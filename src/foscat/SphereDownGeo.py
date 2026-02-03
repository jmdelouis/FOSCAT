
import torch
import torch.nn as nn
import numpy as np
import healpy as hp


class SphereDownGeo(nn.Module):
    """
    Geometric HEALPix downsampling operator (NESTED indexing).

    This module reduces resolution by a factor 2:
        nside_out = nside_in // 2

    Input conventions
    -----------------
    - If in_cell_ids is None:
        x is expected to be full-sphere: [B, C, N_in]
        output is [B, C, K_out] with K_out = len(cell_ids_out) (or N_out if None).
    - If in_cell_ids is provided (fine pixels at nside_in, NESTED):
        x can be either:
          * compact: [B, C, K_in] where K_in = len(in_cell_ids), aligned with in_cell_ids order
          * full-sphere: [B, C, N_in] (also supported)
        output is [B, C, K_out] where cell_ids_out is derived as unique(in_cell_ids // 4),
        unless you explicitly pass cell_ids_out (then it will be intersected with the derived set).

    Modes
    -----
    - mode="smooth": linear downsampling y = M @ x  (M sparse)
    - mode="maxpool": non-linear max over available children (fast)
    """

    def __init__(
        self,
        nside_in: int,
        mode: str = "smooth",
        radius_deg: float | None = None,
        sigma_deg: float | None = None,
        weight_norm: str = "l1",
        cell_ids_out: np.ndarray | list[int] | None = None,
        in_cell_ids: np.ndarray | list[int] | torch.Tensor | None = None,
        use_csr=True,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        self.nside_in = int(nside_in)
        assert (self.nside_in & (self.nside_in - 1)) == 0, "nside_in must be a power of 2."
        self.nside_out = self.nside_in // 2
        assert self.nside_out >= 1, "nside_out must be >= 1."

        self.N_in = 12 * self.nside_in * self.nside_in
        self.N_out = 12 * self.nside_out * self.nside_out

        self.mode = str(mode).lower()
        assert self.mode in ("smooth", "maxpool"), "mode must be 'smooth' or 'maxpool'."

        self.weight_norm = str(weight_norm).lower()
        assert self.weight_norm in ("l1", "l2"), "weight_norm must be 'l1' or 'l2'."

        # ---- Handle reduced-domain inputs (fine pixels) ----
        self.in_cell_ids = self._validate_in_cell_ids(in_cell_ids)
        self.has_in_subset = self.in_cell_ids is not None
        if self.has_in_subset:
            # derive parents
            derived_out = np.unique(self.in_cell_ids // 4).astype(np.int64)
            if cell_ids_out is None:
                self.cell_ids_out = derived_out
            else:
                req_out = self._validate_cell_ids_out(cell_ids_out)
                # keep only those compatible with derived_out (otherwise they'd be all-zero)
                self.cell_ids_out = np.intersect1d(req_out, derived_out, assume_unique=False)
                if self.cell_ids_out.size == 0:
                    raise ValueError(
                        "After intersecting cell_ids_out with unique(in_cell_ids//4), "
                        "no coarse pixel remains. Check your inputs."
                    )
        else:
            self.cell_ids_out = self._validate_cell_ids_out(cell_ids_out)

        self.K_out = int(self.cell_ids_out.size)

        # Column basis for smooth matrix:
        # - full sphere: columns are 0..N_in-1
        # - subset: columns are 0..K_in-1 aligned to self.in_cell_ids
        self.K_in = int(self.in_cell_ids.size) if self.has_in_subset else self.N_in

        if self.mode == "smooth":
            if radius_deg is None:
                # default: include roughly the 4 children footprint
                # (healpy pixel size ~ sqrt(4pi/N), coarse pixel is 4x area)
                radius_deg = 2.0 * hp.nside2resol(self.nside_out, arcmin=True) / 60.0
            if sigma_deg is None:
                sigma_deg = max(radius_deg / 2.0, 1e-6)

            self.radius_deg = float(radius_deg)
            self.sigma_deg = float(sigma_deg)
            self.radius_rad = self.radius_deg * np.pi / 180.0
            self.sigma_rad = self.sigma_deg * np.pi / 180.0
                                        
            M = self._build_down_matrix()  # shape (K_out, K_in or N_in)
              
            self.M = M.coalesce()
            
            if use_csr:
                self.M = self.M.to_sparse_csr().to(self.device)

            self.M_size = M.size()

        else:
            # Precompute children indices for maxpool
            # For subset mode, store mapping from each parent to indices in compact vector,
            # with -1 for missing children.
            children = np.stack(
                [4 * self.cell_ids_out + i for i in range(4)],
                axis=1,
            ).astype(np.int64)  # [K_out, 4] in fine pixel ids (full indexing)

            if self.has_in_subset:
                # map each child pixel id to position in in_cell_ids (compact index)
                pos = self._positions_in_sorted(self.in_cell_ids, children.reshape(-1))
                children_compact = pos.reshape(self.K_out, 4).astype(np.int64)  # -1 if missing
                self.register_buffer(
                    "children_compact",
                    torch.tensor(children_compact, dtype=torch.long, device=self.device),
                )
            else:
                self.register_buffer(
                    "children_full",
                    torch.tensor(children, dtype=torch.long, device=self.device),
                )

        # expose ids as torch buffers for convenience
        self.register_buffer(
            "cell_ids_out_t",
            torch.tensor(self.cell_ids_out.astype(np.int64), dtype=torch.long, device=self.device),
        )
        if self.has_in_subset:
            self.register_buffer(
                "in_cell_ids_t",
                torch.tensor(self.in_cell_ids.astype(np.int64), dtype=torch.long, device=self.device),
            )

    # ---------------- validation helpers ----------------
    def _validate_cell_ids_out(self, cell_ids_out):
        """Return a 1D np.int64 array of coarse cell ids (nside_out)."""
        if cell_ids_out is None:
            return np.arange(self.N_out, dtype=np.int64)

        arr = np.asarray(cell_ids_out, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            raise ValueError("cell_ids_out is empty: provide at least one coarse pixel id.")
        arr = np.unique(arr)
        if arr.min() < 0 or arr.max() >= self.N_out:
            raise ValueError(f"cell_ids_out must be in [0, {self.N_out-1}] for nside_out={self.nside_out}.")
        return arr

    def _validate_in_cell_ids(self, in_cell_ids):
        """Return a 1D np.int64 array of fine cell ids (nside_in) or None."""
        if in_cell_ids is None:
            return None
        if torch.is_tensor(in_cell_ids):
            arr = in_cell_ids.detach().cpu().numpy()
        else:
            arr = np.asarray(in_cell_ids)
        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            raise ValueError("in_cell_ids is empty: provide at least one fine pixel id or None.")
        arr = np.unique(arr)
        if arr.min() < 0 or arr.max() >= self.N_in:
            raise ValueError(f"in_cell_ids must be in [0, {self.N_in-1}] for nside_in={self.nside_in}.")
        return arr

    @staticmethod
    def _positions_in_sorted(sorted_ids: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
        """
        For each query_id, return its index in sorted_ids if present, else -1.
        sorted_ids must be sorted ascending unique.
        """
        q = np.asarray(query_ids, dtype=np.int64)
        idx = np.searchsorted(sorted_ids, q)
        ok = (idx >= 0) & (idx < sorted_ids.size) & (sorted_ids[idx] == q)
        out = np.full(q.shape, -1, dtype=np.int64)
        out[ok] = idx[ok]
        return out

    # ---------------- weights and matrix build ----------------
    def _normalize_weights(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        if w.size == 0:
            return w
        w = np.maximum(w, 0.0)

        if self.weight_norm == "l1":
            s = w.sum()
            if s <= 0.0:
                return np.ones_like(w) / max(w.size, 1)
            return w / s

        # l2
        s2 = (w * w).sum()
        if s2 <= 0.0:
            return np.ones_like(w) / max(np.sqrt(w.size), 1.0)
        return w / np.sqrt(s2)

    def _build_down_matrix(self) -> torch.Tensor:
        nside_in = self.nside_in
        nside_out = self.nside_out
        sigma = float(self.sigma_rad)

        p_out = self.cell_ids_out.astype(np.int64)  # [K]
        K = p_out.size
        offs = np.arange(4, dtype=np.int64)         # [4]

        # --- (A) Choix du voisinage côté coarse
        # Option 1 (minimal, très rapide) : uniquement le parent -> 4 enfants
        #parents = p_out[:, None]                    # [K,1]

        # Option 2 (plus “lisse”) : parent + 8 voisins -> 9 parents -> 36 enfants
        neigh8 = hp.get_all_neighbours(nside_out, p_out, nest=True)  # [8,K] (healpy renvoie souvent [8,K])
        parents = np.concatenate([p_out[None, :], neigh8], axis=0).T # [K,9]
        idx=np.where(parents==-1)
        parents[idx[0],idx[1]]=parents[idx[0],idx[1]-1]

        # --- enfants fins (NESTED) : child_id = 4*parent + {0,1,2,3}
        children = (4 * parents[..., None] + offs[None, None, :]).reshape(K, -1)  # [K, 4] ou [K,36]

        # Si option voisins activée : invalider enfants des parents=-1
        #mask_child = np.repeat(mask_parent, 4, axis=1)               # [K,36]
        #children_flat = children[mask_child]
        #print(mask_child.shape,children.shape,K)
        rows_flat = np.repeat(np.arange(K, dtype=np.int64), children.shape[1])#[mask_child.ravel()]

        # Option minimal (sans voisins) :
        children_flat = children.reshape(-1)                           # [K*4]
        #rows_flat = np.repeat(np.arange(K, dtype=np.int64), children.shape[1])

        # --- Subset: map vers indices compacts
        if self.has_in_subset:
            in_ids = self.in_cell_ids  # trié/unique :contentReference[oaicite:4]{index=4}
            idx = np.searchsorted(in_ids, children_flat)
            ok = (idx >= 0) & (idx < in_ids.size) & (in_ids[idx] == children_flat)
            cols_flat = idx[ok]
            rows_flat2 = rows_flat[ok]
            child_ids_kept = children_flat[ok]
        else:
            cols_flat = children_flat
            rows_flat2 = rows_flat
            child_ids_kept = children_flat

        if rows_flat2.size == 0:
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            vals_t = torch.zeros((0,), dtype=self.dtype, device=self.device)
            return torch.sparse_coo_tensor(indices, vals_t, size=(self.K_out, self.K_in),
                                           device=self.device, dtype=self.dtype).coalesce()

        # --- poids gaussiens (vectorisé)
        # centres coarse: vec0 [K,3]
        vx0, vy0, vz0 = hp.pix2vec(nside_out, p_out, nest=True)
        vec0 = np.stack([vx0, vy0, vz0], axis=1)                        # [K,3]

        # vec des enfants gardés
        vx, vy, vz = hp.pix2vec(nside_in, child_ids_kept, nest=True)
        vec = np.stack([vx, vy, vz], axis=1)                            # [nnz,3]

        # dot(vec, vec0[row])
        dots = np.einsum("ij,ij->i", vec, vec0[rows_flat2])
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        w = np.exp(-2.0 * (ang / max(sigma, 1e-30)) ** 2)

        # --- normalisation par ligne (row-wise), sans boucle
        # On recompose un tableau dense “temporaire” [K, m] via indexation
        m = children.shape[1]
        # position within row (0..m-1)
        jpos = np.tile(np.arange(m, dtype=np.int64), K)
        if self.has_in_subset:
            jpos = jpos.reshape(-1)[ok]  # aligné sur rows_flat2/child_ids_kept
        # si option voisins + mask_child : jpos = jpos[mask_child.reshape(-1)][ok] etc.

        W = np.zeros((K, m), dtype=np.float64)
        W[rows_flat2, jpos] = w

        if self.weight_norm == "l1":
            s = W.sum(axis=1, keepdims=True)
            s[s <= 0] = 1.0
            W /= s
        else:  # l2
            s2 = np.sqrt((W * W).sum(axis=1, keepdims=True))
            s2[s2 <= 0] = 1.0
            W /= s2

        # extraire w normalisés aux mêmes nnz
        w_norm = W[rows_flat2, jpos].astype(np.float32)

        # --- sparse
        rows_t = torch.tensor(rows_flat2, dtype=torch.long, device=self.device)
        cols_t = torch.tensor(cols_flat, dtype=torch.long, device=self.device)
        vals_t = torch.tensor(w_norm, dtype=self.dtype, device=self.device)

        indices = torch.stack([rows_t, cols_t], dim=0)
        return torch.sparse_coo_tensor(indices, vals_t, size=(self.K_out, self.K_in),
                                    device=self.device, dtype=self.dtype).coalesce().to_sparse_csr().to(self.device)
    '''
    def _build_down_matrix(self) -> torch.Tensor:
        """Construct sparse matrix M (K_out, K_in or N_in) for the selected coarse pixels."""
        nside_in = self.nside_in
        nside_out = self.nside_out

        radius_rad = self.radius_rad
        sigma_rad = self.sigma_rad

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        # For subset columns, we use self.in_cell_ids as the basis
        subset_cols = self.has_in_subset
        in_ids = self.in_cell_ids  # np.ndarray or None

        for r, p_out in enumerate(self.cell_ids_out.tolist()):
            theta0, phi0 = hp.pix2ang(nside_out, int(p_out), nest=True)
            vec0 = hp.ang2vec(theta0, phi0)

            neigh = hp.query_disc(nside_in, vec0, radius_rad, inclusive=True, nest=True)
            neigh = np.asarray(neigh, dtype=np.int64)

            if subset_cols:
                # keep only valid fine pixels
                # neigh is not sorted; intersect1d expects sorted
                neigh_sorted = np.sort(neigh)
                keep = np.intersect1d(neigh_sorted, in_ids, assume_unique=False)
                neigh = keep

            # Fallback: if radius query returns nothing in subset mode, at least try the 4 children
            if neigh.size == 0:
                children = (4 * int(p_out) + np.arange(4, dtype=np.int64))
                if subset_cols:
                    pos = self._positions_in_sorted(in_ids, children)
                    ok = pos >= 0
                    if np.any(ok):
                        neigh = children[ok]
                    else:
                        # nothing to connect -> row stays zero
                        continue
                else:
                    neigh = children

            theta, phi = hp.pix2ang(nside_in, neigh, nest=True)
            vec = hp.ang2vec(theta, phi)

            # angular distance via dot product
            dots = np.clip(np.dot(vec, vec0), -1.0, 1.0)
            ang = np.arccos(dots)
            w = np.exp(- 2.0*(ang / sigma_rad) ** 2)

            w = self._normalize_weights(w)

            if subset_cols:
                pos = self._positions_in_sorted(in_ids, neigh)
                # all should be present due to filtering, but guard anyway
                ok = pos >= 0
                neigh_pos = pos[ok]
                w = w[ok]
                if neigh_pos.size == 0:
                    continue
                for c, v in zip(neigh_pos.tolist(), w.tolist()):
                    rows.append(r)
                    cols.append(int(c))
                    vals.append(float(v))
            else:
                for c, v in zip(neigh.tolist(), w.tolist()):
                    rows.append(r)
                    cols.append(int(c))
                    vals.append(float(v))

        if len(rows) == 0:
            # build an all-zero sparse tensor
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            vals_t = torch.zeros((0,), dtype=self.dtype, device=self.device)
            return torch.sparse_coo_tensor(
                indices, vals_t, size=(self.K_out, self.K_in), device=self.device, dtype=self.dtype
            ).coalesce()

        rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=self.device)
        vals_t = torch.tensor(vals, dtype=self.dtype, device=self.device)

        indices = torch.stack([rows_t, cols_t], dim=0)
        M = torch.sparse_coo_tensor(
            indices,
            vals_t,
            size=(self.K_out, self.K_in),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()
        return M
        '''
    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            If has_in_subset:
                - [B,C,K_in] (compact, aligned with in_cell_ids) OR [B,C,N_in] (full sphere)
            Else:
                - [B,C,N_in] (full sphere)

        Returns
        -------
        y : torch.Tensor
            [B,C,K_out]
        cell_ids_out : torch.Tensor
            [K_out] coarse pixel ids (nside_out), aligned with y last dimension.
        """
        if x.dim() != 3:
            raise ValueError("x must be [B, C, N]")

        B, C, N = x.shape
        if self.has_in_subset:
            if N not in (self.K_in, self.N_in):
                raise ValueError(
                    f"x last dim must be K_in={self.K_in} (compact) or N_in={self.N_in} (full), got {N}"
                )
        else:
            if N != self.N_in:
                raise ValueError(f"x last dim must be N_in={self.N_in}, got {N}")

        if self.mode == "smooth":

            # If x is full-sphere but M is subset-based, gather compact inputs
            if self.has_in_subset and N == self.N_in:
                x_use = x.index_select(dim=2, index=self.in_cell_ids_t.to(x.device))
            else:
                x_use = x

            # sparse mm expects 2D: (K_out, K_in) @ (K_in, B*C)
            x2 = x_use.reshape(B * C, -1).transpose(0, 1).contiguous()
            y2 = torch.sparse.mm(self.M, x2)
            y = y2.transpose(0, 1).reshape(B, C, self.K_out).contiguous()
            return y, self.cell_ids_out_t.to(x.device)

        # maxpool
        if self.has_in_subset and N == self.N_in:
            x_use = x.index_select(dim=2, index=self.in_cell_ids_t.to(x.device))
        else:
            x_use = x

        if self.has_in_subset:
            # children_compact: [K_out, 4] indices in 0..K_in-1 or -1
            ch = self.children_compact.to(x.device)  # [K_out,4]
            # gather with masking
            # We build y by iterating 4 children with max
            y = None
            for j in range(4):
                idx = ch[:, j]  # [K_out]
                mask = idx >= 0
                # start with very negative so missing children don't win
                tmp = torch.full((B, C, self.K_out), -torch.inf, device=x.device, dtype=x.dtype)
                if mask.any():
                    tmp[:, :, mask] = x_use.index_select(dim=2, index=idx[mask]).reshape(B, C, -1)
                y = tmp if y is None else torch.maximum(y, tmp)
            # If a parent had no valid children at all, it is -inf -> set to 0
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
            return y, self.cell_ids_out_t.to(x.device)

        else:
            ch = self.children_full.to(x.device)  # [K_out,4] full indices
            # gather children and max
            xch = x_use.index_select(dim=2, index=ch.reshape(-1)).reshape(B, C, self.K_out, 4)
            y = xch.max(dim=3).values
            return y, self.cell_ids_out_t.to(x.device)
