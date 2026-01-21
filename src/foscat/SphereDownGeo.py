import torch
import torch.nn as nn
import numpy as np
import healpy as hp


class SphereDownGeo(nn.Module):
    """Geometric HEALPix downsampling operator (full sphere, NESTED indexing).

    This module reduces resolution by a factor 2:
        nside_out = nside_in // 2

    Modes
    -----
    - mode="smooth": linear downsampling y = M @ x  (M sparse)
    - mode="maxpool": non-linear max over the 4 children (fast)

    Weight normalization (smooth mode)
    ---------------------------------
    - weight_norm="l1": sum(w) = 1  (preserve constants / DC gain)
    - weight_norm="l2": sum(w^2) = 1 (preserve energy locally, not DC)
    - weight_norm="none": no normalization

    Partial-sphere / subset support
    -------------------------------
    Two ways to restrict the operator:

    1) Provide ``cell_ids_out`` (coarse pixel ids at nside_out, NESTED).
       The operator is built ONLY for those outputs.

    2) Provide ``in_cell_ids`` (fine pixel ids at nside_in, NESTED) to
       declare which *input* pixels you consider valid. In that case we
       automatically build the operator ONLY for the corresponding coarse
       parents:

           cell_ids_out = unique(in_cell_ids // 4)

    In that case the output has shape [B, C, K] with K=len(cell_ids_out),
    and ``forward`` returns these coarse ids (same order as provided).

    Notes
    -----
    - Input is still assumed to be a full-sphere vector at nside_in.
    - ``cell_ids_out`` refers to the *coarse* grid (nside_out) ids.
    """

    def __init__(
        self,
        nside_in: int,
        mode: str = "smooth",
        radius_deg: float | None = None,
        sigma_deg: float | None = None,
        weight_norm: str = "l1",
        in_cell_ids=None,
        cell_ids_out: np.ndarray | list[int] | None = None,
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

        # Optional: restrict outputs based on valid input pixel ids
        self.in_cell_ids = self._validate_in_cell_ids(in_cell_ids)

        # Validate / store subset of coarse pixels (either explicit, or derived from in_cell_ids)
        if self.in_cell_ids is not None:
            # NESTED: each coarse pixel has exactly 4 children; parent = child // 4
            derived_out = np.unique((self.in_cell_ids // 4).astype(np.int64))
            self.cell_ids_out = self._validate_cell_ids_out(derived_out)
        else:
            self.cell_ids_out = self._validate_cell_ids_out(cell_ids_out)

        # Convenience tensor version (returned by forward)
        self.register_buffer(
            "cell_ids_out_t",
            torch.as_tensor(self.cell_ids_out, dtype=torch.long, device=self.device),
        )
        self.K_out = int(self.cell_ids_out.size)

        mode = str(mode).lower().strip()
        if mode not in ("smooth", "maxpool"):
            raise ValueError("mode must be either 'smooth' or 'maxpool'.")
        self.mode = mode

        weight_norm = str(weight_norm).lower().strip()
        if weight_norm not in ("l1", "l2", "none"):
            raise ValueError("weight_norm must be 'l1', 'l2', or 'none'.")
        self.weight_norm = weight_norm

        if self.mode == "smooth":
            pix_area = 4.0 * np.pi / self.N_in
            pix_rad = np.sqrt(pix_area)
            pix_deg = pix_rad * 180.0 / np.pi

            if radius_deg is None:
                radius_deg = 3.0 * pix_deg
            if sigma_deg is None:
                sigma_deg = radius_deg / 2.0

            self.radius_deg = float(radius_deg)
            self.sigma_deg = float(sigma_deg)
            self.radius_rad = self.radius_deg * np.pi / 180.0
            self.sigma_rad = self.sigma_deg * np.pi / 180.0

            M = self._build_down_matrix()  # shape (K_out, N_in)

            self.register_buffer("M_indices", M.indices())
            self.register_buffer("M_values", M.values())
            self.M_size = M.size()

        else:
            # children_idx: [K_out, 4]
            children = np.stack(
                [
                    4 * self.cell_ids_out + 0,
                    4 * self.cell_ids_out + 1,
                    4 * self.cell_ids_out + 2,
                    4 * self.cell_ids_out + 3,
                ],
                axis=1,
            ).astype(np.int64)

            if children.max() >= self.N_in or children.min() < 0:
                raise RuntimeError("Child indices out of bounds. Check NESTED/full-sphere assumptions.")

            self.register_buffer(
                "children_idx",
                torch.as_tensor(children, dtype=torch.long, device=self.device),
            )

    def _validate_cell_ids_out(self, cell_ids_out):
        """Return a 1D np.int64 array of coarse cell ids.

        - None -> full sphere [0..N_out-1]
        - Ensures unique & sorted ids
        - Checks bounds
        """
        if cell_ids_out is None:
            return np.arange(self.N_out, dtype=np.int64)

        arr = np.asarray(cell_ids_out, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            raise ValueError("cell_ids_out is empty: provide at least one coarse pixel id.")

        # Unique + sort for deterministic behavior (and simpler debugging)
        arr = np.unique(arr)

        if arr.min() < 0 or arr.max() >= self.N_out:
            raise ValueError(
                f"cell_ids_out must be within [0, {self.N_out - 1}] for nside_out={self.nside_out}. "
                f"Got min={arr.min()}, max={arr.max()}."
            )

        return arr

    def _validate_in_cell_ids(self, in_cell_ids):
        """Return a 1D np.int64 array of fine cell ids, or None.

        Accepts torch tensors or numpy-like inputs.
        Ensures unique & sorted ids, checks bounds.
        """
        if in_cell_ids is None:
            return None

        if isinstance(in_cell_ids, torch.Tensor):
            arr = in_cell_ids.detach().cpu().numpy()
        else:
            arr = np.asarray(in_cell_ids)

        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            raise ValueError("in_cell_ids is empty: provide at least one fine pixel id.")

        arr = np.unique(arr)
        if arr.min() < 0 or arr.max() >= self.N_in:
            raise ValueError(
                f"in_cell_ids must be within [0, {self.N_in - 1}] for nside_in={self.nside_in}. "
                f"Got min={arr.min()}, max={arr.max()}."
            )

        return arr

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        dlat = 0.5 * (lat2 - lat1)
        dlon = 0.5 * (lon2 - lon1)

        sin_dlat = np.sin(dlat)
        sin_dlon = np.sin(dlon)

        a = sin_dlat**2 + np.cos(lat1) * np.cos(lat2) * sin_dlon**2
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))
        return c

    def _normalize_weights(self, w: np.ndarray) -> np.ndarray:
        """Apply the requested normalization to a 1D numpy array of weights."""
        if self.weight_norm == "none":
            return w

        if self.weight_norm == "l1":
            s = w.sum()
            if s <= 0.0:
                return np.ones_like(w) / max(w.size, 1)
            return w / s

        # self.weight_norm == "l2"
        s2 = (w * w).sum()
        if s2 <= 0.0:
            return np.ones_like(w) / max(np.sqrt(w.size), 1.0)
        return w / np.sqrt(s2)

    def _build_down_matrix(self) -> torch.Tensor:
        """Construct sparse matrix M (K_out, N_in) for the selected coarse pixels."""
        N_in = self.N_in
        nside_in = self.nside_in
        nside_out = self.nside_out

        radius_rad = self.radius_rad
        sigma_rad = self.sigma_rad

        rows = []
        cols = []
        vals = []

        # IMPORTANT: output rows are 0..K_out-1, but refer to coarse pixel id p_out
        for r, p_out in enumerate(self.cell_ids_out.tolist()):
            theta0, phi0 = hp.pix2ang(nside_out, int(p_out), nest=True)
            lat0 = 0.5 * np.pi - theta0
            lon0 = phi0

            vec0 = hp.ang2vec(theta0, phi0)
            cand = hp.query_disc(nside_in, vec0, radius_rad, inclusive=True, nest=True)
            cand = np.asarray(cand, dtype=np.int64)

            if cand.size == 0:
                cand = np.array(
                    [4 * p_out + 0, 4 * p_out + 1, 4 * p_out + 2, 4 * p_out + 3],
                    dtype=np.int64,
                )

            theta_c, phi_c = hp.pix2ang(nside_in, cand, nest=True)
            lat_c = 0.5 * np.pi - theta_c
            lon_c = phi_c
            gamma = self._haversine(lat0, lon0, lat_c, lon_c)

            #w = np.exp(-0.5 * (gamma / sigma_rad) ** 2)
            w = np.exp(- (gamma / sigma_rad) ** 2)
            w[gamma > radius_rad] = 0.0
            if w.sum() <= 0.0:
                w[:] = 1.0

            w = self._normalize_weights(w)

            for pix, w_val in zip(cand, w):
                if 0 <= pix < N_in and w_val != 0.0:
                    rows.append(r)
                    cols.append(int(pix))
                    vals.append(float(w_val))

        rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=self.device)
        vals_t = torch.tensor(vals, dtype=self.dtype, device=self.device)

        indices = torch.stack([rows_t, cols_t], dim=0)
        M = torch.sparse_coo_tensor(
            indices,
            vals_t,
            size=(self.K_out, N_in),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()
        return M

    def forward(self, x: torch.Tensor):
        """Apply downsampling.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, C, N_in] at nside_in (full sphere, nested order).

        Returns
        -------
        y : torch.Tensor
            Shape [B, C, K_out] where K_out=len(cell_ids_out).
        cell_ids_out : torch.Tensor
            Coarse pixel ids (NESTED, at nside_out) corresponding to the last dimension of y.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B,C,N], got {tuple(x.shape)}")

        B, C, N_in = x.shape
        if N_in != self.N_in:
            raise AssertionError(f"Expected N_in={self.N_in}, got {N_in}")

        if self.mode == "smooth":
            M = torch.sparse_coo_tensor(
                self.M_indices.to(device=x.device),
                self.M_values.to(device=x.device, dtype=x.dtype),
                size=self.M_size,
                device=x.device,
                dtype=x.dtype,
            )
            x_bc = x.reshape(B * C, N_in)
            y_bc_T = torch.sparse.mm(M, x_bc.T)  # [K_out, B*C]
            y = y_bc_T.T.reshape(B, C, self.K_out)
            return y, self.cell_ids_out_t.to(device=x.device)

        idx = self.children_idx.to(device=x.device)
        gathered = x[:, :, idx]          # [B, C, K_out, 4]
        y = gathered.max(dim=-1).values  # [B, C, K_out]
        return y, self.cell_ids_out_t.to(device=x.device)
