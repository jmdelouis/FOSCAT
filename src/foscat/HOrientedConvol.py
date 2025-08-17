import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.sparse import csr_array
import torch
from scipy.spatial import cKDTree

class HOrientedConvol:
    def __init__(self,nside,KERNELSZ,cell_ids=None,nest=True):
        
        if KERNELSZ % 2 == 0:
            raise ValueError(f"N must be odd so that coordinates are integers from -K..K; got N={N}.")

        self.local_test=False
        
        if cell_ids is None:
            self.cell_ids=np.arange(12*nside**2)
            
            idx_nn = self.knn_healpix_ckdtree(self.cell_ids, 
                KERNELSZ*KERNELSZ, 
                nside,
                nest=nest,
            )
            '''
            idx_nn = self.all_neighbours_batched(
                self.cell_ids, 
                KERNELSZ*KERNELSZ, 
                nside=nside, 
                nest=nest,
                overshoot=2.0,      # candidate pool ≈ 2×N
                parent_batch=4096,  # tune to fit memory
                # out_memmap_path="neighbors_nest.int64.memmap",  # enable for huge M to be tested
                )
            '''
        else:
            try:
                self.cell_ids=cell_ids.cpu().numpy()
            except:
                self.cell_ids=cell_ids
                
            self.local_test=True
            
            idx_nn = self.knn_healpix_ckdtree(self.cell_ids, 
                KERNELSZ*KERNELSZ, 
                nside,
                nest=nest,
            )

            
        mat_pt=self.rotation_matrices_from_healpix(nside,self.cell_ids,nest=nest)

        if self.local_test:
            t,p = hp.pix2ang(nside,self.cell_ids[idx_nn],nest=True)
        else:
            t,p = hp.pix2ang(nside,idx_nn,nest=True)
            
        vec_orig=hp.ang2vec(t,p)

        self.vec_rot = np.einsum('mki,ijk->kmj', vec_orig,mat_pt)

        '''
        if self.local_test:
            idx_nn=self.remap_by_first_column(idx_nn)
        '''
        
        del mat_pt
        del vec_orig
        self.t=t[:,0]
        self.p=p[:,0]
        self.idx_nn=idx_nn
        self.nside=nside
        self.KERNELSZ=KERNELSZ

    def remap_by_first_column(self,idx: np.ndarray) -> np.ndarray:
        """
        Remap the values in `idx` so that:
          - The first column becomes [0, 1, ..., N-1]
          - All other columns are updated accordingly using the same mapping.
        
        Parameters
        ----------
        idx : np.ndarray
            Integer array of shape (N, m).
            Assumes all values in idx are present in the first column (otherwise they get -1).
    
        Returns
        -------
        np.ndarray
            New array with remapped indices.
        """
        if idx.ndim != 2:
            raise ValueError("idx must be a 2D array of shape (N, m)")
        
        N, m = idx.shape
    
        # Create a mapping: original_value_in_first_column -> row_index
        # Example: if idx[:,0] = [101, 505, 303], then mapping = {101:0, 505:1, 303:2}
        keys = idx[:, 0]
        mapping = {v: i for i, v in enumerate(keys)}
    
        # Optional check: ensure all values are in the mapping keys
        # If not, you can raise an error or handle it differently
        # if not np.isin(idx, keys).all():
        #     missing = np.unique(idx[~np.isin(idx, keys)])
        #     raise ValueError(f"Some values are not in idx[:,0]: {missing}")
    
        # Function to get mapped value, or -1 if value is not found
        get = mapping.get
    
        # Apply mapping to all elements (vectorized via np.vectorize)
        out = np.vectorize(lambda v: get(int(v), -1), otypes=[int])(idx)
    
        return out
    
    def rotation_matrices_from_healpix(self,nside, hpix_idx, nest=True):
        """
        Compute rotation matrices that move each Healpix pixel center to the North pole.
        equivalent to rotation matrices R_z(phi) * R_y(-thi) for N points.
    
        Parameters
        ----------
        nside : int
            Healpix Nside resolution.
        hpix_idx : array_like, shape (N,)
            Healpix pixel indices.
        nest : bool, optional
            True if indices are in NESTED ordering, False for RING ordering.
    
        Returns
        -------
        R : ndarray, shape (3, 3, N)
            Rotation matrices for each pixel index.
        """
        
        try:
            hpix_idx = np.asarray(hpix_idx)
        except:
            hpix_idx = hpix_idx.cpu().numpy()
            
        N = hpix_idx.shape[0]
    
        # Get angular coordinates of each pixel center
        theta, phi = hp.pix2ang(nside, hpix_idx, nest=nest)  # theta: colatitude (0=north pole)
        
        # Precompute sines/cosines
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthi = np.cos(-theta)
        sthi = np.sin(-theta)
    
        # Rotation around Z (by phi)
        Rz = np.zeros((3, 3, N))
        Rz[0, 0, :] = cphi
        Rz[0, 1, :] = -sphi
        Rz[1, 0, :] = sphi
        Rz[1, 1, :] = cphi
        Rz[2, 2, :] = 1.0
    
        # Rotation around Y (by -theta)
        Ry = np.zeros((3, 3, N))
        Ry[0, 0, :] = cthi
        Ry[0, 2, :] = -sthi
        Ry[1, 1, :] = 1.0
        Ry[2, 0, :] = sthi
        Ry[2, 2, :] = cthi
    
        # Multiply Rz * Ry for each pixel
        R = np.einsum('ijk,jlk->ilk', Rz, Ry)
        
        return R

    def _choose_depth_for_candidates(self, N, overshoot=2, max_depth=12):
        """
        Pick hierarchy depth d so that ~ 9 * 4**d >= overshoot * N.
        Depth 0 => 9 candidates; 1 => 36; 2 => 144; 3 => 576; 4 => 2304; etc.
        """
        d = 0
        while 9 * (4 ** d) < overshoot * N and d < max_depth:
            d += 1
        return d

    def knn_healpix_ckdtree(self,
        hidx, N, nside, *, nest=True,
        include_self=True,
        vec_dtype=np.float32,
        out_dtype=np.int64
    ):
        """
        k-NN using a cKDTree on unit vectors (exact in Euclidean space).
        Returns LOCAL indices (0..M-1) of the N nearest neighbours per row.
        """
        try:
            hidx = np.asarray(hidx, dtype=np.int64)
        except:
            hidx = hidx.cpu().numpy()
            
        if hidx.ndim != 1:
            raise ValueError("hidx must be 1D")
        M = hidx.size
        if M == 0:
            return np.empty((0, 0), dtype=out_dtype)
        if N <= 0:
            raise ValueError("N must be >= 1")

        # Effective N
        N_eff = min(N, M if include_self else max(M-1, 1))

        # Build unit vectors
        hidx_n = hidx if nest else hp.ring2nest(nside, hidx)
        x, y, z = hp.pix2vec(nside, hidx_n, nest=True)
        V = np.stack([x, y, z], axis=1).astype(vec_dtype, copy=False)  # (M,3)

        tree = cKDTree(V)

        if include_self:
            # Self appears with distance 0 as the first neighbour
            d, idx = tree.query(V, k=N_eff, workers=-1)   # idx shape (M,N)
            return idx.astype(out_dtype, copy=False)
        else:
            # Ask for one extra and drop self
            k = min(N_eff + 1, M)
            d, idx = tree.query(V, k=k, workers=-1)
            # idx can be (M,) if k==1; normalize shapes
            if idx.ndim == 1:
                idx = idx[:, None]
            # Remove self if present (distance 0)
            out = np.empty((M, N_eff), dtype=out_dtype)
            for i in range(M):
                row = idx[i]
                # filter out self (i); keep first N_eff
                row = row[row != i][:N_eff]
                # if M==N and no self, row already size N_eff
                out[i, :row.size] = row
                if row.size < N_eff:
                    # extremely rare (degenerate duplicates); fallback by scores
                    cand = np.setdiff1d(np.arange(M), np.r_[i, row], assume_unique=False)
                    # pick nearest remaining
                    di, ci = tree.query(V[i], k=N_eff - row.size)
                    out[i, row.size:] = np.atleast_1d(ci).astype(out_dtype, copy=False)
            return out
            
    def knn_healpix_subset_nest(self,
        hidx, N, nside, *, nest=True,
        include_self=True,
        alpha=8,                # initial half-window ~ alpha*N
        edge_margin=8,          # if top-N hits within this many indices from L/R, expand
        vec_dtype=np.float32,
        out_dtype=np.int64,
        max_expand_iters=20
    ):
        """
        Nearest neighbours inside a HEALPix subset using NEST-order windows.
        Edge-aware expansion: grow window if selected neighbours sit on the window edges.
        Returns LOCAL indices 0..M-1 (use hidx[idx] to get HEALPix ids).
        """
        # ---- Inputs
        hidx = np.asarray(hidx, dtype=np.int64)
        if hidx.ndim != 1:
            raise ValueError("hidx must be 1D")
        M = hidx.size
        if M == 0:
            return np.empty((0, 0), dtype=out_dtype)
        if N <= 0:
            raise ValueError("N must be >= 1")

        N_eff = min(N, M if include_self else max(M - 1, 1))

        # ---- Convert to NEST if needed, then unit vectors
        hidx_n = hidx if nest else hp.ring2nest(nside, hidx)
        x, y, z = hp.pix2vec(nside, hidx_n, nest=True)
        V = np.stack([x, y, z], axis=1).astype(vec_dtype, copy=False)  # (M,3)

        # ---- Sort by NEST index for locality
        p = np.argsort(hidx_n, kind="mergesort")   # sorted positions -> local indices
        invp = np.empty(M, dtype=np.int64)
        invp[p] = np.arange(M)                     # local index -> position in sorted order

        W0 = max(int(alpha * N_eff), N_eff + 16)
        out = np.empty((M, N_eff), dtype=out_dtype)

        for i in range(M):
            pos = invp[i]
            W = W0
            need = N_eff + (0 if include_self else 1)

            for _ in range(max_expand_iters + 1):
                L = max(0, pos - W)
                R = min(M, pos + W + 1)  # [L, R)
                # Candidate local indices and their positions in the sorted order:
                cand_all = p[L:R]
                pos_all = np.arange(L, R, dtype=np.int64)

                if not include_self:
                    mask = (cand_all != i)
                    cand = cand_all[mask]
                    pos_cand = pos_all[mask]
                else:
                    cand = cand_all
                    pos_cand = pos_all

                if cand.size < need and (L > 0 or R < M):
                    # Not enough candidates yet: expand and retry
                    W = min(max(W * 2, W + 1), M - 1)
                    continue

                # Score candidates by dot product (equiv. to squared Euclidean distance order)
                s = V[i] @ V[cand].T
                kth = N_eff - 1
                top = np.argpartition(-s, kth=kth)[:N_eff]
                order = np.argsort(-s[top])
                top = top[order]

                # Edge-aware check: did we pick neighbours that sit on the window edges?
                sel_pos = pos_cand[top]
                hit_left  = (sel_pos - L).min() <= edge_margin
                hit_right = (R - 1 - sel_pos).min() <= edge_margin
                fully_covered = (L == 0 and R == M)

                if not fully_covered and (hit_left or hit_right):
                    # Likely better neighbours just outside -> expand and re-run
                    W = min(max(W * 2, W + 1), M - 1)
                    continue

                # Done: either no edge hit, or full coverage (exact)
                out[i] = cand[top].astype(out_dtype, copy=False)
                break

            else:
                # Safety: if the loop exits without break (shouldn't happen), fall back to full set
                s = V[i] @ V.T
                if not include_self:
                    s[i] = -np.inf
                top = np.argpartition(-s, kth=N_eff - 1)[:N_eff]
                out[i] = top[np.argsort(-s[top])].astype(out_dtype, copy=False)

        return out


    def knn_healpix_subset(self,
                           hidx, N, nside, *, nest=True,
                           batch=None,                    # rows per block; auto if None
                           vec_dtype=np.float32,          # float32 is enough for unit vectors
                           out_dtype=np.int64,
                           include_self=True,
                           max_memory_bytes=512*1024**2   # ~512 MB working buffer
                           ):
        """
        Return, for each input pixel (subset), the indices (0..M-1) of its N nearest
        neighbours within the subset itself, using squared Euclidean distance in 3D:
            d2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
        where (x,y,z) = hp.pix2vec(nside, hidx, nest).

        Parameters
        ----------
        hidx : (M,) int64
            Subset of HEALPix pixel ids (NEST or RING according to `nest`).
        N : int
            Number of neighbours per row. If include_self=True, the center is one of them.
        nside : int
            HEALPix NSIDE.
        nest : bool, default True
            If False, hidx is converted RING->NEST internally.
        batch : int or None
            Number of query rows processed at once. If None, chosen from max_memory_bytes.
        vec_dtype : dtype
            dtype for the (x,y,z) unit vectors (float32 recommended).
        out_dtype : dtype
            dtype for returned indices.
        include_self : bool
            If True, each row contains its center index.
        max_memory_bytes : int
            Target memory for the score matrix block (approx B * M * 4 bytes).

        Returns
        -------
        idx_nn : (M, N) int array (out_dtype)
            For each input pixel i, indices in [0..M-1] of the N nearest pixels inside `hidx`.
            To get the corresponding HEALPix ids, do: `hidx[idx_nn]`.
        """
        # ---- inputs
        hidx = np.asarray(hidx, dtype=np.int64)
        if hidx.ndim != 1:
            raise ValueError("hidx must be 1D")
        M = hidx.size
        if M == 0:
            return np.empty((0, 0), dtype=out_dtype)
        if N <= 0:
            raise ValueError("N must be >= 1")

        # You cannot return more than M neighbours (or M-1 if self excluded)
        N_eff = min(N, M if include_self else max(M-1, 1))

        # ---- to NEST + unit vectors
        hidx_n = hidx if nest else hp.ring2nest(nside, hidx)
        x, y, z = hp.pix2vec(nside, hidx_n, nest=True)
        V = np.stack([x, y, z], axis=1).astype(vec_dtype, copy=False)   # (M,3), unit-norm

        # ---- choose batch automatically if not provided
        if batch is None:
            # score block S has shape (B, M) with float32 (~4 bytes)
            bytes_per_col = np.dtype(np.float32).itemsize
            B = int(max_memory_bytes // (M * bytes_per_col))
            B = max(1, min(B, M))
        else:
            B = int(max(1, min(batch, M)))

        out = np.empty((M, N_eff), dtype=out_dtype)

        # ---- process by row blocks
        for i0 in range(0, M, B):
            i1 = min(i0 + B, M)
            Bcur = i1 - i0

            # Dot products with entire set: (B,3) @ (3,M) -> (B,M)
            S = V[i0:i1] @ V.T                                   # cosine since unit vectors

            if include_self:
                # Nudge the diagonal so self is strictly the best but stays finite
                S[np.arange(Bcur), np.arange(i0, i1)] += np.float32(1e-6)
            else:
                # Exclude self by setting to -inf
                S[np.arange(Bcur), np.arange(i0, i1)] = np.float32(-np.inf)

            # We want smallest Euclidean distance <=> largest dot product
            kth = N_eff - 1
            top = np.argpartition(-S, kth=kth, axis=1)[:, :N_eff]         # (B, N)
            top_scores = np.take_along_axis(S, top, axis=1)
            order = np.argsort(-top_scores, axis=1)                       # sort by score desc
            top_sorted = np.take_along_axis(top, order, axis=1)           # (B, N)

            out[i0:i1] = top_sorted.astype(out_dtype, copy=False)

        return out
    
    
    def all_neighbours_batched2(
        self,
        hidx,
        N,
        nside,
        *,
        nest=True,
        overshoot=2.0,
        depth=None,
        parent_batch=4096,
        out_memmap_path=None,
        dtype_out=np.int64,
        vec_dtype=np.float32,
    ):
        """
        k-NN on a HEALPix grid (same NSIDE) restricted to the input subset `hidx`.
        Returns LOCAL indices (0..M-1) into `hidx`. Guarantees exactly N *unique* neighbours per row.

        Strategy
        --------
        1) Build a local candidate bank via parent refinement.
        2) Score candidates, take a widened top-K pool (K >= 3N).
        3) Row-wise deduplicate while preserving score order.
        4) If < N after dedup -> global fallback on the subset to fill to N.
        5) Enforce center presence, re-rank, and re-check uniqueness. If still off -> fallback (rare).

        Parameters
        ----------
        hidx : (M,) int array
            Subset pixel IDs (NEST or RING according to `nest`).
        N : int
            Neighbours per row (includes the center). Clipped to M.
        nside : int
            Full-resolution NSIDE (power of 2).
        nest : bool, default True
            Input scheme; internals use NEST.
        overshoot : float, default 2.0
            Kept for compatibility, not relied upon for pool size (we use an explicit pool multiplier).
        depth : int or None
            Parent depth; if None, picked via self._choose_depth_for_candidates.
        parent_batch : int, default 4096
            Number of unique parents processed per batch.
        out_memmap_path : str or None
            If set, results written to mmap file on disk.
        dtype_out : dtype, default np.int64
            Output dtype (local indices).
        vec_dtype : dtype, default np.float32
            dtype for unit vectors.

        Returns
        -------
        out_local : (M, N) array (or memmap)
            Local indices (0..M-1) referencing `hidx`. Each row has exactly N unique entries and contains the center.
        """
        # ---- Basic checks
        hidx = np.asarray(hidx, dtype=np.int64)
        if hidx.ndim != 1:
            raise ValueError("hidx must be 1D (M,)")
        M = hidx.size
        if M == 0:
            return np.empty((0, 0), dtype=dtype_out)
        if N <= 0:
            raise ValueError("N must be >= 1")
        N = int(min(N, M))

        # ---- Work in NEST internally
        if nest:
            hidx_n = hidx
        else:
            hidx_n = hp.ring2nest(nside, hidx)

        # ---- Depth / hierarchy
        D = int(np.log2(nside))
        if (1 << D) != nside:
            raise ValueError(f"nside must be a power of 2, got {nside}")
        if depth is None:
            d = self._choose_depth_for_candidates(N, overshoot=overshoot, max_depth=12)
        else:
            d = int(depth)
        d = max(0, min(d, D))
        nside_parent = nside >> d
        shift = 2 * d            # child index = (parent << (2d)) + [0..4**d - 1]
        K = 1 << shift           # 4**d children per parent (==1 if d==0)

        # ---- Local index lookup (sorted array; OOB-safe)
        perm = np.argsort(hidx_n, kind="mergesort")
        hidx_sorted = hidx_n[perm]
        M_sorted = hidx_sorted.size

        def values_to_local_indices(vals: np.ndarray):
            """Map HEALPix IDs -> local indices (0..M-1) where present, else -1 (OOB-safe)."""
            vals = np.asarray(vals, dtype=np.int64)
            k = np.searchsorted(hidx_sorted, vals, side="left")    # 0..M_sorted
            in_range = (k < M_sorted)
            eq = np.zeros(vals.shape, dtype=bool)
            if np.any(in_range):
                ki = k[in_range]
                eq[in_range] = (hidx_sorted[ki] == vals[in_range])
            valid = in_range & eq
            out = np.full(vals.shape, -1, dtype=np.int64)
            if np.any(valid):
                kv = k[valid]
                out[valid] = perm[kv]
            return out, valid

        # ---- Group centers by parent
        parents_of_center = hidx_n >> shift
        order = np.argsort(parents_of_center, kind="mergesort")
        parents_sorted = parents_of_center[order]
        uparents, starts, counts = np.unique(
            parents_sorted, return_index=True, return_counts=True
        )
        P = uparents.size

        # ---- Output buffer
        if out_memmap_path is None:
            out_local = np.empty((M, N), dtype=dtype_out)
        else:
            out_local = np.memmap(out_memmap_path, mode="w+", dtype=dtype_out, shape=(M, N))

        # ---- Precompute child block and subset vectors
        child_block = np.arange(K, dtype=np.int64)
        xs, ys, zs = hp.pix2vec(nside, hidx_n, nest=True)
        vsubset = np.stack([xs, ys, zs], axis=1).astype(vec_dtype, copy=False)   # (M,3)

        # Helper: row-wise dedup keeping score order
        def dedup_rowwise_keep_order(rows_locals):
            """
            rows_locals: (Nc, L) int array sorted by descending score per row.
            Returns a list of arrays with duplicates removed and -1 removed, preserving order.
            (Implemented with a tiny Python loop over Nc; L is small (<= ~4N), so it's cheap.)
            """
            Nc, L = rows_locals.shape
            cleaned = []
            for r in range(Nc):
                arr = rows_locals[r]
                # Drop -1 early
                arr = arr[arr >= 0]
                if arr.size == 0:
                    cleaned.append(arr)
                    continue
                # Keep first occurrences in current order (= score order)
                # np.unique would sort by value; we want to preserve order -> use mask trick
                seen = set()
                keep = []
                for a in arr:
                    if a not in seen:
                        seen.add(int(a))
                        keep.append(a)
                    if len(keep) == N:  # early stop
                        break
                cleaned.append(np.asarray(keep, dtype=np.int64))
            return cleaned

        i0 = 0
        while i0 < P:
            i1 = min(i0 + parent_batch, P)

            # ----- Parents in this batch -----
            parents_b = uparents[i0:i1]                   # (Pb,)
            Pb = parents_b.size

            # Centers slice in sorted view
            c_lo = starts[i0]
            c_hi = starts[i1] if i1 < P else parents_sorted.size
            centers_sorted_pos = np.arange(c_lo, c_hi, dtype=np.int64)
            centers_idx = order[centers_sorted_pos]       # original local indices (M,)
            centers_par = parents_sorted[centers_sorted_pos]
            Nc = centers_idx.size

            # Relative parent index in [0..Pb-1]
            rel_parent = np.searchsorted(parents_b, centers_par)

            # ----- Parent neighbours at parent NSIDE -----
            neigh8 = hp.get_all_neighbours(nside_parent, parents_b, nest=True)   # (8,Pb)
            neigh8 = np.where(neigh8 < 0, parents_b[None, :], neigh8)            # (8,Pb)
            parents_ext = np.vstack([parents_b[None, :], neigh8])                # (9,Pb)

            # ----- Enumerate children candidates at full NSIDE -----
            children_b = ((parents_ext.T[:, :, None] << shift) + child_block[None, None, :]).reshape(Pb, -1)
            Cb = children_b.shape[1]

            # ----- Candidate unit vectors (bank per parent) -----
            cand_flat = children_b.reshape(-1)                                   # (Pb*Cb,)
            xc, yc, zc = hp.pix2vec(nside, cand_flat, nest=True)
            vcand = np.stack([xc, yc, zc], axis=1).astype(vec_dtype, copy=False).reshape(Pb, Cb, 3)

            # ----- Center unit vectors for this batch -----
            xc, yc, zc = hp.pix2vec(nside, hidx_n[centers_idx], nest=True)
            vcent = np.stack([xc, yc, zc], axis=1).astype(vec_dtype, copy=False) # (Nc,3)

            # ----- Score candidates -----
            vcand_sel = vcand[rel_parent]                                        # (Nc,Cb,3)
            dots = np.sum(vcand_sel * vcent[:, None, :], axis=2)                 # (Nc,Cb)

            # ----- Map candidates -> local subset indices; mask out non-members
            local_idx_flat, is_in_subset_flat = values_to_local_indices(cand_flat)
            is_in_subset = is_in_subset_flat.reshape(Pb, Cb)[rel_parent]         # (Nc,Cb)
            local_idx_mat = local_idx_flat.reshape(Pb, Cb)[rel_parent]           # (Nc,Cb)
            dots[~is_in_subset] = np.float32(-1e30)                              # exclude

            # ----- Take a widened pool (Kpool >= 3N) to help dedup from the bank itself
            Kpool = int(min(Cb, max(3 * N, N + 16)))
            top_pool = np.argpartition(-dots, kth=Kpool - 1, axis=1)[:, :Kpool]  # (Nc,Kpool)
            pool_scores = np.take_along_axis(dots,         top_pool, axis=1)     # (Nc,Kpool)
            pool_locals = np.take_along_axis(local_idx_mat, top_pool, axis=1)    # (Nc,Kpool)
            # Order pool by descending score
            pool_order = np.argsort(-pool_scores, axis=1)
            pool_locals = np.take_along_axis(pool_locals, pool_order, axis=1)    # (Nc,Kpool)

            # ----- Row-wise dedup (preserve score order)
            dedup_lists = dedup_rowwise_keep_order(pool_locals)

            # Build provisional selection; mark rows needing fallback
            sel_locals = np.full((Nc, N), -1, dtype=np.int64)
            need_fallback = np.zeros(Nc, dtype=bool)
            for r, arr in enumerate(dedup_lists):
                if arr.size >= N:
                    sel_locals[r] = arr[:N]
                else:
                    sel_locals[r, :arr.size] = arr
                    need_fallback[r] = True

            # ----- Fallback: complete rows that still miss neighbours
            if np.any(need_fallback):
                rr = np.where(need_fallback)[0]                                   # (R,)
                # Global scores on subset
                dots_global = (vsubset @ vcent[rr].T).T                           # (R,M)
                # For each row, we must exclude already selected (non-negative) indices
                for j, rrow in enumerate(rr):
                    selected = sel_locals[rrow]
                    used = selected[selected >= 0]
                    if used.size:
                        dots_global[j, used] = np.float32(-1e30)                 # exclude already picked
                    # Pick as many as needed
                    need = N - (used.size)
                    kth_g = need - 1 if need > 0 else 0
                    top_g = np.argpartition(-dots_global[j], kth=kth_g)[:need]   # (need,)
                    # Order by score
                    scores_g = dots_global[j, top_g]
                    order_g = np.argsort(-scores_g)
                    fill = top_g[order_g]
                    sel_locals[rrow, used.size:] = fill.astype(np.int64, copy=False)

            # ----- Enforce center presence for all rows and re-rank those affected
            center_local, _ = values_to_local_indices(hidx_n[centers_idx])       # (Nc,)
            missing_center = ~np.any(sel_locals == center_local[:, None], axis=1)
            if np.any(missing_center):
                rr = np.where(missing_center)[0]
                sel_locals[rr, -1] = center_local[rr]                             # force presence
                # Re-rank only these rows by true dot
                vrows = vsubset[sel_locals[rr]]                                   # (R,N,3)
                dots_fix = np.sum(vcent[rr][:, None, :] * vrows, axis=2)          # (R,N)
                sort_fix = np.argsort(-dots_fix, axis=1)
                sel_locals[rr] = np.take_along_axis(sel_locals[rr], sort_fix, axis=1)

            # ----- Final safety: no holes, no dups. If any row still off -> full global replace.
            tmp_sorted = np.sort(sel_locals, axis=1)
            has_dup = np.any(tmp_sorted[:, 1:] == tmp_sorted[:, :-1], axis=1)
            has_neg = np.any(sel_locals < 0, axis=1)
            rows_bad = has_dup | has_neg
            if np.any(rows_bad):
                rr = np.where(rows_bad)[0]
                dots_global = (vsubset @ vcent[rr].T).T                           # (R,M)
                # Force center presence by boosting its score
                c_loc = center_local[rr]
                dots_global[np.arange(rr.size), c_loc] = np.float32(1.0)          # strictly max possible (unit dots)
                # Top-N unique from global
                top_g = np.argpartition(-dots_global, kth=N - 1, axis=1)[:, :N]
                sel_g = np.take_along_axis(top_g, np.argsort(
                    -np.take_along_axis(dots_global, top_g, axis=1), axis=1
                ), axis=1)
                out_local[centers_idx[rr], :] = sel_g.astype(dtype_out, copy=False)

                # Good rows (not bad) -> copy from sel_locals
                good = ~rows_bad
                out_local[centers_idx[good], :] = sel_locals[good].astype(dtype_out, copy=False)
            else:
                out_local[centers_idx, :] = sel_locals.astype(dtype_out, copy=False)

            i0 = i1  # next batch

        return out_local

    def all_neighbours_batched(
        self,
        hidx,
        N,
        nside,
        *,
        nest=True,
        overshoot=2.0,
        depth=None,
        parent_batch=4096,
        out_memmap_path=None,
        dtype_out=np.int64,
        vec_dtype=np.float32,   # use float32 to cut memory by half
    ):
        """
        Vectorized k-NN on a HEALPix grid (same NSIDE), scalable to NSIDE=2**20 and M~1e7.
    
        Parameters
        ----------
        hidx : (M,) int array
            Pixel indices at full NSIDE (NEST or RING according to `nest`).
        N : int
            Number of nearest neighbours to return (includes the center itself).
        nside : int
            Full-resolution NSIDE (must be a power of 2; e.g., 2**20).
        nest : bool, default True
            Input/output indexing scheme. Internally NEST is used either way.
        overshoot : float, default 2.0
            Safety factor for candidate pool size (9*4**d ≥ overshoot*N).
        depth : int or None
            Levels to go up for parent selection (None => chosen from `N` & `overshoot`).
        parent_batch : int, default 4096
            Number of unique parent pixels processed per batch (tune for memory).
        out_memmap_path : str or None
            If provided, results are written to a memmap file on disk; function
            returns the memmap view.
        dtype_out : dtype, default np.int64
            Output dtype for neighbour indices.
        vec_dtype : dtype, default np.float32
            dtype for 3D vectors (float32 reduces memory/IO footprint).
    
        Returns
        -------
        out_idx : (M, N) int array (or memmap)
            For each input pixel, indices of the N nearest pixels (same ordering as input).
        """
        #hidx = np.asarray(hidx, dtype=np.int64)
        M = hidx.size
    
        # -- Work in NEST internally
        if nest:
            hidx_n = hidx
        else:
            hidx_n = hp.ring2nest(nside, hidx)
    
        # -- Choose hierarchy depth d (how far up we go)
        if depth is None:
            d = self._choose_depth_for_candidates(N, overshoot=overshoot, max_depth=12)
        else:
            d = int(depth)
    
        # Clamp d by available depth (nside = 2**D)
        D = int(np.log2(nside))
        if d > D:
            d = D
        nside_parent = nside >> d
        shift = 2 * d          # child index = (parent << (2d)) + [0..4**d-1]
        K = 1 << shift         # 4**d children per parent
    
        # -- Parent for each center (bit-shift in NEST)
        parents_of_center = hidx_n >> shift
    
        # -- Group by parent (single sort; no loop over centers)
        order = np.argsort(parents_of_center, kind="mergesort")
        parents_sorted = parents_of_center[order]
        uparents, starts, counts = np.unique(
            parents_sorted, return_index=True, return_counts=True
        )
        P = uparents.size
    
        # -- Prepare output (RAM or memmap)
        if out_memmap_path is None:
            out_idx_n = np.empty((M, N), dtype=dtype_out)
        else:
            out_idx_n = np.memmap(out_memmap_path, mode="w+", dtype=dtype_out, shape=(M, N))
    
        # -- Precompute child block once (vectorized refinement back down)
        child_block = np.arange(K, dtype=np.int64)
    
        i0 = 0
        while i0 < P:
            i1 = min(i0 + parent_batch, P)
    
            # ---------- parents in this batch ----------
            parents_b = uparents[i0:i1]                 # (Pb,)
            Pb = parents_b.size
    
            # Centers for these parents = single contiguous slice in sorted view
            c_lo = starts[i0]
            c_hi = starts[i1] if i1 < P else parents_sorted.size
            centers_sorted_pos = np.arange(c_lo, c_hi, dtype=np.int64)
            centers_idx = order[centers_sorted_pos]     # indices into original (M,)
            centers_par = parents_sorted[centers_sorted_pos]  # parent id per center, sorted
    
            # Map each center to relative parent index in [0..Pb-1]
            rel_parent = np.searchsorted(parents_b, centers_par)  # (Nc,)
            Nc = rel_parent.size
    
            # ---------- 8 neighbours at parent NSIDE (vectorized) ----------
            # Returns shape (8, Pb); -1 near boundaries -> replace by parent itself
            neigh8 = hp.get_all_neighbours(nside_parent, parents_b, nest=True)
            neigh8 = np.where(neigh8 < 0, parents_b[None, :], neigh8)  # (8,Pb)
            # Stack center parent + its neighbours: (9,Pb)
            parents_ext = np.vstack([parents_b[None, :], neigh8])
    
            # ---------- Enumerate children at full NSIDE ----------
            # children_b: (Pb, 9*K) = for each parent, list of refined children for (self+8 neigh)
            children_b = ((parents_ext.T[:, :, None] << shift) + child_block[None, None, :]).reshape(Pb, -1)
            Cb = children_b.shape[1]  # candidate count per parent
    
            # ---------- Candidate unit vectors (one bank per parent) ----------
            cand_flat = children_b.reshape(-1)                         # (Pb*Cb,)
            xc, yc, zc = hp.pix2vec(nside, cand_flat, nest=True)       # each (Pb*Cb,)
            vcand = np.stack([xc, yc, zc], axis=1).astype(vec_dtype, copy=False).reshape(Pb, Cb, 3)
    
            # ---------- Center unit vectors ----------
            xc, yc, zc = hp.pix2vec(nside, hidx_n[centers_idx], nest=True)  # each (Nc,)
            vcent = np.stack([xc, yc, zc], axis=1).astype(vec_dtype, copy=False)  # (Nc,3)
    
            # ---------- Select per-center candidate bank & score ----------
            # (Nc,Cb,3) broadcast-multiply with (Nc,1,3) -> sum over axis=-1 -> (Nc,Cb)
            vcand_sel = vcand[rel_parent]                    # (Nc,Cb,3)
            dots = np.sum(vcand_sel * vcent[:, None, :], axis=2)  # (Nc,Cb)
    
            # ---------- Top-N indices per row (argpartition is O(Cb)) ----------
            kth = min(N - 1, Cb - 1)
            top_local = np.argpartition(-dots, kth=kth, axis=1)[:, :N]     # (Nc,N)
    
            # Map local candidate indices back to global pixel ids
            children_sel = children_b[rel_parent]                          # (Nc,Cb)
            best_pix = children_sel[np.arange(Nc)[:, None], top_local]     # (Nc,N)
    
            # Optional: stable sort by distance (descending dot) for nicer ordering
            sel_dots = dots[np.arange(Nc)[:, None], top_local]
            sort_in_row = np.argsort(-sel_dots, axis=1)
            best_pix = best_pix[np.arange(Nc)[:, None], sort_in_row]       # (Nc,N)
    
            # ---------- Ensure the center pixel is present ----------
            # (rarely needed; keeps exact behaviour)
            missing = ~np.any(best_pix == hidx_n[centers_idx][:, None], axis=1)
            if np.any(missing):
                rr = np.where(missing)[0]
                best_pix[rr, -1] = hidx_n[centers_idx][rr]
                # Re-sort only modified rows
                xb, yb, zb = hp.pix2vec(nside, best_pix[rr].ravel(), nest=True)
                vb = np.stack([xb, yb, zb], axis=1).astype(vec_dtype, copy=False).reshape(rr.size, N, 3)
                dots_fix = np.sum(vcent[rr][:, None, :] * vb, axis=2)      # (rr,N)
                sort_fix = np.argsort(-dots_fix, axis=1)
                best_pix[rr] = best_pix[rr, np.arange(N)[None, :]][np.arange(rr.size)[:, None], sort_fix]
    
            # ---------- Write back to output (original order) ----------
            out_idx_n[centers_idx, :] = best_pix
    
            i0 = i1  # next parent batch
    
        # Convert back to RING if needed
        if not nest:
            out_idx = hp.nest2ring(nside, out_idx_n)
            return out_idx
        return out_idx_n

    def make_wavelet_matrix(self,
                            orientations,
                            polar=True,
                            norm_mean=True,
                            norm_std=True,
                            return_index=False,
                            return_smooth=False,
                           ):
        
        sigma_gauss = 0.5
        sigma_cosine = 0.5
        if self.KERNELSZ == 5:
            sigma_gauss = 0.5
            sigma_cosine = 0.5
        elif self.KERNELSZ == 3:
            sigma_gauss = 1.0 / np.sqrt(2)
            sigma_cosine = 1.0

        elif self.KERNELSZ == 7:
            sigma_gauss = 0.5
            sigma_cosine = 0.25

        orientations=np.asarray(orientations)
        NORIENT = orientations.shape[0]
        
        rotate=2*((self.t<np.pi/2)-0.5)[None,:,None]
        if polar:
            xx=np.cos(self.p[None,:]+np.pi/2-orientations[:,None])[:,:,None]*self.vec_rot[None,:,:,0]-rotate*np.sin(self.p[None,:]+np.pi/2-orientations[:,None])[:,:,None]*self.vec_rot[None,:,:,1]
        else:
            xx=np.cos(np.pi/2-orientations[:,None,None])*self.vec_rot[None,:,:,0]-np.sin(np.pi/2-orientations[:,None,None])*self.vec_rot[None,:,:,1]
            
        r=(self.vec_rot[None,:,:,0]**2+self.vec_rot[None,:,:,1]**2+(self.vec_rot[None,:,:,2]-1.0)**2)
        
        if return_smooth:
            wsmooth=np.exp(-sigma_gauss*r*self.nside**2)
            '''
            if self.local_test:
                idx=np.where(self.idx_nn==-1)
                wsmooth[0,idx[0],idx[1]]=0.0
            '''    
            if norm_std:
                ww=np.sum(wsmooth,2)
                #print(ww.min(),ww.max())
                wsmooth = wsmooth/ww[:,:,None]

        #for consistency with previous definition
        w=np.exp(-sigma_gauss*r*self.nside**2)*(np.cos(xx*self.nside*sigma_cosine*np.pi)-1J*np.sin(xx*self.nside*sigma_cosine*np.pi))
        '''
        if self.local_test:
            idx=np.where(self.idx_nn==-1)
            for k in range(NORIENT):
                w[k,idx[0],idx[1]]=0.0
        '''        
        if norm_std:
            ww=1/np.sum(abs(w),2)[:,:,None] 
        else:
            ww=1.0
        if norm_mean:
            w = (w.real-np.mean(w.real,2)[:,:,None]+1J*(w.imag-np.mean(w.imag,2)[:,:,None]))*ww
            
        NK=self.idx_nn.shape[1]
        indice_1_0 = np.tile(self.idx_nn.flatten(),NORIENT)
        indice_1_1 = np.tile(np.repeat(self.idx_nn[:,0],NK),NORIENT)+ \
            np.repeat(np.arange(NORIENT),self.idx_nn.shape[0]*self.idx_nn.shape[1])*self.idx_nn.shape[0]
        w = w.flatten()

        '''
        if self.local_test:
            w[indice_1_0==-1]=0.0
            indice_1_0[indice_1_0==-1]=0
        '''
        if return_smooth:
            indice_2_0 = self.idx_nn.flatten()
            indice_2_1 = np.repeat(self.idx_nn[:,0],NK)
            wsmooth = wsmooth.flatten()
            '''
            if self.local_test:
                wsmooth[indice_2_0==-1]=0.0
                indice_2_0[indice_2_0==-1]=0
            '''    
        if return_index:
            if return_smooth:
                return w,np.concatenate([indice_1_0[:,None],indice_1_1[:,None]],1),wsmooth,np.concatenate([indice_2_0[:,None],indice_2_1[:,None]],1)
            
            return w,np.concatenate([indice_1_0[:,None],indice_1_1[:,None]],1)
        
        return csr_array((w, (indice_1_0, indice_1_1)), shape=(12*self.nside**2, 12*self.nside**2*NORIENT))

    
    def make_idx_weights(self,polar=False,gamma=1.0,device='cuda',allow_extrapolation=True):
        
        rotate=2*((self.t<np.pi/2)-0.5)[:,None]
        if polar:
            xx=np.cos(self.p)[:,None]*self.vec_rot[:,:,0]-rotate*np.sin(self.p)[:,None]*self.vec_rot[:,:,1]
            yy=-np.sin(self.p)[:,None]*self.vec_rot[:,:,0]-rotate*np.cos(self.p)[:,None]*self.vec_rot[:,:,1]
        else:
            xx=self.vec_rot[:,:,0]
            yy=self.vec_rot[:,:,1]
        
        self.w_idx,self.w_w = self.bilinear_weights_NxN(xx*nside*gamma,
                                                        yy*nside*gamma,
                                                        allow_extrapolation=allow_extrapolation)

        # Ensure types/devices
        self.idx_nn = torch.Tensor(self.idx_nn).to(device=device, dtype=torch.long)
        self.w_idx  = torch.Tensor(self.w_idx).to(device=device, dtype=torch.long)
        self.w_w    = torch.Tensor(self.w_w).to(device=device, dtype=torch.float64)
    
    def _grid_index(self, xi, yi):
        """
        Map integer grid coords (xi, yi) in {-1,0,1} to flat index in [0..8]
        following the given order (row-major from y=-1 to y=1).
        """
        return (yi + 1) * 3 + (xi + 1)
    
    def bilinear_weights_NxN(self,x, y, allow_extrapolation=True):
        """
        Compute bilinear weights on an N×N integer grid with node coordinates
        (xi, yi) in {-K, ..., +K} × {-K, ..., +K}, where K = N//2 (N must be odd).

        N is attached to the class `N = self.KERNELSZ`
        
        The query point (x, y) is continuous in the same coordinate system.
        For each query, we pick the unit cell [x0, x0+1] × [y0, y0+1] with
        integer corners (x0,y0), (x0+1,y0), (x0,y0+1), (x0+1,y0+1), and compute
        standard bilinear weights relative to (x0, y0).
    
        Parameters
        ----------
        x, y : float or array-like of shape (M,)
            Query coordinates in the integer grid coordinate system.
        N : int
            Grid size (must be odd). Grid nodes are at integer coords
            xi, yi ∈ {-K, ..., +K}, where K = N//2.
        allow_extrapolation : bool, default True
            - If False: clamp (x, y) to [-K, +K] so that tx, ty ∈ [0, 1] and
              weights are non-negative and sum to 1.
            - If True : do not clamp (x, y); we still select the nearest boundary
              cell inside the grid for the indices, but tx, ty may fall outside
              [0, 1], yielding extrapolation (weights can be negative).
    
        Returns
        -------
        idx : ndarray of shape (M, 4), dtype=int64
            Flat indices (0 .. N*N-1) of the four cell-corner nodes in row-major
            order (y from -K to +K, x from -K to +K):
            order = [(x0,y0), (x0+1,y0), (x0,y0+1), (x0+1,y0+1)].
        w : ndarray of shape (M, 4), dtype=float64
            Corresponding bilinear weights for each query point. If
            allow_extrapolation=False and the point is inside the grid, each row
            sums to 1 and all weights are in [0,1].
    
        Notes
        -----
        - This matches your previous 3×3 case when N=3, with the same row-major
          flattening convention.
        - For extrapolation=True, indices are kept in-bounds (clamped to boundary
          cells), while tx, ty > 1 or < 0 are allowed.
        """
        # --- checks & shapes ---
        N=self.KERNELSZ
        
        K = N // 2
    
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        M = x.shape[0]
    
        # --- optionally clamp queries (for pure interpolation) ---
        if not allow_extrapolation:
            x = np.clip(x, -K, K)
            y = np.clip(y, -K, K)
    
        # --- choose the cell: x0=floor(x), y0=floor(y), but keep indices in-bounds
        #     cell must be inside [-K..K-1] × [-K..K-1] so that +1 is valid
        x0 = np.floor(x)
        y0 = np.floor(y)
        x0 = np.clip(x0, -K, K - 1).astype(int)
        y0 = np.clip(y0, -K, K - 1).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
    
        # --- local coords within the cell (unit spacing) ---
        tx = x - x0
        ty = y - y0
    
        # --- bilinear weights ---
        # (x0,y0) w00, (x1,y0) w10, (x0,y1) w01, (x1,y1) w11
        w00 = (1.0 - tx) * (1.0 - ty)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty
        w = np.stack([w00, w10, w01, w11], axis=1)
    
        # --- flat indices in row-major order (y changes slowest) ---
        # index = (yi + K) * N + (xi + K)
        def flat_idx(xi, yi):
            return (yi + K) * N + (xi + K)
    
        i00 = flat_idx(x0, y0)
        i10 = flat_idx(x1, y0)
        i01 = flat_idx(x0, y1)
        i11 = flat_idx(x1, y1)
        idx = np.stack([i00, i10, i01, i11], axis=1).astype(np.int64)
    
        return idx, w

    def Convol_torch(self, im, ww):
        """
        Batched KERNELSZxKERNELSZ neighborhood aggregation in pure PyTorch (generalization of the 3x3 case).
    
        Parameters
        ----------
        im : Tensor, shape (B, C_i, Npix)
            Input features per pixel for a batch of B samples.
        ww : Tensor
            Base mixing weights, indexed along its 'M' dimension by self.w_idx.
            Supported shapes:
              (C_i, C_o, M)
              (C_i, C_o, M, S)
              (B, C_i, C_o, M)
              (B, C_i, C_o, M, S)
    
        Class members (already tensors; will be aligned to im.device/dtype):
        -------------------------------------------------------------------
        self.idx_nn : LongTensor, shape (Npix, P)
            For each center pixel, the P neighbor indices into the Npix axis of `im`.
            (P = K*K for a KxK neighborhood.)
        self.w_idx  : LongTensor, shape (Npix, P) or (Npix, S, P)
            Indices along the 'M' dimension of ww, per (center[, sector], neighbor).
        self.w_w    : Tensor,     shape (Npix, P) or (Npix, S, P)
            Additional scalar weights per neighbor (same layout as w_idx).
    
        Returns
        -------
        out : Tensor, shape (B, C_o, Npix)
            Aggregated output per center pixel for each batch sample.
        """
        # ---- Basic checks ----
        assert im.ndim == 3, f"`im` must be (B, C_i, Npix), got {tuple(im.shape)}"
        assert ww.shape[2]==self.KERNELSZ*self.KERNELSZ, f"`ww` must be (C_i, C_o, KERNELSZ*KERNELSZ), got {tuple(ww.shape)}"
        
        B, C_i, Npix = im.shape
        device = im.device
        dtype  = im.dtype
    
        # Align class tensors to device/dtype
        idx_nn = self.idx_nn.to(device=device, dtype=torch.long)  # (Npix, P)
        w_idx  = self.w_idx.to(device=device, dtype=torch.long)   # (Npix, P) or (Npix, S, P)
        w_w    = self.w_w.to(device=device, dtype=dtype)          # (Npix, P) or (Npix, S, P)
    
        # Neighbor count P inferred from idx_nn
        assert idx_nn.ndim == 2 and idx_nn.size(0) == Npix, \
            f"`idx_nn` must be (Npix, P) with Npix={Npix}, got {tuple(idx_nn.shape)}"
        P = idx_nn.size(1)
    
        # ---- 1) Gather neighbor values from im along the Npix dimension -> (B, C_i, Npix, P)
        # im: (B,C_i,Npix) -> (B,C_i,Npix,1); idx: (1,1,Npix,P) broadcast over (B,C_i)
        rim = torch.take_along_dim(
            im.unsqueeze(-1),
            idx_nn.unsqueeze(0).unsqueeze(0),
            dim=2
        )  # (B, C_i, Npix, P)
    
        # ---- 2) Normalize w_idx / w_w to include a sector dim S ----
        # Target layout: (Npix, S, P)
        if w_idx.ndim == 2:
            # (Npix, P) -> add sector dim S=1
            assert w_idx.size(0) == Npix and w_idx.size(1) == P
            w_idx_eff = w_idx.unsqueeze(1)  # (Npix, 1, P)
            w_w_eff   = w_w.unsqueeze(1)    # (Npix, 1, P)
            S = 1
        elif w_idx.ndim == 3:
            # (Npix, S, P)
            Npix_, S, P_ = w_idx.shape
            assert Npix_ == Npix and P_ == P, \
                f"`w_idx` must be (Npix,S,P) with Npix={Npix}, P={P}, got {tuple(w_idx.shape)}"
            assert w_w.shape == w_idx.shape, "`w_w` must match `w_idx` shape"
            w_idx_eff = w_idx
            w_w_eff   = w_w
        else:
            raise ValueError(f"Unsupported `w_idx` shape {tuple(w_idx.shape)}; expected (Npix,P) or (Npix,S,P)")
    
        # ---- 3) Normalize ww to (B, C_i, C_o, M, S) for uniform gather ----
        if ww.ndim == 3:
            # (C_i, C_o, M) -> (B, C_i, C_o, M, S)
            C_i_w, C_o, M = ww.shape
            assert C_i_w == C_i, f"ww C_i mismatch: {C_i_w} vs im {C_i}"
            ww_eff = ww.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1, S)
    
        elif ww.ndim == 4:
            # Could be (C_i, C_o, M, S) or (B, C_i, C_o, M)
            if ww.shape[0] == C_i and ww.shape[1] != C_i:
                # (C_i, C_o, M, S) -> (B, C_i, C_o, M, S)
                C_i_w, C_o, M, S_w = ww.shape
                assert C_i_w == C_i, f"ww C_i mismatch: {C_i_w} vs im {C_i}"
                assert S_w == S, f"ww S mismatch: {S_w} vs w_idx S {S}"
                ww_eff = ww.unsqueeze(0).expand(B, -1, -1, -1, -1)
            elif ww.shape[0] == B:
                # (B, C_i, C_o, M) -> (B, C_i, C_o, M, S)
                _, C_i_w, C_o, M = ww.shape
                assert C_i_w == C_i, f"ww C_i mismatch: {C_i_w} vs im {C_i}"
                ww_eff = ww.unsqueeze(-1).expand(-1, -1, -1, -1, S)
            else:
                raise ValueError(
                    f"Ambiguous 4D ww shape {tuple(ww.shape)}; expected (C_i,C_o,M,S) or (B,C_i,C_o,M)"
                )
    
        elif ww.ndim == 5:
            # (B, C_i, C_o, M, S)
            assert ww.shape[0] == B and ww.shape[1] == C_i, "ww batch/C_i mismatch"
            _, _, _, M, S_w = ww.shape
            assert S_w == S, f"ww S mismatch: {S_w} vs w_idx S {S}"
            ww_eff = ww
        else:
            raise ValueError(f"Unsupported ww shape {tuple(ww.shape)}")
    
        # ---- 4) Gather along M using w_idx_eff -> (B, C_i, C_o, Npix, S, P)
        idx_exp = w_idx_eff.unsqueeze(0).unsqueeze(0).unsqueeze(0)     # (1,1,1,Npix,S,P)
        rw = torch.take_along_dim(
            ww_eff.unsqueeze(-1),  # (B, C_i, C_o, M, S, 1)
            idx_exp,               # (1,1,1,Npix,S,P) -> broadcast
            dim=3                  # gather along M
        )  # -> (B, C_i, C_o, Npix, S, P)
    
        # ---- 5) Apply extra neighbor weights ----
        rw = rw * w_w_eff.unsqueeze(0).unsqueeze(0).unsqueeze(0)       # (B, C_i, C_o, Npix, S, P)
    
        # ---- 6) Combine neighbor values and weights ----
        # rim: (B, C_i, Npix, P) -> expand to (B, C_i, 1, Npix, 1, P)
        rim_exp = rim[:, :, None, :, None, :]
        # sum over neighbors (P), then over sectors (S), then over input channels (C_i)
        out_ci  = (rim_exp * rw).sum(dim=-1)    # (B, C_i, C_o, Npix, S)
        out_ci  = out_ci.sum(dim=-1)            # (B, C_i, C_o, Npix)
        out     = out_ci.sum(dim=1)             # (B, C_o, Npix)
    
        return out
