import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.sparse import csr_array
import torch

class HOrientedConvol:
    def __init__(self,nside,KERNELSZ,cell_ids=None,nest=True):
        
        if KERNELSZ % 2 == 0:
            raise ValueError(f"N must be odd so that coordinates are integers from -K..K; got N={N}.")
            
        if cell_ids is None:
            self.cell_ids=np.arange(12*nside**2)
        else:
            self.cell_ids=cell_ids

        idx_nn = self.all_neighbours_batched(
            self.cell_ids, 
            KERNELSZ*KERNELSZ, 
            nside=nside, 
            nest=nest,
            overshoot=2.0,      # candidate pool ≈ 2×N
            parent_batch=4096,  # tune to fit memory
            # out_memmap_path="neighbors_nest.int64.memmap",  # enable for huge M to be tested
            )

        mat_pt=self.rotation_matrices_from_healpix(nside,idx_nn[:,0],nest=nest)

        t,p = hp.pix2ang(nside,idx_nn,nest=True)
        vec_orig=hp.ang2vec(t,p)

        self.vec_rot = np.einsum('mki,ijk->kmj', vec_orig,mat_pt)
        del mat_pt
        del vec_orig
        self.t=t[:,0]
        self.p=p[:,0]
        self.idx_nn=idx_nn
        self.nside=nside
        self.KERNELSZ=KERNELSZ
        
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
        hpix_idx = np.asarray(hpix_idx)
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
        hidx = np.asarray(hidx, dtype=np.int64)
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
                            polar=False,
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
            xx=np.cos(self.p[None,:]+orientations[:,None])[:,:,None]*self.vec_rot[None,:,:,0]-rotate*np.sin(self.p[None,:]+orientations[:,None])[:,:,None]*self.vec_rot[None,:,:,1]
        else:
            xx=np.cos(orientations[:,None,None])*self.vec_rot[None,:,:,0]-np.sin(orientations[:,None,None])*self.vec_rot[None,:,:,1]
            
        r=(self.vec_rot[None,:,:,0]**2+self.vec_rot[None,:,:,1]**2)
        
        if return_smooth:
            wsmooth=np.exp(-sigma_gauss*r*self.nside**2)
            
        w=np.exp(-sigma_gauss*r*self.nside**2)*(np.cos(xx*self.nside*sigma_cosine*np.pi)+1J*np.sin(xx*self.nside*sigma_cosine*np.pi))
        
        if norm_mean:
            w = w.real-np.mean(w.real,2)[:,:,None]+1J*(w.imag-np.mean(w.imag,2)[:,:,None])
        if norm_std:
            w = w/(NORIENT*np.std(w,2)[:,:,None])

        if return_index:
            if return_smooth:
                return w.flatten(),np.concatenate([np.tile(self.idx_nn.flatten(),NORIENT)[:,None],
                                               (np.tile(np.repeat(self.idx_nn[:,0],self.KERNELSZ*self.KERNELSZ),NORIENT)+
                                               np.repeat(np.arange(NORIENT),self.idx_nn.shape[0]*self.idx_nn.shape[1])*12*self.nside**2)[:,None]],1),wsmooth.flatten(),np.concatenate([self.idx_nn.flatten()[:,None],
                                               np.repeat(self.idx_nn[:,0],self.KERNELSZ*self.KERNELSZ)[:,None]],1)
            
            return w.flatten(),np.concatenate([np.tile(self.idx_nn.flatten(),NORIENT)[:,None],
                                               (np.tile(np.repeat(self.idx_nn[:,0],self.KERNELSZ*self.KERNELSZ),NORIENT)+
                                               np.repeat(np.arange(NORIENT),self.idx_nn.shape[0]*self.idx_nn.shape[1])*12*self.nside**2)[:,None]],1)
        return csr_array((w.flatten(), 
                              (np.tile(self.idx_nn.flatten(),NORIENT), 
                               np.tile(np.repeat(self.idx_nn[:,0],self.KERNELSZ*self.KERNELSZ),NORIENT)
                               +np.repeat(np.arange(NORIENT),self.idx_nn.shape[0]*self.idx_nn.shape[1])*12*self.nside**2)),
                                 shape=(12*self.nside**2, 12*self.nside**2*NORIENT))

    
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
