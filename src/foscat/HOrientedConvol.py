import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.sparse import csr_array
import torch
from scipy.spatial import cKDTree

class HOrientedConvol:
    def __init__(self,nside,KERNELSZ,cell_ids=None,nest=True):
        
        if KERNELSZ % 2 == 0:
            raise ValueError(f"N must be odd so that coordinates are integers from -K..K; got N={KERNELSZ}.")

        self.local_test=False
        
        if cell_ids is None:
            self.cell_ids=np.arange(12*nside**2)
            
            idx_nn = self.knn_healpix_ckdtree(self.cell_ids, 
                KERNELSZ*KERNELSZ, 
                nside,
                nest=nest,
            )
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
        if self.KERNELSZ == 3:
            sigma_gauss = 1.0 / np.sqrt(2)
            sigma_cosine = 1.0

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
            if norm_std:
                ww=np.sum(wsmooth,2)
                wsmooth = wsmooth/ww[:,:,None]

        #for consistency with previous definition
        w=np.exp(-sigma_gauss*r*self.nside**2)*(np.cos(xx*self.nside*sigma_cosine*np.pi)-1J*np.sin(xx*self.nside*sigma_cosine*np.pi))
              
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

        if return_smooth:
            indice_2_0 = self.idx_nn.flatten()
            indice_2_1 = np.repeat(self.idx_nn[:,0],NK)
            wsmooth = wsmooth.flatten()
            
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
        
        self.w_idx,self.w_w = self.bilinear_weights_NxN(xx*self.nside*gamma,
                                                        yy*self.nside*gamma,
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
        return (yi + self.KERNELSZ//2) * self.KERNELSZ + (xi + self.KERNELSZ//2)
    
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

   
        
