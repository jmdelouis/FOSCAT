import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.sparse import csr_array
import torch
import foscat.scat_cov as sc
from scipy.spatial import cKDTree

class HOrientedConvol:
    def __init__(self,
                 nside,
                 KERNELSZ,
                 cell_ids=None,
                 nest=True,
                 device='cuda',
                 dtype='float64',
                 polar=False,
                 gamma=1.0,
                 allow_extrapolation=True,
                 no_cell_ids=False,
                 ):


        if dtype=='float64':
            self.dtype=torch.float64
        else:
            self.dtype=torch.float32
        
        if KERNELSZ % 2 == 0:
            raise ValueError(f"N must be odd so that coordinates are integers from -K..K; got N={KERNELSZ}.")

        self.local_test=False

        if no_cell_ids==True:
            cell_ids=np.arange(10)
            
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

            if self.cell_ids.ndim==1:
                idx_nn = self.knn_healpix_ckdtree(self.cell_ids, 
                                                  KERNELSZ*KERNELSZ, 
                                                  nside,
                                                  nest=nest,
                                                  )
            else:
                idx_nn = []
                for k in range(self.cell_ids.shape[0]):
                    idx_nn.append(self.knn_healpix_ckdtree(self.cell_ids[k], 
                                                           KERNELSZ*KERNELSZ, 
                                                           nside,
                                                           nest=nest,
                                                           ))
                idx_nn=np.stack(idx_nn,0)
        
        if self.cell_ids.ndim==1:
            mat_pt=self.rotation_matrices_from_healpix(nside,self.cell_ids,nest=nest)
            
            if self.local_test:
                t,p = hp.pix2ang(nside,self.cell_ids[idx_nn],nest=True)
            else:
                t,p = hp.pix2ang(nside,idx_nn,nest=True)

            self.t=t[:,0]
            self.p=p[:,0]
            vec_orig=hp.ang2vec(t,p)

            self.vec_rot = np.einsum('mki,ijk->kmj', vec_orig,mat_pt)

            '''
            if self.local_test:
            idx_nn=self.remap_by_first_column(idx_nn)
            '''
            
            del mat_pt
            del vec_orig
        else:
            
            t,p,vec_rot = [],[],[]
            
            for k in range(self.cell_ids.shape[0]):
                mat_pt=self.rotation_matrices_from_healpix(nside,self.cell_ids[k],nest=nest)
                
                lt,lp = hp.pix2ang(nside,self.cell_ids[k,idx_nn[k]],nest=True)
                
                vec_orig=hp.ang2vec(lt,lp)
                
                l_vec_rot=np.einsum('mki,ijk->kmj', vec_orig,mat_pt)
                vec_rot.append(l_vec_rot)
                
                del vec_orig
                del mat_pt
                
                t.append(lt[:,0])
                p.append(lp[:,0])

                    
            self.t=np.stack(t,0)
            self.p=np.stack(p,0)
            self.vec_rot=np.stack(vec_rot,0)

            del t
            del p
            del vec_rot
                
        self.polar=polar
        self.gamma=gamma
        self.device=device
        self.allow_extrapolation=allow_extrapolation
        self.w_idx=None
        
        self.idx_nn=idx_nn
        self.nside=nside
        self.KERNELSZ=KERNELSZ
        self.nest=nest
        self.f=None

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

    def make_idx_weights_from_cell_ids(self,
                                       i_cell_ids,
                                       polar=False,
                                       gamma=1.0,
                                       device='cuda',
                                       allow_extrapolation=True):
        """
        Accept 1D (Npix,) or 2D (B, Npix) cell_ids and return
        tensors batched sur la 1ère dim (B, ...).
        """
        # → cast numpy
        if torch.is_tensor(i_cell_ids):
            cid = i_cell_ids.detach().cpu().numpy()
        else:
            cid = np.asarray(i_cell_ids)

        # --- 1D: pas de boucle, on calcule une fois, puis on ajoute l'axe batch
        if cid.ndim == 1:
            l_idx_nn, l_w_idx, l_w_w = self.make_idx_weights_from_one_cell_ids(
                cid, polar=polar, gamma=gamma, device=device,
                allow_extrapolation=allow_extrapolation
            )
            idx_nn = torch.as_tensor(l_idx_nn, device=device, dtype=torch.long)[None, ...]  # (1, Npix, P)
            w_idx  = torch.as_tensor(l_w_idx,  device=device, dtype=torch.long)[None, ...]  # (1, Npix, S, P) ou (1, Npix, P)
            w_w    = torch.as_tensor(l_w_w,    device=device, dtype=self.dtype)[None, ...]  # (1, Npix, S, P) ou (1, Npix, P)
            return idx_nn, w_idx, w_w

        # --- 2D: boucle sur b, empilement en (B, ...)
        elif cid.ndim == 2:
            outs = [ self.make_idx_weights_from_one_cell_ids(
                        cid[k], polar=polar, gamma=gamma, device=device,
                        allow_extrapolation=allow_extrapolation)
                     for k in range(cid.shape[0]) ]
            idx_nn = torch.as_tensor(np.stack([o[0] for o in outs], axis=0), device=device, dtype=torch.long)
            w_idx  = torch.as_tensor(np.stack([o[1] for o in outs], axis=0), device=device, dtype=torch.long)
            w_w    = torch.as_tensor(np.stack([o[2] for o in outs], axis=0), device=device, dtype=self.dtype)
            return idx_nn, w_idx, w_w

        else:
            raise ValueError(f"Unsupported cell_ids ndim={cid.ndim}; expected 1 or 2.")
    '''
    def make_idx_weights_from_cell_ids(self,i_cell_ids,
                                       polar=False,
                                       gamma=1.0,
                                       device='cuda',
                                       allow_extrapolation=True):
        if len(i_cell_ids.shape)<2:
            cell_ids=i_cell_ids
            n_cids=1
        else:
            cell_ids=i_cell_ids[0]
            n_cids=i_cell_ids.shape[0]
            
        idx_nn,w_idx,w_w    = [],[],[]
        
        for k in range(n_cids):
            cell_ids=i_cell_ids[k]
            l_idx_nn,l_w_idx,l_w_w = self.make_idx_weights_from_one_cell_ids(cell_ids,
                                                                       polar=polar,
                                                                       gamma=gamma,
                                                                       device=device,
                                                                       allow_extrapolation=allow_extrapolation)
            idx_nn.append(l_idx_nn)
            w_idx.append(l_w_idx)
            w_w.append(l_w_w)
            
        idx_nn = torch.Tensor(np.stack(idx_nn,0)).to(device=device, dtype=torch.long)
        w_idx  = torch.Tensor(np.stack(w_idx,0)).to(device=device, dtype=torch.long)
        w_w    = torch.Tensor(np.stack(w_w,0)).to(device=device, dtype=self.dtype)
        
        return idx_nn,w_idx,w_w
    '''

    def make_idx_weights_from_one_cell_ids(self,
                                           cell_ids,
                                           polar=False,
                                           gamma=1.0,
                                           device='cuda',
                                           allow_extrapolation=True):

        idx_nn = self.knn_healpix_ckdtree(cell_ids, 
                                          self.KERNELSZ*self.KERNELSZ, 
                                          self.nside,
                                          nest=self.nest,
                                          )
        
        mat_pt=self.rotation_matrices_from_healpix(self.nside,cell_ids,nest=self.nest)

        t,p = hp.pix2ang(self.nside,cell_ids[idx_nn],nest=self.nest)
            
        vec_orig=hp.ang2vec(t,p)

        vec_rot = np.einsum('mki,ijk->kmj', vec_orig,mat_pt)
        
        del vec_orig
        del mat_pt
        
        rotate=2*((t<np.pi/2)-0.5)[:,None]
        if polar:
            xx=np.cos(p)[:,None]*vec_rot[:,:,0]-rotate*np.sin(p)[:,None]*vec_rot[:,:,1]
            yy=-np.sin(p)[:,None]*vec_rot[:,:,0]-rotate*np.cos(p)[:,None]*vec_rot[:,:,1]
        else:
            xx=vec_rot[:,:,0]
            yy=vec_rot[:,:,1]

        del vec_rot
        del rotate
        del t
        del p
        
        w_idx,w_w = self.bilinear_weights_NxN(xx*self.nside*gamma,
                                              yy*self.nside*gamma,
                                              allow_extrapolation=allow_extrapolation)
        '''
        # calib : [Npix, K]
        calib = np.zeros((w_idx.shape[0], w_idx.shape[2]))
        # Hypothèses : 
        # w_idx.shape == (Npix, M, K)  et  w_w.shape == (Npix, M, K)
        Npix, M, K = w_idx.shape
        nb_cols = K
        
        # 1) Accumulation par "bincount" avec décalage de ligne
        row_ids = np.arange(Npix, dtype=np.int64)[:, None, None] * nb_cols
        flat_idx = (row_ids + w_idx).ravel()           # indices dans [0, Npix*9)
        weights  = w_w.ravel().astype(np.float64)      # ou dtype de ton choix
        
        calib = np.bincount(flat_idx, weights, minlength=Npix*nb_cols)\
                  .reshape(Npix, nb_cols)
        
        # 2) Réinjection dans norm_a selon w_idx
        norm_a = calib[np.arange(Npix)[:, None, None], w_idx]

        w_w /= norm_a
        w_w  = np.clip(w_w,0.0,1.0)

        w_w[np.isnan(w_w)]=0.0
        '''
        #del xx
        #del yy
        
        return idx_nn,w_idx,w_w,xx,yy
        
    def make_idx_weights(self,polar=False,gamma=1.0,device='cuda',allow_extrapolation=True,return_index=False):
        
        idx_nn,w_idx,w_w = self.make_idx_weights_from_one_cell_ids(self.cell_ids,
                                                                  polar=polar,
                                                                  gamma=gamma,
                                                                  device=device,
                                                                  allow_extrapolation=allow_extrapolation)
        
        # Ensure types/devices
        self.idx_nn = torch.Tensor(idx_nn).to(device=device, dtype=torch.long)
        self.w_idx  = torch.Tensor(w_idx).to(device=device, dtype=torch.long)
        self.w_w    = torch.Tensor(w_w).to(device=device, dtype=self.dtype)
    
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
    
    # --- Add inside class HOrientedConvol, just above Convol_torch ---
    def _convol_single(self, im1: torch.Tensor, ww: torch.Tensor, cell_ids=None, nside=None):
        """
        Single-sample path. im1: (1, C_i, Npix_1). Returns (1, C_o, Npix_1).
        """
        if not isinstance(im1, torch.Tensor):
            im1 = torch.as_tensor(im1, device=self.device, dtype=self.dtype)
        if not isinstance(ww, torch.Tensor):
            ww  = torch.as_tensor(ww, device=self.device, dtype=self.dtype)
        assert im1.ndim == 3 and im1.shape[0] == 1, f"expected (1, C_i, Npix), got {tuple(im1.shape)}"

        # Reuse the existing Convol_torch core by faking B=1 shapes.
        # We call the existing (batched) implementation with B=1.
        return self.Convol_torch(im1, ww, cell_ids=cell_ids, nside=nside)  # returns (1, C_o, Npix_1)

    # --- Replace the first lines of Convol_torch with a dispatcher ---
    def Convol_torch(self, im, ww, cell_ids=None, nside=None):
        """
        Batched KERNELSZxKERNELSZ aggregation.

        Accepts either:
          - im: Tensor (B, C_i, Npix)  with one shared or per-batch (B,Npix) cell_ids
          - im: list/tuple of Tensors, each (C_i, Npix_b), with cell_ids a list of arrays
        """
        import torch

        # (A) Variable-length per-sample path: im is a list/tuple OR cell_ids is a list/tuple
        if isinstance(im, (list, tuple)) or isinstance(cell_ids, (list, tuple)):
            # Normalize to lists
            im_list = im if isinstance(im, (list, tuple)) else [im]
            cid_list = cell_ids if isinstance(cell_ids, (list, tuple)) else [cell_ids] * len(im_list)
            assert len(im_list) == len(cid_list), "im list and cell_ids list must have same length"

            outs = []
            for xb, cb in zip(im_list, cid_list):
                # xb: (C_i, Npix_b) -> (1, C_i, Npix_b)
                if not torch.is_tensor(xb):
                    xb = torch.as_tensor(xb, device=self.device, dtype=self.dtype)
                if xb.dim() == 2:
                    xb = xb.unsqueeze(0)
                elif xb.dim() != 3 or xb.shape[0] != 1:
                    raise ValueError(f"Each sample must be (C,N) or (1,C,N); got {tuple(xb.shape)}")

                yb = self._convol_single(xb, ww, cell_ids=cb, nside=nside)  # (1, C_o, Npix_b)
                outs.append(yb.squeeze(0))  # -> (C_o, Npix_b)
            return outs  # List[Tensor], each (C_o, Npix_b)

        # (B) Standard fixed-length batched path (your current implementation)
        # ... keep your existing Convol_torch body from here unchanged ...
        # (paste your current function body starting from the type casting and assertions)

        # ---- Basic checks / casting ----
        if not isinstance(im, torch.Tensor):
            im = torch.as_tensor(im, device=self.device, dtype=self.dtype)
        if not isinstance(ww, torch.Tensor):
            ww = torch.as_tensor(ww, device=self.device, dtype=self.dtype)

        assert im.ndim == 3, f"`im` must be (B, C_i, Npix), got {tuple(im.shape)}"
        B, C_i, Npix = im.shape
        device = im.device
        dtype  = im.dtype

        # ---- Recompute (idx_nn, w_idx, w_w) depending on cell_ids shape ----
        # target shapes:
        #   idx_nn_eff : (B, Npix, P)
        #   w_idx_eff  : (B, Npix, S, P)
        #   w_w_eff    : (B, Npix, S, P)
        if cell_ids is not None:
            # ---- Recompute (idx_nn, w_idx, w_w) depending on cell_ids shape ----
            # Normaliser: accepter Tensor, ndarray ou list/tuple de 1 élément (cas var-length, B=1)
            if isinstance(cell_ids, (list, tuple)):
                # liste d'ids (souvent longueur 1 en var-length)
                if len(cell_ids) == 1:
                    cid = np.asarray(cell_ids[0])[None, :]  # -> (1, Npix)
                else:
                    # si jamais >1, on essaie d'empiler (doit avoir même Npix par élément)
                    cid = np.stack([np.asarray(c) for c in cell_ids], axis=0)
            elif torch.is_tensor(cell_ids):
                c = cell_ids.detach().cpu().numpy()
                cid = c if c.ndim != 1 else c[None, :]      # uniformiser en 2D quand B=1
            else:
                c = np.asarray(cell_ids)
                cid = c if c.ndim != 1 else c[None, :]

            # cid est maintenant (B, Npix)
            idx_nn_eff, w_idx_eff, w_w_eff = self.make_idx_weights_from_cell_ids(
                cid, nside, device=device
            )
            # shapes: (B, Npix, P), (B, Npix, S, P|P), (B, Npix, S, P|P)
            P = idx_nn_eff.shape[-1]
            S = w_idx_eff.shape[-2] if w_idx_eff.ndim == 4 else 1

            # s’assurer des dtypes/devices
            idx_nn_eff = torch.as_tensor(idx_nn_eff, device=device, dtype=torch.long)
            w_idx_eff  = torch.as_tensor(w_idx_eff,  device=device, dtype=torch.long)
            w_w_eff    = torch.as_tensor(w_w_eff,    device=device, dtype=dtype)
        else:
            # Use precomputed (shared for batch)
            if self.w_idx is None:
                if self.cell_ids.ndim==1:
                    l_cell=self.cell_ids[None,:]
                else:
                    l_cell=self.cell_ids
                    
                idx_nn,w_idx,w_w = self.make_idx_weights_from_cell_ids(
                    l_cell,
                    polar=self.polar,
                    gamma=self.gamma,
                    device=self.device,
                    allow_extrapolation=self.allow_extrapolation)

                self.idx_nn = idx_nn
                self.w_idx  = w_idx
                self.w_w    = w_w
            else:
                idx_nn = self.idx_nn          # (Npix,P)
                w_idx  = self.w_idx           # (Npix,P) or (Npix,S,P)
                w_w    = self.w_w             # (Npix,P) or (Npix,S,P)

            #assert idx_nn.ndim == 3 and idx_nn.size(1) == Npix, \
            #    f"`idx_nn` must be (B,Npix,P) with Npix={Npix}, got {tuple(idx_nn.shape)}"

            P = idx_nn.size(-1)

            if w_idx.ndim == 3:
                S = 1
                w_idx_eff = w_idx[:, :, None, :]  # (B,Npix,1,P)
                w_w_eff   = w_w[:, :, None, :]    # (B,Npix,1,P)
            elif w_idx.ndim == 4:
                S = w_idx.size(2)
                w_idx_eff = w_idx    # (B,Npix,S,P)
                w_w_eff   =  w_w    # (B,Npix,S,P)
            else:
                raise ValueError(f"Unsupported `w_idx` shape {tuple(w_idx.shape)}; expected (Npix,P) or (Npix,S,P)")
            idx_nn_eff = idx_nn               # (B,Npix,P)

        # ---- 1) Gather neighbor values from im along Npix -> (B, C_i, Npix, P)
        rim = torch.take_along_dim(
            im.unsqueeze(-1),                  # (B, C_i, Npix, 1)
            idx_nn_eff[:, None, :, :],         # (B, 1, Npix, P)
            dim=2
        )

        # ---- 2) Normalize ww to (B, C_i, C_o, M, S)
        if ww.ndim == 3:
            C_i_w, C_o, M = ww.shape
            assert C_i_w == C_i, f"ww C_i mismatch: {C_i_w} vs im {C_i}"
            ww_eff = ww.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1, S)
        elif ww.ndim == 4:
            if ww.shape[0] == C_i and ww.shape[1] != C_i:
                # (C_i, C_o, M, S)
                C_i_w, C_o, M, S_w = ww.shape
                assert C_i_w == C_i, f"ww C_i mismatch: {C_i_w} vs im {C_i}"
                assert S_w == S, f"ww S mismatch: {S_w} vs w_idx S {S}"
                ww_eff = ww.unsqueeze(0).expand(B, -1, -1, -1, -1)
            elif ww.shape[0] == B:
                # (B, C_i, C_o, M)
                _, C_i_w, C_o, M = ww.shape
                assert C_i_w == C_i, f"ww C_i mismatch: {C_i_w} vs im {C_i}"
                ww_eff = ww.unsqueeze(-1).expand(-1, -1, -1, -1, S)
            else:
                raise ValueError(f"Ambiguous 4D ww shape {tuple(ww.shape)}; expected (C_i,C_o,M,S) or (B,C_i,C_o,M)")
        elif ww.ndim == 5:
            # (B, C_i, C_o, M, S)
            assert ww.shape[0] == B and ww.shape[1] == C_i, "ww batch/C_i mismatch"
            _, _, _, M, S_w = ww.shape
            assert S_w == S, f"ww S mismatch: {S_w} vs w_idx S {S}"
            ww_eff = ww
        else:
            raise ValueError(f"Unsupported ww shape {tuple(ww.shape)}")
        
        # --- Sanitize shapes: ensure w_idx_eff / w_w_eff == (B, Npix, S, P)
        
        # ---- 3) Gather along M using w_idx_eff -> (B, C_i, C_o, Npix, S, P)
        idx_exp = w_idx_eff[:, None, None, :, :, :]            # (B,1,1,Npix,S,P)
        rw = torch.take_along_dim(
            ww_eff.unsqueeze(-1),                               # (B,C_i,C_o,M,S,1)
            idx_exp,                                           # (B,1,1,Npix,S,P)
            dim=3                                              # gather along M
        )  # -> (B, C_i, C_o, Npix, S, P)
        # ---- 4) Apply extra neighbor weights ----
        rw = rw * w_w_eff[:, None, None, :, :, :]              # (B, C_i, C_o, Npix, S, P)
        # ---- 5) Combine neighbor values and weights ----
        rim_exp = rim[:, :, None, :, None, :]                  # (B, C_i, 1, Npix, 1, P)
        out_ci  = (rim_exp * rw).sum(dim=-1)                   # sum over P -> (B, C_i, C_o, Npix, S)
        out_ci  = out_ci.sum(dim=-1)                           # sum over S -> (B, C_i, C_o, Npix)
        out     = out_ci.sum(dim=1)                            # sum over C_i -> (B, C_o, Npix)

        return out
    
    def _to_numpy_1d(self, ids):
        """Return a 1D numpy array of int64 for a single set of cell ids."""
        import numpy as np, torch
        if isinstance(ids, np.ndarray):
            return ids.reshape(-1).astype(np.int64, copy=False)
        if torch.is_tensor(ids):
            return ids.detach().cpu().to(torch.long).view(-1).numpy()
        # python list/tuple of ints
        return np.asarray(ids, dtype=np.int64).reshape(-1)

    def _is_varlength_batch(self, ids):
        """
        True if ids is a list/tuple of per-sample id arrays (var-length batch).
        False if ids is a single array/tensor of ids (shared for whole batch).
        """
        import numpy as np, torch
        if isinstance(ids, (list, tuple)):
            return True
        if isinstance(ids, np.ndarray) and ids.ndim == 2:
            # This would be a dense (B, Npix) matrix -> NOT var-length list
            return False
        if torch.is_tensor(ids) and ids.dim() == 2:
            return False
        return False

    def Down(self, im, cell_ids=None, nside=None,max_poll=False):
        """
        If `cell_ids` is a single set of ids -> return a single (Tensor, Tensor).
        If `cell_ids` is a list (var-length)           -> return (list[Tensor], list[Tensor]).
        """
        if self.f is None:
            if self.dtype==torch.float64:
                self.f=sc.funct(KERNELSZ=self.KERNELSZ,all_type='float64')
            else:
                self.f=sc.funct(KERNELSZ=self.KERNELSZ,all_type='float32')
                
        if cell_ids is None:
            dim,cdim = self.f.ud_grade_2(im,cell_ids=self.cell_ids,nside=self.nside,max_poll=False)
            return dim,cdim
        
        if nside is None:
            nside = self.nside

        # var-length mode: list/tuple of ids, one per sample
        if self._is_varlength_batch(cell_ids):
            outs, outs_ids = [], []
            B = len(cell_ids)
            for b in range(B):
                cid_b = self._to_numpy_1d(cell_ids[b])
                # extraire le bon échantillon d'`im`
                if torch.is_tensor(im):
                    xb = im[b:b+1]  # (1, C, N_b)
                    yb, ids_b = self.f.ud_grade_2(xb, cell_ids=cid_b, nside=nside,max_poll=max_poll)
                    outs.append(yb.squeeze(0))  # (C, N_b')
                else:
                    # si im est déjà une liste de (C, N_b)
                    xb = im[b]
                    yb, ids_b = self.f.ud_grade_2(xb[None, ...], cell_ids=cid_b, nside=nside,max_poll=max_poll)
                    outs.append(yb.squeeze(0))
                outs_ids.append(torch.as_tensor(ids_b, device=outs[-1].device, dtype=torch.long))
            return outs, outs_ids

        # grille commune (un seul vecteur d'ids)
        cid = self._to_numpy_1d(cell_ids)
        return self.f.ud_grade_2(im, cell_ids=cid, nside=nside,max_poll=False)

    def Up(self, im, cell_ids=None, nside=None, o_cell_ids=None):
        """
        If `cell_ids` / `o_cell_ids` are single arrays  -> return Tensor.
        If they are lists (var-length per sample)       -> return list[Tensor].
        """
        if self.f is None:
            if self.dtype==torch.float64:
                self.f=sc.funct(KERNELSZ=self.KERNELSZ,all_type='float64')
            else:
                self.f=sc.funct(KERNELSZ=self.KERNELSZ,all_type='float32')
                
        if cell_ids is None:
            dim = self.f.up_grade(im,self.nside*2,cell_ids=self.cell_ids,nside=self.nside)
            return dim
        
        if nside is None:
            nside = self.nside

        # var-length: listes parallèles
        if self._is_varlength_batch(cell_ids):
            assert isinstance(o_cell_ids, (list, tuple)) and len(o_cell_ids) == len(cell_ids), \
                "In var-length mode, `o_cell_ids` must be a list with same length as `cell_ids`."
            outs = []
            B = len(cell_ids)
            for b in range(B):
                cid_b  = self._to_numpy_1d(cell_ids[b])      # coarse ids
                ocid_b = self._to_numpy_1d(o_cell_ids[b])    # fine   ids
                if torch.is_tensor(im):
                    xb = im[b:b+1]  # (1, C, N_b_coarse)
                    yb = self.f.up_grade(xb, nside*2, cell_ids=cid_b, nside=nside,
                                         o_cell_ids=ocid_b, force_init_index=True)
                    outs.append(yb.squeeze(0))  # (C, N_b_fine)
                else:
                    xb = im[b]  # (C, N_b_coarse)
                    yb = self.f.up_grade(xb[None, ...], nside*2, cell_ids=cid_b, nside=nside,
                                         o_cell_ids=ocid_b, force_init_index=True)
                    outs.append(yb.squeeze(0))
            return outs

        # grille commune
        cid  = self._to_numpy_1d(cell_ids)
        ocid = self._to_numpy_1d(o_cell_ids) if o_cell_ids is not None else None
        return self.f.up_grade(im, nside*2, cell_ids=cid, nside=nside,
                               o_cell_ids=ocid, force_init_index=True)

    def to_tensor(self,x):
        if self.f is None:
            if self.dtype==torch.float64:
                self.f=sc.funct(KERNELSZ=self.KERNELSZ,all_type='float64')
            else:
                self.f=sc.funct(KERNELSZ=self.KERNELSZ,all_type='float32')
        return self.f.backend.bk_cast(x)
    
    def to_numpy(self,x):
        if isinstance(x,np.ndarray):
            return x
        return x.cpu().numpy()
        

   
        
