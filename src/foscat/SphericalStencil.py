# SPDX-License-Identifier: MIT
# Author: J.-M. Delouis
import numpy as np
import healpy as hp
import foscat.scat_cov as sc
import torch

import numpy as np
import torch
import healpy as hp


class SphericalStencil:
    """
    GPU-accelerated spherical stencil operator for HEALPix convolutions.

    This class implements three phases:
      A) Geometry preparation: build local rotated stencil vectors for each target
         pixel, compute HEALPix neighbor indices and interpolation weights.
      B) Sparse binding: map neighbor indices/weights to available data samples
         (sorted ids), and normalize weights.
      C) Convolution: apply multi-channel kernels to sparse gathered data.

    Once A+B are prepared, multiple convolutions (C) can be applied efficiently
    on the GPU.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    kernel_sz : int
        Size of local stencil (must be odd, e.g. 3, 5, 7).
    gauge_type : str
        Type of gauge :
        'cosmo' use the same definition than
           https://www.aanda.org/articles/aa/abs/2022/12/aa44566-22/aa44566-22.html
        'phi' is define at the pole, could be better for earth observation not using intensivly the pole
    n_gauge : float
        Number of oriented gauges (Default 1).
    blend : bool
        Whether to blend smoothly between axisA and axisB (dual gauge).
    power : float
        Sharpness of blend transition (dual gauge).
    nest : bool
        Use nested ordering if True (default), else ring ordering.
    cell_ids : np.ndarray | torch.Tensor | None
        If given, initialize Step A immediately for these targets.
    device : torch.device | str | None
        Default device (if None, 'cuda' if available else 'cpu').
    dtype : torch.dtype | None
        Default dtype (float32 if None).
    """

    def __init__(
            self,
            nside: int,
            kernel_sz: int,
            *,
            nest: bool = True,
            cell_ids=None,
            device=None,
            dtype=None,
            n_gauges=1,
            gauge_type='cosmo',
            scat_op=None,
    ):
        assert kernel_sz >= 1 and int(kernel_sz) == kernel_sz
        assert kernel_sz % 2 == 1, "kernel_sz must be odd"

        self.nside = int(nside)
        self.KERNELSZ = int(kernel_sz)
        self.P = self.KERNELSZ * self.KERNELSZ

        self.G = n_gauges
        self.gauge_type=gauge_type
        
        self.nest = bool(nest)
        if scat_op is None:
            self.f=sc.funct(KERNELSZ=self.KERNELSZ)
        else:
            self.f=scat_op
            
        # Torch defaults
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.float32
        self.device = torch.device(device)
        self.dtype = dtype

        # Geometry cache
        self.Kb = None
        self.idx_t = None   # (4, K*P) neighbor indices
        self.w_t   = None   # (4, K*P) interpolation weights
        self.ids_sorted_np = None
        self.pos_safe_t = None
        self.w_norm_t   = None
        self.present_t  = None

        # Optionnel : on garde une copie des ids par défaut si fournis
        self.cell_ids_default = None

        # ---- Optional immediate preparation (Step A+B at init) ----
        if cell_ids is not None:
            # Keep a copy of the default target grid (fast-path later)
            cid = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
            self.cell_ids_default = cid.copy()

            # Step A (Torch): build geometry for this grid with G gauges
            th, ph = hp.pix2ang(self.nside, cid, nest=self.nest)
            self.prepare_torch(th, ph, G=self.G)   # fills idx_t/_multi and w_t/_multi

            # Step B (Torch): bind sparse mapping on the class device/dtype
            order = np.argsort(cid)
            self.ids_sorted_np = cid[order]        # cache for fast-path

            if self.G > 1:
                # Multi-gauge binding (produces pos_safe_t_multi, w_norm_t_multi)
                self.bind_support_torch_multi(
                    self.ids_sorted_np,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                # Single-gauge binding (produces pos_safe_t, w_norm_t)
                self.bind_support_torch(
                    self.ids_sorted_np,
                    device=self.device,
                    dtype=self.dtype,
                )


    # ------------------------------------------------------------------
    # Rotation construction in Torch
    # ------------------------------------------------------------------
    @staticmethod
    def _rotation_total_torch(th, ph, alpha=None, G: int = 1, gauge_cosmo=True,device=None, dtype=None):
        """
        Build a batch of rotation matrices with *G gauges* per target.

        Column-vector convention: v' = R @ v.

        Parameters
        ----------
        th : array-like (N,)
            Colatitude.
        ph : array-like (N,)
            Longitude.
        alpha : array-like (N,) or scalar or None
            Base gauge rotation angle around the local normal.
            If None -> 0. For each gauge g in [0..G-1], we add g*pi/G.
        G : int
            Number of gauges to generate per target (>=1).
        device, dtype : torch device/dtype

        Returns
        -------
        R_tot : torch.Tensor, shape (N, G, 3, 3)
            For each target i and gauge g, the matrix:
              R_tot[i,g] = R_gauge(alpha[i] + g*pi/G) @ Rz(ph[i]) @ Ry(th[i])
        """
        assert G >= 1, "G must be >= 1"

        # ---- to torch 1D
        th = torch.as_tensor(th, device=device, dtype=dtype).view(-1)
        ph = torch.as_tensor(ph, device=device, dtype=dtype).view(-1)
        if alpha is None:
            alpha = torch.zeros_like(th)
        else:
            alpha = torch.as_tensor(alpha, device=device, dtype=dtype).view(-1)

        device = th.device
        dtype  = th.dtype
        N = th.shape[0]

        # ---- base rotation R_base = Rz(ph) @ Ry(th), shape (N,3,3)
        ct, st = torch.cos(th), torch.sin(th)
        cp, sp = torch.cos(ph), torch.sin(ph)

        R_base = torch.zeros((N, 3, 3), device=device, dtype=dtype)
        # row 0
        R_base[:, 0, 0] = cp * ct
        R_base[:, 0, 1] = -sp
        R_base[:, 0, 2] = cp * st
        # row 1
        R_base[:, 1, 0] = sp * ct
        R_base[:, 1, 1] = cp
        R_base[:, 1, 2] = sp * st
        # row 2
        R_base[:, 2, 0] = -st
        R_base[:, 2, 1] = 0.0
        R_base[:, 2, 2] = ct

        # local normal n = third column of R_base, shape (N,3)
        n = R_base[:, :, 2]
        n = n / torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(1e-12)  # safe normalize
        
        # per-target sign: +1 if th <= pi/2 else -1
        sign = torch.where(th <= (np.pi/2), torch.ones_like(th), -torch.ones_like(th))  # (N,)

        # base gauge shifts (always positive)
        g_shifts = torch.arange(G, device=device, dtype=dtype) * (np.pi / G)            # (G,)

        # broadcast with sign: (N,G)
        if gauge_cosmo:
            alpha_g = alpha[:, None] + sign[:, None] * g_shifts[None, :]
        else:
            alpha_g = alpha[:, None] +  g_shifts[None, :]

        ca = torch.cos(alpha_g)  # (N,G)
        sa = torch.sin(alpha_g)  # (N,G)

        # ---- expand normal to (N,G,3)
        n_g = n[:, None, :].expand(N, G, 3)         # (N,G,3)
        nx, ny, nz = n_g[..., 0], n_g[..., 1], n_g[..., 2]

        # skew-symmetric K(n_g), shape (N,G,3,3)
        K = torch.zeros((N, G, 3, 3), device=device, dtype=dtype)
        K[..., 0, 1] = -nz; K[..., 0, 2] =  ny
        K[..., 1, 0] =  nz; K[..., 1, 2] = -nx
        K[..., 2, 0] = -ny; K[..., 2, 1] =  nx

        # outer(n,n) and identity
        outer = n_g.unsqueeze(-1) * n_g.unsqueeze(-2)    # (N,G,3,3)
        I = torch.eye(3, device=device, dtype=dtype).view(1,1,3,3).expand(N, G, 3, 3)

        # ---- Rodrigues per gauge: R_gauge(N,G,3,3)
        R_gauge = I * ca.view(N, G, 1, 1) + K * sa.view(N, G, 1, 1) + \
                  outer * (1.0 - ca).view(N, G, 1, 1)

        # ---- broadcast multiply with base: R_base_g(N,G,3,3)
        R_base_g = R_base.unsqueeze(1).expand(N, G, 3, 3)
        R_tot = torch.matmul(R_gauge, R_base_g)          # (N,G,3,3)
        return R_tot

    # ------------------------------------------------------------------
    # Torch-based get_interp_weights wrapper
    # ------------------------------------------------------------------
    @staticmethod
    def get_interp_weights_from_vec_torch(
        nside: int,
        vec,
        *,
        nest: bool = True,
        device=None,
        dtype=None,
        chunk_size=1_000_000,
    ):
        """
        Torch wrapper for healpy.get_interp_weights using input vectors.

        Parameters
        ----------
        nside : int
            HEALPix resolution.
        vec : torch.Tensor (...,3)
            Direction vectors (not necessarily normalized).
        nest : bool
            Nested ordering if True (default).
        device, dtype : Torch device/dtype.
        chunk_size : int
            Number of points per healpy call on CPU.

        Returns
        -------
        idx_t : LongTensor (4, *leading)
        w_t   : Tensor (4, *leading)
        """
        if not isinstance(vec, torch.Tensor):
            vec = torch.as_tensor(vec, device=device, dtype=dtype)
        else:
            device = vec.device if device is None else device
            dtype  = vec.dtype if dtype is None else dtype
            vec = vec.to(device=device, dtype=dtype)

        orig_shape = vec.shape[:-1]
        M = int(np.prod(orig_shape)) if len(orig_shape) else 1
        v = vec.reshape(M, 3)

        eps = torch.finfo(vec.dtype).eps
        r = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(eps)
        v_unit = v / r
        x, y, z = v_unit[:, 0], v_unit[:, 1], v_unit[:, 2]

        theta = torch.acos(z.clamp(-1.0, 1.0))
        phi = torch.atan2(y, x)
        two_pi = torch.tensor(2*np.pi, device=device, dtype=dtype)
        phi = (phi % two_pi)

        theta_np = theta.detach().cpu().numpy()
        phi_np   = phi.detach().cpu().numpy()

        idx_accum, w_accum = [], []
        for start in range(0, M, chunk_size):
            stop = min(start + chunk_size, M)
            t_chunk, p_chunk = theta_np[start:stop], phi_np[start:stop]
            idx_np, w_np = hp.get_interp_weights(nside, t_chunk, p_chunk, nest=nest)
            idx_accum.append(idx_np)
            w_accum.append(w_np)

        idx_np_all = np.concatenate(idx_accum, axis=1) if len(idx_accum) > 1 else idx_accum[0]
        w_np_all   = np.concatenate(w_accum,   axis=1) if len(w_accum) > 1 else w_accum[0]

        idx_t = torch.as_tensor(idx_np_all, device=device, dtype=torch.long)
        w_t   = torch.as_tensor(w_np_all,   device=device, dtype=dtype)

        if len(orig_shape):
            idx_t = idx_t.view(4, *orig_shape)
            w_t   = w_t.view(4, *orig_shape)

        return idx_t, w_t

    # ------------------------------------------------------------------
    # Step A: geometry preparation fully in Torch
    # ------------------------------------------------------------------
    def prepare_torch(self, th, ph, alpha=None, G: int = 1):
        """
        Prepare rotated stencil and HEALPix neighbors/weights in Torch for *G gauges*.

        Parameters
        ----------
        th, ph : array-like, shape (K,)
            Target colatitudes/longitudes.
        alpha : array-like (K,) or scalar or None
            Base gauge angle about the local normal at each target. If None -> 0.
            For each gauge g in [0..G-1], the effective angle is alpha + g*pi/G.
        G : int (>=1)
            Number of gauges to generate per target.

        Side effects
        ------------
        Sets:
          - self.Kb = K
          - self.G  = G
          - self.idx_t_multi : (G, 4, K*P) LongTensor (neighbors per gauge)
          - self.w_t_multi   : (G, 4, K*P) Tensor     (weights   per gauge)
          - For backward compat when G==1:
              self.idx_t : (4, K*P)
              self.w_t   : (4, K*P)

        Returns
        -------
        idx_t_multi : torch.LongTensor, shape (G, 4, K*P)
        w_t_multi   : torch.Tensor,     shape (G, 4, K*P)
        """
        # --- sanitize inputs on CPU (angles) then use class device/dtype
        th = np.asarray(th, float).reshape(-1)
        ph = np.asarray(ph, float).reshape(-1)
        K = th.size
        self.Kb = K
        self.G  = int(G)
        assert self.G >= 1, "G must be >= 1"

        # --- build the local (P,3) stencil once on device
        P = self.P
        vec_np = np.zeros((P, 3), dtype=float)
        grid = (np.arange(self.KERNELSZ) - self.KERNELSZ // 2) / self.nside
        vec_np[:, 0] = np.tile(grid, self.KERNELSZ)
        vec_np[:, 1] = np.repeat(grid, self.KERNELSZ)
        vec_np[:, 2] = 1.0 - np.sqrt(vec_np[:, 0]**2 + vec_np[:, 1]**2)
        vec_t = torch.as_tensor(vec_np, device=self.device, dtype=self.dtype)     # (P,3)

        # --- rotation matrices for all targets & gauges: (K,G,3,3)
        if alpha is None:
            if self.gauge_type=='cosmo':
                alpha=2*((th>np.pi/2)-0.5)*ph
            else:
                alpha=0.0*th
            
        R_t = self._rotation_total_torch(
            th, ph, alpha, G=self.G, gauge_cosmo=(self.gauge_type=='cosmo'),
            device=self.device, dtype=self.dtype
        )  # shape (K,G,3,3)

        # --- rotate stencil for each (target, gauge): (K,G,P,3)
        #     einsum over local stencil (P,3) with rotation (K,G,3,3)
        rotated = torch.einsum('kgij,pj->kgpi', R_t, vec_t)  # (K,G,P,3)

        # --- query HEALPix (neighbors+weights) in one call over (K*G*P)
        rotated_flat = rotated.reshape(-1, 3)  # (K*G*P, 3)
        idx_t, w_t = self.get_interp_weights_from_vec_torch(
            self.nside,
            rotated_flat,
            nest=self.nest,
            device=self.device,
            dtype=self.dtype,
        )  # each (4, K*G*P)

        # --- reshape back to split gauges:
        # current: (4, K*G*P) -> (4, K, G, P) -> (G, 4, K, P) -> (G, 4, K*P)
        idx_t = idx_t.view(4, K, self.G, P).permute(2, 0, 1, 3).reshape(self.G, 4, K*P)
        w_t   = w_t.view(4, K, self.G, P).permute(2, 0, 1, 3).reshape(self.G, 4, K*P)

        # --- cache multi-gauge versions
        self.idx_t_multi = idx_t   # (G, 4, K*P)
        self.w_t_multi   = w_t     # (G, 4, K*P)

        # --- backward compatibility: when G==1, also fill single-gauge fields
        if self.G == 1:
            self.idx_t = idx_t[0]  # (4, K*P)
            self.w_t   = w_t[0]    # (4, K*P)
        else:
            # when multi-gauge, you can pick a default (e.g., gauge 0) if legacy code asks
            # but better to adapt bind/apply to consume the multi-gauge tensors.
            self.idx_t = None
            self.w_t   = None

        return self.idx_t_multi, self.w_t_multi

    def bind_support_torch_multi(self, ids_sorted_np, *, device=None, dtype=None):
        """
        Multi-gauge sparse binding (Step B) AVEC logique 'domaine réduit':
          - poids des voisins hors domaine mis à 0
          - renormalisation par colonne à 1
          - si colonne vide: fallback sur le pixel cible (centre du stencil)

        Produit:
          self.pos_safe_t_multi : (G, 4, K*P)
          self.w_norm_t_multi   : (G, 4, K*P)
          self.present_t_multi  : (G, 4, K*P)
        """
        assert hasattr(self, 'idx_t_multi') and self.idx_t_multi is not None, \
            "Call prepare_torch(..., G>0) before bind_support_torch_multi(...)"
        assert hasattr(self, 'w_t_multi') and self.w_t_multi is not None

        if device is None: device = self.device
        if dtype  is None: dtype  = self.dtype

        self.ids_sorted_np = np.asarray(ids_sorted_np, dtype=np.int64).reshape(-1)
        ids_sorted = torch.as_tensor(self.ids_sorted_np, device=device, dtype=torch.long)

        G, _, M = self.idx_t_multi.shape
        K = self.Kb
        P = self.P
        assert M == K*P, "idx_t_multi second axis must have K*P columns"

        # index du centre du stencil (en flatten P)
        p_ref = (self.KERNELSZ // 2) * (self.KERNELSZ + 1)  # ex. 5 -> 12

        pos_list, present_list, wnorm_list = [], [], []

        for g in range(G):
            idx = self.idx_t_multi[g].to(device=device, dtype=torch.long)   # (4, M)
            w   = self.w_t_multi[g].to(device=device, dtype=dtype)          # (4, M)

            # positions dans ids_sorted
            pos = torch.searchsorted(ids_sorted, idx.reshape(-1)).view(4, M)
            in_range = pos < ids_sorted.numel()
            cmp_vals = torch.full_like(idx, -1)
            cmp_vals[in_range] = ids_sorted[pos[in_range]]
            present = (cmp_vals == idx)                                     # (4, M) bool

            # Colonnes sans AUCUN voisin présent
            empty_cols = ~present.any(dim=0)                                # (M,)
            if empty_cols.any():
                p_ref = (self.KERNELSZ // 2) * (self.KERNELSZ + 1)
                k_id = torch.div(torch.arange(M, device=device), P, rounding_mode='floor')  # (M,)
                ref_cols = k_id * P + p_ref
                src = ref_cols[empty_cols]

                # copie idx/w de la colonne 'centre'
                idx[:, empty_cols] = idx[:, src]
                w[:,   empty_cols] = w[:,   src]

                # --- Recompute presence/pos safely on those columns
                idx_e = idx[:, empty_cols].reshape(-1)           # (4*M_empty,)
                pos_e = torch.searchsorted(ids_sorted, idx_e)    # (4*M_empty,)
                valid_e = pos_e < ids_sorted.numel()
                pos_e_clipped = pos_e.clamp_max(max(ids_sorted.numel()-1, 0)).to(torch.long)
                cmp_e = ids_sorted[pos_e_clipped]
                present_e = valid_e & (cmp_e == idx_e)          # (4*M_empty,)

                present[:, empty_cols] = present_e.view(4, -1)
                pos[:,      empty_cols] = pos_e_clipped.view(4, -1)

            # Met à zéro les poids absents puis renormalise à 1 par colonne
            w = w * present
            colsum = w.sum(dim=0, keepdim=True)
            zero_cols = (colsum == 0)
            if zero_cols.any():
                w[0, zero_cols[0]] = present[0, zero_cols[0]].to(w.dtype)
                colsum = w.sum(dim=0, keepdim=True)
            w_norm = w / colsum.clamp_min(1e-12)

            pos_safe = torch.where(present, pos, torch.zeros_like(pos))

            pos_list.append(pos_safe)
            present_list.append(present)
            wnorm_list.append(w_norm)

        self.pos_safe_t_multi = torch.stack(pos_list, dim=0)     # (G, 4, M)
        self.present_t_multi  = torch.stack(present_list, dim=0) # (G, 4, M)
        self.w_norm_t_multi   = torch.stack(wnorm_list, dim=0)   # (G, 4, M)

        # miroir device/dtype runtime
        self.device = device
        self.dtype  = dtype

    def bind_support_torch(self, ids_sorted_np, *, device=None, dtype=None):
        """
        Single-gauge sparse binding (Step B) AVEC logique 'domaine réduit':
          - poids des voisins hors domaine mis à 0
          - renormalisation par colonne à 1
          - si colonne vide: fallback sur le pixel cible (centre du stencil)
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        self.ids_sorted_np = np.asarray(ids_sorted_np, dtype=np.int64)
        ids_sorted = torch.as_tensor(self.ids_sorted_np, device=device, dtype=torch.long)

        idx = self.idx_t.to(device=device, dtype=torch.long)    # (4, K*P)
        w   = self.w_t.to(device=device, dtype=dtype)           # (4, K*P)

        K = self.Kb
        P = self.P
        M = K * P

        # positions dans ids_sorted
        pos = torch.searchsorted(ids_sorted, idx.reshape(-1)).view(4, M)
        in_range = pos < ids_sorted.shape[0]
        cmp_vals = torch.full_like(idx, -1)
        cmp_vals[in_range] = ids_sorted[pos[in_range]]
        present = (cmp_vals == idx)                              # (4, M)

        # Fallback colonnes vides -> centre du stencil
        p_ref = (self.KERNELSZ // 2) * (self.KERNELSZ + 1)
        empty_cols = ~present.any(dim=0)                         # (M,)
        if empty_cols.any():
            k_id = torch.div(torch.arange(M, device=device), P, rounding_mode='floor')  # (M,)
            ref_cols = k_id * P + p_ref
            src = ref_cols[empty_cols]

            # copie idx/w de la colonne 'centre'
            idx[:, empty_cols] = idx[:, src]
            w[:,   empty_cols] = w[:,   src]

            # --- Recompute presence/pos safely on those columns
            idx_e = idx[:, empty_cols].reshape(-1)               # (4*M_empty,)
            pos_e = torch.searchsorted(ids_sorted, idx_e)        # (4*M_empty,)
            # valid positions strictly inside [0, len)
            valid_e = pos_e < ids_sorted.numel()
            pos_e_clipped = pos_e.clamp_max(max(ids_sorted.numel()-1, 0)).to(torch.long)
            cmp_e = ids_sorted[pos_e_clipped]
            present_e = valid_e & (cmp_e == idx_e)               # (4*M_empty,)

            # reshape back
            present[:, empty_cols] = present_e.view(4, -1)
            pos[:,      empty_cols] = pos_e_clipped.view(4, -1)

        # Zéro poids absents + renormalisation à 1
        w = w * present
        colsum = w.sum(dim=0, keepdim=True)
        zero_cols = (colsum == 0)
        if zero_cols.any():
            # force 1 sur la première ligne disponible (ici ligne 0)
            w[0, zero_cols[0]] = present[0, zero_cols[0]].to(w.dtype)
            colsum = w.sum(dim=0, keepdim=True)
        w_norm = w / colsum.clamp_min(1e-12)

        self.pos_safe_t = torch.where(present, pos, torch.zeros_like(pos))
        self.w_norm_t   = w_norm
        self.present_t  = present

        self.device = device
        self.dtype  = dtype

    
    '''
    def bind_support_torch_multi(self, ids_sorted_np, *, device=None, dtype=None):
        """
        Multi-gauge sparse binding (Step B).
        Uses self.idx_t_multi / self.w_t_multi prepared by prepare_torch(..., G>1)
        and builds, for each gauge g, (pos_safe, w_norm, present).

        Parameters
        ----------
        ids_sorted_np : np.ndarray (K,)
            Sorted pixel ids for available samples (matches the last axis of your data).
        device, dtype : torch device/dtype for the produced mapping tensors.

        Side effects
        ------------
        Sets:
          - self.ids_sorted_np  : (K,)
          - self.pos_safe_t_multi : (G, 4, K*P)  LongTensor
          - self.w_norm_t_multi   : (G, 4, K*P)  Tensor
          - self.present_t_multi  : (G, 4, K*P)  BoolTensor
          - (and mirrors device/dtype in self.device/self.dtype)
        """
        assert hasattr(self, 'idx_t_multi') and self.idx_t_multi is not None, \
            "Call prepare_torch(..., G>0) before bind_support_torch_multi(...)"
        assert hasattr(self, 'w_t_multi') and self.w_t_multi is not None

        if device is None: device = self.device
        if dtype  is None: dtype  = self.dtype

        self.ids_sorted_np = np.asarray(ids_sorted_np, dtype=np.int64).reshape(-1)
        ids_sorted = torch.as_tensor(self.ids_sorted_np, device=device, dtype=torch.long)

        G, _, M = self.idx_t_multi.shape
        K = self.Kb
        P = self.P
        assert M == K*P, "idx_t_multi second axis must have K*P columns"

        pos_list, present_list, wnorm_list = [], [], []

        for g in range(G):
            idx = self.idx_t_multi[g].to(device=device, dtype=torch.long)   # (4, M)
            w   = self.w_t_multi[g].to(device=device, dtype=dtype)          # (4, M)

            pos = torch.searchsorted(ids_sorted, idx.reshape(-1)).view(4, M)
            in_range = pos < ids_sorted.numel()
            cmp_vals = torch.full_like(idx, -1)
            cmp_vals[in_range] = ids_sorted[pos[in_range]]
            present = (cmp_vals == idx)

            # normalize weights per column after masking
            w = w * present
            colsum = w.sum(dim=0, keepdim=True).clamp_min(1e-12)
            w_norm = w / colsum

            pos_safe = torch.where(present, pos, torch.zeros_like(pos))

            pos_list.append(pos_safe)
            present_list.append(present)
            wnorm_list.append(w_norm)

        self.pos_safe_t_multi = torch.stack(pos_list, dim=0)     # (G, 4, M)
        self.present_t_multi  = torch.stack(present_list, dim=0) # (G, 4, M)
        self.w_norm_t_multi   = torch.stack(wnorm_list, dim=0)   # (G, 4, M)

        # mirror runtime placement
        self.device = device
        self.dtype  = dtype
    
    # ------------------------------------------------------------------
    # Step B: bind support Torch
    # ------------------------------------------------------------------
    def bind_support_torch(self, ids_sorted_np, *, device=None, dtype=None):
        """
        Map HEALPix neighbor indices (from Step A) to actual data samples
        sorted by pixel id. Produces pos_safe and normalized weights.

        Parameters
        ----------
        ids_sorted_np : np.ndarray (K,)
            Sorted pixel ids for available data.
        device, dtype : Torch device/dtype for results.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        self.ids_sorted_np = np.asarray(ids_sorted_np, dtype=np.int64)
        ids_sorted = torch.as_tensor(self.ids_sorted_np, device=device, dtype=torch.long)

        idx = self.idx_t.to(device=device, dtype=torch.long)
        w   = self.w_t.to(device=device, dtype=dtype)

        M = self.Kb * self.P
        idx = idx.view(4, M)
        w   = w.view(4, M)

        pos = torch.searchsorted(ids_sorted, idx.reshape(-1)).view(4, M)
        in_range = pos < ids_sorted.shape[0]
        cmp_vals = torch.full_like(idx, -1)
        cmp_vals[in_range] = ids_sorted[pos[in_range]]
        present = (cmp_vals == idx)

        w = w * present
        colsum = w.sum(dim=0, keepdim=True).clamp_min(1e-12)
        w_norm = w / colsum

        self.pos_safe_t = torch.where(present, pos, torch.zeros_like(pos))
        self.w_norm_t   = w_norm
        self.present_t  = present
        self.device = device
        self.dtype  = dtype
    '''
    # ------------------------------------------------------------------
    # Step C: apply convolution (already Torch in your code)
    # ------------------------------------------------------------------
    def apply_multi(self, data_sorted_t: torch.Tensor, kernel_t: torch.Tensor):
        """
        Apply multi-gauge convolution.

        Inputs
        ------
        data_sorted_t : (B, Ci, K)  torch.Tensor on self.device/self.dtype
        kernel_t      : either
                        - (Ci, Co_g, P)         : shared kernel for all gauges
                        - (G, Ci, Co_g, P)      : per-gauge kernels

        Returns
        -------
        out : (B, G*Co_g, K) torch.Tensor
        """
        assert hasattr(self, 'pos_safe_t_multi') and self.pos_safe_t_multi is not None, \
            "Call bind_support_torch_multi(...) before apply_multi(...)"
        B, Ci, K = data_sorted_t.shape
        G, _, M = self.pos_safe_t_multi.shape
        assert M == K * self.P

        # normalize kernel to per-gauge
        if kernel_t.dim() == 3:
            Ci_k, Co_g, P = kernel_t.shape
            assert Ci_k == Ci and P == self.P
            kernel_g = kernel_t[None, ...].expand(G, -1, -1, -1)   # (G, Ci, Co_g, P)
        elif kernel_t.dim() == 4:
            Gk, Ci_k, Co_g, P = kernel_t.shape
            assert Gk == G and Ci_k == Ci and P == self.P
            kernel_g = kernel_t
        else:
            raise ValueError("kernel_t must be (Ci,Co_g,P) or (G,Ci,Co_g,P)")

        outs = []
        for g in range(G):
            pos_safe = self.pos_safe_t_multi[g]             # (4, K*P)
            w_norm   = self.w_norm_t_multi[g]               # (4, K*P)

            # gather four neighbors then weight -> (B,Ci,K,P)
            vals_g = []
            for j in range(4):
                vj = data_sorted_t.index_select(2, pos_safe[j].reshape(-1))   # (B,Ci,K*P)
                vj = vj.view(B, Ci, K, self.P)
                vals_g.append(vj * w_norm[j].view(1, 1, K, self.P))
            tmp = sum(vals_g)   # (B,Ci,K,P)

            # spatial+channel mixing with kernel of this gauge -> (B,Co_g,K)
            yg = torch.einsum('bckp,cop->bok', tmp, kernel_g[g])
            outs.append(yg)

        # concat the gauges along channel dimension: (B, G*Co_g, K)
        return torch.cat(outs, dim=1)

    def apply(self, data_sorted_t, kernel_t):
        """
        Apply the (Ci,Co,P) kernel to batched sparse data (B,Ci,K)
        using precomputed pos_safe and w_norm. Runs fully on GPU.

        Parameters
        ----------
        data_sorted_t : torch.Tensor (B,Ci,K)
            Input data aligned with ids_sorted.
        kernel_t : torch.Tensor (Ci,Co,P)
            Convolution kernel.

        Returns
        -------
        out : torch.Tensor (B,Co,K)
        """
        assert self.pos_safe_t is not None and self.w_norm_t is not None
        B, Ci, K = data_sorted_t.shape
        Ci_k, Co, P = kernel_t.shape
        assert Ci_k == Ci and P == self.P

        vals = []
        for j in range(4):
            vj = data_sorted_t.index_select(2, self.pos_safe_t[j].reshape(-1))
            vj = vj.view(B, Ci, K, P)
            vals.append(vj * self.w_norm_t[j].view(1, 1, K, P))
        tmp = sum(vals)   # (B,Ci,K,P)

        out = torch.einsum('bckp,cop->bok', tmp, kernel_t)
        return out

    def _Convol_Torch(self, data: torch.Tensor, kernel: torch.Tensor, cell_ids=None) -> torch.Tensor:
        """
        Convenience entry point with automatic single- or multi-gauge dispatch.

        Behavior
        --------
        - If `cell_ids is None`: use cached geometry (prepare_torch) and sparse mapping
          (bind_support_torch or bind_support_torch_multi) already stored in the class,
          re-binding Step-B to `data`'s device/dtype when needed, then apply.
        - If `cell_ids` is provided: compute geometry + sparse mapping for these cells
          using the class' gauge setup (including the number of gauges G prepared by
          `prepare_torch(..., G)`), reorder `data` to match the sorted ids, apply
          (single or multi), and finally unsort to the original `cell_ids` order.

        Parameters
        ----------
        data :  (B, Ci, K) torch.float
            Sparse map values. Last axis K must equal the number of target pixels.
        kernel : torch.Tensor
            - Single-gauge path: (Ci, Co, P) where P = kernel_sz**2.
            - Multi-gauge path:  (Ci, Co_g, P)   shared kernel for all gauges, OR
                                 (G, Ci, Co_g, P) per-gauge kernels.
              The output channels will be Co (single) or G*Co_g (multi).
        cell_ids : Optional[np.ndarray | torch.Tensor], shape (K,)
            Target HEALPix pixels. If None, re-use the class' cached targets.

        Returns
        -------
        out : torch.Tensor, shape (B, Co, K)
              Co = Co (single gauge) or Co = G*Co_g (multi-gauge).
        """
        assert isinstance(data, torch.Tensor) and isinstance(kernel, torch.Tensor), \
            "data and kernel must be torch.Tensors"
        device = data.device
        dtype  = data.dtype

        B, Ci, K_data = data.shape
        P = self.P
        P_k = kernel.shape[-1]
        assert P_k == P, f"kernel P={P_k} must equal kernel_sz**2 = {P}"

        def _to_np_1d(ids):
            if isinstance(ids, torch.Tensor):
                return ids.detach().cpu().numpy().astype(np.int64, copy=False)
            return np.asarray(ids, dtype=np.int64).reshape(-1)

        def _has_multi_bind():
            return (getattr(self, 'G', 1) > 1 and
                    getattr(self, 'pos_safe_t_multi', None) is not None and
                    getattr(self, 'w_norm_t_multi',   None) is not None)

        # ----------------------------
        # Case 1: new target ids given
        # ----------------------------
        if cell_ids is not None:
            cell_ids_np = _to_np_1d(cell_ids)

            # A) geometry with class' G (defaults to 1 if not set)
            G = getattr(self, 'G', 1)
            th, ph = hp.pix2ang(self.nside, cell_ids_np, nest=self.nest)
            self.prepare_torch(th, ph, alpha=None, G=G)   # fills idx_t/_multi, w_t/_multi

            # B) sort ids and reorder data accordingly
            order = np.argsort(cell_ids_np)
            ids_sorted_np = cell_ids_np[order]
            assert K_data == ids_sorted_np.size, \
                "data last dimension must equal number of provided cell_ids"

            order_t = torch.as_tensor(order, device=device, dtype=torch.long)
            data_sorted_t = data[..., order_t]            # (B, Ci, K) aligned with ids_sorted_np

            # C) bind sparse support
            if G > 1:
                self.bind_support_torch_multi(ids_sorted_np, device=device, dtype=dtype)
                out_sorted = self.apply_multi(data_sorted_t, kernel)   # (B, G*Co_g, K)
            else:
                self.bind_support_torch(ids_sorted_np, device=device, dtype=dtype)
                out_sorted = self.apply(data_sorted_t, kernel)         # (B, Co, K)

            # D) unsort back to original order
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(order.size)
            inv_idx = torch.as_tensor(inv_order, device=device, dtype=torch.long)
            return out_sorted[..., inv_idx]

        # -----------------------------------------------
        # Case 2: fast path on cached geometry + mapping
        # -----------------------------------------------
        if self.ids_sorted_np is None:
            if getattr(self, 'cell_ids_default', None) is not None:
                self.ids_sorted_np = np.sort(self.cell_ids_default)
            else:
                raise AssertionError(
                    "No cached targets. Either pass `cell_ids` once or initialize the class with `cell_ids=`."
                )

        if _has_multi_bind():
            # rebind if device/dtype changed
            if (self.device != device) or (self.dtype != dtype):
                self.bind_support_torch_multi(self.ids_sorted_np, device=device, dtype=dtype)
            return self.apply_multi(data, kernel)

        # single-gauge cached path
        need_rebind = (
            getattr(self, 'pos_safe_t', None) is None or
            getattr(self, 'w_norm_t',   None) is None or
            self.device != device or
            self.dtype  != dtype
        )
        if need_rebind:
            self.bind_support_torch(self.ids_sorted_np, device=device, dtype=dtype)
        return self.apply(data, kernel)

    def Convol_torch(self, im, ww, cell_ids=None, nside=None):
        """
        Batched KERNELSZ x KERNELSZ aggregation (dispatcher).

        Supports:
          - im: Tensor (B, Ci, K) with
              * cell_ids is None              -> use cached targets (fast path)
              * cell_ids is 1D (K,)           -> one shared grid for whole batch
              * cell_ids is 2D (B, K)         -> per-sample grids, same length; returns (B, Co, K)
              * cell_ids is list/tuple        -> per-sample grids (var-length allowed)
          - im: list/tuple of Tensors, each (Ci, K_b) with cell_ids list/tuple

        Notes
        -----
        - Kernel shapes accepted:
            * single/multi shared:     (Ci, Co_g, P)
            * per-gauge kernels:       (G, Ci, Co_g, P)
          The low-level _Convol_Torch will choose between apply/apply_multi
          depending on the class state (G>1 and multi-bind present).
        """
        import numpy as np
        import torch

        def _dev_dtype_like(x: torch.Tensor):
            if not isinstance(x, torch.Tensor):
                raise TypeError("Expected a torch.Tensor for device/dtype inference.")
            return x.device, x.dtype

        def _prepare_kernel(k: torch.Tensor, device, dtype):
            if not isinstance(k, torch.Tensor):
                raise TypeError("kernel (ww) must be a torch.Tensor")
            return k.to(device=device, dtype=dtype)

        def _to_np_ids(ids):
            if isinstance(ids, torch.Tensor):
                return ids.detach().cpu().numpy().astype(np.int64, copy=False)
            return np.asarray(ids, dtype=np.int64)

        class _NsideContext:
            def __init__(self, obj, nside_new):
                self.obj = obj
                self.nside_old = obj.nside
                self.nside_new = int(nside_new) if nside_new is not None else obj.nside
            def __enter__(self):
                self.obj.nside = self.nside_new
                return self
            def __exit__(self, exc_type, exc, tb):
                self.obj.nside = self.nside_old

        # ---------------- main dispatcher ----------------
        if isinstance(im, torch.Tensor):
            device, dtype = _dev_dtype_like(im)
            kernel = _prepare_kernel(ww, device, dtype)

            with _NsideContext(self, nside):
                # (A) Fast path: no ids provided -> delegate fully to _Convol_Torch
                if cell_ids is None:
                    return self._Convol_Torch(im, kernel, cell_ids=None)

                # Normalise numpy/tensor ragged inputs
                if isinstance(cell_ids, np.ndarray) and cell_ids.dtype == object:
                    cell_ids = list(cell_ids)

                # (B) One shared grid for entire batch: 1-D ids
                if isinstance(cell_ids, (np.ndarray, torch.Tensor)) and getattr(cell_ids, "ndim", 1) == 1:
                    return self._Convol_Torch(im, kernel, cell_ids=_to_np_ids(cell_ids))

                # (C) Per-sample grids, same length: 2-D ids (B, K)
                if isinstance(cell_ids, (np.ndarray, torch.Tensor)) and getattr(cell_ids, "ndim", 0) == 2:
                    B = im.shape[0]
                    if isinstance(cell_ids, torch.Tensor):
                        assert cell_ids.shape[0] == B, "cell_ids first dim must match batch size B"
                        ids2d = cell_ids.detach().cpu().numpy().astype(np.int64, copy=False)
                    else:
                        ids2d = np.asarray(cell_ids, dtype=np.int64)
                        assert ids2d.shape[0] == B, "cell_ids first dim must match batch size B"

                    outs = []
                    for b in range(B):
                        x_b   = im[b:b+1]                       # (1, Ci, K_b)
                        ids_b = ids2d[b]                        # (K_b,)
                        y_b   = self._Convol_Torch(x_b, kernel, cell_ids=ids_b)  # (1, Co, K_b)
                        outs.append(y_b)
                    return torch.cat(outs, dim=0)              # (B, Co, K)

                # (D) Per-sample grids, variable length: list/tuple
                if isinstance(cell_ids, (list, tuple)):
                    B = im.shape[0]
                    assert len(cell_ids) == B, "cell_ids list length must match batch size B"
                    outs = []
                    lengths = []
                    for b in range(B):
                        ids_b_np = _to_np_ids(cell_ids[b])
                        lengths.append(ids_b_np.size)
                        x_b = im[b:b+1]                        # (1, Ci, K_b)
                        y_b = self._Convol_Torch(x_b, kernel, cell_ids=ids_b_np)  # (1, Co, K_b)
                        outs.append(y_b)
                    if len(set(lengths)) == 1:
                        return torch.cat(outs, dim=0)          # (B, Co, K)
                    else:
                        return [y.squeeze(0) for y in outs]    # list[(Co, K_b)]

                raise TypeError("Unsupported type for cell_ids with tensor input.")

        # Case: im is list/tuple of (Ci, K_b) tensors (var-length samples)
        if isinstance(im, (list, tuple)):
            assert isinstance(cell_ids, (list, tuple)) and len(cell_ids) == len(im), \
                "When im is a list, cell_ids must be a list of same length."
            assert len(im) > 0, "Empty list for `im`."

            device, dtype = _dev_dtype_like(im[0])
            kernel = _prepare_kernel(ww, device, dtype)

            outs = []
            with _NsideContext(self, nside):
                lengths = []
                tmp = []
                for x_b, ids_b in zip(im, cell_ids):
                    assert isinstance(x_b, torch.Tensor), "Each sample in `im` must be a torch.Tensor"
                    assert x_b.device == device and x_b.dtype == dtype, "All samples must share device/dtype."
                    x_b   = x_b.unsqueeze(0)  # (1, Ci, K_b)
                    ids_b = _to_np_ids(ids_b)
                    y_b   = self._Convol_Torch(x_b, kernel, cell_ids=ids_b)  # (1, Co, K_b)
                    tmp.append(y_b)
                    lengths.append(y_b.shape[-1])
                if len(set(lengths)) == 1:
                    return torch.cat(tmp, dim=0)  # (B, Co, K)
                else:
                    return [y.squeeze(0) for y in tmp]

        raise TypeError("`im` must be either a torch.Tensor (B,Ci,K) or a list of (Ci,K_b) tensors.")

    def make_matrix(
        self,
        kernel: torch.Tensor,
        cell_ids=None,
        *,
        return_sparse_tensor: bool = False,
        chunk_k: int = 4096,
    ):
        """
        Build the sparse COO matrix M such that applying M to vec(data) reproduces
        the spherical convolution performed by Convol_torch/_Convol_Torch.

        Supports single- and multi-gauge:
          - kernel shape (Ci, Co_g, P)       -> shared across G gauges, output Co = G*Co_g
          - kernel shape (G, Ci, Co_g, P)    -> per-gauge kernels, same output Co = G*Co_g

        Parameters
        ----------
        kernel : torch.Tensor
            (Ci, Co_g, P) or (G, Ci, Co_g, P) with P = kernel_sz**2.
            Must be on the device/dtype where you want the resulting matrix.
        cell_ids : array-like of shape (K,) or torch.Tensor, optional
            Target pixel IDs (NESTED if self.nest=True).
            If None, uses the grid already cached in the class (fast path).
            If provided, we prepare geometry & sparse binding for these ids.
        return_sparse_tensor : bool, default False
            If True, return a coalesced torch.sparse_coo_tensor of shape (Co*K, Ci*K).
            Else, return (weights, indices, shape) where:
              - indices is a LongTensor of shape (2, nnz) with [row; col]
              - weights is a Tensor of shape (nnz,)
              - shape is the (rows, cols) tuple
        chunk_k : int, default 4096
            Chunk size over target pixels to limit peak memory.

        Returns
        -------
        If return_sparse_tensor:
            M : torch.sparse_coo_tensor of shape (Co*K, Ci*K), coalesced
        else:
            weights : torch.Tensor (nnz,)
            indices : torch.LongTensor (2, nnz)   with [row; col]
            shape   : tuple[int, int]  (Co*K, Ci*K)

        Notes
        -----
        - The resulting matrix implements the same interpolation-and-mixing as the
          GPU path (gather 4 neighbors -> normalize -> apply spatial+channel kernel),
          and matches the output of Convol_torch for the same (kernel, cell_ids).
        - For multi-gauge, rows are grouped as concatenated gauges: first all
          Co_g channels for gauge 0 over all K, then gauge 1, etc.
        """
        import numpy as np
        import torch
        import healpy as hp

        device = kernel.device
        k_dtype = kernel.dtype

        # --- validate kernel & normalize shapes
        if kernel.dim() == 3:
            # shared across gauges
            Ci, Co_g, P = kernel.shape
            per_gauge = False
        elif kernel.dim() == 4:
            Gk, Ci, Co_g, P = kernel.shape
            per_gauge = True
            if hasattr(self, 'G'):
                assert Gk == self.G, f"kernel first dim G={Gk} must match self.G={self.G}"
            else:
                self.G = int(Gk)
        else:
            raise ValueError("kernel must be (Ci,Co_g,P) or (G,Ci,Co_g,P)")

        assert P == self.P, f"kernel P={P} must equal kernel_sz**2={self.P}"

        # --- geometry + binding for these ids (or use cached)
        def _to_np_ids(ids):
            if ids is None:
                return None
            if isinstance(ids, torch.Tensor):
                return ids.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            return np.asarray(ids, dtype=np.int64).reshape(-1)

        cell_ids_np = _to_np_ids(cell_ids)

        if cell_ids_np is not None:
            # Step A: geometry (Torch) with the class' number of gauges
            G = int(getattr(self, 'G', 1))
            th, ph = hp.pix2ang(self.nside, cell_ids_np, nest=self.nest)
            self.prepare_torch(th, ph, alpha=None, G=G)

            # Step B: bind on sorted ids, and remember K
            order = np.argsort(cell_ids_np)
            ids_sorted_np = cell_ids_np[order]
            K = ids_sorted_np.size

            if G > 1:
                self.bind_support_torch_multi(ids_sorted_np, device=device, dtype=k_dtype)
            else:
                self.bind_support_torch(ids_sorted_np, device=device, dtype=k_dtype)
        else:
            # use cached mapping
            if getattr(self, 'ids_sorted_np', None) is None:
                raise AssertionError("No cached targets; pass `cell_ids` or init the class with `cell_ids=`.")
            K = self.ids_sorted_np.size
            # rebind to the kernel device/dtype if needed
            if getattr(self, 'G', 1) > 1:
                if (self.device != device) or (self.dtype != k_dtype):
                    self.bind_support_torch_multi(self.ids_sorted_np, device=device, dtype=k_dtype)
            else:
                if (self.device != device) or (self.dtype != k_dtype):
                    self.bind_support_torch(self.ids_sorted_np, device=device, dtype=k_dtype)

        G = int(getattr(self, 'G', 1))
        Co_total = (G * Co_g)  # output channels including gauges
        shape = (Co_total * K, Ci * K)

        # --- choose mapping tensors (multi vs single)
        is_multi = (G > 1) and (getattr(self, 'pos_safe_t_multi', None) is not None)
        if is_multi:
            pos_all_g = self.pos_safe_t_multi.to(device=device)   # (G,4,K*P)
            w_all_g   = self.w_norm_t_multi.to(device=device, dtype=k_dtype)
        else:
            pos_all   = self.pos_safe_t.to(device=device)         # (4,K*P)
            w_all     = self.w_norm_t.to(device=device, dtype=k_dtype)

        # --- precompute channel row/col bases
        # rows: for (co_total, k_out) -> co_total*K + k_out
        # cols: for (ci, k_in)        -> ci*K       + k_in
        row_base = (torch.arange(Co_total, device=device, dtype=torch.long) * K)[:, None]  # (Co_total, 1)
        col_base = (torch.arange(Ci,       device=device, dtype=torch.long) * K)[:, None]  # (Ci, 1)
        

        rows_all, cols_all, vals_all = [], [], []

        # --- helper to add one gauge block (gauge g -> Co_g*K rows)
        def _accumulate_for_gauge(g, pos_g, w_g, ker_g):
            """
            pos_g : (4, K*P) long
            w_g   : (4, K*P) float
            ker_g : (Ci, Co_g, P)
            """
            # process by chunks in k to control memory
            for start in range(0, K, chunk_k):
                stop = min(start + chunk_k, K)
                Kb = stop - start
                cols_span = torch.arange(start * self.P, stop * self.P, device=device, dtype=torch.long)

                pos = pos_g[:, cols_span].view(4, Kb, self.P)   # (4, Kb, P)
                w   = w_g[:, cols_span].view(4, Kb, self.P)     # (4, Kb, P)

                # rows_gauge: indices de lignes pour cette jauge g
                # Chaque jauge occupe un bloc de Co_g canaux de sortie pour CHAQUE pixel (K)
                # donc offset = g*Co_g
                rows_gauge = (torch.arange(Co_g, device=device, dtype=torch.long) + g*Co_g)[:, None] * K \
                           + (start + torch.arange(Kb, device=device, dtype=torch.long))[None, :]
                # -> shape (Co_g, Kb)
                rows = rows_gauge[:, :, None, None, None]               # (Co_g, Kb,1,1,1)
                rows = rows.expand(Co_g, Kb, Ci, 4, self.P)              # (Co_g, Kb, Ci, 4, P)

                # cols: indices colonnes = (ci*K + pix)
                cols_pix = pos.permute(1, 0, 2)                          # (Kb, 4, P)
                cols_pix = cols_pix[None, :, None, :, :]                  # (1, Kb, 1, 4, P)
                cols = col_base + cols_pix                                # (Ci, Kb, 1, 4, P)
                cols = cols.permute(2, 1, 0, 3, 4)                         # (1, Kb, Ci, 4, P)
                cols = cols.expand(Co_g, Kb, Ci, 4, self.P)

                # values = kernel(ci, co_g, p) * w(4,kb,p)
                k_exp = ker_g.permute(1, 0, 2)                 # (Co_g, Ci, P)
                k_exp = k_exp[:, None, :, None, :]             # (Co_g, 1, Ci, 1, P)

                # CORRECTION: remettre les axes de w en (Kb,4,P) avant broadcast
                w_exp = w.permute(1, 0, 2)[None, :, None, :, :]  # (1, Kb, 1, 4, P)
                w_exp = w_exp.expand(Co_g, Kb, Ci, 4, self.P)    # (Co_g, Kb, Ci, 4, P)

                vals = k_exp * w_exp                           # (Co_g, Kb, Ci, 4, P)

                rows_all.append(rows.reshape(-1))
                cols_all.append(cols.reshape(-1))
                vals_all.append(vals.reshape(-1))
                

        # --- accumulate either single- or multi-gauge
        if is_multi:
            # (a) shared kernel (Ci, Co_g, P) -> repeat over gauges
            if not per_gauge and kernel.dim() == 3:
                for g in range(G):
                    _accumulate_for_gauge(g, pos_all_g[g], w_all_g[g], kernel.to(device=device, dtype=k_dtype))
            # (b) per-gauge kernel (G, Ci, Co_g, P)
            else:
                for g in range(G):
                    _accumulate_for_gauge(g, pos_all_g[g], w_all_g[g], kernel[g].to(device=device, dtype=k_dtype))
        else:
            # G == 1 (single-gauge path)
            g = 0
            _accumulate_for_gauge(g, pos_all, w_all, kernel if kernel.dim() == 3 else kernel[0])

        rows = torch.cat(rows_all, dim=0)
        cols = torch.cat(cols_all, dim=0)
        vals = torch.cat(vals_all, dim=0)

        
        indices = torch.stack([cols, rows], dim=0)             # (2, nnz) invert rows/cols for foscat needs

        if return_sparse_tensor:
            M = torch.sparse_coo_tensor(indices, vals, size=shape, device=device, dtype=k_dtype).coalesce()
            return M
        else:
            return vals, indices, shape


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
            dim,cdim = self.f.ud_grade_2(im,cell_ids=self.cell_ids,nside=self.nside,max_poll=max_poll)
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
        return torch.tensor(x,device=self.device,dtype=self.dtype)
    
    def to_numpy(self,x):
        if isinstance(x,np.ndarray):
            return x
        return x.cpu().numpy()
        
