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
    gauge : str
        Gauge type for local orientation: 'phi', 'axis', 'dual'. (Default 'dual')
    axisA, axisB : tuple of float
        Reference axes for axis/dual gauges.
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
        gauge: str = 'dual',
        axisA=(1.0, 0.0, 0.0),
        axisB=(0.0, 1.0, 0.0),
        blend: bool = True,
        power: float = 4.0,
        nest: bool = True,
        cell_ids=None,
        device=None,
        dtype=None,
            scat_op=None,
    ):
        assert kernel_sz >= 1 and int(kernel_sz) == kernel_sz
        assert kernel_sz % 2 == 1, "kernel_sz must be odd"

        self.nside = int(nside)
        self.KERNELSZ = int(kernel_sz)
        self.P = self.KERNELSZ * self.KERNELSZ

        self.gauge = gauge
        self.axisA = np.asarray(axisA, float) / np.linalg.norm(axisA)
        self.axisB = np.asarray(axisB, float) / np.linalg.norm(axisB)
        self.blend = bool(blend)
        self.power = float(power)
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

        # Optional immediate preparation
        if cell_ids is not None:
            cid = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
            self.cell_ids_default = cid.copy()          # mémoriser la grille par défaut

            # Step A (Torch) : géométrie pour cette grille
            th, ph = hp.pix2ang(self.nside, cid, nest=self.nest)
            self.prepare_torch(th, ph)

            # Step B (Torch) : bind épars pour ces ids (triés)
            order = np.argsort(cid)
            self.ids_sorted_np = cid[order]             # cache pour fast-path
            self.bind_support_torch(self.ids_sorted_np, device=self.device, dtype=self.dtype)


    # ------------------------------------------------------------------
    # Rotation construction in Torch
    # ------------------------------------------------------------------
    @staticmethod
    def _rotation_total_torch(th, ph, alpha=None, device=None, dtype=None):
        """
        Build batch of rotation matrices R_tot for each (theta, phi, alpha).

        Column-vector convention: v' = R @ v.

        Parameters
        ----------
        th : array-like (N,)
            Colatitude.
        ph : array-like (N,)
            Longitude.
        alpha : array-like (N,) or None
            Gauge rotation angle around local normal. If None, set to zero.
        device, dtype : Torch device/dtype.

        Returns
        -------
        R_tot : torch.Tensor, shape (N, 3, 3)
        """
        th = torch.as_tensor(th, device=device, dtype=dtype).view(-1)
        ph = torch.as_tensor(ph, device=device, dtype=dtype).view(-1)
        if alpha is None:
            alpha = torch.zeros_like(th)
        else:
            alpha = torch.as_tensor(alpha, device=device, dtype=dtype).view(-1)
        N = th.shape[0]

        ct, st = torch.cos(th), torch.sin(th)
        cp, sp = torch.cos(ph), torch.sin(ph)

        R_base = torch.zeros((N, 3, 3), device=device, dtype=dtype)
        R_base[:, 0, 0] = cp * ct
        R_base[:, 0, 1] = -sp
        R_base[:, 0, 2] = cp * st
        R_base[:, 1, 0] = sp * ct
        R_base[:, 1, 1] = cp
        R_base[:, 1, 2] = sp * st
        R_base[:, 2, 0] = -st
        R_base[:, 2, 1] = 0.0
        R_base[:, 2, 2] = ct

        # local normal
        n = R_base[:, :, 2]
        n = n / torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(1e-12)
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]

        ca, sa = torch.cos(alpha), torch.sin(alpha)
        K = torch.zeros((N, 3, 3), device=device, dtype=dtype)
        K[:, 0, 1] = -nz; K[:, 0, 2] = ny
        K[:, 1, 0] = nz;  K[:, 1, 2] = -nx
        K[:, 2, 0] = -ny; K[:, 2, 1] = nx

        outer = n.unsqueeze(2) * n.unsqueeze(1)
        I = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

        R_gauge = I * ca.view(N, 1, 1) + K * sa.view(N, 1, 1) + \
                  outer * (1.0 - ca).view(N, 1, 1)

        R_tot = torch.matmul(R_gauge, R_base)
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
    def prepare_torch(self, th, ph, alpha=None):
        """
        Prepare rotated stencil vectors and neighbor indices/weights in Torch.

        Parameters
        ----------
        th, ph : array-like (K,)
            Target colatitudes/longitudes.
        alpha : array-like (K,) or None
            Gauge rotation angles (default 0).

        Side effects
        ------------
        Sets self.Kb, self.idx_t, self.w_t
        """
        th = np.asarray(th, float)
        ph = np.asarray(ph, float)
        self.Kb = th.size

        # Build local stencil (P,3) once
        vec_np = np.zeros((self.P, 3), dtype=float)
        grid = (np.arange(self.KERNELSZ) - self.KERNELSZ // 2) / self.nside
        vec_np[:, 0] = np.tile(grid, self.KERNELSZ)
        vec_np[:, 1] = np.repeat(grid, self.KERNELSZ)
        vec_np[:, 2] = 1.0 - np.sqrt(vec_np[:, 0]**2 + vec_np[:, 1]**2)
        vec_t = torch.as_tensor(vec_np, device=self.device, dtype=self.dtype)

        # Rotation matrices for all targets
        R_t = self._rotation_total_torch(th, ph, alpha, device=self.device, dtype=self.dtype)

        # Rotate stencil for each target
        rotated_t = torch.einsum('kij,pj->kpi', R_t, vec_t)   # (K,P,3)

        # Get neighbors/weights in Torch
        idx_t, w_t = self.get_interp_weights_from_vec_torch(
            self.nside,
            rotated_t.reshape(-1, 3),
            nest=self.nest,
            device=self.device,
            dtype=self.dtype,
        )

        self.idx_t = idx_t
        self.w_t   = w_t
        return idx_t, w_t

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

    # ------------------------------------------------------------------
    # Step C: apply convolution (already Torch in your code)
    # ------------------------------------------------------------------
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
        Convenience entry point:
          - If cell_ids is None: use cached geometry (prepare_torch) and sparse mapping
            (bind_support_torch) already stored in the class, then apply.
          - If cell_ids is provided: compute geometry + sparse mapping for these cells
            (using the class' gauge and options), reorder `data` accordingly, and apply.

        Parameters
        ----------
        data :  (B, Ci, K) torch.float on self.device
            Sparse map values (last axis must match the number of target pixels).
        kernel : (Ci, Co, P) torch.float on self.device
            Spatial kernel per (input, output) channel pair (P = kernel_sz**2).
        cell_ids : Optional[np.ndarray | torch.Tensor], shape (K,)
            Target HEALPix pixels. If None, re-use the class' cached targets.

        Returns
        -------
        out : (B, Co, K) torch.float
        """
        assert isinstance(data, torch.Tensor) and isinstance(kernel, torch.Tensor), \
            "data and kernel must be torch.Tensors"
        device = data.device
        dtype  = data.dtype

        B, Ci, K_data = data.shape
        Ci_k, Co, P_k = kernel.shape
        assert P_k == self.P, f"kernel P={P_k} must equal kernel_sz**2 = {self.P}"
        assert Ci_k == Ci,    f"kernel Ci={Ci_k} must match data Ci={Ci}"

        # Case 1: cell_ids provided -> compute everything for these targets
        if cell_ids is not None:
            # to numpy 1D array
            if isinstance(cell_ids, torch.Tensor):
                cell_ids_np = cell_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                cell_ids_np = np.asarray(cell_ids, dtype=np.int64)

            # angles for the new targets
            th, ph = hp.pix2ang(self.nside, cell_ids_np, nest=self.nest)

            # A) NumPy: neighbors/weights w.r.t. class gauge/options
            self.prepare_torch(th, ph)

            # B) Torch: sort ids and reorder data last axis to match ids_sorted
            order = np.argsort(cell_ids_np)
            ids_sorted_np = cell_ids_np[order]
            assert K_data == ids_sorted_np.size, \
                "data last dimension must equal number of provided cell_ids"

            # Reorder data to match sorted ids on last axis (in-place view-safe)
            data_sorted_t = data[..., torch.as_tensor(order, device=device, dtype=torch.long)]

            # Bind sparse support on GPU
            self.bind_support_torch(
                ids_sorted_np,
                device=device,
                dtype=dtype,
            )

            # C) Torch apply
            out = self.apply(data_sorted_t, kernel)   # (B, Co, K)
            # Return in the same order as input cell_ids (unsort to original order)
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(order.size)
            inv_idx = torch.as_tensor(inv_order, device=device, dtype=torch.long)
            out = out[..., inv_idx]                   # back to original cell_ids order
            return out

    
        # Case 2: cell_ids is None -> use cached geometry & mapping
        # Fast path: utiliser la géométrie/mapping en cache.
        # Si le mapping n’existe pas encore OU s’il n’est pas au bon device/dtype,
        # on refait uniquement l’étape B (bind_support_torch) avec le device/dtype des données.
        assert self.ids_sorted_np is not None, \
            "No cached targets. Either pass `cell_ids` or initialize the class with `cell_ids=`."

        need_rebind = (
            self.pos_safe_t is None or
            self.w_norm_t   is None or
            self.device != data.device or
            self.dtype  != data.dtype
        )
        if need_rebind:
            self.bind_support_torch(
                self.ids_sorted_np,
                device=data.device,
                dtype=data.dtype,
            )
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
                # Fast path: no ids provided -> ensure Step B is bound to this device/dtype
                
                if cell_ids is None:
                    assert self.ids_sorted_np is not None, \
                        "No cached targets. Pass `cell_ids` once (or init the class with `cell_ids=`)."
                    if (self.pos_safe_t is None or self.w_norm_t is None or
                        self.device != im.device or self.dtype != im.dtype):
                        self.bind_support_torch(self.ids_sorted_np, device=im.device, dtype=im.dtype)
                    return self._Convol_Torch(im, kernel, cell_ids=None)

                # Normalize numpy/tensor cell_ids
                if isinstance(cell_ids, np.ndarray) and cell_ids.dtype == object:
                    # ragged -> treat as list
                    cell_ids = list(cell_ids)

                # One shared grid for entire batch: 1-D
                if isinstance(cell_ids, (np.ndarray, torch.Tensor)) and getattr(cell_ids, "ndim", 1) == 1:
                    return self._Convol_Torch(im, kernel, cell_ids=_to_np_ids(cell_ids))

                # Per-sample grids, same length: 2-D (B, K)
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
                        x_b   = im[b:b+1]                  # (1, Ci, K_b)
                        ids_b = ids2d[b]                   # (K_b,)
                        y_b   = self._Convol_Torch(x_b, kernel, cell_ids=ids_b)  # (1, Co, K_b)
                        outs.append(y_b)
                    return torch.cat(outs, dim=0)         # (B, Co, K)

                # Per-sample grids, variable length: list/tuple
                if isinstance(cell_ids, (list, tuple)):
                    B = im.shape[0]
                    assert len(cell_ids) == B, "cell_ids list length must match batch size B"
                    outs = []
                    # Try to detect if all lengths are equal -> stack; else return list
                    lengths = []
                    for b in range(B):
                        ids_b_np = _to_np_ids(cell_ids[b])
                        lengths.append(ids_b_np.size)
                        x_b   = im[b:b+1]
                        y_b   = self._Convol_Torch(x_b, kernel, cell_ids=ids_b_np)  # (1, Co, K_b)
                        outs.append(y_b)
                    if len(set(lengths)) == 1:
                        return torch.cat(outs, dim=0)     # (B, Co, K)
                    else:
                        return [y.squeeze(0) for y in outs]  # list[(Co, K_b)]

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
                # detect equal lengths to stack
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

    '''
    def Convol_torch(self, im, ww, cell_ids=None, nside=None):
        """
        Batched KERNELSZ x KERNELSZ aggregation (dispatcher).

        Accepts either:
          1) im: Tensor (B, Ci, K)
             - cell_ids is None              -> use class-cached targets (prepare_numpy + bind_support_torch already done)
             - cell_ids is 1D (K,)           -> one shared grid for the whole batch
             - cell_ids is list[(K_b,)]      -> per-sample grids (var-length allowed)

          2) im: list/tuple of Tensors, each (Ci, K_b)
             - cell_ids must be list/tuple of same length with matching lengths per sample

        Returns
        -------
          - If a single common grid is used: Tensor (B, Co, K_out)
          - If var-length per sample:       list[ Tensor (Co, K_b_out) ]
        """
        import numpy as np
        import torch

        # ---- helpers local to this method (no pollution of the class API)
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
            """Temporarily override self.nside during this call (restored on exit)."""
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
            # Case 1: im is (B, Ci, K) tensor
            device, dtype = _dev_dtype_like(im)
            kernel = _prepare_kernel(ww, device, dtype)

            with _NsideContext(self, nside):
                if cell_ids is None:
                    # Use cached geometry/mapping
                    return self._Convol_Torch(im, kernel, cell_ids=None)

                # One shared grid for entire batch
                if isinstance(cell_ids, (np.ndarray, torch.Tensor)) and getattr(cell_ids, "ndim", 1) == 1:
                    return self._Convol_Torch(im, kernel, cell_ids=_to_np_ids(cell_ids))

                # Per-sample grids (list/tuple)
                if isinstance(cell_ids, (list, tuple)):
                    B = im.shape[0]
                    assert len(cell_ids) == B, "cell_ids list length must match batch size B"
                    outs = []
                    for b in range(B):
                        x_b   = im[b:b+1]                   # (1, Ci, K_b)
                        ids_b = _to_np_ids(cell_ids[b])     # (K_b,)
                        y_b   = self._Convol_Torch(x_b, kernel, cell_ids=ids_b)  # (1, Co, K_b_out)
                        outs.append(y_b.squeeze(0))         # (Co, K_b_out)
                    return torch.stack(outs)

                raise TypeError("Unsupported type for cell_ids with tensor input.")

        # Case 2: im is list/tuple of (Ci, K_b) tensors (var-length samples)
        if isinstance(im, (list, tuple)):
            assert isinstance(cell_ids, (list, tuple)) and len(cell_ids) == len(im), \
                "When im is a list, cell_ids must be a list of same length."
            assert len(im) > 0, "Empty list for `im`."

            # Infer device/dtype from the first sample
            device, dtype = _dev_dtype_like(im[0])
            kernel = _prepare_kernel(ww, device, dtype)

            outs = []
            with _NsideContext(self, nside):
                for x_b, ids_b in zip(im, cell_ids):
                    assert isinstance(x_b, torch.Tensor), "Each sample in `im` must be a torch.Tensor"
                    assert x_b.device == device and x_b.dtype == dtype, "All samples must share device/dtype."
                    # Promote (Ci,K_b) -> (1,Ci,K_b) and dispatch
                    x_b   = x_b.unsqueeze(0)
                    ids_b = _to_np_ids(ids_b)
                    y_b   = self._Convol_Torch(x_b, kernel, cell_ids=ids_b)  # (1, Co, K_b_out)
                    outs.append(y_b.squeeze(0))                               # (Co, K_b_out)
            return torch.stack(outs,0)

        raise TypeError("`im` must be either a torch.Tensor (B,Ci,K) or a list of (Ci,K_b) tensors.")
    '''
    def make_sparse_matrix(
        self,
        cell_ids,
        kernel: torch.Tensor,
        *,
        return_sparse_tensor: bool = False,
        chunk_k: int = 4096,
    ):
        """
        Build the sparse COO matrix M so that, for each batch, applying M to
        vec(data) (flattened along channels then pixels) yields vec(out) for the
        same (cell_ids, kernel) as _Convol_Torch.

        Parameters
        ----------
        cell_ids : array-like (K,)
            Target HEALPix pixel ids (NESTED if self.nest=True).
            If these differ from the current cached targets, geometry and sparse
            mapping will be (re)computed using the class' gauge/options.
        kernel : (Ci, Co, P) torch.float on self.device
            Spatial kernel per (input, output) channel pair.
            P must equal self.P (kernel_sz**2).
        return_sparse_tensor : bool
            If True, returns a coalesced torch.sparse_coo_tensor instead of (indices, weights).
        chunk_k : int
            Number of target pixels to process per chunk (controls memory).

        Returns
        -------
        If return_sparse_tensor=False:
            (indices, weights, shape)
            indices : (2, nnz) torch.long   with [row; col]
            weights : (nnz,) torch.float
            shape   : tuple (Co*K, Ci*K)
        If return_sparse_tensor=True:
            M : torch.sparse_coo_tensor  (Co*K, Ci*K), coalesced
        """
        device = self.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- Ensure kernel on device and valid shape
        assert isinstance(kernel, torch.Tensor), "kernel must be a torch.Tensor"
        kernel = kernel.to(device)
        Ci_k, Co, P_k = kernel.shape
        assert P_k == self.P, f"kernel P={P_k} must equal kernel_sz**2={self.P}"

        # ---- Prepare geometry + sparse mapping for the requested cell_ids
        # Convert ids to numpy
        if isinstance(cell_ids, torch.Tensor):
            cell_ids_np = cell_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            cell_ids_np = np.asarray(cell_ids, dtype=np.int64)

        # Angles
        th, ph = hp.pix2ang(self.nside, cell_ids_np, nest=self.nest)
        # A) NumPy geometry for these targets
        self.prepare_numpy(th, ph)

        # B) Torch sparse binding for these targets (sorted order)
        order = np.argsort(cell_ids_np)
        ids_sorted_np = cell_ids_np[order]
        self.bind_support_torch(ids_sorted_np, device=device, dtype=kernel.dtype)

        # K size (targets) and bases for rows/cols
        K = self.Kb
        Ci = Ci_k

        # Precompute channel row/col bases
        # row for (co, k_out): row = co*K + k_out
        # col for (ci, k_in) : col = ci*K + k_in
        co_base = (torch.arange(Co, device=device, dtype=torch.long) * K)[:, None]  # (Co,1)
        ci_base = (torch.arange(Ci, device=device, dtype=torch.long) * K)[:, None]  # (Ci,1)

        # We'll build lists of chunk-level indices/values and concat, then coalesce
        rows_all = []
        cols_all = []
        vals_all = []

        # Convenience views of precomputed (4, K*P)
        pos_all = self.pos_safe_t          # long
        w_all   = self.w_norm_t            # float

        # Loop over chunks of target pixels to avoid huge intermediates
        for start in range(0, K, chunk_k):
            stop = min(start + chunk_k, K)
            Kb = stop - start
            # slice over columns belonging to these k in [start,stop)
            # For each k, its P columns occupy [k*P, (k+1)*P)
            col_span = torch.arange(start * self.P, stop * self.P, device=device, dtype=torch.long)

            pos = pos_all[:, col_span]   # (4, Kb*P)
            w   = w_all[:, col_span]     # (4, Kb*P)

            # reshape as (4, Kb, P)
            pos = pos.view(4, Kb, self.P)
            w   = w.view( 4, Kb, self.P)

            # -----------------------------
            # Build COO entries:
            # For every (co, k_out, ci, j, p), we add:
            #   row = co*K + (start + k_out)
            #   col = ci*K + pos[j, k_out, p]
            #   val = kernel[ci, co, p] * w[j, k_out, p]
            # -----------------------------

            # rows: shape (Co, Kb, Ci, 4, P) after broadcasting
            rows = co_base + (start + torch.arange(Kb, device=device, dtype=torch.long))[None, :]
            rows = rows[:, :, None, None, None]                   # (Co,Kb,1,1,1)
            rows = rows.expand(Co, Kb, Ci, 4, self.P)             # (Co,Kb,Ci,4,P)

            # cols: base from pos (4,Kb,P) -> (1,Kb,1,4,P) then add channel base
            cols_pix = pos[None, :, :]                            # (1,4,Kb,P)
            cols_pix = cols_pix.permute(0, 2, 0+1, 1, 3)          # (1,Kb,4,P) -> to align later
            # Easier: expand step-by-step
            cols_pix = pos.permute(1, 0, 2)                       # (Kb,4,P)
            cols_pix = cols_pix[None, :, None, :, :]              # (1,Kb,1,4,P)
            cols = ci_base + cols_pix                              # (Ci, Kb, 1, 4, P)
            cols = cols.permute(2, 1, 0, 3, 4)                    # (1,Kb,Ci,4,P) for broadcasting
            cols = cols.expand(Co, Kb, Ci, 4, self.P)             # match rows shape

            # values: kernel (Ci,Co,P) and w (4,Kb,P)
            k_cp = kernel.permute(1, 0, 2).transpose(0,1)         # no-op; keep (Ci,Co,P)
            k_cp = kernel                                          # (Ci,Co,P)
            k_exp = k_cp[None, :, :, None, :]                      # (1,Ci,Co,1,P)
            k_exp = k_exp.permute(2, 0, 1, 3, 4)                   # (Co,1,Ci,1,P)

            w_exp = w[None, :, :, :]                               # (1,4,Kb,P)
            w_exp = w_exp.permute(0, 2, 1, 0, 3)                   # -> (1,Kb,4,1,P)
            w_exp = w_exp.permute(0, 1, 3, 2, 4)                   # (1,Kb,1,4,P)
            w_exp = w_exp.expand(Co, Kb, Ci, 4, self.P)            # (Co,Kb,Ci,4,P)

            vals = k_exp * w_exp                                   # (Co,Kb,Ci,4,P)

            # Flatten all dims to 1D COO triplets
            rows = rows.reshape(-1)                                 # (nnz_chunk,)
            cols = cols.reshape(-1)
            vals = vals.reshape(-1)

            rows_all.append(rows)
            cols_all.append(cols)
            vals_all.append(vals)

        # Concatenate all chunks
        rows = torch.cat(rows_all, dim=0)
        cols = torch.cat(cols_all, dim=0)
        vals = torch.cat(vals_all, dim=0)

        # Build sparse COO and coalesce to sum duplicates (same (row,col))
        indices = torch.stack([rows, cols], dim=0)                 # (2, nnz)
        shape = (Co * K, Ci * K)
        M = torch.sparse_coo_tensor(indices, vals, size=shape, device=device, dtype=kernel.dtype).coalesce()

        if return_sparse_tensor:
            return M
        else:
            # return coalesced indices/values/shape
            return M.indices(), M.values(), M.shape

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
        
