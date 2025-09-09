# SPDX-License-Identifier: MIT
# Author: J.-M. Delouis
import numpy as np
import healpy as hp
import foscat.scat_cov as sc
import torch


class SphericalStencil:
    """
    Memory-friendly HEALPix convolution scaffolding with reusable geometry.
    - Step A (NumPy): build rotated stencil + HEALPix neighbors/weights for targets
    - Step B (Torch): map neighbors onto a sparse support and normalize weights
    - Step C (Torch): apply multi-channel kernel on batched sparse maps

    API summary
    -----------
    st = SphericalStencil(nside, kernel_sz, gauge='dual', ...)
    st.prepare_numpy(th, ph)                      # A: once per (targets, gauge, nside, kernel_sz)
    st.bind_support_torch(ids_sorted_np, device, dtype)   # B: once per sparse support (ids order)
    out = st.apply(data_sorted_t, kernel_t)       # C: many times; data_sorted: (B,Ci,K); kernel: (Ci,Co,P)

    Shapes
    ------
    - targets: K = len(th) = len(ph)  (also the length of `ids_sorted_np`)
    - data_sorted_t: (B, Ci, K)  ️ aligned to `ids_sorted_np` (same pixel order)
    - kernel_t:      (Ci, Co, P) with P = kernel_sz**2
    - output:        (B, Co, K)
    """
    # ---- Replace your current __init__ with this one (or merge the additions) ----
    def __init__(
        self,
        nside: int,
        kernel_sz: int,
        *,
        gauge: str = 'dual',       # 'phi' | 'axis' | 'dual'
        axisA=(1.0, 0.0, 0.0),     # for 'axis'/'dual' gauges
        axisB=(0.0, 1.0, 0.0),     # for 'dual' gauge
        blend: bool = True,        # soft blend for 'dual'
        power: float = 4.0,        # blend sharpness
        nest: bool = True,
        # NEW:
        cell_ids: np.ndarray | torch.Tensor | None = None,
        device = 'cuda',
        dtype = torch.float32,
        sort_ids: bool = True,
    ):
        assert kernel_sz >= 1 and int(kernel_sz) == kernel_sz
        self.nside     = int(nside)
        # Keep both names to stay compatible with your other code
        self.kernel_sz = int(kernel_sz)
        self.KERNELSZ  = self.kernel_sz
        self.P         = self.kernel_sz * self.kernel_sz

        self.gauge     = gauge
        self.axisA     = np.asarray(axisA, float) / np.linalg.norm(axisA)
        self.axisB     = np.asarray(axisB, float) / np.linalg.norm(axisB)
        self.blend     = bool(blend)
        self.power     = float(power)
        self.nest      = bool(nest)

        # Geometry (NumPy) — filled by prepare_numpy
        self.Kb = None
        self.idx_np = None  # (4, K*P)
        self.w_np   = None  # (4, K*P)

        # Sparse mapping (Torch) — filled by bind_support_torch
        self.ids_sorted_np = None
        self.device = None
        self.dtype  = None
        self.pos_safe_t = None  # (4, K*P) long
        self.w_norm_t   = None  # (4, K*P) float
        self.present_t  = None  # (4, K*P) bool

        # If you still rely on self.f elsewhere, keep it:
        self.f = None

        # ---------- Helper: default device/dtype ----------
        def _pick_device_dtype(dev_in, dt_in):
            dev = torch.device(
                dev_in if dev_in is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
            )
            dt = dt_in if dt_in is not None else torch.float32
            return dev, dt

        # ---------- Optional immediate precompute A+B ----------
        if cell_ids is not None:
            # to numpy
            if isinstance(cell_ids, torch.Tensor):
                cell_ids_np = cell_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                cell_ids_np = np.asarray(cell_ids, dtype=np.int64)

            # Step A: geometry for these targets (NumPy)
            th, ph = hp.pix2ang(self.nside, cell_ids_np, nest=self.nest)
            self.prepare_numpy(th, ph)  # fills self.idx_np, self.w_np, self.Kb

            # Choose sorted ids (recommended for fast searchsorted)
            if sort_ids:
                order = np.argsort(cell_ids_np)
                self.ids_sorted_np = cell_ids_np[order]
            else:
                self.ids_sorted_np = cell_ids_np

            # Step B: bind support on chosen device/dtype
            dev, dt = _pick_device_dtype(device, dtype)
            self.bind_support_torch(
                self.ids_sorted_np,
                device=dev,
                dtype=dt,
            )
            # Now the class is fully ready for fast convolutions on this support.
    '''
    # ------------------------- ctor -------------------------
    def __init__(
        self,
        nside: int,
        kernel_sz: int,
        *,
        gauge: str = 'dual',       # 'phi' | 'axis' | 'dual'
        axisA=(1.0, 0.0, 0.0),     # for 'axis'/'dual' gauges
        axisB=(0.0, 1.0, 0.0),     # for 'dual' gauge
        blend: bool = True,        # soft blend for 'dual'
        power: float = 4.0,        # blend sharpness
        nest: bool = True,
    ):
        assert kernel_sz >= 1 and int(kernel_sz) == kernel_sz
        self.nside     = int(nside)
        self.KERNELSZ = int(kernel_sz)
        self.P         = self.KERNELSZ * self.KERNELSZ
        self.gauge     = gauge
        self.axisA     = np.asarray(axisA, float) / np.linalg.norm(axisA)
        self.axisB     = np.asarray(axisB, float) / np.linalg.norm(axisB)
        self.blend     = bool(blend)
        self.power     = float(power)
        self.nest      = bool(nest)

        # Geometry (NumPy) — filled by prepare_numpy
        self.Kb = None          # number of targets
        self.idx_np = None      # (4, Kb*P) neighbor ids per stencil column
        self.w_np   = None      # (4, Kb*P) barycentric weights

        # Sparse mapping (Torch) — filled by bind_support_torch
        self.ids_sorted_np = None
        self.device = None
        self.dtype  = None
        self.pos_safe_t = None  # (4, Kb*P) torch.long
        self.w_norm_t   = None  # (4, Kb*P) torch.float
        self.present_t  = None  # (4, Kb*P) torch.bool

        self.f = None

    '''
    # ==================== Step A: NumPy geometry ====================

    def prepare_numpy(self, th, ph):
        """
        Build rotated stencil at each target (NumPy) and get HEALPix 4-neighbor
        indices + interpolation weights for each stencil point.

        Parameters
        ----------
        th, ph : array-like (K,)
            Target colatitude/longitude in radians.

        Fills
        -----
        self.Kb, self.idx_np, self.w_np
        """
        th = np.asarray(th, float)
        ph = np.asarray(ph, float)
        assert th.shape == ph.shape
        self.Kb = th.size

        # Local stencil in the pole-tangent frame
        vec = np.zeros((self.P, 3), dtype=float)
        grid = (np.arange(self.KERNELSZ) - self.KERNELSZ // 2) / self.nside
        vec[:, 0] = np.tile(grid, self.KERNELSZ)
        vec[:, 1] = np.repeat(grid, self.KERNELSZ)
        vec[:, 2] = 1.0 - np.sqrt(vec[:, 0]**2 + vec[:, 1]**2)

        # Rotate the stencil to each (th,ph) according to the selected gauge
        R = self._rotation_total(th, ph)                 # (K,3,3)
        rotated = np.einsum('kij,pj->kpi', R, vec)       # (K,P,3)

        # Query HEALPix: 4 neighbors + weights per stencil column
        t2, p2 = hp.vec2ang(rotated.reshape(-1, 3))
        self.idx_np, self.w_np = hp.get_interp_weights(self.nside, t2, p2, nest=self.nest)
        return self.idx_np, self.w_np  # handy return

    # ==================== Step B: Torch sparse binding ====================

    def bind_support_torch(
        self,
        ids_sorted_np: np.ndarray,
        *,
        ref_rule: str = 'same_as_before',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Map (idx_np, w_np) onto the compact sparse support given by ids_sorted_np,
        fix empty columns, and column-normalize weights — all on GPU.

        Parameters
        ----------
        ids_sorted_np : (K,) numpy.int64
            Sorted HEALPix pixel ids of your sparse map (must align with data[..., K]).
        ref_rule : str
            'same_as_before' -> fallback reference = kernel_sz + kernel_sz//2
            else uses center P//2.
        device, dtype : torch device/dtype (defaults: cuda / float32)

        Fills
        -----
        self.pos_safe_t : (4, K*P) torch.long
        self.w_norm_t   : (4, K*P) torch.float
        self.present_t  : (4, K*P) torch.bool
        """
        assert self.idx_np is not None and self.w_np is not None and self.Kb is not None, \
            "Call prepare_numpy(th, ph) before bind_support_torch(...)"

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.float32

        self.ids_sorted_np = np.asarray(ids_sorted_np, dtype=np.int64)
        assert self.ids_sorted_np.ndim == 1 and self.ids_sorted_np.size == self.Kb, \
            "ids_sorted_np must have length K == number of targets"

        # Move inputs to GPU
        ids_sorted = torch.as_tensor(self.ids_sorted_np, device=device, dtype=torch.long)
        idx = torch.as_tensor(self.idx_np, device=device, dtype=torch.long)     # (4, K*P)
        w   = torch.as_tensor(self.w_np,   device=device, dtype=dtype)          # (4, K*P)

        M = self.Kb * self.P

        # Locate neighbors (GPU)
        idx_flat = idx.reshape(-1)                                              # (4*M,)
        pos = torch.searchsorted(ids_sorted, idx_flat, right=False)
        in_range = pos < ids_sorted.numel()
        cmp = torch.full_like(idx_flat, -1, device=device)
        cmp[in_range] = ids_sorted[pos[in_range]]
        present_flat = in_range & (cmp == idx_flat)

        pos = pos.view(4, M)
        present = present_flat.view(4, M)

        # Handle empty columns
        empty_cols = ~present.any(dim=0)
        if empty_cols.any():
            if ref_rule == 'same_as_before':
                ref = self.KERNELSZ + self.KERNELSZ // 2
            else:
                ref = (self.P // 2)

            k_id = torch.arange(M, device=device, dtype=torch.long) // self.P
            ref_cols = k_id * self.P + ref
            src = ref_cols[empty_cols]

            idx[:, empty_cols] = idx[:, src]
            w[:,   empty_cols] = w[:,   src]

            # recompute presence/pos for those columns
            idx_e = idx[:, empty_cols].reshape(-1)
            pos_e = torch.searchsorted(ids_sorted, idx_e, right=False)
            inr_e = pos_e < ids_sorted.numel()
            cmp_e = torch.full_like(idx_e, -1, device=device)
            cmp_e[inr_e] = ids_sorted[pos_e[inr_e]]
            present[:, empty_cols] = (inr_e & (cmp_e == idx_e)).view(4, -1)
            pos[:, empty_cols] = pos_e.view(4, -1)

        # Normalize weights column-wise (zero missing neighbors)
        w = w * present
        colsum = w.sum(dim=0)
        nz = colsum > 0
        if nz.any():
            w[:, nz] = w[:, nz] / colsum[nz].unsqueeze(0)

        pos_safe = torch.where(present, pos, torch.zeros_like(pos))  # 0 where missing

        # Store on the class
        self.device = device
        self.dtype  = dtype
        self.pos_safe_t = pos_safe.to(torch.long)
        self.w_norm_t   = w.to(dtype)
        self.present_t  = present
        return self.pos_safe_t, self.w_norm_t

    # ==================== Step C: Torch apply ====================

    def apply(self, data_sorted_t: torch.Tensor, kernel_t: torch.Tensor) -> torch.Tensor:
        """
        Apply the (Ci,Co,P) kernel to batched sparse data (B,Ci,K) using precomputed
        (pos_safe, w_norm). Everything runs on GPU.

        Parameters
        ----------
        data_sorted_t : (B, Ci, K) torch.float  (device must match self.device)
            Sparse map aligned to ids_sorted_np (same order as in bind_support_torch).
        kernel_t      : (Ci, Co, P) torch.float (device must match self.device)

        Returns
        -------
        out : (B, Co, K) torch.float
        """
        assert self.pos_safe_t is not None and self.w_norm_t is not None, \
            "Call bind_support_torch(...) before apply(...)"

        assert data_sorted_t.device == self.device
        assert kernel_t.device      == self.device
        assert data_sorted_t.dtype  == self.dtype
        assert kernel_t.dtype       == self.dtype

        B, Ci, K = data_sorted_t.shape
        Ci_k, Co, P_k = kernel_t.shape
        assert K == self.Kb, "data last dimension must match number of targets"
        assert Ci_k == Ci,  "kernel input channels must match data channels"
        assert P_k == self.P, "kernel P must equal kernel_sz**2"

        M = self.Kb * self.P

        # Gather 4 neighbors (tiny loop of 4 is fine)
        vals = []
        for j in range(4):
            vj = data_sorted_t[..., self.pos_safe_t[j]]     # (B, Ci, M)
            vals.append(vj)
        vals = torch.stack(vals, dim=2)                     # (B, Ci, 4, M)

        # Weighted sum over neighbors → (B, Ci, M) then reshape → (B, Ci, K, P)
        tmp_cols = (vals * self.w_norm_t[None, None, :, :]).sum(dim=2)  # (B, Ci, M)
        tmp = tmp_cols.view(B, Ci, self.Kb, self.P)                     # (B, Ci, K, P)

        # Spatial + channel contraction
        out = torch.einsum('bckp,cop->bok', tmp, kernel_t)  # (B, Co, K)
        return out

    # ---------------- rotations + gauges (NumPy, internal) ----------------

    def _rotation_total(self, th, ph):
        """Build R_tot(K,3,3) in NumPy according to selected gauge."""
        if self.gauge == 'phi':
            return self._rotation_total_phi(th, ph)
        elif self.gauge == 'axis':
            return self._rotation_total_axis(th, ph, self.axisA)
        elif self.gauge == 'dual':
            return self._rotation_total_dual(th, ph, self.axisA, self.axisB, self.blend, self.power)
        raise ValueError("gauge must be 'phi', 'axis', or 'dual'")

    @staticmethod
    def _Rz_batch(phi):
        c, s = np.cos(phi), np.sin(phi)
        R = np.zeros((phi.size, 3, 3), dtype=float)
        R[:, 0, 0] = c;  R[:, 0, 1] = -s
        R[:, 1, 0] = s;  R[:, 1, 1] =  c
        R[:, 2, 2] = 1.0
        return R

    @staticmethod
    def _Ry_batch(th):
        c, s = np.cos(th), np.sin(th)
        R = np.zeros((th.size, 3, 3), dtype=float)
        R[:, 0, 0] = c;  R[:, 0, 2] = s
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -s; R[:, 2, 2] = c
        return R

    @staticmethod
    def _rodrigues_about_axes(n, alpha):
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        ca, sa = np.cos(alpha), np.sin(alpha)
        I = np.eye(3)[None, :, :]
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
        Kmat = np.zeros((n.shape[0], 3, 3))
        Kmat[:, 0, 1] = -nz; Kmat[:, 0, 2] =  ny
        Kmat[:, 1, 0] =  nz; Kmat[:, 1, 2] = -nx
        Kmat[:, 2, 0] = -ny; Kmat[:, 2, 1] =  nx
        outer = n[:, :, None] * n[:, None, :]
        return I * ca[:, None, None] + sa[:, None, None] * Kmat + (1.0 - ca)[:, None, None] * outer

    def _rotation_total_phi(self, th, ph):
        Rb = self._Rz_batch(ph) @ self._Ry_batch(th)
        n  = (Rb @ np.array([0.0, 0.0, 1.0])).reshape(-1, 3)
        Rg = self._rodrigues_about_axes(n, ph)
        return np.einsum('kij,kjl->kil', Rg, Rb)

    def _rotation_total_axis(self, th, ph, axis, eps=1e-12):
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        Rb = self._Rz_batch(ph) @ self._Ry_batch(th)
        ez = np.array([0.0, 0.0, 1.0])
        ex = np.array([1.0, 0.0, 0.0])
        n  = (Rb @ ez).reshape(-1, 3)
        ux = (Rb @ ex).reshape(-1, 3)
        a_dot_n = (axis[None, :] * n).sum(axis=1)
        e1 = axis[None, :] - a_dot_n[:, None] * n
        e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), eps)
        s = (n * np.cross(ux, e1)).sum(axis=1)
        c = (ux * e1).sum(axis=1)
        alpha = np.arctan2(s, c)
        Rg = self._rodrigues_about_axes(n, alpha)
        return np.einsum('kij,kjl->kil', Rg, Rb)

    @staticmethod
    def _rot_to_quat(R):
        K = R.shape[0]
        q = np.empty((K, 4), dtype=float)
        tr = np.trace(R, axis1=1, axis2=2)
        for i in range(K):
            if tr[i] > 0:
                S = np.sqrt(tr[i] + 1.0) * 2.0
                q[i, 0] = 0.25 * S
                q[i, 1] = (R[i, 2, 1] - R[i, 1, 2]) / S
                q[i, 2] = (R[i, 0, 2] - R[i, 2, 0]) / S
                q[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) / S
            else:
                j = np.argmax(np.diag(R[i]))
                if j == 0:
                    S = np.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2.0
                    q[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / S
                    q[i, 1] = 0.25 * S
                    q[i, 2] = (R[i, 0, 1] + R[i, 1, 0]) / S
                    q[i, 3] = (R[i, 0, 2] + R[i, 2, 0]) / S
                elif j == 1:
                    S = np.sqrt(1.0 - R[i, 0, 0] + R[i, 1, 1] - R[i, 2, 2]) * 2.0
                    q[i, 0] = (R[i, 0, 2] - R[i, 2, 0]) / S
                    q[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / S
                    q[i, 2] = 0.25 * S
                    q[i, 3] = (R[i, 1, 2] + R[i, 2, 1]) / S
                else:
                    S = np.sqrt(1.0 - R[i, 0, 0] - R[i, 1, 1] + R[i, 2, 2]) * 2.0
                    q[i, 0] = (R[i, 1, 0] - R[i, 0, 1]) / S
                    q[i, 1] = (R[i, 0, 2] + R[i, 2, 0]) / S
                    q[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / S
                    q[i, 3] = 0.25 * S
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return q

    @staticmethod
    def _quat_to_rot(q):
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.empty((q.shape[0], 3, 3), dtype=float)
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - z*w)
        R[:, 0, 2] = 2 * (x*z + y*w)
        R[:, 1, 0] = 2 * (x*y + z*w)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - x*w)
        R[:, 2, 0] = 2 * (x*z - y*w)
        R[:, 2, 1] = 2 * (y*z + x*w)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R

    def _rotation_total_dual(self, th, ph, axisA, axisB, blend, power):
        Rb = self._Rz_batch(ph) @ self._Ry_batch(th)
        n = (Rb @ np.array([0.0, 0.0, 1.0])).reshape(-1, 3)
        aA = axisA / np.linalg.norm(axisA)
        aB = axisB / np.linalg.norm(axisB)
        qA = np.linalg.norm(np.cross(aA[None, :], n), axis=1)
        qB = np.linalg.norm(np.cross(aB[None, :], n), axis=1)

        if not blend:
            useA = qA >= qB
            R = np.empty((th.size, 3, 3), dtype=float)
            if np.any(useA):
                R[useA]  = self._rotation_total_axis(th[useA],  ph[useA],  aA)
            if np.any(~useA):
                R[~useA] = self._rotation_total_axis(th[~useA], ph[~useA], aB)
            return R

        wA = (qA ** power); wB = (qB ** power)
        wsum = np.clip(wA + wB, 1e-12, None)
        wA /= wsum; wB /= wsum

        RA = self._rotation_total_axis(th, ph, aA)
        RB = self._rotation_total_axis(th, ph, aB)
        qA_ = self._rot_to_quat(RA); qB_ = self._rot_to_quat(RB)
        q = wA[:, None] * qA_ + wB[:, None] * qB_
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return self._quat_to_rot(q)

    def _Convol_Torch(self, data: torch.Tensor, kernel: torch.Tensor, cell_ids=None) -> torch.Tensor:
        """
        Convenience entry point:
          - If cell_ids is None: use cached geometry (prepare_numpy) and sparse mapping
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
            self.prepare_numpy(th, ph)

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
        
