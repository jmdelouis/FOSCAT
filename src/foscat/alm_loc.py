
import numpy as np
import healpy as hp

from foscat.alm import alm as _alm
import torch

class alm_loc(_alm):
    """
    Local/partial-sky variant of foscat.alm.alm.

    Key design choice (to match alm.py exactly when full-sky is provided):
    - Reuse *all* Legendre/normalization machinery from the parent class (alm),
      i.e. shift_ph(), compute_legendre_m(), ratio_mm, A/B recurrences, etc.
      This is critical for matching alm.map2alm() numerically.

    Differences vs alm.map2alm():
    - Input map is [..., n] with explicit (nside, cell_ids)
    - Only rings touched by cell_ids are processed.
    - For rings with full coverage, we run the exact same FFT+tiling logic as alm.comp_tf()
      (but only for those rings) -> bitwise comparable up to backend FFT differences.
    - For rings with partial coverage, we compute a *partial DFT* for m=0..mmax,
      using the same phase convention as alm.comp_tf():
          FFT kernel uses exp(-i 2pi (m mod Nring) j / Nring)
          then apply the per-ring shift exp(-i m phi0) via self.matrix_shift_ph
    """

    def __init__(self, backend=None, lmax=24, limit_range=1e10):
        super().__init__(backend=backend, lmax=lmax, nside=None, limit_range=limit_range)

    # --------- helpers: ring layout identical to alm.ring_th/ring_ph ----------
    @staticmethod
    def _ring_starts_sizes(nside: int):
        starts = []
        sizes = []
        n = 0
        for k in range(nside - 1):
            N = 4 * (k + 1)
            starts.append(n); sizes.append(N)
            n += N
        for _ in range(2 * nside + 1):
            N = 4 * nside
            starts.append(n); sizes.append(N)
            n += N
        for k in range(nside - 1):
            N = 4 * (nside - 1 - k)
            starts.append(n); sizes.append(N)
            n += N
        return np.asarray(starts, np.int64), np.asarray(sizes, np.int32)

    def _to_ring_ids(self, nside: int, cell_ids: np.ndarray, nest: bool) -> np.ndarray:
        if nest:
            return hp.nest2ring(nside, cell_ids)
        return cell_ids

    def _group_by_ring(self, nside: int, ring_ids: np.ndarray):
        """
        Returns:
          ring_idx: ring number (0..4*nside-2) per pixel
          pos:      position along ring (0..Nring-1) per pixel
          order:    sort order grouping by ring then pos
          starts,sizes: ring layout
        """
        starts, sizes = self._ring_starts_sizes(nside)

        # ring index = last start <= ring_id
        ring_idx = np.searchsorted(starts, ring_ids, side="right") - 1
        ring_idx = ring_idx.astype(np.int32)

        pos = (ring_ids - starts[ring_idx]).astype(np.int32)

        order = np.lexsort((pos, ring_idx))
        return ring_idx, pos, order, starts, sizes

    # ------------------ local Fourier transform per ring ---------------------
    def comp_tf_loc(self, im, nside: int, cell_ids, nest: bool = False, realfft: bool = True, mmax=None):
        """
        Returns:
          rings_used: 1D np.ndarray of ring indices present
          ft: backend tensor of shape [..., nrings_used, mmax+1] (complex)
               where last axis is m, ring axis matches rings_used order.
        """
        nside = int(nside)
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        if mmax is None:
            mmax = min(self.lmax, 3 * nside - 1)
        mmax = int(mmax)

        # Ensure parent caches for this nside exist (matrix_shift_ph, A/B, ratio_mm, etc.)
        self.shift_ph(nside)

        ring_ids = self._to_ring_ids(nside, cell_ids, nest)
        ring_idx, pos, order, starts, sizes = self._group_by_ring(nside, ring_ids)

        ring_idx = ring_idx[order]
        pos = pos[order]

        i_im = self.backend.bk_cast(im)
        i_im = self.backend.bk_gather(i_im, order, axis=-1)  # reorder last axis

        rings_used, start_ptr, counts = np.unique(ring_idx, return_index=True, return_counts=True)

        # Build output per ring as list then concat
        out_per_ring = []
        for r, s0, cnt in zip(rings_used.tolist(), start_ptr.tolist(), counts.tolist()):
            Nring = int(sizes[r])
            p = pos[s0:s0+cnt]

            v = self.backend.bk_gather(i_im, np.arange(s0, s0+cnt, dtype=np.int64), axis=-1)

            if cnt == Nring:
                # Full ring: exact same FFT+tiling logic as alm.comp_tf for 1 ring
                # Need data ordered by pos (already grouped, but ensure pos is 0..N-1)
                if not np.all(p == np.arange(Nring, dtype=p.dtype)):
                    # reorder within ring
                    sub_order = np.argsort(p)
                    v = self.backend.bk_gather(v, sub_order, axis=-1)

                if realfft:
                    tmp = self.rfft2fft(v)
                else:
                    tmp = self.backend.bk_fft(v)

                l_n = tmp.shape[-1]
                if l_n < mmax + 1:
                    repeat_n = (mmax // l_n) + 1
                    tmp = self.backend.bk_tile(tmp, repeat_n, axis=-1)

                tmp = tmp[..., :mmax+1]

                # Apply per-ring shift exp(-i m phi0) exactly like alm.comp_tf
                shift = self.matrix_shift_ph[nside][r, :mmax+1]  # [m]
                tmp = tmp * shift
                out_per_ring.append(self.backend.bk_expand_dims(tmp, axis=-2))  # [...,1,m]
            else:
                # Partial ring: partial DFT for required m, using same aliasing as FFT branch
                m_vec = np.arange(mmax+1, dtype=np.int64)
                m_mod = (m_vec % Nring).astype(np.int64)

                # angles: 2pi * pos * m_mod / Nring
                ang = (2.0 * np.pi / Nring) * p.astype(np.float64)[:, None] * m_mod[None, :].astype(np.float64)
                ker = np.exp(-1j * ang).astype(np.complex128)  # [cnt, m]

                ker_bk = self.backend.bk_cast(ker)

                # v is [..., cnt]; we want [..., m] = sum_cnt v*ker
                tmp = self.backend.bk_reduce_sum(
                    self.backend.bk_expand_dims(v, axis=-1) * ker_bk,
                    axis=-2
                )  # [..., m]

                shift = self.matrix_shift_ph[nside][r, :mmax+1]  # [m] true m shift
                tmp = tmp * shift
                out_per_ring.append(self.backend.bk_expand_dims(tmp, axis=-2))  # [...,1,m]

        ft = self.backend.bk_concat(out_per_ring, axis=-2)  # [..., nrings, m]
        return np.asarray(rings_used, dtype=np.int32), ft

    # ---------------------------- map -> alm --------------------------------
    def map2alm_loc(self, im, nside: int, cell_ids, nest: bool = False, lmax=None):
        nside = int(nside)
        if lmax is None:
            lmax = min(self.lmax, 3 * nside - 1)
        lmax = int(lmax)

        # Ensure a batch dimension like alm.map2alm expects
        _added_batch = False
        if hasattr(im, 'ndim') and im.ndim == 1:
            im = im[None, :]
            _added_batch = True
        elif (not hasattr(im, 'ndim')) and len(im.shape) == 1:
            im = im[None, :]
            _added_batch = True

        rings_used, ft = self.comp_tf_loc(im, nside=nside, cell_ids=cell_ids, nest=nest, realfft=True, mmax=lmax)

        # cos(theta) on used rings
        co_th = np.cos(self.ring_th(nside)[rings_used])

        # ft is [..., R, m]
        alm_out = None
        
        

        for m in range(lmax + 1):
            # IMPORTANT: reuse alm.compute_legendre_m and its normalization exactly
            plm = self.compute_legendre_m(co_th, m, lmax, nside) / (12 * nside**2)  # [L,R]
            plm_bk = self.backend.bk_cast(plm)

            ft_m = ft[..., :, m]  # [..., R]
            tmp = self.backend.bk_reduce_sum(
                self.backend.bk_expand_dims(ft_m, axis=-2) * plm_bk,
                axis=-1
            )  # [..., L]
            l_vals = np.arange(m, lmax + 1, dtype=np.float64)
            scale = np.sqrt(2.0 * l_vals + 1.0)

            # convertir scale en backend tensor (torch) sur le bon device
            scale_t = self.backend.bk_cast(scale)  # ou un helper équivalent
            # reshape pour broadcast si nécessaire: [1, L] ou [L]
            shape = (1,) * (tmp.ndim - 1) + (scale_t.shape[0],)
            scale_t = scale_t.reshape(shape)
            
            tmp = tmp * scale_t
            if m == 0:
                alm_out = tmp
            else:
                alm_out = self.backend.bk_concat([alm_out, tmp], axis=-1)
        if _added_batch:
            alm_out = alm_out[0]
        return alm_out

    # ---------------------------- alm -> Cl ---------------------------------
    def anafast_loc(self, im, nside: int, cell_ids, nest: bool = False, lmax=None):
        if lmax is None:
            lmax = min(self.lmax, 3 * nside - 1)
        lmax = int(lmax)

        alm = self.map2alm_loc(im, nside=nside, cell_ids=cell_ids, nest=nest, lmax=lmax)
        
        # Unpack and compute Cl with correct real-field folding:
        cl = torch.zeros((lmax + 1,), dtype=alm.dtype, device=alm.device)
        
        idx = 0
        for m in range(lmax + 1):
            L = lmax - m + 1
            a = alm[..., idx:idx+L]
            idx += L
            p = self.backend.bk_real(a * self.backend.bk_conjugate(a))
            # sum over any batch dims
            p = self.backend.bk_reduce_sum(p, axis=tuple(range(p.ndim-1))) if p.ndim > 1 else p
            if m == 0:
                cl[m:] += p
            else:
                cl[m:] += 2.0 * p
        denom = (2*torch.arange(lmax+1,dtype=p.dtype, device=alm.device)+1)
        cl = cl / denom
        return cl
