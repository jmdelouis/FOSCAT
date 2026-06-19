"""
alm_loc_optim.py
================
Optimised variant of alm_loc for spherical-harmonic computation
on a restricted HEALPix domain.

Two key optimisations relative to alm_loc
----------------------------------------------
1.  Restriction in m  (longitude axis)
    For each ring r with cnt pixels out of N_ring total:
      - full ring    : mode m valid if  (m % N_ring) <= N_ring // 2
      - partial ring : mode m valid if  (m % N_ring) <= cnt // 2   (partial Nyquist)
    For each m, only rings satisfying this criterion feed
    the Legendre projection.

2.  Restriction in l  (latitude axis)
    With R_m valid rings for mode m, the Legendre system has R_m
    equations.  At most R_m ell degrees can be constrained.
    Hence:
        lmax_eff(m) = min(lmax,  m + R_m - 1)
    and Legendre polynomials are computed only up to this ceiling.

Resulting savings
-------------------
La projection de Legendre passe de O(R × (lmax - m + 1))
to O(R_m × (lmax_eff(m) - m + 1)), which is quadratically more
favorable pour les petits domaines.

Public API (mirroring alm_loc)
---------------------------------
analyze_domain(nside, cell_ids, nest, lmax)
    → rings_used, counts, sizes, valid_idx_per_m, lmax_eff_per_m, m_valid

map2alm_loc_optim(im, nside, cell_ids, nest, lmax)
    → alm_per_m, m_valid, lmax_eff_per_m   (sparse alm)

anafast_loc_optim(im, nside, cell_ids, nest, lmax)
    → cl [lmax+1], m_count [lmax+1]

sparse_to_dense(alm_per_m, m_valid, lmax_eff_per_m, lmax)
    → dense alm vector (zeros for uncalculated modes),
      compatible avec la mise en page de alm_loc.map2alm_loc
"""

import numpy as np
import torch

from foscat.alm_loc import alm_loc


class alm_loc_optim(alm_loc):

    def __init__(self, backend=None, lmax=24, limit_range=1e10):
        super().__init__(backend=backend, lmax=lmax, limit_range=limit_range)

    # ================================================================== #
    #  Analyse du domaine : quels (l, m) sont contraints par le patch ?   #
    # ================================================================== #

    def analyze_domain(self, nside: int, cell_ids, nest: bool = False,
                       lmax: int = None):
        """
        Pre-compute the set of (l, m) modes that are effectively
        constrained by the partial patch described by cell_ids.

        Parameters
        ----------
        nside    : HEALPix resolution
        cell_ids : indices de pixels (ring ou nested)
        nest     : True si cell_ids utilise l'ordre NESTED
        lmax     : maximum multipole (default: min(self.lmax, 3*nside-1))

        Returns
        -------
        rings_used       : ndarray int32 [R]    indices of present rings
        counts           : ndarray int32 [R]    nb de pixels par ring
        sizes            : ndarray int32         N_ring pour chaque ring du nside
        valid_idx_per_m  : dict  m -> ndarray int32  indices dans rings_used
        lmax_eff_per_m   : dict  m -> int   lmax effectif pour ce m
        m_valid          : list[int]  valeurs de m ayant au moins 1 ring valide
        """
        nside = int(nside)
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        if lmax is None:
            lmax = min(self.lmax, 3 * nside - 1)
        lmax = int(lmax)

        # Initialise les caches parent (matrix_shift_ph, A/B, ratio_mm…)
        self.shift_ph(nside)

        ring_ids = self._to_ring_ids(nside, cell_ids, nest)
        ring_idx, pos, order, starts, sizes = self._group_by_ring(nside, ring_ids)
        ring_idx_sorted = ring_idx[order]

        rings_used, start_ptr, counts = np.unique(
            ring_idx_sorted, return_index=True, return_counts=True
        )

        # N_ring for each used ring
        Nrings = sizes[rings_used].astype(np.int64)   # [R]
        nyquist = (counts // 2).astype(np.int64)       # [R]  Nyquist partiel
        is_full = counts == Nrings                     # [R]  bool

        valid_idx_per_m: dict = {}
        lmax_eff_per_m: dict = {}
        m_valid: list = []

        for m in range(lmax + 1):
            m_mod = (m % Nrings).astype(np.int64)     # aliased frequency [R]

            # Full ring   : standard FFT Nyquist criterion (m_mod <= N_ring/2)
            # Partial ring: reduced Nyquist criterion          (m_mod <= cnt/2)
            valid_mask = (is_full & (m_mod <= Nrings // 2)) | \
                         (~is_full & (m_mod <= nyquist))

            valid = np.where(valid_mask)[0].astype(np.int32)

            if valid.size > 0:
                valid_idx_per_m[m] = valid
                # With R_m rings we constrain at most R_m ell degrees
                lmax_eff_per_m[m] = int(min(lmax, m + valid.size - 1))
                m_valid.append(m)

        return rings_used, counts, sizes, valid_idx_per_m, lmax_eff_per_m, m_valid

    # ================================================================== #
    #  map -> sparse alm (optimised)                                      #
    # ================================================================== #

    def map2alm_loc_optim(self, im, nside: int, cell_ids,
                          nest: bool = False, lmax: int = None):
        """
        Compute spherical-harmonic coefficients on a partial patch,
        restricted to the (l, m) modes effectively constrained by the
        partial sky coverage.

        Parameters
        ----------
        im       : [..., n_pixels]  map values on the patch
        nside    : HEALPix resolution
        cell_ids : pixel indices in the patch
        nest     : True if NESTED ordering
        lmax     : maximum multipole

        Returns
        -------
        alm_per_m      : list of tensors, one per valid m.
                         alm_per_m[i] a la forme [..., lmax_eff(m)-m+1]
                         pour m = m_valid[i]
        m_valid        : list[int]   m values in the same order
        lmax_eff_per_m : dict  m -> int
        """
        nside = int(nside)
        if lmax is None:
            lmax = min(self.lmax, 3 * nside - 1)
        lmax = int(lmax)

        # Gestion dimension batch
        _added_batch = False
        if hasattr(im, 'ndim') and im.ndim == 1:
            im = im[None, :]
            _added_batch = True
        elif not hasattr(im, 'ndim') and len(im.shape) == 1:
            im = im[None, :]
            _added_batch = True

        # Analyse du domaine (appelle aussi shift_ph)
        rings_used, counts, sizes, valid_idx_per_m, lmax_eff_per_m, m_valid = \
            self.analyze_domain(nside, cell_ids, nest=nest, lmax=lmax)

        # Fourier transform per ring — identical to comp_tf_loc
        # ft : [..., R, lmax+1]
        _, ft = self.comp_tf_loc(im, nside=nside, cell_ids=cell_ids,
                                 nest=nest, realfft=True, mmax=lmax)

        # cos(theta) for all used rings
        co_th_all = np.cos(self.ring_th(nside)[rings_used])   # [R]

        alm_per_m = []

        for m in m_valid:
            vidx    = valid_idx_per_m[m]     # indices dans rings_used, [R_m]
            lmax_m  = lmax_eff_per_m[m]      # lmax effectif pour ce m
            n_l     = lmax_m - m + 1         # number of ell degrees computed

            co_th_m = co_th_all[vidx]        # [R_m]

            # Legendre polynomials P_{lm}(cos θ) for l = m..lmax_m
            # Forme : [n_l, R_m]
            # Pass lmax_m instead of lmax → savings on the recurrence
            plm = self.compute_legendre_m(co_th_m, m, lmax_m, nside) \
                  / (12 * nside**2)
            plm_bk = self.backend.bk_cast(plm)   # [n_l, R_m]

            # Coefficients de Fourier au mode m pour les rings valides
            # ft[..., vidx, m] : [..., R_m]
            ft_m = ft[..., vidx, m]

            # Projection : [..., n_l] = sum_{R_m} ft_m * P_{lm}
            # extended ft_m: [..., 1, R_m]  ×  plm [n_l, R_m]  → sum over R_m
            tmp = self.backend.bk_reduce_sum(
                self.backend.bk_expand_dims(ft_m, axis=-2) * plm_bk,
                axis=-1
            )   # [..., n_l]

            # sqrt(2l+1) weighting (consistent with map2alm_loc)
            l_vals = np.arange(m, lmax_m + 1, dtype=np.float64)
            scale  = self.backend.bk_cast(np.sqrt(2.0 * l_vals + 1.0))
            scale  = scale.reshape((1,) * (tmp.ndim - 1) + (n_l,))
            tmp    = tmp * scale

            if _added_batch:
                tmp = tmp[0]

            alm_per_m.append(tmp)

        return alm_per_m, m_valid, lmax_eff_per_m

    # ================================================================== #
    #  Sparse -> dense conversion (compatibility with alm_loc)            #
    # ================================================================== #

    def sparse_to_dense(self, alm_per_m, m_valid, lmax_eff_per_m, lmax: int):
        """
        Convertit l'alm creux (sortie de map2alm_loc_optim) vers le vecteur
        flat dense array used by alm_loc.map2alm_loc.

        Format dense : [m=0: l=0..lmax | m=1: l=1..lmax | …]
        Uncalculated modes are filled with zeros.

        Parameters
        ----------
        alm_per_m      : liste de tenseurs (sortie de map2alm_loc_optim)
        m_valid        : list[int]
        lmax_eff_per_m : dict m -> int
        lmax           : maximum multipole used during computation

        Retourne
        --------
        out : tensor [..., total_alm]  same dtype and device as input
        """
        if not alm_per_m:
            raise ValueError("alm_per_m est vide.")

        sample       = alm_per_m[0]
        batch_shape  = sample.shape[:-1] if sample.ndim > 1 else ()
        device       = sample.device
        dtype        = sample.dtype
        total        = sum(lmax - m + 1 for m in range(lmax + 1))

        out = torch.zeros(batch_shape + (total,), dtype=dtype, device=device)

        # Offset of each m in the dense vector
        offset = 0
        m_to_offset = {}
        for m in range(lmax + 1):
            m_to_offset[m] = offset
            offset += lmax - m + 1

        for m, alm_m in zip(m_valid, alm_per_m):
            lmax_m = lmax_eff_per_m[m]
            n_l    = lmax_m - m + 1
            off    = m_to_offset[m]
            out[..., off:off + n_l] = alm_m

        return out

    # ================================================================== #
    #  alm -> Cl  (optimised)                                             #
    # ================================================================== #

    def anafast_loc_optim(self, im, nside: int, cell_ids,
                          nest: bool = False, lmax: int = None):
        """
        Estimate the angular power spectrum Cl on a partial patch.

        Only the (l, m) modes effectively constrained by the patch
        contribute to the estimate.  For each l, Cl is normalised by
        the number of available m modes (graceful degradation rather than
        dilution by zero modes).

        Parameters
        ----------
        im       : [..., n_pixels]
        nside    : HEALPix resolution
        cell_ids : pixel indices in the patch
        nest     : True if NESTED ordering
        lmax     : maximum multipole

        Retourne
        --------
        cl      : tenseur [..., lmax+1]
        m_count : ndarray int32 [lmax+1]
                  number of m modes contributing to each l
                  (flags poorly constrained multipoles:
                   m_count[l] == 0  →  Cl[l] undefined)
        """
        nside = int(nside)
        if lmax is None:
            lmax = min(self.lmax, 3 * nside - 1)
        lmax = int(lmax)

        alm_per_m, m_valid, lmax_eff_per_m = self.map2alm_loc_optim(
            im, nside=nside, cell_ids=cell_ids, nest=nest, lmax=lmax
        )

        # Forme batch et device
        if alm_per_m:
            sample      = alm_per_m[0]
            batch_shape = sample.shape[:-1] if sample.ndim > 1 else ()
            device      = sample.device
        else:
            batch_shape = ()
            device      = torch.device('cpu')

        cl      = torch.zeros(batch_shape + (lmax + 1,),
                              dtype=torch.float64, device=device)
        m_count = np.zeros(lmax + 1, dtype=np.int32)

        for m, alm_m in zip(m_valid, alm_per_m):
            lmax_m = lmax_eff_per_m[m]

            # |a_{lm}|^2  pour l = m..lmax_m
            power  = self.backend.bk_real(
                alm_m * self.backend.bk_conjugate(alm_m)
            )   # batch + (lmax_m - m + 1,)

            # Facteur 2 pour m > 0 (modes m et -m)
            weight = 1.0 if m == 0 else 2.0
            cl[..., m:lmax_m + 1] += weight * power
            m_count[m:lmax_m + 1] += 1 if m == 0 else 2

        # Normalisation: divide by the number of modes contributing to each l
        # (et non par (2l+1) global, qui supposerait le ciel complet)
        norm = np.where(m_count > 0,
                        m_count.astype(np.float64),
                        1.0)   # avoid division by zero
        norm_t = torch.tensor(norm, dtype=torch.float64, device=device)
        norm_t = norm_t.reshape((1,) * len(batch_shape) + (lmax + 1,))
        cl = cl / norm_t

        return cl, m_count

    # ================================================================== #
    #  Utility: domain summary (diagnostic)                               #
    # ================================================================== #

    def domain_summary(self, nside: int, cell_ids, nest: bool = False,
                       lmax: int = None):
        """
        Print a human-readable summary of the partial domain:
        sky fraction covered, number of rings, effective (m, l) mode count
        vs brute-force, and estimated computation gain.
        """
        nside = int(nside)
        if lmax is None:
            lmax = min(self.lmax, 3 * nside - 1)
        lmax = int(lmax)

        rings_used, counts, sizes, valid_idx_per_m, lmax_eff_per_m, m_valid = \
            self.analyze_domain(nside, cell_ids, nest=nest, lmax=lmax)

        n_pix_patch = int(counts.sum())
        n_pix_full  = 12 * nside**2
        f_sky       = n_pix_patch / n_pix_full

        n_rings_full  = 4 * nside - 1
        n_rings_patch = len(rings_used)

        # Brute-force cost (original alm_loc): sum_m  R × (lmax - m + 1)
        cost_full = sum(n_rings_patch * (lmax - m + 1) for m in range(lmax + 1))

        # Optimised cost: sum_m  R_m × (lmax_eff(m) - m + 1)
        cost_opt = sum(
            len(valid_idx_per_m[m]) * (lmax_eff_per_m[m] - m + 1)
            for m in m_valid
        )

        n_alm_full = sum(lmax - m + 1 for m in range(lmax + 1))
        n_alm_opt  = sum(lmax_eff_per_m[m] - m + 1 for m in m_valid)

        print(f"=== Domain summary  (nside={nside}, lmax={lmax}) ===")
        print(f"  Sky fraction         : {f_sky:.3%}  ({n_pix_patch}/{n_pix_full} pixels)")
        print(f"  Rings used           : {n_rings_patch}/{n_rings_full}")
        print(f"  Valid m values       : {len(m_valid)}/{lmax+1}  "
              f"(first gap at m={m_valid[-1]+1 if m_valid else 0})")
        print(f"  alm coefficients     : {n_alm_opt}/{n_alm_full}  "
              f"({100*n_alm_opt/n_alm_full:.1f}%)")
        print(f"  Legendre ops (estim) : {cost_opt}/{cost_full}  "
              f"→ speedup ×{cost_full/max(cost_opt,1):.1f}")
        print()

        return {
            'f_sky'         : f_sky,
            'n_rings'       : n_rings_patch,
            'n_m_valid'     : len(m_valid),
            'n_alm_opt'     : n_alm_opt,
            'n_alm_full'    : n_alm_full,
            'speedup_estim' : cost_full / max(cost_opt, 1),
            'lmax_eff_per_m': lmax_eff_per_m,
            'm_valid'       : m_valid,
        }
