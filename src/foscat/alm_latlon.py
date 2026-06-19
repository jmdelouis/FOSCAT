"""
alm_latlon.py
=============
Spherical-harmonic transform for maps defined on an arbitrary
colatitude / longitude grid organised into rings.

Differences with respect to foscat.alm.alm
-----------------------------------------
- No dependency on HEALPix for pixel positioning.
- Rings may have arbitrary colatitudes (not HEALPix colatitudes)
  and arbitrary longitudes (not necessarily uniform).
- Longitude step: direct DFT when ring φ values are irregular;
  FFT + phase shift (like alm.comp_tf) when φ are uniformly spaced.
- Colatitude step: same Legendre recurrence as alm.compute_legendre_m,
  evaluated at the cosines of the provided colatitudes.
- Quadrature weights: trapezoidal in θ (sin θ · Δθ) × uniform in φ (2π/N_r)
  by default, or user-supplied weight array.

Main API
--------------
build_rings_from_latlon(lat, lon, atol)
    Groups a flat list of (lat, lon) into rings of equal colatitude.

compute_weights(ring_theta, ring_phi_list, ring_counts)
    Computes the quadrature weights per pixel (steradians).

comp_tf_latlon(im, ring_phi_list, ring_counts, pixel_weights, mmax)
    Ring-weighted DFT → ft[..., R, mmax+1].

map2alm_latlon(im, ring_theta, ring_phi_list, ring_counts, lmax, weights)
    Map → alm.

anafast_latlon(im, ring_theta, ring_phi_list, ring_counts, lmax, weights)
    Map → Cl.

alm2map_latlon(alm, ring_theta, ring_phi_list, ring_counts, lmax)
    alm → map (synthesis).

Minimal example
---------------
    import numpy as np
    from foscat.alm_latlon import alm_latlon

    # Regular grid (ntheta=64 rings × nphi=128 pixels per ring)
    ntheta, nphi = 64, 128
    theta_1d = np.linspace(np.pi / (2*ntheta), np.pi - np.pi/(2*ntheta), ntheta)
    phi_1d   = np.linspace(0, 2*np.pi*(1 - 1/nphi), nphi)

    lat  = np.repeat(theta_1d, nphi)        # colatitude of each pixel
    lon  = np.tile(phi_1d,     ntheta)      # longitude of each pixel

    ring_theta, ring_phi_list, ring_counts, sort_idx = \\
        alm_latlon.build_rings_from_latlon(lat, lon)

    obj  = alm_latlon(lmax=32)
    im   = np.random.randn(ntheta * nphi)

    alm_coeffs = obj.map2alm_latlon(
        im[sort_idx], ring_theta, ring_phi_list, ring_counts
    )
    cl = obj.anafast_latlon(
        im[sort_idx], ring_theta, ring_phi_list, ring_counts
    )
"""

import numpy as np
import torch

from foscat.alm import alm as _alm


class alm_latlon(_alm):
    """
    Spherical-harmonic transform on an arbitrary lat/lon grid organised into rings.
    """

    def __init__(self, backend=None, lmax=24, limit_range=1e10):
        # nside=None: no HEALPix grid, maxlog computed from lmax
        super().__init__(backend=backend, lmax=lmax, nside=None,
                         limit_range=limit_range)

    # ================================================================== #
    #  Build rings from a flat (lat, lon) array                           #
    # ================================================================== #

    @staticmethod
    def build_rings_from_latlon(lat, lon, atol=1e-10, convention='colatitude_rad'):
        """
        Group a flat list of pixels into rings of equal colatitude.

        Parameters
        ----------
        lat  : array [N]  angular coordinate of each pixel (see convention).
        lon  : array [N]  longitudinal coordinate of each pixel (see convention).
        atol : tolerance in radians for grouping two pixels into the same ring.
        convention : str  format of the input coordinates.

            'colatitude_rad'  (default)
                lat = colatitude θ in RADIANS    0 → π
                lon = longitude  φ in RADIANS    0 → 2π

            'colatitude_deg'
                lat = colatitude θ in DEGREES    0° → 180°
                lon = longitude  φ in DEGREES    0° → 360°

            'geographic_rad'
                lat = geographic latitude in RADIANS   −π/2 → +π/2
                lon = longitude in RADIANS             −π → +π  or  0 → 2π

            'geographic_deg'
                lat = geographic latitude in DEGREES   −90° → +90°
                lon = longitude in DEGREES             −180° → +180°  or  0° → 360°

        All conventions are converted internally to colatitude + longitude
        in radians before processing.

        Returns
        -------
        ring_theta    : ndarray [R]            colatitude θ (radians) per ring
        ring_phi_list : list[ndarray [N_r]]    longitudes  φ (radians) per ring
        ring_counts   : ndarray int64 [R]      number of pixels per ring
        sort_idx      : ndarray int64 [N]      permutation  im_sorted = im[sort_idx]
        """
        lat = np.asarray(lat, dtype=np.float64).ravel()
        lon = np.asarray(lon, dtype=np.float64).ravel()

        conv = convention.lower().strip()
        if conv == 'colatitude_rad':
            theta = lat
            phi   = lon
        elif conv == 'colatitude_deg':
            theta = np.radians(lat)
            phi   = np.radians(lon)
        elif conv == 'geographic_rad':
            theta = np.pi / 2.0 - lat
            phi   = lon % (2.0 * np.pi)
        elif conv == 'geographic_deg':
            theta = np.radians(90.0 - lat)
            phi   = np.radians(lon) % (2.0 * np.pi)
        else:
            raise ValueError(
                f"Unknown convention: '{convention}'. "
                "Accepted values: 'colatitude_rad', 'colatitude_deg', "
                "'geographic_rad', 'geographic_deg'."
            )

        N = len(theta)

        # Sort by colatitude then by longitude
        order = np.lexsort((phi, theta))
        lat_s = theta[order]
        lon_s = phi[order]

        # Find ring boundaries (colatitude jump > atol)
        breaks = np.where(np.diff(lat_s) > atol)[0] + 1
        ring_starts = np.concatenate([[0], breaks])
        ring_ends   = np.concatenate([breaks, [N]])

        ring_theta    = np.array([lat_s[s] for s in ring_starts], dtype=np.float64)
        ring_phi_list = [lon_s[s:e] for s, e in zip(ring_starts, ring_ends)]
        ring_counts   = np.array([e - s for s, e in zip(ring_starts, ring_ends)],
                                 dtype=np.int64)

        return ring_theta, ring_phi_list, ring_counts, order

    # ================================================================== #
    #  Quadrature weights                                                 #
    # ================================================================== #

    @staticmethod
    def compute_weights(ring_theta, ring_phi_list, ring_counts,
                        quadrature='trapeze'):
        """
        Computes the quadrature weights per pixel (steradians).

        Parameters
        ----------
        ring_theta    : [R] colatitudes in radians
        ring_phi_list : list[ndarray]  longitudes per ring
        ring_counts   : [R] number of pixels per ring
        quadrature    : str  quadrature method in θ.

            'trapeze'         (default)
                Trapezoidal rule:  w_θ = sin(θ_r) × Δθ_r
                Suitable for regular θ grids.

            'gauss_legendre'
                Exact Gauss-Legendre weights.
                Required for Gaussian grids
                (ERA5, ECMWF, IFS, ARPEGE…) where the colatitudes are the
                zeros of P_R(cos θ).  The integral ∫f dΩ = ∫f dφ dx
                (x = cos θ) is then exact up to ℓ ≈ 2R-1.

            'equal_area'
                Equal weights: 4π / N_total.
                For equal-area grids (HEALPix).

        Returns
        -------
        weights : ndarray float64 [N_total]
        """
        ring_theta  = np.asarray(ring_theta,  dtype=np.float64)
        ring_counts = np.asarray(ring_counts, dtype=np.int64)
        R       = len(ring_theta)
        N_total = int(ring_counts.sum())
        all_w   = []

        # ---- θ weights according to chosen method ----
        if quadrature == 'trapeze':
            w_theta = np.empty(R, dtype=np.float64)
            for r in range(R):
                if R == 1:
                    dth = np.pi
                elif r == 0:
                    dth = (ring_theta[1] - ring_theta[0]) / 2.0
                elif r == R - 1:
                    dth = (ring_theta[-1] - ring_theta[-2]) / 2.0
                else:
                    dth = (ring_theta[r + 1] - ring_theta[r - 1]) / 2.0
                w_theta[r] = abs(np.sin(ring_theta[r]) * dth)

        elif quadrature == 'gauss_legendre':
            # GL nodes are x_r = cos(θ_r) ∈ [-1, 1].
            # np.polynomial.legendre.leggauss(R) returns nodes and weights
            # for ∫₋₁¹ f(x) dx ≈ Σ w_r f(x_r).
            # Since dΩ = dφ dx (with x = cos θ), the θ weights are directly
            # the GL weights (sin θ is absorbed into dx = -sin θ dθ).
            x_provided = np.cos(ring_theta)          # provided nodes
            gl_nodes, gl_weights = np.polynomial.legendre.leggauss(R)
            # GL nodes are sorted in ascending order; cos θ is decreasing
            # (θ increasing), so we align them by sorting.
            sort_gl  = np.argsort(gl_nodes)          # -1 → +1 (ascending)
            sort_prov = np.argsort(x_provided)       # cos θ ascending
            gl_w_sorted = gl_weights[sort_gl]        # weights aligned on ascending x
            # Reorder to original ring order (θ ascending ≡ x descending)
            # sort_prov[i] = ring index of the i-th smallest cos θ
            w_theta = np.empty(R, dtype=np.float64)
            w_theta[sort_prov] = gl_w_sorted
            # Verification: GL nodes must match cos(θ_r)
            max_err = np.max(np.abs(np.sort(x_provided) - np.sort(gl_nodes)))
            if max_err > 1e-6:
                import warnings
                warnings.warn(
                    f"gauss_legendre: provided colatitudes do not match "
                    f"GL nodes (max error = {max_err:.2e}). "
                    "Check that the grid is a Gaussian grid with "
                    f"{R} latitude points.",
                    UserWarning
                )

        elif quadrature == 'equal_area':
            total_area = 4.0 * np.pi
            w_theta = np.full(R, total_area / N_total)  # will be weighted by N_r below
            # Pour equal_area, le poids par pixel est uniforme : 4π/N_total
            weights = np.full(N_total, total_area / N_total, dtype=np.float64)
            return weights

        else:
            raise ValueError(
                f"Unknown quadrature: '{quadrature}'. "
                "Accepted values: 'trapeze', 'gauss_legendre', 'equal_area'."
            )

        # ---- φ weights (common to both non-equal_area θ methods) ----
        for r in range(R):
            N_r   = int(ring_counts[r])
            phi_r = np.asarray(ring_phi_list[r], dtype=np.float64)

            if N_r == 1:
                w_phi = np.array([2.0 * np.pi])
            else:
                sorted_phi = np.sort(phi_r)
                dphi       = np.diff(sorted_phi)
                if np.ptp(dphi) < 1e-10 * (2 * np.pi / N_r):
                    w_phi = np.full(N_r, 2.0 * np.pi / N_r)
                else:
                    gap_wrap = (sorted_phi[0] + 2.0 * np.pi) - sorted_phi[-1]
                    dp_ext   = np.concatenate([[gap_wrap], dphi, [gap_wrap]])
                    w_sorted = (dp_ext[:-1] + dp_ext[1:]) / 2.0
                    back     = np.argsort(np.argsort(phi_r))
                    w_phi    = w_sorted[back]

            all_w.append(w_theta[r] * w_phi)

        return np.concatenate(all_w)
    # ================================================================== #
    #  Fourier transform per ring                                         #
    # ================================================================== #

    @staticmethod
    def _check_uniform(phi, tol=1e-10):
        """
        Return (True, phi0, N) if the φ values are uniformly spaced
        at dphi = 2π/N, otherwise (False, None, None).
        The φ values need not be sorted.
        """
        N = len(phi)
        if N <= 1:
            return True, float(phi[0]) if N == 1 else 0.0, N
        sorted_phi = np.sort(phi)
        dphi = np.diff(sorted_phi)
        mean_dp = 2.0 * np.pi / N
        if np.ptp(dphi) < tol * mean_dp:
            return True, float(sorted_phi[0]), N
        return False, None, None

    def comp_tf_latlon(self, im, ring_phi_list, ring_counts, pixel_weights, mmax):
        """
        Pixel-weighted DFT for each ring.

        For a ring with uniformly spaced φ values, uses the
        FFT + phase shift (same logic as alm.comp_tf).
        For an irregular ring, performs a direct DFT.

        Parameters
        ----------
        im            : [..., N_total]  map (backend tensor or ndarray)
        ring_phi_list : list[ndarray]   longitudes per ring
        ring_counts   : ndarray int64   number of pixels per ring
        pixel_weights : ndarray float64 [N_total]  quadrature weights
        mmax          : int  maximum frequency

        Returns
        -------
        ft : tensor [..., R, mmax+1]  complex
        """
        R         = len(ring_counts)
        m_vec     = np.arange(mmax + 1, dtype=np.float64)
        im_bk     = self.backend.bk_cast(im)

        out = []
        offset = 0

        for r in range(R):
            N_r   = int(ring_counts[r])
            phi_r = np.asarray(ring_phi_list[r], dtype=np.float64)
            w_r   = pixel_weights[offset:offset + N_r]
            v     = im_bk[..., offset:offset + N_r]   # [..., N_r]
            offset += N_r

            is_unif, phi0, _ = self._check_uniform(phi_r)

            if is_unif:
                # ---- FFT + phase shift ----
                # Sort v in ascending φ order
                sort_phi = np.argsort(phi_r)
                v_sorted = self.backend.bk_gather(v, sort_phi, axis=-1)

                # Uniform φ weights: w_r[j] = w_th * (2π/N_r)
                # Absorb the θ weight (constant per ring) as a scalar
                w_scalar = float(w_r[0])  # all φ weights equal for a uniform ring
                v_sorted  = v_sorted * w_scalar

                # Real FFT → full spectrum
                tmp = self.rfft2fft(v_sorted)   # [..., N_r]

                l_n = tmp.shape[-1]
                if l_n < mmax + 1:
                    repeat_n = (mmax // l_n) + 1
                    tmp = self.backend.bk_tile(tmp, repeat_n, axis=-1)
                tmp = tmp[..., :mmax + 1]       # [..., mmax+1]

                # Phase shift: exp(-i m phi0) for m = 0..mmax
                shift = np.exp(-1j * m_vec * phi0).astype(np.complex128)
                shift_bk = self.backend.bk_cast(shift)
                tmp = tmp * shift_bk

            else:
                # ---- Direct weighted DFT ----
                # kernel[j, m] = w_r[j] * exp(-i m phi_r[j])
                ang = np.outer(phi_r, m_vec)                   # [N_r, M]
                ker = (np.exp(-1j * ang) * w_r[:, None])       # [N_r, M]
                ker_bk = self.backend.bk_cast(ker.astype(np.complex128))

                # ft[..., m] = sum_j v[..., j] * ker[j, m]
                tmp = self.backend.bk_reduce_sum(
                    self.backend.bk_expand_dims(v, axis=-1) * ker_bk,
                    axis=-2
                )  # [..., mmax+1]

            out.append(self.backend.bk_expand_dims(tmp, axis=-2))  # [..., 1, mmax+1]

        return self.backend.bk_concat(out, axis=-2)   # [..., R, mmax+1]

    # ================================================================== #
    #  map → alm                                                          #
    # ================================================================== #

    def map2alm_latlon(self, im, ring_theta, ring_phi_list, ring_counts,
                       lmax=None, weights=None, quadrature='trapeze'):
        """
        Compute the alm coefficients of a map defined on an arbitrary
        grid organised into rings.

        Parameters
        ----------
        im            : [..., N_total]  map values, ordered ring by ring
                        (use sort_idx from build_rings_from_latlon
                        if the map is initially in arbitrary order)
        ring_theta    : [R] colatitude (radians) of each ring
        ring_phi_list : list[ndarray] or ndarray [N_total]
                        longitudes (radians) per ring (or flat array)
        ring_counts   : [R] number of pixels per ring
        lmax          : maximum multipole (default: self.lmax)
        weights       : [N_total] per-pixel weights (steradians).
                        If None, trapezoidal quadrature is computed
                        automatically.  Pass weights='uniform' to
                        use 1/N_total (alm.map2alm convention).

        Returns
        -------
        alm_out : [..., n_alm]  complex,  n_alm = Σ_{m=0}^{lmax} (lmax-m+1)
                  Layout: [m=0: l=0..lmax | m=1: l=1..lmax | …]
        """
        if lmax is None:
            lmax = self.lmax
        lmax = int(lmax)

        ring_theta  = np.asarray(ring_theta, dtype=np.float64)
        ring_counts = np.asarray(ring_counts, dtype=np.int64)
        N_total     = int(ring_counts.sum())
        phi_list    = self._parse_phi_list(ring_phi_list, ring_counts)

        # Handle batch dimension
        _added_batch = False
        if hasattr(im, 'ndim') and im.ndim == 1:
            im = im[None, :]
            _added_batch = True
        elif not hasattr(im, 'ndim') and len(im.shape) == 1:
            im = im[None, :]
            _added_batch = True

        # Quadrature weights
        if weights is None:
            pixel_weights = self.compute_weights(ring_theta, phi_list, ring_counts,
                                                 quadrature=quadrature)
        elif isinstance(weights, str) and weights == 'uniform':
            pixel_weights = np.ones(N_total, dtype=np.float64) / N_total
        else:
            pixel_weights = np.asarray(weights, dtype=np.float64)

        # DFT per ring: ft[..., R, mmax+1]
        ft = self.comp_tf_latlon(im, phi_list, ring_counts, pixel_weights, mmax=lmax)

        # cos(θ) per ring for the Legendre recurrence
        co_th = np.cos(ring_theta)          # [R]

        # Legendre projection
        alm_out = None
        for m in range(lmax + 1):
            # compute_legendre_m returns sqrt(4π) · P_lm^norm(cos θ), shape [L, R].
            # Normalised spherical harmonics are Y_lm = sqrt((2l+1)/4π) · P_lm^norm.
            # The missing factor is sqrt(2l+1)/(4π); applied here so that
            # the projection gives a_lm = ∫ f Y_lm* dΩ (healpy/standard convention).
            plm    = self.compute_legendre_m(co_th, m, lmax, nside=1)   # [L, R]
            l_vals  = np.arange(m, lmax + 1, dtype=np.float64)           # [L]
            ylm_factor = np.sqrt(2.0 * l_vals + 1.0) / (4.0 * np.pi)    # [L]
            plm    = plm * ylm_factor[:, np.newaxis]                      # [L, R]
            plm_bk = self.backend.bk_cast(plm)   # [L, R]

            ft_m = ft[..., :, m]   # [..., R]

            # alm[..., l-m] = sum_r plm[l-m, r] * ft[..., r, m]
            # ft_m  [..., 1, R]  ×  plm [L, R]  →  sum_R  →  [..., L]
            tmp = self.backend.bk_reduce_sum(
                self.backend.bk_expand_dims(ft_m, axis=-2) * plm_bk,
                axis=-1
            )  # [..., L]

            if m == 0:
                alm_out = tmp
            else:
                alm_out = self.backend.bk_concat([alm_out, tmp], axis=-1)

        if _added_batch:
            alm_out = alm_out[0]

        return alm_out

    # ================================================================== #
    #  alm → map (synthesis)                                              #
    # ================================================================== #

    def alm2map_latlon(self, alm, ring_theta, ring_phi_list, ring_counts, lmax=None):
        """
        Synthesis: reconstruct the map from alm coefficients.

        Parameters
        ----------
        alm           : [..., n_alm]  harmonic coefficients
        ring_theta    : [R] colatitudes (radians)
        ring_phi_list : list[ndarray] or ndarray [N_total] longitudes
        ring_counts   : [R] number of pixels per ring
        lmax          : maximum multipole used when computing the alm

        Returns
        -------
        im_out : [..., N_total]  reconstructed map
        """
        if lmax is None:
            lmax = self.lmax
        lmax = int(lmax)

        ring_theta  = np.asarray(ring_theta, dtype=np.float64)
        ring_counts = np.asarray(ring_counts, dtype=np.int64)
        phi_list    = self._parse_phi_list(ring_phi_list, ring_counts)
        N_total     = int(ring_counts.sum())
        R           = len(ring_theta)

        co_th = np.cos(ring_theta)  # [R]

        # Build the Fourier part per ring: ft[r, m] = sum_l alm[l,m] * plm[l-m, r]
        # then reconstruct the map via inverse DFT

        _added_batch = False
        if hasattr(alm, 'ndim') and alm.ndim == 1:
            alm = alm[None, :]
            _added_batch = True

        # ft_synth[..., R, mmax+1]
        batch_shape = alm.shape[:-1]
        ft_synth = torch.zeros(
            batch_shape + (R, lmax + 1),
            dtype=torch.complex128,
            device=alm.device if hasattr(alm, 'device') else torch.device('cpu')
        )

        idx = 0
        for m in range(lmax + 1):
            L      = lmax - m + 1
            alm_m  = alm[..., idx:idx + L]       # [..., L]
            idx   += L

            plm    = self.compute_legendre_m(co_th, m, lmax, nside=1)  # [L, R]
            plm_bk = self.backend.bk_cast(plm)

            # ft[..., r, m] = sum_l alm_m[..., l-m] * plm[l-m, r]
            # alm_m [..., L, 1] × plm [L, R] → sum_L → [..., R]
            contrib = self.backend.bk_reduce_sum(
                self.backend.bk_expand_dims(alm_m, axis=-1) * plm_bk,
                axis=-2
            )   # [..., R]

            ft_synth[..., :, m] = contrib

        # Inverse DFT per ring: im[r, j] = Re( sum_m ft[r,m] * exp(i m phi_j) )
        out_per_ring = []
        for r in range(R):
            N_r   = int(ring_counts[r])
            phi_r = np.asarray(phi_list[r], dtype=np.float64)
            m_vec = np.arange(lmax + 1, dtype=np.float64)

            # kernel[m, j] = exp(i m phi_j)  [mmax+1, N_r]
            ang = np.outer(m_vec, phi_r)
            ker = np.exp(1j * ang).astype(np.complex128)   # [M, N_r]
            ker_bk = self.backend.bk_cast(ker)

            ft_r = ft_synth[..., r, :]   # [..., M]

            # im[r, j] = Re( sum_m ft[r,m] * exp(i m phi_j) )
            # ft_r [..., M, 1] × ker [M, N_r] → sum_M → [..., N_r]
            pix = self.backend.bk_reduce_sum(
                self.backend.bk_expand_dims(ft_r, axis=-1) * ker_bk,
                axis=-2
            )   # [..., N_r]

            out_per_ring.append(self.backend.bk_real(pix))

        im_out = self.backend.bk_concat(out_per_ring, axis=-1)   # [..., N_total]

        if _added_batch:
            im_out = im_out[0]

        return im_out

    # ================================================================== #
    #  map → Cl                                                           #
    # ================================================================== #

    def anafast_latlon(self, im, ring_theta, ring_phi_list, ring_counts,
                       lmax=None, weights=None, quadrature='trapeze'):
        """
        Estimate the power spectrum Cl of a map on an arbitrary grid.

        Returns
        -------
        cl : tensor [..., lmax+1]
        """
        if lmax is None:
            lmax = self.lmax
        lmax = int(lmax)

        alm = self.map2alm_latlon(
            im, ring_theta, ring_phi_list, ring_counts,
            lmax=lmax, weights=weights, quadrature=quadrature
        )

        batch_shape = alm.shape[:-1] if alm.ndim > 1 else ()
        device = alm.device if hasattr(alm, 'device') else torch.device('cpu')

        cl = torch.zeros(batch_shape + (lmax + 1,),
                         dtype=torch.float64, device=device)

        idx = 0
        for m in range(lmax + 1):
            L = lmax - m + 1
            a = alm[..., idx:idx + L]
            idx += L

            p = self.backend.bk_real(a * self.backend.bk_conjugate(a))

            weight = 1.0 if m == 0 else 2.0
            cl[..., m:m + L] += weight * p

        # Normalisation par (2l+1)
        denom = (2.0 * torch.arange(lmax + 1, dtype=torch.float64, device=device) + 1.0)
        denom = denom.reshape((1,) * len(batch_shape) + (lmax + 1,))
        cl = cl / denom

        return cl

    # ================================================================== #
    #  Utilitaires internes                                               #
    # ================================================================== #

    @staticmethod
    def _parse_phi_list(ring_phi_list, ring_counts):
        """
        Accepte ring_phi_list comme :
          - liste de tableaux
          - flat array [N_total] (split according to ring_counts)
        Retourne toujours une liste de tableaux float64.
        """
        ring_counts = np.asarray(ring_counts, dtype=np.int64)
        if isinstance(ring_phi_list, np.ndarray) and ring_phi_list.ndim == 1:
            splits = np.cumsum(ring_counts)[:-1]
            return [a.astype(np.float64)
                    for a in np.split(ring_phi_list, splits)]
        return [np.asarray(p, dtype=np.float64) for p in ring_phi_list]

    def grid_summary(self, ring_theta, ring_phi_list, ring_counts, lmax=None):
        """
        Print a summary of the grid and the estimated computation cost.
        """
        if lmax is None:
            lmax = self.lmax

        ring_counts = np.asarray(ring_counts, dtype=np.int64)
        N_total = int(ring_counts.sum())
        R       = len(ring_theta)

        n_unif = sum(
            1 for r in range(R)
            if self._check_uniform(np.asarray(ring_phi_list[r]))[0]
        )

        cost_fft  = sum(
            (mmax_r + 1) * np.log2(max(2, int(ring_counts[r])))
            for r, mmax_r in enumerate([lmax] * R)
            if self._check_uniform(np.asarray(ring_phi_list[r]))[0]
        )
        cost_dft  = sum(
            int(ring_counts[r]) * (lmax + 1)
            for r in range(R)
            if not self._check_uniform(np.asarray(ring_phi_list[r]))[0]
        )

        print(f"=== Grid summary  (lmax={lmax}) ===")
        print(f"  Total pixels  : {N_total}")
        print(f"  Rings         : {R}")
        print(f"  Uniform rings : {n_unif}/{R}  (FFT acceleration)")
        print(f"  θ range       : [{np.degrees(ring_theta.min()):.2f}°, "
              f"{np.degrees(ring_theta.max()):.2f}°]")
        print(f"  N_pix/ring    : min={ring_counts.min()}, "
              f"max={ring_counts.max()}, mean={ring_counts.mean():.1f}")
        print(f"  n_alm         : {sum(lmax - m + 1 for m in range(lmax + 1))}")
