"""
alm_latlon.py
=============
Transformée en harmoniques sphériques pour des cartes définies sur une
grille arbitraire en colatitude / longitude, organisée en rings.

Différences par rapport à foscat.alm.alm
-----------------------------------------
- Aucune dépendance à HEALPix pour le positionnement des pixels.
- Les rings peuvent avoir des colatitudes quelconques (pas les colatitudes
  HEALPix) et des longitudes quelconques (pas nécessairement uniformes).
- Étape longitude : DFT directe si les φ d'un ring sont irréguliers ;
  FFT + déphasage (comme alm.comp_tf) si les φ sont uniformément espacés.
- Étape colatitude : même récurrence de Legendre que alm.compute_legendre_m,
  évaluée aux cosinus des colatitudes fournies.
- Poids de quadrature : trapèze en θ (sin θ · Δθ) × uniforme en φ (2π/N_r)
  par défaut, ou tableau de poids fourni par l'utilisateur.

API principale
--------------
build_rings_from_latlon(lat, lon, atol)
    Classe une liste plate de (lat, lon) en rings de même colatitude.

compute_weights(ring_theta, ring_phi_list, ring_counts)
    Calcule les poids de quadrature par pixel (stéradians).

comp_tf_latlon(im, ring_phi_list, ring_counts, pixel_weights, mmax)
    DFT pondérée par ring → ft[..., R, mmax+1].

map2alm_latlon(im, ring_theta, ring_phi_list, ring_counts, lmax, weights)
    Carte → alm.

anafast_latlon(im, ring_theta, ring_phi_list, ring_counts, lmax, weights)
    Carte → Cl.

alm2map_latlon(alm, ring_theta, ring_phi_list, ring_counts, lmax)
    alm → carte (synthèse).

Exemple minimal
---------------
    import numpy as np
    from foscat.alm_latlon import alm_latlon

    # Grille régulière (ntheta=64 rings × nphi=128 pixels par ring)
    ntheta, nphi = 64, 128
    theta_1d = np.linspace(np.pi / (2*ntheta), np.pi - np.pi/(2*ntheta), ntheta)
    phi_1d   = np.linspace(0, 2*np.pi*(1 - 1/nphi), nphi)

    lat  = np.repeat(theta_1d, nphi)        # colatitude de chaque pixel
    lon  = np.tile(phi_1d,     ntheta)      # longitude de chaque pixel

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
    Transformée en harmoniques sphériques sur une grille lat/lon arbitraire
    organisée en rings.
    """

    def __init__(self, backend=None, lmax=24, limit_range=1e10):
        # nside=None : pas de grille HEALPix, maxlog calculé depuis lmax
        super().__init__(backend=backend, lmax=lmax, nside=None,
                         limit_range=limit_range)

    # ================================================================== #
    #  Construction des rings depuis un tableau plat (lat, lon)           #
    # ================================================================== #

    @staticmethod
    def build_rings_from_latlon(lat, lon, atol=1e-10):
        """
        Groupe une liste plate de pixels en rings de même colatitude.

        Paramètres
        ----------
        lat  : array [N]  colatitude θ de chaque pixel (radians, 0..π).
               (si la valeur est en degrés ou en latitude géographique,
               convertir avant : θ = π/2 − lat_geo  ou  θ = lat_rad)
        lon  : array [N]  longitude φ de chaque pixel (radians, 0..2π).
        atol : tolérance en radians pour regrouper deux pixels dans le
               même ring (défaut 1e-10).

        Retourne
        --------
        ring_theta  : ndarray [R]          colatitude de chaque ring (triée)
        ring_phi_list : list[ndarray [N_r]]  longitudes par ring
        ring_counts : ndarray int64 [R]    nombre de pixels par ring
        sort_idx    : ndarray int64 [N]    permutation pour réordonner im :
                        im_sorted = im[sort_idx]
        """
        lat = np.asarray(lat, dtype=np.float64).ravel()
        lon = np.asarray(lon, dtype=np.float64).ravel()
        N = len(lat)

        # Trier par colatitude puis par longitude
        order = np.lexsort((lon, lat))
        lat_s = lat[order]
        lon_s = lon[order]

        # Trouver les frontières de ring (saut de colatitude > atol)
        breaks = np.where(np.diff(lat_s) > atol)[0] + 1
        ring_starts = np.concatenate([[0], breaks])
        ring_ends   = np.concatenate([breaks, [N]])

        ring_theta    = np.array([lat_s[s] for s in ring_starts], dtype=np.float64)
        ring_phi_list = [lon_s[s:e] for s, e in zip(ring_starts, ring_ends)]
        ring_counts   = np.array([e - s for s, e in zip(ring_starts, ring_ends)],
                                 dtype=np.int64)

        return ring_theta, ring_phi_list, ring_counts, order

    # ================================================================== #
    #  Poids de quadrature                                                #
    # ================================================================== #

    @staticmethod
    def compute_weights(ring_theta, ring_phi_list, ring_counts):
        """
        Calcule les poids de quadrature par pixel (stéradians).

        Règle :
        - En θ : règle des trapèzes  →  Δθ_r = |θ_{r+1} − θ_{r-1}| / 2
          (moitié d'intervalle aux bords)  ×  sin(θ_r)
        - En φ : uniforme  →  2π / N_r  si ring uniformément espacé,
                 sinon trapèze enroulé sur les φ triés.

        Retourne
        --------
        weights : ndarray float64 [N_total]  poids dans l'ordre ring-par-ring
        """
        ring_theta  = np.asarray(ring_theta, dtype=np.float64)
        ring_counts = np.asarray(ring_counts, dtype=np.int64)
        R = len(ring_theta)
        all_w = []

        for r in range(R):
            # ---- poids en θ ----
            if R == 1:
                dth = np.pi
            elif r == 0:
                dth = (ring_theta[1] - ring_theta[0]) / 2.0
            elif r == R - 1:
                dth = (ring_theta[-1] - ring_theta[-2]) / 2.0
            else:
                dth = (ring_theta[r + 1] - ring_theta[r - 1]) / 2.0
            w_th = abs(np.sin(ring_theta[r]) * dth)

            # ---- poids en φ ----
            N_r   = int(ring_counts[r])
            phi_r = np.asarray(ring_phi_list[r], dtype=np.float64)

            if N_r == 1:
                w_phi = np.array([2.0 * np.pi])
            else:
                sorted_phi = np.sort(phi_r)
                dphi = np.diff(sorted_phi)
                # Vérifier uniformité (tolérance relative)
                if np.ptp(dphi) < 1e-10 * (2 * np.pi / N_r):
                    w_phi = np.full(N_r, 2.0 * np.pi / N_r)
                else:
                    # Trapèze enroulé (wrap-around)
                    gap_wrap = (sorted_phi[0] + 2.0 * np.pi) - sorted_phi[-1]
                    dp_ext   = np.concatenate([[gap_wrap], dphi, [gap_wrap]])
                    w_sorted = (dp_ext[:-1] + dp_ext[1:]) / 2.0
                    # Réordonner dans l'ordre original des φ
                    back = np.argsort(np.argsort(phi_r))
                    w_phi = w_sorted[back]

            all_w.append(w_th * w_phi)

        return np.concatenate(all_w)

    # ================================================================== #
    #  Transformée de Fourier par ring                                    #
    # ================================================================== #

    @staticmethod
    def _check_uniform(phi, tol=1e-10):
        """
        Retourne (True, phi0, N) si les φ sont uniformément espacés
        à dphi = 2π/N, sinon (False, None, None).
        Les φ peuvent ne pas être triés.
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
        DFT pondérée par pixel pour chaque ring.

        Pour un ring avec des φ uniformément espacés, on utilise
        la FFT + déphasage (même logique que alm.comp_tf).
        Pour un ring irrégulier, on fait la DFT directe.

        Paramètres
        ----------
        im            : [..., N_total]  carte (backend tensor ou ndarray)
        ring_phi_list : list[ndarray]   longitudes par ring
        ring_counts   : ndarray int64   nombre de pixels par ring
        pixel_weights : ndarray float64 [N_total]  poids quadrature
        mmax          : int  fréquence maximale

        Retourne
        --------
        ft : tensor [..., R, mmax+1]  complexe
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
                # ---- FFT + déphasage ----
                # Trier v dans l'ordre croissant de φ
                sort_phi = np.argsort(phi_r)
                v_sorted = self.backend.bk_gather(v, sort_phi, axis=-1)

                # Poids en φ uniformes : w_r[j] = w_th * (2π/N_r)
                # On absorbe le poids θ (constant par ring) comme scalaire
                w_scalar = float(w_r[0])  # tous les poids φ sont égaux pour un ring uniforme
                v_sorted  = v_sorted * w_scalar

                # FFT réelle → spectre complet
                tmp = self.rfft2fft(v_sorted)   # [..., N_r]

                l_n = tmp.shape[-1]
                if l_n < mmax + 1:
                    repeat_n = (mmax // l_n) + 1
                    tmp = self.backend.bk_tile(tmp, repeat_n, axis=-1)
                tmp = tmp[..., :mmax + 1]       # [..., mmax+1]

                # Déphasage : exp(-i m phi0) pour m = 0..mmax
                shift = np.exp(-1j * m_vec * phi0).astype(np.complex128)
                shift_bk = self.backend.bk_cast(shift)
                tmp = tmp * shift_bk

            else:
                # ---- DFT directe pondérée ----
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
                       lmax=None, weights=None):
        """
        Calcule les coefficients alm d'une carte définie sur une grille
        arbitraire organisée en rings.

        Paramètres
        ----------
        im            : [..., N_total]  valeurs de la carte, ordonnées ring
                        par ring (utiliser sort_idx de build_rings_from_latlon
                        si la carte est initialement dans un ordre quelconque)
        ring_theta    : [R] colatitude (radians) de chaque ring
        ring_phi_list : list[ndarray] ou ndarray [N_total]
                        longitudes (radians) par ring (ou tableau plat)
        ring_counts   : [R] nombre de pixels par ring
        lmax          : multipôle maximal (défaut : self.lmax)
        weights       : [N_total] poids par pixel (stéradians).
                        Si None, la quadrature trapèze est calculée
                        automatiquement.  Passer weights='uniform' pour
                        utiliser 1/N_total (convention alm.map2alm).

        Retourne
        --------
        alm_out : [..., n_alm]  complexe,  n_alm = Σ_{m=0}^{lmax} (lmax-m+1)
                  Mise en page : [m=0: l=0..lmax | m=1: l=1..lmax | …]
        """
        if lmax is None:
            lmax = self.lmax
        lmax = int(lmax)

        ring_theta  = np.asarray(ring_theta, dtype=np.float64)
        ring_counts = np.asarray(ring_counts, dtype=np.int64)
        N_total     = int(ring_counts.sum())
        phi_list    = self._parse_phi_list(ring_phi_list, ring_counts)

        # Gestion de la dimension batch
        _added_batch = False
        if hasattr(im, 'ndim') and im.ndim == 1:
            im = im[None, :]
            _added_batch = True
        elif not hasattr(im, 'ndim') and len(im.shape) == 1:
            im = im[None, :]
            _added_batch = True

        # Poids de quadrature
        if weights is None:
            pixel_weights = self.compute_weights(ring_theta, phi_list, ring_counts)
        elif isinstance(weights, str) and weights == 'uniform':
            pixel_weights = np.ones(N_total, dtype=np.float64) / N_total
        else:
            pixel_weights = np.asarray(weights, dtype=np.float64)

        # DFT par ring : ft[..., R, mmax+1]
        ft = self.comp_tf_latlon(im, phi_list, ring_counts, pixel_weights, mmax=lmax)

        # cos(θ) par ring pour la récurrence de Legendre
        co_th = np.cos(ring_theta)          # [R]

        # Projection de Legendre
        alm_out = None
        for m in range(lmax + 1):
            # plm : [lmax-m+1, R]  (nside inutilisé dans compute_legendre_m)
            plm    = self.compute_legendre_m(co_th, m, lmax, nside=1)
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
    #  alm → map (synthèse)                                               #
    # ================================================================== #

    def alm2map_latlon(self, alm, ring_theta, ring_phi_list, ring_counts, lmax=None):
        """
        Synthèse : reconstruit la carte depuis les alm.

        Paramètres
        ----------
        alm           : [..., n_alm]  coefficients harmoniques
        ring_theta    : [R] colatitudes (radians)
        ring_phi_list : list[ndarray] ou ndarray [N_total] longitudes
        ring_counts   : [R] nombre de pixels par ring
        lmax          : multipôle maximal utilisé lors du calcul des alm

        Retourne
        --------
        im_out : [..., N_total]  carte reconstruite
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

        # Construire la partie Fourier par ring : ft[r, m] = sum_l alm[l,m] * plm[l-m, r]
        # puis reconstruire la carte par DFT inverse

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

        # DFT inverse par ring : im[r, j] = Re( sum_m ft[r,m] * exp(im phi_j) )
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

            # im[r, j] = Re( sum_m ft[r,m] * exp(im phi_j) )
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
                       lmax=None, weights=None):
        """
        Estime le spectre de puissance Cl d'une carte sur grille arbitraire.

        Retourne
        --------
        cl : tensor [..., lmax+1]
        """
        if lmax is None:
            lmax = self.lmax
        lmax = int(lmax)

        alm = self.map2alm_latlon(
            im, ring_theta, ring_phi_list, ring_counts,
            lmax=lmax, weights=weights
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
          - tableau plat [N_total] (découpé selon ring_counts)
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
        Affiche un résumé de la grille et le coût de calcul estimé.
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
