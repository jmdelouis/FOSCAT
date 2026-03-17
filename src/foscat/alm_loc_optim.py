"""
alm_loc_optim.py
================
Variante optimisée de alm_loc pour le calcul des harmoniques sphériques
sur un domaine HEALPix restreint.

Deux optimisations clés par rapport à alm_loc
----------------------------------------------
1.  Restriction en m  (axe longitude)
    Pour chaque ring r avec cnt pixels sur N_ring totaux :
      - ring plein   : mode m valide si  (m % N_ring) <= N_ring // 2
      - ring partiel : mode m valide si  (m % N_ring) <= cnt // 2   (Nyquist partiel)
    Pour chaque m, seuls les rings satisfaisant ce critère alimentent
    la projection de Legendre.

2.  Restriction en l  (axe latitude)
    Avec R_m rings valides pour le mode m, le système de Legendre dispose
    de R_m équations.  On ne peut contraindre au plus que R_m degrés ell.
    On pose donc :
        lmax_eff(m) = min(lmax,  m + R_m - 1)
    et on ne calcule les polynômes de Legendre que jusqu'à ce plafond.

Économie résultante
-------------------
La projection de Legendre passe de O(R × (lmax - m + 1))
à O(R_m × (lmax_eff(m) - m + 1)), ce qui est quadratiquement plus
favorable pour les petits domaines.

API publique (miroir de alm_loc)
---------------------------------
analyze_domain(nside, cell_ids, nest, lmax)
    → rings_used, counts, sizes, valid_idx_per_m, lmax_eff_per_m, m_valid

map2alm_loc_optim(im, nside, cell_ids, nest, lmax)
    → alm_per_m, m_valid, lmax_eff_per_m   (alm creux)

anafast_loc_optim(im, nside, cell_ids, nest, lmax)
    → cl [lmax+1], m_count [lmax+1]

sparse_to_dense(alm_per_m, m_valid, lmax_eff_per_m, lmax)
    → vecteur alm dense (zéros pour les modes non calculés),
      compatible avec la mise en page de alm_loc.map2alm_loc
"""

import numpy as np
import torch

from alm_loc import alm_loc


class alm_loc_optim(alm_loc):

    def __init__(self, backend=None, lmax=24, limit_range=1e10):
        super().__init__(backend=backend, lmax=lmax, limit_range=limit_range)

    # ================================================================== #
    #  Analyse du domaine : quels (l, m) sont contraints par le patch ?   #
    # ================================================================== #

    def analyze_domain(self, nside: int, cell_ids, nest: bool = False,
                       lmax: int = None):
        """
        Pré-calcule l'ensemble des modes (l, m) qui sont effectivement
        contraints par le patch partiel décrit par cell_ids.

        Paramètres
        ----------
        nside    : résolution HEALPix
        cell_ids : indices de pixels (ring ou nested)
        nest     : True si cell_ids utilise l'ordre NESTED
        lmax     : multipôle maximal (défaut : min(self.lmax, 3*nside-1))

        Retourne
        --------
        rings_used       : ndarray int32 [R]    indices des rings présents
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

        # N_ring pour chaque ring utilisé
        Nrings = sizes[rings_used].astype(np.int64)   # [R]
        nyquist = (counts // 2).astype(np.int64)       # [R]  Nyquist partiel
        is_full = counts == Nrings                     # [R]  bool

        valid_idx_per_m: dict = {}
        lmax_eff_per_m: dict = {}
        m_valid: list = []

        for m in range(lmax + 1):
            m_mod = (m % Nrings).astype(np.int64)     # fréquence aliasée [R]

            # Ring plein  : critère Nyquist FFT standard (m_mod <= N_ring/2)
            # Ring partiel: critère Nyquist réduit      (m_mod <= cnt/2)
            valid_mask = (is_full & (m_mod <= Nrings // 2)) | \
                         (~is_full & (m_mod <= nyquist))

            valid = np.where(valid_mask)[0].astype(np.int32)

            if valid.size > 0:
                valid_idx_per_m[m] = valid
                # Avec R_m rings on contraint au plus R_m degrés ell
                lmax_eff_per_m[m] = int(min(lmax, m + valid.size - 1))
                m_valid.append(m)

        return rings_used, counts, sizes, valid_idx_per_m, lmax_eff_per_m, m_valid

    # ================================================================== #
    #  map -> alm creux (optimisé)                                        #
    # ================================================================== #

    def map2alm_loc_optim(self, im, nside: int, cell_ids,
                          nest: bool = False, lmax: int = None):
        """
        Calcule les coefficients harmoniques sphériques sur un patch partiel,
        en se limitant aux modes (l, m) effectivement contraints par la
        couverture partielle du ciel.

        Paramètres
        ----------
        im       : [..., n_pixels]  valeurs de la carte sur le patch
        nside    : résolution HEALPix
        cell_ids : indices de pixels dans le patch
        nest     : True si ordre NESTED
        lmax     : multipôle maximal

        Retourne
        --------
        alm_per_m      : liste de tenseurs, un par m valide.
                         alm_per_m[i] a la forme [..., lmax_eff(m)-m+1]
                         pour m = m_valid[i]
        m_valid        : list[int]   valeurs de m dans le même ordre
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

        # Transformée de Fourier par ring — identique à comp_tf_loc
        # ft : [..., R, lmax+1]
        _, ft = self.comp_tf_loc(im, nside=nside, cell_ids=cell_ids,
                                 nest=nest, realfft=True, mmax=lmax)

        # cos(theta) pour tous les rings utilisés
        co_th_all = np.cos(self.ring_th(nside)[rings_used])   # [R]

        alm_per_m = []

        for m in m_valid:
            vidx    = valid_idx_per_m[m]     # indices dans rings_used, [R_m]
            lmax_m  = lmax_eff_per_m[m]      # lmax effectif pour ce m
            n_l     = lmax_m - m + 1         # nb de degrés ell calculés

            co_th_m = co_th_all[vidx]        # [R_m]

            # Polynômes de Legendre P_{lm}(cos θ) pour l = m..lmax_m
            # Forme : [n_l, R_m]
            # On passe lmax_m au lieu de lmax → économie sur la récurrence
            plm = self.compute_legendre_m(co_th_m, m, lmax_m, nside) \
                  / (12 * nside**2)
            plm_bk = self.backend.bk_cast(plm)   # [n_l, R_m]

            # Coefficients de Fourier au mode m pour les rings valides
            # ft[..., vidx, m] : [..., R_m]
            ft_m = ft[..., vidx, m]

            # Projection : [..., n_l] = sum_{R_m} ft_m * P_{lm}
            # ft_m étendu : [..., 1, R_m]  ×  plm [n_l, R_m]  → somme sur R_m
            tmp = self.backend.bk_reduce_sum(
                self.backend.bk_expand_dims(ft_m, axis=-2) * plm_bk,
                axis=-1
            )   # [..., n_l]

            # Pondération sqrt(2l+1) (cohérente avec map2alm_loc)
            l_vals = np.arange(m, lmax_m + 1, dtype=np.float64)
            scale  = self.backend.bk_cast(np.sqrt(2.0 * l_vals + 1.0))
            scale  = scale.reshape((1,) * (tmp.ndim - 1) + (n_l,))
            tmp    = tmp * scale

            if _added_batch:
                tmp = tmp[0]

            alm_per_m.append(tmp)

        return alm_per_m, m_valid, lmax_eff_per_m

    # ================================================================== #
    #  Conversion creux -> dense (compatibilité avec alm_loc)             #
    # ================================================================== #

    def sparse_to_dense(self, alm_per_m, m_valid, lmax_eff_per_m, lmax: int):
        """
        Convertit l'alm creux (sortie de map2alm_loc_optim) vers le vecteur
        dense plat utilisé par alm_loc.map2alm_loc.

        Format dense : [m=0: l=0..lmax | m=1: l=1..lmax | …]
        Les modes non calculés sont remplis de zéros.

        Paramètres
        ----------
        alm_per_m      : liste de tenseurs (sortie de map2alm_loc_optim)
        m_valid        : list[int]
        lmax_eff_per_m : dict m -> int
        lmax           : multipôle maximal utilisé lors du calcul

        Retourne
        --------
        out : tenseur [..., total_alm]  dtype et device identiques à l'entrée
        """
        if not alm_per_m:
            raise ValueError("alm_per_m est vide.")

        sample       = alm_per_m[0]
        batch_shape  = sample.shape[:-1] if sample.ndim > 1 else ()
        device       = sample.device
        dtype        = sample.dtype
        total        = sum(lmax - m + 1 for m in range(lmax + 1))

        out = torch.zeros(batch_shape + (total,), dtype=dtype, device=device)

        # Décalage de chaque m dans le vecteur dense
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
    #  alm -> Cl  (optimisé)                                              #
    # ================================================================== #

    def anafast_loc_optim(self, im, nside: int, cell_ids,
                          nest: bool = False, lmax: int = None):
        """
        Estime le spectre de puissance angulaire Cl sur un patch partiel.

        Seuls les modes (l, m) effectivement contraints par le patch
        contribuent à l'estimation.  Pour chaque l, Cl est normalisé par
        le nombre de modes m disponibles (dégradation gracieuse plutôt que
        dilution par des modes nuls).

        Paramètres
        ----------
        im       : [..., n_pixels]
        nside    : résolution HEALPix
        cell_ids : indices de pixels dans le patch
        nest     : True si ordre NESTED
        lmax     : multipôle maximal

        Retourne
        --------
        cl      : tenseur [..., lmax+1]
        m_count : ndarray int32 [lmax+1]
                  nombre de modes m contribuant à chaque l
                  (permet de signaler les multipôles mal contraints :
                   m_count[l] == 0  →  Cl[l] non défini)
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

        # Normalisation : on divise par le nombre de modes contribuant à chaque l
        # (et non par (2l+1) global, qui supposerait le ciel complet)
        norm = np.where(m_count > 0,
                        m_count.astype(np.float64),
                        1.0)   # évite la division par zéro
        norm_t = torch.tensor(norm, dtype=torch.float64, device=device)
        norm_t = norm_t.reshape((1,) * len(batch_shape) + (lmax + 1,))
        cl = cl / norm_t

        return cl, m_count

    # ================================================================== #
    #  Utilitaire : résumé du domaine (diagnostic)                        #
    # ================================================================== #

    def domain_summary(self, nside: int, cell_ids, nest: bool = False,
                       lmax: int = None):
        """
        Affiche un résumé lisible du domaine partiel :
        fraction de ciel couverte, nb de rings, nb de modes (m, l) effectifs
        vs brut, et gain de calcul estimé.
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

        # Coût brut (alm_loc original) : sum_m  R × (lmax - m + 1)
        cost_full = sum(n_rings_patch * (lmax - m + 1) for m in range(lmax + 1))

        # Coût optimisé : sum_m  R_m × (lmax_eff(m) - m + 1)
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
