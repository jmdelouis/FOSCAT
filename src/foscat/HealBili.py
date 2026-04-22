"""
HealBili: Bilinear weights from a curvilinear (theta, phi) source grid to arbitrary HEALPix targets.

This module provides a class `HealBili` that, given a *curvilinear* source grid of angular
coordinates (theta[y,x], phi[y,x]) on the sphere (e.g., a tangent-plane grid built on an ellipsoid),
computes **bilinear interpolation weights** to map values from that grid onto arbitrary target
directions specified by HEALPix angles (heal_theta[n], heal_phi[n]).

Key idea
--------
Because the source grid is not rectilinear in index-space, we cannot assume a simple affine mapping
from (i,j) to angles. Instead, for each target direction (theta_h, phi_h), we:
  1) locate a nearby source cell (seed) by nearest neighbor search on the unit sphere;
  2) consider up to 4 candidate quads around the seed: [(i0,j0),(i0+1,j0),(i0,j0+1),(i0+1,j0+1)];
  3) project the 4 corner unit vectors and the target onto a **local tangent plane** built at the
     quad barycenter;
  4) *invert* the bilinear mapping f(s,t) from the quad corners to the plane point using Newton,
     retrieving (s,t) in [0,1]^2;
  5) build the 4 bilinear weights [(1-s)(1-t), s(1-t), (1-s)t, st] and the 4 linear indices
     into the source image (row-major, j*W + i).

If no candidate quad cleanly contains the point, we choose the one with the smallest residual in the
plane and clamp (s,t) to [0,1].

The code is NumPy-only by default, but can optionally use `scipy.spatial.cKDTree` for a faster nearest
neighbor seed search by setting `prefer_kdtree=True` (falls back automatically if SciPy is absent).

Usage
-----
>>> hb = HealBili(src_theta, src_phi, prefer_kdtree=True)
>>> I, W = hb.compute_weights(heal_theta, heal_phi)
>>> # Apply to a source image `img` of shape (H,W):
>>> vals = hb.apply_weights(img, I, W)  # shape (N,)

All angles must be in **radians**. theta is colatitude (0 at north pole), phi is longitude in [0, 2*pi).
"""
from __future__ import annotations

from typing import Tuple
import healpy as hp
import numpy as np

try:
    from scipy.spatial import cKDTree  # optional
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    cKDTree = None
    _HAVE_SCIPY = False


class HealBili:
    """Compute bilinear interpolation weights from a curvilinear (theta, phi) grid to HEALPix targets.

    Parameters
    ----------
    src_theta : np.ndarray, shape (H, W)
        Source **colatitude** (radians) at each grid node.
    src_phi : np.ndarray, shape (H, W)
        Source **longitude** (radians) at each grid node.
    prefer_kdtree : bool, default False
        If True and SciPy is available, use cKDTree on unit vectors for a faster nearest-neighbor seed.
        Falls back to blocked brute-force dot-product search otherwise.
    """

    def __init__(self, src_theta: np.ndarray, src_phi: np.ndarray, *, prefer_kdtree: bool = False) -> None:
        if src_theta.shape != src_phi.shape or src_theta.ndim != 2:
            raise ValueError("src_theta and src_phi must have the same 2D shape (H, W)")
        self.src_theta = np.asarray(src_theta, dtype=float)
        self.src_phi = np.asarray(src_phi, dtype=float)
        self.H, self.W = self.src_theta.shape
        # Precompute unit vectors of source grid nodes
        self._Vsrc = self._sph_to_vec(self.src_theta.ravel(), self.src_phi.ravel())  # (H*W, 3)
        self.prefer_kdtree = bool(prefer_kdtree) and _HAVE_SCIPY
        if self.prefer_kdtree:  # optional acceleration
            self._kdtree = cKDTree(self._Vsrc)
        else:
            self._kdtree = None

    # -----------------------------
    # Public API
    # -----------------------------
    def compute_weights(
            self,
            level,
            cell_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bilinear weights/indices for target HEALPix angles.

        Parameters
        ----------
        cell_ids : np.ndarray, shape (N,)
            Target **cell_ids** .

        Returns
        -------
        I : np.ndarray, shape (4, N), dtype=int64
            Linear indices of the 4 source corners per target (row-major: j*W + i). Invalid corners are -1.
        W : np.ndarray, shape (4, N), dtype=float64
            Bilinear weights aligned with `I`. Weights are set to 0.0 for invalid corners and normalized to sum to 1
            when at least one corner is valid.
        """
        #compute the coordinate of the selected cell_ids
        heal_theta, heal_phi = hp.pix2ang(2**level,cell_ids,nest=True)

        ht = np.asarray(heal_theta, dtype=float).ravel()
        hpt = np.asarray(heal_phi, dtype=float).ravel()
        if ht.shape != hpt.shape:
            raise ValueError("heal_theta and heal_phi must have the same 1D shape (N,)")
        N = ht.size

        # Target unit vectors
        Vtgt = self._sph_to_vec(ht, hpt)  # (N,3)

        # 1) Choose a seed node for each target (nearest source grid node on the sphere)
        seed_flat = self._nearest_source_indices(Vtgt)
        seed_j, seed_i = np.divmod(seed_flat, self.W)

        # 2) For each target, test up to 4 candidate quads around the seed; pick the best
        I = np.full((4, N), -1, dtype=np.int64)
        W = np.zeros((4, N), dtype=float)
        candidates = [(0, 0), (-1, 0), (0, -1), (-1, -1)]  # offsets for (i0,j0)

        for n in range(N):
            v = Vtgt[n]
            best = None  # (score_tuple, s, t, i0, j0, (idx00, idx10, idx01, idx11))

            for di, dj in candidates:
                i0 = seed_i[n] + di
                j0 = seed_j[n] + dj
                if i0 < 0 or j0 < 0 or i0 + 1 >= self.W or j0 + 1 >= self.H:
                    continue  # out of bounds

                idx00 = j0 * self.W + i0
                idx10 = j0 * self.W + (i0 + 1)
                idx01 = (j0 + 1) * self.W + i0
                idx11 = (j0 + 1) * self.W + (i0 + 1)

                v00 = self._Vsrc[idx00]
                v10 = self._Vsrc[idx10]
                v01 = self._Vsrc[idx01]
                v11 = self._Vsrc[idx11]

                # Local tangent plane at the quad barycenter
                vC = v00 + v10 + v01 + v11
                vC /= np.linalg.norm(vC)
                ex, ey, _ = self._tangent_axes_from_vec(vC)

                # Project 4 corners + target onto (ex, ey)
                P00 = np.array(self._project_to_plane(v00, ex, ey))
                P10 = np.array(self._project_to_plane(v10, ex, ey))
                P01 = np.array(self._project_to_plane(v01, ex, ey))
                P11 = np.array(self._project_to_plane(v11, ex, ey))
                P   = np.array(self._project_to_plane(v,   ex, ey))

                # Invert bilinear mapping f(s,t) = P
                s, t, ok, resid = self._invert_bilinear(P00, P10, P01, P11, P)

                # Prefer in-bounds solutions; otherwise smallest residual
                score = (0, resid) if ok else (1, resid)
                if (best is None) or (score < best[0]):
                    best = (score, s, t, i0, j0, (idx00, idx10, idx01, idx11))

            if best is None:
                continue  # leave weights at 0 and indices at -1

            _, s, t, i0, j0, (idx00, idx10, idx01, idx11) = best

            # Bilinear weights
            w00 = (1.0 - s) * (1.0 - t)
            w10 = s * (1.0 - t)
            w01 = (1.0 - s) * t
            w11 = s * t

            I[:, n] = np.array([idx00, idx10, idx01, idx11], dtype=np.int64)
            W[:, n] = np.array([w00, w10, w01, w11], dtype=float)

            # Normalize for numerical safety
            sW = W[:, n].sum()
            if sW > 0:
                W[:, n] /= sW

        return I, W

    def apply_weights(self, img: np.ndarray, I: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Apply precomputed (I, W) to a source image to obtain values at the HEALPix targets.

        Parameters
        ----------
        img : np.ndarray, shape (H, W)
            Source image values defined on the same grid as (src_theta, src_phi).
        I : np.ndarray, shape (4, N), dtype=int64
            Linear indices (row-major) of corner samples; -1 for invalid corners.
        W : np.ndarray, shape (4, N), dtype=float64
            Bilinear weights aligned with I.

        Returns
        -------
        vals : np.ndarray, shape (N,)
            Interpolated values at the target directions.
        """
        if img.shape != (self.H, self.W):
            raise ValueError(f"img must have shape {(self.H, self.W)}, got {img.shape}")
        img_flat = img.reshape(-1)
        N = I.shape[1]
        vals = np.zeros(N, dtype=float)
        for k in range(4):
            idx = I[k]
            w = W[k]
            m = idx >= 0
            vals[m] += w[m] * img_flat[idx[m]]
        return vals

    # -----------------------------
    # Internal helpers (geometry)
    # -----------------------------
    @staticmethod
    def _sph_to_vec(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """(theta, phi) -> unit vectors (x,y,z). theta=colat, phi=lon, radians."""
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        x = st * cp
        y = st * sp
        z = ct
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def _tangent_axes_from_vec(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return an orthonormal basis (ex, ey, ez=v_hat) for the tangent plane at v."""
        ez = v / np.linalg.norm(v)
        a = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(a, ez)) > 0.9:  # avoid near-colinearity
            a = np.array([0.0, 1.0, 0.0], dtype=float)
        ex = a - np.dot(a, ez) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        return ex, ey, ez

    @staticmethod
    def _project_to_plane(vec: np.ndarray, ex: np.ndarray, ey: np.ndarray) -> Tuple[float, float]:
        """Project 3D unit vector `vec` onto plane spanned by (ex, ey): returns (x, y)."""
        return float(np.dot(vec, ex)), float(np.dot(vec, ey))

    @staticmethod
    def _bilinear_f(s: float, t: float, P00: np.ndarray, P10: np.ndarray, P01: np.ndarray, P11: np.ndarray) -> np.ndarray:
        """Bilinear blend of 4 points in R^2."""
        return ((1 - s) * (1 - t)) * P00 + (s * (1 - t)) * P10 + ((1 - s) * t) * P01 + (s * t) * P11

    @staticmethod
    def _bilinear_jacobian(s: float, t: float, P00: np.ndarray, P10: np.ndarray, P01: np.ndarray, P11: np.ndarray) -> np.ndarray:
        """2x2 Jacobian of the bilinear map at (s,t)."""
        dFds = (-(1 - t)) * P00 + (1 - t) * P10 + (-t) * P01 + t * P11
        dFdt = (-(1 - s)) * P00 + (-s) * P10 + (1 - s) * P01 + s * P11
        return np.stack([dFds, dFdt], axis=-1)

    @classmethod
    def _invert_bilinear(cls, P00: np.ndarray, P10: np.ndarray, P01: np.ndarray, P11: np.ndarray, P: np.ndarray,
                         max_iter: int = 10, tol: float = 1e-9) -> Tuple[float, float, bool, float]:
        """Invert the bilinear map f(s,t)=P with a Newton loop; return (s,t, ok, residual)."""
        # Initial guess from parallelogram (ignore cross term)
        A = np.column_stack([P10 - P00, P01 - P00])  # 2x2
        b = P - P00
        try:
            st0 = np.linalg.lstsq(A, b, rcond=None)[0]
            s, t = float(st0[0]), float(st0[1])
        except np.linalg.LinAlgError:  # fallback
            s, t = 0.5, 0.5

        for _ in range(max_iter):
            F = cls._bilinear_f(s, t, P00, P10, P01, P11)
            r = P - F
            if np.linalg.norm(r) < tol:
                break
            J = cls._bilinear_jacobian(s, t, P00, P10, P01, P11)
            try:
                delta = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                break
            s += float(delta[0])
            t += float(delta[1])

        # Clamp to [0,1] softly and compute residual
        s_c = min(max(s, 0.0), 1.0)
        t_c = min(max(t, 0.0), 1.0)
        F_end = cls._bilinear_f(s_c, t_c, P00, P10, P01, P11)
        resid = float(np.linalg.norm(P - F_end))
        ok = (0.0 <= s <= 1.0) and (0.0 <= t <= 1.0)
        return s_c, t_c, ok, resid

    # -----------------------------
    # Internal helpers (search)
    # -----------------------------
    def _nearest_source_indices(self, Vtgt: np.ndarray) -> np.ndarray:
        """Return flat indices of nearest source nodes for each target unit vector."""
        if self._kdtree is not None:  # fast path
            _, nn = self._kdtree.query(Vtgt, k=1)
            return nn.astype(np.int64)
        # Brute-force in blocks to limit memory
        N = Vtgt.shape[0]
        out = np.empty(N, dtype=np.int64)
        block = 20000
        VsrcT = self._Vsrc.T  # (3, H*W)
        for start in range(0, N, block):
            end = min(N, start + block)
            D = Vtgt[start:end] @ VsrcT  # cosine similarities
            out[start:end] = np.argmax(D, axis=1)
        return out


__all__ = ["HealBili"]
