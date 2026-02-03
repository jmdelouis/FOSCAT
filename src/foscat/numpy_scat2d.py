"""
Self-contained NumPy reimplementation of the 2D FOSCAT scattering pipeline.

This module rebuilds the FoCUS wavelet filters and runs the simplified
S0/S1/S2 scattering steps using only NumPy.  It mirrors the ordering and shapes
expected from :mod:`foscat.scat_cov2D` so it can be plugged into the test
scripts without importing the rest of the library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ScatCoefficients:
    """Container for the scattering outputs."""

    S0: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S2L: np.ndarray
    j1: np.ndarray
    j2: np.ndarray


class NumpyScat2D:
    """Lightweight NumPy-only scattering implementation."""

    def __init__(
        self,
        NORIENT: int = 4,
        LAMBDA: float = 1.2,
        KERNELSZ: int = 5,
        slope: float = 1.0,
        DODIV: bool = False,
        use_median: bool = False,
    ):
        self.NORIENT = NORIENT + 2 if DODIV else NORIENT
        self.base_orient = NORIENT
        self.LAMBDA = LAMBDA
        self.KERNELSZ = KERNELSZ
        self.slope = slope
        self.DODIV = DODIV
        self.use_median = use_median

        self.real_filters, self.imag_filters, self.smooth_filter = (
            self._build_wavelets()
        )

    # ------------------------------------------------------------------
    # Wavelet construction (ported from FoCUS.__init__)
    # ------------------------------------------------------------------
    def _build_wavelets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = self.KERNELSZ
        norient = self.base_orient
        l_NORIENT = self.NORIENT

        wwc = np.zeros((l_NORIENT, k * k), dtype=np.float32)
        wws = np.zeros_like(wwc)

        x = np.repeat(np.arange(k) - k // 2, k).reshape(k, k)
        y = x.T

        if norient == 1:
            xx = (3 / float(k)) * self.LAMBDA * x
            yy = (3 / float(k)) * self.LAMBDA * y
            if k == 5:
                w_smooth = np.exp(-(xx**2 + yy**2))
                tmp = np.exp(-2 * (xx**2 + yy**2)) - 0.25 * np.exp(
                    -0.5 * (xx**2 + yy**2)
                )
            else:
                w_smooth = np.exp(-0.5 * (xx**2 + yy**2))
                tmp = np.exp(-2 * (xx**2 + yy**2)) - 0.25 * np.exp(
                    -0.5 * (xx**2 + yy**2)
                )

            wwc[0] = tmp.flatten() - tmp.mean()
            wws[0] = np.zeros_like(tmp).flatten()
            sigma = math.sqrt(np.mean(wwc[:, 0] ** 2))
            wwc[0] /= sigma
            wws[0] /= sigma
            w_smooth = w_smooth.flatten()
        else:
            for i in range(norient):
                a = (norient - 1 - i) / float(norient) * np.pi
                if k < 5:
                    xx = (3 / float(k)) * self.LAMBDA * (x * np.cos(a) + y * np.sin(a))
                    yy = (3 / float(k)) * self.LAMBDA * (x * np.sin(a) - y * np.cos(a))
                else:
                    xx = (3 / 5) * self.LAMBDA * (x * np.cos(a) + y * np.sin(a))
                    yy = (3 / 5) * self.LAMBDA * (x * np.sin(a) - y * np.cos(a))

                if k == 5:
                    w_smooth = np.exp(
                        -2 * ((3.0 / float(k) * xx) ** 2 + (3.0 / float(k) * yy) ** 2)
                    )
                else:
                    w_smooth = np.exp(-0.5 * (xx**2 + yy**2))

                tmp1 = np.cos(yy * np.pi) * w_smooth
                tmp2 = np.sin(yy * np.pi) * w_smooth

                wwc[i] = tmp1.flatten() - tmp1.mean()
                wws[i] = tmp2.flatten() - tmp2.mean()
                sigma = np.mean(w_smooth)
                wwc[i] /= sigma
                wws[i] /= sigma

                if self.DODIV and i == 0:
                    r = xx**2 + yy**2
                    theta = np.arctan2(yy, xx)
                    theta[k // 2, k // 2] = 0.0
                    tmp1 = r * np.cos(2 * theta) * w_smooth
                    tmp2 = r * np.sin(2 * theta) * w_smooth
                    wwc[norient] = tmp1.flatten() - tmp1.mean()
                    wws[norient] = tmp2.flatten() - tmp2.mean()
                    sigma = np.mean(w_smooth)
                    wwc[norient] /= sigma
                    wws[norient] /= sigma

                    tmp1 = r * np.cos(2 * theta + np.pi)
                    tmp2 = r * np.sin(2 * theta + np.pi)
                    wwc[norient + 1] = tmp1.flatten() - tmp1.mean()
                    wws[norient + 1] = tmp2.flatten() - tmp2.mean()
                    sigma = np.mean(w_smooth)
                    wwc[norient + 1] /= sigma
                    wws[norient + 1] /= sigma

                w_smooth = w_smooth.flatten()

        smooth = (self.slope * (w_smooth / np.sum(w_smooth))).astype(np.float32)
        smooth = smooth.reshape(k, k)
        real = wwc.reshape(self.NORIENT, k, k).astype(np.float32)
        imag = wws.reshape(self.NORIENT, k, k).astype(np.float32)
        return real, imag, smooth

    # ------------------------------------------------------------------
    # Basic tensor utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_batch(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image[None, ...]
        if image.ndim != 3:
            raise ValueError("Expected [H, W] or [B, H, W] input")
        return image

    @staticmethod
    def _reflect_pad(x: np.ndarray, pad: int) -> np.ndarray:
        if pad == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

    @staticmethod
    def _extract_patches(x: np.ndarray, k: int) -> np.ndarray:
        """Sliding window view with shape [B, C, H, W, k, k]."""
        b, c, h, w = x.shape
        shape = (b, c, h - k + 1, w - k + 1, k, k)
        strides = x.strides + x.strides[-2:]
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    def _conv2d(self, x: np.ndarray, filters: np.ndarray) -> np.ndarray:
        """2D convolution with reflect padding and unit stride."""
        b, h, w = x.shape
        x_padded = self._reflect_pad(x[:, None, :, :], filters.shape[-1] // 2)
        patches = self._extract_patches(x_padded, filters.shape[-1])
        # patches: [B, 1, H, W, K, K]
        filt = filters[:, None, :, :]  # [F, 1, K, K]
        res = np.tensordot(patches, filt, axes=((1, 4, 5), (1, 2, 3)))
        # res shape [B, H, W, F]
        return np.moveaxis(res, -1, 1)  # [B, F, H, W]

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        res = self._conv2d(x, self.smooth_filter[None, :, :])
        return res[:, 0, :, :] if res.shape[1] == 1 else res

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _downsample2(x: np.ndarray) -> np.ndarray:
        b, *rest = x.shape
        h, w = rest[-2], rest[-1]
        reshaped = x.reshape(b, *rest[:-2], h // 2, 2, w // 2, 2)
        return reshaped.mean(axis=(-1, -3))

    @staticmethod
    def _bilinear_resize(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        b, h, w = x.shape
        yy = np.linspace(0, h - 1, out_h)
        xx = np.linspace(0, w - 1, out_w)
        Y, X = np.meshgrid(yy, xx, indexing="ij")

        x0 = np.floor(X).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.floor(Y).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)

        dx = X - x0
        dy = Y - y0

        Ia = x[:, y0, x0]
        Ib = x[:, y0, x1]
        Ic = x[:, y1, x0]
        Id = x[:, y1, x1]

        wa = (1 - dx) * (1 - dy)
        wb = dx * (1 - dy)
        wc = (1 - dx) * dy
        wd = dx * dy

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    # ------------------------------------------------------------------
    def eval(self, image: np.ndarray) -> ScatCoefficients:
        img = self._ensure_batch(np.asarray(image, dtype=np.float32))
        nside = min(img.shape[1], img.shape[2])
        if nside <= self.KERNELSZ:
            raise ValueError("Input must be larger than the kernel")
        jmax = int(np.log(nside - self.KERNELSZ) / np.log(2))

        if self.KERNELSZ > 3:
            scale = 2 if self.KERNELSZ == 5 else 4
            img = self._bilinear_resize(img, img.shape[1] * scale, img.shape[2] * scale)

        vmask = np.ones((1, img.shape[1], img.shape[2]), dtype=np.float32)

        s0 = img.mean(axis=(1, 2))

        s1_list = []
        s2_blocks = []
        s2l_blocks = []
        l_image = img
        l2_image = None

        for j1 in range(jmax):
            c_image = self._conv2d(l_image, self.real_filters) + 1j * self._conv2d(
                l_image, self.imag_filters
            )
            conj = c_image * np.conjugate(c_image)
            conj_mag = np.sqrt(np.abs(conj))

            l_s1 = conj_mag.mean(axis=(2, 3), keepdims=True)  # [B, F, 1, 1]
            s1_list.append(np.moveaxis(l_s1[:, :, 0, 0], 1, -1))

            if l2_image is None:
                l2_image = conj_mag[:, None, :, :, :]
            else:
                l2_image = np.concatenate(
                    [conj_mag[:, None, :, :, :], l2_image], axis=1
                )

            b, jc, io, h, w = l2_image.shape
            flat = l2_image.reshape(b * jc * io, h, w)

            c2_pos = self._conv2d(
                self._relu(flat), self.real_filters
            ) + 1j * self._conv2d(self._relu(flat), self.imag_filters)
            c2_neg = self._conv2d(
                self._relu(-flat), self.real_filters
            ) + 1j * self._conv2d(self._relu(-flat), self.imag_filters)

            c2_pos = c2_pos.reshape(b, jc, io, self.NORIENT, h, w)
            c2_neg = c2_neg.reshape(b, jc, io, self.NORIENT, h, w)

            conj2p = c2_pos * np.conjugate(c2_pos)
            conj2m = c2_neg * np.conjugate(c2_neg)

            conj2p_l1 = np.sqrt(np.abs(conj2p))
            conj2m_l1 = np.sqrt(np.abs(conj2m))

            l_s2 = (
                (conj2p - conj2m).mean(axis=(-2, -1)).reshape(b, jc * io, self.NORIENT)
            )
            l_s2l1 = (
                (conj2p_l1 - conj2m_l1)
                .mean(axis=(-2, -1))
                .reshape(b, jc * io, self.NORIENT)
            )

            s2_blocks.append(l_s2l1)
            s2l_blocks.append(l_s2)

            if j1 != jmax - 1:
                vmask = self._downsample2(self._smooth(vmask))
                down = self._downsample2(self._smooth(l2_image.reshape(-1, h, w)))
                nh, nw = down.shape[-2:]
                l2_image = down.reshape(b, jc, io, nh, nw)
                l_image = self._downsample2(self._smooth(l_image))

        s1 = np.stack(s1_list, axis=1)
        s2 = np.concatenate(s2_blocks, axis=1)
        s2l = np.concatenate(s2l_blocks, axis=1)

        s2j1 = []
        s2j2 = []
        for j1 in range(jmax):
            s2j1.extend(list(range(j1 + 1)))
            s2j2.extend([j1] * (j1 + 1))

        return ScatCoefficients(
            S0=s0,
            S1=s1,
            S2=s2,
            S2L=s2l,
            j1=np.asarray(s2j1, dtype=int),
            j2=np.asarray(s2j2, dtype=int),
        )


def compute_scattering(
    image: np.ndarray,
    NORIENT: int = 4,
    LAMBDA: float = 1.2,
    KERNELSZ: int = 5,
    slope: float = 1.0,
    DODIV: bool = False,
) -> ScatCoefficients:
    """Convenience wrapper to compute 2D scattering with NumPy only."""

    op = NumpyScat2D(
        NORIENT=NORIENT,
        LAMBDA=LAMBDA,
        KERNELSZ=KERNELSZ,
        slope=slope,
        DODIV=DODIV,
    )
    return op.eval(image)


__all__ = ["NumpyScat2D", "compute_scattering", "ScatCoefficients"]
