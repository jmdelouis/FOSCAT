"""
UNET for HEALPix (nested) using Foscat oriented convolutions.

This module defines a lightweight, U-Net–like encoder/decoder that operates on
signals defined on the HEALPix sphere (nested scheme). It leverages Foscat's
`HOrientedConvol` for orientation-aware convolutions and `funct` utilities for
upgrade/downgrade (change of nside) operations.

Key design choices
------------------
• **Flat parameter vector**: all convolution kernels are stored in a single
  1‑D vector `self.x`. The dictionaries `self.wconv` and `self.t_wconv` map
  layer indices to *offsets* within that vector.
• **HEALPix-aware down/up-sampling**: down-sampling uses
  `self.f.ud_grade_2`, and up-sampling uses `self.f.up_grade`, both with
  per-level `cell_ids` to preserve locality and orientation information.
• **Skip connections**: U‑Net skip connections are implemented by concatenating
  encoder features with downgraded/upsampled paths along the channel axis.

Shape convention
----------------
All tensors follow the Foscat backend shape `(batch, channels, npix)`.

Dependencies
------------
- foscat.scat_cov as `sc`
- foscat.SphericalStencil as `hs`

Example
-------
>>> import numpy as np
>>> from UNET import UNET
>>> nside = 8
>>> npix = 12 * nside * nside
>>> # Your backend tensor should be created via foscat backend; here we show a placeholder np.array
>>> x = np.random.randn(1, 1, npix).astype(np.float32)
>>> # cell_ids should be provided for the highest resolution (nside)
>>> # and must be consistent with the nested scheme expected by Foscat.
>>> # Example placeholder (use the real one from your pipeline):
>>> cell_ids = np.arange(npix, dtype=np.int64)
>>> net = UNET(in_nside=nside, n_chan_in=1, cell_ids=cell_ids)
>>> y = net.eval(net.f.backend.bk_cast(x))  # forward pass

Notes
-----
- This implementation assumes `cell_ids` is provided for the input resolution
  `in_nside`. It propagates/derives the coarser `cell_ids` across levels.
- Some constructor parameters are reserved for future use (see docstring).
"""

from typing import Dict, Optional
import numpy as np

import foscat.scat_cov as sc
import foscat.SphericalPencil as hs


class UNET:
    """U‑Net–like network on HEALPix (nested) using Foscat oriented convolutions.

    The network is built as an encoder/decoder (down/upsampling) tower. Each
    level performs two oriented convolutions. All kernels are packed in a flat
    parameter vector `self.x` to simplify optimization with external solvers.

    Parameters
    ----------
    nparam : int, optional
        Reserved for future use. Currently unused.
    KERNELSZ : int, optional
        Spatial kernel size (k × k) used by oriented convolutions. Default is 3.
    NORIENT : int, optional
        Reserved for future use (number of orientations). Currently unused.
    chanlist : Optional[list[int]], optional
        Number of output channels per encoder level. If ``None``, it defaults to
        ``[4 * 2**k for k in range(log2(in_nside))]``. The length of this list
        defines the number of encoder/decoder levels.
    in_nside : int, optional
        Input HEALPix nside. Must be a power of two for the implicit
        ``log2(in_nside)`` depth when ``chanlist`` is not given.
    n_chan_in : int, optional
        Number of input channels at the finest resolution. Default is 1.
    cell_ids : array-like of int, required
        Pixel identifiers at the input resolution (nested indexing). They are
        used to build oriented convolutions and to derive coarser grids.
        **Must not be ``None``.**
    SEED : int, optional
        Reserved for future use (random initialization seed). Currently unused.
    filename : Optional[str], optional
        Reserved for future use (checkpoint I/O). Currently unused.

    Attributes
    ----------
    f : object
        Foscat helper exposing the backend and grade/convolution utils.
    KERNELSZ : int
        Effective kernel size used by all convolutions.
    chanlist : list[int]
        Channels per encoder level.
    wconv, t_wconv : Dict[int, int]
        Offsets into the flat parameter vector `self.x` for encoder/decoder
        convolutions respectively.
    hconv, t_hconv : Dict[int, hs.SphericalPencil]
        Per-level oriented convolution operators for encoder/decoder.
    l_cell_ids : Dict[int, np.ndarray]
        Per-level cell ids for downsampled grids (encoder side).
    m_cell_ids : Dict[int, np.ndarray]
        Per-level cell ids for upsampled grids (decoder side). Mirrors levels of
        ``l_cell_ids`` but indexed from the decoder traversal.
    x : backend tensor (1‑D)
        Flat vector holding *all* convolution weights.
    nside : int
        Input nside (finest resolution).
    n_chan_in : int
        Number of channels at input.

    Notes
    -----
    - The constructor prints informative messages about the architecture layout
      (channels and pixel counts) to ease debugging.
    - The implementation keeps the logic identical to the original code; only
      comments, docstrings and variable explanations are added.
    """

    def __init__(
        self,
        nparam: int = 1,
        KERNELSZ: int = 3,
        NORIENT: int = 4,
        chanlist: Optional[list] = None,
        in_nside: int = 1,
        n_chan_in: int = 1,
        cell_ids: Optional[np.ndarray] = None,
        SEED: int = 1234,
        filename: Optional[str] = None,
    ):
        # Foscat function wrapper providing backend and grade ops
        self.f = sc.funct(KERNELSZ=KERNELSZ)

        # If no channel plan is provided, build a default pyramid depth of
        # log2(in_nside) levels with channels growing as 4 * 2**k
        if chanlist is None:
            nlayer = int(np.log2(in_nside))
            chanlist = [4 * 2 ** k for k in range(nlayer)]
        else:
            nlayer = len(chanlist)
        print("N_layer ", nlayer)

        # Internal registries
        n = 0  # running offset in the flat parameter vector
        wconv: Dict[int, int] = {}  # encoder weight offsets
        hconv: Dict[int, hs.SphericalPencil] = {}  # encoder conv operators
        l_cell_ids: Dict[int, np.ndarray] = {}  # encoder level cell ids
        self.KERNELSZ = KERNELSZ
        kernelsz = self.KERNELSZ

        # -----------------------------
        # Encoder (downsampling) build
        # -----------------------------
        l_nside = in_nside
        # NOTE: the original code assumes cell_ids is provided; we keep that
        # contract and copy to avoid side effects.
        l_cell_ids[0] = cell_ids.copy()
        # Create a dummy data tensor to probe shapes; real data arrives in eval()
        l_data = self.f.backend.bk_cast(np.ones([1, 1, l_cell_ids[0].shape[0]]))
        l_chan = n_chan_in
        print("Initial chan %d Npix=%d" % (l_chan, l_data.shape[2]))

        for l in range(nlayer):
            print("Layer %d Npix=%d" % (l, l_data.shape[2]))

            # Record offset for first conv at this level: (in -> chanlist[l])
            wconv[2 * l] = n
            nw = l_chan * chanlist[l] * kernelsz * kernelsz
            print("Layer %d conv [%d,%d]" % (l, l_chan, chanlist[l]))
            n += nw

            # Record offset for second conv at this level: (chanlist[l] -> chanlist[l])
            wconv[2 * l + 1] = n
            nw = chanlist[l] * chanlist[l] * kernelsz * kernelsz
            print("Layer %d conv [%d,%d]" % (l, chanlist[l], chanlist[l]))
            n += nw

            # Build oriented convolution operator for this level
            hconvol = hs.SphericalPencil(l_nside, 3, cell_ids=l_cell_ids[l])
            hconvol.make_idx_weights()  # precompute indices/weights once
            hconv[l] = hconvol

            # Downsample features and propagate cell ids to the next level
            l_data, n_cell_ids = self.f.ud_grade_2(
                l_data, cell_ids=l_cell_ids[l], nside=l_nside
            )
            l_cell_ids[l + 1] = self.f.backend.to_numpy(n_cell_ids)
            l_nside //= 2

            # +1 channel to concatenate the downgraded input (skip-like feature)
            l_chan = chanlist[l] + 1

        # Freeze encoder bookkeeping
        self.n_cnn = n
        self.l_cell_ids = l_cell_ids
        self.wconv = wconv
        self.hconv = hconv

        # -----------------------------
        # Decoder (upsampling) build
        # -----------------------------
        m_cell_ids: Dict[int, np.ndarray] = {}
        m_cell_ids[0] = l_cell_ids[nlayer]
        t_wconv: Dict[int, int] = {}  # decoder weight offsets
        t_hconv: Dict[int, hs.SphericalPencil] = {}  # decoder conv operators

        for l in range(nlayer):
            # Upsample features to the previous (finer) resolution
            l_chan += 1  # account for concatenation before first conv at this level
            l_data = self.f.up_grade(
                l_data,
                l_nside * 2,
                cell_ids=l_cell_ids[nlayer - l],
                o_cell_ids=l_cell_ids[nlayer - 1 - l],
                nside=l_nside,
            )
            print("Transpose Layer %d Npix=%d" % (l, l_data.shape[2]))

            m_cell_ids[l] = l_cell_ids[nlayer - 1 - l]
            l_nside *= 2

            # First decoder conv: (l_chan -> l_chan)
            t_wconv[2 * l] = n
            nw = l_chan * l_chan * kernelsz * kernelsz
            print("Transpose Layer %d conv [%d,%d]" % (l, l_chan, l_chan))
            n += nw

            # Second decoder conv: (l_chan -> out_chan)
            t_wconv[2 * l + 1] = n
            out_chan = 1
            if nlayer - 1 - l > 0:
                out_chan += chanlist[nlayer - 1 - l]
            print("Transpose Layer %d conv [%d,%d]" % (l, l_chan, out_chan))
            nw = l_chan * out_chan * kernelsz * kernelsz
            n += nw

            # Build oriented convolution operator for this decoder level
            hconvol = hs.SphericalPencil(l_nside, 3, cell_ids=m_cell_ids[l])
            hconvol.make_idx_weights()
            t_hconv[l] = hconvol

            # Update channel count after producing out_chan
            l_chan = out_chan

        print("Final chan %d Npix=%d" % (out_chan, l_data.shape[2]))

        # Freeze decoder bookkeeping
        self.n_cnn = n
        self.m_cell_ids = l_cell_ids  # mirror of encoder ids (kept for backward compat)
        self.t_wconv = t_wconv
        self.t_hconv = t_hconv

        # Initialize flat parameter vector with small random values
        self.x = self.f.backend.bk_cast((np.random.rand(n) - 0.5) / self.KERNELSZ)

        # Expose config
        self.nside = in_nside
        self.n_chan_in = n_chan_in
        self.chanlist = chanlist

    def get_param(self):
        """Return the flat parameter vector that stores all convolution kernels.

        Returns
        -------
        backend tensor (1‑D)
            The Foscat backend representation (e.g., NumPy/Torch/TF tensor)
            holding all convolution weights in a single vector.
        """
        return self.x

    def set_param(self, x):
        """Overwrite the flat parameter vector with externally provided values.

        This is useful when optimizing parameters with an external optimizer or
        when restoring weights from a checkpoint (after proper conversion to the
        Foscat backend type).

        Parameters
        ----------
        x : array-like (1‑D)
            New values for the flat parameter vector. Must match `self.x` size.
        """
        self.x = self.f.backend.bk_cast(x)

    def eval(self, data):
        """Run a forward pass through the encoder/decoder.

        Parameters
        ----------
        data : backend tensor, shape (B, C, Npix)
            Input signal at resolution `self.nside` (finest grid). `C` must
            equal `self.n_chan_in`.

        Returns
        -------
        backend tensor, shape (B, C_out, Npix)
            Network output at the input resolution. `C_out` is `1` at the top
            level, or `1 + chanlist[level]` for intermediate decoder levels.

        Notes
        -----
        The forward comprises two stages:
        (1) **Encoder**: for each level `l`, apply two oriented convolutions
            ("conv -> conv"), downsample to the next coarser grid, and
            concatenate with a downgraded copy of the running input (`m_data`).
        (2) **Decoder**: for each level, upsample to the finer grid, concatenate
            with the stored encoder feature (skip connection), then apply two
            oriented convolutions ("conv -> conv") to produce `out_chan`.
        """
        # Encoder state
        l_nside = self.nside
        l_chan = self.n_chan_in
        l_data = data
        m_data = data  # running copy of input used for the additional concat
        nlayer = len(self.chanlist)
        kernelsz = self.KERNELSZ
        ud_data: Dict[int, object] = {}  # stores per-level skip features

        # -----------------
        # Encoder traversal
        # -----------------
        for l in range(nlayer):
            # Fetch weights for conv (in -> chanlist[l]) and reshape to Foscat backend
            nw = l_chan * self.chanlist[l] * kernelsz * kernelsz
            ww = self.x[self.wconv[2 * l] : self.wconv[2 * l] + nw]
            ww = self.f.backend.bk_reshape(
                ww, [l_chan, self.chanlist[l], kernelsz * kernelsz]
            )
            l_data = self.hconv[l].Convol_torch(l_data, ww)

            # Second conv (chanlist[l] -> chanlist[l])
            nw = self.chanlist[l] * self.chanlist[l] * kernelsz * kernelsz
            ww = self.x[self.wconv[2 * l + 1] : self.wconv[2 * l + 1] + nw]
            ww = self.f.backend.bk_reshape(
                ww, [self.chanlist[l], self.chanlist[l], kernelsz * kernelsz]
            )
            l_data = self.hconv[l].Convol_torch(l_data, ww)

            # Downsample features and store skip connection
            l_data, _ = self.f.ud_grade_2(
                l_data, cell_ids=self.l_cell_ids[l], nside=l_nside
            )
            ud_data[l] = m_data

            # Also downgrade the running input for the auxiliary concat
            m_data, _ = self.f.ud_grade_2(
                m_data, cell_ids=self.l_cell_ids[l], nside=l_nside
            )

            # Concatenate along channels: [ downgraded_input , features ]
            l_data = self.f.backend.bk_concat([m_data, l_data], 1)

            l_nside //= 2
            l_chan = self.chanlist[l] + 1  # account for the concat above

        # -----------------
        # Decoder traversal
        # -----------------
        for l in range(nlayer):
            # Upsample to finer grid
            l_chan += 1  # due to upcoming concat with ud_data
            l_data = self.f.up_grade(
                l_data,
                l_nside * 2,
                cell_ids=self.l_cell_ids[nlayer - l],
                o_cell_ids=self.l_cell_ids[nlayer - 1 - l],
                nside=l_nside,
            )

            # Concatenate with encoder skip features
            l_data = self.f.backend.bk_concat([ud_data[nlayer - 1 - l], l_data], 1)
            l_nside *= 2

            # Determine output channels at this level
            out_chan = 1
            if nlayer - 1 - l > 0:
                out_chan += self.chanlist[nlayer - 1 - l]

            # First decoder conv (l_chan -> l_chan)
            nw = l_chan * l_chan * kernelsz * kernelsz
            ww = self.x[self.t_wconv[2 * l] : self.t_wconv[2 * l] + nw]
            ww = self.f.backend.bk_reshape(ww, [l_chan, l_chan, kernelsz * kernelsz])
            c_data = self.t_hconv[l].Convol_torch(l_data, ww)

            # Second decoder conv (l_chan -> out_chan)
            nw = l_chan * out_chan * kernelsz * kernelsz
            ww = self.x[self.t_wconv[2 * l + 1] : self.t_wconv[2 * l + 1] + nw]
            ww = self.f.backend.bk_reshape(ww, [l_chan, out_chan, kernelsz * kernelsz])
            l_data = self.t_hconv[l].Convol_torch(c_data, ww)

            # Update channel count for next iteration
            l_chan = out_chan

        return l_data


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

    
# -----------------------------
# Unit tests (smoke tests)
# -----------------------------
# Run with:  python UNET.py  (or)  python UNET.py -q  for quieter output
# These tests assume Foscat and its dependencies are installed.


def _dummy_cell_ids(nside: int) -> np.ndarray:
    """Return a simple identity mapping for HEALPix nested pixel IDs.

    Notes
    -----
    Replace with your pipeline's real `cell_ids` if you have a precomputed
    mapping consistent with Foscat/HEALPix nested ordering.
    """
    return np.arange(12 * nside * nside, dtype=np.int64)


if __name__ == "__main__":
    import unittest

    class TestUNET(unittest.TestCase):
        """Lightweight smoke tests for shape and parameter plumbing."""

        def setUp(self):
            self.nside = 4  # small grid for fast tests (npix = 192)
            self.chanlist = [4, 8]  # two-level encoder/decoder
            self.batch = 2
            self.channels = 1
            self.npix = 12 * self.nside * self.nside
            self.cell_ids = _dummy_cell_ids(self.nside)
            self.net = UNET(
                in_nside=self.nside,
                n_chan_in=self.channels,
                chanlist=self.chanlist,
                cell_ids=self.cell_ids,
            )

        def test_forward_shape(self):
            # random input
            x = np.random.randn(self.batch, self.channels, self.npix).astype(np.float32)
            x = self.net.f.backend.bk_cast(x)
            y = self.net.eval(x)
            # expected output: same npix, 1 channel at the very top
            self.assertEqual(y.shape[0], self.batch)
            self.assertEqual(y.shape[1], 1)
            self.assertEqual(y.shape[2], self.npix)
            # sanity: no NaNs
            y_np = self.net.f.backend.to_numpy(y)
            self.assertFalse(np.isnan(y_np).any())

        def test_param_roundtrip_and_determinism(self):
            x = np.random.randn(self.batch, self.channels, self.npix).astype(np.float32)
            x = self.net.f.backend.bk_cast(x)

            # forward twice -> identical outputs with fixed params
            y1 = self.net.eval(x)
            y2 = self.net.eval(x)
            y1_np = self.net.f.backend.to_numpy(y1)
            y2_np = self.net.f.backend.to_numpy(y2)
            np.testing.assert_allclose(y1_np, y2_np, rtol=0, atol=0)

            # perturb parameters -> output should (very likely) change
            p = self.net.get_param()
            p_np = self.net.f.backend.to_numpy(p).copy()
            if p_np.size > 0:
                p_np[0] += 1.0
                self.net.set_param(p_np)
                y3 = self.net.eval(x)
                y3_np = self.net.f.backend.to_numpy(y3)
                with self.assertRaises(AssertionError):
                    np.testing.assert_allclose(y1_np, y3_np, rtol=0, atol=0)

    unittest.main()
 
