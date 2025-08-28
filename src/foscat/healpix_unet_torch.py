"""
HEALPix U-Net (nested) with Foscat + PyTorch niceties
----------------------------------------------------
GPU by default (when available), with graceful CPU fallback if Foscat ops are CPU-only.

- ReLU + BatchNorm after each convolution (encoder & decoder)
- Segmentation/Regression heads with optional final activation
- PyTorch-ified: inherits from nn.Module, standard state_dict
- Device management: tries CUDA first; if Foscat HOrientedConvol cannot run on CUDA, falls back to CPU

Shape convention: (B, C, Npix)

Requirements: foscat (scat_cov.funct + HOrientedConvol.Convol_torch must be differentiable on torch tensors)
"""
from __future__ import annotations
from typing import List, Optional, Literal, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import foscat.scat_cov as sc
import foscat.HOrientedConvol as hs
import matplotlib.pyplot as plt


class HealpixUNet(nn.Module):
    """U-Net-like architecture on the HEALPix sphere using Foscat oriented convolutions.

    Parameters
    ----------
    in_nside : int
        Input HEALPix nside (nested scheme).
    n_chan_in : int
        Number of input channels.
    chanlist : list[int]
        Channels per encoder level (depth = len(chanlist)). Example: [16, 32, 64].
    cell_ids : np.ndarray
        Cell indices for the finest resolution (nside = in_nside) in nested scheme.
    KERNELSZ : int, default 3
        Spatial kernel size K (K x K) for oriented convolution.
    task : {'regression','segmentation'}, default 'regression'
        Chooses the head and default activation.
    out_channels : int, default 1
        Number of output channels (e.g. num_classes for segmentation).
    final_activation : {'none','sigmoid','softmax'} | None
        If None, uses sensible default per task: 'none' for regression, 'softmax' for segmentation (multi-class),
        'sigmoid' for segmentation when out_channels==1.
    device : str | torch.device | None, default: 'cuda' if available else 'cpu'
        Preferred device. The module will probe whether Foscat ops can run on CUDA; if not,
        it will fallback to CPU and keep all parameters/buffers on CPU for consistency.
    prefer_foscat_gpu : bool, default True
        When device is CUDA, try to move Foscat operators (internal tensors) to CUDA and do a dry-run.
        If the dry-run fails, everything falls back to CPU.

    Notes
    -----
    - Two oriented convolutions per level. After each conv: BatchNorm1d + ReLU.
    - Downsampling uses foscat ``ud_grade_2``; upsampling uses ``up_grade``.
    - Convolution kernels are explicit parameters (shape [C_in, C_out, K*K]) and applied via ``HOrientedConvol.Convol_torch``.
    - Foscat ops device is auto-probed to avoid CPU/CUDA mismatches.
    """

    def __init__(
        self,
        *,
        in_nside: int,
        n_chan_in: int,
        chanlist: List[int],
        cell_ids: np.ndarray,
        KERNELSZ: int = 3,
        task: Literal['regression', 'segmentation'] = 'regression',
        out_channels: int = 1,
        final_activation: Optional[Literal['none', 'sigmoid', 'softmax']] = None,
        device: Optional[torch.device | str] = None,
        prefer_foscat_gpu: bool = True,
    ) -> None:
        super().__init__()

        if cell_ids is None:
            raise ValueError("cell_ids must be provided for the finest resolution.")
        if len(chanlist) == 0:
            raise ValueError("chanlist must be non-empty (depth >= 1).")

        self.in_nside = int(in_nside)
        self.n_chan_in = int(n_chan_in)
        self.chanlist = list(map(int, chanlist))
        self.KERNELSZ = int(KERNELSZ)
        self.task = task
        self.out_channels = int(out_channels)
        self.prefer_foscat_gpu = bool(prefer_foscat_gpu)

        # Choose default final activation if not given
        if final_activation is None:
            if task == 'regression':
                self.final_activation = 'none'
            else:  # segmentation
                self.final_activation = 'sigmoid' if out_channels == 1 else 'softmax'
        else:
            self.final_activation = final_activation

        # Resolve preferred device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Foscat functional wrapper (backend + grade ops)
        self.f = sc.funct(KERNELSZ=self.KERNELSZ)

        # ---------- Build multi-resolution bookkeeping ----------
        depth = len(self.chanlist)
        self.l_cell_ids: List[np.ndarray] = [None] * (depth + 1)  # per encoder level + bottom
        self.l_cell_ids[0] = np.asarray(cell_ids)

        enc_nsides: List[int] = [self.in_nside]
        current_nside = self.in_nside

        # dummy data to propagate shapes/ids through ud_grade_2
        npix = 12 * current_nside * current_nside
        l_data = self.f.backend.bk_cast(np.zeros((1, 1, cell_ids.shape[0]), dtype=np.float32))

        for l in range(depth):
            # downsample once to get next level ids and new data shape
            l_data, next_ids = self.f.ud_grade_2(
                l_data, cell_ids=self.l_cell_ids[l], nside=current_nside
            )
            self.l_cell_ids[l + 1] = self.f.backend.to_numpy(next_ids)
            current_nside //= 2
            enc_nsides.append(current_nside)

        self.enc_nsides = enc_nsides  # [in, in/2, ..., in/2**depth]

        # ---------- Oriented convolutions per level (encoder & decoder) ----------
        self.hconv_enc: List[hs.HOrientedConvol] = []
        self.hconv_dec: List[hs.HOrientedConvol] = []

        # encoder conv weights and BN
        self.enc_w1 = nn.ParameterList()
        self.enc_bn1 = nn.ModuleList()
        self.enc_w2 = nn.ParameterList()
        self.enc_bn2 = nn.ModuleList()

        inC = self.n_chan_in
        for l, outC in enumerate(self.chanlist):
            # operator at encoder level l
            hc = hs.HOrientedConvol(self.enc_nsides[l], self.KERNELSZ, cell_ids=self.l_cell_ids[l])
            hc.make_idx_weights()
            self.hconv_enc.append(hc)

            # conv1: inC -> outC
            w1 = torch.empty(inC, outC, self.KERNELSZ * self.KERNELSZ)
            nn.init.kaiming_uniform_(w1.view(inC * outC, -1), a=np.sqrt(5))
            self.enc_w1.append(nn.Parameter(w1))
            self.enc_bn1.append(nn.BatchNorm1d(outC))

            # conv2: outC -> outC
            w2 = torch.empty(outC, outC, self.KERNELSZ * self.KERNELSZ)
            nn.init.kaiming_uniform_(w2.view(outC * outC, -1), a=np.sqrt(5))
            self.enc_w2.append(nn.Parameter(w2))
            self.enc_bn2.append(nn.BatchNorm1d(outC))

            inC = outC  # next level input channels

        # decoder conv weights and BN (mirrored levels)
        self.dec_w1 = nn.ParameterList()
        self.dec_bn1 = nn.ModuleList()
        self.dec_w2 = nn.ParameterList()
        self.dec_bn2 = nn.ModuleList()

        for d in range(depth):
            level = depth - 1 - d  # encoder level we are going back to
            hc = hs.HOrientedConvol(self.enc_nsides[level], self.KERNELSZ, cell_ids=self.l_cell_ids[level])
            hc.make_idx_weights()
            self.hconv_dec.append(hc)

            upC = self.chanlist[level + 1] if level + 1 < depth else self.chanlist[level]
            skipC = self.chanlist[level]
            inC_dec = upC + skipC
            outC_dec = skipC

            w1 = torch.empty(inC_dec, outC_dec, self.KERNELSZ * self.KERNELSZ)
            nn.init.kaiming_uniform_(w1.view(inC_dec * outC_dec, -1), a=np.sqrt(5))
            self.dec_w1.append(nn.Parameter(w1))
            self.dec_bn1.append(nn.BatchNorm1d(outC_dec))

            w2 = torch.empty(outC_dec, outC_dec, self.KERNELSZ * self.KERNELSZ)
            nn.init.kaiming_uniform_(w2.view(outC_dec * outC_dec, -1), a=np.sqrt(5))
            self.dec_w2.append(nn.Parameter(w2))
            self.dec_bn2.append(nn.BatchNorm1d(outC_dec))

        # Output head (on finest grid, channels = chanlist[0])
        self.head_hconv = hs.HOrientedConvol(self.in_nside, self.KERNELSZ, cell_ids=self.l_cell_ids[0])
        self.head_hconv.make_idx_weights()
        head_inC = self.chanlist[0]
        self.head_w = nn.Parameter(torch.empty(head_inC, self.out_channels, self.KERNELSZ * self.KERNELSZ))
        nn.init.kaiming_uniform_(self.head_w.view(head_inC * self.out_channels, -1), a=np.sqrt(5))
        self.head_bn = nn.BatchNorm1d(self.out_channels) if self.task == 'segmentation' else None

        # ---- Decide runtime device (probe Foscat on CUDA, else CPU) ----
        self.runtime_device = self._probe_and_set_runtime_device(self.device)

    # -------------------------- device plumbing --------------------------
    def _move_hconv_tensors(self, hc: hs.HOrientedConvol, device: torch.device) -> None:
        """Best-effort: move any torch.Tensor attribute of HOrientedConvol to device."""
        for name, val in list(vars(hc).items()):
            try:
                if torch.is_tensor(val):
                    setattr(hc, name, val.to(device))
                elif isinstance(val, (list, tuple)) and val and torch.is_tensor(val[0]):
                    setattr(hc, name, type(val)([v.to(device) for v in val]))
            except Exception:
                # silently ignore non-tensor or protected attributes
                pass

    @torch.no_grad()
    def _probe_and_set_runtime_device(self, preferred: torch.device) -> torch.device:
        """Try to run a tiny Foscat conv on preferred device; fallback to CPU if it fails."""
        if preferred.type == 'cuda' and self.prefer_foscat_gpu:
            try:
                # move module params/buffers first
                super().to(preferred)
                # move Foscat operator internals
                for hc in self.hconv_enc + self.hconv_dec + [self.head_hconv]:
                    self._move_hconv_tensors(hc, preferred)
                # dry run on level 0
                npix0 = int(len(self.l_cell_ids[0]))
                x_try = torch.zeros(1, self.n_chan_in, npix0, device=preferred)
                y_try = self.hconv_enc[0].Convol_torch(x_try, self.enc_w1[0])
                # success -> stay on CUDA
                self._foscat_device = preferred
                return preferred
            except Exception as e:
                # fallback to CPU; keep error for info
                self._gpu_probe_error = repr(e)
                pass
        # CPU fallback
        cpu = torch.device('cpu')
        super().to(cpu)
        for hc in self.hconv_enc + self.hconv_dec + [self.head_hconv]:
            self._move_hconv_tensors(hc, cpu)
        self._foscat_device = cpu
        return cpu

    def set_device(self, device: torch.device | str) -> torch.device:
        """Request a (re)device; will probe Foscat and return the actual runtime device used."""
        device = torch.device(device)
        self.device = device
        self.runtime_device = self._probe_and_set_runtime_device(device)
        return self.runtime_device

    # -------------------------- forward --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C_in, Npix)
            Input tensor on `in_nside` grid.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError("Input must be (B, C, Npix)")

        # Ensure input lives on the runtime (probed) device
        x = x.to(self.runtime_device)

        B, C, N = x.shape
        if C != self.n_chan_in:
            raise ValueError(f"Expected {self.n_chan_in} input channels, got {C}")

        # Encoder
        skips: List[torch.Tensor] = []
        l_data = x
        current_nside = self.in_nside
        for l, outC in enumerate(self.chanlist):
            # conv1 + BN + ReLU
            l_data = self.hconv_enc[l].Convol_torch(l_data, self.enc_w1[l])
            l_data = self.enc_bn1[l](l_data)
            l_data = F.relu(l_data, inplace=True)

            # conv2 + BN + ReLU
            l_data = self.hconv_enc[l].Convol_torch(l_data, self.enc_w2[l])
            l_data = self.enc_bn2[l](l_data)
            l_data = F.relu(l_data, inplace=True)

            # save skip at this resolution
            skips.append(l_data)

            # downsample (except bottom level) -> ensure output is on runtime_device
            if l < len(self.chanlist) - 1:
                l_data, _ = self.f.ud_grade_2(
                    l_data, cell_ids=self.l_cell_ids[l], nside=current_nside
                )
                if isinstance(l_data, torch.Tensor) and l_data.device != self.runtime_device:
                    l_data = l_data.to(self.runtime_device)
                current_nside //= 2

        # Decoder
        for d in range(len(self.chanlist)):
            level = len(self.chanlist) - 1 - d  # corresponding encoder level

            if level < len(self.chanlist) - 1:
                # upsample to next finer grid
                # upsample to next finer grid (from level+1 -> level)
                src_nside = self.enc_nsides[level + 1]    # current (coarser)
                tgt_nside = self.enc_nsides[level]        # next finer (== src*2)
                # Foscat up_grade signature expects current (coarse) ids in `cell_ids`
                # and target (fine) ids in `o_cell_ids` (matching original UNET code).
                l_data = self.f.up_grade(
                    l_data,
                    tgt_nside,
                    cell_ids=self.l_cell_ids[level + 1],  # source (coarser) ids
                    o_cell_ids=self.l_cell_ids[level],     # target (finer) ids
                    nside=src_nside,
                )
                if isinstance(l_data, torch.Tensor) and l_data.device != self.runtime_device:
                    l_data = l_data.to(self.runtime_device)

            # concat with skip features at this resolution
            concat = self.f.backend.bk_concat([skips[level], l_data], 1)
            l_data = concat.to(self.runtime_device) if torch.is_tensor(concat) else concat

            # apply decoder convs on this grid
            hc = self.hconv_dec[d]
            l_data = hc.Convol_torch(l_data, self.dec_w1[d])
            l_data = self.dec_bn1[d](l_data)
            l_data = F.relu(l_data, inplace=True)

            l_data = hc.Convol_torch(l_data, self.dec_w2[d])
            l_data = self.dec_bn2[d](l_data)
            l_data = F.relu(l_data, inplace=True)

        # Head on finest grid
        out = self.head_hconv.Convol_torch(l_data, self.head_w)
        if self.head_bn is not None:
            out = self.head_bn(out)
        if self.final_activation == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.final_activation == 'softmax':
            out = torch.softmax(out, dim=1)
        return out

    # -------------------------- utilities --------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, batch_size: int = 8) -> torch.Tensor:
        self.eval()
        outs = []
        for i in range(0, x.shape[0], batch_size):
            outs.append(self.forward(x[i : i + batch_size]))
        return torch.cat(outs, dim=0)

    # -----------------------------
    # Kernel extraction & plotting
    # -----------------------------
    def _arch_shapes(self):
        """Return expected (in_c, out_c) per conv for encoder/decoder.

        Returns
        -------
        enc_shapes : list[tuple[tuple[int,int], tuple[int,int]]]
            For each level `l`, ((in1, out1), (in2, out2)) for the two encoder convs.
        dec_shapes : list[tuple[tuple[int,int], tuple[int,int]]]
            For each level `l`, ((in1, out1), (in2, out2)) for the two decoder convs.
        """
        nlayer = len(self.chanlist)
        enc_shapes = []
        l_chan = self.n_chan_in
        for l in range(nlayer):
            enc_shapes.append(((l_chan, self.chanlist[l]), (self.chanlist[l], self.chanlist[l])))
            l_chan = self.chanlist[l] + 1

        dec_shapes = []
        l_chan = self.chanlist[-1] + 1
        for l in range(nlayer):
            in1 = l_chan + 1
            out2 = 1 + (self.chanlist[nlayer - 1 - l] if (nlayer - 1 - l) > 0 else 0)
            dec_shapes.append(((in1, in1), (in1, out2)))
            l_chan = out2
        return enc_shapes, dec_shapes

    def extract_kernels(self, stage: str = "encoder", layer: int = 0, conv: int = 0):
        """Extract raw convolution kernels for a given stage/level/conv.

        Parameters
        ----------
        stage : {"encoder", "decoder"}
            Which part of the network to inspect.
        layer : int
            Pyramid level (0 = finest encoder level / bottommost decoder level).
        conv : int
            0 for the first conv at that level, 1 for the second conv.

        Returns
        -------
        np.ndarray
            Array of shape (in_c, out_c, K, K) containing the spatial kernels.
        """
        assert stage in {"encoder", "decoder"}
        assert conv in {0, 1}
        K = self.KERNELSZ
        enc_shapes, dec_shapes = self._arch_shapes()

        if stage == "encoder":
            if conv==0:
                w = self.enc_w1[layer]
            else:
                w = self.enc_w2[layer]
        else:
            if conv==0:
                w = self.dec_w1[layer]
            else:
                w = self.dec_w2[layer]

        w_np = self.f.backend.to_numpy(w.detach())
        return w_np.reshape(w.shape[0],w.shape[1],K,K)

    def plot_kernels(
        self,
        stage: str = "encoder",
        layer: int = 0,
        conv: int = 0,
        fixed: str = "in",
        index: int = 0,
        max_tiles: int = 16,
    ):
        """Quick visualization of kernels on a grid using matplotlib.

        Parameters
        ----------
        stage : {"encoder", "decoder"}
            Which tower to visualize.
        layer : int
            Level to visualize.
        conv : int
            0 or 1: first or second conv in the level.
        fixed : {"in", "out"}
            If "in", show kernels for a fixed input channel across many outputs.
            If "out", show kernels for a fixed output channel across many inputs.
        index : int
            Channel index to fix (according to `fixed`).
        max_tiles : int
            Maximum number of tiles to display.
        """
        import math
        import matplotlib.pyplot as plt

        W = self.extract_kernels(stage=stage, layer=layer, conv=conv)
        ic, oc, K,_ = W.shape

        if fixed == "in":
            idx = min(index, ic - 1)
            tiles = [W[idx, j] for j in range(oc)]
            title = f"{stage} L{layer} C{conv} | in={idx}"
        else:
            idx = min(index, oc - 1)
            tiles = [W[i, idx] for i in range(ic)]
            title = f"{stage} L{layer} C{conv} | out={idx}"

        tiles = tiles[:max_tiles]
        n = len(tiles)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

        plt.figure(figsize=(2.5 * cols, 2.5 * rows))
        for i, ker in enumerate(tiles, 1):
            ax = plt.subplot(rows, cols, i)
            ax.imshow(ker)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

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

def fit(
        model: HealpixUNet,
        x_train: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        *,
        n_epoch: int = 10,
        view_epoch: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        clip_grad_norm: float | None = None,
        verbose: bool = True,
        optimizer:  Literal['ADAM', 'LBFGS'] = 'LBFGS',
) -> dict:
    """Train helper for the torch-ified HEALPix U-Net.

    Optimizes all registered parameters (kernels + BN affine) with Adam on MSE for regression,
    or CrossEntropy/BCE for segmentation.

    Device policy
    -------------
    Uses the model's probed runtime device (CUDA if Foscat conv works there; otherwise CPU).
    """
    import numpy as _np
    from torch.utils.data import TensorDataset, DataLoader

    # Ensure model is on its runtime device (already probed in __init__)
    model.to(model.runtime_device)

    def _to_t(x):
        if isinstance(x, torch.Tensor):
            return x.float().to(model.runtime_device)
        return torch.from_numpy(_np.asarray(x)).float().to(model.runtime_device)

    x_t = _to_t(x_train)
    y_t = _to_t(y_train)

    # choose loss
    if model.task == 'regression':
        criterion = nn.MSELoss()
    else:
        if model.out_channels == 1:
            criterion = nn.BCEWithLogitsLoss() if model.final_activation == 'none' else nn.BCELoss()
        else:
            if y_t.dim() == 3:
                y_t = y_t.argmax(dim=1)
            criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(x_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if optimizer=='ADAM':
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        l_n_epoch=n_epoch
        n_inter=0
    else:
        optim = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=n_epoch*5, line_search_fn="strong_wolfe")
        l_n_epoch=n_epoch//20
        n_inter=20

    history: List[float] = []
    model.train()

    for epoch in range(l_n_epoch):
        for k in range(n_inter):
            epoch_loss = 0.0
            n_samples = 0
            for xb, yb in loader:
                # LBFGS a besoin d'un closure qui recalcule loss et gradients
                if isinstance(optim, torch.optim.LBFGS):
                    def closure():
                        optim.zero_grad(set_to_none=True)
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        loss.backward()
                        return loss

                    _ = optim.step(closure)  # LBFGS appelle plusieurs fois le closure
                    # on recalcule la loss finale pour l’agg (sans gradient)
                    with torch.no_grad():
                        preds = model(xb)
                        loss = criterion(preds, yb)

                else:
                    optim.zero_grad(set_to_none=True)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    if clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optim.step()

                bs = xb.shape[0]
                epoch_loss += loss.item() * bs
                n_samples += bs

            epoch_loss /= max(1, n_samples)
            history.append(epoch_loss)
            if verbose and (epoch*n_inter+k+1)%view_epoch==0:
                print(f"[epoch {epoch*n_inter+k+1}/{l_n_epoch*n_inter}] loss={epoch_loss:.6f}")

    return {"loss": history}


__all__ = ["HealpixUNet", "fit"]
