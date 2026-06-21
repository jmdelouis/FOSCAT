"""
SynthHalfUNet2D — fast texture synthesis via a decoder-only U-Net.

Architecture
------------
Input: white noise  z ~ N(0,1),  shape [N, 1, H, W]

Skip connections (frozen wavelet bank, no encoder):
  At each scale j (j=0 finest, j=Jmax coarsest):
    z_j     = AvgPool2d^j(z)                          [N, 1,  H/2^j, W/2^j]
    skip_j  = [Re(ψ_l ★ z_j), Im(ψ_l ★ z_j)]_{l}   [N, 2L, H/2^j, W/2^j]

  where {ψ_l} are the L FOSCAT oriented complex wavelets (fixed, from FoCUS).

  At the coarsest level Jmax an extra low-frequency channel is added:
    z_avg = z_Jmax                                     [N, 1,  H/2^Jmax, W/2^Jmax]
  so the initial input to the decoder is [skip_Jmax, z_avg] → 2L+1 channels.

Decoder (all parameters are learned):
  x = InitBlock( cat[skip_Jmax, z_avg] )              [N, C[0], H/2^Jmax, W/2^Jmax]
  for j in Jmax-1 .. 0:
    x = Upsample(x, ×2)
    x = DecBlock_j( cat[x, skip_j] )                  [N, C[Jmax-j], H/2^j, W/2^j]
  out = Conv1×1(x)                                     [N, out_ch, H, W]

Each DecBlock is:  Conv3×3 → BN → LeakyReLU → Conv3×3 → BN → LeakyReLU

Training loss:
  z_i ~ N(0,1)  (sampled fresh each epoch)
  x_i = network(z_i)
  loss = reduce_distance( mean_scat_cov({x_i}), target_scat_cov )
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock2D(nn.Module):
    """Two stacked Conv3×3 + InstanceNorm + LeakyReLU layers.

    InstanceNorm is used instead of BatchNorm so that each sample in the batch
    is normalised independently.  BatchNorm computes statistics across the
    batch dimension and therefore averages out the sample-to-sample variation
    introduced by the noise input z, causing mode collapse (all z produce the
    same output).  InstanceNorm has no such effect.
    """

    def __init__(self, in_ch: int, out_ch: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class SynthHalfUNet2D(nn.Module):
    """Decoder-only U-Net for fast scattering-covariance texture synthesis.

    At inference time a single forward pass (milliseconds) replaces the slow
    gradient-descent synthesis (minutes).  The network is trained by
    overfitting to a single target scattering covariance using
    :func:`train_synth_unet`.

    Parameters
    ----------
    scat_op : FoCUS
        A FOSCAT FoCUS object initialised with ``use_2D=True``.  Its oriented
        wavelet kernels (``ww_RealT[1]``, ``ww_ImagT[1]``) are extracted and
        registered as **frozen** buffers (not trained).
    Jmax : int
        Number of wavelet scales (decoder depth).  The spatial resolution at
        the coarsest level is H/2^Jmax × W/2^Jmax.
    channel_list : list[int] or None
        Feature-map channels at each decoder level, ordered **coarsest →
        finest**: ``channel_list[0]`` is used at level Jmax (coarsest),
        ``channel_list[-1]`` is used just before the output Conv1×1.
        Must have length ``Jmax + 1``.
        Default: ``[min(32·2^(Jmax-j), 256) for j in 0..Jmax]``,
        e.g. ``[256, 128, 64, 32]`` for Jmax=3.
    out_channels : int
        Number of output image channels (1 for single-channel textures).
        Generate N independent samples by setting the batch size of z to N.

    Examples
    --------
    >>> import torch
    >>> from foscat.SynthHalfUNet2D import SynthHalfUNet2D, train_synth_unet, generate_samples
    >>> import foscat.scat_cov2D as sc
    >>>
    >>> scat_op = sc.funct(NORIENT=4, KERNELSZ=3, use_2D=True)
    >>> target = torch.tensor(my_image_2d)           # shape [H, W]
    >>>
    >>> # Train (overfits to the scattering covariance of target)
    >>> model = train_synth_unet(target, scat_op, Jmax=3, n_epochs=2000)
    >>>
    >>> # Generate new samples instantly
    >>> samples = generate_samples(model, n_samples=8, H=256, W=256)
    >>> # samples: torch.Tensor [8, 1, 256, 256]
    """

    def __init__(
        self,
        scat_op,
        Jmax: int,
        channel_list: list[int] | None = None,
        out_channels: int = 1,
    ):
        super().__init__()

        self.Jmax = Jmax
        L = scat_op.NORIENT
        K = scat_op.KERNELSZ
        self.norient = L
        self.kernelsz = K
        pad = K // 2

        # ------------------------------------------------------------------ #
        # Frozen wavelet bank — extracted from the FoCUS object               #
        # Shape [L, K, K] → reshaped to [L, 1, K, K] for F.conv2d            #
        # ------------------------------------------------------------------ #
        def _to_tensor(t):
            if isinstance(t, np.ndarray):
                return torch.tensor(t, dtype=torch.float32)
            return t.detach().float()

        wc = _to_tensor(scat_op.ww_RealT[1]).reshape(L, 1, K, K)
        ws = _to_tensor(scat_op.ww_ImagT[1]).reshape(L, 1, K, K)
        self.register_buffer("_wc", wc)   # real part  [L, 1, K, K]
        self.register_buffer("_ws", ws)   # imag part  [L, 1, K, K]

        # ------------------------------------------------------------------ #
        # Channel layout                                                       #
        # channel_list[0] = coarsest (Jmax), channel_list[-1] = finest (0)   #
        # ------------------------------------------------------------------ #
        if channel_list is None:
            channel_list = [min(32 * 2 ** (Jmax - j), 256) for j in range(Jmax + 1)]
        assert len(channel_list) == Jmax + 1, (
            f"channel_list must have Jmax+1={Jmax+1} entries, got {len(channel_list)}"
        )
        self.channel_list = channel_list

        # Each skip contains: Re(ψ_l ★ z_j), Im(ψ_l ★ z_j), z_j
        # The extra z_j channel gives the decoder a direct, unfiltered path
        # from the noise to each decoder level, preventing mode collapse.
        skip_ch = 2 * L + 1   # real + imaginary (2L) + raw z (1)

        # ------------------------------------------------------------------ #
        # Decoder blocks                                                       #
        # ------------------------------------------------------------------ #
        # InitBlock: input = skip_Jmax (2L+1 ch, which already contains z_Jmax)
        self.init_block = ConvBlock2D(skip_ch, channel_list[0])

        # One ConvBlock per upsampling step (Jmax steps, from Jmax-1 to 0)
        self.decoder_blocks = nn.ModuleList()
        for k in range(Jmax):
            in_ch  = channel_list[k] + skip_ch   # upsampled features + skip
            out_ch = channel_list[k + 1]
            self.decoder_blocks.append(ConvBlock2D(in_ch, out_ch))

        # 1×1 projection to output channels
        self.output_conv = nn.Conv2d(channel_list[-1], out_channels, kernel_size=1)

        self._pad = pad

    # ------------------------------------------------------------------ #
    # Wavelet skip-connection bank                                         #
    # ------------------------------------------------------------------ #

    def _wavelet_skips(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Return skip tensors at every scale.

        Each skip contains the oriented wavelet responses of z at that scale
        **plus the raw (downsampled) noise z_j itself**.  The extra z_j
        channel gives the decoder a direct, unfiltered path from the noise to
        each spatial resolution, which is the primary mechanism for producing
        diverse outputs.

        Returns
        -------
        skips : list of tensors, len = Jmax+1
            ``skips[j]`` has shape ``[N, 2L+1, H/2^j, W/2^j]``
            Channels: ``[Re(ψ_0★z_j), …, Re(ψ_{L-1}★z_j),``
                        ``Im(ψ_0★z_j), …, Im(ψ_{L-1}★z_j),``
                        ``z_j]``
            (j=0 finest, j=Jmax coarsest).
        """
        pad = self._pad
        wc = self._wc   # [L, 1, K, K]
        ws = self._ws

        skips = []
        z_j = z                   # start at full resolution [N, 1, H, W]
        for j in range(self.Jmax + 1):
            if j > 0:
                z_j = F.avg_pool2d(z_j, kernel_size=2, stride=2)
            real = F.conv2d(z_j, wc, padding=pad)   # [N, L, H/2^j, W/2^j]
            imag = F.conv2d(z_j, ws, padding=pad)
            # Concatenate wavelet responses + raw noise: [N, 2L+1, H/2^j, W/2^j]
            skips.append(torch.cat([real, imag, z_j], dim=1))

        return skips   # index 0 = finest, index Jmax = coarsest

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate synthesised images from white noise.

        Parameters
        ----------
        z : torch.Tensor, shape [N, 1, H, W]
            White noise input.  N independent samples are produced in parallel.

        Returns
        -------
        torch.Tensor, shape [N, out_channels, H, W]
        """
        skips = self._wavelet_skips(z)   # skips[0]=finest .. skips[Jmax]=coarsest
        # Each skip[j] already contains [Re, Im, z_j] — 2L+1 channels.

        # ---- Initial block at coarsest scale ----
        # skip[Jmax] = [Re(ψ★z_J), Im(ψ★z_J), z_J]  — 2L+1 channels
        x = self.init_block(skips[self.Jmax])

        # ---- Decode from Jmax-1 down to 0 ----
        for k, block in enumerate(self.decoder_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            j = self.Jmax - 1 - k           # scale of the skip to fuse
            x = block(torch.cat([x, skips[j]], dim=1))

        return self.output_conv(x)


# ---------------------------------------------------------------------------
# Microcanonical loss helper
# ---------------------------------------------------------------------------

def _microcanonical_loss(synth_sc, target_sc, eps: float = 1e-6) -> torch.Tensor:
    """Microcanonical scattering-covariance loss.

    Instead of penalising each generated image individually, this loss
    constrains the **distribution** of statistics across the N-sample batch:

    .. math::

        \\mathcal{L} = \\sum_k
            \\frac{(\\bar{\\Phi}_k - \\Phi^*_k)^2}{\\sigma^2_k + \\varepsilon}

    where :math:`\\bar{\\Phi}_k` and :math:`\\sigma^2_k` are the empirical
    mean and variance of coefficient *k* across the N generated images.

    **Key property:** if all generated images are identical (mode collapse),
    :math:`\\sigma^2_k \\to 0` and the loss diverges, naturally penalising
    collapse and encouraging diversity.

    Parameters
    ----------
    synth_sc : scat_cov
        Scattering covariance of the N generated images (batch dimension N).
    target_sc : scat_cov
        Scattering covariance of the target image (batch dimension 1).
    eps : float
        Floor added to the variance to avoid division by zero.

    Returns
    -------
    torch.Tensor (scalar)
    """
    # Locate a non-None tensor to get device / dtype
    _ref = next(
        (getattr(synth_sc, a) for a in ("S2", "S3", "S4", "S1", "S3P", "S0")
         if getattr(synth_sc, a, None) is not None),
        None,
    )
    device = _ref.device if _ref is not None else "cpu"
    dtype  = _ref.dtype  if _ref is not None else torch.float32

    loss = torch.zeros([], device=device, dtype=dtype if not torch.is_complex(_ref) else _ref.real.dtype)

    def _term(s_t, t_t):
        if s_t is None or t_t is None:
            return 0.0
        # target may have batch=1; take mean over it to be safe
        t = t_t.mean(dim=0)          # [...] — removes batch dimension
        if torch.is_complex(s_t):
            s_r, s_i = s_t.real, s_t.imag
            mean_r = s_r.mean(dim=0)
            mean_i = s_i.mean(dim=0)
            var_r  = s_r.var(dim=0, unbiased=False).clamp(min=eps)
            var_i  = s_i.var(dim=0, unbiased=False).clamp(min=eps)
            return (
                ((mean_r - t.real) ** 2 / var_r).sum()
                + ((mean_i - t.imag) ** 2 / var_i).sum()
            )
        else:
            mean = s_t.mean(dim=0)
            var  = s_t.var(dim=0, unbiased=False).clamp(min=eps)
            return ((mean - t) ** 2 / var).sum()

    for attr in ("S0", "S1", "S2", "S3", "S3P", "S4"):
        loss = loss + _term(getattr(synth_sc, attr, None),
                            getattr(target_sc, attr, None))

    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_synth_unet(
    target_image: torch.Tensor,
    scat_op,
    Jmax: int,
    n_samples: int = 4,
    channel_list: list[int] | None = None,
    out_channels: int = 1,
    lr: float = 1e-3,
    n_epochs: int = 2000,
    eval_frequency: int = 100,
    norm: str = "auto",
    Jmax_scat=None,
    edge: bool = False,
    iso_ang: bool = False,
    fft_ang: bool = False,
    fft_nharm: int = 1,
    fft_imaginary: bool = True,
    microcanonical: bool = True,
    micro_eps: float = 1e-6,
    device: str | None = None,
) -> SynthHalfUNet2D:
    """Train a SynthHalfUNet2D to reproduce the scattering covariance of a target.

    The network is over-fitted to a single target (no generalisation).  Once
    trained, drawing a new z ~ N(0,1) and calling ``model(z)`` generates a new
    texture sample in a single forward pass.

    Parameters
    ----------
    target_image : torch.Tensor, shape [1, H, W] or [H, W]
        The reference texture whose scattering covariance we want to match.
    scat_op : FoCUS
        Initialised FOSCAT operator (``use_2D=True``).
    Jmax : int
        Number of upsampling levels in the U-Net decoder.
    n_samples : int
        Noise batch size per training step.  More samples → more stable
        gradient, more memory.
    channel_list : list[int] or None
        Feature channels per decoder level, ordered **coarsest → finest**.
        Default: ``[min(32·2^(Jmax-j), 256) for j in 0..Jmax]``.
    out_channels : int
        Output channels per image (1 for single-channel textures).
    lr : float
        Initial Adam learning rate.
    n_epochs : int
        Number of training epochs.
    eval_frequency : int
        Print loss every this many epochs.
    norm : str
        Normalisation passed to ``scat_op.eval`` (e.g. ``'auto'``).
    Jmax_scat : int or None
        Maximum wavelet scale for the scattering-covariance loss.  ``None``
        uses all scales available in ``scat_op``.  **Independent of the U-Net
        depth** ``Jmax``.
    edge : bool
        If ``True``, pass ``edge=True`` to ``scat_op.eval`` to compute
        statistics on edge pixels as well.  Must match how the target
        statistics are used in the rest of the pipeline.
    iso_ang : bool
        If ``True``, apply :meth:`~foscat.scat_cov.scat_cov.iso_mean` to the
        scattering covariance after ``eval``, collapsing the orientation axes
        to a single isotropic mean.  Cannot be combined with ``fft_ang``.
    fft_ang : bool
        If ``True``, apply :meth:`~foscat.scat_cov.scat_cov.fft_ang` to the
        scattering covariance after ``eval``, projecting the orientation axes
        onto the first ``fft_nharm`` Fourier harmonics.  Cannot be combined
        with ``iso_ang``.
    fft_nharm : int
        Number of harmonics kept by ``fft_ang`` (beyond the DC term).
        Default 1.  Ignored when ``fft_ang=False``.
    fft_imaginary : bool
        If ``True`` (default), keep both cosine and sine components in
        ``fft_ang``, giving rotation-invariant amplitudes.
        Ignored when ``fft_ang=False``.
    microcanonical : bool
        If ``True`` (default), use the **microcanonical loss**
        :func:`_microcanonical_loss`: the batch mean of statistics must match
        the target, normalised by the batch variance.  Mode collapse is
        naturally penalised because variance → 0 makes the loss diverge.
        If ``False``, use the classical per-sample distance (each generated
        image must independently match the target) averaged over the batch.
        Requires ``n_samples >= 2``.
    micro_eps : float
        Variance floor for the microcanonical loss (prevents division by
        exactly zero at the very start of training).  Default ``1e-6``.
    device : str or None
        ``'cuda'``, ``'cpu'``, or None (auto-detect).

    Returns
    -------
    SynthHalfUNet2D
        The trained network (in eval mode).

    Examples
    --------
    >>> model = train_synth_unet(
    ...     target, scat_op, Jmax=4,
    ...     edge=True,          # same edge handling as scat_op.synthesis
    ...     iso_ang=True,       # isotropic loss (fewer statistics)
    ...     n_epochs=2000,
    ... )

    >>> model = train_synth_unet(
    ...     target, scat_op, Jmax=4,
    ...     fft_ang=True,       # orientation-aware Fourier loss
    ...     fft_nharm=1,
    ...     fft_imaginary=True,
    ...     n_epochs=3000,
    ... )
    """
    if iso_ang and fft_ang:
        raise ValueError("iso_ang and fft_ang are mutually exclusive.")
    if microcanonical and n_samples < 2:
        raise ValueError(
            "microcanonical loss requires n_samples >= 2 "
            "(variance needs at least two data points)."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Helper: reduce statistics after eval -------------------------- #
    def _reduce(sc):
        """Apply iso_ang / fft_ang reduction (or none) to a scat_cov."""
        if iso_ang:
            return sc.iso_mean()
        if fft_ang:
            return sc.fft_ang(nharm=fft_nharm, imaginary=fft_imaginary)
        return sc

    # ---- Prepare target image ----------------------------------------- #
    if target_image.dim() == 2:
        target_image = target_image.unsqueeze(0)   # [1, H, W]
    H, W = target_image.shape[-2], target_image.shape[-1]
    target_image = target_image.to(device)

    # ---- Compute target scattering covariance (once) ------------------- #
    with torch.no_grad():
        target_sc = _reduce(
            scat_op.eval(target_image, Jmax=Jmax_scat, norm=norm, edge=edge)
        )

    # ---- Build model --------------------------------------------------- #
    model = SynthHalfUNet2D(
        scat_op,
        Jmax=Jmax,
        channel_list=channel_list,
        out_channels=out_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # ---- Training loop ------------------------------------------------- #
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        z = torch.randn(n_samples, 1, H, W, device=device)
        x_synth = model(z)                          # [N, out_ch, H, W]
        x_input = x_synth[:, 0, :, :]              # [N, H, W]

        synth_sc = _reduce(
            scat_op.eval(x_input, Jmax=Jmax_scat, norm=norm, edge=edge)
        )

        if microcanonical:
            # Microcanonical loss: (mean_k - target_k)^2 / var_k
            # — naturally penalises mode collapse (var → 0 → loss diverges)
            loss = _microcanonical_loss(synth_sc, target_sc, eps=micro_eps)
        else:
            # Classical: each sample independently matches the target
            loss = scat_op.reduce_distance(synth_sc, target_sc) / n_samples

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % eval_frequency == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{n_epochs}  loss={loss.item():.6f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Convenience: generate samples from a trained model
# ---------------------------------------------------------------------------

def generate_samples(
    model: SynthHalfUNet2D,
    n_samples: int,
    H: int,
    W: int,
    device: str | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Draw n_samples textures from a trained SynthHalfUNet2D.

    Parameters
    ----------
    model : SynthHalfUNet2D
        A trained network (output of :func:`train_synth_unet`).
    n_samples : int
        Number of independent textures to generate.
    H, W : int
        Spatial dimensions of the output.
    device : str or None
        Target device.  Defaults to the device of the model parameters.
    seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    torch.Tensor, shape [n_samples, out_channels, H, W]
    """
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 1, H, W, device=device)
        return model(z)
