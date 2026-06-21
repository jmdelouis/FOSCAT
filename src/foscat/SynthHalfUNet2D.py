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
        Number of wavelet scales (decoder depth).
    n_samples : int
        Number of independent synthesis samples generated per training step
        (= batch size of the noise z).  More samples → more stable gradient,
        but more memory.
    channel_list : list[int] or None
        Feature channels per decoder level, ordered **coarsest → finest**
        (same convention as :class:`SynthHalfUNet2D`).
        Default: ``[min(32·2^(Jmax-j), 256) for j in 0..Jmax]``.
    out_channels : int
        Output channels of the network (1 for single-channel textures).
    lr : float
        Learning rate for Adam.
    n_epochs : int
        Number of training epochs.
    eval_frequency : int
        Print loss every this many epochs.
    norm : str
        Normalisation passed to ``scat_op.eval``.
    Jmax_scat : int or None
        Maximum wavelet scale used when computing scattering covariances
        (passed as ``Jmax`` to ``scat_op.eval``).  ``None`` (default) lets
        FOSCAT use all scales available in ``scat_op`` — recommended.
        **Note**: this is independent of the U-Net depth ``Jmax``, which
        controls how many upsampling levels the decoder has.
    device : str or None
        ``'cuda'``, ``'cpu'``, or None (auto-detect).

    Returns
    -------
    SynthHalfUNet2D
        The trained network (in eval mode).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Jmax_scat=None → FOSCAT uses all scales available in scat_op.
    # Do NOT default to the U-Net Jmax: the two are independent parameters.

    # ---- Prepare target image ----------------------------------------- #
    if target_image.dim() == 2:
        target_image = target_image.unsqueeze(0)   # [1, H, W]
    H, W = target_image.shape[-2], target_image.shape[-1]
    target_image = target_image.to(device)

    # ---- Compute target scattering covariance -------------------------- #
    with torch.no_grad():
        # scat_op.eval expects [batch, H, W] (2D mode)
        target_sc = scat_op.eval(
            target_image,
            Jmax=Jmax_scat,   # None → use all available scales
            norm=norm,
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

        # Sample fresh white noise each epoch
        z = torch.randn(n_samples, 1, H, W, device=device)

        # Forward pass → [N, out_channels, H, W]
        x_synth = model(z)   # [N, 1, H, W]

        # Compute scattering covariance of all N synthesised maps
        # scat_op.eval expects [batch, H, W]
        x_input = x_synth[:, 0, :, :]   # [N, H, W]
        synth_sc = scat_op.eval(x_input, Jmax=Jmax_scat, norm=norm)

        # Loss: mean scat-cov distance across the N samples
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
