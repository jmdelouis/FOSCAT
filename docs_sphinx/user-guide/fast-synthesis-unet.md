# Fast Synthesis with a Decoder U-Net

The classical synthesis workflow in FOSCAT uses **L-BFGS-B gradient descent**:
starting from Gaussian noise, the optimizer iteratively adjusts pixel values
until the scattering-covariance statistics of the synthesised image match the
target. This produces excellent quality but can take minutes per image —
and every new sample requires restarting the optimisation.

`SynthHalfUNet2D` offers a complementary approach: train a small neural network
once (a few thousand epochs), then generate as many new samples as needed with a
**single forward pass** (milliseconds).

---

## Core idea

Instead of searching for one good image, we train a **generative network** that
maps white noise to texture. The network is over-fitted to a single target
scattering covariance: after training, any Gaussian noise vector $z$ fed to the
network produces a new image whose statistics match the target.

The key architectural choice is how to make the network **multi-scale** without
an encoder. The solution: the skip connections that normally carry encoder
features are replaced by **oriented wavelet responses of the input noise** — the
same FOSCAT wavelets used by `FoCUS`, applied to progressively downsampled
versions of $z$.

---

## Architecture: decoder-only U-Net

```
z ~ N(0,1)   [N, 1, H, W]
│
├─ Wavelet bank (frozen FOSCAT kernels)
│   ├── ψ_l ★ z          → skip[0]  [N, 2L, H,      W     ]  finest
│   ├── ψ_l ★ Pool(z)    → skip[1]  [N, 2L, H/2,    W/2   ]
│   ├── ψ_l ★ Pool²(z)   → skip[2]  [N, 2L, H/4,    W/4   ]
│   └── ψ_l ★ Pool^J(z)  → skip[J]  [N, 2L, H/2^J,  W/2^J ]  coarsest
│
│  Decoder  (parameters learned during training)
│
├── InitBlock( cat[skip[J], Pool^J(z)] )  →  [N, C[0], H/2^J, W/2^J]
│                                              ↑ 2L+1 input channels
├── Upsample ×2  →  cat(skip[J-1])  →  ConvBlock  →  [N, C[1], H/2^(J-1), W/2^(J-1)]
├── Upsample ×2  →  cat(skip[J-2])  →  ConvBlock  →  [N, C[2], H/2^(J-2), W/2^(J-2)]
│   ...
├── Upsample ×2  →  cat(skip[0])   →  ConvBlock  →  [N, C[J], H,          W         ]
│
└── Conv 1×1  →  x_out   [N, out_ch, H, W]
```

**Wavelet skip connections.** At each scale $j$, the noise is downsampled $2^j$
times and then convolved with all $L$ oriented FOSCAT wavelets:

$$\text{skip}_j = \bigl[\operatorname{Re}(\psi_l \star z_j),\;
                         \operatorname{Im}(\psi_l \star z_j)\bigr]_{l=0}^{L-1}
\qquad z_j = \operatorname{AvgPool}^j(z)$$

This gives $2L$ channels per scale (real and imaginary parts of each of the $L$
oriented wavelets). The real and imaginary parts together preserve the full
complex wavelet response, allowing the decoder to reconstruct both the amplitude
and phase of the angular modulation at each scale.

**Why not use the modulus** $|\psi_l \star z_j|$ as skip? The modulus discards
the phase of the wavelet response, limiting the diversity of generated samples.
Keeping both components lets the network exploit the full information in $z$.

**Low-frequency channel.** At the coarsest scale $J$, the spatially averaged
noise $z_J = \operatorname{AvgPool}^J(z)$ is concatenated alongside the wavelet
skips, providing a direct low-frequency anchor. The InitBlock input therefore has
$2L + 1$ channels.

**ConvBlock.** Each decoder block applies two rounds of:

$$\operatorname{Conv}_{3\times3} \to \operatorname{BatchNorm} \to \operatorname{LeakyReLU}(0.2)$$

No skip connections from an encoder; the wavelet responses of $z$ play that role.

---

## Training objective

The network $G_\theta$ is trained by minimising the scattering-covariance
distance between its outputs and the target statistics $\Phi^*$:

$$\mathcal{L}(\theta) =
  \frac{1}{N} \sum_{i=1}^{N}
  \mathrm{dist}\!\bigl(\Phi(G_\theta(z_i)),\; \Phi^*\bigr),
  \qquad z_i \sim \mathcal{N}(0, I)$$

where $\mathrm{dist}$ is `scat_op.reduce_distance` (sum of squared
differences) and $\Phi^*$ is computed once from the target image with
`scat_op.eval(..., norm='auto')`.

At each epoch, a fresh batch of $N$ noise vectors is drawn, ensuring the network
learns to map *any* noise to a valid texture rather than memorising one solution.
The optimizer is Adam with a cosine-annealing learning-rate schedule.

---

## Parameters

### `SynthHalfUNet2D`

| Parameter | Type | Description |
|-----------|------|-------------|
| `scat_op` | `FoCUS` | FOSCAT operator initialised with `use_2D=True`. Wavelet kernels are extracted once and frozen. |
| `Jmax` | `int` | Decoder depth (number of upsampling steps). The coarsest feature map has resolution $H/2^{J} \times W/2^{J}$. **Independent of the number of scales in `scat_op`.** |
| `channel_list` | `list[int]` or `None` | Feature-map channels per level, ordered **coarsest → finest**, length `Jmax+1`. Default: `[min(32·2^(J-j), 256) for j=0..J]`, e.g. `[256,128,64,32]` for `Jmax=3`. |
| `out_channels` | `int` | Output channels per image. Use `1` for single-channel textures; generate $N$ samples by setting the noise batch size to $N$. |

### `train_synth_unet`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_image` | — | Reference texture, shape `[H,W]` or `[1,H,W]`. |
| `scat_op` | — | FOSCAT 2D operator. |
| `Jmax` | — | U-Net decoder depth (see above). |
| `n_samples` | `4` | Noise batch size per training step. Larger → more stable gradient, more memory. |
| `channel_list` | `None` | See above. |
| `out_channels` | `1` | Output image channels. |
| `lr` | `1e-3` | Initial learning rate (Adam). |
| `n_epochs` | `2000` | Training epochs. |
| `eval_frequency` | `100` | Print loss every N epochs. |
| `norm` | `'auto'` | Normalisation passed to `scat_op.eval`. Must be the same at training and evaluation time. |
| `Jmax_scat` | `None` | Maximum scale for scattering-covariance computation. `None` uses all scales available in `scat_op` — recommended. **Do not confuse with the U-Net `Jmax`.** |
| `edge` | `False` | If `True`, pass `edge=True` to `scat_op.eval`. Controls whether edge pixels contribute to the statistics. Must match the convention used in the rest of the pipeline. |
| `iso_ang` | `False` | Apply `iso_mean()` after `eval`, collapsing orientation axes to a single isotropic mean. Reduces the number of loss terms and speeds up training. Mutually exclusive with `fft_ang`. |
| `fft_ang` | `False` | Apply `fft_ang()` after `eval`, projecting the orientation axes onto the first `fft_nharm` Fourier harmonics. Keeps orientation information in a compact form. Mutually exclusive with `iso_ang`. |
| `fft_nharm` | `1` | Number of harmonics beyond DC kept by `fft_ang`. Ignored when `fft_ang=False`. |
| `fft_imaginary` | `True` | If `True`, keep both cosine and sine components in `fft_ang` (rotation-invariant amplitudes). Ignored when `fft_ang=False`. |
| `device` | auto | `'cuda'` or `'cpu'`. Defaults to CUDA if available. |

### `generate_samples`

```python
samples = generate_samples(model, n_samples=16, H=256, W=256, seed=42)
# → torch.Tensor [16, 1, 256, 256]
```

| Parameter | Description |
|-----------|-------------|
| `model` | Trained `SynthHalfUNet2D` (output of `train_synth_unet`). |
| `n_samples` | Number of independent textures to generate. |
| `H`, `W` | Spatial dimensions (must match the training image). |
| `seed` | Optional random seed for reproducibility. |

---

## Channel layout

`channel_list` is indexed **from coarsest (index 0) to finest (last index)**,
one entry per decoder level:

```
Jmax = 3, image 256×256, channel_list = [256, 128, 64, 32]

 Level  Scale  Resolution   channel_list[k]
 ─────  ─────  ──────────   ───────────────
   0    j=3    32  × 32     256   ← InitBlock output
   1    j=2    64  × 64     128
   2    j=1    128 × 128     64
   3    j=0    256 × 256     32   ← last decoder block, feeds Conv1×1
```

A **uniform layout** (same channels at all scales) is also valid and sometimes
easier to tune:

```python
channel_list = [64, 64, 64, 64]   # 64 channels everywhere, Jmax=3
```

---

## Two independent `Jmax` parameters

A common source of confusion: there are two separate depth parameters.

| Parameter | Controls | Who uses it |
|-----------|----------|-------------|
| U-Net `Jmax` (arg to `SynthHalfUNet2D` / `train_synth_unet`) | How many upsampling levels the decoder has. Determines the coarsest spatial resolution of the feature maps. | The network only. |
| `Jmax_scat` (arg to `train_synth_unet`) | How many wavelet scales are used when computing the scattering covariance loss. | `scat_op.eval` only. |

They are **independent**: you can have a deep decoder (`Jmax=5`) with a
conservative scattering loss (`Jmax_scat=3`), or vice versa.
Setting `Jmax_scat=None` (default) tells FOSCAT to use all scales it was
initialised with — the safest choice.

---

## Complete usage example

```python
import torch
import foscat.scat_cov2D as sc
from foscat.SynthHalfUNet2D import train_synth_unet, generate_samples

# 1. Create FOSCAT operator
scat_op = sc.funct(NORIENT=4, KERNELSZ=3, use_2D=True)

# 2. Load target image (numpy array → torch tensor)
import numpy as np
target = torch.tensor(np.load("my_texture.npy"), dtype=torch.float32)  # [H, W]

# 3. Train the decoder U-Net
#    - Jmax=4: 4 upsampling levels  (coarsest map = H/16 × W/16 for H=256)
#    - channel_list: 128 channels at coarsest, 8 at finest
#    - edge=True: include edge pixels in the statistics (match synthesis convention)
#    - fft_ang=True: orientation-aware Fourier loss (keeps DC = iso_mean + harmonics)
#    - Jmax_scat=None: use all FOSCAT scales
model = train_synth_unet(
    target,
    scat_op,
    Jmax=4,
    channel_list=[128, 64, 32, 16, 8],
    n_samples=8,
    lr=1e-3,
    n_epochs=3000,
    eval_frequency=200,
    edge=True,
    fft_ang=True,        # or iso_ang=True for isotropic loss
    fft_nharm=1,
    fft_imaginary=True,
)

# 4. Generate new samples instantly
samples = generate_samples(model, n_samples=16, H=256, W=256, seed=0)
# samples: torch.Tensor shape [16, 1, 256, 256]

# 5. Save / display
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0].cpu().numpy(), cmap="gray")
    ax.axis("off")
plt.suptitle("Synthesised textures (single forward pass each)")
plt.tight_layout()
plt.show()

# 6. Save the trained model for later
torch.save(model.state_dict(), "synth_unet_texture.pt")

# 7. Reload
model2 = SynthHalfUNet2D(scat_op, Jmax=4, channel_list=[128, 64, 32, 16, 8])
model2.load_state_dict(torch.load("synth_unet_texture.pt"))
model2.eval()
```

---

## Comparison with gradient-descent synthesis

| | `scat_op.synthesis()` | `SynthHalfUNet2D` |
|---|---|---|
| Method | L-BFGS-B gradient descent | Neural network (Adam) |
| Training cost | None | ~minutes (once per texture) |
| Inference cost per sample | Minutes | Milliseconds |
| Quality | Reference | Comparable after sufficient training |
| Multiple samples | Restart from scratch | Instant (single forward pass) |
| Memory | Low | Depends on `channel_list` |
| Geometry | 2D and HEALPix | 2D only (this module) |
| Orientation reduction | `iso_ang`, `fft_ang` | Not built-in (uses raw scat-cov) |

The two approaches are complementary: gradient descent is the gold standard for
a single high-quality synthesis; the U-Net is preferable whenever many
independent samples are needed quickly (Monte Carlo studies, uncertainty
estimation, data augmentation).
