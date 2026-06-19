# `synthesis` — Complete parameter reference

**Module:** `foscat.scat_cov`, `foscat.scat_cov2D`, `foscat.scat_cov1D`

`synthesis` is the high-level entry point for field synthesis, denoising, and
texture generation. It wraps the full L-BFGS-B optimisation loop and handles
multi-resolution scheduling automatically.

```python
result = scat_op.synthesis(
    image_target,
    reference     = None,
    nstep         = 4,
    seed          = 1234,
    Jmax          = None,
    edge          = False,
    to_gaussian   = False,
    use_variance  = True,
    synthesised_N = 1,
    input_image   = None,
    grd_mask      = None,
    in_mask       = None,
    iso_ang       = False,
    fft_ang       = False,
    fft_nharm     = 1,
    fft_imaginary = True,
    EVAL_FREQUENCY= 100,
    NUM_EPOCHS    = 300,
    scat_cov_method = 'eval',
    n_up          = 0,
)
```

---

## How synthesis works

The optimiser searches for a field $u$ whose scattering-covariance statistics
$\Phi(u)$ match those of `image_target`:

$$u^* = \arg\min_u \; \mathcal{L}(u), \qquad
  \mathcal{L}(u) = \sum_k \frac{(\Phi(u)_k - \Phi(d)_k)^2}{\sigma_k^2}$$

where $\sigma_k$ is the per-coefficient variance of `image_target` (when
`use_variance=True`). Gradients are computed via PyTorch autograd and the
minimiser is `scipy.optimize.fmin_l_bfgs_b`.

The multi-resolution schedule (`nstep`) runs this minimisation from coarse to
fine resolution, using each step's result as the warm start for the next.

---

## Parameters

### `image_target` *(required)*

The reference field whose statistics are to be reproduced.

- **HEALPix:** ndarray shape `(npix,)` or `(B, npix)` with `npix = 12·nside²`
- **2D:** ndarray shape `(H, W)` or `(B, H, W)`
- **1D:** ndarray shape `(N,)` or `(B, N)`

The input should be normalised (zero mean, unit variance is standard):

```python
xnorm = (image - np.mean(image)) / np.std(image)
result = scat_op.synthesis(xnorm, ...)
```

---

### `nstep` *(default: 4)*

Number of resolution levels in the multi-resolution optimisation.

The algorithm builds a resolution pyramid by repeated 2× downsampling of
`image_target`. It then synthesises from coarsest to finest, using each result
as the initial condition for the next level.

| `nstep` | Resolutions visited (2D example, 256×256 target) |
|---------|--------------------------------------------------|
| 1 | 256×256 only (direct synthesis, slow convergence) |
| 2 | 128×128 → 256×256 |
| 3 | 64×64 → 128×128 → 256×256 |
| 4 | 32×32 → 64×64 → 128×128 → 256×256 |

**Capped automatically** if `nstep > jmax - 1` (the map is too small for that
many downsampling steps). For a 256×256 2D map, the maximum useful `nstep` is
about 6.

```python
# Multiscale synthesis — recommended for maps larger than ~64×64
result = scat_op.synthesis(xnorm, nstep=3, NUM_EPOCHS=300)

# Single-scale (faster, but worse convergence at large resolutions)
result = scat_op.synthesis(xnorm, nstep=1, NUM_EPOCHS=1000)
```

---

### `n_up` *(default: 0)*

Number of **extra upsampling steps** beyond the target size, keeping the same
`Jmax` (same wavelet scales).

With `n_up=1`, after completing synthesis at N×N, the algorithm continues at
2N×2N. The scattering statistics used are still those of the N×N target — only
the domain is larger. The result is a field that locally matches the statistics
of the original target, embedded in a larger canvas.

| `n_up` | Output size (2D, N×N target) | Comment |
|--------|------------------------------|---------|
| 0 | N×N | Standard synthesis |
| 1 | 2N×2N | Same statistics, larger domain |
| 2 | 4N×4N | Two extra levels |

The `Jmax` used during n_up steps is pinned to the value effective for the N×N
target (computed from its size and `KERNELSZ`), so wavelet filters and the norm
cache (`P1_dic`) remain consistent.

```python
# Synthesise a 512×512 map whose statistics match a 256×256 target
result = scat_op.synthesis(xnorm_256, nstep=3, n_up=1, NUM_EPOCHS=300)
# result.shape == (512, 512)
```

---

### `NUM_EPOCHS` *(default: 300)*

Maximum number of L-BFGS-B iterations **per resolution level**.

The optimiser may stop earlier if the convergence criterion (`factr`) is reached.
A good rule of thumb: convergence is typically achieved in 100–500 iterations
per level for moderate-size maps.

The total wall-clock time scales as `nstep × NUM_EPOCHS`.

---

### `EVAL_FREQUENCY` *(default: 100)*

Print the current loss value every N L-BFGS-B iterations.

---

### `seed` *(default: 1234)*

Random seed for the Gaussian white noise used as the initial condition at the
coarsest resolution level (`k=0`).

Change `seed` to generate statistically independent realisations of the same
target:

```python
results = [scat_op.synthesis(xnorm, seed=s, nstep=3) for s in range(10)]
```

Has no effect when `input_image` is provided (the initial condition is then
taken from `input_image`).

---

### `synthesised_N` *(default: 1)*

Number of independent synthetic maps to produce in a single run (batch
synthesis).

All `synthesised_N` maps are optimised simultaneously with the same loss, which
amortises the per-iteration cost. The output has an extra leading dimension:

```python
result = scat_op.synthesis(xnorm, synthesised_N=4, nstep=3)
# result.shape == (4, H, W)  for 2D
```

When `synthesised_N > 1` the reference statistics are computed as the
batch-mean of `image_target` (if batched) or from the single target map
replicated for all `synthesised_N` maps.

---

### `use_variance` *(default: True)*

Controls the loss weighting.

- **`True` (recommended):** each coefficient is divided by its standard deviation
  estimated from `image_target`. This makes the loss scale-invariant across
  spatial scales — large-scale and small-scale coefficients contribute equally
  regardless of their dynamic range.

  $$\mathcal{L}(u) = \sum_k \frac{(\Phi(u)_k - \Phi(d)_k)^2}{\sigma_k^2}$$

- **`False`:** all coefficients are weighted equally. Dominated by the
  largest-amplitude statistics (usually the coarsest scales). Useful if you
  want to enforce exact coefficient values rather than normalised deviations.

---

### `Jmax` *(default: None)*

Maximum wavelet scale index included in the loss. When `None`, all scales
available for the map size are used.

Reduce `Jmax` to match only small-scale statistics (fast textures) while
leaving large-scale structure free:

```python
# Use only scales j < 4 — ignore large-scale correlations
result = scat_op.synthesis(xnorm, Jmax=4, nstep=3)
```

During the multi-resolution schedule, `Jmax` at level k is automatically
decremented by 1 for each coarser level: `Jmax_k = Jmax - (nstep - 1 - k)`.

---

### `edge` *(default: False)*

Whether the map has non-periodic boundaries (limited spatial domain).

- **`False`:** assumes periodic / full-sphere boundary conditions. The
  convolutions wrap around the edges.
- **`True`:** activates an internal edge mask that down-weights pixels near the
  boundary at each resolution level. Use for rectangular images or
  partial-sky HEALPix patches where the field does not wrap.

Automatically set to `True` when `in_mask` is provided.

---

### `in_mask` *(default: None)*

A binary (or soft) mask that marks **invalid pixels in the input data** —
pixels that should not contribute to the reference statistics.

Shape must match `image_target`. Pixels with `in_mask = 0` are excluded from
the scattering-covariance computation at every scale level (the mask is
downsampled at each coarser level). Setting `in_mask` also enables `edge=True`
internally.

Use case: partial-sky CMB analysis, images with missing regions, survey
footprints.

```python
mask = np.ones_like(image_target)
mask[bad_pixels] = 0.0
result = scat_op.synthesis(xnorm, in_mask=mask, NUM_EPOCHS=300)
```

---

### `grd_mask` *(default: None)*

A binary mask controlling **which pixels of the synthesised map are free to
move**.

Pixels where `grd_mask = 0` are frozen to their values in `image_target`;
only pixels where `grd_mask = 1` are updated by the optimiser.

Use case: inpainting (reconstruct missing regions while preserving observed
pixels), or constrained synthesis where part of the field is fixed.

```python
grd_mask = np.zeros_like(image_target)
grd_mask[missing_pixels] = 1.0   # only synthesise these pixels
result = scat_op.synthesis(xnorm, grd_mask=grd_mask, NUM_EPOCHS=500)
```

At each multi-resolution level the gradient mask is downsampled to match the
current resolution.

---

### `reference` *(default: None)*

A second reference field for **cross-scattering-covariance** synthesis.

When provided, the loss matches the cross-statistics between the synthesised
map and a fixed second field (rather than the auto-statistics of
`image_target`):

$$\mathcal{L}(u) = \sum_k (\Phi_\text{cross}(u,\, d_2)_k - \Phi_\text{cross}(d_1,\, d_2)_k)^2$$

where $d_1$ = `image_target`, $d_2$ = `reference`.

Use case: component separation, where the synthesised field must match a
specific covariance structure with a known companion field.

```python
result = scat_op.synthesis(
    cmb_estimate,
    reference=dust_template,
    NUM_EPOCHS=300,
)
```

---

### `iso_ang` *(default: False)*

Whether to use **isotropically averaged** statistics in the loss.

- **`False`:** the loss is computed on the full oriented statistics (S1, S2,
  S3, S3P, S4 with all orientation indices). The synthesised map can reproduce
  anisotropic structures (filaments, oriented textures).

- **`True`:** the statistics are collapsed to their rotationally invariant
  content via `iso_mean()` before computing the loss:

  | Stat | Shape before iso | Shape after iso | Reduction |
  |------|-----------------|-----------------|-----------|
  | S1, S2 | `(..., L)` | `(...)` | simple mean over `L` |
  | S3, S3P | `(..., L, L)` | `(..., L)` | mean over `l1` at fixed `l2-l1` |
  | S4 | `(..., L, L, L)` | `(..., L, L)` | mean over `l1` at fixed `(l2-l1, l3-l1)` |

  Use for fields that are statistically isotropic (no preferred direction).
  Reduces the number of constraints and accelerates convergence.

```python
# Isotropic synthesis — suitable for CMB-like fields
result = scat_op.synthesis(xnorm, iso_ang=True, nstep=3)
```

```{note}
`iso_ang` and `fft_ang` should not be used together. `iso_ang` is the harder
reduction (mean only); `fft_ang` keeps angular variation and is the recommended
soft alternative.
```

---

### `fft_ang` *(default: False)* / `fft_nharm` *(default: 1)* / `fft_imaginary` *(default: True)*

A **softer alternative to `iso_ang`** that compresses the orientation axes to
their first Fourier harmonics instead of collapsing them to a single mean.
The three parameters work together:

| Parameter | Type | Default | Role |
|-----------|------|---------|------|
| `fft_ang` | bool | `False` | Enable Fourier angular compression |
| `fft_nharm` | int | `1` | Number of harmonics to keep beyond DC |
| `fft_imaginary` | bool | `True` | Keep both cos and sin components |

#### What `fft_ang=True` keeps (with `fft_nharm=1, fft_imaginary=True`)

For each orientation axis L, three coefficients are computed:

| Output index | Content | Physical meaning |
|---|---|---|
| `[…, 0]` | DC = $\frac{1}{L}\sum_l S[l]$ | Same as `iso_ang` — global mean |
| `[…, 1]` | $\sum_l \cos\!\bigl(\tfrac{2\pi l}{L}\bigr)\cdot S[l]$ | In-phase first harmonic |
| `[…, 2]` | $\sum_l \sin\!\bigl(\tfrac{2\pi l}{L}\bigr)\cdot S[l]$ | Quadrature first harmonic |

The **amplitude** $A_1 = \sqrt{c_1^2 + s_1^2}$ measures the strength of the
dominant angular variation and is **rotation-invariant** (independent of the
absolute orientation of the image). This is why `fft_imaginary=True` is
strongly recommended: with only the cosine (`fft_imaginary=False`) a field
oriented at 90° gives $c_1 \approx 0$ even when strongly anisotropic.

#### Shapes after `fft_ang(nharm=1, imaginary=True)` → `nout = 3`

| Statistic | Before | After | Factor |
|-----------|--------|-------|--------|
| S1, S2 | `(…, L)` | `(…, 3)` | ×3/L |
| S3, S3P | `(…, L, L)` | `(…, 3, 3)` | ×9/L² |
| S4 | `(…, L, L, L)` | `(…, 3, 3, 3)` | ×27/L³ |

For S3/S4 the projection is the **tensor product** of independent 1D Fourier
projections on each orientation axis. For S3 with L=4 this reduces from 16 to 9
orientation coefficients per scale pair, keeping the full angular power.

#### `fft_ang` vs `iso_ang` — choosing the right reduction

| | `iso_ang=True` | `fft_ang=True` |
|---|---|---|
| S1/S2 output | scalar (1 value) | 3 values (DC + cos + sin) |
| Angular information kept | mean only | mean + amplitude + phase of dominant mode |
| Number of constraints | minimum | moderate (×3 per axis vs ×1) |
| Suited for | strictly isotropic fields | fields with preferred but variable orientation |
| Cost per iteration | lowest | slightly higher |

```python
# fft_ang synthesis — preserves angular variation amplitude
result = scat_op.synthesis(xnorm, fft_ang=True, nstep=3, NUM_EPOCHS=300)

# Keep two harmonics (DC + first two angular modes):
result = scat_op.synthesis(xnorm, fft_ang=True, fft_nharm=2, nstep=3)

# With fft_imaginary=False only if the image orientation is known and fixed:
result = scat_op.synthesis(xnorm, fft_ang=True, fft_imaginary=False, nstep=3)
```

---

### `to_gaussian` *(default: False)*

Apply a Gaussianisation transform to `image_target` before synthesis, then
invert it at the end.

- **`True`:** maps the histogram of `image_target` to a standard Gaussian
  before computing target statistics. The final output is mapped back to the
  original histogram. Useful for highly non-Gaussian fields (log-normal dust
  emission, etc.) where the scattering statistics alone do not fully constrain
  the one-point distribution.

- **`False`:** the histogram of the synthesised map is controlled only by the
  scattering statistics (which capture some non-Gaussian content but are not
  guaranteed to reproduce the full marginal distribution).

---

### `input_image` *(default: None)*

A warm-start initial condition for the coarsest resolution level.

When `None`, the optimiser starts from Gaussian white noise (seeded by `seed`).
When provided, the field `input_image` is downsampled to each resolution level
and used as the initial guess at `k=0`.

Use case: iterative refinement, re-synthesis starting from a previous result,
or injecting a prior.

```python
# Refine a previous result
result_v1 = scat_op.synthesis(xnorm, nstep=3, NUM_EPOCHS=100)
result_v2 = scat_op.synthesis(xnorm, nstep=3, NUM_EPOCHS=500,
                               input_image=result_v1)
```

---

### `scat_cov_method` *(default: `'eval'`)*

Internal method used to compute scattering covariances.

- **`'eval'`** *(recommended):* uses `funct.eval()` with `norm='auto'`.
  The normalisation is cached after the first call (`clean_norm()`), which
  makes subsequent iterations fast. Works for all geometry types.

- Any other value: uses the legacy `scattering_cov()` path (2D only). Kept
  for backward compatibility; in practice `'eval'` is always preferred.

---

## Return value

- If `synthesised_N == 1`: ndarray with the same shape as `image_target`
  (the leading batch dimension is squeezed).
- If `synthesised_N > 1`: ndarray with a leading dimension of size
  `synthesised_N`.
- If `n_up > 0` (2D only): the output has shape `(2^n_up · H, 2^n_up · W)`.

---

## Complete examples

### Minimal 2D synthesis

```python
import foscat.scat_cov2D as sc
import numpy as np

scat_op = sc.funct(NORIENT=4)
xnorm = (image - np.mean(image)) / np.std(image)

result = scat_op.synthesis(xnorm, seed=10, nstep=3, NUM_EPOCHS=300)
```

### Batch synthesis with masking

```python
mask = np.ones_like(xnorm)
mask[invalid] = 0.0

results = scat_op.synthesis(
    xnorm,
    in_mask       = mask,
    synthesised_N = 4,
    nstep         = 3,
    iso_ang       = True,
    NUM_EPOCHS    = 500,
)
# results.shape == (4, H, W)
```

### Upsampled synthesis (n_up)

```python
# Synthesise a 512×512 map matching the statistics of a 256×256 target
result = scat_op.synthesis(
    xnorm_256,
    nstep      = 3,
    n_up       = 1,      # one extra 2× upsampling after full resolution
    NUM_EPOCHS = 300,
)
# result.shape == (512, 512)
```

### Inpainting (frozen observed pixels)

```python
grd_mask = np.zeros_like(xnorm)
grd_mask[hole_pixels] = 1.0   # only these pixels move

result = scat_op.synthesis(
    xnorm,
    grd_mask   = grd_mask,
    nstep      = 3,
    edge       = True,
    NUM_EPOCHS = 500,
)
```

### Cross-covariance synthesis (component separation)

```python
# Synthesise CMB-like field matching cross-statistics with a dust template
result = scat_op.synthesis(
    cmb_estimate,
    reference  = dust_template,
    nstep      = 3,
    iso_ang    = True,
    NUM_EPOCHS = 300,
)
```

### Isotropic Gaussianised synthesis

```python
result = scat_op.synthesis(
    image_target,
    to_gaussian = True,
    iso_ang     = True,
    nstep       = 4,
    NUM_EPOCHS  = 500,
)
```
