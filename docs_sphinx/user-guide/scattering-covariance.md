# Scattering Covariance — FoCUS / scat_cov

**Modules:** `foscat.scat_cov` (HEALPix / spherical), `foscat.scat_cov2D` (2D planar), `foscat.scat_cov1D` (1D)

The primary entry point is the `funct` factory function in each module. It constructs
a `FoCUS` instance — the core operator that builds wavelet filters, manages per-scale
stencil tables, and evaluates scattering-covariance statistics.

---

## Quick start

```python
import foscat.scat_cov as sc        # HEALPix / spherical
import foscat.scat_cov2D as sc2d    # 2D planar

# Create an operator for HEALPix maps (nside up to nstep_max resolution levels)
scat_op = sc.funct(KERNELSZ=5, NORIENT=4, OSTEP=1, nstep_max=4, all_type='float64')

# Evaluate statistics on a full-sky map
import numpy as np
nside = 64
x = np.random.randn(12 * nside**2)
stat = scat_op.eval(x)

print("Number of descriptors:", stat.numel)
```

---

## `funct` — constructor (HEALPix)

```python
foscat.scat_cov.funct(
    NORIENT       = 4,
    LAMBDA        = 1.2,
    KERNELSZ      = 3,
    slope         = 1.0,
    all_type      = "float32",
    nstep_max     = 20,
    padding       = "SAME",
    gpupos        = 0,
    mask_thres    = None,
    mask_norm     = False,
    isMPI         = False,
    TEMPLATE_PATH = None,
    BACKEND       = "torch",
    use_2D        = False,
    use_1D        = False,
    return_data   = False,
    DODIV         = False,
    use_median    = False,
    InitWave      = None,
    silent        = True,
    mpi_size      = 1,
    mpi_rank      = 0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NORIENT` | int | 4 | Number of wavelet orientations. Orientations are uniformly spaced over $[0, \pi)$. Use 1 for an isotropic (orientation-averaged) operator. |
| `LAMBDA` | float | 1.2 | Frequency parameter of the complex Morlet wavelet. Higher values shift the wavelet towards higher spatial frequencies. |
| `KERNELSZ` | int | 3 | Side length $K$ of the local stencil ($K \times K$ neighbours per pixel). Common values: 3, 5. Larger values increase the per-layer receptive field. |
| `slope` | float | 1.0 | Activation slope applied to wavelet modulus (leaky-ReLU-like). Rarely needs changing. |
| `all_type` | str | `"float32"` | Numerical precision: `"float32"` or `"float64"`. Use `float64` for synthesis when high numerical accuracy matters. |
| `nstep_max` | int | 20 | Maximum number of resolution levels (HEALPix `nside` halvings). The operator stops when it reaches `nside=1` or `nstep_max` levels, whichever comes first. |
| `padding` | str | `"SAME"` | Padding mode for 2D operators: `"SAME"` or `"VALID"`. HEALPix operators always use neighbour tables; this parameter is relevant only in 2D mode. |
| `gpupos` | int | 0 | Index into the list of available CUDA devices. Wraps around the number of GPUs. |
| `mask_thres` | float\|None | None | If set, pixels with mask value below this threshold are excluded from statistics. |
| `mask_norm` | bool | False | Normalise statistics by the fraction of unmasked pixels at each scale. |
| `isMPI` | bool | False | Enable MPI-parallel mode (requires `mpi4py`). Each rank computes statistics on a subset of maps and results are reduced. |
| `TEMPLATE_PATH` | str\|None | None | Path for the wavelet stencil cache. Defaults to `~/.FOSCAT/data/`. |
| `BACKEND` | str | `"torch"` | Computation backend. Currently only `"torch"` is maintained. |
| `use_2D` | bool | False | If True, operate on 2D grids instead of HEALPix (equivalent to using `scat_cov2D.funct`). |
| `use_1D` | bool | False | If True, operate on 1D arrays (equivalent to using `scat_cov1D.funct`). |
| `return_data` | bool | False | If True, `eval()` returns raw intermediate arrays instead of a `scat_cov` statistics object. For advanced debugging. |
| `DODIV` | bool | False | Add two extra divergence-sensitive wavelet orientations (used for polarisation analysis). |
| `use_median` | bool | False | Use median instead of mean for spatial pooling. More robust to pixel outliers. |
| `InitWave` | array\|None | None | Override the initial wavelet kernels with a custom array. Shape `[NORIENT, KERNELSZ**2]`. |
| `silent` | bool | True | Suppress progress output during initialisation. |
| `mpi_size` | int | 1 | Total number of MPI ranks (set automatically when `isMPI=True`). |
| `mpi_rank` | int | 0 | Rank of this process (set automatically when `isMPI=True`). |

### Key attributes after construction

```python
scat_op.NORIENT      # int  — number of orientations
scat_op.KERNELSZ     # int  — kernel side length
scat_op.LAMBDA       # float — wavelet frequency parameter
scat_op.nstep_max    # int  — number of resolution levels
scat_op.all_type     # str  — dtype string
scat_op.BACKEND      # str  — "torch"
scat_op.backend      # BkTorch instance (low-level backend wrapper)
scat_op.ngpu         # int  — number of detected CUDA devices
scat_op.gpupos       # int  — active GPU index
```

---

## `eval` — computing statistics

```python
stat = scat_op.eval(x, mask=None, norm=None)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | ndarray or Tensor, shape `(npix,)` or `(batch, npix)` | Input field(s). For HEALPix, `npix = 12 * nside**2`. For batches, axis 0 is the batch dimension. |
| `mask` | ndarray, shape `(npix,)`, optional | Binary or soft mask. Pixels with mask = 0 are excluded. |
| `norm` | scat_cov object, optional | Normalisation statistics. If provided, each coefficient is divided by the corresponding reference value before returning. |

**Returns:** a `scat_cov` statistics object (see below).

### Batched evaluation

```python
# batch of 8 maps at nside=64
batch = np.random.randn(8, 12 * 64**2)
stat = scat_op.eval(batch)   # scat_cov object with batch dimension
```

---

## `scat_cov` — the statistics object

`eval()` returns a `scat_cov` instance containing the coefficient arrays:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `S0` | `(B, Nj)` | Mean wavelet power per scale |
| `S1` | `(B, Nj, L)` | First-order modulus mean — per scale `j` and orientation `l` |
| `S2` | `(B, Nj1, Nj2, L)` | Second-order scattering mean — pairs `(j1, j2)` with `j2 ≤ j1`, one orientation `l` |
| `S3` | `(B, Nj1, Nj2, L, L)` | Cross-scale covariance — pairs `(j1, j2)`, two orientations `(l1, l2)` |
| `S3P` | `(B, Nj1, Nj2, L, L)` | Cross-orientation covariance — same shape as S3 |
| `S4` | `(B, Nj1, Nj2, L, L, L)` | Second-order cross-scale covariance — scale triplets, three orientations `(l1, l2, l3)` |
| `numel` | int | Total number of real-valued coefficients after flattening |

`B` = batch size, `Nj*` = number of active scale (pairs/triplets), `L` = `NORIENT`.

### Arithmetic

`scat_cov` objects support element-wise arithmetic:

```python
diff = stat_a - stat_b      # difference
sq   = diff ** 2            # element-wise square
loss = sq.reduce_mean_batch(sq)  # scalar per batch element → mean over coefficients
```

### Isotropic angular averaging — `iso_mean` / `iso_ang`

For statistically isotropic fields, only the **relative** orientation between wavelet
pairs matters, not the absolute angle.  `iso_mean()` reduces the orientation axes
by averaging over all global rotations, keeping only rotationally-invariant
combinations.

```python
stat_iso = stat.iso_mean()          # reduce to isotropic descriptors
stat_full = stat.iso_mean(repeat=True)  # reduce then broadcast back to original shape
```

The reduction is different for each statistic:

**S1, S2** — shape `(..., L)` → `(...)`:

Simple mean over the single orientation axis:

$$S_1^\text{iso}[j] = \frac{1}{L}\sum_{l=0}^{L-1} S_1[j,\, l]$$

**S3, S3P** — shape `(..., L, L)` → `(..., L)`:

Only the angular difference $\Delta l = l_2 - l_1 \bmod L$ is invariant.
The output index is $\Delta l$:

$$S_3^\text{iso}[j_1,j_2,\,\Delta l] = \frac{1}{L}\sum_{l_1=0}^{L-1} S_3\!\left[j_1,j_2,\;l_1,\;(l_1+\Delta l)\bmod L\right]$$

The $L$ output values correspond to angular separations $\Delta l \cdot \pi/L \in \{0, \pi/L, \ldots, (L-1)\pi/L\}$.

**S4** — shape `(..., L, L, L)` → `(..., L, L)`:

S4 has three orientation indices $(l_1, l_2, l_3)$, one per scale in the triplet.
The two invariant quantities are both pairwise differences relative to $l_1$:

$$\Delta l_{12} = (l_2 - l_1)\bmod L, \qquad \Delta l_{13} = (l_3 - l_1)\bmod L$$

$$S_4^\text{iso}[j_1,j_2,j_3,\,\Delta l_{12},\,\Delta l_{13}]
= \frac{1}{L}\sum_{l_1=0}^{L-1}
  S_4\!\left[l_1,\;(l_1+\Delta l_{12})\bmod L,\;(l_1+\Delta l_{13})\bmod L\right]$$

Result shape: `(..., L, L)` — a $L\times L$ matrix of relative-angle pairs.
This is implemented via the `_iso_orient3` matrix in `BkBase.calc_iso_orient3`.

**Usage with `iso_ang=True` in synthesis:**

```python
# Compute statistics and immediately reduce to isotropic descriptors
stat = scat_op.eval(x, norm='auto')
stat_iso = stat.iso_mean()   # S1,S2 → (...), S3,S3P → (...,L), S4 → (...,L,L)

# Use iso_ang directly in synthesis:
result = scat_op.synthesis(xnorm, iso_ang=True, NUM_EPOCHS=300)
```

---

### Soft angular compression — `fft_ang`

`iso_mean` is a hard reduction: it collapses each orientation axis to a single
number (the mean), discarding all information about angular variation.
`fft_ang` is a softer alternative that keeps the first few Fourier harmonics
along each orientation axis, preserving the *amplitude* of the angular variation.

```python
stat_fft = stat.fft_ang(nharm=1, imaginary=True)
```

**What is kept (nharm=1):**

| Output index | Content |
|---|---|
| `[…, 0]` | DC — mean over orientations (identical to `iso_mean`) |
| `[…, 1]` | Cosine projection: $\sum_l \cos(2\pi l/L) \cdot S[l]$ |
| `[…, 2]` | Sine projection: $\sum_l \sin(2\pi l/L) \cdot S[l]$ |

**Shapes after `fft_ang(nharm=1, imaginary=True)`** (`nout = 3`):

| Statistic | Before | After |
|-----------|--------|-------|
| S1, S2 | `(..., L)` | `(..., 3)` |
| S3, S3P | `(..., L, L)` | `(..., 3, 3)` |
| S4 | `(..., L, L, L)` | `(..., 3, 3, 3)` |

For S3/S4 the projection is the **tensor product** of independent 1D Fourier
projections on each orientation axis.

**Why `imaginary=True` is essential for rotation invariance:**

With `imaginary=False` only the cosine component is kept. A field whose dominant
orientation sits at the zero-crossing of cosine (e.g. 90° for L=4) would give
a near-zero first-harmonic coefficient despite being strongly anisotropic.

With `imaginary=True` both cosine and sine are kept, so the **amplitude**

$$A_1 = \sqrt{c_1^2 + s_1^2}$$

is **rotation-invariant** regardless of the image orientation. This is the
recommended mode whenever results must not depend on the absolute rotation of
the input field.

```python
import numpy as np

stat_fft = stat.fft_ang(nharm=1, imaginary=True)

# Rotation-invariant angular amplitude for S2:
A1_S2 = np.sqrt(stat_fft.S2[..., 1]**2 + stat_fft.S2[..., 2]**2)
```

**Using `fft_ang` directly in `synthesis`:**

The `fft_ang` parameter is a first-class option of `synthesis`, exactly like `iso_ang`:

```python
# Soft angular compression — keeps DC + first harmonic amplitude (rotation-invariant)
result = scat_op.synthesis(xnorm, fft_ang=True, NUM_EPOCHS=300)

# Keep two harmonics:
result = scat_op.synthesis(xnorm, fft_ang=True, fft_nharm=2, NUM_EPOCHS=300)
```

`fft_ang=True` is applied to both the target statistics and the statistics evaluated on the current
candidate map at every optimisation step, so the loss is always comparing Fourier-compressed
statistics in a consistent space.

**Advanced: custom loss with `fft_ang` (manual control):**

```python
from foscat.Synthesis import Loss, Synthesis

def fft_loss(u, scat_op, args):
    ref_fft = args[0]
    learn = scat_op.eval(u, norm='auto').fft_ang(nharm=1, imaginary=True)
    return scat_op.reduce_distance(learn, ref_fft)

target_fft = scat_op.eval(xnorm, norm='auto').fft_ang(nharm=1, imaginary=True)
loss = Loss(fft_loss, scat_op, target_fft)
solver = Synthesis([loss])
result = solver.run(x0, NUM_EPOCHS=300)
```

### Serialisation

```python
stat.save("my_statistics")          # writes my_statistics.pkl
stat2 = foscat.scat_cov.read("my_statistics")
```

### Converting to NumPy

```python
stat_np = stat.numpy()   # returns a scat_cov where all arrays are numpy ndarray
```

### Flattening

```python
vec = stat.flattenMask()  # (B, numel) numpy array — all coefficients concatenated
```

---

## `ud_grade` — changing HEALPix resolution

```python
# Downsample the operator to a coarser nside (useful for multi-resolution workflows)
low_res_map = scat_op.ud_grade_2(x, nside_out=32)
```

---

## Masking

Pass a binary mask to exclude invalid pixels (survey boundaries, missing data):

```python
mask = np.ones(12 * nside**2)
mask[bad_pixels] = 0.0

stat = scat_op.eval(x, mask=mask)
```

With `mask_norm=True` the statistics are rescaled by the fraction of valid pixels
at each resolution level, so that partially masked statistics remain comparable to
full-sky ones.

---

## 2D operator

`foscat.scat_cov2D.funct` has the same signature as the HEALPix version. The
differences are:

- `x` has shape `(H, W)` or `(B, H, W)` — a regular image grid.
- `OSTEP` is not supported (all orientation pairs are evaluated).
- Resolution levels are obtained by 2× average pooling.

```python
import foscat.scat_cov2D as sc2d

scat_op = sc2d.funct(KERNELSZ=5, NORIENT=4, all_type='float64')
image = np.random.randn(256, 256)
stat = scat_op.eval(image)
```
