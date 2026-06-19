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

| Attribute | Shape (HEALPix, batch=B) | Description |
|-----------|--------------------------|-------------|
| `S0` | `(B, nscale)` | Mean wavelet power per scale |
| `S1` | `(B, nscale, NORIENT)` | First-order modulus mean per scale and orientation |
| `S2` | `(B, nscale, NORIENT, nscale2, NORIENT)` | Second-order scattering mean (`nscale2 < nscale`) |
| `S3` | `(B, nscale, NORIENT, nscale2, NORIENT)` | Cross-scale covariance (complex) |
| `S3P` | `(B, nscale, NORIENT, NORIENT)` | Cross-orientation covariance (complex) |
| `S4` | `(B, nscale, NORIENT, nscale2, NORIENT, nscale2, NORIENT)` | Second-order covariance (complex) |
| `numel` | int | Total number of real-valued coefficients after flattening |

### Arithmetic

`scat_cov` objects support element-wise arithmetic:

```python
diff = stat_a - stat_b      # difference
sq   = diff ** 2            # element-wise square
loss = sq.reduce_mean_batch(sq)  # scalar per batch element → mean over coefficients
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
