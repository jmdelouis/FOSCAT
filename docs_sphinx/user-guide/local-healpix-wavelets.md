# Local HEALPix Wavelets

FOSCAT supports processing on **incomplete or regional spherical domains** —
arbitrary subsets of HEALPix pixels. This is important for:

- Observations that do not cover the full sky (survey footprints, telescope pointings).
- Region-specific analysis (ocean basin, continent, atmospheric window).
- Training neural network models on a regional patch while preserving spherical geometry.

This workflow corresponds to `local_foscat.ipynb` and `CNN_local.ipynb` in
`demo-foscat-pangeo-eosc`.

---

## Why local HEALPix instead of planar projection?

Projecting a spherical region onto a flat grid introduces angular distortions that
grow with the size of the patch. FOSCAT's local operators keep all data and
computations on the sphere using the HEALPix NESTED neighbour tables, so spatial
statistics remain consistent with their spherical geometry across all scales.

---

## Defining a local domain

A local domain is specified by an array of HEALPix NESTED pixel indices at the
finest resolution (`in_nside`):

```python
import numpy as np
import healpy as hp

nside = 64
# select a disc around (lon=0°, lat=30°), radius 20°
centre_vec = hp.ang2vec(np.radians(60), np.radians(0))   # colatitude, longitude
cell_ids = hp.query_disc(nside, centre_vec, np.radians(20), nest=True)

print(f"Regional domain: {len(cell_ids)} pixels out of {12*nside**2}")
```

---

## Local scattering statistics

Pass `cell_ids` to restrict `eval` to the local domain:

```python
import foscat.scat_cov as sc

scat_op = sc.funct(KERNELSZ=5, NORIENT=4, nstep_max=3, all_type='float64')

# full-sky map
full_map = np.load("my_map.npy")

# evaluate statistics only on the regional subset
local_data = full_map[cell_ids]    # shape (N_cells,)

# the operator handles local neighbour lookup automatically
local_stat = scat_op.eval_local(local_data, cell_ids, nside)
```

:::{note}
The internal neighbour stencil is constructed from the full HEALPix geometry.
Pixels at the boundary of the domain receive contributions from their full-sphere
neighbours; if a neighbour falls outside `cell_ids`, it is treated as zero (or as
the masked value if `mask_thres` is set).
:::

---

## Local U-Net with `HealpixUNet`

`HealpixUNet` natively supports regional domains via the `cell_ids` parameter:

```python
from foscat.healpix_unet_torch import HealpixUNet

model = HealpixUNet(
    in_nside   = 64,
    n_chan_in  = 1,
    chanlist   = [16, 32, 64],
    cell_ids   = cell_ids,       # ← regional domain
    KERNELSZ   = 3,
    task       = 'regression',
    out_channels = 1,
)

# Input shape: (B, C, N_cells) — not the full K = 12*nside²
x_local = full_map[cell_ids][np.newaxis, np.newaxis, :]   # (1, 1, N_cells)
y_pred  = model(torch.tensor(x_local).float())
```

The model scatters regional pixels to a full-sphere buffer, performs spherical
convolutions, then gathers the result back to the regional subset. Boundary pixels
receive zero-padded contributions from outside the domain, which is acceptable for
most analysis tasks.

---

## Multi-resolution downsampling on a local domain

HEALPix NESTED ordering makes multi-resolution local analysis straightforward:
pixel `p` at `nside` has children `{4p, 4p+1, 4p+2, 4p+3}` at `2*nside`.
Coarsening a local domain is therefore a simple index operation:

```python
# coarsen cell_ids from nside=64 to nside=32
cell_ids_32 = np.unique(cell_ids // 4)
```

This preserves the exact spatial coverage of the domain at every resolution level.

---

## Practical notes

**Consistent ordering.** Keep `nest=True` throughout. Mixing RING and NESTED
orderings silently produces wrong neighbour lookups.

**Boundary padding.** The local stencil sets neighbours outside `cell_ids` to
zero. For pixels near the domain boundary, the wavelet response is therefore
influenced by this zero-padding. For quantitative studies, expand the domain by
one stencil radius ($\approx \lfloor K/2 \rfloor$ pixel spacings at the finest level)
to create a guard band, and restrict the statistics computation to the inner core.

**Scale compatibility.** After downsampling, verify that `nside / 2^k ≥ 1` for
all levels `k ≤ nstep_max`. The operator stops automatically when `nside=1` but
raises an error if the coarsened domain becomes empty.

**Missing data vs physical zeros.** Use `mask` to distinguish missing pixels
(excluded from statistics) from physical zero-value pixels (included). The
`mask_thres` parameter in `FoCUS` sets the threshold below which pixels are
considered invalid.
