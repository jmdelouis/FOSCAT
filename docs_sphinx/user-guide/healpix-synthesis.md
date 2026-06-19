# HEALPix Synthesis

Synthesis on HEALPix maps creates a full-sky spherical field whose
scattering-covariance statistics match a target observation. The synthesised
field is *statistically equivalent* to the target — not a copy — and can
serve as a new realisation of the same physical process.

Typical applications: CMB foreground emulation, interstellar medium (ISM) dust
maps, large-scale structure mock catalogues.

This workflow corresponds to `Demo_Synthesis.ipynb` in `demo-foscat-pangeo-eosc`.

---

## Workflow overview

```
  Target map  d  (nside, HEALPix NESTED)
        │
        ▼
  ┌─────────────────────────────────────┐
  │  scat_op = sc.funct(...)            │
  │  target_stat = scat_op.eval(d)      │
  └─────────────────────────────────────┘
        │  Φ(d) — scattering-covariance descriptor
        ▼
  ┌─────────────────────────────────────┐
  │  Define loss:  L(u) = ‖Φ(u)−Φ(d)‖² │
  └─────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Synthesis(loss).run(x0)            │
  │  x0 = Gaussian noise                │
  └─────────────────────────────────────┘
        │
        ▼
  Synthetic map  u*  with  Φ(u*) ≈ Φ(d)
```

---

## Minimal example

```python
import numpy as np
import healpy as hp
import foscat.scat_cov as sc
from foscat.Synthesis import Loss, Synthesis

# --- target map ---
nside = 64
target_map = hp.read_map("my_dust_map.fits", field=0)   # or np.random.randn(...)

# --- scattering operator ---
scat_op = sc.funct(
    KERNELSZ = 5,
    NORIENT  = 4,
    OSTEP    = 1,
    nstep_max = 4,
    all_type = 'float64',
    silent   = True,
)
target_stat = scat_op.eval(target_map)

# --- loss ---
def synth_loss(x, scat_op, args):
    ref = args[0]
    stat = scat_op.eval(x)
    return stat.reduce_mean_batch((stat - ref) ** 2)

loss = Loss(synth_loss, scat_op, target_stat)

# --- synthesis ---
x0 = np.random.randn(12 * nside**2)
solver = Synthesis([loss], eta=0.03)
result = solver.run(x0, NUM_EPOCHS=300, EVAL_FREQUENCY=10)
```

---

## Normalised loss (recommended)

Dividing the squared difference by the target statistics makes the loss
dimensionless and balances contributions from all coefficient groups:

```python
def synth_loss_norm(x, scat_op, args):
    ref, ref_sq = args
    stat = scat_op.eval(x)
    diff = (stat - ref) ** 2
    return stat.reduce_mean_batch(diff / ref_sq)

target_sq = target_stat ** 2
loss = Loss(synth_loss_norm, scat_op, target_stat, target_sq)
```

---

## Working with masks

When the target is a partial-sky map, pass the survey mask so that statistics
are computed only over valid pixels. The gradient mask ensures the synthesised
field is only updated inside the footprint:

```python
mask = np.zeros(12 * nside**2)
mask[valid_pixels] = 1.0

target_stat = scat_op.eval(target_map, mask=mask)

result = solver.run(x0, NUM_EPOCHS=300, grd_mask=mask)
```

---

## Multi-resolution warm start

Synthesising at full `nside` from scratch requires many epochs. A warm start
synthesises first at low resolution and progressively upsamples:

```python
import healpy as hp

# Step 1: coarse synthesis at nside=16
scat_16 = sc.funct(KERNELSZ=5, NORIENT=4, nstep_max=2, all_type='float64')
stat_16  = scat_16.eval(hp.ud_grade(target_map, 16))
loss_16  = Loss(synth_loss, scat_16, stat_16)

x0_16 = np.random.randn(12 * 16**2)
result_16 = Synthesis([loss_16]).run(x0_16, NUM_EPOCHS=200)

# Step 2: upsample and refine at nside=64
x0_64 = hp.ud_grade(result_16, 64)
scat_64 = sc.funct(KERNELSZ=5, NORIENT=4, nstep_max=4, all_type='float64')
stat_64 = scat_64.eval(target_map)
loss_64 = Loss(synth_loss, scat_64, stat_64)

result_64 = Synthesis([loss_64]).run(x0_64, NUM_EPOCHS=300)
```

---

## Practical notes

- **Normalisation.** Normalise the target map (zero mean, unit variance) before
  synthesis. Restore the original mean and variance afterwards: the synthesis
  only preserves the *shape* of the statistics, not absolute amplitude unless
  you include S0 in the loss.

- **`nstep_max`.** Set `nstep_max` to the number of HEALPix resolution levels
  you want to capture. For `nside=64` (6 halvings from `nside=1`), `nstep_max=4`
  captures the four coarsest scales. Increasing it recovers more large-scale
  structure at the cost of more computation.

- **`NORIENT`.** Use 4 for most geophysical fields. Use 8 for highly anisotropic
  or filamentary fields (e.g. ISM emission, galactic synchrotron). Use 1 for
  purely isotropic statistics.

- **Convergence check.** The final loss should be below `1e-2` for a convincing
  synthesis. Plot the loss history with `np.array(solver.history)`.

- **`nest=True`.** FOSCAT always works in HEALPix NESTED ordering. Pass
  `nest=True` to `healpy` functions (e.g. `hp.read_map`, `hp.mollview`).
