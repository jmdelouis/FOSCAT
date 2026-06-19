# 2D Synthesis and Denoising

The 2D operator `foscat.scat_cov2D` applies the scattering-covariance framework to
regular image grids. The workflow is identical to HEALPix synthesis but operates on
`(H, W)` arrays rather than HEALPix maps.

This workflow corresponds to `Synthesis2D.ipynb` and `Denoising-2D.ipynb` in
`demo-foscat-pangeo-eosc`.

---

## 2D synthesis

```python
import numpy as np
import foscat.scat_cov2D as sc2d
from foscat.Synthesis import Loss, Synthesis

# --- target image ---
image = np.load("cloud_fields.npy")   # shape (H, W), e.g. (256, 256)

# --- operator ---
scat_op = sc2d.funct(KERNELSZ=5, NORIENT=4, all_type='float64')
target_stat = scat_op.eval(image)

# --- loss ---
def synth_loss(x, scat_op, args):
    ref = args[0]
    stat = scat_op.eval(x)
    return stat.reduce_mean_batch((stat - ref) ** 2)

loss = Loss(synth_loss, scat_op, target_stat)

# --- synthesis ---
x0 = np.random.randn(*image.shape)
solver = Synthesis([loss], eta=0.03)
synthetic = solver.run(x0, NUM_EPOCHS=500, EVAL_FREQUENCY=20)
```

---

## Denoising

Denoising constrains the output field using statistics from a *cleaner* reference
distribution, rather than the noisy observation itself. The loss has two terms:

1. **Fidelity:** the synthesised field should remain close to the noisy observation
   in pixel space (prevents trivial solutions).
2. **Statistics:** the synthesised field's scattering statistics should match those
   of a clean reference or a model distribution.

```python
import foscat.scat_cov2D as sc2d
from foscat.Synthesis import Loss, Synthesis

# noisy_image: (H, W) — the observed (noisy) field
# clean_ref  : (H, W) — a clean reference or ensemble average

scat_op = sc2d.funct(KERNELSZ=5, NORIENT=4, all_type='float64')
clean_stat = scat_op.eval(clean_ref)

# Loss 1: statistical constraint (match clean statistics)
def stat_loss(x, scat_op, args):
    ref_stat = args[0]
    stat = scat_op.eval(x)
    return stat.reduce_mean_batch((stat - ref_stat) ** 2)

# Loss 2: pixel-level fidelity to the noisy observation
def fidelity_loss(x, scat_op, args):
    obs, sigma2 = args
    x_np = scat_op.backend.to_numpy(x)
    return scat_op.backend.bk_mean((x_np - obs) ** 2 / sigma2)

loss_stat     = Loss(stat_loss,     scat_op, clean_stat)
loss_fidelity = Loss(fidelity_loss, scat_op, noisy_image, noise_variance)

solver = Synthesis([loss_stat, loss_fidelity], eta=0.01)
denoised = solver.run(noisy_image, NUM_EPOCHS=300)
```

---

## Batched synthesis

Synthesise multiple independent realisations in parallel by passing a batched
initial field:

```python
n_real = 8
x0_batch = np.random.randn(n_real, *image.shape)

# scat_cov2D.eval handles the batch dimension automatically
stat = scat_op.eval(image)          # reference from a single image
target_batch = stat                  # same target applied to all realisations

result_batch = solver.run(x0_batch, NUM_EPOCHS=400)
# result_batch.shape: (n_real, H, W)
```

---

## Practical notes

- **Image size.** Powers of 2 (e.g. 128×128, 256×256, 512×512) work best with
  the multi-resolution downsampling. Non-power-of-2 sizes are supported but may
  produce slightly asymmetric stencils at the finest level.

- **Normalisation.** Subtract the mean and divide by the standard deviation before
  computing statistics. This makes the loss scale-invariant and greatly improves
  convergence speed.

- **`nstep_max`.** For a 256×256 image, 4–5 levels captures all relevant scales.
  Beyond that, the coarse-level stencil covers the entire image and provides no
  additional spatial information.

- **Boundary effects.** By default the 2D operator uses `padding="SAME"`. For
  periodic fields (e.g. turbulence with periodic boundary conditions) this
  matches the field geometry; for non-periodic fields consider `padding="VALID"`
  to avoid wrap-around artefacts.
