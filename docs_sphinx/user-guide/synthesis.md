# Synthesis Engine — `Synthesis` and `Loss`

**Module:** `foscat.Synthesis`

The `Synthesis` class is the optimisation engine that drives field synthesis,
denoising, and component separation in FOSCAT. It implements an **Adam gradient
descent loop** where the loss — and therefore the gradient — is evaluated
through a differentiable PyTorch graph containing the scattering-covariance
operator.

---

## Architecture overview

```
  Initial field  x_0  (random Gaussian noise or a prior)
        │
        ▼
  ┌───────────────────────────────────────────────────────┐
  │  Adam optimiser  (η, β₁, β₂, ε, decay)               │
  │                                                       │
  │   ┌──────────────────────────────────────────────┐   │
  │   │  Loss.eval(x, batch)                         │   │
  │   │   → FoCUS.eval(x)  →  scat_cov stats        │   │
  │   │   → user-defined L(stats, target_stats)      │   │
  │   │   → scalar loss value                        │   │
  │   └──────────────────────────────────────────────┘   │
  │                                                       │
  │   autograd  →  ∇ₓ L  →  Adam step  →  x_{t+1}       │
  └───────────────────────────────────────────────────────┘
        │
        ▼
  Converged field  x*  with  Φ(x*) ≈ Φ(d_ref)
```

---

## `Loss` — wrapping a user loss function

```python
foscat.Synthesis.Loss(
    function,
    scat_operator,
    *param,
    name         = "",
    batch        = None,
    batch_data   = None,
    batch_update = None,
    info_callback = False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `function` | callable | — | The loss function `f(x, scat_op, args)` → scalar tensor. See signature details below. |
| `scat_operator` | FoCUS | — | The scattering operator used inside `function`. Also determines the backend and device. |
| `*param` | any | — | Extra arguments passed as a tuple to `function` as its `args` parameter. |
| `name` | str | `""` | Optional label printed in the optimisation log. |
| `batch` | callable\|None | None | Function `batch(batch_data, istep) → batch_indices` for mini-batch synthesis. If None, the full field is used at every step. |
| `batch_data` | any | None | Data passed to the `batch` function at each step. |
| `batch_update` | callable\|None | None | Called after every Adam step to update `batch_data` (e.g. for stochastic noise models). |
| `info_callback` | bool | False | If True, `function` must return `(loss, info)` and the info string is printed at log frequency. |

### Loss function signature

```python
def my_loss(x, scat_op, args):
    """
    x       : torch.Tensor, shape (npix,) or (batch, npix)
    scat_op : FoCUS instance
    args    : tuple of extra arguments passed at Loss construction time
    returns : scalar torch.Tensor (differentiable)
    """
    target_stat = args[0]
    stat = scat_op.eval(x)
    diff = stat - target_stat
    return stat.reduce_mean_batch(diff ** 2)
```

### Minimal example

```python
from foscat.Synthesis import Loss
import foscat.scat_cov as sc

scat_op = sc.funct(KERNELSZ=5, NORIENT=4, all_type='float64')
target_stat = scat_op.eval(target_map)

def scattering_loss(x, scat_op, args):
    ref = args[0]
    stat = scat_op.eval(x)
    return stat.reduce_mean_batch((stat - ref) ** 2)

loss = Loss(scattering_loss, scat_op, target_stat)
```

---

## `Synthesis` — the optimisation loop

```python
foscat.Synthesis.Synthesis(
    loss_list,
    eta        = 0.03,
    beta1      = 0.9,
    beta2      = 0.999,
    epsilon    = 1e-7,
    decay_rate = 0.999,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_list` | list[Loss] | — | List of `Loss` objects. Total gradient is the sum over all losses — enables multi-component optimisation with a single solver. |
| `eta` | float | 0.03 | Adam learning rate $\eta$. |
| `beta1` | float | 0.9 | Adam first-moment decay $\beta_1$. |
| `beta2` | float | 0.999 | Adam second-moment decay $\beta_2$. |
| `epsilon` | float | 1e-7 | Adam numerical stabiliser $\epsilon$. |
| `decay_rate` | float | 0.999 | Multiplicative learning-rate decay applied every iteration: $\eta_t = \eta \cdot r^t$. Set to 1.0 to disable decay. |

### `run` — execute the optimisation

```python
result = solver.run(
    x0,
    NUM_EPOCHS      = 1000,
    EVAL_FREQUENCY  = 10,
    SHOWGPU         = False,
    batchsz         = None,
    grd_mask        = None,
    idx_grd         = None,
    KEEP_TRACK      = None,
    MESSAGE         = "",
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x0` | ndarray, shape `(npix,)` or `(npix_total,)` | — | Initial field. Typically Gaussian noise with the same variance as the target. |
| `NUM_EPOCHS` | int | 1000 | Number of Adam gradient steps. |
| `EVAL_FREQUENCY` | int | 10 | Print the current loss every `EVAL_FREQUENCY` iterations. |
| `SHOWGPU` | bool | False | Include GPU memory usage in the printed output. |
| `batchsz` | int\|None | None | Mini-batch size (number of pixels used per gradient step). If None, all pixels are used. |
| `grd_mask` | ndarray\|None | None | Binary mask applied to the gradient before each Adam step. Pixels with mask = 0 are frozen. |
| `idx_grd` | ndarray\|None | None | Index array selecting the pixels to optimise. Useful for partial inpainting. |
| `KEEP_TRACK` | callable\|None | None | Called at each log step with the current info dict. Use for custom progress tracking. |
| `MESSAGE` | str | `""` | Prefix string prepended to each log line. |

**Returns:** ndarray, same shape as `x0` — the optimised field.

---

## Multiple losses

Pass a list to `Synthesis` to optimise multiple objectives simultaneously:

```python
from foscat.Synthesis import Loss, Synthesis

loss_scat  = Loss(scattering_loss,    scat_op, target_stat,   name="scat")
loss_power = Loss(power_spectrum_loss, scat_op, target_Cl,     name="Cl")

solver = Synthesis([loss_scat, loss_power])
result = solver.run(x0, NUM_EPOCHS=500)
```

The total gradient at each step is $\nabla_x \mathcal{L}_\text{scat} + \nabla_x \mathcal{L}_{C_\ell}$.

---

## Gradient masking

To synthesise only within a survey footprint while holding the complement fixed:

```python
grd_mask = survey_mask.astype(np.float64)   # 1 inside survey, 0 outside
result = solver.run(x0, NUM_EPOCHS=500, grd_mask=grd_mask)
```

---

## Practical tips

**Learning rate.** Start with `eta=0.03`. If the loss decreases too slowly,
increase to 0.1. If it diverges, decrease to 0.01.

**Initialisation.** Initialise `x0` with the same power spectrum as the target
to speed up convergence (use `healpy.synfast` for HEALPix, or filter a Gaussian
white noise for 2D).

**Multi-resolution warm start.** For large `nside`, first synthesise at low
resolution (e.g. `nside=16`) and upsample the result as the initialisation for
the full resolution. This dramatically reduces the number of epochs needed.

**Convergence.** The loss should decrease monotonically (with occasional plateaus).
If it oscillates, reduce `eta` or increase `decay_rate` to anneal more aggressively.
A final loss below `1e-3` generally produces visually convincing syntheses.

**dtype.** Use `all_type='float64'` for high-precision synthesis tasks. The
additional memory and compute cost is modest at typical HEALPix resolutions.
