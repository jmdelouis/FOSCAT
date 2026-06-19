# Synthesis Engine — `Synthesis` and `Loss`

**Module:** `foscat.Synthesis`

The `Synthesis` class drives field synthesis, denoising, and component separation
in FOSCAT. It wraps **L-BFGS-B** (`scipy.optimize.fmin_l_bfgs_b`) — a
quasi-Newton method that builds a low-memory approximation of the inverse Hessian
from the gradient history. L-BFGS-B converges much faster than first-order
optimisers (Adam, SGD) for smooth, moderately high-dimensional problems like
scattering-covariance synthesis.

---

## Architecture overview

```
  Initial field  x_0  (random Gaussian noise or a prior)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  scipy.optimize.fmin_l_bfgs_b                           │
  │                                                         │
  │   loop (up to NUM_EPOCHS iterations):                   │
  │   ┌─────────────────────────────────────────────────┐   │
  │   │  calc_grad(x)                                   │   │
  │   │   → FoCUS.eval(x)  →  scat_cov stats           │   │
  │   │   → user loss  L(stats, target_stats)           │   │
  │   │   → scalar loss  +  ∇ₓ L  (via autograd)       │   │
  │   └─────────────────────────────────────────────────┘   │
  │                                                         │
  │   L-BFGS-B step  →  x_{t+1}                            │
  │   (Hessian approximation updated from gradient history) │
  └─────────────────────────────────────────────────────────┘
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
    name          = "",
    batch         = None,
    batch_data    = None,
    batch_update  = None,
    info_callback = False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `function` | callable | — | Loss function. Signature: `f(x, scat_op, args) → scalar tensor` (no-batch mode) or `f(x, batch, scat_op, args) → scalar tensor` (batch mode). Must return a differentiable PyTorch scalar. |
| `scat_operator` | FoCUS | — | The scattering operator used inside `function`. Determines the backend and device. |
| `*param` | any | — | Extra arguments packed into a tuple and passed to `function` as `args`. |
| `name` | str | `""` | Optional label printed in the iteration log. |
| `batch` | callable\|None | None | `batch(batch_data, istep) → batch_indices`. Enables mini-batch gradient computation. If None, the full field is used at every step. |
| `batch_data` | any | None | Data passed to `batch` at each step. |
| `batch_update` | callable\|None | None | Called between L-BFGS-B restarts (see `NUM_STEP_BIAS`) to update `batch_data`. |
| `info_callback` | bool | False | If True, `function` must return `(loss, info_str)` and the string is printed at log frequency. |

### Loss function signature

```python
def my_loss(x, scat_op, args):
    """
    x       : torch.Tensor — current field, shape (npix,) or (batch, npix)
    scat_op : FoCUS instance
    args    : tuple of extra arguments
    returns : scalar torch.Tensor, differentiable w.r.t. x
    """
    target_stat = args[0]
    stat = scat_op.eval(x)
    return stat.reduce_mean_batch((stat - target_stat) ** 2)
```

### Minimal example

```python
from foscat.Synthesis import Loss
import foscat.scat_cov as sc

scat_op     = sc.funct(KERNELSZ=5, NORIENT=4, all_type='float64')
target_stat = scat_op.eval(target_map)

def scattering_loss(x, scat_op, args):
    ref  = args[0]
    stat = scat_op.eval(x)
    return stat.reduce_mean_batch((stat - ref) ** 2)

loss = Loss(scattering_loss, scat_op, target_stat)
```

---

## `Synthesis` — the optimisation loop

```python
foscat.Synthesis.Synthesis(loss_list)
```

The constructor only takes the list of `Loss` objects. Optimiser hyperparameters
are passed to `run()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `loss_list` | list[Loss] | One or more `Loss` objects. The total gradient is the sum across all losses, enabling multi-component optimisation. |

### `run` — execute L-BFGS-B

```python
result = solver.run(
    in_x,
    NUM_EPOCHS           = 100,
    EVAL_FREQUENCY       = 100,
    factr                = 10.0,
    NUM_STEP_BIAS        = 1,
    LEARNING_RATE        = 0.03,
    DECAY_RATE           = 0.95,
    batchsz              = 1,
    totalsz              = 1,
    grd_mask             = None,
    idx_grd              = None,
    KEEP_TRACK           = None,
    SHOWGPU              = False,
    MESSAGE              = "",
    axis                 = 0,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_x` | ndarray, shape `(npix,)` | — | Initial field. Typically Gaussian noise with power matched to the target. |
| `NUM_EPOCHS` | int | 100 | Maximum number of L-BFGS-B iterations (`maxiter`). The optimiser may stop earlier if the convergence criterion is met. |
| `EVAL_FREQUENCY` | int | 100 | Print the current loss every N iterations (via the L-BFGS-B callback). |
| `factr` | float | 10.0 | L-BFGS-B convergence tolerance: the iteration stops when `(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) ≤ factr × eps_machine`. Smaller values run longer and converge more tightly. Typical range: `1e7` (fast/loose) to `10` (tight). |
| `NUM_STEP_BIAS` | int | 1 | Number of L-BFGS-B restarts. At each restart the Hessian approximation is reset and `batch_update` is called on all losses. Useful for progressive noise schedules. |
| `LEARNING_RATE` | float | 0.03 | *Vestigial — not used by L-BFGS-B.* |
| `DECAY_RATE` | float | 0.95 | *Vestigial — not used by L-BFGS-B.* |
| `batchsz` | int | 1 | Number of noise realisations used per gradient evaluation (for stochastic losses). |
| `totalsz` | int | 1 | Total batch pool size. When `totalsz > batchsz`, the gradient is accumulated over `totalsz // batchsz` sub-batches. |
| `grd_mask` | ndarray\|None | None | Binary mask applied to the gradient before each L-BFGS-B step. Pixels with mask = 0 are frozen. |
| `idx_grd` | ndarray\|None | None | Index array selecting the pixels to optimise (partial inpainting). |
| `KEEP_TRACK` | callable\|None | None | Called at each log step with the current info dict for custom progress tracking. |
| `SHOWGPU` | bool | False | Print GPU memory usage alongside the loss. |
| `MESSAGE` | str | `""` | Prefix prepended to each log line. |
| `axis` | int | 0 | Axis along which to extract the output map (relevant for multi-component problems). |

**Returns:** ndarray, same shape as `in_x` — the optimised field.

---

## Why L-BFGS-B?

L-BFGS-B uses second-order information (an approximation of the Hessian) to
take much larger steps than a first-order method like Adam. For
scattering-covariance synthesis:

- The loss landscape is smooth (sum of squared differences of smooth statistics).
- The number of free variables is `npix` (up to ~800 000 for `nside=256`) —
  within the range where L-BFGS-B is effective.
- Typical convergence in 50–300 iterations vs. thousands for Adam.

The key parameter is `factr`: set it to a large value (`1e6`) for a quick
approximate synthesis; reduce it (`10` or `1.0`) for high-fidelity results.
Leave `NUM_EPOCHS` large enough that the `factr` criterion triggers before
`maxiter` is hit.

---

## Multiple losses

```python
from foscat.Synthesis import Loss, Synthesis

loss_scat  = Loss(scattering_loss,    scat_op, target_stat,  name="scat")
loss_power = Loss(power_spec_loss,    scat_op, target_Cl,    name="Cl")

solver = Synthesis([loss_scat, loss_power])
result = solver.run(x0, NUM_EPOCHS=200, factr=10.0)
```

The total gradient at each L-BFGS-B step is
$\nabla_x \mathcal{L}_\text{scat} + \nabla_x \mathcal{L}_{C_\ell}$.

---

## Gradient masking

```python
grd_mask = survey_mask.astype(np.float64)   # 1 inside survey, 0 outside
result = solver.run(x0, NUM_EPOCHS=300, grd_mask=grd_mask)
```

---

## Practical tips

**`factr`.** Start with `factr=1e7` for a quick test. For production synthesis
use `factr=10` or lower. With `factr=1e-32` (effectively disabled), convergence
is determined entirely by `NUM_EPOCHS`.

**`NUM_EPOCHS`.** Set it higher than you expect to need — L-BFGS-B stops
automatically when `factr` is satisfied. A typical synthesis at `nside=64`
converges in 100–300 iterations.

**Initialisation.** A Gaussian white noise map works well. For faster convergence,
pre-colour the noise with the target power spectrum (`healpy.synfast`).

**Multi-resolution warm start.** For large `nside`, synthesise first at low
resolution and upsample the result as the initial guess for the full resolution.
L-BFGS-B benefits significantly from a good initialisation.

**`get_history()`.** Retrieve the loss curve after `run()`:

```python
result = solver.run(x0, NUM_EPOCHS=300)
loss_curve = solver.get_history()   # ndarray, one value per logged iteration
```
