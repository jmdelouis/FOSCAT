# Component Separation

Component separation uses scattering-covariance statistics as differentiable
morphological priors to separate mixed physical fields. Because the statistics
encode multi-scale, cross-orientation correlations, they capture morphological
*shape* beyond what a simple power spectrum can describe — making them powerful
priors for separating physically distinct components.

This workflow corresponds to `Remove_CMB.ipynb` in `demo-foscat-pangeo-eosc`.

---

## The problem

Given an observed mixture:

$$d = s_1 + s_2 + \cdots + s_n$$

find component maps $\hat{s}_1, \ldots, \hat{s}_n$ such that:

- $\hat{s}_1 + \cdots + \hat{s}_n \approx d$ (mixture consistency)
- $\Phi(\hat{s}_i) \approx \Phi_i^\text{ref}$ (each component matches its expected morphology)

where $\Phi_i^\text{ref}$ are reference statistics for each component (from a
training set, a physical model, or a prior observation).

This is an underdetermined inverse problem: FOSCAT regularises it with scattering
statistics rather than with smoothness or sparsity priors.

---

## General workflow

```
  Observed mixture d = s₁ + s₂  (known)
        │
        ▼  optimise over (ŝ₁, ŝ₂)
  ┌──────────────────────────────────────────────────────────┐
  │  L_mix   = ‖ŝ₁ + ŝ₂ − d‖²          (mixture fidelity)  │
  │  L_stat₁ = ‖Φ(ŝ₁) − Φ(s₁_ref)‖²    (morphology prior)  │
  │  L_stat₂ = ‖Φ(ŝ₂) − Φ(s₂_ref)‖²    (morphology prior)  │
  └──────────────────────────────────────────────────────────┘
        │
        ▼
  Synthesis([L_mix, L_stat₁, L_stat₂]).run(x0)
        │
        ▼
  ŝ₁*, ŝ₂*  — separated components
```

---

## Example: CMB-like background removal

```python
import numpy as np
import foscat.scat_cov as sc
from foscat.Synthesis import Loss, Synthesis

nside = 64
npix  = 12 * nside**2

# --- input data ---
mixture   = np.load("observed_mixture.npy")   # d = foreground + CMB
cmb_ref   = np.load("cmb_simulations.npy")    # ensemble of CMB realisations
fg_ref    = np.load("foreground_ref.npy")     # reference foreground map

# --- operator ---
scat_op = sc.funct(KERNELSZ=5, NORIENT=4, nstep_max=4, all_type='float64')

# compute reference statistics (average over CMB ensemble if available)
cmb_stat = scat_op.eval(cmb_ref)
fg_stat  = scat_op.eval(fg_ref)

# --- optimise over (ŝ_fg, ŝ_cmb) jointly ---
# pack both fields into a single vector: x = [ŝ_fg | ŝ_cmb]
def mixture_loss(x, scat_op, args):
    d = args[0]
    s_fg  = x[:npix]
    s_cmb = x[npix:]
    residual = (s_fg + s_cmb - d) ** 2
    return scat_op.backend.bk_mean(residual)

def fg_stat_loss(x, scat_op, args):
    ref = args[0]
    s_fg = x[:npix]
    stat = scat_op.eval(s_fg)
    return stat.reduce_mean_batch((stat - ref) ** 2)

def cmb_stat_loss(x, scat_op, args):
    ref = args[0]
    s_cmb = x[npix:]
    stat = scat_op.eval(s_cmb)
    return stat.reduce_mean_batch((stat - ref) ** 2)

loss_mix  = Loss(mixture_loss,  scat_op, mixture)
loss_fg   = Loss(fg_stat_loss,  scat_op, fg_stat)
loss_cmb  = Loss(cmb_stat_loss, scat_op, cmb_stat)

# initialise: foreground = mixture, CMB = zero
x0 = np.concatenate([mixture, np.zeros(npix)])

solver = Synthesis([loss_mix, loss_fg, loss_cmb], eta=0.01)
result = solver.run(x0, NUM_EPOCHS=500, EVAL_FREQUENCY=20)

fg_hat  = result[:npix]
cmb_hat = result[npix:]
```

---

## Weighting losses

The three losses above have different scales. If one dominates, weight them
explicitly by scaling your loss function return values:

```python
def cmb_stat_loss(x, scat_op, args):
    ref, weight = args
    s_cmb = x[npix:]
    stat = scat_op.eval(s_cmb)
    return weight * stat.reduce_mean_batch((stat - ref) ** 2)

loss_cmb = Loss(cmb_stat_loss, scat_op, cmb_stat, 0.5)  # weight = 0.5
```

A good heuristic: at the first iteration, print each loss component separately
and set weights so they are all $O(1)$.

---

## Gradient masking for partial-sky separation

When the observation covers only part of the sky, freeze pixels outside the
survey footprint and only update the internal region:

```python
grd_mask = survey_mask.astype(np.float64)
grd_mask_full = np.tile(grd_mask, 2)   # one mask per component

result = solver.run(x0, NUM_EPOCHS=500, grd_mask=grd_mask_full)
```

---

## Practical notes

- **Reference quality.** The quality of separation depends critically on the
  reference statistics $\Phi_i^\text{ref}$. Averaging over an ensemble of
  simulations gives better priors than a single reference map.

- **Amplitude ambiguity.** Scattering statistics are not amplitude-preserving by
  default. If the amplitude ratio between components matters, add a power-spectrum
  constraint (e.g. via `healpy.anafast`) as an additional loss term.

- **Number of iterations.** Component separation typically needs more iterations
  than simple synthesis (500–1000 epochs). The loss may decrease slowly if the
  two components are morphologically similar.

- **Initialisation.** Initialising with the observed mixture for the first
  component and zeros for the second often converges faster than random noise.
