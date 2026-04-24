# HEALPix synthesis

HEALPix synthesis creates a spherical field whose scattering-covariance statistics match a target map.

This workflow corresponds to `Demo_Synthesis.ipynb`.

## Basic workflow

```python
import numpy as np
import foscat.scat_cov as sc
from foscat.Synthesis import Synthesis

nside = 64
target_map = np.random.randn(12 * nside**2)

scat_op = sc.funct(KERNELSZ=5, NORIENT=4, OSTEP=1, all_type='float64')
target_stat = scat_op.eval(target_map)
```

Then define a loss object:

```python
class ScatLoss:
    def __init__(self, scat_op, target_stat):
        self.scat_op = scat_op
        self.target_stat = target_stat

    def eval(self, x, batch, return_all=False):
        stat = self.scat_op.eval(x)
        return stat.reduce_mean_batch((stat - self.target_stat) ** 2)
```

Run synthesis:

```python
x0 = np.random.randn(target_map.size)
solver = Synthesis(ScatLoss(scat_op, target_stat))
result = solver.run(x0, EVAL_FREQUENCY=10, NUM_EPOCHS=50)
```

## Practical notes

- Use `nest=True` consistently.
- Normalize target maps before synthesis when amplitudes vary strongly.
- Start with small `nside` and increase resolution progressively.
- Use multi-resolution initialization when the target contains large-scale structures.
