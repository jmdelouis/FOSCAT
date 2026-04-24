# 2D synthesis

2D synthesis uses the same statistical principle as HEALPix synthesis, but on regular image grids.

This workflow corresponds to `Synthesis2D.ipynb`.

```python
import numpy as np
import foscat.scat_cov2D as sc2d
from foscat.Synthesis import Synthesis

image = np.random.randn(256, 256)
scat_op = sc2d.funct(KERNELSZ=5, NORIENT=4, all_type='float64')
target_stat = scat_op.eval(image)
```

Define a loss:

```python
class Scat2DLoss:
    def __init__(self, scat_op, target_stat):
        self.scat_op = scat_op
        self.target_stat = target_stat

    def eval(self, x, batch, return_all=False):
        stat = self.scat_op.eval(x)
        return stat.reduce_mean_batch((stat - self.target_stat) ** 2)
```

Optimize:

```python
x0 = np.random.randn(*image.shape)
solver = Synthesis(Scat2DLoss(scat_op, target_stat))
synthetic = solver.run(x0, EVAL_FREQUENCY=10, NUM_EPOCHS=50)
```

## Denoising

The `Denoising-2D.ipynb` notebook uses the same machinery, but constrains the solution with statistics from a cleaner reference or from a target distribution. This is useful when pixel-level noise should be removed while preserving multi-scale textures.
