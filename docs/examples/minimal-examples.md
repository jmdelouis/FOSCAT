# Minimal examples

## HEALPix statistics

```python
import numpy as np
import foscat.scat_cov as sc

nside = 32
x = np.random.randn(12 * nside**2)

scat_op = sc.funct(KERNELSZ=5, NORIENT=4, OSTEP=1, all_type='float64')
stat = scat_op.eval(x)
```

## 2D statistics

```python
import numpy as np
import foscat.scat_cov2D as sc2d

x = np.random.randn(128, 128)
scat_op = sc2d.funct(KERNELSZ=5, NORIENT=4, all_type='float64')
stat = scat_op.eval(x)
```

## HEALPix neural network sketch

```python
from foscat.healpix_unet_torch import HealpixUNet

model = HealpixUNet(
    nside=64,
    in_channels=1,
    out_channels=1,
)
```

Check the local installed version for the exact signature if this constructor changes.
