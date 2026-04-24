# FOSCAT

FOSCAT is a Python package for wavelet/scattering-based analysis and synthesis of data on regular 2D grids and on the sphere with HEALPix pixelization.

This README replaces the former dependency on `FOSCAT_DEMO`. The recommended examples are now the notebooks from `demo-foscat-pangeo-eosc`, especially:

- `Demo_Synthesis.ipynb` - global HEALPix synthesis from scattering-covariance statistics.
- `Synthesis2D.ipynb` - 2D image/field synthesis.
- `Denoising-2D.ipynb` - denoising by statistical constraints.
- `Remove_CMB.ipynb` - component separation / CMB-like background removal.
- `local_foscat.ipynb` and `CNN_local.ipynb` - local HEALPix domains and local spherical convolutions.

## Install

```bash
pip install foscat
```

For the examples, a fuller environment is usually useful:

```bash
micromamba create -n foscat python=3.10
micromamba activate foscat
pip install foscat
pip install tensorflow torch healpy xarray gcsfs zarr jupyterlab
```

## Minimal HEALPix synthesis sketch

```python
import numpy as np
import foscat.scat_cov as sc
from foscat.Synthesis import Synthesis

nside = 64
target_map = np.random.randn(12 * nside**2)

scat_op = sc.funct(KERNELSZ=5, NORIENT=4, OSTEP=1, all_type='float64')
target_stat = scat_op.eval(target_map)

class MyLoss:
    def __init__(self, scat_op, target_stat):
        self.scat_op = scat_op
        self.target_stat = target_stat

    def eval(self, x, batch, return_all=False):
        stat = self.scat_op.eval(x)
        loss = stat.reduce_mean_batch((stat - self.target_stat) ** 2)
        return loss

x0 = np.random.randn(target_map.size)
solver = Synthesis(MyLoss(scat_op, target_stat))
result = solver.run(x0, EVAL_FREQUENCY=10, NUM_EPOCHS=50)
```

## Documentation

This repository contains a MkDocs documentation skeleton:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

A minimal Sphinx bridge is also provided in `docs_sphinx/` for teams preferring Sphinx.

## Main sections

- Getting started: installation and concepts.
- User guide: scattering covariance, HEALPix synthesis, 2D synthesis, component separation, and local wavelet convolutions.
- Examples: mapping from maintained notebooks to documentation pages.
- Reference: practical API entry points and best practices.
