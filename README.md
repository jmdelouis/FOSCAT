# FOSCAT

**FOSCAT** (Field cOmpression via Scattering CovariAnce Transform) is a Python
library for wavelet/scattering-based statistical analysis, synthesis, denoising,
and deep learning on 2D fields and on the sphere with
[HEALPix](https://healpix.sourceforge.io/) pixelisation.

The central idea: any spatial field can be summarised by a compact vector of
**scattering-covariance statistics** Φ(d) — encoding multi-scale, cross-orientation
correlations beyond what a power spectrum captures. FOSCAT can then *synthesise* a
new field matching those statistics, use them as a loss for denoising or component
separation, or feed them into neural-network architectures that operate directly on
spherical geometry.

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jmdelouis.github.io/FOSCAT/)
[![PyPI](https://img.shields.io/pypi/v/foscat)](https://pypi.org/project/foscat/)
[![CI](https://github.com/jmdelouis/FOSCAT/actions/workflows/ci.yml/badge.svg)](https://github.com/jmdelouis/FOSCAT/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

---

## Install

```bash
pip install foscat
```

Full environment for running the example notebooks:

```bash
micromamba create -n foscat python=3.10
micromamba activate foscat
pip install foscat torch healpy xarray gcsfs zarr jupyterlab matplotlib scipy
```

> **Backend note:** FOSCAT uses PyTorch as its only active backend (TensorFlow
> and NumPy backends are no longer maintained).

---

## Minimal HEALPix synthesis

```python
import numpy as np
import foscat.scat_cov as sc
from foscat.Synthesis import Loss, Synthesis

nside = 64
target_map = np.random.randn(12 * nside**2)

# build the scattering operator and compute target statistics
scat_op     = sc.funct(KERNELSZ=5, NORIENT=4, nstep_max=4, all_type='float64')
target_stat = scat_op.eval(target_map)

# define a differentiable loss
def synth_loss(x, scat_op, args):
    ref  = args[0]
    stat = scat_op.eval(x)
    return stat.reduce_mean_batch((stat - ref) ** 2)

loss   = Loss(synth_loss, scat_op, target_stat)
solver = Synthesis([loss], eta=0.03)

x0     = np.random.randn(12 * nside**2)
result = solver.run(x0, NUM_EPOCHS=300, EVAL_FREQUENCY=10)
```

---

## What FOSCAT can do

| Capability | Entry point | Example notebook |
|---|---|---|
| HEALPix synthesis | `foscat.scat_cov.funct` + `Synthesis` | `Demo_Synthesis.ipynb` |
| 2D image synthesis | `foscat.scat_cov2D.funct` + `Synthesis` | `Synthesis2D.ipynb` |
| Denoising | two-loss synthesis (statistics + fidelity) | `Denoising-2D.ipynb` |
| Component separation | multi-loss synthesis | `Remove_CMB.ipynb` |
| Local spherical analysis | `cell_ids` in `FoCUS` / `HealpixUNet` | `local_foscat.ipynb` |
| Spherical U-Net (regression/segmentation) | `HealpixUNet` | `CNN_local.ipynb`, `CNN_ecmwf.ipynb` |

Example notebooks are maintained in
[demo-foscat-pangeo-eosc](https://github.com/jmdelouis/demo-foscat-pangeo-eosc).

---

## Documentation

Full documentation (installation, mathematical background, API reference, user
guide) is published at:

**https://jmdelouis.github.io/FOSCAT/**

To build the docs locally:

```bash
cd docs_sphinx
make install   # pip install requirements.txt
make html      # → _build/html/index.html
make livehtml  # live-reload on save (requires sphinx-autobuild)
```

---

## Citation

If you use FOSCAT in a publication, please cite:

```bibtex
@software{foscat,
  author  = {Delouis, Jean-Marc and Foulquier, Theo},
  title   = {{FOSCAT}: Field cOmpression via Scattering CovariAnce Transform},
  year    = {2026},
  url     = {https://github.com/jmdelouis/FOSCAT},
  version = {2026.04.1},
}
```

The underlying statistical framework is described in:

- Allys et al. (2019), *Rwst, a comprehensive statistical description of the
  non-Gaussian structures in the ISM*, A&A 629, A115.
- Allys et al. (2020), *New interpretable statistics for large-scale structure
  analysis and generation*, Phys. Rev. D 102, 103506.

---

## License

BSD 3-Clause. See [LICENSE](LICENSE).
