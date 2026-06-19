# Installation

## Requirements

- Python ≥ 3.9
- [PyTorch](https://pytorch.org/) ≥ 2.0 (the active backend; TensorFlow and NumPy backends are no longer maintained)
- [healpy](https://healpy.readthedocs.io/) — HEALPix Python bindings
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)

## Install from PyPI

The simplest path:

```bash
pip install foscat
```

## Full environment (recommended for notebooks)

For running the example notebooks from `demo-foscat-pangeo-eosc`:

::::{tab-set}

:::{tab-item} micromamba / conda

```bash
micromamba create -n foscat python=3.10
micromamba activate foscat
pip install foscat
pip install torch healpy xarray gcsfs zarr jupyterlab matplotlib scipy
```

:::

:::{tab-item} pip only

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install foscat torch healpy xarray zarr jupyterlab matplotlib scipy
```

:::

::::

## GPU support

FOSCAT's synthesis loop and neural-network modules run on GPU automatically when
PyTorch detects a CUDA-capable device. Install PyTorch with CUDA support first:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

When instantiating a `FoCUS` operator you can select the GPU with `gpupos`:

```python
import foscat.scat_cov as sc

# Use GPU 0 (default)
scat_op = sc.funct(KERNELSZ=5, NORIENT=4, gpupos=0)

# Use GPU 1 in a multi-GPU node
scat_op = sc.funct(KERNELSZ=5, NORIENT=4, gpupos=1)
```

## MPI-parallel synthesis

For distributed HPC jobs with `mpi4py`:

```bash
pip install mpi4py
mpirun -n 4 python my_synthesis_script.py
```

Pass `isMPI=True` to `FoCUS`:

```python
scat_op = sc.funct(KERNELSZ=5, NORIENT=4, isMPI=True)
```

## Install from source (development)

```bash
git clone https://github.com/jmdelouis/FOSCAT.git
cd FOSCAT
pip install -e .
```

## Verify the installation

```python
import foscat.scat_cov as sc
import numpy as np

nside = 16
x = np.random.randn(12 * nside**2)
op = sc.funct(KERNELSZ=3, NORIENT=4, nstep_max=2, silent=True)
stat = op.eval(x)
print("FOSCAT installed successfully. Descriptor size:", stat.numel)
```

## FOSCAT data cache

The first time you instantiate `FoCUS` at a given `(nside, KERNELSZ, NORIENT)`
combination, it computes and stores wavelet stencil tables in
`~/.FOSCAT/data/`. Subsequent instantiations load from cache. Ensure that
directory is writable (or pass a custom path via `TEMPLATE_PATH`).
