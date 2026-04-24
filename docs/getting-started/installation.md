# Installation

Install the package from PyPI:

```bash
pip install foscat
```

For notebook workflows, use a dedicated environment:

```bash
micromamba create -n foscat python=3.10
micromamba activate foscat
pip install foscat
pip install tensorflow torch healpy xarray gcsfs zarr jupyterlab
```

## Main dependencies

Depending on the workflow, you may need TensorFlow or PyTorch:

- `numpy`, `matplotlib`
- `healpy`
- `spherical`
- `tensorflow` for differentiable synthesis workflows used in several historical notebooks
- `torch` for recent HEALPix neural-network modules
- `xarray`, `zarr`, `gcsfs` for Pangeo/EOSC-style data workflows

## Recommended HEALPix convention

The examples generally use HEALPix in `nest=True` ordering. Keep this convention consistent across data loading, masking, degradation, synthesis, and visualization.
