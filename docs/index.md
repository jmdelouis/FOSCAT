# FOSCAT documentation

FOSCAT is a Python library for statistical analysis, synthesis, reconstruction, and learning on spatial fields. It supports both regular 2D images and spherical maps represented with HEALPix.

The central idea is to describe a field through wavelet/scattering statistics and then use these statistics either as descriptors, loss functions, or constraints for synthesis.

## What FOSCAT is useful for

- Synthesizing HEALPix maps with prescribed scattering-covariance statistics.
- Synthesizing and denoising regular 2D images or fields.
- Separating components, for example removing a CMB-like background from a target signal.
- Building local wavelet convolutions on incomplete HEALPix domains.
- Training neural networks that operate directly on HEALPix geometry.

## Maintained example notebooks

The current examples should be taken from `demo-foscat-pangeo-eosc`, not from the former `FOSCAT_DEMO` link.

| Notebook | Topic |
|---|---|
| `Demo_Synthesis.ipynb` | Global HEALPix synthesis |
| `Synthesis2D.ipynb` | 2D synthesis |
| `Denoising-2D.ipynb` | Denoising with statistical constraints |
| `Remove_CMB.ipynb` | Component separation |
| `local_foscat.ipynb` | Local HEALPix domain construction |
| `CNN_local.ipynb` | Local HEALPix convolutional model |
| `CNN_ecmwf.ipynb` | Neural network on HEALPix fields |
