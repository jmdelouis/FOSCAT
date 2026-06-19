# FOSCAT: Scattering Analysis and Synthesis on the Sphere

**FOSCAT** is a Python library for wavelet/scattering-based statistical analysis,
synthesis, and deep learning on 2D fields and on the sphere represented with
[HEALPix](https://healpix.sourceforge.io/) pixelisation.

The central idea: any spatial field can be summarised by a compact vector of
**scattering-covariance statistics** Φ(d). Given a target field, FOSCAT can
*synthesise* a new realisation that matches those statistics — without ever
copying pixels from the original. The same statistics serve as loss functions,
descriptors for classification, and morphological constraints for denoising and
component separation.

---

## Quick install

```bash
pip install foscat
```

See {doc}`installation` for GPU setup and a full environment recipe.

---

## Start here

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Overview
:link: overview
:link-type: doc

Mathematical background: wavelet filters, scattering-covariance coefficients S0–S4,
synthesis as a differentiable inverse problem, and the PyTorch backend.
:::

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install from PyPI, set up a full environment, enable GPU acceleration.
:::

:::{grid-item-card} API Reference
:link: autoapi/index
:link-type: doc

Auto-generated documentation of all classes and functions.
:::

:::{grid-item-card} Changelog
:link: changelog
:link-type: doc

Version history and release notes.
:::

::::

---

## Scattering operators

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Scattering covariance (FoCUS / scat_cov)
:link: user-guide/scattering-covariance
:link-type: doc

The core operator. Creates oriented wavelet filters on HEALPix or 2D grids,
evaluates S0–S4 coefficient sets, supports batching and masking.
:::

:::{grid-item-card} Synthesis engine
:link: user-guide/synthesis
:link-type: doc

L-BFGS-B differentiable optimisation loop. Plug in any differentiable loss
to synthesise or reconstruct fields matching target statistics.
:::

:::{grid-item-card} synthesis — all options
:link: user-guide/synthesis-options
:link-type: doc

Complete reference for every parameter of `scat_op.synthesis()`: multi-resolution
schedule, masking, upsampling, iso_ang, cross-covariance, and more.
:::

::::

---

## Workflows

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} HEALPix synthesis
:link: user-guide/healpix-synthesis
:link-type: doc

Generate full-sky maps whose scattering-covariance statistics match a target
HEALPix observation. Typical applications: CMB, dust, ISM emissivity.
:::

:::{grid-item-card} 2D synthesis
:link: user-guide/2d-synthesis
:link-type: doc

Synthesise and denoise planar fields (ocean, atmosphere, cloud images) using
the 2D scattering-covariance operator.
:::

:::{grid-item-card} Component separation
:link: user-guide/component-separation
:link-type: doc

Separate morphologically distinct components from a mixture using scattering
statistics as differentiable morphological priors.
:::

:::{grid-item-card} Local HEALPix wavelets
:link: user-guide/local-healpix-wavelets
:link-type: doc

Process incomplete or regional spherical domains. Build local stencils and
apply wavelet convolutions on arbitrary subsets of the HEALPix sphere.
:::

:::{grid-item-card} HEALPix neural networks
:link: user-guide/neural-networks
:link-type: doc

Train U-Net and graph-convolutional models that operate directly on HEALPix
geometry using FOSCAT oriented convolutions as the spatial primitive.
:::

::::

---

## Resources

- {doc}`overview` — Architecture and design philosophy
- {doc}`installation` — Installation guide
- {doc}`autoapi/index` — Full API reference

```{toctree}
---
maxdepth: 1
caption: Getting Started
hidden: true
---
installation
overview
```

```{toctree}
---
maxdepth: 2
caption: User Guide
hidden: true
---
user-guide/scattering-covariance
user-guide/synthesis
user-guide/synthesis-options
user-guide/healpix-synthesis
user-guide/2d-synthesis
user-guide/component-separation
user-guide/local-healpix-wavelets
user-guide/neural-networks
```

```{toctree}
---
maxdepth: 1
caption: Reference
hidden: true
---
reference/module-map
autoapi/index
```

```{toctree}
---
maxdepth: 1
caption: About
hidden: true
---
changelog
```
