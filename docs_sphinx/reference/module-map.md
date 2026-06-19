# Module Map

Complete index of FOSCAT modules, classes, and key functions.

---

## Scattering operators

| Module | Entry point | Description |
|--------|-------------|-------------|
| `foscat.scat_cov` | `funct(...)` | HEALPix / full-sphere scattering-covariance operator. Returns a `FoCUS` instance. |
| `foscat.scat_cov2D` | `funct(...)` | 2D planar scattering-covariance operator. |
| `foscat.scat_cov1D` | `funct(...)` | 1D scattering-covariance operator. |
| `foscat.FoCUS` | `FoCUS` | Core class: wavelet filter construction, GPU dispatch, stencil management. |

## Synthesis

| Module | Entry point | Description |
|--------|-------------|-------------|
| `foscat.Synthesis` | `Synthesis` | L-BFGS-B optimisation loop for synthesis / reconstruction. |
| `foscat.Synthesis` | `Loss` | Wraps a user loss function and a scattering operator. |

## Statistics objects

| Module | Class | Description |
|--------|-------|-------------|
| `foscat.scat_cov` | `scat_cov` | Container for S0–S4 coefficient arrays with arithmetic and serialisation. |
| `foscat.scat_cov2D` | `scat_cov` | 2D variant of the statistics object. |

## Neural networks

| Module | Class | Description |
|--------|-------|-------------|
| `foscat.healpix_unet_torch` | `HealpixUNet` | PyTorch U-Net on HEALPix geometry with oriented spherical convolutions. |
| `foscat.GCNN` | `GCNN` | Graph-convolutional network using FOSCAT scattering layers. |
| `foscat.CNN` | `CNN` | Flat CNN on HEALPix (simple baseline). |
| `foscat.UNET` | `UNET` | Legacy U-Net (TensorFlow-era, kept for compatibility). |
| `foscat.healpix_vit_torch` | `HealpixViT` | Vision Transformer on HEALPix tokens. |
| `foscat.healpix_vit_skip` | `HealpixViTSkip` | ViT with skip connections. |
| `foscat.planar_vit` | `PlanarViT` | ViT on equirectangular grids. |
| `foscat.unet_2_d_from_healpix_params` | — | 2D U-Net parameterised from a HEALPix model. |

## Spherical geometry

| Module | Class / function | Description |
|--------|-----------------|-------------|
| `foscat.SphericalStencil` | `Convol_torch` | Oriented spherical convolution on HEALPix neighbour stencils. |
| `foscat.HOrientedConvol` | `HOrientedConvol` | High-level oriented convolution wrapper. |
| `foscat.SphereDownGeo` | `SphereDownGeo` | Geodesic downsampling between HEALPix resolutions. |
| `foscat.SphereUpGeo` | `SphereUpGeo` | Geodesic upsampling between HEALPix resolutions. |
| `foscat.HealBili` | — | Bilinear interpolation between HEALPix resolutions. |
| `foscat.HealSpline` | — | Spline interpolation on HEALPix. |
| `foscat.CircSpline` | — | Circular spline interpolation. |
| `foscat.Spline1D` | — | 1D spline interpolation utilities. |

## Spherical harmonic analysis

| Module | Key functions | Description |
|--------|--------------|-------------|
| `foscat.alm` | `map2alm`, `alm2map`, … | Full-sky spherical harmonic decomposition. |
| `foscat.alm_latlon` | — | Alm analysis on lat/lon grids. |
| `foscat.alm_loc` | — | Local spherical harmonic analysis on regional domains. |
| `foscat.alm_loc_optim` | — | Optimised local alm routines. |

## Backends

| Module | Class | Description |
|--------|-------|-------------|
| `foscat.backend` | `foscat_backend` | Abstract base class for all backends. |
| `foscat.BkTorch` | `BkTorch` | Active PyTorch backend (GPU + CPU). |
| `foscat.BkTensorflow` | `BkTensorflow` | *Deprecated.* TensorFlow backend (not maintained). |
| `foscat.BkNumpy` | `BkNumpy` | *Deprecated.* NumPy backend — read-only; cannot run `Synthesis`. |

## Loss backends (internal)

| Module | Description |
|--------|-------------|
| `foscat.loss_backend_torch` | Gradient computation via PyTorch autograd. |
| `foscat.loss_backend_tens` | *Deprecated.* Gradient computation via TF GradientTape. |

## Visualisation

| Module | Key functions | Description |
|--------|--------------|-------------|
| `foscat.Plot` | `plot_map`, `plot_stat`, … | Convenience plotting functions (HEALPix Mollweide, statistics curves). |

## xarray integration

| Module | Description |
|--------|-------------|
| `foscat.xarray.accessor` | Registers a `.foscat` accessor on xarray DataArrays. |
| `foscat.xarray.statistics` | Statistics computations integrated with xarray labels. |
| `foscat.xarray.parameters` | Parameter management for xarray workflows. |
