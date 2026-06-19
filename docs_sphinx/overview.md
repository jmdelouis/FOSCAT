# Overview

**FOSCAT** (Field cOmpression via Scattering CovariAnce Transform) is a Python library
for statistical characterisation, synthesis, denoising, and deep learning on spatial
fields — on regular 2D grids and on the sphere discretised with
[HEALPix](https://healpix.sourceforge.io/) pixelisation.

---

## Why scattering covariance?

A power spectrum $P(\ell)$ captures the second-order statistics of a field but is
blind to higher-order structure: two fields with identical power spectra can look
completely different (e.g. a Gaussian random field versus an intermittent turbulent
field or a CMB foreground). Scattering covariance statistics go further: they encode
*cross-scale correlations* and *phase information* in a differentiable, compact
descriptor that is suitable as a loss function for gradient-based optimisation.

Concretely, FOSCAT implements the **Wavelet Scattering Covariance** transform of
Mallat (2012) and its extensions to the sphere (Allys et al. 2019, 2020). The key
properties are:

- **Multi-scale**: coefficients are computed at multiple spatial resolutions
  (`nstep_max` levels, each reducing `nside` by a factor of 2).
- **Multi-orientation**: complex Morlet wavelets are defined at `NORIENT` orientations
  uniformly covering $[0, \pi)$.
- **Differentiable**: the full pipeline is implemented in PyTorch, enabling
  gradient flow through the statistics for synthesis and learning.
- **Equivariant**: S0 and S1 are rotationally averaged; S2, S3, and S4 encode
  orientation-dependent cross-scale covariances.

---

## Wavelet filters

At each scale $j$ and orientation $\theta_k = k\pi / N_\text{orient}$ ($k = 0,\ldots,N_\text{orient}-1$), FOSCAT
constructs a complex Morlet wavelet kernel $\psi_{j,k}$ defined on a $K \times K$
stencil centred on each pixel:

$$
\psi_{j,k}(x, y) =
  e^{-\frac{1}{2}(u^2 + v^2)}
  \bigl( \cos(\pi v) + i\,\sin(\pi v) \bigr)
  - \text{mean}
$$

where $(u, v) = \Lambda \, R(\theta_k) (x, y) / (K/2)$ are the rotated, scaled
local coordinates and $\Lambda$ is the frequency parameter (`LAMBDA`, default
1.2). The kernel is mean-subtracted and normalised so that all orientations have
equal energy.

On HEALPix maps the stencil neighbours of each pixel are looked up from
precomputed tables stored in `~/.FOSCAT/data/`; on 2D grids a standard sliding
window is used.

---

## Scattering-covariance coefficients

Given an input field $d$ and the wavelet operator $W_{j,k}$, FOSCAT computes the
following coefficient groups.

### S0 — mean power per scale

$$S_0(j) = \langle |W_{j} \star d|^2 \rangle_\text{pixels}$$

A single real value per scale: the spatially averaged modulus-squared of the
wavelet response. Equivalent to the power spectrum band-passed to scale $j$.

### S1 — first-order wavelet modulus mean

$$S_1(j, k) = \langle |W_{j,k} \star d| \rangle_\text{pixels}$$

Mean of the wavelet modulus at each scale and orientation. Captures the
*intensity* of features at each scale/orientation without phase information.

### S2 — second-order scattering mean

$$S_2(j_1, k_1, j_2, k_2) =
  \langle \bigl|W_{j_2,k_2} \star |W_{j_1,k_1} \star d|\bigr|^2 \rangle_\text{pixels}
  \quad j_2 > j_1$$

The classical scattering coefficient. Applies a second wavelet to the modulus of
the first response — capturing energy at scale $j_2$ in the *envelope* of scale
$j_1$ features. This encodes intermittency and multi-scale coupling.

### S3 — cross-scale covariance (complex)

$$S_3(j_1, k_1, j_2, k_2) =
  \langle (W_{j_1,k_1} \star d) \cdot \overline{(W_{j_2,k_2} \star d)} \rangle_\text{pixels}
  \quad j_2 > j_1$$

Complex covariance between wavelet coefficients at two different scales. The
imaginary part encodes *phase relationships* between scales, which are invisible
in $S_0$ and $S_2$.

### S3P — cross-orientation phase covariance

$$S_{3P}(j, k_1, k_2) =
  \langle (W_{j,k_1} \star d) \cdot \overline{(W_{j,k_2} \star d)} \rangle_\text{pixels}$$

Cross-orientation covariance at the *same* scale. Measures oriented texture
anisotropy (e.g. filamentary versus isotropic structures).

### S4 — second-order covariance (complex)

$$S_4(j_1, k_1, j_2, k_2) =
  \langle (W_{j_2,k_2} \star |W_{j_1,k_1} \star d|) \cdot
          \overline{(W_{j_2,k_2} \star |W_{j_1',k_1'} \star d|)}
  \rangle_\text{pixels}$$

Covariance of two second-order scattering envelopes at the same coarse scale.
Together with S3, this provides a complete second-order characterisation of the
non-Gaussian morphology of the field.

---

## The synthesis problem

Given a target field $d_\text{ref}$, synthesis finds a new field $u$ such that:

$$\Phi(u) \approx \Phi(d_\text{ref})$$

where $\Phi = [S_0, S_1, S_2, S_3, S_{3P}, S_4]$ is the concatenated
scattering-covariance descriptor. The optimisation problem is:

$$u^* = \arg\min_u \mathcal{L}(u), \quad
\mathcal{L}(u) = \sum_k
  \frac{\bigl(\Phi(u)_k - \Phi(d_\text{ref})_k\bigr)^2}{\sigma_k^2}$$

where $\sigma_k$ is a per-coefficient normalisation (typically the value of
$\Phi(d_\text{ref})_k$ itself, making the loss scale-invariant).

This is solved by gradient descent: since the entire pipeline is differentiable in
PyTorch, $\nabla_u \mathcal{L}(u)$ is computed via autograd and the weights are
updated via **L-BFGS-B** (`scipy.optimize.fmin_l_bfgs_b`), a quasi-Newton method that exploits second-order information to converge in tens to a few hundred iterations. The `Synthesis` class handles the optimisation loop.

---

## Backend

FOSCAT currently supports **PyTorch only** (the TensorFlow and NumPy backends
present in earlier versions are no longer maintained). All internal tensors live on
the device selected at `FoCUS` construction time (`gpupos` parameter). The synthesis
loop operates entirely on GPU when one is available, falling back to CPU gracefully.

MPI-parallel synthesis is supported via `mpi4py` for distributed HPC workflows
(`isMPI=True`).

---

## Module map

| Module | Class / function | Purpose |
|--------|-----------------|---------|
| `foscat.scat_cov` | `funct` | HEALPix / spherical scattering-covariance operator |
| `foscat.scat_cov2D` | `funct` | 2D planar scattering-covariance operator |
| `foscat.scat_cov1D` | `funct` | 1D scattering-covariance operator |
| `foscat.FoCUS` | `FoCUS` | Core low-level class: wavelet construction, convolutions, GPU dispatch |
| `foscat.Synthesis` | `Synthesis`, `Loss` | L-BFGS-B optimisation loop for synthesis / reconstruction |
| `foscat.healpix_unet_torch` | `HealpixUNet` | PyTorch U-Net on HEALPix geometry |
| `foscat.GCNN` | `GCNN` | Graph-convolutional neural network on HEALPix |
| `foscat.CNN` | `CNN` | Classical CNN on HEALPix |
| `foscat.UNET` | `UNET` | Legacy U-Net (TensorFlow-era, kept for compatibility) |
| `foscat.healpix_vit_torch` | `HealpixViT` | Vision Transformer on HEALPix tokens |
| `foscat.alm` | — | Spherical harmonic (alm) utilities |
| `foscat.alm_loc` | — | Local spherical harmonic analysis |
| `foscat.SphericalStencil` | `Convol_torch` | Oriented convolution on HEALPix neighbours |
| `foscat.HealBili` | — | Bilinear interpolation between HEALPix resolutions |
| `foscat.HealSpline` | — | Spline interpolation on HEALPix |
| `foscat.SphereDownGeo` | `SphereDownGeo` | Geodesic downsampling |
| `foscat.SphereUpGeo` | `SphereUpGeo` | Geodesic upsampling |
| `foscat.xarray.accessor` | — | xarray integration (`.foscat` accessor) |

---

## References

- Mallat, S. (2012). *Group Invariant Scattering.* Comm. Pure Appl. Math., 65, 1331–1398.
- Allys, E. et al. (2019). *Rwst, a comprehensive statistical description of the
  non-Gaussian structures in the ISM.* A&A, 629, A115.
- Allys, E. et al. (2020). *New interpretable statistics for large-scale structure
  analysis and generation.* Phys. Rev. D, 102, 103506.
- Gorski, K. M. et al. (2005). *HEALPix: A Framework for High-Resolution Discretization
  and Fast Analysis of Data Distributed on the Sphere.* ApJ, 622, 759–771.
