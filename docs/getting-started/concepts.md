# Core concepts

## Scattering covariance

FOSCAT computes wavelet/scattering statistics from an input field. For a target field `d`, the operator computes a set of coefficients:

```text
Phi(d)
```

A synthesis problem seeks a new field `u` such that:

```text
Phi(u) ~= Phi(d)
```

A typical optimization loss is:

```text
L(u) = sum_k ((Phi(d)_k - Phi(u)_k)^2 / sigma_k^2)
```

where `sigma_k` is a coefficient normalization or variance.

## HEALPix maps

A full-sky HEALPix map at resolution `nside` contains:

```python
npix = 12 * nside**2
```

Typical shapes are:

```python
# one map
x.shape == (12 * nside**2,)

# batch of maps
x.shape == (n_maps, 12 * nside**2)
```

## 2D maps

For planar data, FOSCAT exposes analogous operators through `foscat.scat_cov2D`.

```python
import foscat.scat_cov2D as sc2d

scat_op = sc2d.funct(KERNELSZ=5, NORIENT=4, all_type='float64')
```

## Differentiable synthesis

Synthesis is an optimization loop:

1. compute reference statistics from the target;
2. initialize a candidate map, often with Gaussian noise;
3. minimize a differentiable loss between target and candidate statistics;
4. return a synthetic field matching the target statistics.
