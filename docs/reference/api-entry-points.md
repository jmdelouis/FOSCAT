# API entry points

This page lists the practical entry points used by the examples.

## Scattering covariance

```python
import foscat.scat_cov as sc
import foscat.scat_cov2D as sc2d
```

Main classes/functions:

- `foscat.scat_cov.funct`: HEALPix/spherical scattering-covariance operator.
- `foscat.scat_cov2D.funct`: 2D scattering-covariance operator.
- `eval(x)`: compute statistics for an input field.

## Synthesis

```python
from foscat.Synthesis import Synthesis
```

Main usage:

```python
solver = Synthesis(loss_object)
result = solver.run(x0, NUM_EPOCHS=50, EVAL_FREQUENCY=10)
```

The `loss_object` is expected to provide an `eval(self, x, batch, return_all=False)` method.

## HEALPix neural networks

```python
from foscat.healpix_unet_torch import HealpixUNet
from foscat.UNET import UNET
```

Use the Torch implementation for recent workflows where possible. The older `UNET` module is useful for compatibility with historical notebooks.

## Local and geometric operators

Relevant modules in the source tree include:

- `foscat.HealBili`
- `foscat.HealSpline`
- `foscat.SphericalStencil`
- `foscat.SphereDownGeo`
- `foscat.SphereUpGeo`
- `foscat.HOrientedConvol`

These support interpolation, local stencils, down/up sampling, and oriented spherical convolution.
