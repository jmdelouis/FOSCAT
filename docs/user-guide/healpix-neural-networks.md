# HEALPix neural networks

FOSCAT contains neural-network modules designed to operate on HEALPix geometry.

Relevant modules include:

```python
from foscat.healpix_unet_torch import HealpixUNet
```

and older TensorFlow-oriented modules such as:

```python
from foscat.UNET import UNET
```

## Global HEALPix U-Net

A HEALPix U-Net follows an encoder/decoder design:

1. apply local spherical convolutions;
2. reduce resolution across HEALPix levels;
3. expand back to the original resolution;
4. combine skip connections.

## Local HEALPix U-Net

For regional domains, provide valid cell identifiers so the model only processes the available subset of the sphere.

```python
model = HealpixUNet(
    nside=64,
    in_channels=1,
    out_channels=1,
)
```

Parameter names may evolve; check the installed source for the exact constructor signature.

## Use cases

- Learning mappings between geophysical fields.
- Filling missing spherical observations.
- Emulating physical operators on the sphere.
- Hybrid scattering/neural workflows.
