# Scattering covariance

The two most common entry points are:

```python
import foscat.scat_cov as sc       # HEALPix / spherical maps
import foscat.scat_cov2D as sc2d   # 2D planar maps
```

Create an operator:

```python
scat_op = sc.funct(
    KERNELSZ=5,
    NORIENT=4,
    OSTEP=1,
    all_type='float64',
)
```

For 2D fields:

```python
scat_op = sc2d.funct(
    KERNELSZ=5,
    NORIENT=4,
    all_type='float64',
)
```

## Important parameters

| Parameter | Meaning |
|---|---|
| `KERNELSZ` | Spatial support of the local wavelet kernel |
| `NORIENT` | Number of wavelet orientations |
| `OSTEP` | Orientation sampling step, used mainly in spherical/HEALPix workflows |
| `all_type` | Numerical precision/backend dtype |

## Computing statistics

```python
target_stat = scat_op.eval(target_map)
```

The returned object contains scattering-covariance coefficients and helper methods used by the synthesis code. In notebooks, losses are usually expressed by comparing two such objects.
