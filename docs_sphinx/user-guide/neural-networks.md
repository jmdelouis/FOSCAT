# HEALPix Neural Networks

FOSCAT provides neural-network architectures designed to operate directly on
HEALPix geometry using **oriented spherical convolutions** as the spatial
primitive. Unlike standard convolutions applied to equirectangular projections,
FOSCAT's convolutions are defined on the sphere and do not introduce projection
distortions.

This workflow corresponds to `CNN_local.ipynb` and `CNN_ecmwf.ipynb` in
`demo-foscat-pangeo-eosc`.

---

## Spatial primitive: oriented spherical convolution

All neural-network modules in FOSCAT use `SphericalStencil.Convol_torch` as the
convolution primitive. For each HEALPix pixel $p$, a $K \times K$ stencil of
neighbour pixel indices is looked up from a precomputed table; the convolution is
then a dot product between the kernel weights and the neighbour values.

The stencil tables are the same ones used by `FoCUS` for the scattering operator,
and are cached in `~/.FOSCAT/data/`. This means the first call at a given
`(nside, KERNELSZ)` pair triggers a one-time initialisation.

---

## `HealpixUNet`

**Module:** `foscat.healpix_unet_torch`

A U-Net-style encoder–decoder on the HEALPix sphere. Each level applies two
oriented spherical convolutions with BatchNorm and ReLU, then downsamples or
upsamples via `ud_grade_2` / `up_grade`. Skip connections carry encoder features
to the decoder at matching resolutions.

### Architecture

```
Input  [B, C_in, N_cells]     N_cells = len(cell_ids)

  enc[0]  DoubleConv  C_in     → chanlist[0]
  down[0] ud_grade_2           N_cells → N_cells/4
  enc[1]  DoubleConv  chanlist[0] → chanlist[1]
  down[1] ud_grade_2
  ...
  enc[L]  DoubleConv  chanlist[L-1] → chanlist[L]   (bottleneck)

  up[0]   up_grade             N_cells/4^L → N_cells/4^(L-1)
  dec[0]  DoubleConv  chanlist[L]+chanlist[L-1] → chanlist[L-1]  (concat skip)
  ...
  dec[L-1] DoubleConv chanlist[1]+chanlist[0] → chanlist[0]

  out_conv  Conv1d(chanlist[0], out_channels, 1)  → [B, out_channels, N_cells]

Output [B, out_channels, N_cells]
```

Each "DoubleConv" block: `SphericalConv → BN → ReLU → SphericalConv → BN → ReLU`.

### Constructor

```python
from foscat.healpix_unet_torch import HealpixUNet

model = HealpixUNet(
    in_nside        = 64,
    n_chan_in        = 1,
    chanlist         = [16, 32, 64],
    cell_ids         = cell_ids,
    KERNELSZ         = 3,
    gauge_type       = "phi",
    G                = 1,
    task             = "regression",
    out_channels     = 1,
    final_activation = None,
    device           = None,
    down_type        = "max",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_nside` | int | — | HEALPix nside for the input domain (NESTED). |
| `n_chan_in` | int | — | Number of input channels. |
| `chanlist` | list[int] | — | Channel count at each encoder level, e.g. `[16, 32, 64]`. Depth = `len(chanlist)`. |
| `cell_ids` | ndarray | — | NESTED pixel indices of the regional domain at `in_nside`. Shape `(N_cells,)`. |
| `KERNELSZ` | int | 3 | Stencil side length $K$ ($K \times K$ neighbours per pixel). |
| `gauge_type` | str | `"phi"` | Orientation convention for the gauge frame. `"phi"`: aligned with the longitude direction (preferred for Earth-observation). `"cosmo"`: standard cosmological convention. |
| `G` | int | 1 | Number of gauge orientations. `G > 1` increases intermediate channel count by $G$. |
| `task` | str | `"regression"` | Output head: `"regression"` (no final activation by default) or `"segmentation"` (softmax / sigmoid). |
| `out_channels` | int | 1 | Number of output channels (e.g. number of classes for segmentation). |
| `final_activation` | str\|None | None | Override the default activation: `"none"`, `"sigmoid"`, or `"softmax"`. |
| `device` | str\|device\|None | auto | Target device. Defaults to CUDA if available; falls back to CPU if FOSCAT ops cannot run on CUDA. |
| `down_type` | str | `"max"` | Downsampling strategy: `"max"` (max over 4 NESTED children) or `"mean"` (average). |
| `prefer_foscat_gpu` | bool | True | Try CUDA for FOSCAT ops and fall back to CPU if a dry-run fails. |

### Methods

**`forward(x) → Tensor`**

```python
# x: (B, C_in, N_cells)
y = model(x)    # (B, out_channels, N_cells)
```

**`fit(x_train, y_train, ...) → dict`**

```python
history = model.fit(
    x_train,           # (N, C_in, N_cells)
    y_train,           # (N, out_channels, N_cells)
    x_val   = None,
    y_val   = None,
    n_epoch     = 100,
    batch_size  = 16,
    lr          = 1e-3,
    weight_decay = 1e-5,
    view_epoch  = 10,
    loss_fn     = None,    # defaults to F.mse_loss
)
# returns {"train_loss": [...], "val_loss": [...]}
```

**`predict(x, batch_size=16) → ndarray`**

Batched inference without gradient. Returns `(N, out_channels, N_cells)` on CPU.

### Example — regression on a regional domain

```python
import numpy as np
import torch
from foscat.healpix_unet_torch import HealpixUNet
import healpy as hp

nside = 64
cell_ids = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0), np.radians(30), nest=True)

model = HealpixUNet(
    in_nside=nside, n_chan_in=6, chanlist=[16, 32, 64],
    cell_ids=cell_ids, KERNELSZ=3, out_channels=1,
)

# Training data: X_atm shape (N, 6, N_cells), Y_sst shape (N, 1, N_cells)
history = model.fit(X_atm, Y_sst, x_val=X_val, y_val=Y_val,
                    n_epoch=100, lr=1e-3)

sst_pred = model.predict(X_test)   # (N_test, 1, N_cells)
```

---

## `GCNN` — graph-convolutional neural network

**Module:** `foscat.GCNN`

A graph-convolutional network that uses the FOSCAT scattering operator as the
convolution layer. Suitable for regression tasks over the full sphere or a
regional subset.

```python
from foscat.GCNN import GCNN

model = GCNN(
    nparam   = 1,       # number of output scalars per pixel
    KERNELSZ = 3,
    NORIENT  = 4,
    chanlist = [1, 16, 32, 16, 1],
    in_nside = 64,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nparam` | int | 1 | Number of output channels per pixel. |
| `KERNELSZ` | int | 3 | Wavelet stencil side length. |
| `NORIENT` | int | 4 | Number of wavelet orientations. |
| `chanlist` | list[int] | [] | Channel sizes at each layer. The network depth equals `len(chanlist) - 1`. |
| `in_nside` | int | 1 | Input HEALPix nside. |

---

## Choosing between architectures

| Architecture | Best for | Notes |
|---|---|---|
| `HealpixUNet` | Dense prediction, regression, segmentation | Full U-Net with skip connections; most flexible |
| `GCNN` | Global scalar regression, lightweight models | Scattering-based graph conv; fewer parameters |
| `CNN` | Simple per-pixel classification | Flat CNN; use as baseline |
| `healpix_vit_torch.HealpixViT` | Long-range dependencies, global patterns | Vision Transformer on HEALPix tokens |

---

## Hybrid scattering + neural workflows

FOSCAT's neural networks are differentiable and can be embedded in synthesis or
component separation loops. For example, a trained `HealpixUNet` can serve as a
learned morphological prior:

```python
from foscat.Synthesis import Loss, Synthesis

# trained_model: a HealpixUNet that maps noise → clean field

def neural_prior_loss(x, scat_op, args):
    model = args[0]
    x_clean = model(torch.tensor(x).unsqueeze(0).unsqueeze(0).float())
    return scat_op.backend.bk_mean((x - x_clean.squeeze().numpy()) ** 2)

loss = Loss(neural_prior_loss, scat_op, trained_model)
solver = Synthesis([loss])
result = solver.run(noisy_map, NUM_EPOCHS=200)
```
