# Changelog

## 2026.04.1 (current)

### Added

- `HealpixUNet` — PyTorch U-Net on HEALPix geometry with oriented spherical
  convolutions, BatchNorm, and ReLU; supports regional domains via `cell_ids`.
- `HealpixViT` / `HealpixViTSkip` — Vision Transformer architectures on HEALPix
  tokens with optional skip connections.
- `PlanarViT` — Vision Transformer on equirectangular grids.
- `alm_loc_optim` — optimised local spherical harmonic routines.
- xarray accessor (`foscat.xarray`) — integrates FOSCAT statistics with
  xarray-labelled datasets.
- MPI-parallel synthesis improvements (`isMPI=True`).
- `DODIV` mode in `FoCUS` — additional divergence-sensitive wavelet orientations
  for polarisation analysis.

### Changed

- **PyTorch is now the only active backend.** TensorFlow and NumPy backends
  remain in the source tree for reference but are not maintained and will raise
  a `NotImplementedError` at construction time.
- `FoCUS` version string updated to `"2026.04.1"`.
- Default `TEMPLATE_PATH` is `~/.FOSCAT/data/` (unchanged).

### Deprecated

- `foscat.BkTensorflow` — TensorFlow backend. Use PyTorch.
- `foscat.BkNumpy` — NumPy backend. Use PyTorch.
- `foscat.UNET` — legacy TensorFlow U-Net. Use `HealpixUNet`.

---

## Earlier versions

Version history prior to 2026 is tracked in the git log:

```bash
git log --oneline --follow src/foscat/FoCUS.py
```
