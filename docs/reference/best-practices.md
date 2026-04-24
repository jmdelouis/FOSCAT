# Best practices

## Normalize inputs

For synthesis, remove large offsets and normalize amplitudes before optimization. Store the inverse transform so the generated field can be mapped back to physical units.

## Keep HEALPix ordering consistent

Most examples assume `nest=True`. Mixing nested and ring ordering is a common source of silent errors.

## Treat missing data explicitly

For incomplete observations, keep a mask or list of valid cell ids. Do not encode missing data as zero unless zero is physically meaningful.

## Start small

Begin with low `nside`, small images, and a small number of epochs. Increase resolution and optimization length after validating the workflow.

## Tune wavelet parameters

- Larger `KERNELSZ` captures broader local structures but increases cost.
- Larger `NORIENT` captures more directional information but increases the number of coefficients.
- Multi-resolution synthesis is often more stable for fields with strong large-scale structure.

## Backend choice

Use TensorFlow for compatibility with older synthesis notebooks. Use PyTorch for recent HEALPix neural-network modules.
