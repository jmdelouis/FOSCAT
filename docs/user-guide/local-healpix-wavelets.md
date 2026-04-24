# Local HEALPix wavelets

FOSCAT also supports local processing on incomplete HEALPix domains. This is important when observations do not cover the full sphere or when one wants to train a model on a regional patch while preserving spherical geometry.

This workflow corresponds to `local_foscat.ipynb` and `CNN_local.ipynb`.

## Why local HEALPix?

Projection to a plane can introduce distortions. Local HEALPix operators keep the data on the sphere and operate with neighborhood relations derived from the HEALPix geometry.

## Typical ingredients

- A list of valid HEALPix cells/pixels.
- Data values associated with those cells.
- Resolution changes through local downsampling/upsampling.
- Local wavelet or neural convolution layers.

## Practical advice

- Keep the same HEALPix ordering across all arrays.
- Store the valid cell indices explicitly.
- Verify that local downsampling and upsampling preserve alignment with the original domain.
- For masked or incomplete data, distinguish missing values from physical zeros.
