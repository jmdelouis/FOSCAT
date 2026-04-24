# Component separation

Component separation uses scattering or wavelet statistics as constraints to separate mixed physical fields.

The `Remove_CMB.ipynb` notebook illustrates a CMB-like removal problem. The generic structure is:

1. start from an observed mixture;
2. define one or several components to estimate;
3. compute statistics or losses that encourage each component to match expected morphology;
4. optimize the component maps jointly or sequentially.

## Generic sketch

```python
mixture = signal + contaminant

# Estimate signal_hat and contaminant_hat such that:
# mixture ~= signal_hat + contaminant_hat
# Phi(signal_hat) matches signal statistics
# Phi(contaminant_hat) matches contaminant statistics
```

FOSCAT is useful here because scattering statistics capture morphology beyond a simple power spectrum.
