from dataclasses import dataclass, field

import numpy as np

import foscat.scat_cov as sc


@dataclass
class Parameters:
    """
    parameters for the scattering covariance transform

    Parameters
    ----------
    n_orientations : int
        The number of orientations of the kernel
    kernel_size : int
        The size of the kernel in cells / pixels
    jmax_delta : float
        Typically, the scattering covariance coefficients are calculated over N scales,
        starting from the smallest scale (j=0) up to the largest scale (j_{\text{max}}=\log_2(\text{nside})).
        Adjusting this parameter eliminates the coefficients at the largest scale
        where j_{\text{max}}=\log_2(\text{nside})-j_{\text{max\_delta}}.
    dtype : str or numpy.dtype, default: "float64"
        The dtype to use for the transform.
    backend : {"numpy", "torch", "tensorflow"}, default: "tensorflow"
        The compute backend.
    """

    n_orientations: int
    kernel_size: int
    jmax_delta: float
    dtype: str | np.dtype = "float64"
    backend: str = "tensorflow"

    cache: sc.funct | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.cache = sc.funct(
            NORIENT=self.n_orientations,
            KERNELSZ=self.kernel_size,
            JmaxDelta=self.jmax_delta,
            all_type=self.dtype,
            BACKEND=self.backend,
        )
