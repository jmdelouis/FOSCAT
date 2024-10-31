from dataclasses import dataclass, field

import numpy as np

import foscat.scat_cov as sc


@dataclass
class Parameters:
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
