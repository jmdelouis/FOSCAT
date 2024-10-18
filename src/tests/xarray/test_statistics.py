import xarray as xr
from dataclasses import dataclass, field
import pytest

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst

from foscat.xarray import statistics
import foscat.scat_cov as sc

@dataclass
class Backend:
    name: str
    BACKEND: int = field(init=False)

    def __post_init__(self):
        self.BACKEND = statistics.backends.index(self.name)


def scat_cov():
    backends = st.builds(Backend, st.sampled_from(statistics.backends))

    batch_size = st.integers(min_value=1, max_value=5)
    type_size = st.just(2)
    masks_size = st.integers(min_value=1, max_value=5)
    scales_size = st.integers(min_value=1, max_value=6)

    orientations_size = st.integers(min_value=1, max_value=8)
    st.shared(masks_size, key="scales1")

    S0 = npst.arrays(
        dtype=st.just("float64"), shape=st.tuples(st.shared(batch_size, key="batches"), type_size)
    )
    S1 = npst.arrays(
        dtype=st.just("float64"),
        shape=st.tuples(
            st.shared(batch_size, key="batches"),
            st.shared(masks_size, key="masks"),
            st.shared(scales_size, key="scales1"),
            st.shared(orientations_size, key="orientations")
        ),
    )
    P00 = npst.arrays(
        dtype=st.just("float64"),
        shape=st.tuples(
            st.shared(batch_size, key="batches"),
            st.shared(masks_size, key="masks"),
            st.shared(scales_size, key="scales1"),
            st.shared(orientations_size, key="orientations")
        ),
    )
    C01 = npst.arrays(
        dtype=st.just("complex128"),
        shape=st.tuples(
            st.shared(batch_size, key="batches"),
            st.shared(masks_size, key="masks"),
            st.shared(scales_size, key="scales2"),
            st.shared(orientations_size, key="orientations"),
            st.shared(orientations_size, key="orientations")
        ),
    )
    C10 = npst.arrays(
        dtype=st.just("complex128"),
        shape=st.tuples(
            st.shared(batch_size, key="batches"),
            st.shared(masks_size, key="masks"),
            st.shared(scales_size, key="scales2"),
            st.shared(orientations_size, key="orientations"),
            st.shared(orientations_size, key="orientations"),
        ),
    )
    C11 = npst.arrays(
        dtype=st.just("complex128"),
        shape=st.tuples(
            st.shared(batch_size, key="batches"),
            st.shared(masks_size, key="masks"),
            st.shared(scales_size, key="scales3"),
            st.shared(orientations_size, key="orientations"),
            st.shared(orientations_size, key="orientations"),
            st.shared(orientations_size, key="orientations"),
        ),
    )

    return st.builds(
        sc.scat_cov,
        s0=S0,
        p00=P00,
        c01=C01,
        c10=C10 | st.just(None),
        c11=C11,
        s1=S1,
        backend=backends,
    )


@pytest.mark.parametrize("batch_dim", ["batches", "batches1"])
@given(scat_cov())
def test_scat_cov_to_xarray(batch_dim, scat_cov):
    actual = statistics._scat_cov_to_xarray(scat_cov, batch_dim=batch_dim)

    assert isinstance(actual, xr.Dataset)
    assert batch_dim in actual.dims
    assert actual.attrs["foscat_backend"] == scat_cov.backend.name
