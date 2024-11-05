from dataclasses import dataclass, field

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import given

import foscat.scat_cov as sc
from foscat.xarray import statistics


@dataclass
class Backend:
    name: str
    BACKEND: int = field(init=False)

    def __post_init__(self):
        inverted_backends = {value: key for key, value in statistics.backends.items()}
        self.BACKEND = inverted_backends[self.name]


class FakeParameters:
    def __init__(self, cache):
        self.cache = cache


class _Parameters:
    def __init__(self, backend):
        self.backend = Backend(backend)

    def eval(self, data1, data2=None, *, calc_var=False, **kwargs):
        ref = sc.scat_cov(
            s0=np.zeros(shape=(3, 2)),
            p00=np.zeros(shape=(3, 1, 5, 4)),
            c01=np.zeros(shape=(3, 1, 3, 4, 4)),
            c10=None,
            c11=np.zeros(shape=(3, 1, 7, 4, 4, 4)),
            s1=np.zeros(shape=(3, 1, 5, 4)),
            backend=self.backend,
        )

        if calc_var:
            return ref, ref
        else:
            return ref


def scat_cov():
    backends = st.builds(Backend, st.sampled_from(list(statistics.backends.values())))

    batch_size = st.integers(min_value=1, max_value=5)
    type_size = st.just(2)
    masks_size = st.integers(min_value=1, max_value=5)
    scales_size = st.integers(min_value=1, max_value=6)

    orientations_size = st.integers(min_value=1, max_value=8)
    st.shared(masks_size, key="scales1")

    S0 = npst.arrays(
        dtype=st.just("float64"),
        shape=st.tuples(st.shared(batch_size, key="batches"), type_size),
    )
    S1 = npst.arrays(
        dtype=st.just("float64"),
        shape=st.tuples(
            st.shared(batch_size, key="batches"),
            st.shared(masks_size, key="masks"),
            st.shared(scales_size, key="scales1"),
            st.shared(orientations_size, key="orientations"),
        ),
    )
    P00 = npst.arrays(
        dtype=st.just("float64"),
        shape=st.tuples(
            st.shared(batch_size, key="batches"),
            st.shared(masks_size, key="masks"),
            st.shared(scales_size, key="scales1"),
            st.shared(orientations_size, key="orientations"),
        ),
    )
    C01 = npst.arrays(
        dtype=st.just("complex128"),
        shape=st.tuples(
            st.shared(batch_size, key="batches"),
            st.shared(masks_size, key="masks"),
            st.shared(scales_size, key="scales2"),
            st.shared(orientations_size, key="orientations"),
            st.shared(orientations_size, key="orientations"),
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


@pytest.mark.parametrize("variances", [True, False])
@pytest.mark.parametrize("backend", ["numpy", "torch", "tensorflow"])
def test_reference_statistics(variances, backend):
    params = FakeParameters(cache=_Parameters(backend))
    arr = xr.DataArray(
        [[0, 0], [1, 1], [2, 2]],
        dims=("time", "cells"),
        coords={"cell_ids": ("cells", [0, 1]), "time": [-1, 0, 1]},
    )

    actual = statistics.reference_statistics(arr, params, variances=variances)
    data_vars = {
        "S0": (["time", "type"], np.zeros(shape=(3, 2))),
        "P00": (
            ["time", "masks", "scales1", "orientations_1"],
            np.zeros(shape=(3, 1, 5, 4)),
        ),
        "C01": (
            ["time", "masks", "scales2", "orientations_1", "orientations_2"],
            np.zeros(shape=(3, 1, 3, 4, 4)),
        ),
        "C11": (
            [
                "time",
                "masks",
                "scales3",
                "orientations_1",
                "orientations_2",
                "orientations_3",
            ],
            np.zeros(shape=(3, 1, 7, 4, 4, 4)),
        ),
        "S1": (
            ["time", "masks", "scales1", "orientations_1"],
            np.zeros(shape=(3, 1, 5, 4)),
        ),
    }
    variance_vars = {f"var_{n}": v for n, v in data_vars.items()} if variances else {}

    expected = xr.Dataset(
        data_vars=data_vars | variance_vars,
        coords={"time": [-1, 0, 1], "types": ("type", ["mean", "variance"])},
        attrs={"foscat_backend": backend, "use_1d": False},
    )
    xr.testing.assert_equal(actual, expected)
