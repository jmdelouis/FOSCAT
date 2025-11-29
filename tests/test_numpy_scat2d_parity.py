import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from foscat import scat_cov2D
from foscat.numpy_scat2d import compute_scattering


torch = pytest.importorskip("torch", reason="torch backend required for parity check")


@pytest.mark.parametrize("kernel_size", [5])
def test_numpy_matches_foscat(kernel_size):
    rng = np.random.default_rng(0)
    image = rng.standard_normal((32, 32), dtype=np.float32)

    numpy_scat = compute_scattering(image, KERNELSZ=kernel_size)

    focus = scat_cov2D.funct(
        BACKEND="torch",
        NORIENT=4,
        LAMBDA=1.2,
        KERNELSZ=kernel_size,
        slope=1.0,
        DODIV=False,
        silent=True,
        use_median=False,
    )
    focus_scat = focus.eval(image)

    def to_numpy(val):
        if hasattr(val, "detach"):
            return val.detach().cpu().numpy()
        if hasattr(val, "numpy"):
            return val.numpy()
        return np.asarray(val)

    torch_s0 = np.squeeze(to_numpy(focus_scat.S0))
    torch_s1 = np.squeeze(to_numpy(focus_scat.S1))
    torch_s2 = np.squeeze(to_numpy(focus_scat.S2))
    torch_s2l = np.squeeze(to_numpy(focus_scat.S2L))

    assert np.allclose(torch_s0, numpy_scat.S0, rtol=1e-5, atol=1e-6)
    assert torch_s1.shape == numpy_scat.S1.shape
    assert torch_s2.shape == numpy_scat.S2.shape
    assert torch_s2l.shape == numpy_scat.S2L.shape

    assert np.allclose(torch_s1, numpy_scat.S1, rtol=1e-4, atol=1e-5)
    assert np.allclose(torch_s2, numpy_scat.S2, rtol=1e-4, atol=1e-5)
    assert np.allclose(torch_s2l, numpy_scat.S2L, rtol=1e-4, atol=1e-5)

    assert np.array_equal(focus_scat.j1, numpy_scat.j1)
    assert np.array_equal(focus_scat.j2, numpy_scat.j2)
