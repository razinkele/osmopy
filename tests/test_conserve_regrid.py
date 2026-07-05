import numpy as np
from osmose.forcing.conserve_regrid import split_conserve


def test_total_conserved():
    f = np.array([[[4.0, 8.0], [0.0, 16.0]]])
    out = split_conserve(f, 2)
    assert out.shape == (1, 4, 4) and np.isclose(out.sum(), f.sum())
    assert np.allclose(out[0, 0:2, 0:2], 1.0)


def test_not_x16_regression():
    f = np.ones((1, 5, 5))
    assert np.isclose(split_conserve(f, 4).sum(), f.sum())
