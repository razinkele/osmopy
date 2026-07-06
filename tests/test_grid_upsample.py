import numpy as np
from osmose.forcing.grid_upsample import block_replicate


def test_block_replicate_preserves_values_and_shape():
    a = np.array([[-99.0, 1.0], [0.0, 0.5]])
    out = block_replicate(a, 2)
    assert out.shape == (4, 4)
    assert np.array_equal(out[0:2, 0:2], np.full((2, 2), -99.0))
    assert np.array_equal(out[2:4, 2:4], np.full((2, 2), 0.5))
