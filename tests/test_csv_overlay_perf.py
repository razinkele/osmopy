import time

import numpy as np
import pandas as pd

from ui.pages.grid_helpers import load_csv_overlay


def test_csv_overlay_performance(tmp_path):
    """Large grid should complete in under 500ms."""
    p = tmp_path / "large.csv"
    ny, nx = 100, 200
    np.random.seed(123)
    data = np.random.rand(ny, nx) * 10
    data[0, :] = -99
    pd.DataFrame(data).to_csv(p, sep=";", header=False, index=False)
    start = time.perf_counter()
    cells = load_csv_overlay(p, ul_lat=50.0, ul_lon=-5.0, lr_lat=43.0, lr_lon=5.0, nx=nx, ny=ny)
    elapsed = time.perf_counter() - start
    assert cells is not None
    assert elapsed < 0.5, f"load_csv_overlay took {elapsed:.3f}s for {ny}x{nx} grid"
