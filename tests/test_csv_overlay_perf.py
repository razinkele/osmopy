import time

import numpy as np
import pandas as pd

from ui.pages.grid_helpers import load_csv_overlay


def test_csv_overlay_performance(tmp_path):
    """Large grid should parse in well under 1.5s of CPU time.

    Two-part contention hardening so the guard survives pytest-xdist -n auto:
    - process_time (CPU time), not perf_counter (wall-clock): immune to the
      scheduler descheduling this process while other workers run.
    - a 1.5s ceiling (~5-6x the serial cost) rather than a tight 0.5s: under
      full core saturation even CPU time inflates (memory-bandwidth contention
      means each CPU-second does less work; measured 0.53s at -n 28). The loose
      ceiling absorbs that while still catching a gross (>5x) regression — which
      is all an absolute-time microbenchmark can reliably detect anyway.
    """
    p = tmp_path / "large.csv"
    ny, nx = 100, 200
    np.random.seed(123)
    data = np.random.rand(ny, nx) * 10
    data[0, :] = -99
    pd.DataFrame(data).to_csv(p, sep=";", header=False, index=False)
    start = time.process_time()
    cells = load_csv_overlay(p, ul_lat=50.0, ul_lon=-5.0, lr_lat=43.0, lr_lon=5.0, nx=nx, ny=ny)
    elapsed = time.process_time() - start
    assert cells is not None
    assert elapsed < 1.5, f"load_csv_overlay took {elapsed:.3f}s CPU for {ny}x{nx} grid"
