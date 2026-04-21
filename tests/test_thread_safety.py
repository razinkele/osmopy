"""Tests that simulation state is isolated per-run (no module globals)."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

from osmose.engine._netcdf import open_dataset_safe
from osmose.engine.processes.predation import enable_diet_tracking, get_diet_matrix
from osmose.engine.simulate import SimulationContext


def test_diet_matrix_not_global():
    """get_diet_matrix without context should return None."""
    matrix = get_diet_matrix()
    assert matrix is None, "Diet matrix should not persist as module-level state"


def test_diet_matrix_with_context():
    """get_diet_matrix returns the matrix from the context when tracking is enabled."""
    ctx = SimulationContext()
    enable_diet_tracking(n_schools=3, n_species=2, ctx=ctx)
    matrix = get_diet_matrix(ctx=ctx)
    assert matrix is not None
    assert matrix.shape == (3, 2)


def test_contexts_are_independent():
    """Two SimulationContext instances should not share state."""
    ctx1 = SimulationContext()
    ctx2 = SimulationContext()

    ctx1.diet_tracking_enabled = True
    ctx1.diet_matrix = np.ones((2, 2), dtype=np.float64)
    ctx1.tl_weighted_sum = np.ones(5, dtype=np.float64)

    assert ctx2.diet_tracking_enabled is False
    assert ctx2.diet_matrix is None
    assert ctx2.tl_weighted_sum is None


def test_config_dir_isolated():
    """config_dir should be per-context, not shared."""
    ctx1 = SimulationContext(config_dir="/path/a")
    ctx2 = SimulationContext(config_dir="/path/b")
    assert ctx1.config_dir == "/path/a"
    assert ctx2.config_dir == "/path/b"


def test_open_dataset_safe_releases_handle(tmp_path):
    """open_dataset_safe should eagerly load and close the file handle.

    The returned Dataset must be usable after the source file is deleted.
    """
    path = tmp_path / "forcing.nc"
    src = xr.Dataset({"biomass": (("time", "y", "x"), np.ones((3, 4, 5)))})
    src.to_netcdf(path)

    ds = open_dataset_safe(path)
    path.unlink()

    assert ds["biomass"].shape == (3, 4, 5)
    assert ds["biomass"].values.sum() == 60


def test_open_dataset_safe_concurrent_threads(tmp_path):
    """Regression: parallel calibration at n_parallel>1 used to abort with
    'NetCDF: Can't open HDF5 attribute' / 'double free or corruption' because
    the netCDF4 C extension is not thread-safe for concurrent opens. This
    test exercises the serialized+eager-load path on many threads at once.
    """
    path = tmp_path / "forcing.nc"
    src = xr.Dataset({"biomass": (("time", "y", "x"), np.arange(60.0).reshape(3, 4, 5))})
    src.to_netcdf(path)

    def _open_and_read() -> float:
        ds = open_dataset_safe(path)
        return float(ds["biomass"].values.sum())

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _open_and_read(), range(32)))

    expected = float(np.arange(60.0).sum())
    assert all(r == expected for r in results)
