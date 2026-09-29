"""H9: Numba parallel-vs-sequential JIT determinism.

Runs a short Python-engine simulation of the ``eec_full`` fixture at 1 thread and at several
threads and requires the per-step biomass/abundance series to be EXACTLY equal.

Fixture choice is load-bearing (#113). This file used ``data/minimal``, whose grid mask has
ZERO ocean cells: every school stays at cell (-1, -1), ``mortality()``'s ``valid_indices`` is
empty every step, and the parallel mortality kernel is never dispatched -- so the comparison
passed whatever the kernel did. ``eec_full`` has 460 ocean cells, and ``_run_with_threads``
asserts ``ocean_mask.sum() > 1`` so the test can never go vacuous again. The sibling guard
``tests/test_thread_policy.py::test_mortality_bit_identical_across_thread_counts`` uses the same
fixture for the same reason.

Thread count is set with ``numba.set_num_threads`` only. The old helper also assigned the
``NUMBA_NUM_THREADS`` environment variable with no restore, a process-global leak inherited by
any later ``forkserver`` child; the conftest autouse ``_restore_numba_thread_state`` fixture puts
the runtime count back after each test.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# numba may not always be available in some test environments
try:
    import numba as _nb
except ImportError:  # pragma: no cover
    _nb = None


REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
_EEC = REPO / "data" / "eec_full" / "eec_all-parameters.csv"


def _run_with_threads(n_threads: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Run eec_full for 1 year at a fixed Numba thread count; return numeric output arrays."""
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.engine.grid import Grid

    cfg = OsmoseConfigReader().read(str(_EEC))
    cfg = {k: v for k, v in cfg.items() if k != ""}

    grid = Grid.from_netcdf(
        str(_EEC.parent / cfg["grid.netcdf.file"]), mask_var=cfg.get("grid.var.mask", "mask")
    )
    n_ocean = int(grid.ocean_mask.sum())
    assert n_ocean > 1, (
        f"fixture has {n_ocean} ocean cells: the parallel mortality kernel would never run and "
        "this determinism test would pass vacuously (#113)"
    )

    cfg["simulation.rng.fixed"] = "true"
    cfg["simulation.time.nyear"] = "1"  # determinism, not science

    if _nb is not None:
        _nb.set_num_threads(n_threads)
    results = PythonEngine().run_in_memory(cfg, seed=seed)

    def _numeric_array(df) -> np.ndarray:
        numeric = df.select_dtypes(include=["number"]).drop(columns=["Time"], errors="ignore")
        return numeric.to_numpy(dtype=np.float64)

    return {
        "biomass_full": _numeric_array(results.biomass()),
        "abundance_full": _numeric_array(results.abundance()),
    }


@pytest.mark.filterwarnings("ignore:Swapping size ratios")
@pytest.mark.skipif(_nb is None, reason="numba not installed")
@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >=2 cores to compare thread counts")
def test_mortality_deterministic_across_thread_counts() -> None:
    """Per-step biomass/abundance must be EXACTLY equal at 1 thread and at several threads."""
    many = min(4, os.cpu_count() or 1)
    assert many > 1  # never degenerate into comparing a run against itself
    seq = _run_with_threads(n_threads=1, seed=42)
    par = _run_with_threads(n_threads=many, seed=42)
    np.testing.assert_array_equal(
        seq["biomass_full"], par["biomass_full"], err_msg=f"biomass diverges, 1 vs {many} threads"
    )
    np.testing.assert_array_equal(
        seq["abundance_full"],
        par["abundance_full"],
        err_msg=f"abundance diverges, 1 vs {many} threads",
    )


@pytest.mark.filterwarnings("ignore:Swapping size ratios")
@pytest.mark.skipif(_nb is None, reason="numba not installed")
def test_single_thread_is_deterministic() -> None:
    """Single-threaded reruns of the same config must be byte-equal.

    This is a stronger floor than H9: regardless of whether multi-threading
    is deterministic, two runs at 1 Numba thread with the same seed
    must produce identical output. Pre-empts a class of regressions where
    a per-call non-determinism (e.g. dict-iteration order) leaks into the
    engine state.
    """
    a = _run_with_threads(n_threads=1, seed=42)
    b = _run_with_threads(n_threads=1, seed=42)
    np.testing.assert_array_equal(
        a["biomass_full"],
        b["biomass_full"],
        err_msg="single-thread reruns produced different biomass time-series",
    )
    np.testing.assert_array_equal(
        a["abundance_full"],
        b["abundance_full"],
        err_msg="single-thread reruns produced different abundance time-series",
    )
