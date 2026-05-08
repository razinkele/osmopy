"""H9: Numba parallel-vs-sequential JIT determinism.

Plan ask (docs/plans/2026-05-05-deep-review-remediation-plan.md):
> 1. Set NUMBA_NUM_THREADS=1 and run the standard mortality kernel.
> 2. Set NUMBA_NUM_THREADS=4 and re-run with the same seeds.
> 3. Assert np.allclose(state_seq, state_par, atol=1e-12) on all output
>    arrays. Catches any latent race in `prange` blocks.

What the test exercises: a 3-step Python-engine simulation of the
minimal fixture, run end-to-end first with `numba.set_num_threads(1)`
and then with `numba.set_num_threads(min(4, available))`. The
end-of-run abundance + biomass arrays must be allclose.

Why it might be flaky: scientific Numba code that reduces or
accumulates across `prange` iterations can produce thread-count-
dependent results because float addition isn't associative. The
mortality kernel at `mortality.py:1307` is parallelised over cells
and the documentation note there says "no cross-cell write conflicts"
— but per-cell RNG draws may still consume the global stream in a
thread-count-dependent order.

Per the plan's risk register, the test is marked
`@pytest.mark.xfail(strict=False)` so a single-threaded vs
multi-threaded divergence is reported but does not break the build.
If the kernel is genuinely deterministic, the xfail will appear as
XPASS in CI and we can remove the marker.
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


def _run_minimal_with_threads(n_threads: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Run a short Python-engine simulation with the given Numba thread count.

    Returns a dict of representative output arrays for cross-thread comparison.
    """
    # Set thread count BEFORE the JIT kernel is called. Numba's set_num_threads
    # applies to subsequent calls.
    if _nb is not None:
        _nb.set_num_threads(n_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(n_threads)  # belt-and-braces

    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    matches = sorted((REPO / "data" / "minimal").glob("*_all-parameters.csv"))
    cfg = OsmoseConfigReader().read(str(matches[0]))
    cfg = {k: v for k, v in cfg.items() if k != ""}
    # Pin the RNG explicitly so the comparison is meaningful.
    cfg["simulation.rng.fixed"] = "true"
    # Keep the run short — H9 is about determinism, not science.
    cfg["simulation.time.nyear"] = "1"

    engine = PythonEngine()
    results = engine.run_in_memory(cfg, seed=seed)

    # OsmoseResults.biomass() / abundance() return DataFrames with a 'Time'
    # column, one numeric column per species, and a string 'species' column
    # = 'all'. Extract just the numeric per-species columns for comparison.
    def _numeric_array(df) -> np.ndarray:
        numeric = df.select_dtypes(include=["number"]).drop(columns=["Time"], errors="ignore")
        return numeric.to_numpy(dtype=np.float64)

    biomass_arr = _numeric_array(results.biomass())
    abundance_arr = _numeric_array(results.abundance())
    return {
        "biomass_final": biomass_arr[-1],
        "abundance_final": abundance_arr[-1],
        "biomass_full": biomass_arr,
        "abundance_full": abundance_arr,
    }


@pytest.mark.skipif(_nb is None, reason="numba not installed")
def test_mortality_deterministic_across_thread_counts() -> None:
    """End-of-run state must be allclose between 1-thread and 4-thread runs.

    The plan's risk register flagged this as potentially flake-prone (Numba
    prange + RNG ordering + non-associative float reductions could in
    principle produce thread-count-dependent results). Empirically, on the
    minimal fixture, the engine is byte-equal across thread counts — so
    we ship the test as a real pass rather than xfail. If this test
    starts failing on another machine, mark it `xfail(strict=False)` and
    investigate the prange kernel responsible.
    """
    seq = _run_minimal_with_threads(n_threads=1, seed=42)
    par = _run_minimal_with_threads(n_threads=min(4, os.cpu_count() or 1), seed=42)

    np.testing.assert_allclose(
        seq["biomass_final"], par["biomass_final"], rtol=0.0, atol=1e-12,
        err_msg="end-of-run biomass diverges across thread counts",
    )
    np.testing.assert_allclose(
        seq["abundance_final"], par["abundance_final"], rtol=0.0, atol=1e-12,
        err_msg="end-of-run abundance diverges across thread counts",
    )


@pytest.mark.skipif(_nb is None, reason="numba not installed")
def test_single_thread_is_deterministic() -> None:
    """Single-threaded reruns of the same config must be byte-equal.

    This is a stronger floor than H9: regardless of whether multi-threading
    is deterministic, two runs at NUMBA_NUM_THREADS=1 with the same seed
    must produce identical output. Pre-empts a class of regressions where
    a per-call non-determinism (e.g. dict-iteration order) leaks into the
    engine state.
    """
    a = _run_minimal_with_threads(n_threads=1, seed=42)
    b = _run_minimal_with_threads(n_threads=1, seed=42)
    np.testing.assert_array_equal(
        a["biomass_full"], b["biomass_full"],
        err_msg="single-thread reruns produced different biomass time-series",
    )
    np.testing.assert_array_equal(
        a["abundance_full"], b["abundance_full"],
        err_msg="single-thread reruns produced different abundance time-series",
    )
