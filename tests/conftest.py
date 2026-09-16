"""Shared pytest fixtures for the OSMOSE test suite.

Fixtures here are available to all test modules without explicit imports.
Only fixtures that are (or will be) used across multiple test files are placed
here; file-specific fixtures stay in their own test modules.
"""

import os
import tempfile
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
import pytest

from osmose.plotly_theme import ensure_templates
from osmose.schema import build_registry
from tests._xdist_support import worker_numba_cache_dir, worker_numba_thread_cap

# Give each xdist worker its own numba cache dir BEFORE any engine kernel
# compiles. The engine's @njit(cache=True) kernels otherwise share one
# __pycache__, and parallel workers race to write the same .nbi/.nbc files on a
# cold cache. Must run at conftest import time (before any test imports the
# engine). No-op on serial runs (PYTEST_XDIST_WORKER unset).
_worker_cache = worker_numba_cache_dir(
    os.environ.get("PYTEST_XDIST_WORKER"), Path(tempfile.gettempdir())
)
if _worker_cache is not None:
    _worker_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_worker_cache)

# Cap each xdist worker's numba thread pool, for the same reason and at the same
# moment: numba reads NUMBA_NUM_THREADS once, at ITS import, which has not happened
# yet here (conftest's own imports above do not pull numba in -- verified). Without
# this, every worker sizes its pool from the LOGICAL cpu count while `-n auto`
# spawns one worker per PHYSICAL cpu, so the machine is ~14x oversubscribed and
# wall-clock budgets in subprocesses blow (PR #148). No-op on serial runs.
#
# `setdefault`, so an explicit NUMBA_NUM_THREADS from the developer still wins.
# `os.sched_getaffinity` where available, because numba derives its own maximum the
# same way and the value must never exceed it.
_worker_threads = worker_numba_thread_cap(
    os.environ.get("PYTEST_XDIST_WORKER"),
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1),
)
if _worker_threads is not None:
    os.environ.setdefault("NUMBA_NUM_THREADS", _worker_threads)

# Register the "osmose" Plotly template once before any test runs. Without this,
# tests that render charts (tests/test_ui_results.py, test_ui_charts.py, etc.)
# fail in isolation because the template is normally registered only when app.py
# calls ensure_templates() at server startup. Running conftest.py triggers the
# registration during pytest collection, before any test module is imported.
ensure_templates()

# Skip e2e modules at collection time when playwright is unavailable. The e2e
# files import `playwright.sync_api` at module top, which raises ImportError
# during pytest collection — before `addopts = "-m 'not e2e'"` can filter them
# out. CI doesn't install playwright (it's not in [dev]), so this guard keeps
# CI collection clean while still letting local devs run e2e when playwright
# is installed.
# Skip browser test modules at collection time when their imports are unavailable.
# Both `test_e2e_*.py` and `test_visual_regression.py` import `playwright.sync_api` at
# module top, which raises ImportError during collection before `-m` filtering can
# exclude them. `test_visual_compare.py` is intentionally NOT guarded here: it imports
# only numpy/Pillow (Pillow is in [dev]) and runs in the normal suite.
collect_ignore_glob: list[str] = []
if find_spec("playwright") is None:
    collect_ignore_glob.append("test_e2e_*.py")
if find_spec("playwright") is None or find_spec("PIL") is None:
    collect_ignore_glob.append("test_visual_regression.py")

# Register a deterministic Hypothesis profile for the property-based tests
# (tests/test_*_properties.py). Guarded by find_spec so the whole suite still
# COLLECTS when hypothesis is absent — a bare top-level `import hypothesis`
# would fail collection of every test. database=None + derandomize=True keep CI
# and local runs byte-identical; deadline=None avoids flaky timing failures.
if find_spec("hypothesis") is not None:
    from hypothesis import settings

    settings.register_profile(
        "ci", max_examples=150, deadline=None, derandomize=True, database=None
    )
    settings.load_profile("ci")


# ---------------------------------------------------------------------------
# Schema / registry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def full_registry():
    """Build the complete OSMOSE schema registry once per test session.

    Use this fixture when you need the full 150+ parameter registry.
    (Unlike the ``registry`` fixture in test_registry.py, which is a
    minimal hand-built registry used to test the ParameterRegistry API
    itself.)
    """
    return build_registry()


# ---------------------------------------------------------------------------
# Minimal valid config dict
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Minimal valid OSMOSE config dict with 3 species."""
    return {
        "simulation.nspecies": "3",
        "simulation.time.nyear": "10",
        "simulation.time.ndtperyear": "24",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
        "species.name.sp2": "Hake",
        "species.linf.sp0": "19.5",
        "species.linf.sp1": "23.0",
        "species.linf.sp2": "130.0",
    }


# ---------------------------------------------------------------------------
# DataFrame fixtures for analysis / plotting tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_biomass_df():
    """Long-format biomass DataFrame with 2 species over 3 timesteps."""
    return pd.DataFrame(
        {
            "time": [1, 1, 2, 2, 3, 3],
            "species": ["A", "B", "A", "B", "A", "B"],
            "biomass": [100.0, 200.0, 110.0, 190.0, 120.0, 180.0],
        }
    )


@pytest.fixture
def sample_yield_df():
    """Long-format yield DataFrame with 2 species over 2 timesteps."""
    return pd.DataFrame(
        {
            "time": [1, 1, 2, 2],
            "species": ["A", "B", "A", "B"],
            "yield": [50.0, 100.0, 55.0, 95.0],
        }
    )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Pre-created temporary output directory for test runs."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Calibration dashboard fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_results_dir(tmp_path, monkeypatch) -> Path:
    """Redirect all checkpoint writes to tmp_path."""
    monkeypatch.setattr("osmose.calibration.checkpoint.RESULTS_DIR", tmp_path)
    try:
        import scripts.calibrate_baltic as cb_mod

        monkeypatch.setattr(cb_mod, "RESULTS_DIR", tmp_path, raising=False)
    except ImportError:
        pass
    try:
        import ui.pages.calibration_handlers as ch_mod

        monkeypatch.setattr(ch_mod, "RESULTS_DIR", tmp_path, raising=False)
    except ImportError:
        pass
    return tmp_path


@pytest.fixture
def synthetic_two_species_targets():
    """A 2-species banded-loss target list."""
    from scripts.calibrate_baltic import BiomassTarget

    targets = [
        BiomassTarget(species="sp_a", target=1.0, lower=0.5, upper=1.5, weight=1.0),
        BiomassTarget(species="sp_b", target=2.0, lower=1.5, upper=2.5, weight=1.0),
    ]
    species_names = ["sp_a", "sp_b"]
    return targets, species_names


@pytest.fixture
def synthetic_stats_in_band():
    return {
        "sp_a_mean": 1.0,
        "sp_a_cv": 0.1,
        "sp_a_trend": 0.01,
        "sp_b_mean": 2.0,
        "sp_b_cv": 0.1,
        "sp_b_trend": 0.01,
    }


@pytest.fixture
def synthetic_stats_sp_b_out_of_band():
    return {
        "sp_a_mean": 1.0,
        "sp_a_cv": 0.1,
        "sp_a_trend": 0.01,
        "sp_b_mean": 5.0,
        "sp_b_cv": 0.1,
        "sp_b_trend": 0.01,
    }


# ---------------------------------------------------------------------------
# Numba warmup — session-scoped, opt-in
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def numba_warmup() -> None:  # type: ignore[return]
    """Warm Numba's JIT cache once per pytest session.

    The OSMOSE engine compiles ~20-25 s of native code on first run; subsequent
    runs are <2 s. Tests that run the engine should request this fixture so
    the JIT cost is paid once per session, not once per test.

    OPT-IN (not autouse) — tests that don't run the engine (schema-only,
    MCP-credential, etc.) shouldn't pay this cost. The tutorial test requests
    it via its `baseline_run` and `perturbed_run` fixtures.

    Runs a minimal 1-year Baltic simulation. Output is discarded.
    """
    import tempfile
    from pathlib import Path

    from osmose.engine import PythonEngine

    # Local import to avoid circular dependency at conftest load time
    from tests._tutorial_config import build_config

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        cfg = build_config(work, n_year=1)
        PythonEngine().run_in_memory(config=cfg, seed=0)
    # No yield — one-shot setup with no teardown.


# ---------------------------------------------------------------------------
# Numba thread-count hygiene
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_numba_thread_state():
    """No test may leave Numba's thread count different from how it found it.

    Stated as an invariant rather than as a patch at each call site, because the
    call sites keep growing and one of them leaks in a shape that is easy to miss:
    ``tests/test_sp1b_recalibration.py`` calls ``set_num_threads(1)`` as its FIRST
    statement and only then reaches ``pytest.skip()``, so it poisons its worker
    while REPORTING AS SKIPPED. ``tests/test_jit_determinism.py`` leaks too -- its
    last test ends at 1 thread.

    Why it matters under ``pytest -n auto``: ``set_num_threads`` is thread-local,
    and pytest runs every test on the worker's main thread, so "thread-local" means
    "persists for that worker process's whole life". With ``--dist loadfile`` the
    files that land on a poisoned worker afterwards run the engine single-threaded.
    Measured 2026-09-15: a probe file run after ``test_jit_determinism.py`` in one
    process sees ``get_num_threads() == 1`` and ``NUMBA_NUM_THREADS == "1"``.

    This is a PERFORMANCE and reproducibility-of-timing issue, NOT a correctness
    one: engine output is bit-identical across thread counts, asserted on two
    different fixtures by ``test_thread_policy.py::test_mortality_bit_identical_
    across_thread_counts`` (EEC, ``np.array_equal``) and
    ``test_jit_determinism.py::test_mortality_deterministic_across_thread_counts``
    (minimal, atol=1e-12). So this fixture does not change any test's numbers.

    Restores the env var as well as the runtime count: the runtime call is
    thread-local, but ``NUMBA_NUM_THREADS`` is process-global and is inherited by
    subprocesses -- which is exactly the leak
    ``test_thread_policy.py::test_cap_does_not_leak_into_forkserver_worker`` had to
    defuse by hand.

    Guarded on ``sys.modules`` so the thousands of non-engine tests never import
    numba just to be cleaned up after.
    """
    import sys

    def _snapshot() -> int | None:
        numba = sys.modules.get("numba")
        try:
            return numba.get_num_threads() if numba is not None else None
        except Exception:  # noqa: BLE001 — numba present but threading layer unusable
            return None

    saved_threads = _snapshot()
    saved_env = os.environ.get("NUMBA_NUM_THREADS")

    yield

    if os.environ.get("NUMBA_NUM_THREADS") != saved_env:
        if saved_env is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = saved_env

    current = _snapshot()
    if current is not None and current != saved_threads:
        # saved_threads is None when numba was imported DURING the test; fall back
        # to numba's own configured maximum, which is the count a fresh worker has.
        numba = sys.modules["numba"]
        target = saved_threads if saved_threads is not None else numba.config.NUMBA_NUM_THREADS
        try:
            numba.set_num_threads(target)
        except Exception:  # noqa: BLE001 — never let cleanup mask a real failure
            pass
