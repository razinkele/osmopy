"""Unit tests for the single-run Numba thread policy resolver."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from osmose.engine import thread_policy as tp


@pytest.fixture
def restore_numba_threads():
    """Save/restore Numba's active thread count so tests don't leak into each other."""
    numba = pytest.importorskip("numba")
    saved = numba.get_num_threads()
    yield
    numba.set_num_threads(saved)


def _fake_topology(monkeypatch, affinity, core_of):
    """Make logical/physical budgets deterministic: `affinity` = set of logical cpu ids,
    `core_of` = dict cpu_id -> (pkg, core) or None (unreadable)."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(affinity))
    monkeypatch.setattr(tp, "_core_key", lambda cpu: core_of.get(cpu))


def test_ht_box_auto_is_physical(monkeypatch):
    # 28 logical, cpu N and N+14 share a physical core -> 14 physical.
    aff = set(range(28))
    core_of = {c: ("0", str(c % 14)) for c in aff}
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.logical_budget() == 28
    assert tp.physical_budget() == 14
    assert tp.resolve_engine_threads(None) == 14
    assert tp.resolve_engine_threads(0) == 14


def test_no_ht_box_auto_is_all_cores(monkeypatch):
    aff = set(range(8))
    core_of = {c: ("0", str(c)) for c in aff}  # 8 distinct cores
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.physical_budget() == 8
    assert tp.resolve_engine_threads(None) == 8  # no-op: physical == logical


def test_container_quota_respected(monkeypatch):
    aff = {0, 1, 2, 3}
    core_of = {0: ("0", "0"), 1: ("0", "1"), 2: ("0", "2"), 3: ("0", "3")}
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.logical_budget() == 4
    assert tp.physical_budget() == 4  # min(4 physical, 4 budget)


def test_sys_read_failure_falls_back_to_logical(monkeypatch):
    aff = set(range(28))
    _fake_topology(monkeypatch, aff, {c: None for c in aff})  # every /sys read fails
    assert tp.physical_budget() == 28  # fallback = today's behavior, NOT 1


def test_explicit_request_honored_and_capped(monkeypatch):
    aff = set(range(28))
    core_of = {c: ("0", str(c % 14)) for c in aff}
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.resolve_engine_threads(20) == 20  # honored
    assert tp.resolve_engine_threads(100) == 28  # capped to logical budget
    assert tp.resolve_engine_threads(1) == 1


def test_degenerate_topology_falls_back(monkeypatch):
    # /sys reads succeed but claim all 28 cpus are one physical core (SMT width 28 > 4).
    aff = set(range(28))
    core_of = {c: ("0", "0") for c in aff}
    _fake_topology(monkeypatch, aff, core_of)
    assert tp.physical_budget() == 28  # distrusted -> logical budget, NOT 1


def test_per_cpu_tolerance_skips_unreadable(monkeypatch):
    # One cpu's /sys entry is unreadable; the rest map to 14 physical cores.
    aff = set(range(28))
    core_of = {c: ("0", str(c % 14)) for c in aff}
    core_of[7] = None
    _fake_topology(monkeypatch, aff, core_of)
    # cpu 7 skipped; remaining 27 cpus still cover all 14 core ids -> 14 physical.
    assert tp.physical_budget() == 14


def test_apply_returns_zero_when_numba_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)  # `import numba` -> ImportError
    assert tp.apply_single_run_threads(4) == 0


def test_apply_sets_and_returns_resolved(monkeypatch, restore_numba_threads):
    numba = pytest.importorskip("numba")
    monkeypatch.setattr(tp, "resolve_engine_threads", lambda requested: 3)
    n = tp.apply_single_run_threads(999)
    assert n == 3
    assert numba.get_num_threads() == 3


def test_python_engine_thread_applies_policy(monkeypatch):
    """_python_engine_thread must pass the raw py_threads value to the policy
    and post a 'done' outcome — without running a real engine."""
    import queue

    from ui.pages import run as run_mod

    recorded = {}

    def _fake_apply(requested):
        recorded["requested"] = requested
        return 14

    monkeypatch.setattr(run_mod, "apply_single_run_threads", _fake_apply)

    class _FakeEngine:
        def run(self, *a, **k):
            return "RESULT"

    monkeypatch.setattr(run_mod, "PythonEngine", _FakeEngine)
    done_q: queue.Queue = queue.Queue()
    run_mod._python_engine_thread("cfg", "out", None, None, done_q, n_threads=7)

    assert recorded["requested"] == 7
    kind, result, msg = done_q.get_nowait()
    assert kind == "done"
    assert result == "RESULT"


def test_benchmark_main_applies_policy(monkeypatch):
    """benchmark_engine.main() must call apply_single_run_threads without running
    the engine (run_benchmark is stubbed)."""
    import scripts.benchmark_engine as be

    called = {}

    def _fake_apply(requested=None):
        called["hit"] = True
        return 12

    monkeypatch.setattr(be, "apply_single_run_threads", _fake_apply)
    monkeypatch.setattr(
        be,
        "run_benchmark",
        lambda *a, **k: {"elapsed_s": 0.1, "final_biomass": {"sp0": 1.0}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_engine.py", "--config", "minimal", "--years", "1", "--repeats", "1"],
    )
    be.main()
    assert called.get("hit") is True


def _run_eec_for_determinism(n_years: int, seed: int, threads: int):
    """Run the eec_full fixture at a fixed Numba thread count; return (grid, outputs).

    CRITICAL: use eec_full (460 ocean cells), NOT data/minimal — minimal's grid mask
    has ZERO ocean cells, so every school stays at cell (-1,-1), mortality()'s
    valid_indices is empty every timestep, n_cells==0, and the prange kernel under
    test is NEVER invoked. A determinism check on minimal would pass vacuously
    (kernel skipped), proving nothing. eec_full's 460 cells make the parallel
    cell-loop actually run.
    """
    import numba

    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "eec_full", "eec_all-parameters.csv"
    )
    reader = OsmoseConfigReader()
    raw = reader.read(cfg_path)
    raw["simulation.time.nyear"] = str(n_years)
    cfg = EngineConfig.from_dict(raw)
    grid = Grid.from_netcdf(
        os.path.join(os.path.dirname(cfg_path), raw["grid.netcdf.file"]),
        mask_var=raw.get("grid.var.mask", "mask"),
    )
    numba.set_num_threads(threads)
    return grid, simulate(cfg, grid, np.random.default_rng(seed))


@pytest.mark.filterwarnings("ignore:Swapping size ratios")
@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >=2 cores to compare thread counts")
def test_mortality_bit_identical_across_thread_counts(restore_numba_threads):
    """The mortality prange is race-free: 1 thread vs many threads must be EXACTLY
    equal (np.array_equal, not allclose). Two hardcoded, different counts so this
    cannot degenerate into comparing a run against itself. Uses eec_full so the
    kernel actually runs; the ocean-cell assertion guarantees this can never pass
    vacuously (see `_run_eec_for_determinism`)."""
    pytest.importorskip("numba")
    hi = os.cpu_count()
    grid1, out1 = _run_eec_for_determinism(1, 123, 1)
    assert grid1.ocean_mask.sum() > 1, (
        "fixture has no ocean cells — the mortality kernel would be skipped and this "
        "determinism test would pass vacuously"
    )
    _, outN = _run_eec_for_determinism(1, 123, hi)
    b1 = np.asarray(out1[-1].biomass, dtype=np.float64)
    bN = np.asarray(outN[-1].biomass, dtype=np.float64)
    assert np.array_equal(b1, bN), f"biomass differs between 1 and {hi} threads"


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >=2 cores to observe a cap")
def test_cap_is_thread_local_same_process(restore_numba_threads):
    """Same-process (thread-backed) calibration paths are unaffected: a main-thread
    cap is invisible to a sibling thread (numba.set_num_threads is thread-local)."""
    import threading

    numba = pytest.importorskip("numba")
    default = numba.config.NUMBA_NUM_THREADS
    if default < 2:
        pytest.skip("default thread count is 1; cannot distinguish a cap")
    tp.apply_single_run_threads(1)
    assert numba.get_num_threads() == 1
    seen = []
    t = threading.Thread(target=lambda: seen.append(numba.get_num_threads()))
    t.start()
    t.join()
    assert seen[0] == default, "single-run cap leaked into a sibling thread"


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >=2 cores to observe a cap")
def test_cap_does_not_leak_into_forkserver_worker(monkeypatch, restore_numba_threads):
    """Regression guard: the single-run cap must stay a *thread-local runtime*
    call (`numba.set_num_threads`), so calibration's forkserver workers keep the
    unrestricted default. Note the cap cannot leak into a forkserver worker via
    the runtime path anyway — workers fork from the persistent server process, not
    the calling thread, and set_num_threads is thread-local (see the sibling
    thread-local test). So a RED here most likely means the cap was reintroduced
    as an inherited env var / process-global (the exact bug this file's
    NUMBA_NUM_THREADS cleanup defuses), NOT a runtime-cap leak — debug there, not
    in apply_single_run_threads. Cap to 1, then a forkserver worker must still see
    the default."""
    import multiprocessing
    import multiprocessing.forkserver
    from concurrent.futures import ProcessPoolExecutor

    numba = pytest.importorskip("numba")
    default = numba.config.NUMBA_NUM_THREADS
    if default < 2:
        pytest.skip("default thread count is 1; cannot distinguish a cap")

    from tests.helpers import numba_thread_count

    # Defuse a cross-file hazard: tests/test_jit_determinism.py sets
    # os.environ["NUMBA_NUM_THREADS"] and never restores it. multiprocessing's
    # forkserver process is a singleton that persists (and is reused) for the
    # rest of THIS interpreter's life once started, so if any earlier test in
    # the same process (e.g. a real ProcessPoolExecutor(forkserver) pool, as
    # osmose/calibration's "process" backend uses) started it while that env
    # var was polluted, every worker forked from it inherits the stale value
    # forever after — clearing the env here would be too late for an
    # already-running server. `_stop()` is multiprocessing's own sanctioned
    # test hook ("Method used by unit tests to stop the server") to force a
    # fresh respawn once the env is clean, regardless of what ran before.
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    multiprocessing.forkserver._forkserver._stop()
    tp.apply_single_run_threads(1)
    assert numba.get_num_threads() == 1
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        worker_threads = ex.submit(numba_thread_count, None).result(timeout=120)
    assert worker_threads == default, "single-run cap leaked into a forkserver worker"


def test_worker_init_does_not_reset_numba_threads():
    """Calibration's real worker initializer (osmose/calibration/problem.py::_worker_init)
    must NOT set a Numba thread count — the single-run cap must never be reintroduced
    there. Static guard so this invariant survives future edits to _worker_init.

    (The thread-ISOLATION guarantee itself comes from the forkserver boundary + Numba's
    thread-locality, exercised by the two tests above. Reconstructing a full
    OsmoseCalibrationProblem via _worker_init here — it takes an _EvalSpec and rebuilds
    the whole problem — would add heavy, fragile calibration coupling for no extra thread
    coverage, since _worker_init is thread-agnostic; this static guard is the proportionate
    check that it cannot reintroduce a cap.)"""
    import inspect

    from osmose.calibration import problem

    src = inspect.getsource(problem._worker_init)
    assert "set_num_threads" not in src, "_worker_init must not touch the Numba thread count"
