"""Process backend parity + robustness for NSGA-II calibration."""

from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest

from osmose.calibration.objectives import BiomassRMSEObjective, _biomass_long
from osmose.calibration.problem import (
    FreeParameter,
    OsmoseCalibrationProblem,
    _resolve_worker_count,
)
from osmose.schema import build_registry

_MINIMAL = Path(__file__).resolve().parent.parent / "data" / "minimal" / "osm_all-parameters.csv"
_skip_no_minimal = pytest.mark.skipif(not _MINIMAL.exists(), reason="minimal config absent")


def _baseline_obs(work_dir):
    """Run the engine once on the unmodified minimal config → use its biomass as 'observed'
    so the objective is finite (no >50% inf abort)."""
    p = OsmoseCalibrationProblem(
        free_params=[FreeParameter("predation.efficiency.critical.sp0", 0.3, 0.5)],
        objective_fns=[lambda r: 0.0],
        base_config_path=_MINIMAL,
        work_dir=work_dir,
    )
    res = p._run_python_engine({}, run_id=0)
    assert res is not None
    with res as r:
        return _biomass_long(r.biomass())  # long form for use as 'observed'


def _make_problem(work_dir, obs, backend, registry=None):
    return OsmoseCalibrationProblem(
        free_params=[FreeParameter("predation.efficiency.critical.sp0", 0.3, 0.5)],
        objective_fns=[BiomassRMSEObjective(obs)],
        base_config_path=_MINIMAL,
        work_dir=work_dir,
        n_parallel=2,
        parallel_backend=backend,
        registry=registry,
    )


@_skip_no_minimal
def test_biomass_objective_handles_wide_engine_output(tmp_path):
    """BiomassRMSEObjective must consume the engine's WIDE biomass() output (it reshapes to long)
    and return a finite value — guards the pre-existing wide/long mismatch the functor now fixes."""
    obs = _baseline_obs(tmp_path / "base")  # long-form baseline
    p = _make_problem(tmp_path / "p", obs, "thread")
    res = p._run_python_engine({}, run_id=0)
    with res as r:
        val = BiomassRMSEObjective(obs)(r)  # must NOT KeyError
    assert np.isfinite(val)


@_skip_no_minimal
@pytest.mark.parametrize("with_registry", [False, True])
def test_thread_process_parity(tmp_path, with_registry):
    obs = _baseline_obs(tmp_path / "base")
    reg = build_registry() if with_registry else None  # exercises the use_registry mapping
    X = np.array([[0.35], [0.45]])
    pt = _make_problem(tmp_path / "t", obs, "thread", registry=reg)
    pp = _make_problem(tmp_path / "p", obs, "process", registry=reg)
    out_t, out_p = {}, {}
    try:
        pt._evaluate(X, out_t)
        pp._evaluate(X, out_p)
    finally:
        pp.shutdown_pool()
    np.testing.assert_allclose(out_t["F"], out_p["F"])


def test_resolve_worker_count(monkeypatch):
    monkeypatch.delenv("OSMOSE_NSGA2_WORKERS", raising=False)
    assert _resolve_worker_count(4) == 4
    monkeypatch.setenv("OSMOSE_NSGA2_WORKERS", "8")
    assert _resolve_worker_count(4) == 8
    monkeypatch.setenv("OSMOSE_NSGA2_WORKERS", "999")
    assert _resolve_worker_count(4) == 32
    monkeypatch.setenv("OSMOSE_NSGA2_WORKERS", "0")
    assert _resolve_worker_count(4) == 1


def test_shutdown_pool_idempotent(tmp_path):
    p = _make_problem(tmp_path, None, "thread")
    p.shutdown_pool()  # no pool created yet → no-op
    p.shutdown_pool()


@_skip_no_minimal
def test_broken_pool_submit_path_recovers(tmp_path, monkeypatch):
    """submit() raising BrokenProcessPool (pool broke idle between gens) must not propagate;
    pending candidates are retried on a rebuilt pool and end finite."""
    obs = _baseline_obs(tmp_path / "base")
    p = _make_problem(tmp_path / "p", obs, "process")
    real_ensure = p._ensure_pool
    calls = {"n": 0}

    class _BrokenOnce:
        def submit(self, *a, **k):
            raise BrokenProcessPool("boom")

    def _ensure():
        calls["n"] += 1
        return _BrokenOnce() if calls["n"] == 1 else real_ensure()

    monkeypatch.setattr(p, "_ensure_pool", _ensure)
    out = {}
    try:
        p._evaluate(np.array([[0.35], [0.45]]), out)  # 2 candidates
    finally:
        p.shutdown_pool()
    assert np.isfinite(out["F"]).all() and calls["n"] >= 2


def test_ensure_pool_discards_broken(tmp_path):
    """_ensure_pool must rebuild when the persisted pool is _broken (broke idle between gens)."""
    p = _make_problem(tmp_path, None, "process")

    class _Stub:
        _broken = True

        def shutdown(self, **k):  # noqa: D401
            pass

    p._executor = _Stub()
    new = p._ensure_pool()  # must discard the broken stub and build a real ProcessPoolExecutor
    try:
        assert new is not p._executor or not getattr(new, "_broken", False)
        assert not isinstance(new, _Stub)
    finally:
        p.shutdown_pool()


@_skip_no_minimal
def test_broken_pool_result_path_preserves_finished(tmp_path, monkeypatch):
    """BrokenProcessPool from fut.result() (a worker died mid-eval) must not discard an
    already-finished candidate; the survivor's value is preserved and the broken one is retried.

    Deterministic, no real worker death: a stub pool returns a REAL Future pre-loaded with a
    result for candidate 0 and with a BrokenProcessPool exception for candidate 1 (as_completed
    accepts real Futures). On retry, the real pool scores candidate 1 finite.
    """
    obs = _baseline_obs(tmp_path / "base")
    p = _make_problem(tmp_path / "p", obs, "process")
    real_ensure = p._ensure_pool
    calls = {"n": 0}

    class _MixedPool:
        def submit(self, fn, i, params):
            fut = Future()
            if i == 0:
                fut.set_result([1.23])  # candidate 0 "finishes"
            else:
                fut.set_exception(BrokenProcessPool("died"))  # candidate 1's worker died
            return fut

    def _ensure():
        calls["n"] += 1
        return _MixedPool() if calls["n"] == 1 else real_ensure()

    monkeypatch.setattr(p, "_ensure_pool", _ensure)
    out = {}
    try:
        p._evaluate(np.array([[0.35], [0.45]]), out)
    finally:
        p.shutdown_pool()
    assert out["F"][0, 0] == 1.23  # candidate 0 preserved, NOT clobbered to inf
    assert np.isfinite(out["F"][1]).all()  # candidate 1 retried on the rebuilt pool → finite
    assert calls["n"] >= 2
