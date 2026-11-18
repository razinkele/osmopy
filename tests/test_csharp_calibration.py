"""Plan task 1.12, acceptance: calibration with ``engine="csharp"``.

- One generation runs on the process backend (a forkserver pool), one thread per C# run, and
  the CLR loads in the workers only, never in this pytest process.
- A cached Python result is not served to a C# request: the C# generation, given the same
  candidates and cache directory as a Python one, runs every candidate and writes its own
  entries.

The engine-name plumbing and the key difference are unit-tested in tests/test_engine_select.py;
this is the behaviour end to end. It needs the built engine: it skips without it, unless
``OSMOSE_REQUIRE_CSHARP=1`` (the csharp-engine CI job), where it fails.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.calibration.objectives import BiomassRMSEObjective, _biomass_long
from osmose.calibration.problem import (
    CSHARP_THREADS_PER_EVAL,
    FreeParameter,
    OsmoseCalibrationProblem,
)
from osmose.demo import osmose_demo
from osmose.engine_cs import csharp_engine_available

PARAMS = [FreeParameter("species.k.sp0", 0.2, 0.6)]
X = np.array([[0.25], [0.35], [0.45], [0.55]])  # one generation of four candidates


def _need_engine() -> None:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")


def _base_config(tmp_path: Path) -> Path:
    """Bay of Biscay, one year, with the minimal outputs the C# engine requires."""
    cfg = Path(osmose_demo("bay_of_biscay", tmp_path / "demo")["config_file"])
    with cfg.open("a") as f:
        f.write("\nsimulation.time.nyear;1\noutput.minimal.enabled;true\n")
    return cfg


def _problem(cfg, work, obs, cache, **kw) -> OsmoseCalibrationProblem:
    return OsmoseCalibrationProblem(
        free_params=PARAMS, objective_fns=[BiomassRMSEObjective(obs)], base_config_path=cfg,
        work_dir=work, enable_cache=True, cache_dir=cache, **kw,
    )  # fmt: skip


def test_a_csharp_generation_runs_on_forkserver_workers_and_never_reuses_python_results(
    tmp_path, monkeypatch
):
    _need_engine()
    monkeypatch.setenv("OSMOSE_NSGA2_WORKERS", "2")  # an inherited value would resize the pool
    from osmose.engine_cs import bridge

    cfg, cache = _base_config(tmp_path), tmp_path / "cache"
    # "Observed" = the Python engine's own unmodified run, so every objective is finite.
    base = _problem(cfg, tmp_path / "base", None, None)
    baseline = base._run_python_engine({}, run_id=0)
    assert baseline is not None, "the Python baseline run failed (see the log)"
    with baseline as r:
        obs = _biomass_long(r.biomass())

    py = _problem(cfg, tmp_path / "py", obs, cache)  # in this process: Python is safe here
    out_py: dict = {}
    py._evaluate(X, out_py)
    py_entries = set(cache.iterdir())
    assert len(py_entries) == len(X)

    cs = _problem(cfg, tmp_path / "cs", obs, cache, engine="csharp", n_parallel=2,
                  parallel_backend="process")  # fmt: skip
    out_cs: dict = {}
    try:
        cs._evaluate(X, out_cs)
        assert cs._executor is not None
        assert cs._executor._mp_context.get_start_method() == "forkserver"
        assert cs._executor._max_workers == 2
    finally:
        cs.shutdown_pool()

    # The workers ran the C# engine; this process never hosted the CLR.
    assert "clr" not in sys.modules and bridge._engine is None
    assert np.all(np.isfinite(out_cs["F"]))
    # Not served from the Python cache: each candidate's C# entry was absent after the Python
    # generation and is present after the C# one (the candidates' overrides, as the problem
    # builds them), and nothing else was added.
    cs_entries = {cache / f"{cs._cache_key({PARAMS[0].key: str(x[0])})}.json" for x in X}
    assert not cs_entries & py_entries
    assert set(cache.iterdir()) - py_entries == cs_entries


def test_each_csharp_evaluation_asks_for_one_thread(tmp_path, monkeypatch):
    """The threads a C# run gets in calibration, without loading the CLR: a stand-in engine."""
    from osmose.engine import select

    seen: list[dict] = []

    class Stub:
        def run_in_memory(self, cfg, seed=0):
            return None

    def fake_get_engine(name, **kwargs):
        seen.append({"name": name, **kwargs})
        return Stub()

    monkeypatch.setattr(select, "get_engine", fake_get_engine)
    cfg = tmp_path / "cfg.csv"
    cfg.write_text("simulation.time.nyear;1\n")
    py = _problem(cfg, tmp_path / "w", None, None)
    cs = object.__new__(OsmoseCalibrationProblem)  # bypass the up-front run check
    cs.__dict__.update(py.__dict__, engine="csharp")
    cs._run_python_engine({}, run_id=0)
    py._run_python_engine({}, run_id=0)
    assert CSHARP_THREADS_PER_EVAL == 1
    assert seen == [{"name": "csharp", "threads": 1}, {"name": "python"}]
