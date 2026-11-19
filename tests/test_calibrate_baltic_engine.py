"""``scripts/calibrate_baltic.py --engine csharp`` (the owner's G2 decision, 2026-11-19:
calibration stays opt-in, and the Baltic script can use the C# engine when asked).

What must hold, and is checked here without the built engine (the engine itself is stubbed; the
real one runs in ``tests/test_csharp_calibrate_baltic.py``, in a child process):

- the engine reaches EVERY evaluation: the DE workers', the checkpoint's, the re-rank's, the
  validation's. One call site left on the default would score a C# calibration on the Python
  engine and say nothing;
- a process that hosts the .NET runtime never forks afterwards;
- a calibration the C# engine cannot run is refused before anything runs, on the config that
  runs, and one whose first evaluation fails is stopped instead of optimised as penalties;
- a C# run never replaces a Python run's files;
- the Python path is what it was.
"""

from __future__ import annotations

import ast
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from scripts import calibrate_baltic as cb

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calibrate_baltic.py"


# --------------------------------------------------------------------------------------------
# The engine reaches every evaluation
# --------------------------------------------------------------------------------------------
def _calls(name: str) -> list[ast.Call]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


@pytest.mark.parametrize(
    ("callee", "n_sites"),
    [
        ("run_simulation", 3),  # the objective's, the multi-seed validation's, --validate's
        ("make_objective", 2),  # the optimizer's objective, the re-rank's
        ("_ObjectiveWrapper", 1),
        ("isolated_eval_map", 1),
        ("_dispatch_optimizer", 1),
        ("validate_calibration", 1),
        ("run_calibration", 1),
    ],
)
def test_every_call_site_names_the_engine(callee, n_sites):
    """Static, so a call site added later cannot slip through on the default. The count pins
    that the scan sees the sites it is meant to."""
    sites = _calls(callee)
    assert len(sites) == n_sites, [s.lineno for s in sites]
    for site in sites:
        assert "engine" in {kw.arg for kw in site.keywords}, f"{callee} at line {site.lineno}"


class _Result:
    returncode = 1  # run_simulation returns {} at once, before it reads any output


def test_run_simulation_builds_the_engine_it_is_asked_for(monkeypatch):
    from osmose.engine import select
    from osmose.engine.thread_policy import physical_budget

    built: list[tuple] = []

    class _Runner:
        def run(self, cfg, output_dir, seed):
            built.append(("run", cfg["simulation.time.nyear"], cfg["output.minimal.enabled"], seed))
            return _Result()

    def fake_get_engine(name, **kwargs):
        built.append(("get_engine", name, kwargs))
        return _Runner()

    monkeypatch.setattr(select, "get_engine", fake_get_engine)
    assert (
        cb.run_simulation({"a": "1"}, {}, n_years=3, seed=5, engine="csharp", engine_threads=3)
        == {}
    )
    assert built == [("get_engine", "csharp", {"threads": 3}), ("run", "3", "true", 5)]
    built.clear()
    cb.run_simulation({"a": "1"}, {}, n_years=3, seed=5, engine="csharp")  # no count given
    assert built[0] == ("get_engine", "csharp", {"threads": physical_budget()})
    with pytest.raises(ValueError, match="engine must be one of"):
        cb.run_simulation({"a": "1"}, {}, engine="java")


def test_the_python_path_still_builds_python_engine_by_its_name(monkeypatch):
    """Tests across the suite patch ``osmose.engine.PythonEngine``; the default path must keep
    looking it up there, and must not go through the selector."""
    from osmose.engine import select

    seen: list[str] = []

    class _Fake:
        def run(self, cfg, output_dir, seed):
            seen.append("PythonEngine.run")
            return _Result()

    monkeypatch.setattr("osmose.engine.PythonEngine", _Fake, raising=False)
    monkeypatch.setattr(select, "get_engine", lambda *a, **k: seen.append("get_engine"))
    assert cb.run_simulation({"a": "1"}, {}, n_years=1, seed=0) == {}
    assert cb.run_simulation({"a": "1"}, {}, n_years=1, seed=0, engine="python") == {}
    assert seen == ["PythonEngine.run", "PythonEngine.run"]


def test_the_objective_carries_the_engine_through_a_pickle(monkeypatch):
    targets = cb.load_targets()
    seen: list[dict] = []
    monkeypatch.setattr(cb, "run_simulation", lambda *a, **kw: seen.append(kw) or {})
    for kwargs, want in (
        ({"engine": "csharp", "engine_threads": 2}, ("csharp", 2)),
        ({}, ("python", None)),
    ):
        # A round trip of an object built on the line below: nothing untrusted is loaded.
        objective = pickle.loads(
            pickle.dumps(cb.make_objective({"a": "1"}, targets, ["species.k.sp0"], **kwargs))
        )
        assert objective(np.array([0.0])) == 1e6  # the stub returns no stats
        assert (seen[-1]["engine"], seen[-1]["engine_threads"]) == want
    with pytest.raises(ValueError, match="engine must be one of"):
        cb.make_objective({"a": "1"}, targets, ["species.k.sp0"], engine="java")


def test_only_the_python_engine_gets_numba_s_worker_rule(monkeypatch):
    """The C# engine takes its threads per run, from the objective; its workers must not import
    Numba for nothing."""
    applied: list[int] = []
    monkeypatch.setattr(cb, "apply_worker_threads", lambda n: applied.append(n) or n)
    cb._init_worker(3)
    cb._init_worker(4, "python")
    cb._init_worker(5, "csharp")
    assert applied == [3, 4]


# --------------------------------------------------------------------------------------------
# A process that hosts the runtime never forks afterwards
# --------------------------------------------------------------------------------------------
def test_clr_hosted_reads_the_process_and_loads_nothing(monkeypatch):
    from osmose import engine_cs

    assert engine_cs.clr_hosted() is False  # pytest never hosts the runtime
    assert "clr" not in sys.modules, "asking must not load it"
    monkeypatch.setitem(sys.modules, "clr", types.ModuleType("clr"))
    assert engine_cs.clr_hosted() is True
    monkeypatch.delitem(sys.modules, "clr")
    bridge = types.ModuleType("osmose.engine_cs.bridge")
    bridge._engine = None  # imported, nothing loaded yet
    monkeypatch.setitem(sys.modules, "osmose.engine_cs.bridge", bridge)
    assert engine_cs.clr_hosted() is False
    bridge._engine = object()
    assert engine_cs.clr_hosted() is True


def _never(*_a, **_k):
    raise AssertionError("an evaluation ran although the fork was refused")


def test_each_fork_site_refuses_once_the_runtime_is_hosted(monkeypatch):
    """Measured 2026-11-19: a child forked from a process that has run the C# engine dies on its
    first managed call, and the isolated map would score it the penalty. So the refusal is at
    the fork, as an error."""
    from osmose import engine_cs

    x = np.array([0.0])
    cb._assert_may_fork("a clean process")  # not hosted: no error
    monkeypatch.setattr(engine_cs, "clr_hosted", lambda: True)
    with pytest.raises(RuntimeError, match="the isolated-eval map: .*must not fork"):
        cb.isolated_eval_map(5.0, 2)(_never, [x])
    with pytest.raises(RuntimeError, match="the forked re-evaluation: .*must not fork"):
        cb._ForkedReeval(_never, timeout_s=5.0)(x)
    with pytest.raises(RuntimeError, match="the DE worker pool: .*must not fork"):
        cb._dispatch_optimizer(
            "de", _never, [(0.0, 1.0)], [0.5], np.full((5, 1), 0.5),
            maxiter=1, popsize=5, tol=0.01, workers=2, seed=1,
        )  # fmt: skip


# --------------------------------------------------------------------------------------------
# Refused before anything runs, on the config that runs
# --------------------------------------------------------------------------------------------
def _stock() -> dict[str, str]:
    from osmose.config.reader import OsmoseConfigReader

    return dict(OsmoseConfigReader().read(cb.BALTIC_CONFIG)) | {"output.minimal.enabled": "true"}


def test_a_csharp_calibration_that_cannot_work_is_refused_with_its_reason(monkeypatch):
    from osmose import engine_cs

    monkeypatch.setattr(engine_cs, "csharp_engine_available", lambda: (False, "not built here"))
    with pytest.raises(SystemExit, match="not available here: not built here"):
        cb._require_csharp(_stock())
    monkeypatch.setattr(engine_cs, "csharp_engine_available", lambda: (True, ""))
    cb._require_csharp(_stock())  # the Baltic config as shipped runs on the C# engine
    with pytest.raises(
        SystemExit, match=r"cannot run this calibration's config: .*selectivity_logistic"
    ):
        cb._require_csharp(_stock() | {"fisheries.selectivity.type.fsh0": "1"})
    cb._require_csharp(_stock() | {"species.lmax.sp0": "25"})  # a growth cap runs
    for optimizer in ("cmaes", "surrogate-de"):
        with pytest.raises(SystemExit, match="--optimizer de only"):
            cb._require_csharp(_stock(), optimizer)


def test_the_check_sees_the_phase_s_edits_and_the_starting_point(monkeypatch):
    """A phase can switch a feature on, and the free parameters are config keys too. Checking
    the stock file would miss both."""
    from osmose import engine_cs

    seen: dict = {}

    def capture(effective, optimizer="de"):
        seen.update(config=dict(effective), optimizer=optimizer)
        raise SystemExit("captured")

    monkeypatch.setattr(cb, "_require_csharp", capture)
    with pytest.raises(SystemExit, match="captured"):
        cb.run_calibration(phase="1d", engine="csharp")
    config = seen["config"]
    assert config["species.accessibility2fish.sp9"] == "0.4"  # phase 1d's own edit
    assert config["output.minimal.enabled"] == "true"
    keys, _bounds, x0 = cb.get_phase1b_params()  # 1d frees the same parameters as 1b
    for key, value in cb.expand_param_overrides(keys, np.asarray(x0), True).items():
        assert config[key] == value
    assert seen["optimizer"] == "de"

    # End to end with the real check: the depletion base switches on a feature the C# engine
    # does not implement, and the refusal names it.
    monkeypatch.undo()
    monkeypatch.setattr(engine_cs, "csharp_engine_available", lambda: (True, ""))
    monkeypatch.setattr(cb, "_preflight_csharp", _never)  # a missing refusal fails here, at once
    with pytest.raises(SystemExit, match="cannot run this calibration's config: .*depletable"):
        cb.run_calibration(phase="1", a2=True, engine="csharp")
    with pytest.raises(SystemExit, match="--optimizer de only"):
        cb.run_calibration(phase="1b", optimizer="cmaes", engine="csharp")


def test_a_first_evaluation_that_fails_stops_the_run(monkeypatch):
    """A failed evaluation scores the 1e6 penalty and so does a crashed worker: without this, an
    engine that cannot run here hands the optimizer a flat landscape for hours."""
    calls: list = []

    class _Reeval:
        outcome: object = 3.25

        def __init__(self, evaluator, timeout_s):
            calls.append((evaluator, timeout_s))

        def __call__(self, x):
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

    monkeypatch.setattr(cb, "_ForkedReeval", _Reeval)
    assert cb._preflight_csharp("objective", [0.1, 0.2], timeout_s=77.0) == 3.25
    assert calls == [("objective", 77.0)]  # in a forked child, not in this process
    for outcome, message in (
        (1e6, "returned the failure penalty"),
        (float("nan"), "returned the failure penalty"),
        (RuntimeError("child exited without a result"), "failed: child exited without a result"),
        (TimeoutError("exceeded 77 s"), "failed: exceeded 77 s"),
    ):
        _Reeval.outcome = outcome
        with pytest.raises(SystemExit, match=message):
            cb._preflight_csharp("objective", [0.1], timeout_s=77.0)


# --------------------------------------------------------------------------------------------
# A whole (stubbed) run: every evaluation on the engine asked for, files that cannot collide
# --------------------------------------------------------------------------------------------
def _stub_run(monkeypatch, tmp_path, engine: str, workers: str = "1") -> tuple[list, dict]:
    from osmose.calibration import history

    calls: list[dict] = []

    def fake_run_simulation(config, overrides, **kw):
        calls.append(kw)
        level = 1000.0 + sum(float(v) for v in overrides.values())
        return {f"{sp}_mean": level for sp in cb.SPECIES_NAMES}

    monkeypatch.setattr(cb, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(cb, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "history")
    monkeypatch.setenv("OSMOSE_DE_WORKERS", workers)
    monkeypatch.setattr(cb, "_require_csharp", lambda *a, **k: None)
    preflight: list = []
    monkeypatch.setattr(
        cb, "_preflight_csharp",
        lambda objective, x0, timeout_s: preflight.append(objective.engine_threads) or 1.0,
    )  # fmt: skip
    saved = cb.run_calibration(
        phase="1b", maxiter=1, popsize=5, popsize_mult=0, n_seeds=2, n_years=3,
        checkpoint_every=1, engine=engine,
    )  # fmt: skip
    return calls, {"saved": saved, "preflight": preflight}


@pytest.mark.parametrize("engine", ["csharp", "python"])
def test_a_whole_run_uses_one_engine_and_writes_under_its_name(monkeypatch, tmp_path, engine):
    from osmose.engine.thread_policy import physical_budget, worker_thread_count

    calls, out = _stub_run(monkeypatch, tmp_path, engine)
    # DE (two generations of five), the checkpoint, five re-rank seeds, the final residuals and
    # two validation seeds: every one of them on the engine asked for.
    assert len(calls) >= 10 + 5 + 1 + 2
    assert {kw["engine"] for kw in calls} == {engine}
    tag = "_csharp" if engine == "csharp" else ""
    results = tmp_path / "results"
    assert sorted(p.name for p in results.iterdir()) == [
        f"phase1b{tag}_checkpoint.json", f"phase1b{tag}_results.json",
    ]  # fmt: skip
    assert json.loads((results / f"phase1b{tag}_results.json").read_text())["engine"] == engine
    assert out["saved"]["engine"] == engine
    (run_file,) = (tmp_path / "history").iterdir()
    assert json.loads(run_file.read_text())["engine"] == engine
    if engine == "csharp":
        # One worker: the worker rule gives the whole machine. Checked before the optimizer.
        assert out["preflight"] == [worker_thread_count(1) or physical_budget()]
        threads = [kw.get("engine_threads") for kw in calls]
        assert threads[0] == out["preflight"][0] and threads[-1] is None  # serial again after DE
    else:
        assert out["preflight"] == []  # the Python path has no preflight: it is unchanged
        assert {kw.get("engine_threads") for kw in calls} == {None}


def test_the_csharp_workers_split_the_cores_as_the_python_workers_do(monkeypatch, tmp_path):
    """With W workers each C# evaluation gets physical cores // W threads, the rule the Python
    workers' Numba threads follow, and the engine name reaches the pool."""
    from osmose.engine.thread_policy import WORKER_THREADS_ENV, physical_budget

    seen: dict = {}

    def stop(optimizer, objective, *args, **kwargs):
        seen.update(
            threads=objective.engine_threads, engine=kwargs["engine"], workers=kwargs["workers"]
        )
        raise SystemExit("dispatched")

    monkeypatch.delenv(WORKER_THREADS_ENV, raising=False)
    monkeypatch.setattr(cb, "_dispatch_optimizer", stop)
    with pytest.raises(SystemExit, match="dispatched"):
        _stub_run(monkeypatch, tmp_path, "csharp", workers="4")
    assert seen == {"threads": max(1, physical_budget() // 4), "engine": "csharp", "workers": 4}


# --------------------------------------------------------------------------------------------
# Which earlier results a run reads
# --------------------------------------------------------------------------------------------
def _results(directory: Path, name: str, value: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps({"parameters": {"species.k.sp0": value}}))


def test_an_engine_reads_its_own_earlier_results_first(monkeypatch, tmp_path):
    """Phase 2 and --validate read an earlier phase's results. A C# run reads the C# file; with
    none, it reads the Python engine's (the calibration of record) and SAYS SO, because that is
    a cross-check and not a C# chain. The Python engine reads only its own file."""
    monkeypatch.setattr(cb, "RESULTS_DIR", tmp_path)
    assert cb._earlier_phase_results("1", "python") == (None, "")
    assert cb._earlier_phase_results("1", "csharp") == (None, "")
    _results(tmp_path, "phase1_results.json", 0.25)
    assert cb._earlier_phase_results("1", "python") == (tmp_path / "phase1_results.json", "")
    assert cb._earlier_phase_results("1", "csharp") == (
        tmp_path / "phase1_results.json", cb.CROSS_ENGINE_NOTE,
    )  # fmt: skip
    _results(tmp_path, "phase1_csharp_results.json", 0.5)
    assert cb._earlier_phase_results("1", "csharp") == (tmp_path / "phase1_csharp_results.json", "")
    # A C# file never reaches a Python run.
    assert cb._earlier_phase_results("1", "python") == (tmp_path / "phase1_results.json", "")
    (tmp_path / "phase1_results.json").unlink()
    assert cb._earlier_phase_results("1", "python") == (None, "")


@pytest.mark.parametrize(
    ("engine", "files", "value", "says"),
    [
        ("python", {"phase1_results.json": 0.25}, "0.25", "Loaded phase 1 parameters (1 params)\n"),
        ("csharp", {"phase1_results.json": 0.25}, "0.25",
         "(1 params) from phase1_results.json, calibrated on the PYTHON engine"),
        ("csharp", {"phase1_results.json": 0.25, "phase1_csharp_results.json": 0.5}, "0.5",
         "(1 params) from phase1_csharp_results.json\n"),
        ("csharp", {"phase1_csharp_results.json": 0.5}, "0.5",
         "(1 params) from phase1_csharp_results.json\n"),
        ("python", {"phase1_csharp_results.json": 0.5}, None, "No calibration results found"),
    ],
)  # fmt: skip
def test_validation_runs_the_parameters_of_the_engine_asked_for(
    monkeypatch, tmp_path, capsys, engine, files, value, says
):
    for name, v in files.items():
        _results(tmp_path, name, v)
    monkeypatch.setattr(cb, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cb, "_require_csharp", lambda *a, **k: None)
    calls: list[tuple] = []

    def fake_run_simulation(config, overrides, **kw):
        calls.append((dict(overrides), kw["engine"]))
        return {f"{sp}_mean": 1000.0 for sp in cb.SPECIES_NAMES}

    monkeypatch.setattr(cb, "run_simulation", fake_run_simulation)
    cb.validate_calibration(n_years=1, n_seeds=2, engine=engine)
    assert says in capsys.readouterr().out
    if value is None:
        assert calls == []  # a C# calibration is not a Python one: nothing to validate
    else:
        assert calls == [({"species.k.sp0": value}, engine)] * 2


def test_phase_two_inherits_from_the_engine_asked_for(monkeypatch, tmp_path, capsys):
    _results(tmp_path, "phase1_results.json", 0.25)
    _results(tmp_path, "phase1_csharp_results.json", 0.5)
    monkeypatch.setattr(cb, "RESULTS_DIR", tmp_path)
    seen: dict = {}

    def stop(base_config, targets, param_keys, **kw):
        seen.update(k=base_config["species.k.sp0"], engine=kw["engine"])
        raise SystemExit("objective")

    monkeypatch.setattr(cb, "make_objective", stop)
    monkeypatch.setattr(cb, "_require_csharp", lambda *a, **k: None)
    for engine, want, line in (
        ("python", "0.25", "calibrated params from phase1_results.json\n"),
        ("csharp", "0.5", "calibrated params from phase1_csharp_results.json\n"),
    ):
        with pytest.raises(SystemExit, match="objective"):
            cb.run_calibration(phase="2", engine=engine)
        assert seen == {"k": want, "engine": engine}
        assert line in capsys.readouterr().out
    (tmp_path / "phase1_csharp_results.json").unlink()
    with pytest.raises(SystemExit, match="objective"):
        cb.run_calibration(phase="2", engine="csharp")
    assert seen == {"k": "0.25", "engine": "csharp"}
    assert "from phase1_results.json, calibrated on the PYTHON engine" in capsys.readouterr().out
