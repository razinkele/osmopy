"""Plan task 1.12: choosing an engine by name (C# engine spec 9.3), without loading the CLR."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem
from osmose.calibration.problem import _EvalSpec as EvalSpec
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.select import IN_PROCESS_ENGINES, get_engine

ROOT = Path(__file__).resolve().parents[1]
PARAMS = [FreeParameter("species.k.sp0", 0.1, 0.5)]


def test_get_engine_by_name():
    from osmose.engine_cs import CSharpEngine

    assert IN_PROCESS_ENGINES == ("python", "csharp")
    assert isinstance(get_engine("python"), PythonEngine)
    cs = get_engine("csharp", threads=3)
    assert isinstance(cs, CSharpEngine) and cs.threads == 3  # constructing it loads nothing
    with pytest.raises(ValueError, match="unknown in-process engine 'java'"):
        get_engine("java")


def test_no_environment_variable_switches_the_engine(monkeypatch):
    """Spec 9.3: every number must come from a NAMED engine."""
    for var in ("OSMOSE_ENGINE", "OSMOSE_DEFAULT_ENGINE"):
        monkeypatch.setenv(var, "csharp")
    assert isinstance(get_engine("python"), PythonEngine)
    src = (ROOT / "osmose" / "engine" / "select.py").read_text()
    assert "os.environ" not in src and "getenv(" not in src


def _problem(tmp_path, **kw):
    return OsmoseCalibrationProblem(
        free_params=PARAMS, objective_fns=[lambda r: 0.0], base_config_path=kw.pop("base", tmp_path / "cfg"),
        work_dir=tmp_path / "work", jar_path=kw.pop("jar_path", None), **kw,
    )  # fmt: skip


def test_calibration_engine_names_and_the_java_alias(tmp_path):
    assert _problem(tmp_path).engine == "python"
    java = _problem(tmp_path, use_java_engine=True, jar_path=tmp_path / "x.jar")
    assert java.engine == "java" and java.use_java_engine
    assert _problem(tmp_path, engine="java", jar_path=tmp_path / "x.jar").use_java_engine
    with pytest.raises(ValueError, match="contradicts"):
        _problem(tmp_path, use_java_engine=True, engine="csharp", jar_path=tmp_path / "x.jar")
    with pytest.raises(ValueError, match="unknown engine"):
        _problem(tmp_path, engine="fortran")


def test_a_csharp_calibration_of_an_unrunnable_config_is_refused_up_front(tmp_path):
    """Every candidate would otherwise fail, and be scored as a failure, one by one.

    The refusal has two causes, and each is asserted by its own message, so that this test
    cannot pass on an unavailable engine while the config check goes untested (it did: the
    stock config it used became runnable in task 2.6c, and only a run WITH the engine noticed).
    Logistic fishing selectivity is a feature the engine does not implement. A growth cap
    (species.lmax) was the example here until the engine ran it: such a config is accepted."""
    from osmose.engine_cs import csharp_engine_available

    stock = Path(osmose_demo("eec_full", tmp_path / "stock")["config_file"])
    capped = Path(osmose_demo("eec_full", tmp_path / "capped")["config_file"])
    capped.write_text(capped.read_text() + "\nspecies.lmax.sp0;25\n")
    logistic = Path(osmose_demo("eec_full", tmp_path / "logistic")["config_file"])
    fishing = logistic.parent / "eec_param-fishing.csv"  # the key is set there: change it there
    was = "fisheries.selectivity.type.fsh0;0"
    assert was in fishing.read_text()
    fishing.write_text(fishing.read_text().replace(was, "fisheries.selectivity.type.fsh0;1"))
    available, _ = csharp_engine_available()
    if not available:
        with pytest.raises(ValueError, match="engine='csharp': the C# engine is unavailable"):
            _problem(tmp_path, base=logistic, engine="csharp")
        return
    with pytest.raises(ValueError, match=r"cannot run this config: .*selectivity_logistic"):
        _problem(tmp_path, base=logistic, engine="csharp")
    assert _problem(tmp_path, base=stock, engine="csharp").engine == "csharp"  # as shipped: runs
    assert _problem(tmp_path, base=capped, engine="csharp").engine == "csharp"


def test_the_engine_reaches_the_workers_and_the_cache_key(tmp_path):
    py = _problem(tmp_path)
    assert py._eval_spec().engine == "python"
    # The Python key is unchanged by this task (existing caches stay valid).
    from osmose import __version__

    overrides = {"species.k.sp0": "0.3"}
    raw = f"{sorted(overrides.items())}|python-{__version__}|{py._base_config_hash}"
    assert py._cache_key(overrides) == hashlib.sha256(raw.encode()).hexdigest()
    cs = object.__new__(OsmoseCalibrationProblem)  # bypass the up-front run check
    cs.__dict__.update(py.__dict__, engine="csharp")
    assert cs._cache_key(overrides) != py._cache_key(overrides)
    assert cs._eval_spec().engine == "csharp"  # what each spawned worker rebuilds the problem with
    assert EvalSpec.__dataclass_fields__["engine"].default == "python"  # old pickles: Python


def test_cli_run_dispatches_in_process_engines(tmp_path, monkeypatch):
    from osmose import cli
    from osmose.engine import select

    seen = {}

    class Fake:
        def run(self, raw, output_dir, seed=0):
            seen.update(raw=raw, output_dir=output_dir, seed=seed)

    monkeypatch.setattr(select, "get_engine", lambda name: seen.setdefault("name", name) and Fake())
    cfg = Path(osmose_demo("bay_of_biscay", tmp_path)["config_file"])
    out = tmp_path / "o"
    monkeypatch.setattr(
        sys,
        "argv",
        ["osmose", "run", str(cfg), "--engine", "csharp", "--seed", "4", "--output", str(out)],
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code in (0, None)
    assert (
        seen["name"] == "csharp" and seen["seed"] == 4 and seen["output_dir"] == out and seen["raw"]
    )


def test_calibrate_baltic_no_longer_refuses_the_csharp_engine_itself():
    """Since the owner's G2 decision (2026-11-19) the script carries the engine into its
    evaluations. What it still refuses is a calibration the C# engine cannot run, with the
    engine's own reason: tests/test_calibrate_baltic_engine.py."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibrate_baltic

    assert not hasattr(calibrate_baltic, "_refuse_unplumbed_engine")
    assert calibrate_baltic.ENGINES == ("python", "csharp")
    source = (ROOT / "scripts" / "calibrate_baltic.py").read_text(encoding="utf-8")
    assert "does not carry it into its evaluations yet" not in source
