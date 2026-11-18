"""Plan task 1.8: ``CSharpEngine`` end to end (spec 9.2, 8.4, 8.5).

Every C# run happens in a child process (``tests/_csharp_e2e_child.py``): the CLR never enters
pytest. What is checked:
- ``run_in_memory`` frames have the Python engine's shape, columns and dtypes, and the leading
  records (no draw involved yet) equal Python's, through the Python-owned result buffers;
- ``run`` writes the same file set;
- a cancel stops within one step and leaves the written records valid and the rest untouched;
  a pre-set cancel token and a past deadline raise ``SimulationCancelled`` / ``CSharpEngineTimeout``;
- a stock config (no ``output.minimal``) is refused naming the full output families;
- a ``KeyboardInterrupt`` or a SIGALRM-raised timeout in the waiting thread is re-raised only
  after the worker has joined (spec 8.5's buffer-lifetime invariant).
None of these assertions depends on timing; triggers key on the engine's own step counter.

The test needs the built engine. It skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine_cs import CSharpEngine, CSharpEngineError, csharp_engine_available

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "tests" / "_csharp_e2e_child.py"
FRAMES = ("biomass", "abundance", "yield_biomass", "mortality")


@pytest.fixture(scope="module", autouse=True)
def engine_ok():
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")


def _raw(demo: str, d: Path, nyear: int, minimal: bool = True, extra: dict | None = None) -> dict:
    raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, d)["config_file"])))
    raw["simulation.time.nyear"] = str(nyear)
    if minimal:
        raw["output.minimal.enabled"] = "true"
    return raw | (extra or {})


def _child(
    mode: str, raw: dict, out: Path, arg: str = "", timeout: int = 900
) -> subprocess.CompletedProcess:
    cfg = out.parent / f"{out.name}.raw.json"
    cfg.write_text(json.dumps(raw))
    args = [sys.executable, str(CHILD), str(ROOT), mode, str(cfg), str(out)] + (
        [arg] if arg else []
    )
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc


@pytest.mark.parametrize(
    ("demo", "nyear", "leading"), [("bay_of_biscay", 1, 4), ("eec_full", 2, 2)]
)
def test_run_in_memory_frames_match_the_python_engine(demo, nyear, leading, tmp_path):
    raw = _raw(demo, tmp_path, nyear, extra={"output.recordfrequency.ndt": "1"})
    _child("memory", raw, tmp_path / "cs.pkl", "11")
    cs = pickle.loads((tmp_path / "cs.pkl").read_bytes())
    py = PythonEngine().run_in_memory(raw, seed=11)
    for name in FRAMES:
        a, b = cs[name], getattr(py, name)()
        assert a.shape == b.shape, name
        assert list(a.columns) == list(b.columns), name
        assert list(a.dtypes) == list(b.dtypes), name
        num = a.select_dtypes("number")
        if name != "mortality":  # mortality() is a RATE: an emptied stage reads inf (CLAUDE.md)
            assert np.isfinite(num.to_numpy()).all() and (num.to_numpy() >= 0).all(), name
        # The leading records involve no draw yet (seeding, larva deaths on unfed eggs).
        head_a, head_b = (
            num.iloc[:leading].to_numpy(),
            b.select_dtypes("number").iloc[:leading].to_numpy(),
        )
        np.testing.assert_allclose(head_a, head_b, rtol=1e-9, atol=0, err_msg=name)
    assert cs["mortality"].select_dtypes("number").to_numpy().sum() > 0  # the run did something


def test_run_writes_the_same_file_set(tmp_path):
    raw = _raw("bay_of_biscay", tmp_path, 1)
    _child("files", raw, tmp_path / "cs")
    PythonEngine().run(raw, tmp_path / "py", seed=3)

    def tree(root: Path) -> list[str]:
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

    assert tree(tmp_path / "cs") == tree(tmp_path / "py") and tree(tmp_path / "py")


def test_cancel_stops_within_one_step_and_leaves_records_valid(tmp_path):
    raw = _raw("bay_of_biscay", tmp_path, 2, extra={"output.recordfrequency.ndt": "2"})
    k = 7
    _child("cancel_at", raw, tmp_path / "c.json", str(k))
    r = json.loads((tmp_path / "c.json").read_text())
    assert r["status"] == 1  # Cancelled
    assert k <= r["steps"] <= k + 1 < r["n_steps"]
    assert r["records"] == r["steps"] // 2
    assert r["finite_rows"] == [i < r["records"] for i in range(len(r["finite_rows"]))]
    assert r["record_step"][: r["records"]] == [2 * i + 1 for i in range(r["records"])]
    assert r["control"][1:3] == [r["steps"], r["records"]]


@pytest.mark.parametrize(
    ("kind", "error"), [("precancel", "SimulationCancelled"), ("deadline", "CSharpEngineTimeout")]
)
def test_precancel_and_deadline_stop_at_step_zero(kind, error, tmp_path):
    _child("error", _raw("bay_of_biscay", tmp_path, 1), tmp_path / "e.json", kind)
    r = json.loads((tmp_path / "e.json").read_text())
    assert r["raised"] == error and "step 0/" in r["message"]


def test_stock_configs_are_refused_naming_the_full_output_families(tmp_path):
    raw = _raw("eec_full", tmp_path, 1, minimal=False)
    with pytest.raises(CSharpEngineError, match=r"full_outputs \(phase 2\)"):
        CSharpEngine().run_in_memory(raw)  # refused before the CLR is touched


def test_live_map_observer_is_refused(tmp_path):
    with pytest.raises(CSharpEngineError, match="live map"):
        CSharpEngine().run(
            _raw("bay_of_biscay", tmp_path, 1), tmp_path / "o", step_observer=lambda *a: None
        )


@pytest.mark.parametrize(
    ("kind", "error"), [("int", "KeyboardInterrupt"), ("alarm", "_SimTimeout")]
)
def test_signals_are_reraised_only_after_the_worker_joined(kind, error, tmp_path):
    raw = _raw("eec_full", tmp_path, 10)
    _child("interrupted", raw, tmp_path / "s.json", kind)
    r = json.loads((tmp_path / "s.json").read_text())
    assert r["raised"] == error, r  # a run that finished first would report None: a failure
    # The engine had RETURNED when the exception surfaced: no step completes afterwards. (A
    # thread-liveness probe is not the invariant: the worker may still be exiting.)
    assert r["cancel_slot"] == 1
    assert r["steps_at_catch"] == r["steps_later"] < r["n_steps"]


PHASES = ["movement", "regroup", "mortality", "life_cycle", "outputs"]


def test_run_summary_carries_the_phase_split(tmp_path):
    """Task 1.11: RunSummary's per-phase seconds. The phases are intervals of the step loop
    on the engine's own clock, so they cannot exceed the run's elapsed time; and they are
    nearly all of it (measured 99.4% on BoB, 99.7% on EEC), so a unit slip or a dropped phase
    falls outside [0.9, 1] of the elapsed time."""
    _child("summary", _raw("bay_of_biscay", tmp_path, 1), tmp_path / "s.json")
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["steps"] == s["n_steps"] == 24
    assert list(s["phases"]) == PHASES
    assert all(v > 0 for v in s["phases"].values()), s["phases"]
    total = sum(s["phases"].values())
    assert 0.9 * s["elapsed"] <= total <= s["elapsed"], (total, s["elapsed"])


def test_benchmark_and_profile_scripts_run_the_csharp_engine(tmp_path):
    """``--engine csharp`` in scripts/benchmark_engine.py and scripts/profile_engine.py (task
    1.11), each in its own process."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))  # scripts/ has no osmose: this wins (CLAUDE.md)

    def script(*args: str) -> None:
        proc = subprocess.run(
            [sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900
        )
        assert proc.returncode == 0, proc.stderr[-4000:]

    bench, prof = tmp_path / "bench.json", tmp_path / "prof.json"
    script("scripts/benchmark_engine.py", "--engine", "csharp", "--config", "examples",
           "--years", "1", "--repeats", "1", "--warmup", "0", "--threads", "2",
           "--output", str(bench))  # fmt: skip
    b = json.loads(bench.read_text())
    assert (b["engine"], b["threads"], b["minimal"]) == ("csharp", 2, True)
    assert list(b["phases_s"]) == PHASES and b["bundle_s"] > 0 and b["engine_s"] > 0
    assert set(b["final_biomass"]) and all(v >= 0 for v in b["final_biomass"].values())
    script("scripts/profile_engine.py", "--engine", "csharp", "--config", "examples",
           "--years", "1", "--json", str(prof))  # fmt: skip
    p = json.loads(prof.read_text())
    assert p["engine"] == "csharp" and list(p["phases_s"]) == PHASES
    assert p["cold_first_call_s"] > 0 and p["engine_elapsed_s"] > 0
    assert "bundle build" in p["stages_s"] and "write_outputs()" in p["stages_s"]
