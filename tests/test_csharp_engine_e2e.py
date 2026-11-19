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


def test_a_run_on_a_worker_thread_is_cancelled_by_a_token_set_from_another_thread(tmp_path):
    """What the UI's Cancel button does (task 2.7): the engine is on a worker thread, and the
    token is set from another thread while it runs. The other cancel tests set the token before
    the start, write the control slot directly, or raise a signal on the main thread."""
    raw = _raw("eec_full", tmp_path, 10)
    k = 5
    _child("cancel_thread", raw, tmp_path / "c.json", str(k))
    r = json.loads((tmp_path / "c.json").read_text())
    assert r["raised"] == "SimulationCancelled" and r["joined"], r
    assert k <= r["steps_at_join"] < r["n_steps"]  # stopped mid-run (a finished run raises nothing)
    assert r["steps_at_join"] == r["steps_later"]  # the engine had returned: no step afterwards
    assert r["message"] == f"cancelled at step {r['steps_at_join']}/{r['n_steps']}"
    assert r["cancel_slot"] == 1 and r["seconds_to_stop"] < 5.0
    assert r["files"] == []  # a cancelled run writes nothing


@pytest.mark.parametrize(
    ("kind", "error"), [("precancel", "SimulationCancelled"), ("deadline", "CSharpEngineTimeout")]
)
def test_precancel_and_deadline_stop_at_step_zero(kind, error, tmp_path):
    _child("error", _raw("bay_of_biscay", tmp_path, 1), tmp_path / "e.json", kind)
    r = json.loads((tmp_path / "e.json").read_text())
    assert r["raised"] == error and "step 0/" in r["message"]


# ------------------------------------------------- stock configs: the full outputs (task 2.6c)

STOCK = {"eec_full": 2, "baltic": 1, "bay_of_biscay": 1}  # demo: years


def _families_row(record, name: str, n_species: int) -> np.ndarray | None:
    """A ``StepOutput`` field as the row the engine writes (None for a field left None)."""
    from tests.test_csharp_outputs_families import BY_AGE, BY_SIZE, PRESENT

    v = getattr(record, name)
    if v is None:
        return None
    if name in BY_AGE:
        return np.concatenate([v[sp] for sp in range(n_species)])
    if name in BY_SIZE:
        return np.stack([v[sp] for sp in range(n_species)])
    if name in PRESENT:
        row = np.full(n_species, np.nan)
        for sp, x in v.items():
            row[sp] = x
        return row
    return np.asarray(v, dtype=np.float64)


@pytest.mark.parametrize("demo", sorted(STOCK))
def test_a_stock_run_hands_on_exactly_the_records_the_engine_computed(demo, tmp_path):
    """The in-memory transport, end to end: a stock config (every output family the demo ships
    with) through ``CSharpEngine``, against the same bundle run by the probe, which reads the
    engine's own results without the caller's buffers. Every family of every record must be the
    same, bit for bit: the shapes the two sides allocate, the names, the order, the conversion
    into ``StepOutput``."""
    from osmose.engine.config import EngineConfig
    from osmose.engine_cs.build import build_bundle, family_shapes
    from tests.test_csharp_outputs_families import _same
    from tests.test_csharp_pipeline import FAMILIES, _probe

    extra = {"output.recordfrequency.ndt": "6", "output.step0.include": "true"}
    raw = _raw(demo, tmp_path, STOCK[demo], minimal=False, extra=extra)
    _child("outputs", raw, tmp_path / "cs.pkl", "11")
    got = pickle.loads((tmp_path / "cs.pkl").read_bytes())
    records = got["records"]

    config = EngineConfig.from_dict(raw)
    build_bundle(config, raw, PythonEngine()._resolve_grid(raw), seed=11).save(tmp_path / "m.bin")
    probe = _probe("Run", tmp_path / "m.bin", tmp_path / "probe.bin")
    names = list(family_shapes(config))
    assert got["resource_names"] == config.resource_names and config.resource_names
    assert len(records) == len(probe.get("out.record_step")) == 1 + STOCK[demo] * 4
    assert [r.step for r in records] == probe.get("out.record_step").tolist()
    assert [r.window_steps for r in records] == probe.get("out.window_steps").tolist()

    nf = config.n_species
    for i, r in enumerate(records):
        for cs_name, py_name in FAMILIES:
            a, b = probe.get(f"out.{cs_name}")[i], np.asarray(getattr(r, py_name))
            assert a.tobytes() == b.tobytes(), (cs_name, i)
        for name in names:
            row, want = _families_row(r, name, nf), probe.get(f"out.{name}")[i]
            if row is None:  # only the diet of the initial record, which no step preceded
                assert name == "diet_by_species" and r.step == -1 and not want.any()
            else:
                assert _same(row, want), (name, i)

    # The demo's own switches: every family it ships with came through, and did something.
    want = {"abundance_by_stage", "mortality_rate_by_cause_stage", "diet_by_species",
            "biomass_by_age", "biomass_by_size", "abundance_by_size", "mean_tl", "mean_size",
            "yield_n"} | ({"ssb"} if demo == "baltic" else set())  # fmt: skip
    assert want <= set(names)
    # (Over the run, not in the last record: Bay of Biscay's fish stop feeding within the year.)
    last = records[-1]
    assert last.diet_by_species is not None
    assert sum(r.diet_by_species.sum() for r in records if r.diet_by_species is not None) > 0
    assert any(r.mean_tl for r in records) and any(r.mean_size for r in records)
    assert last.biomass_by_age and sum(v.sum() for v in last.biomass_by_age.values()) > 0
    assert last.abundance_by_stage is not None and last.abundance_by_stage.sum() > 0
    if demo == "baltic":  # the background predators' rows, and SSB
        assert last.diet_by_species.shape[0] == nf + 2 and last.diet_by_species[nf:].sum() > 0
        assert last.ssb is not None and last.ssb.sum() > 0


def test_the_stock_baltic_run_is_the_same_at_1_2_and_14_threads(tmp_path):
    """The spec's determinism goal on the config AS SHIPPED, every output family on: the diet
    rows are written per predator school inside the parallel kernel, and the families are folded
    from what the threads leave behind. The records must not depend on the thread count, and
    must depend on the seed (or the digest would be blind)."""
    extra = {"output.recordfrequency.ndt": "6", "output.step0.include": "true"}
    raw = _raw("baltic", tmp_path, 2, minimal=False, extra=extra)
    _child("thread_digests", raw, tmp_path / "d.json", "1,7")
    got = json.loads((tmp_path / "d.json").read_text())
    assert set(got) == {f"{seed}:{threads}" for seed in (1, 7) for threads in (1, 2, 14)}
    for seed in (1, 7):
        one, two, many = (got[f"{seed}:{threads}"] for threads in (1, 2, 14))
        assert one["records"] == two["records"] == many["records"] > 2
        assert one["digest"] == two["digest"], f"seed {seed}: 1 and 2 threads differ"
        assert one["digest"] == many["digest"], f"seed {seed}: 1 and 14 threads differ"
    assert got["1:1"]["digest"] != got["7:1"]["digest"]
    # The digest covers the families Baltic ships with, not only the minimal outputs.
    assert {
        "biomass", "abundance", "yield_by_species", "mortality_by_cause", "abundance_by_stage",
        "biomass_by_age", "ssb", "yield_n", "mean_size", "mean_tl", "diet_by_species",
    } <= set(got["1:1"]["fields"])  # fmt: skip


def test_a_run_leaves_its_arrays_to_no_later_run(tmp_path):
    """Spec 10.4's memory budget, for a process that runs many simulations (a calibration
    worker, the UI). A run's own arrays are dead when it returns. Left to the collector's
    schedule they stayed for two or three more runs (measured on Baltic, 2026-11-19: a managed
    heap sawing between 30 and 88 MB, resident memory up to 324 MB against the Python engine's
    309). ``OsmoseEngine.RunSimulation`` collects them before it returns: the heap is back to a
    megabyte or two after EVERY run, and every run shows a full collection."""
    raw = _raw("baltic", tmp_path, 3, minimal=False)
    _child("heap_after_runs", raw, tmp_path / "h.json", "5")
    rows = json.loads((tmp_path / "h.json").read_text())
    assert len(rows) == 5
    heaps, gen2 = [r["heap_mb"] for r in rows], [r["gen2"] for r in rows]
    assert max(heaps) < 5.0, heaps
    assert gen2 == sorted(set(gen2)), gen2  # strictly rising: a full collection in every run


def _structure(frame) -> tuple:
    return (frame.shape, list(frame.columns), [str(t) for t in frame.dtypes])


@pytest.mark.parametrize("demo", sorted(STOCK))
def test_a_stock_run_gives_the_frames_the_python_engine_gives(demo, tmp_path):
    """Same keys, shapes, column labels and dtypes as the Python engine's frames for the same
    config. The values differ (another random stream), the structure must not: it includes the
    prey labels of the diet frames, which need the resource names the run publishes."""
    raw = _raw(demo, tmp_path, 1, minimal=False)
    _child("frames", raw, tmp_path / "cs.pkl", "11")
    cs = pickle.loads((tmp_path / "cs.pkl").read_bytes())
    py = PythonEngine().run_in_memory(raw, seed=11)._csv_cache
    assert sorted(cs) == sorted(py) and len(py) > 10
    labels: set[str] = set()
    for key in py:
        a, b = cs[key], py[key]
        if isinstance(b, dict):
            assert sorted(a) == sorted(b), key
            for sub in b:
                assert _structure(a[sub]) == _structure(b[sub]), (key, sub)
                labels.update(map(str, b[sub].columns))
        else:
            assert _structure(a) == _structure(b), key
            labels.update(map(str, b.columns))
    resource = {"eec_full": "Dinoflagellates", "baltic": "Benthos", "bay_of_biscay": "SmallPhyto"}
    assert any(resource[demo] in label for label in labels)  # a resource is a prey label


def _header(path: Path) -> list[str]:
    """The lines of a CSV before its first data row (a row whose first field is a number)."""
    head = []
    for line in path.read_text().splitlines():
        first = line.split(";")[0].split(",")[0].strip().strip('"')
        try:
            float(first)
        except ValueError:
            head.append(line)
            continue
        break
    return head


@pytest.mark.parametrize("demo", sorted(STOCK))
def test_a_stock_run_writes_the_files_the_python_engine_writes(demo, tmp_path):
    raw = _raw(demo, tmp_path, 1, minimal=False)
    _child("files", raw, tmp_path / "cs")
    PythonEngine().run(raw, tmp_path / "py", seed=3)

    def tree(root: Path) -> list[str]:
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

    files = tree(tmp_path / "py")
    assert tree(tmp_path / "cs") == files and len(files) > 20
    assert any("diet" in f.lower() for f in files) and any("mortality" in f.lower() for f in files)
    for rel in files:
        if not rel.endswith(".csv"):
            continue
        a, b = tmp_path / "cs" / rel, tmp_path / "py" / rel
        assert _header(a) == _header(b), rel
        assert len(a.read_text().splitlines()) == len(b.read_text().splitlines()), rel


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
