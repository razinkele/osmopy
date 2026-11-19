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

# demo: years. Benguela since 2026-11-19: it switches incoming flux on and gives no species a
# flux file, which the capability check no longer takes for a use of the feature.
STOCK = {"eec_full": 2, "baltic": 1, "bay_of_biscay": 1, "benguela": 1}
#: Demos that ship with most output switches off (Benguela: biomass, yield, mortality and the
#: life-stage outputs only). What they do ship must still match; the checks that a config
#: exercises the diet and the distributions do not apply to them.
SPARSE = {"benguela"}


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
    last = records[-1]
    assert last.abundance_by_stage is not None and last.abundance_by_stage.sum() > 0
    if demo in SPARSE:
        assert {"abundance_by_stage", "mortality_rate_by_cause_stage"} <= set(names)
        assert last.diet_by_species is None and last.biomass_by_age is None  # switched off
        assert last.biomass.sum() > 0
        return
    want = {"abundance_by_stage", "mortality_rate_by_cause_stage", "diet_by_species",
            "biomass_by_age", "biomass_by_size", "abundance_by_size", "mean_tl", "mean_size",
            "yield_n"} | ({"ssb"} if demo == "baltic" else set())  # fmt: skip
    assert want <= set(names)
    # (Over the run, not in the last record: Bay of Biscay's fish stop feeding within the year.)
    assert last.diet_by_species is not None
    assert sum(r.diet_by_species.sum() for r in records if r.diet_by_species is not None) > 0
    assert any(r.mean_tl for r in records) and any(r.mean_size for r in records)
    assert last.biomass_by_age and sum(v.sum() for v in last.biomass_by_age.values()) > 0
    assert last.abundance_by_stage is not None and last.abundance_by_stage.sum() > 0
    if demo == "baltic":  # the background predators' rows, and SSB
        assert last.diet_by_species.shape[0] == nf + 2 and last.diet_by_species[nf:].sum() > 0
        assert last.ssb is not None and last.ssb.sum() > 0


_MINIMAL_FIELDS = {"biomass", "abundance", "yield_by_species", "mortality_by_cause"}
STOCK_FAMILIES = {
    "baltic": _MINIMAL_FIELDS | {
        "abundance_by_stage", "biomass_by_age", "ssb", "yield_n", "mean_size", "mean_tl",
        "diet_by_species",
    },
    "benguela": _MINIMAL_FIELDS | {"abundance_by_stage", "mortality_rate_by_cause_stage"},
}  # fmt: skip


@pytest.mark.parametrize("demo", ["baltic", "benguela"])
def test_a_stock_run_is_the_same_at_1_2_and_14_threads(demo, tmp_path):
    """The spec's determinism goal on the config AS SHIPPED, every output family on: the diet
    rows are written per predator school inside the parallel kernel, and the families are folded
    from what the threads leave behind. The records must not depend on the thread count, and
    must depend on the seed (or the digest would be blind)."""
    extra = {"output.recordfrequency.ndt": "6", "output.step0.include": "true"}
    raw = _raw(demo, tmp_path, 2, minimal=False, extra=extra)
    _child("thread_digests", raw, tmp_path / "d.json", "1,7")
    got = json.loads((tmp_path / "d.json").read_text())
    assert set(got) == {f"{seed}:{threads}" for seed in (1, 7) for threads in (1, 2, 14)}
    for seed in (1, 7):
        one, two, many = (got[f"{seed}:{threads}"] for threads in (1, 2, 14))
        assert one["records"] == two["records"] == many["records"] > 2
        assert one["digest"] == two["digest"], f"seed {seed}: 1 and 2 threads differ"
        assert one["digest"] == many["digest"], f"seed {seed}: 1 and 14 threads differ"
    assert got["1:1"]["digest"] != got["7:1"]["digest"]
    # The digest covers the families the demo ships with, not only the minimal outputs.
    assert STOCK_FAMILIES[demo] <= set(got["1:1"]["fields"]), sorted(got["1:1"]["fields"])


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
    assert sorted(cs) == sorted(py) and len(py) > (3 if demo in SPARSE else 10)
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
    if demo in SPARSE:  # no diet frame, so no prey labels: the species columns are there
        assert any("anchovy" in label.lower() for label in labels)
        return
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
    assert tree(tmp_path / "cs") == files and len(files) > (10 if demo in SPARSE else 20)
    assert any("mortality" in f.lower() for f in files)
    assert any("diet" in f.lower() for f in files) != (demo in SPARSE)
    for rel in files:
        if not rel.endswith(".csv"):
            continue
        a, b = tmp_path / "cs" / rel, tmp_path / "py" / rel
        assert _header(a) == _header(b), rel
        assert len(a.read_text().splitlines()) == len(b.read_text().splitlines()), rel


# ------------------------------------------ task 2.2: the growth cap and immobile species

CAP_CM = 12.0  # above every EEC egg size: the dogfish hatches at 9.5 cm
FINE_SIZE_CLASSES = {
    "output.distrib.bysize.min": "0",
    "output.distrib.bysize.max": "60",
    "output.distrib.bysize.incr": "2",
}
FIRST_CLASS_ABOVE_THE_CAP = 7  # [14, 16) cm: the class [12, 14) holds a fish at the cap
IMMOBILE = 3  # the EEC species given the movement method "none"


def _run_records(engine: str, raw: dict, out: Path, seed: int = 5) -> list:
    """The records of one run: the Python engine here, the C# engine in a child process."""
    if engine == "csharp":
        _child("outputs", raw, out, str(seed))
        # The file was written a moment ago by this test's own child process.
        return pickle.loads(out.read_bytes())["records"]
    import osmose.engine.simulate as sim

    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, seed)
    return sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)


def _above_the_cap(records: list, n_species: int) -> float:
    return float(
        sum(
            np.asarray(r.abundance_by_size[sp])[FIRST_CLASS_ABOVE_THE_CAP:].sum()
            for r in records
            for sp in range(n_species)
        )
    )


@pytest.mark.parametrize("engine", ["python", "csharp"])
def test_no_fish_outgrows_its_cap_on_a_whole_run(engine, tmp_path):
    """``species.lmax`` over a whole run (no shipped config sets it). Every EEC species is
    capped at 12 cm from the start. No record may hold a fish in a size class above the cap,
    and fish must have reached it. The uncapped run is the control: there the same classes
    fill, so an empty class is the cap's doing and not the output's.

    (A cap below a species' egg size would leave its newly laid eggs above the cap until
    their first growth step, on both engines: the cap is applied in growth.)"""
    base = _raw("eec_full", tmp_path, 2, minimal=False, extra=FINE_SIZE_CLASSES)
    n = int(base["simulation.nspecies"])
    capped = base | {f"species.lmax.sp{i}": str(CAP_CM) for i in range(n)}
    records = _run_records(engine, capped, tmp_path / "capped.pkl")
    assert len(records) >= 2
    assert _above_the_cap(records, n) == 0.0
    at_the_cap = sum(
        float(np.asarray(records[-1].abundance_by_size[sp])[FIRST_CLASS_ABOVE_THE_CAP - 1])
        for sp in range(n)
    )
    assert at_the_cap > 0
    assert _above_the_cap(_run_records(engine, base, tmp_path / "free.pkl"), n) > 0


@pytest.mark.parametrize("engine", ["python", "csharp"])
def test_an_immobile_species_is_never_placed_and_takes_no_part(engine, tmp_path):
    """The movement method ``none``: the species' schools are never put on the grid. They are
    not in any cell, so nothing eats them, they starve of nothing, and no fleet catches them.
    They are still there and counted. The control is the same run with the species moving:
    there it is eaten and caught, so a zero is the method's doing."""
    from osmose.engine.state import MortalityCause as C

    base = _raw("eec_full", tmp_path, 2, minimal=False)
    still = base | {f"movement.distribution.method.sp{IMMOBILE}": "none"}
    in_a_cell = [int(C.PREDATION), int(C.STARVATION), int(C.FISHING)]

    def losses(records: list) -> float:
        return float(
            sum(np.asarray(r.mortality_by_cause)[IMMOBILE, in_a_cell].sum() for r in records)
        )

    records = _run_records(engine, still, tmp_path / "still.pkl")
    assert losses(records) == 0.0
    assert all(float(np.asarray(r.yield_by_species)[IMMOBILE]) == 0.0 for r in records)
    assert float(np.asarray(records[-1].abundance)[IMMOBILE]) > 0
    others = [sp for sp in range(int(base["simulation.nspecies"])) if sp != IMMOBILE]
    assert float(np.asarray(records[-1].mortality_by_cause)[others][:, in_a_cell].sum()) > 0
    assert losses(_run_records(engine, base, tmp_path / "moving.pkl")) > 0


def test_an_immobile_species_reads_the_same_on_both_engines(tmp_path):
    """The engines draw from different generators, so whole runs are compared in distribution.
    An immobile species is the exception: no draw touches a school that is never placed. Its
    series are the same at two seeds on the Python engine (and a moving species' are not, so
    the comparison is not blind), and the C# engine gives the same numbers."""
    base = _raw("eec_full", tmp_path, 2, minimal=False, extra={"output.recordfrequency.ndt": "4"})
    still = base | {f"movement.distribution.method.sp{IMMOBILE}": "none"}

    def series(records: list, name: str, sp: int = IMMOBILE) -> np.ndarray:
        return np.array([float(np.asarray(getattr(r, name))[sp]) for r in records])

    one = _run_records("python", still, tmp_path / "p5.pkl", seed=5)
    other = _run_records("python", still, tmp_path / "p6.pkl", seed=6)
    cs = _run_records("csharp", still, tmp_path / "c5.pkl", seed=5)
    assert len(one) == len(other) == len(cs) == 12
    moving = next(sp for sp in range(int(base["simulation.nspecies"])) if sp != IMMOBILE)
    assert not np.array_equal(series(one, "abundance", moving), series(other, "abundance", moving))
    for name in ("abundance", "biomass"):
        ref = series(one, name)
        assert ref[-1] > 0 and np.array_equal(ref, series(other, name)), name
        np.testing.assert_allclose(series(cs, name), ref, rtol=1e-12, atol=0.0, err_msg=name)
    # What it does lose, it loses to the one cause that needs no cell, on both engines alike.
    lost = np.array([np.asarray(r.mortality_by_cause)[IMMOBILE] for r in one]).sum(axis=0)
    lost_cs = np.array([np.asarray(r.mortality_by_cause)[IMMOBILE] for r in cs]).sum(axis=0)
    assert np.count_nonzero(lost) == 1 and lost.sum() > 0
    np.testing.assert_allclose(lost_cs, lost, rtol=1e-12, atol=0.0)


def test_a_capped_run_with_an_immobile_species_is_the_same_at_1_2_and_14_threads(tmp_path):
    base = _raw("eec_full", tmp_path, 2, minimal=False, extra=FINE_SIZE_CLASSES)
    n = int(base["simulation.nspecies"])
    raw = base | {f"species.lmax.sp{i}": str(CAP_CM) for i in range(n)}
    raw |= {f"movement.distribution.method.sp{IMMOBILE}": "none"}
    _child("thread_digests", raw, tmp_path / "d.json", "1,7")
    got = json.loads((tmp_path / "d.json").read_text())
    for seed in (1, 7):
        one, two, many = (got[f"{seed}:{threads}"] for threads in (1, 2, 14))
        assert one["records"] == two["records"] == many["records"] >= 2
        assert one["digest"] == two["digest"] == many["digest"], f"seed {seed}"
    assert got["1:1"]["digest"] != got["7:1"]["digest"]


# ------------------------------------------------ task 3.6: the depensation gate

GATE_OFF = {"reproduction.depensation.gate.enabled": "false"}


def _cod_west(records: list) -> np.ndarray:
    return np.array([float(np.asarray(r.biomass)[0]) for r in records])


def test_the_depensation_demo_runs_and_its_gate_reaches_the_output(tmp_path):
    """The shipped ``baltic_depensation`` demo, refused until its gate was ported. The gate is
    on cod_west with an S50 of 90 000 t, far above the stock, so it all but shuts recruitment
    as soon as cod spawns for real. Until then cod is seeded, and a seeded species is spared:
    the first year is the same numbers with the gate on and off. By the fifth year the gated
    stock is a fraction of the other."""
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    raw = _raw("baltic_depensation", tmp_path, 5, minimal=False)
    assert raw["reproduction.depensation.gate.s50.sp0"] == "90000"
    assert csharp_engine_block_reason(raw) is None
    on = _run_records("csharp", raw, tmp_path / "on.pkl")
    off = _run_records("csharp", raw | GATE_OFF, tmp_path / "off.pkl")
    assert len(on) == len(off) == 5
    assert np.asarray(on[0].biomass).tobytes() == np.asarray(off[0].biomass).tobytes()
    assert 0 < _cod_west(on)[-1] < 0.5 * _cod_west(off)[-1]
    assert _cod_west(off)[-1] > _cod_west(off)[1]  # ungated, the stock is still building up


def test_the_depensation_demo_is_the_same_at_1_2_and_14_threads(tmp_path):
    """Five years, so that the gate acts within the run."""
    raw = _raw("baltic_depensation", tmp_path, 5, minimal=False)
    _child("thread_digests", raw, tmp_path / "d.json", "1,7")
    got = json.loads((tmp_path / "d.json").read_text())
    for seed in (1, 7):
        one, two, many = (got[f"{seed}:{threads}"] for threads in (1, 2, 14))
        assert one["records"] == two["records"] == many["records"] == 5
        assert one["digest"] == two["digest"] == many["digest"], f"seed {seed}"
    assert got["1:1"]["digest"] != got["7:1"]["digest"]


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
