"""Plan task 2.1: background (forced) species, feature background_species.

With it, Baltic needs no feature the C# engine lacks, so it runs (with minimal outputs).

- The bundle's tables, and the schools ``Background.Inject`` builds from them, equal
  ``BackgroundState.get_schools`` field for field, on Baltic's NetCDF forcing and on a synthetic
  variant with a uniform biomass and a proportion time series shorter than the run (clamped).
- The background totals equal ``_collect_background_outputs`` on the state a real Baltic run
  holds after mortality, taken through a store regrouped by cell: the sums run in the Python
  state's order, and a cell-ordered sum would differ.
- A Baltic run through ``CSharpEngine``: the background species' biomass and abundance columns
  equal the Python engine's at every record (nothing preys on a seal or a cormorant, so they do
  not depend on any draw), frames have Python's shape, and results are byte-identical at 1, 2
  and 14 threads with the death-accounting check on.

The CLR-hosting checks run in child processes and need the built engine.
"""

from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.background import BackgroundState
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import csharp_engine_block_reason, required_features
from tests.test_csharp_engine_e2e import _child

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ("out.biomass", "out.abundance", "out.mortality", "out.yield")
FIELDS = (
    "species_id", "age_dt", "first_feeding_age_dt", "cell_x", "cell_y",
    "length", "weight", "abundance", "trophic_level",
)  # fmt: skip


class _Stop(Exception):
    pass


def _need_engine() -> None:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")


def _baltic(d: Path, nyear: int = 2, extra: dict | None = None) -> dict:
    raw = dict(OsmoseConfigReader().read(Path(osmose_demo("baltic", d)["config_file"])))
    raw |= {"output.minimal.enabled": "true", "simulation.time.nyear": str(nyear)} | (extra or {})
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    return {"raw": raw, "config": config, "grid": grid}


def _probe(method: str, *args) -> Bundle:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        f"f = bridge.load().Diagnostics.{method}; "
        "a = [int(x) if x.lstrip('-').isdigit() else x for x in sys.argv[2:]]; f(*a)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), *map(str, args)],
        capture_output=True, text=True, timeout=1200, cwd=ROOT,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-4000:]
    return Bundle.load(args[-1])


def _assert_schools_equal(case: dict, d: Path, steps) -> None:
    config, grid = case["config"], case["grid"]
    state = BackgroundState(config=config.raw_config, grid=grid, engine_config=config)
    model = d / "model.bin"
    build_bundle(config, case["raw"], grid, seed=11).save(model)
    for step in steps:
        want = state.get_schools(step)
        got = _probe("BackgroundProbe.Schools", model, step, d / f"s{step}.bin")
        assert got.get("bkg.per_step") == len(want) > 0
        for name in FIELDS:
            a = np.asarray(got.get(f"bkg.{name}"))
            b = np.asarray(getattr(want, name)).astype(a.dtype)
            assert a.tobytes() == b.tobytes(), (step, name, np.flatnonzero(a != b)[:5])
        assert (np.asarray(got.get("bkg.flags")) == 4).all()  # IsBackground, nothing else
        assert np.asarray(got.get("bkg.ordinal")).tolist() == list(range(len(want)))
        assert want.is_background.all() and (want.first_feeding_age_dt == -1).all()


# ------------------------------------------------------------------ the bundle (no CLR)


def test_baltic_needs_no_feature_the_engine_lacks(tmp_path):
    case = _baltic(tmp_path)
    features = required_features(case["config"], case["raw"])
    assert "background_species" in features
    assert csharp_engine_block_reason(case["raw"]) is None  # Baltic runs, with minimal outputs
    b = build_bundle(case["config"], case["raw"], case["grid"], seed=1)
    n_ocean = int(case["grid"].ocean_mask.sum())
    assert b.get("background.class.offsets").tolist() == [0, 2, 4]  # seal and cormorant, 2 classes
    assert b.get("background.ocean_cells").shape == (n_ocean,)
    assert b.get("background.proportion_by_step").shape == (48, 4)
    assert b.get("background.base").shape[1] == n_ocean
    frames = b.get("background.frame_by_step")
    assert frames.shape == (48, 2) and len(np.unique(frames[:, 0])) > 1  # the forcing moves in time


def test_a_config_without_background_species_carries_no_background_entries(tmp_path):
    raw = dict(OsmoseConfigReader().read(Path(osmose_demo("eec_full", tmp_path)["config_file"])))
    raw |= {"output.minimal.enabled": "true", "simulation.time.nyear": "1"}
    config, grid, *_ = PythonEngine()._prepare_run(raw, 0)
    b = build_bundle(config, raw, grid, seed=1)
    assert "background_species" not in required_features(config, raw)
    assert not [n for n in b.names() if n.startswith("background.")]


# ------------------------------------------------------------------ the schools (tier E)


def test_injected_schools_equal_get_schools_on_baltic(tmp_path):
    _need_engine()
    _assert_schools_equal(_baltic(tmp_path), tmp_path, (0, 1, 13, 23, 24, 47))


def test_injected_schools_follow_a_forcing_that_varies_in_time_and_space(tmp_path):
    """Baltic's own predator file holds the same field in all 24 frames, so on it a wrong
    step-to-frame map, or a wrong ocean-cell order, would go unseen. Here the seal (sp15) is
    forced by a random field of 12 frames a year on a grid of another shape (so it is regridded):
    step 1 still reads frame 0, step 2 reads frame 1, and every cell differs."""
    import xarray as xr

    _need_engine()
    nc = tmp_path / "seal.nc"
    data = np.random.default_rng(7).uniform(0.5, 9.0, (12, 20, 25))
    ds = xr.Dataset({"GreySeal": (["time", "latitude", "longitude"], data)})
    ds.to_netcdf(nc)
    ds.close()
    case = _baltic(
        tmp_path,
        extra={"species.file.sp15": str(nc), "species.biomass.nsteps.year.sp15": "12"},
    )
    b = build_bundle(case["config"], case["raw"], case["grid"], seed=1)
    seal_frames = b.get("background.frame_by_step")[:, 0]
    assert seal_frames[:5].tolist() == [0, 0, 1, 1, 2] and seal_frames[24:27].tolist() == [0, 0, 1]
    assert len(np.unique(seal_frames)) == 12
    _assert_schools_equal(case, tmp_path, (0, 1, 2, 13, 23, 24, 47))
    state = BackgroundState(
        config=case["config"].raw_config, grid=case["grid"], engine_config=case["config"]
    )
    first, later = state.get_schools(0), state.get_schools(13)
    seal = first.species_id == case["config"].n_species
    assert first.abundance[seal].tobytes() != later.abundance[seal].tobytes()  # frames differ
    assert len(np.unique(first.abundance[seal])) > 100  # and so do cells


def test_injected_schools_equal_get_schools_with_uniform_biomass_and_a_proportion_series(tmp_path):
    """What Baltic does not use: a uniform total (the seal, sp15) and a proportion time series
    of three rows on a 48-step run (the cormorant, sp16), which is clamped at its last row."""
    _need_engine()
    series = tmp_path / "proportions.csv"
    series.write_text("small;large\n0.9;0.1\n0.5;0.5\n0.2;0.8\n")
    case = _baltic(
        tmp_path,
        extra={
            "species.biomass.total.sp15": "4500.0",
            "species.size.proportion.file.sp16": str(series),
        },
    )
    b = build_bundle(case["config"], case["raw"], case["grid"], seed=1)
    prop = b.get("background.proportion_by_step")
    assert prop[:4, 2].tolist() == [0.9, 0.5, 0.2, 0.2] and prop[47, 3] == 0.8  # clamped
    assert len(np.unique(b.get("background.frame_by_step")[:, 0])) == 1  # uniform: one frame
    _assert_schools_equal(case, tmp_path, (0, 1, 2, 3, 30))


# ------------------------------------------------------------------ the totals (tier E)


def _capture_after_mortality(case: dict, step: int) -> dict:
    """The state ``_collect_background_outputs`` receives at ``step`` of a real run (focal
    schools, then that step's background schools, after mortality), and what it returns."""
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(case["raw"], 11)
    real = sim._collect_background_outputs
    got: dict = {}
    calls = {"n": 0}

    def spy(state, config_, n_focal):
        out = real(state, config_, n_focal)
        if calls["n"] == step:
            got.update(state=copy.deepcopy(state), n_focal=n_focal, out=out)
            raise _Stop
        calls["n"] += 1
        return out

    sim._collect_background_outputs = spy
    try:
        sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    except _Stop:
        pass
    finally:
        sim._collect_background_outputs = real
    assert got, "the run ended before the step"
    return got


def test_background_totals_equal_the_engines_through_a_regrouped_store(tmp_path):
    _need_engine()
    case = _baltic(tmp_path)
    got = _capture_after_mortality(case, 30)
    st, n_focal = got["state"], got["n_focal"]
    assert st.is_background[n_focal:].all() and not st.is_background[:n_focal].any()
    state = Bundle(1)
    for name, dt in [
        ("species_id", np.int32), ("abundance", np.float64), ("weight", np.float64),
        ("cell_x", np.int32), ("cell_y", np.int32), ("is_background", np.uint8),
    ]:  # fmt: skip
        state.set_array(f"state.{name}", np.ascontiguousarray(getattr(st, name), dtype=dt))
    state.save(tmp_path / "state.bin")
    model = tmp_path / "model.bin"
    build_bundle(case["config"], case["raw"], case["grid"], seed=11).save(model)
    res = _probe("BackgroundProbe.Totals", model, tmp_path / "state.bin", tmp_path / "t.bin")
    want_biomass, want_abundance = got["out"]
    assert np.asarray(res.get("bkg.biomass")).tobytes() == want_biomass.tobytes()
    assert np.asarray(res.get("bkg.abundance")).tobytes() == want_abundance.tobytes()
    assert (want_biomass[case["config"].n_species :] > 0).all()
    # The order matters: the same schools added up cell by cell (the store's order after the
    # regroup) round differently for at least one species.
    n = case["config"].n_species + case["config"].n_background
    by_cell = np.zeros(n)
    bkg = np.arange(n_focal, len(st))
    order = bkg[np.argsort(st.cell_y[bkg] * case["grid"].nx + st.cell_x[bkg], kind="stable")]
    np.add.at(by_cell, st.species_id[order], st.biomass[order])
    assert by_cell.tobytes() != want_biomass.tobytes()
    # The removal: every background school is gone, every live focal school is left, and the
    # focal schools keep their relative order within a cell.
    alive = np.flatnonzero(st.abundance[:n_focal] != 0)
    left = np.asarray(res.get("bkg.left_ids"))[: int(res.get("bkg.left_count"))]
    assert sorted(left.tolist()) == alive.tolist()
    assert int(res.get("bkg.removed")) == int((st.abundance[n_focal:] != 0).sum())
    cells = st.cell_y[left] * case["grid"].nx + st.cell_x[left]
    located = cells[st.cell_x[left] >= 0]
    assert (np.diff(located) >= 0).all()  # still grouped by cell
    same_cell = np.diff(cells) == 0
    assert (np.diff(left)[same_cell] > 0).all()  # and stable inside a cell


# ------------------------------------------------------------------ Baltic end to end


def test_a_baltic_run_matches_the_python_engine_on_its_background_species(tmp_path):
    _need_engine()
    case = _baltic(tmp_path, nyear=1, extra={"output.recordfrequency.ndt": "1"})
    _child("memory", case["raw"], tmp_path / "cs.pkl", "11")
    # Trusted input: the pickle was just written by this test's own child process into tmp_path.
    cs = pickle.loads((tmp_path / "cs.pkl").read_bytes())
    py = PythonEngine().run_in_memory(case["raw"], seed=11)
    names = list(case["config"].all_species_names)
    background = names[case["config"].n_species :]
    assert background == ["GreySeal", "Cormorant"]
    for frame in ("biomass", "abundance"):
        a, b = cs[frame], getattr(py, frame)()
        assert a.shape == b.shape and list(a.columns) == list(b.columns), frame
        assert len(a) == 24
        for sp in background:  # no draw reaches them: equal at every record
            assert a[sp].to_numpy().tobytes() == b[sp].to_numpy().tobytes(), (frame, sp)
            assert (a[sp] > 0).all()
        focal = a[names[: case["config"].n_species]].to_numpy()
        assert np.isfinite(focal).all() and (focal >= 0).all() and focal[-1].sum() > 0
    assert cs["mortality"].select_dtypes("number").to_numpy().sum() > 0


def test_baltic_runs_are_byte_identical_across_thread_counts_with_checks_on(tmp_path):
    _need_engine()
    case = _baltic(tmp_path)
    runs = {}
    for threads in (1, 2, 14):
        b = build_bundle(case["config"], case["raw"], case["grid"], seed=5, threads=threads, checks=True)  # fmt: skip
        b.save(tmp_path / f"m{threads}.bin")
        out = _probe(
            "PipelineProbe.Run", tmp_path / f"m{threads}.bin", tmp_path / f"o{threads}.bin"
        )
        status, _, _, peak = out.get("out.summary")
        assert status == 0 and peak > 1000
        runs[threads] = b"".join(np.ascontiguousarray(out.get(n)).tobytes() for n in RESULTS)
    assert runs[1] == runs[2] == runs[14]


def test_injection_and_removal_allocate_nothing_at_one_thread(tmp_path):
    """Injection appends 9,856 schools every step and removal takes them out again; neither may
    allocate. Measured over 24 steps of model year 12. Baltic's population is still growing
    then, and whenever a cell sets a new record occupancy its mortality scratch grows once
    (seen as occasional bursts in other windows, e.g. steps 340 to 364): that is not this
    feature. The seed and thread count are fixed, so the window is reproducible; a per-step
    allocation would show in every window."""
    _need_engine()
    case = _baltic(tmp_path, nyear=14)
    build_bundle(case["config"], case["raw"], case["grid"], seed=3, threads=1).save(
        tmp_path / "m.bin"
    )
    allocated, cap_before, cap_after, schools = _probe(
        "PipelineProbe.Allocations", tmp_path / "m.bin", 300, 24, tmp_path / "a.bin"
    ).get("out.allocations")
    assert schools > 20_000
    assert cap_before == cap_after, "the store grew inside the window: choose a later one"
    assert allocated == 0, f"{allocated} bytes allocated over 24 steps"
