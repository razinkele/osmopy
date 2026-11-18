"""Plan task 1.7: the C# life cycle against the Python engine, stage by stage.

Each stage (floor, growth, aging, reproduction) is captured at its own entry and exit in a real
run (spies on ``simulate``'s ``_floor_sub_one_fish``, ``_growth``, ``_aging_mortality`` and
``_reproduction``), and ``LifeCycleProbe`` runs the C# stage on Python's own input, so a
last-ulp difference in growth cannot flip a maturity test downstream.

Tiers (spec 6.8): the floor, aging and reproduction are exact here (the probe keeps Python's
school order, so the spawning-stock sum is added in the same order; no curve, no ``exp``).
Growth involves ``exp`` and ``pow``: it is compared at 1e-12 against the stock code, and its
lengths bit for bit against the same code with ``growth.np.exp`` replaced by ``math.exp``.

Reproduction runs on EEC and Bay of Biscay; Baltic's stock-recruitment curve and recruitment
gates are phase 2. Synthetic inputs (production functions as the reference) cover what a capture
may not show: fewer eggs than schools, and the seeding window's closed side.

The test needs the built engine. It skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import osmose.engine.processes.growth as growth_mod
import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import MortalityCause
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle

ROOT = Path(__file__).resolve().parents[1]
CASES = {"eec_full": 30, "bay_of_biscay": 9, "baltic": 129}
STAGES = {"floor": "_floor_sub_one_fish", "growth": "_growth", "aging": "_aging_mortality", "reproduction": "_reproduction"}  # fmt: skip
TIER_L = 1e-12


class _Stop(Exception):
    pass


def _capture(raw: dict, step: int) -> dict:
    """(input, output) of each life-cycle stage at ``step``, from a real run."""
    raw = dict(raw)
    raw["output.minimal.enabled"] = "true"
    raw["simulation.time.nyear"] = str(step // 24 + 1)
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    got: dict = {"stages": {}}
    real = {stage: getattr(sim, name) for stage, name in STAGES.items()}
    current = {"step": -1}
    real_mortality = sim._mortality

    def mortality_spy(state, resources, config_, rng_, grid_, step=0, species_rngs=None, ctx=None):
        current["step"] = step
        return real_mortality(
            state, resources, config_, rng_, grid_, step=step, species_rngs=species_rngs, ctx=ctx
        )

    def make_spy(stage):
        def spy(state, *args, **kwargs):
            out = real[stage](state, *args, **kwargs)
            if current["step"] == step:
                got["stages"][stage] = (copy.deepcopy(state), copy.deepcopy(out))
                if stage == "reproduction":
                    raise _Stop
            return out

        return spy

    sim._mortality = mortality_spy
    for stage, name in STAGES.items():
        setattr(sim, name, make_spy(stage))
    try:
        sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    except _Stop:
        pass
    finally:
        sim._mortality = real_mortality
        for stage, name in STAGES.items():
            setattr(sim, name, real[stage])
    assert set(got["stages"]) == set(STAGES), set(got["stages"])
    got.update(config=config, grid=grid, raw=raw, step=step)
    return got


def _write_state(st, path: Path) -> None:
    b = Bundle(1)
    for name, dt in [
        ("species_id", np.int32), ("age_dt", np.int32), ("length", np.float64), ("weight", np.float64),
        ("abundance", np.float64), ("trophic_level", np.float64), ("is_out", np.uint8),
        ("is_egg", np.uint8), ("from_seeding", np.uint8), ("first_feeding_age_dt", np.int32),
        ("pred_success_rate", np.float64), ("cell_x", np.int32), ("cell_y", np.int32),
        ("starvation_rate", np.float64),
    ]:  # fmt: skip
        b.set_array(f"state.{name}", np.ascontiguousarray(getattr(st, name), dtype=dt))
    b.set_array(
        "state.dead_aging",
        np.ascontiguousarray(st.n_dead[:, MortalityCause.AGING], dtype=np.float64),
    )
    b.save(path)


def _run_probe(model: Path, state: Path, stage: str, step: int, out: Path) -> Bundle:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.LifeCycleProbe.Run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])"
    )
    args = [
        sys.executable,
        "-c",
        code,
        str(ROOT),
        str(model),
        str(state),
        stage,
        str(step),
        str(out),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return Bundle.load(out)


def _stage(got: dict, stage: str, tmp: Path, state=None) -> Bundle:
    st = got["stages"][stage][0] if state is None else state
    _write_state(st, tmp / f"{stage}_in.bin")
    return _run_probe(
        got["dir"] / "model.bin",
        tmp / f"{stage}_in.bin",
        stage,
        got["step"],
        tmp / f"{stage}_out.bin",
    )


FIELDS = {
    "species_id": "species_id", "age_dt": "age_dt", "first_feeding_age_dt": "first_feeding_age_dt",
    "cell_x": "cell_x", "cell_y": "cell_y", "length": "length", "weight": "weight",
    "abundance": "abundance", "trophic_level": "trophic_level", "starvation_rate": "starvation_rate",
    "is_egg": "is_egg", "from_seeding": "from_seeding", "is_out": "is_out",
}  # fmt: skip


def _assert_exact(res: Bundle, want, names) -> None:
    for name in names:
        got = np.asarray(res.get(f"life.{name}"))
        exp = np.asarray(getattr(want, FIELDS[name])).astype(got.dtype)
        assert got.shape == exp.shape, name
        assert got.tobytes() == exp.tobytes(), (name, np.flatnonzero(got != exp)[:5])


@pytest.fixture(scope="module")
def cycled(tmp_path_factory) -> dict:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    out = {}
    for demo, step in CASES.items():
        d = tmp_path_factory.mktemp(demo)
        raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, d)["config_file"])))
        got = _capture(raw, step)
        build_bundle(got["config"], got["raw"], got["grid"], seed=11).save(d / "model.bin")
        got["dir"] = d
        out[demo] = got
    return out


@pytest.mark.parametrize("demo", sorted(CASES))
def test_floor_equals_the_engine(cycled, demo, tmp_path):
    got = cycled[demo]
    st_in, st_out = got["stages"]["floor"]
    sub_one = (st_in.abundance > 0) & (st_in.abundance < 1)
    assert sub_one.any() and (st_in.abundance >= 1).any()  # both sides of the floor present
    _assert_exact(_stage(got, "floor", tmp_path), st_out, ["abundance"])


@pytest.mark.parametrize("demo", sorted(CASES))
def test_growth_matches_the_engine_at_tier_l(cycled, demo, tmp_path, record_property):
    got = cycled[demo]
    st_in, st_out = got["stages"]["growth"]
    res = _stage(got, "growth", tmp_path)
    worst = {}
    for name in ("length", "weight"):
        a, b = np.asarray(res.get(f"life.{name}")), getattr(st_out, name)
        worst[name] = float(np.max(np.abs(a - b) / np.abs(b)))
    record_property("worst_relative_error", worst)
    assert all(v <= TIER_L for v in worst.values()), worst
    grew = st_out.length - st_in.length
    sr, csr = st_in.pred_success_rate, got["config"].critical_success_rate[st_in.species_id]
    assert (grew > 0).any() and (sr < csr).any() and (sr >= csr).any() and (st_in.age_dt == 0).any()


class _LibmNumpy:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def exp(x):
        a = np.asarray(x, dtype=np.float64)
        return np.fromiter(
            (math.exp(v) for v in a.ravel()), dtype=np.float64, count=a.size
        ).reshape(a.shape)


@pytest.mark.parametrize("demo", sorted(CASES))
def test_growth_lengths_are_exact_against_libm_exp(cycled, demo, tmp_path):
    """The same growth with ``growth.np.exp`` replaced by the C library's: lengths bit for bit."""
    got = cycled[demo]
    st_in, st_out = got["stages"]["growth"]
    stock = growth_mod.growth(copy.deepcopy(st_in), got["config"], np.random.default_rng(0))
    assert stock.length.tobytes() == st_out.length.tobytes()  # the capture is reproducible
    old = growth_mod.np
    growth_mod.np = _LibmNumpy()
    try:
        libm = growth_mod.growth(copy.deepcopy(st_in), got["config"], np.random.default_rng(0))
    finally:
        growth_mod.np = old
    res = _stage(got, "growth", tmp_path)
    assert np.asarray(res.get("life.length")).tobytes() == libm.length.tobytes()


@pytest.mark.parametrize("demo", sorted(CASES))
def test_aging_equals_the_engine(cycled, demo, tmp_path):
    got = cycled[demo]
    _, st_out = got["stages"]["aging"]
    res = _stage(got, "aging", tmp_path)
    _assert_exact(res, st_out, ["abundance"])
    want = st_out.n_dead[:, MortalityCause.AGING]
    assert np.asarray(res.get("life.dead_aging")).tobytes() == want.tobytes()


def test_aging_threshold_on_both_sides(cycled, tmp_path):
    """No captured step need contain a school at its lifespan. Put schools at ``lifespan - 1``
    (dies) and ``lifespan - 2`` (lives), production ``aging_mortality`` as the reference."""
    from osmose.engine.processes.natural import aging_mortality

    got = cycled["eec_full"]
    st = copy.deepcopy(got["stages"]["aging"][0])
    lifespan = got["config"].lifespan_dt[st.species_id]
    st.age_dt[0::2] = lifespan[0::2] - 1
    st.age_dt[1::2] = lifespan[1::2] - 2
    want = aging_mortality(copy.deepcopy(st), got["config"])
    res = _stage(got, "aging", tmp_path, state=st)
    _assert_exact(res, want, ["abundance"])
    assert (
        np.asarray(res.get("life.dead_aging")).tobytes()
        == want.n_dead[:, MortalityCause.AGING].tobytes()
    )
    assert (want.abundance[0::2] == 0).all() and (want.abundance[1::2] == st.abundance[1::2]).all()


REPRO = ["eec_full", "bay_of_biscay"]  # Baltic: SR curve + gates are phase 2
ALL_FIELDS = list(FIELDS)


@pytest.mark.parametrize("demo", REPRO)
def test_reproduction_equals_the_engine(cycled, demo, tmp_path):
    got = cycled[demo]
    st_in, st_out = got["stages"]["reproduction"]
    res = _stage(got, "reproduction", tmp_path)
    _assert_exact(res, st_out, ALL_FIELDS)
    n_new = len(st_out) - len(st_in)
    assert n_new > 0  # eggs were laid
    seeded = np.asarray(res.get("life.seeded")).astype(bool)
    if demo == "bay_of_biscay":
        assert seeded.any()  # no BoB fish matures: the seeding rule makes the eggs
    else:
        assert (np.asarray(res.get("life.ssb"))[~seeded] > 0).any()  # a real spawning stock


def test_reproduction_with_fewer_eggs_than_schools(cycled, tmp_path):
    """A spawning stock so small that one species lays fewer eggs than it has schools: one school
    then holds them all. Synthetic input, production ``reproduction`` as the reference."""
    got = cycled["eec_full"]
    config, step = got["config"], got["step"]
    st_in = got["stages"]["reproduction"][0]
    res = _stage(got, "reproduction", tmp_path)
    eggs, seeded = np.asarray(res.get("life.eggs")), np.asarray(res.get("life.seeded")).astype(bool)
    sp = int(np.flatnonzero((eggs > config.n_schools[: len(eggs)]) & ~seeded)[0])
    small = copy.deepcopy(st_in)
    small.abundance[small.species_id == sp] *= 0.5 * config.n_schools[sp] / eggs[sp]
    want = reproduction(copy.deepcopy(small), config, step, np.random.default_rng(0))
    _assert_exact(_stage(got, "reproduction", tmp_path, state=small), want, ALL_FIELDS)
    assert (want.species_id[len(small) :] == sp).sum() == 1  # exactly one egg school for it


@pytest.mark.parametrize("open_window", [True, False])
def test_seeding_rule_on_both_sides_of_the_window(cycled, tmp_path, open_window):
    """An empty spawning stock (SSB exactly 0) seeds while ``step < seeding_max_step`` and not at
    ``step == seeding_max_step``. The window is moved onto the captured step by rebuilding the
    bundle from a config copy; production ``reproduction`` is the reference."""
    got = cycled["eec_full"]
    step = got["step"]
    cfg = copy.copy(got["config"])
    cfg.seeding_max_step = np.full_like(
        got["config"].seeding_max_step, step + 1 if open_window else step
    )
    model = tmp_path / "model.bin"
    build_bundle(cfg, got["raw"], got["grid"], seed=11).save(model)
    empty = copy.deepcopy(got["stages"]["reproduction"][0])
    empty.abundance[:] = 0.0
    want = reproduction(copy.deepcopy(empty), cfg, step, np.random.default_rng(0))
    _write_state(empty, tmp_path / "in.bin")
    res = _run_probe(model, tmp_path / "in.bin", "reproduction", step, tmp_path / "out.bin")
    _assert_exact(res, want, ALL_FIELDS)
    seeded = np.asarray(res.get("life.seeded")).astype(bool)
    assert seeded.all() if open_window else not seeded.any()
    assert (len(want) > len(empty)) == open_window


def test_growth_cap_at_lmax(cycled, tmp_path):
    """Every shipped config leaves lmax at +inf (the cap is the lmax_cap feature). Cap each
    species at its median length through a config copy: lengths must equal production growth's
    (with libm exp) bit for bit, and the cap must bind."""
    got = cycled["eec_full"]
    st_in = got["stages"]["growth"][0]
    cfg = copy.copy(got["config"])
    cfg.lmax = np.array(got["config"].lmax, dtype=np.float64)
    assert np.isinf(cfg.lmax).all()
    for sp in np.unique(st_in.species_id):
        cfg.lmax[sp] = float(np.median(st_in.length[st_in.species_id == sp]))
    model = tmp_path / "model.bin"
    build_bundle(cfg, got["raw"], got["grid"], seed=11).save(model)
    old = growth_mod.np
    growth_mod.np = _LibmNumpy()
    try:
        want = growth_mod.growth(copy.deepcopy(st_in), cfg, np.random.default_rng(0))
    finally:
        growth_mod.np = old
    _write_state(st_in, tmp_path / "in.bin")
    res = _run_probe(model, tmp_path / "in.bin", "growth", got["step"], tmp_path / "out.bin")
    assert np.asarray(res.get("life.length")).tobytes() == want.length.tobytes()
    assert (want.length == cfg.lmax[st_in.species_id]).sum() > 10  # the cap binds


def test_maturity_age_binds(cycled, tmp_path):
    """If a config matures by size alone the age condition never binds. Raise each spawning
    species' maturity age to the median age of its mature schools: production ``reproduction``
    is the reference, and the spawning stock must shrink."""
    got = cycled["eec_full"]
    config, step = got["config"], got["step"]
    st_in = got["stages"]["reproduction"][0]
    cfg = copy.copy(config)
    cfg.maturity_age_dt = np.array(config.maturity_age_dt).copy()
    sp_ids = st_in.species_id
    mature = (
        (st_in.length >= config.maturity_size[sp_ids])
        & (st_in.age_dt >= config.maturity_age_dt[sp_ids])
        & (st_in.abundance > 0)
    )
    for sp in np.unique(sp_ids[mature]):
        cfg.maturity_age_dt[sp] = int(np.median(st_in.age_dt[mature & (sp_ids == sp)])) + 1
    model = tmp_path / "model.bin"
    build_bundle(cfg, got["raw"], got["grid"], seed=11).save(model)
    want = reproduction(copy.deepcopy(st_in), cfg, step, np.random.default_rng(0))
    _write_state(st_in, tmp_path / "in.bin")
    res = _run_probe(model, tmp_path / "in.bin", "reproduction", step, tmp_path / "out.bin")
    _assert_exact(res, want, ALL_FIELDS)
    base = _stage(got, "reproduction", tmp_path)
    shrunk = np.asarray(res.get("life.ssb")) < np.asarray(base.get("life.ssb"))
    assert shrunk.any()
