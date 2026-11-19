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
import osmose.engine.processes.reproduction as reproduction_mod
import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import MortalityCause
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import required_features

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


REPRO = sorted(CASES)  # Baltic too since task 2.3: its Shepherd curves and its RV gate
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


# ------------------------------------------------------------------ task 2.3: recruitment


def _repro_case(got: dict, tmp: Path, state=None, **changes):
    """Production ``reproduction`` and the C# stage on a copy of the captured config with
    ``changes`` set (the bundle is rebuilt from it). Returns (config, want, res). Each call
    writes to its own directory: a loaded bundle is a memory map of its file, so a second case
    written to the same names would silently replace the first one's results."""
    cfg = copy.copy(got["config"])
    for name, value in changes.items():
        setattr(cfg, name, value)
    st = got["stages"]["reproduction"][0] if state is None else state
    want = reproduction(copy.deepcopy(st), cfg, got["step"], np.random.default_rng(0))
    d = tmp / f"case{sum(1 for _ in tmp.glob('case*'))}"
    d.mkdir()
    build_bundle(cfg, got["raw"], got["grid"], seed=11).save(d / "model.bin")
    _write_state(st, d / "in.bin")
    res = _run_probe(d / "model.bin", d / "in.bin", "reproduction", got["step"], d / "out.bin")
    return cfg, want, res


def _eggs(res: Bundle) -> np.ndarray:
    return np.array(res.get("life.eggs"))  # a copy, not a view of the mapped file


def test_baltic_reproduction_needs_its_curves(cycled):
    """The exact Baltic comparison above is only worth something if the Shepherd curves change
    the answer: without them production itself lays different eggs. (The RV gate does NOT change
    it at the captured step: Baltic's factor is 1.0 for its first twelve model years. It has its
    own test below.)"""
    got = cycled["baltic"]
    cfg, step = got["config"], got["step"]
    st_in, st_out = got["stages"]["reproduction"]
    assert set(cfg.recruitment_type[: cfg.n_species]) == {"shepherd"}
    other = copy.copy(cfg)
    other.recruitment_type = ["none"] * len(cfg.recruitment_type)
    out = reproduction(copy.deepcopy(st_in), other, step, np.random.default_rng(0))
    assert out.abundance.tobytes() != st_out.abundance.tobytes()


@pytest.mark.parametrize("year", [16, 49])
def test_baltic_rv_gate_bites_in_low_volume_years_and_clamps_past_its_series(
    cycled, year, tmp_path
):
    """Baltic's own reproductive-volume series (47 years, raw_cap): 0.6 in model year 16, and
    past its end (year 49) the last value, 0.32, by clamping. ``reproduction`` is a function of
    (state, config, step), so the state captured at step 129 serves at any step: here mid-year,
    in eastern cod's spawning season, on a 50-year config."""
    from osmose.engine.processes.recruitment_gate import rv_gate_factor

    got = cycled["baltic"]
    raw = got["raw"] | {"simulation.time.nyear": "50"}
    cfg, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    step = year * cfg.n_dt_per_year + 12
    cod_east = int(np.flatnonzero(cfg.rv_gate_enabled)[0])
    factor = float(rv_gate_factor(cfg, step)[cod_east])
    assert factor == pytest.approx({16: 0.6, 49: 0.32}[year], abs=1e-3)
    assert factor == cfg.rv_gate_factor_by_index[min(year, len(cfg.rv_gate_factor_by_index) - 1)]
    st_in = got["stages"]["reproduction"][0]
    want = reproduction(copy.deepcopy(st_in), cfg, step, np.random.default_rng(0))
    model = tmp_path / "model.bin"
    build_bundle(cfg, raw, grid, seed=11).save(model)
    _write_state(st_in, tmp_path / "in.bin")
    res = _run_probe(model, tmp_path / "in.bin", "reproduction", step, tmp_path / "out.bin")
    _assert_exact(res, want, ALL_FIELDS)
    ungated = copy.copy(cfg)
    ungated.rv_gate_factor_by_index = None
    free = reproduction(copy.deepcopy(st_in), ungated, step, np.random.default_rng(0))
    assert free.abundance.tobytes() != want.abundance.tobytes()  # the gate changes the answer
    assert (
        not np.asarray(res.get("life.seeded")).astype(bool)[cod_east] and _eggs(res)[cod_east] > 0
    )


@pytest.mark.parametrize("curve", ["beverton_holt", "ricker", "hockey_stick", "shepherd"])
def test_each_recruitment_curve_equals_the_engine(cycled, curve, tmp_path):
    """Every species gets ``curve``, with its characteristic SSB below the stock for the even
    species and above it for the odd ones (both sides of the hockey stick's breakpoint), and a
    non-integer Shepherd exponent. Ricker's ``exp`` is NumPy's in production and the C library's
    in C#, so the exact reference swaps it (as for growth); production is within tier L."""
    got = cycled["eec_full"]
    n = got["config"].n_species
    base = _stage(got, "reproduction", tmp_path)
    ssb = np.asarray(base.get("life.ssb"))  # after seeding: what the curve sees
    half = np.where(np.arange(n) % 2 == 0, 0.37, 2.3) * ssb
    changes = {
        "recruitment_type": [curve] * n,
        "recruitment_ssb_half": half,
        "shepherd_beta": np.full(n, 1.63),
    }
    cfg, stock, res = _repro_case(got, tmp_path, **changes)
    assert required_features(cfg, got["raw"]) == ["core", "stock_recruitment"]
    old = reproduction_mod.np
    reproduction_mod.np = _LibmNumpy()
    try:
        st_in = got["stages"]["reproduction"][0]
        libm = reproduction(copy.deepcopy(st_in), cfg, got["step"], np.random.default_rng(0))
    finally:
        reproduction_mod.np = old
    _assert_exact(res, libm, ALL_FIELDS)
    assert np.allclose(stock.abundance, libm.abundance, rtol=TIER_L, atol=0.0)
    if curve != "ricker":
        assert stock.abundance.tobytes() == libm.abundance.tobytes()  # no exp: production itself
    eggs, linear = _eggs(res), _eggs(base)
    lays = linear > 0  # a species out of its spawning season lays nothing either way
    below = ssb <= half
    assert (lays & below).any() and (lays & ~below).any()
    assert (eggs[~lays] == 0).all()
    if curve == "hockey_stick":
        assert (eggs[below] == linear[below]).all()  # at or under the breakpoint: no correction
        assert (eggs[lays & ~below] < linear[lays & ~below]).all()
    else:
        assert (eggs[lays] < linear[lays]).all()  # every other curve bites at any positive stock


@pytest.mark.parametrize("mode", ["stock_recruitment", "linear"])
def test_a_seeded_species_takes_the_curve_or_the_linear_count_by_seeding_mode(
    cycled, mode, tmp_path
):
    """GitHub #143: with ``seeding_mode = linear`` a seeded species skips the curve (Java's
    proportional seeding); by default it goes through it. An emptied state seeds every species."""
    got = cycled["eec_full"]
    n, step = got["config"].n_species, got["step"]
    empty = copy.deepcopy(got["stages"]["reproduction"][0])
    empty.abundance[:] = 0.0
    changes = {
        "seeding_max_step": np.full_like(got["config"].seeding_max_step, step + 1),
        "recruitment_type": ["beverton_holt"] * n,
        "recruitment_ssb_half": np.asarray(got["config"].seeding_biomass)[:n] * 0.5,
        "seeding_mode": mode,
    }
    cfg, want, res = _repro_case(got, tmp_path, state=empty, **changes)
    assert ("seeding_linear" in required_features(cfg, got["raw"])) == (mode == "linear")
    _assert_exact(res, want, ALL_FIELDS)
    assert np.asarray(res.get("life.seeded")).astype(bool).all()
    _, _, plain = _repro_case(
        got, tmp_path, state=empty, seeding_max_step=changes["seeding_max_step"]
    )
    lays = _eggs(plain) > 0
    assert lays.any()
    if mode == "linear":
        assert _eggs(res).tobytes() == _eggs(plain).tobytes()  # the curve is skipped
    else:
        # Beverton-Holt at ssb = 2 * ssb_half: a third of the linear count (to rounding).
        assert np.allclose(_eggs(res)[lays], _eggs(plain)[lays] / 3.0, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("series", [[0.5, 0.25], [0.4]])
def test_rv_gate_scales_unseeded_enabled_species_and_clamps_past_its_series(
    cycled, series, tmp_path
):
    """The captured step is in model year 1: a two-year series gives year 1's factor, a one-year
    series is clamped to its last value (never wrapped). The gate is on for one seeded species
    and one with a real stock; only the second is scaled. Production is the reference."""
    got = cycled["eec_full"]
    n, step = got["config"].n_species, got["step"]
    assert step // got["config"].n_dt_per_year == 1
    base = _stage(got, "reproduction", tmp_path)
    seeded = np.asarray(base.get("life.seeded")).astype(bool)
    # Both must LAY EGGS at this step: scaling a species that lays none changes nothing, and a
    # gate wrongly applied to a seeded species would go unseen (it did, in a mutation check).
    on_seeded = int(np.flatnonzero(seeded & (_eggs(base) > 0))[0])
    on_real = int(np.flatnonzero(~seeded & (_eggs(base) > 0))[0])
    enabled = np.zeros(n, dtype=bool)
    enabled[[on_seeded, on_real]] = True
    cfg, want, res = _repro_case(
        got, tmp_path, rv_gate_enabled=enabled,
        rv_gate_factor_by_index=np.asarray(series, dtype=np.float64), rv_gate_offset=0,
    )  # fmt: skip
    assert required_features(cfg, got["raw"]) == ["core", "rv_gate"]
    _assert_exact(res, want, ALL_FIELDS)
    eggs, plain = _eggs(res), _eggs(base)
    assert eggs[on_real] == plain[on_real] * series[-1]
    untouched = np.arange(n) != on_real
    assert eggs[untouched].tobytes() == plain[untouched].tobytes()  # the seeded one included


# --------------------------------------------------------- task 3.6: the depensation gate


def _depensation_case(got: dict, tmp: Path, theta: float, ratio: float, **more):
    """The gate on one seeded species that lays eggs and on one with a real stock, whose S50 is
    ``ratio`` times that stock. Returns (config, want, res, base, real species, seeded species,
    the stocks, S50)."""
    n = got["config"].n_species
    base = _stage(got, "reproduction", tmp)
    seeded = np.asarray(base.get("life.seeded")).astype(bool)
    ssb = np.array(base.get("life.ssb"))  # after seeding: what the gate sees
    lays = _eggs(base) > 0
    on_seeded = int(np.flatnonzero(seeded & lays)[0])
    on_real = int(np.flatnonzero(~seeded & lays)[0])
    enabled = np.zeros(n, dtype=bool)
    enabled[[on_seeded, on_real]] = True
    s50 = np.ones(n, dtype=np.float64)
    s50[on_seeded] = 2.0 * ssb[on_seeded]  # would halve it, if a seeded species were gated
    s50[on_real] = ratio * ssb[on_real]
    cfg, want, res = _repro_case(
        got, tmp, depensation_gate_enabled=enabled, depensation_s50=s50,
        depensation_theta=np.full(n, theta, dtype=np.float64), **more,
    )  # fmt: skip
    return cfg, want, res, base, on_real, on_seeded, ssb, s50


@pytest.mark.parametrize("theta", [1.0, 2.5, 4.0, 300.0])
@pytest.mark.parametrize("ratio", [0.37, 2.3])
def test_depensation_gate_scales_an_unseeded_enabled_species_by_the_allee_term(
    cycled, theta, ratio, tmp_path
):
    """A = S^theta / (S50^theta + S^theta), on the stock of the step: above one half when the
    stock is above its S50, below it otherwise. A seeded species is spared, so that the
    bootstrap from an empty stock cannot be trapped. Production is the reference, bit for bit
    (the power is the C library's on both sides).

    theta = 300 is the case in which the two powers do not fit a double: both engines then
    write the same term as ratios, and must still agree to the last bit."""
    from osmose.engine.processes.depensation_gate import depensation_factor

    got = cycled["eec_full"]
    cfg, want, res, base, real, seeded, ssb, s50 = _depensation_case(got, tmp_path, theta, ratio)
    assert required_features(cfg, got["raw"]) == ["core", "depensation_gate"]
    _assert_exact(res, want, ALL_FIELDS)
    eggs, plain = _eggs(res), _eggs(base)
    one = np.ones(1, dtype=bool)
    allee = float(depensation_factor(ssb[[real]], s50[[real]], np.array([theta]), one)[0])
    if theta < 100:  # the plain formula, where it can be evaluated
        assert allee == ssb[real] ** theta / (s50[real] ** theta + ssb[real] ** theta)
        assert 0.0 < allee < 1.0
    else:
        assert np.isinf(max(ssb[real], s50[real]) ** theta)  # the case is the overflowing one
    assert np.isfinite(eggs).all() and eggs[real] == plain[real] * allee
    assert (allee > 0.5) == (ratio < 1.0)
    untouched = np.arange(cfg.n_species) != real
    assert eggs[untouched].tobytes() == plain[untouched].tobytes()  # the seeded one included
    assert plain[seeded] > 0


def test_depensation_gate_comes_after_the_rv_gate(cycled, tmp_path):
    """Both gates on the same species: the eggs are scaled by the one, then by the other, in
    production's order. Two multiplications in another order need not round alike, so the
    order is part of what must be equal."""
    got = cycled["eec_full"]
    n = got["config"].n_species
    base = _stage(got, "reproduction", tmp_path)
    seeded = np.asarray(base.get("life.seeded")).astype(bool)
    real = int(np.flatnonzero(~seeded & (_eggs(base) > 0))[0])
    rv = np.zeros(n, dtype=bool)
    rv[real] = True
    cfg, want, res, base, real2, _seeded, ssb, s50 = _depensation_case(
        got, tmp_path, 2.5, 1.7, rv_gate_enabled=rv,
        rv_gate_factor_by_index=np.array([0.9, 0.3]), rv_gate_offset=0,
    )  # fmt: skip
    assert real2 == real
    assert required_features(cfg, got["raw"]) == ["core", "rv_gate", "depensation_gate"]
    _assert_exact(res, want, ALL_FIELDS)
    allee = ssb[real] ** 2.5 / (s50[real] ** 2.5 + ssb[real] ** 2.5)
    assert _eggs(res)[real] == (_eggs(base)[real] * 0.3) * allee
