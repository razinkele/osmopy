"""Plan task 2.6a: the life-stage outputs of the C# engine (family ``Stage`` of ``full_outputs``).

``abundance_by_stage``, ``mortality_by_cause_stage`` and ``mortality_rate_by_cause_stage`` against
the Python engine's ``_collect_by_life_stage``, ``mortality_rates_from_counts`` and
``_average_step_outputs``.

A. Record windows, bit for bit. K captured states are collected as K steps and folded, with the
   same school order on both sides. The states are edited so that every branch of the rate
   formula is met: a stage a step emptied (``inf``, and the window stays ``inf``), a stage with
   survivors and no deaths, a stage with no schools. ``np.log`` is the C library's for the
   reference, as .NET's ``Math.Log`` is.
B. One production step. The reference split is the one ``simulate`` took BEFORE reproduction;
   the C# step takes its own inside ``AfterMortality``. Sums run in a different order (the store
   is regrouped), so tier L for the counts; the rates must then be exactly the rate function of
   the C# counts. A split taken after reproduction is shown to differ.
C. Whole runs: the records no draw has influenced equal Python's; the minimal families do not
   change when the stage family is on; deaths by stage add up to deaths by cause; results do not
   depend on the thread count.

``full_outputs`` is not flipped by this task (2.6c does it), so production still refuses a config
that is not minimal; the probes read such a bundle. The test needs the built engine. It skips
without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

import osmose.engine.simulate as sim
from osmose.engine import PythonEngine
from osmose.engine.simulate import StepOutput, _collect_by_life_stage, mortality_rates_from_counts
from osmose.engine_cs.build import (
    FAMILY_STAGE,
    STAGE_BUFFERS,
    allocate_results,
    build_bundle,
    output_families,
)
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.engine import _step_outputs
from tests.test_csharp_pipeline import (
    FAMILIES,
    STEP_CASES,
    _capture_step,
    _post_reproduction_state,
    _probe,
    _python_records,
    _raw,
    _worst,
    _write_states,
    engine_ok,  # noqa: F401  (a fixture)
)

FULL = {"output.minimal.enabled": "false"}
TIER_L = 1e-12
EGGS, JUVENILE, ADULT = 0, 1, 2
ADDITIONAL, FISHING = 2, 3


class _LibmLog:
    """``numpy``, except that ``log`` is the C library's (``math.log``), as ``Math.Log`` is."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def log(x):
        a = np.asarray(x, dtype=np.float64)
        out = [math.log(v) if v > 0 else (-math.inf if v == 0 else math.nan) for v in a.ravel()]
        return np.asarray(out, dtype=np.float64).reshape(a.shape)


@pytest.fixture
def libm_log():
    saved = sim.np
    sim.np = _LibmLog()
    yield
    sim.np = saved


def _stage_of(state, config) -> np.ndarray:
    mature = (state.length >= config.maturity_size[state.species_id]) & (
        state.age_dt >= config.maturity_age_dt[state.species_id]
    )
    return np.where(state.is_egg, EGGS, np.where(mature, ADULT, JUVENILE))


def _bytes_equal(a, b) -> bool:
    return np.asarray(a, dtype=np.float64).tobytes() == np.asarray(b, dtype=np.float64).tobytes()


# ------------------------------------------------------------------------ the contract


def test_the_stage_family_is_declared_exactly_for_a_config_that_is_not_minimal(tmp_path):
    for extra, families in ((FULL, FAMILY_STAGE), ({}, 0)):
        raw = _raw("eec_full", tmp_path, 1, extra)
        config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
        assert output_families(config) == families
        b = build_bundle(config, raw, grid, seed=11)
        assert b.get("output.families") == families
        buffers = allocate_results(b, config)
        n, nf = len(buffers.record_step), config.n_species
        shapes = {name: getattr(buffers, name) for name in STAGE_BUFFERS}
        if families:
            assert shapes["abundance_by_stage"].shape == (n, nf, 3)
            assert shapes["mortality_by_cause_stage"].shape == (n, nf, 8, 3)
            assert shapes["mortality_rate_by_cause_stage"].shape == (n, nf, 8, 3)
            assert all(b.get(f"result.{name}") is shapes[name] for name in STAGE_BUFFERS)
        else:
            assert all(v is None for v in shapes.values())
            assert not any(f"result.{name}" in b.names() for name in STAGE_BUFFERS)


def test_step_outputs_carry_the_stage_rows_and_none_for_a_minimal_run(tmp_path):
    raw = _raw("eec_full", tmp_path, 1, FULL)
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    buffers = allocate_results(build_bundle(config, raw, grid, seed=11), config)
    assert buffers.abundance_by_stage is not None
    assert buffers.mortality_rate_by_cause_stage is not None
    buffers.abundance_by_stage[:] = np.arange(buffers.abundance_by_stage.size).reshape(
        buffers.abundance_by_stage.shape
    )
    buffers.mortality_rate_by_cause_stage[0, 1, 3, 1] = np.inf
    out = _step_outputs(buffers, 1)
    assert len(out) == 1 and isinstance(out[0], StepOutput)
    assert out[0].abundance_by_stage is not None
    assert out[0].mortality_by_cause_stage is not None
    assert out[0].mortality_rate_by_cause_stage is not None
    assert np.array_equal(out[0].abundance_by_stage, buffers.abundance_by_stage[0])
    assert out[0].mortality_by_cause_stage.shape == (config.n_species, 8, 3)
    assert out[0].mortality_rate_by_cause_stage[1, 3, 1] == np.inf

    raw = _raw("eec_full", tmp_path, 1)
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    minimal = _step_outputs(allocate_results(build_bundle(config, raw, grid, seed=11), config), 1)
    assert minimal[0].abundance_by_stage is None and minimal[0].mortality_by_cause_stage is None
    assert minimal[0].mortality_rate_by_cause_stage is None


# ------------------------------------------------------------------------ A. windows

EMPTIED, UNTOUCHED, ABSENT = 4, 7, 9  # the states (steps) that carry the three edits


def _edit(states: list, config) -> dict:
    """Edits three of the states and says where: an emptied juvenile stage, an adult stage with
    survivors and no deaths at all, and an egg stage with neither schools nor deaths."""
    st = states[EMPTIED]
    stage = _stage_of(st, config)
    focal = st.species_id < config.n_species
    kills = st.n_dead.sum(axis=1) > 0
    counts = np.bincount(
        st.species_id[focal & (stage == JUVENILE) & kills], minlength=config.n_species
    )
    emptied = int(np.argmax(counts))
    assert counts[emptied] > 0
    st.abundance[(st.species_id == emptied) & (stage == JUVENILE)] = 0.0
    states[EMPTIED] = st.replace(biomass=st.abundance * st.weight)

    st = states[UNTOUCHED]
    stage = _stage_of(st, config)
    adults = np.bincount(
        st.species_id[(st.species_id < config.n_species) & (stage == ADULT) & (st.abundance > 0)],
        minlength=config.n_species,
    )
    untouched = int(np.argmax(adults))
    assert adults[untouched] > 0
    st.n_dead[(st.species_id == untouched) & (stage == ADULT)] = 0.0

    st = states[ABSENT]
    eggs = (st.species_id < config.n_species) & (_stage_of(st, config) == EGGS)
    absent = int(np.argmax(np.bincount(st.species_id[eggs], minlength=config.n_species)))
    gone = eggs & (st.species_id == absent)
    assert gone.any()
    st.abundance[gone] = 0.0
    st.n_dead[gone] = 0.0
    states[ABSENT] = st.replace(biomass=st.abundance * st.weight)
    return {"emptied": emptied, "untouched": untouched, "absent": absent}


@pytest.fixture(scope="module")
def stage_states(tmp_path_factory) -> dict:
    out = {}
    for demo, step in (("tutorial", 129), ("eec_full", 30)):
        d = tmp_path_factory.mktemp(demo)
        raw = _raw(demo, d, step // 24 + 1, FULL)
        config, grid, raw, st = _post_reproduction_state(raw, step)
        rng = np.random.default_rng(7)
        states = []
        for _ in range(12):  # twelve "steps": the same schools, perturbed abundance and deaths
            s = copy.deepcopy(st)
            s.abundance[:] *= rng.uniform(0.5, 1.5, len(s))
            s.n_dead[:] *= rng.uniform(0.5, 1.5, s.n_dead.shape)
            states.append(s.replace(biomass=s.abundance * s.weight))
        out[demo] = (config, grid, raw, states, _edit(states, config))
    return out


@pytest.mark.parametrize("demo", ["tutorial", "eec_full"])
@pytest.mark.parametrize(("freq", "step0"), [(5, False), (5, True), (1, False), (12, False)])
def test_windows_equal_the_python_engines_bit_for_bit(
    engine_ok,  # noqa: F811
    libm_log,
    stage_states,
    demo,
    freq,
    step0,
    tmp_path,
):
    config, grid, raw, states, where = stage_states[demo]
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = freq, step0
    assert not cfg.output_minimal
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(states, tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    want = _python_records(states, cfg, freq, step0)
    assert np.array_equal(res.get("out.record_step"), [r.step for r in want])
    for name in STAGE_BUFFERS:
        got = res.get(f"out.{name}")
        assert got.shape[0] == len(want)
        for i, r in enumerate(want):
            assert _bytes_equal(got[i], getattr(r, name)), (name, i)
    for cs_name, py_name in FAMILIES:  # the minimal families are collected as before
        got = res.get(f"out.{cs_name}")
        for i, r in enumerate(want):
            assert _bytes_equal(got[i], getattr(r, py_name)), (cs_name, i)

    # The edits are inside the compared records, and read as the formula says.
    first = 1 if step0 else 0
    rates = res.get("out.mortality_rate_by_cause_stage")
    deaths = res.get("out.mortality_by_cause_stage")
    emptied_record = first + (EMPTIED - first) // freq
    juvenile = rates[emptied_record, where["emptied"], :, JUVENILE]
    killed = deaths[emptied_record, where["emptied"], :, JUVENILE] > 0
    assert np.isinf(juvenile).any() and (juvenile[~np.isinf(juvenile)] == 0).all()
    assert np.isinf(juvenile).sum() <= killed.sum()
    assert np.isfinite(rates[emptied_record, where["emptied"], :, ADULT]).all()
    if freq == 1:  # one step per record: each edit reads as its branch of the formula says
        abundance = res.get("out.abundance_by_stage")
        assert (rates[UNTOUCHED, where["untouched"], :, ADULT] == 0).all()  # no deaths
        assert abundance[UNTOUCHED, where["untouched"], ADULT] > 0  # ...with survivors
        assert abundance[ABSENT, where["absent"], EGGS] == 0  # no schools at all
        assert (deaths[ABSENT, where["absent"], :, EGGS] == 0).all()
        assert (rates[ABSENT, where["absent"], :, EGGS] == 0).all()
        assert not np.isinf(rates[EMPTIED + 1]).any()  # the emptied step is one record only


def test_a_window_sums_per_step_rates_and_is_not_a_rate_of_the_window_counts(
    engine_ok,  # noqa: F811
    libm_log,
    stage_states,
    tmp_path,
):
    """Java's convention, and the trap CLAUDE.md records: the record is the SUM of per-step
    rates. One rate formed from the window's summed counts is a different number."""
    config, grid, raw, states, _ = stage_states["eec_full"]
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = 4, False
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(states[:4], tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    rates = res.get("out.mortality_rate_by_cause_stage")[0]
    per_step = [_collect_by_life_stage(s, cfg) for s in states[:4]]
    summed = np.sum(
        [
            mortality_rates_from_counts(d.transpose(0, 2, 1), a).transpose(0, 2, 1)
            for a, d in per_step
        ],
        axis=0,
    )
    assert _bytes_equal(rates, summed)
    of_window = mortality_rates_from_counts(
        res.get("out.mortality_by_cause_stage")[0].transpose(0, 2, 1),
        res.get("out.abundance_by_stage")[0],
    ).transpose(0, 2, 1)
    live = rates > 0
    assert live.any() and not np.allclose(rates[live], of_window[live], rtol=0.05)


# ------------------------------------------------------------------- B. one whole step


@pytest.mark.parametrize(
    ("demo", "edge_ages"), [("bay_of_biscay", False), ("eec_full", False), ("eec_full", True)]
)
def test_one_step_takes_the_split_before_reproduction_as_production_does(
    engine_ok,  # noqa: F811
    demo,
    edge_ages,
    tmp_path,
    record_property,
):
    step = STEP_CASES[demo]
    got = _capture_step(_raw(demo, tmp_path, step // 24 + 1, FULL), step, edge_ages)
    config, out = got["config"], got["output"]
    assert not config.output_minimal and out.abundance_by_stage is not None
    build_bundle(config, got["raw"], got["grid"], seed=11).save(tmp_path / "model.bin")
    _write_states([got["entry"]], tmp_path / "entry.bin")
    from tests.test_csharp_mortality_step import _write_draws

    _write_draws(got["record"], tmp_path / "draws.bin")
    res = _probe(
        "Step", tmp_path / "model.bin", tmp_path / "entry.bin", tmp_path / "draws.bin", step,
        tmp_path / "out.bin",
    )  # fmt: skip
    abundance = res.get("out.abundance_by_stage")[0]
    deaths = res.get("out.mortality_by_cause_stage")[0]
    rates = res.get("out.mortality_rate_by_cause_stage")[0]
    worst = {
        "abundance_by_stage": _worst(abundance, out.abundance_by_stage),
        "mortality_by_cause_stage": _worst(deaths, out.mortality_by_cause_stage),
    }
    record_property("worst_relative_error", worst)
    assert all(v <= TIER_L for v in worst.values()), worst

    # The rates are exactly the rate function of the C# counts (the C library's log)...
    saved, sim.np = sim.np, _LibmLog()
    try:
        of_counts = mortality_rates_from_counts(deaths.transpose(0, 2, 1), abundance)
    finally:
        sim.np = saved
    assert _bytes_equal(rates, of_counts.transpose(0, 2, 1))
    # ...and agree with production's to the accuracy the counts allow (Z = -ln(1 - small)
    # amplifies a last-bit difference in the counts by 1 / Z).
    assert np.array_equal(np.isinf(rates), np.isinf(out.mortality_rate_by_cause_stage))
    finite = np.isfinite(rates)
    assert np.allclose(rates[finite], out.mortality_rate_by_cause_stage[finite], rtol=1e-6)

    # The order is what this test is for: the same step split AFTER reproduction is another
    # answer (the new eggs are counted, and this step's survivors have aged out of the egg stage).
    late_abundance, late_deaths = _collect_by_life_stage(got["final"], config)
    assert _worst(late_abundance, out.abundance_by_stage) > 1e-3
    assert out.abundance_by_stage[:, EGGS].sum() > 0 and out.mortality_by_cause_stage.sum() > 0
    if demo == "eec_full":
        assert out.abundance_by_stage[:, ADULT].sum() > 0
        assert _worst(late_deaths, out.mortality_by_cause_stage) > 1e-3


# --------------------------------------------------------------------- C. whole runs


def _run(raw: dict, tmp_path, name: str, threads: int = 1) -> tuple[Bundle, object]:
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    d = tmp_path / name
    d.mkdir()
    build_bundle(config, raw, grid, seed=11, threads=threads).save(d / "model.bin")
    return _probe("Run", d / "model.bin", d / "out.bin"), config


@pytest.mark.parametrize(("demo", "leading"), [("eec_full", 2), ("bay_of_biscay", 4)])
def test_whole_run_leading_records_equal_python(engine_ok, demo, leading, tmp_path):  # noqa: F811
    raw = _raw(demo, tmp_path, 1, FULL | {"output.recordfrequency.ndt": "1"})
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    python = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    res, _ = _run(raw, tmp_path, "full")
    assert np.array_equal(res.get("out.record_step"), [r.step for r in python])
    for i in range(leading):  # no draw has acted yet: seeding, then larva deaths on unfed eggs
        for name in ("abundance_by_stage", "mortality_by_cause_stage"):
            assert _worst(res.get(f"out.{name}")[i], getattr(python[i], name)) <= 1e-9, (name, i)
        got = res.get("out.mortality_rate_by_cause_stage")[i]
        want = python[i].mortality_rate_by_cause_stage
        assert np.array_equal(np.isinf(got), np.isinf(want)), i
        assert np.allclose(got[np.isfinite(got)], want[np.isfinite(want)], rtol=1e-6), i
    if demo == "eec_full":
        # Step 1's larva deaths fall on schools that are eggs when they die. Reproduction then
        # ages them out of the egg stage, so only a split taken before it books them as eggs.
        eggs = res.get("out.mortality_by_cause_stage")[1][:, ADDITIONAL, EGGS]
        assert eggs.sum() > 0 and python[1].mortality_by_cause_stage[:, ADDITIONAL, EGGS].sum() > 0
        assert res.get("out.mortality_by_cause_stage")[1][:, ADDITIONAL, JUVENILE].sum() == 0


@pytest.mark.parametrize("demo", ["eec_full", "tutorial"])
def test_a_full_run_leaves_the_minimal_families_as_they_were(engine_ok, demo, tmp_path):  # noqa: F811
    extra = {"output.recordfrequency.ndt": "6", "output.step0.include": "true"}
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    minimal, _ = _run(_raw(demo, tmp_path / "a", 2, extra), tmp_path, "minimal")
    full, config = _run(_raw(demo, tmp_path / "b", 2, FULL | extra), tmp_path, "full")
    assert "out.abundance_by_stage" not in minimal.names()
    for name in ("record_step", "window_steps", "biomass", "abundance", "mortality", "yield"):
        assert minimal.get(f"out.{name}").tobytes() == full.get(f"out.{name}").tobytes(), name

    # Deaths by stage partition deaths by cause (another grouping of the same sum).
    deaths = full.get("out.mortality_by_cause_stage")
    by_cause = full.get("out.mortality")
    assert deaths.shape == (*by_cause.shape, 3)
    assert np.allclose(deaths.sum(axis=3), by_cause, rtol=1e-9, atol=0.0)
    assert by_cause.sum() > 0 and (deaths.sum(axis=(0, 1, 2)) > 0).all()  # all three stages die
    abundance = full.get("out.abundance_by_stage")
    assert (abundance >= 0).all() and (abundance[1:].sum(axis=(0, 1)) > 0).all()
    assert (full.get("out.abundance_by_stage")[0] == 0).all()  # the initial record: no schools
    rates = full.get("out.mortality_rate_by_cause_stage")
    assert not np.isnan(rates).any() and (rates >= 0).all()
    assert ((rates > 0) <= (deaths > 0)).all()  # a cause that did not kill reads exactly 0
    assert config.n_background == (2 if demo == "tutorial" else 0)


def test_the_stage_records_do_not_depend_on_the_thread_count(engine_ok, tmp_path):  # noqa: F811
    raw = _raw("eec_full", tmp_path, 2, FULL)
    runs = [_run(raw, tmp_path, f"t{t}", threads=t)[0] for t in (1, 4)]
    for name in STAGE_BUFFERS:
        assert runs[0].get(f"out.{name}").tobytes() == runs[1].get(f"out.{name}").tobytes(), name
    assert runs[0].get("out.mortality_by_cause_stage").sum() > 0
