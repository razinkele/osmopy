"""Plan task 2.6c: the C# engine's diet (``diet_by_species``, from which the diet matrix and the
predator pressure are derived), against the Python engine's kernel, ``aggregate_diet_by_species``
and ``_average_step_outputs``.

A. The contract: one switch, one bit; the prey-column count is the reference's own (the focal
   species, then the resources ``ResourceState`` discovers), and ``build_bundle`` publishes the
   resource names as ``simulate`` does.
B. The aggregation and the window, bit for bit: per-school diet rows (background predators'
   included) are added per predator species and summed over the window.
C. One production step. The kernel's per-school rows against the Python per-cell oracle on the
   same draws, and the step's diet against production's: on EEC, on Bay of Biscay, and on the
   Baltic-derived config, whose background predators (seal, cormorant) have rows of their own.
D. The prey column of a BACKGROUND school. The reference books a school prey under its species
   id, and a background species' id is ``n_species + k``: the column of resource ``k``. Nothing
   eats a seal on Baltic, so the case is forced here by shrinking some background schools.
E. Whole runs: the diet changes nothing else (the kernel is where it is collected, so this is
   the check that matters); results do not depend on the thread count.

The test needs the built engine. It skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from osmose.engine.output import aggregate_diet_by_species
from osmose.engine.resources import ResourceState
from osmose.engine.simulate import _average_step_outputs, _collect_outputs, _strip_background
from osmose.engine_cs import build
from osmose.engine_cs.build import (
    allocate_results,
    build_bundle,
    diet_columns,
    family_shapes,
    output_families,
)
from osmose.engine_cs.engine import _step_outputs
from tests.test_csharp_mortality_step import _write_draws
from tests.test_csharp_outputs_families import OFF
from tests.test_csharp_pipeline import (
    FAMILIES,
    _capture_step,
    _probe,
    _raw,
    _worst,
    _write_states,
    engine_ok,  # noqa: F401  (a fixture)
)

DIET = OFF | {"output.diet.composition.enabled": "true"}
TIER_L = 1e-12
PREDATION = 0
STEPS = {"eec_full": 30, "bay_of_biscay": 9, "tutorial": 129}


# ------------------------------------------------------------------------ A. the contract


def test_the_switch_turns_on_the_diet_family_and_nothing_else(tmp_path):
    on = EngineConfig.from_dict(_raw("eec_full", tmp_path, 1, DIET))
    assert on.diet_output_enabled
    assert output_families(on) == build.FAMILY_STAGE | build.FAMILY_DIET == 513
    off = EngineConfig.from_dict(_raw("eec_full", tmp_path, 1, OFF))
    assert output_families(off) == build.FAMILY_STAGE
    # Predator pressure is derived from the diet; its own switch collects nothing.
    pressure = OFF | {"output.diet.pressure.enabled": "true"}
    assert output_families(EngineConfig.from_dict(_raw("eec_full", tmp_path, 1, pressure))) == 1
    minimal = DIET | {"output.minimal.enabled": "true"}
    assert output_families(EngineConfig.from_dict(_raw("eec_full", tmp_path, 1, minimal))) == 0


@pytest.mark.parametrize(("demo", "n_resources"), [("eec_full", 10), ("tutorial", None)])
def test_the_prey_columns_are_the_reference_s(demo, n_resources, tmp_path):
    raw = _raw(demo, tmp_path, 1, DIET)
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    assert config.resource_names == []  # published at run time, not by the config
    b = build_bundle(config, raw, grid, seed=11)
    names = [r.name for r in ResourceState(config=config.raw_config, grid=grid).species]
    assert config.resource_names == names and len(names) > 0  # as simulate() publishes them
    if n_resources is not None:
        assert len(names) == n_resources and names[0] == "Dinoflagellates"
    n_species_all = config.n_species + config.n_background
    assert b.get("output.diet_columns") == diet_columns(config) == n_species_all + len(names)
    buffers = allocate_results(b, config)
    n, rows = len(buffers.record_step), config.n_species + config.n_background
    assert buffers.families["diet_by_species"].shape == (n, rows, n_species_all + len(names))
    assert list(family_shapes(config))[-1] == "diet_by_species"
    assert config.n_background == (2 if demo == "tutorial" else 0)

    (tmp_path / "off").mkdir()
    off = _raw(demo, tmp_path / "off", 1, OFF)
    config, grid, *_ = PythonEngine()._prepare_run(off, 11)
    b = build_bundle(config, off, grid, seed=11)
    assert "output.diet_columns" not in b.names()
    assert "diet_by_species" not in allocate_results(b, config).families
    assert config.resource_names == names  # published whatever the outputs


def test_step_outputs_carry_the_diet_and_none_for_the_initial_record(tmp_path):
    """``simulate`` leaves ``diet_by_species`` None on the initial record (no mortality step ran
    before it), and the diet frames skip it; a zero matrix there would add a row."""
    raw = _raw("eec_full", tmp_path, 1, DIET | {"output.step0.include": "true"})
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    buffers = allocate_results(build_bundle(config, raw, grid, seed=11), config)
    diet = buffers.families["diet_by_species"]
    diet[:] = np.arange(diet.size).reshape(diet.shape)
    buffers.record_step[:] = [-1, 23]
    out = _step_outputs(buffers, 2)
    assert out[0].step == -1 and out[0].diet_by_species is None
    assert out[1].diet_by_species is not None and np.array_equal(out[1].diet_by_species, diet[1])

    raw = _raw("eec_full", tmp_path, 1, OFF)
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    off = _step_outputs(allocate_results(build_bundle(config, raw, grid, seed=11), config), 1)
    assert off[0].diet_by_species is None


@pytest.mark.parametrize("demo", ["eec_full", "tutorial"])
def test_records_of_a_stock_python_run_survive_the_buffers_and_both_consumers(demo, tmp_path):
    """As the 2.6b round trip, with the diet and every family the demo ships with: a real
    Python run's records, packed into the result buffers and rebuilt, give back every
    ``StepOutput`` field, the same frames and byte-identical files. The diet and the predator
    pressure files are among them, with the resources as prey labels."""
    import dataclasses

    import pandas as pd

    import osmose.engine.simulate as sim
    from osmose.engine.output import write_outputs
    from osmose.engine.simulate import StepOutput
    from osmose.results import OsmoseResults
    from tests.test_csharp_engine_e2e import _families_row
    from tests.test_csharp_outputs_families import _fields_equal

    extra = {
        "output.minimal.enabled": "false",
        "output.recordfrequency.ndt": "6",
        "output.step0.include": "true",
    }
    raw = _raw(demo, tmp_path, 1, extra)
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    python = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    published = list(config.resource_names)
    config.resource_names = []  # as before any run: the builder must publish them itself
    buffers = allocate_results(build_bundle(config, raw, grid, seed=11), config)
    assert config.resource_names == published and published
    assert "diet_by_species" in buffers.families and len(buffers.record_step) == len(python) == 5
    nf = config.n_species
    for i, r in enumerate(python):
        buffers.record_step[i], buffers.window_steps[i] = r.step, r.window_steps
        buffers.biomass[i], buffers.abundance[i] = r.biomass, r.abundance
        buffers.mortality[i], buffers.yield_[i] = r.mortality_by_cause, r.yield_by_species
        for name, rows in buffers.families.items():
            row = _families_row(r, name, nf)
            if row is not None:  # the initial record has no diet: its row stays zero
                rows[i] = row
    assert python[0].step == -1 and python[0].diet_by_species is None
    rebuilt = _step_outputs(buffers, len(python))

    for i, (a, b) in enumerate(zip(python, rebuilt, strict=True)):
        for f in dataclasses.fields(StepOutput):
            assert _fields_equal(getattr(a, f.name), getattr(b, f.name)), (i, f.name)
    assert python[-1].diet_by_species is not None and python[-1].diet_by_species.sum() > 0

    frames = [OsmoseResults.from_outputs(o, config, grid)._csv_cache for o in (python, rebuilt)]
    assert sorted(frames[0]) == sorted(frames[1])
    for key in frames[0]:
        a, b = frames[0][key], frames[1][key]
        if isinstance(a, dict):
            assert sorted(a) == sorted(b), key
            for sub in a:
                pd.testing.assert_frame_equal(a[sub], b[sub], check_exact=True, obj=f"{key}/{sub}")
        else:
            pd.testing.assert_frame_equal(a, b, check_exact=True, obj=str(key))

    for name, records in (("python", python), ("rebuilt", rebuilt)):
        write_outputs(records, tmp_path / name, config, grid=grid)
    files = [
        sorted(p.relative_to(tmp_path / n) for p in (tmp_path / n).rglob("*") if p.is_file())
        for n in ("python", "rebuilt")
    ]
    assert files[0] == files[1]
    assert any("diet" in str(f).lower() for f in files[0])
    assert any("pressure" in str(f).lower() for f in files[0])
    for rel in files[0]:
        assert (tmp_path / "python" / rel).read_bytes() == (
            tmp_path / "rebuilt" / rel
        ).read_bytes(), rel
    diet_file = next(f for f in files[0] if "dietmatrix" in str(f).lower().replace("_", ""))
    assert published[0] in (tmp_path / "python" / diet_file).read_text()


# -------------------------------------------------------- B. the aggregation and the window


@pytest.fixture(scope="module")
def captures(tmp_path_factory) -> dict:
    """One production step per demo, with the diet on (see C)."""
    out = {}
    for demo, step in STEPS.items():
        d = tmp_path_factory.mktemp(demo)
        out[demo] = _capture_step(_raw(demo, d, step // 24 + 1, DIET), step)
    return out


@pytest.mark.parametrize("demo", ["tutorial", "eec_full"])
@pytest.mark.parametrize(("freq", "step0"), [(4, False), (4, True), (1, False), (6, False)])
def test_the_aggregation_and_the_window_equal_the_python_engines_bit_for_bit(
    engine_ok,  # noqa: F811
    captures,
    demo,
    freq,
    step0,
    tmp_path,
):
    got = captures[demo]
    cfg, state = copy.copy(got["config"]), got["entry"]  # the entry state: background included
    cfg.output_record_frequency, cfg.output_step0_include = freq, step0
    cols, n_pred = diet_columns(cfg), cfg.n_species + cfg.n_background
    rng = np.random.default_rng(3)
    k = 6
    states = [copy.deepcopy(state) for _ in range(k)]
    # Sparse random rows of very different sizes, so that the order of the additions shows.
    diets = [
        np.where(
            rng.random((len(state), cols)) < 0.3,
            rng.random((len(state), cols)) * 10.0 ** rng.integers(-6, 7, (len(state), cols)),
            0.0,
        )
        for _ in range(k)
    ]
    if step0:
        diets[0][:] = 0.0  # the initial state: no mortality step ran before it
    build_bundle(cfg, got["raw"], got["grid"], seed=11).save(tmp_path / "model.bin")
    _write_states(states, tmp_path / "states.bin", {"state.diet": np.concatenate(diets)})
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")

    # simulate()'s order and recording rule: the diet is aggregated while the background
    # schools are still in the state, the other outputs are collected after they left; the
    # initial record has no diet; a window sums its steps'.
    focal = _strip_background(state, int((~state.is_background).sum()))
    assert len(focal) < len(state) or demo == "eec_full"
    first = 1 if step0 else 0
    records, acc = [], []
    if step0:
        records.append(_collect_outputs(focal, cfg, step=-1))
    for step in range(k - first):
        per_species = aggregate_diet_by_species(diets[first + step], state.species_id, n_pred)
        acc.append(_collect_outputs(focal, cfg, step, diet_by_species=per_species))
        if (step + 1) % freq == 0:
            records.append(_average_step_outputs(acc, freq, step))
            acc = []
    if acc:
        records.append(_average_step_outputs(acc, len(acc), k - first - 1))

    out = res.get("out.diet_by_species")
    assert out.shape == (len(records), n_pred, cols)
    assert np.array_equal(res.get("out.record_step"), [r.step for r in records])
    for i, r in enumerate(records):
        if r.diet_by_species is None:
            assert r.step == -1 and not out[i].any()  # the initial record: nothing was eaten
        else:
            assert out[i].tobytes() == np.asarray(r.diet_by_species).tobytes(), i
    for cs_name, py_name in FAMILIES:  # the minimal families are collected as before
        mine = res.get(f"out.{cs_name}")
        for i, r in enumerate(records):
            assert mine[i].tobytes() == np.asarray(getattr(r, py_name)).tobytes(), (cs_name, i)
    if demo == "tutorial":  # the background predators have rows, after the focal species
        assert state.is_background.any() and out[-1, cfg.n_species :].sum() > 0
    # Not a sum that any order gives: the same rows added school by school in reverse differ.
    one = aggregate_diet_by_species(diets[-1], state.species_id, n_pred)
    backwards = aggregate_diet_by_species(diets[-1][::-1], state.species_id[::-1], n_pred)
    assert (one != backwards).any()


# ------------------------------------------------------------------- C. one whole step


def _step(got: dict, tmp_path):
    build_bundle(got["config"], got["raw"], got["grid"], seed=11).save(tmp_path / "model.bin")
    _write_states([got["entry"]], tmp_path / "entry.bin")
    _write_draws(got["record"], tmp_path / "draws.bin")
    return _probe(
        "Step", tmp_path / "model.bin", tmp_path / "entry.bin", tmp_path / "draws.bin",
        got["step"], tmp_path / "out.bin",
    )  # fmt: skip


@pytest.mark.parametrize("demo", ["bay_of_biscay", "eec_full", "tutorial"])
def test_one_production_step_equals_the_python_kernel_and_its_aggregate(
    engine_ok,  # noqa: F811
    captures,
    demo,
    tmp_path,
    record_property,
):
    got = captures[demo]
    config, entry, out = got["config"], got["entry"], got["output"]
    res = _step(got, tmp_path)
    cols, nf = diet_columns(config), config.n_species
    rows = res.get("s.diet")  # one row per school of the entry state, in its order
    assert rows.shape == got["diet"].shape == (len(entry), cols)
    worst = {
        "per_school": _worst(rows, got["diet"]),
        "by_species": _worst(res.get("out.diet_by_species")[0], out.diet_by_species),
    }
    for cs_name, py_name in FAMILIES:  # the step itself still equals production's
        worst[cs_name] = _worst(res.get(f"out.{cs_name}")[0], getattr(out, py_name))
    record_property("worst_relative_error", worst)
    assert all(v <= TIER_L for v in worst.values()), worst
    assert np.array_equal(rows == 0, got["diet"] == 0)  # the same predators ate the same prey

    # Positive witnesses: schools ate schools and resources, and every school's row is its own.
    # (Bay of Biscay's fish eat no plankton: its accessibility matrix is transposed, CLAUDE.md.)
    assert got["diet"][:, :nf].sum() > 0
    assert (got["diet"][:, nf:].sum() > 0) == (demo != "bay_of_biscay")
    fed = np.flatnonzero(got["diet"].sum(axis=1) > 0)
    assert len(fed) > 20 and len(np.unique(got["diet"][fed], axis=0)) > 0.9 * len(fed)
    assert out.diet_by_species.shape == (nf + config.n_background, cols)
    if demo == "tutorial":
        # The seal and the cormorant eat (their rows follow the focal species'), and their
        # schools were still in the store when the diet was taken.
        background = out.diet_by_species[nf:]
        assert entry.is_background.any() and (background.sum(axis=1) > 0).all()
        assert got["diet"][entry.is_background].sum() > 0
        # Nothing eats them: Baltic's seals and cormorants are far too large to be prey.
        assert not got["after_mortality"].n_dead[entry.is_background, PREDATION].any()


# ------------------------------------------------- D. a background school as prey


def test_a_background_prey_is_booked_in_its_own_column_on_both_engines(
    engine_ok,  # noqa: F811
    tmp_path,
):
    """Issue #322. A school prey is booked at its species id, and a background school's is
    ``n_species + k``. The prey axis now has a column there (focal, background, resources, as
    Java's DietOutput), where it used to be the column of resource ``k``. Inert on Baltic as
    shipped; here every third background school is shrunk to 8 cm so that fish eat it. Both
    engines must book it in the background species' own column, and nowhere else."""

    def shrink(state, config):
        background = np.flatnonzero(state.is_background)
        state.length[background[::3]] = 8.0

    step = STEPS["tutorial"]
    got = _capture_step(_raw("tutorial", tmp_path, step // 24 + 1, DIET), step, edit=shrink)
    config, entry, after = got["config"], got["entry"], got["after_mortality"]
    nf, nb = config.n_species, config.n_background
    eaten = after.n_dead[:, PREDATION] * entry.weight  # tonnes of each school eaten
    assert eaten[entry.is_background].sum() > 0  # the forced case happened

    res = _step(got, tmp_path)
    rows = res.get("s.diet")
    assert rows.shape[1] == nf + nb + len(config.resource_names)
    assert _worst(rows, got["diet"]) <= TIER_L
    assert _worst(res.get("out.diet_by_species")[0], got["output"].diet_by_species) <= TIER_L

    # Where it was booked, on the C# engine's own rows: every species column, the background
    # ones included, holds what was eaten of that species' schools and nothing else.
    by_prey_species = np.bincount(entry.species_id, weights=eaten, minlength=nf + nb)
    assert by_prey_species[nf:].sum() > 0
    assert np.allclose(rows[:, : nf + nb].sum(axis=0), by_prey_species, rtol=1e-9)
    # The resources are the last columns; a predator's row sums to what it ate.
    assert rows[:, nf + nb :].sum() > 0
    assert np.allclose(rows.sum(axis=1), after.preyed_biomass[: len(rows)], rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------- E. whole runs


def _run(raw: dict, tmp_path, name: str, threads: int = 1):
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    d = tmp_path / name
    d.mkdir()
    build_bundle(config, raw, grid, seed=11, threads=threads).save(d / "model.bin")
    return _probe("Run", d / "model.bin", d / "out.bin"), config


@pytest.mark.parametrize("demo", ["eec_full", "tutorial"])
def test_collecting_the_diet_changes_nothing_else(engine_ok, demo, tmp_path):  # noqa: F811
    """The diet is collected inside the mortality kernel. A run with it must give the same
    schools as a run without: every other family byte for byte, at the same seed."""
    extra = {"output.recordfrequency.ndt": "6", "output.step0.include": "true"}
    for sub in ("a", "b", "c"):
        (tmp_path / sub).mkdir()
    minimal, _ = _run(_raw(demo, tmp_path / "a", 2, extra), tmp_path, "minimal")
    without, _ = _run(_raw(demo, tmp_path / "b", 2, OFF | extra), tmp_path, "without")
    full, config = _run(_raw(demo, tmp_path / "c", 2, DIET | extra), tmp_path, "with")
    for name in ("record_step", "window_steps", "biomass", "abundance", "mortality", "yield"):
        assert minimal.get(f"out.{name}").tobytes() == full.get(f"out.{name}").tobytes(), name
    for name in build.STAGE_BUFFERS:
        assert without.get(f"out.{name}").tobytes() == full.get(f"out.{name}").tobytes(), name
    assert "out.diet_by_species" not in without.names()

    diet = full.get("out.diet_by_species")
    nf = config.n_species
    assert diet.shape[1:] == (nf + config.n_background, diet_columns(config))
    assert not diet[0].any() and (diet >= 0).all() and np.isfinite(diet).all()
    assert diet[1:, :nf, :nf].sum() > 0 and diet[1:, :nf, nf:].sum() > 0  # fish and plankton
    if demo == "tutorial":
        assert (diet[1:, nf:].sum(axis=(0, 2)) > 0).all()  # both background predators eat
    # What fish ate of fish is what predation killed: the biomass eaten of each prey species
    # is its predation deaths times the weight of the schools they came from. The outputs hold
    # deaths in numbers, so only the sign can be checked here: a species nobody ate lost nobody.
    eaten = diet[1:, :, :nf].sum(axis=(0, 1))
    killed = full.get("out.mortality")[1:, :, PREDATION].sum(axis=0)
    assert np.array_equal(eaten > 0, killed > 0)


@pytest.mark.parametrize("demo", ["eec_full", "tutorial"])
def test_a_whole_run_s_diet_is_the_size_of_the_python_engines(engine_ok, demo, tmp_path):  # noqa: F811
    """The two engines draw from different random streams, so no value can be compared. But the
    step tests above start from one state and one step: they cannot see a diet that is not
    cleared between steps, or one taken after the background schools left. Over a run such a
    diet is many times too large, or lacks its background rows; so each window's total, and
    each main predator's, must be of the Python engine's size."""
    import osmose.engine.simulate as sim

    raw = _raw(demo, tmp_path, 2, DIET | {"output.recordfrequency.ndt": "12"})
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    python = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    res, _ = _run(raw, tmp_path, "cs")
    mine = res.get("out.diet_by_species")
    theirs = np.stack([r.diet_by_species for r in python])
    assert mine.shape == theirs.shape and len(python) == 4
    for i in range(1, 4):  # the first half-year is seeding: little is eaten yet
        assert 0.5 < mine[i].sum() / theirs[i].sum() < 2.0, i  # measured: 0.89 to 1.04
        by_predator, want = mine[i].sum(axis=1), theirs[i].sum(axis=1)
        main = want > 0.05 * want.sum()  # measured 2026-11-18: 0.77 to 1.38 for these
        assert main.sum() >= 3
        ratio = by_predator[main] / want[main]
        assert (ratio > 1 / 3).all() and (ratio < 3).all(), (i, ratio)
    if demo == "tutorial":
        nf = config.n_species
        assert (theirs[1:, nf:].sum(axis=(0, 2)) > 0).all()  # the reference's seals eat too
        ratio = mine[1:, nf:].sum(axis=(0, 2)) / theirs[1:, nf:].sum(axis=(0, 2))
        assert (ratio > 0.5).all() and (ratio < 2.0).all(), ratio


@pytest.mark.parametrize("demo", ["eec_full", "tutorial"])
def test_the_diet_does_not_depend_on_the_thread_count(engine_ok, demo, tmp_path):  # noqa: F811
    raw = _raw(demo, tmp_path, 2, DIET)
    runs = [_run(raw, tmp_path, f"t{t}", threads=t)[0] for t in (1, 2, 14)]
    for name in ("diet_by_species", "biomass", "mortality"):
        one = runs[0].get(f"out.{name}").tobytes()
        assert all(r.get(f"out.{name}").tobytes() == one for r in runs[1:]), name
    assert runs[0].get("out.diet_by_species").sum() > 0
