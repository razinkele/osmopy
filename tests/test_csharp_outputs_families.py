"""Plan task 2.6b: the C# engine's distributions by age and by size, SSB, yield in numbers, mean
size and mean trophic level, against the Python engine's ``_collect_distributions``,
``_collect_ssb``, ``_collect_yield_n``, ``_collect_mean_size``, ``_collect_mean_tl`` and
``_average_step_outputs``.

A. The contract: the family mask follows the switches ``_collect_outputs`` itself reads, one bit
   per ``StepOutput`` field; the binning tables are the reference's own.
B. Record windows, bit for bit, with the same school order on both sides. The states carry
   lengths exactly on class edges, below the first and beyond the last, ages beyond a species'
   last class, a species unfed in some steps of a window (mean TL over the steps that have it),
   a species unfed through a whole window (absent from the record), and a species with no
   survivor in one step.
C. One production step, at tier L (the C# store is regrouped, so sums run in another order).
D. Whole runs: the records no draw has influenced equal Python's; the minimal families do not
   change; by-age and by-size add up to the same total; the distributions include the eggs the
   abundance output leaves out; results do not depend on the thread count.

``full_outputs`` is not flipped by this task (2.6c does it), so production still refuses a config
that is not minimal; the probes read such a bundle. The test needs the built engine. It skips
without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import osmose.engine.simulate as sim
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from osmose.engine.simulate import StepOutput, _collect_outputs
from osmose.engine_cs import build
from osmose.engine_cs.build import (
    age_bin_offsets,
    allocate_results,
    build_bundle,
    family_shapes,
    output_families,
    size_edges,
)
from osmose.engine_cs.engine import _step_outputs
from tests.test_csharp_mortality_step import _write_draws
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

#: The switch that turns each family on, and a second one where the reference reads two.
SWITCHES = {
    "biomass_by_age": ("output.biomass.byage.enabled",),
    "abundance_by_age": ("output.abundance.byage.enabled",),
    "biomass_by_size": ("output.biomass.bysize.enabled",),
    "abundance_by_size": ("output.abundance.bysize.enabled",),
    "mean_tl": ("output.tl.enabled", "output.meantl.enabled"),
    "yield_n": ("output.yield.abundance.enabled", "output.yield.abundance.netcdf.enabled"),
    "mean_size": ("output.size.enabled", "output.size.netcdf.enabled"),
    "ssb": ("output.ssb.enabled", "output.ssb.netcdf.enabled"),
}
NAMES = tuple(SWITCHES)
OFF = (
    {"output.minimal.enabled": "false", "output.diet.composition.enabled": "false"}  # diet: 2.6c
    | {k: "false" for keys in SWITCHES.values() for k in keys}
)
#: Every family on, with length classes that leave schools below the first and beyond the last.
ALL = (
    OFF
    | {keys[0]: "true" for keys in SWITCHES.values()}
    | {
        "output.distrib.bysize.min": "2",
        "output.distrib.bysize.max": "40",
        "output.distrib.bysize.incr": "4",
    }
)
TIER_L = 1e-12
BY_AGE, BY_SIZE, PRESENT = NAMES[:2], NAMES[2:4], ("mean_tl", "mean_size")


def _row(record: StepOutput, name: str, n_species: int) -> np.ndarray:
    """A Python record's family as the row the C# engine writes."""
    v = getattr(record, name)
    assert v is not None, name
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


def _same(got, want) -> bool:
    """Bit for bit, where NaN (an absent species) is one value whatever its sign bit."""
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if got.shape != want.shape or not np.array_equal(np.isnan(got), np.isnan(want)):
        return False
    live = ~np.isnan(got)
    return got[live].tobytes() == want[live].tobytes()


# ------------------------------------------------------------------------ A. the contract


def _config(tmp_path, extra: dict):
    return EngineConfig.from_dict(_raw("eec_full", tmp_path, 1, extra))


@pytest.mark.parametrize("name", NAMES)
def test_each_switch_turns_on_its_family_and_nothing_else(name, tmp_path):
    bit = getattr(build, f"FAMILY_{name.upper()}")
    for key in SWITCHES[name]:
        config = _config(tmp_path, OFF | {key: "true"})
        assert output_families(config) == build.FAMILY_STAGE | bit, key
        assert set(family_shapes(config)) == {*build.STAGE_BUFFERS, name}, key
    assert output_families(_config(tmp_path, OFF)) == build.FAMILY_STAGE
    assert output_families(_config(tmp_path, ALL | {"output.minimal.enabled": "true"})) == 0


def test_the_bits_are_the_engines(tmp_path):
    """The C# ``OutputFamilies`` values, pinned here so that a renumbering on one side fails."""
    bits = {name: getattr(build, f"FAMILY_{name.upper()}") for name in NAMES}
    assert bits == dict(zip(NAMES, (2, 4, 8, 16, 32, 64, 128, 256), strict=True))
    assert build.FAMILY_STAGE == 1
    assert output_families(_config(tmp_path, ALL)) == 511


def test_the_tables_exist_exactly_with_their_families(tmp_path):
    raw = _raw("eec_full", tmp_path, 1, ALL)
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    b = build_bundle(config, raw, grid, seed=11)
    offsets, edges = b.get("output.age_bin_offsets"), b.get("output.size_edges")
    assert isinstance(offsets, np.ndarray) and isinstance(edges, np.ndarray)
    assert offsets.dtype == np.int32 and np.array_equal(offsets, age_bin_offsets(config))
    assert np.array_equal(edges, [2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0])
    assert np.array_equal(edges, size_edges(config))

    only_size = _raw("eec_full", tmp_path, 1, OFF | {"output.abundance.bysize.enabled": "true"})
    config, grid, *_ = PythonEngine()._prepare_run(only_size, 11)
    b = build_bundle(config, only_size, grid, seed=11)
    assert "output.size_edges" in b.names() and "output.age_bin_offsets" not in b.names()
    minimal = _raw("eec_full", tmp_path, 1)
    config, grid, *_ = PythonEngine()._prepare_run(minimal, 11)
    names = build_bundle(config, minimal, grid, seed=11).names()
    assert "output.size_edges" not in names and "output.age_bin_offsets" not in names


def test_the_buffers_have_the_shapes_the_reference_returns(tmp_path):
    """Every family's record has the shape of the field ``_collect_outputs`` returns for it, on
    an empty state (shapes depend on the config alone)."""
    raw = _raw("eec_full", tmp_path, 1, ALL)
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    from osmose.engine.state import SchoolState

    out = _collect_outputs(SchoolState.create(0), config, 0)
    shapes = family_shapes(config)
    assert list(shapes) == [*build.STAGE_BUFFERS, *NAMES]
    nf = config.n_species
    for name in NAMES:
        want = np.zeros(nf) if name in PRESENT else _row(out, name, nf)
        assert shapes[name] == want.shape, name
    buffers = allocate_results(build_bundle(config, raw, grid, seed=11), config)
    assert buffers.age_bin_offsets is not None
    for name, shape in shapes.items():
        assert buffers.families[name].shape == (len(buffers.record_step), *shape), name
    lengths = [len(out.biomass_by_age[sp]) for sp in range(nf)]
    assert np.array_equal(np.diff(buffers.age_bin_offsets), lengths)

    # Lifespans that are not whole years: the classes are ceil(lifespan / steps per year) + 1.
    odd = copy.copy(config)
    odd.lifespan_dt = config.lifespan_dt + np.arange(len(config.lifespan_dt)) % 7 + 1
    assert (odd.lifespan_dt[:nf] % odd.n_dt_per_year != 0).any()
    out = _collect_outputs(SchoolState.create(0), odd, 0)
    lengths = [len(out.biomass_by_age[sp]) for sp in range(nf)]
    assert np.array_equal(np.diff(age_bin_offsets(odd)), lengths)
    assert not np.array_equal(age_bin_offsets(odd), buffers.age_bin_offsets)


def test_step_outputs_rebuild_the_dicts_and_leave_out_absent_species(tmp_path):
    raw = _raw("eec_full", tmp_path, 1, ALL)
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    buffers = allocate_results(build_bundle(config, raw, grid, seed=11), config)
    nf, offsets = config.n_species, buffers.age_bin_offsets
    assert offsets is not None
    rng = np.random.default_rng(1)
    for a in buffers.families.values():
        a[:] = rng.random(a.shape)
    buffers.families["mean_tl"][0, 3] = np.nan
    buffers.families["mean_size"][0, :] = np.nan
    out = _step_outputs(buffers, 1)[0]
    f = {name: a[0] for name, a in buffers.families.items()}

    assert out.biomass_by_age is not None and out.abundance_by_age is not None
    assert sorted(out.biomass_by_age) == list(range(nf))
    for sp in range(nf):
        assert np.array_equal(
            out.biomass_by_age[sp], f["biomass_by_age"][offsets[sp] : offsets[sp + 1]]
        )
        assert np.array_equal(
            out.abundance_by_age[sp], f["abundance_by_age"][offsets[sp] : offsets[sp + 1]]
        )
    assert out.biomass_by_size is not None and out.abundance_by_size is not None
    assert all(
        np.array_equal(out.biomass_by_size[sp], f["biomass_by_size"][sp]) for sp in range(nf)
    )
    assert all(
        np.array_equal(out.abundance_by_size[sp], f["abundance_by_size"][sp]) for sp in range(nf)
    )
    assert out.mean_tl is not None and out.mean_size is not None
    assert sorted(out.mean_tl) == [sp for sp in range(nf) if sp != 3]
    assert all(out.mean_tl[sp] == f["mean_tl"][sp] for sp in out.mean_tl)
    assert out.mean_size == {}  # a dict, with every species absent
    assert out.ssb is not None and out.yield_n is not None
    assert np.array_equal(out.ssb, f["ssb"]) and np.array_equal(out.yield_n, f["yield_n"])

    raw = _raw("eec_full", tmp_path, 1, OFF)
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    off = _step_outputs(allocate_results(build_bundle(config, raw, grid, seed=11), config), 1)[0]
    assert off.abundance_by_stage is not None  # the stage family is part of every full run
    for name in NAMES:
        assert getattr(off, name) is None, name


def _fields_equal(a, b) -> bool:
    """Two ``StepOutput`` field values: arrays, dicts of arrays or floats, scalars, None."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and sorted(a) == sorted(b)
            and all(_fields_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.tobytes() == b.tobytes()
    return type(a) is type(b) and a == b


def test_records_of_a_python_run_survive_the_buffers_and_both_consumers(tmp_path):
    """The Python half of the transport, on what the consumers actually read: a real run's
    records are packed into the result buffers as the C# engine fills them and rebuilt by
    ``_step_outputs``. Every ``StepOutput`` field must come back as it was, so a field the
    Python engine fills and this path drops is caught; and ``OsmoseResults.from_outputs`` and
    ``write_outputs`` must give the same frames and the same files from both lists."""
    import dataclasses

    import pandas as pd

    from osmose.engine.output import write_outputs
    from osmose.results import OsmoseResults

    extra = ALL | {
        "output.recordfrequency.ndt": "6",
        "output.step0.include": "true",
        "output.diet.composition.enabled": "false",  # the diet family is task 2.6c
        "output.diet.pressure.enabled": "false",
    }
    raw = _raw("eec_full", tmp_path, 1, extra)
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    python = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    buffers = allocate_results(build_bundle(config, raw, grid, seed=11), config)
    assert len(buffers.record_step) == len(python) == 5
    nf = config.n_species
    for i, r in enumerate(python):
        buffers.record_step[i], buffers.window_steps[i] = r.step, r.window_steps
        buffers.biomass[i], buffers.abundance[i] = r.biomass, r.abundance
        buffers.mortality[i], buffers.yield_[i] = r.mortality_by_cause, r.yield_by_species
        for name in build.STAGE_BUFFERS:
            buffers.families[name][i] = getattr(r, name)
        for name in NAMES:
            buffers.families[name][i] = _row(r, name, nf)
    rebuilt = _step_outputs(buffers, len(python))

    for i, (a, b) in enumerate(zip(python, rebuilt, strict=True)):
        for f in dataclasses.fields(StepOutput):
            assert _fields_equal(getattr(a, f.name), getattr(b, f.name)), (i, f.name)
    last = python[-1]
    assert last.mean_tl and last.mean_size and last.ssb is not None and last.ssb.sum() > 0
    assert len(last.mean_tl) < nf or len(python[1].mean_tl or {}) < nf  # a species was absent

    frames = [OsmoseResults.from_outputs(o, config, grid)._csv_cache for o in (python, rebuilt)]
    assert sorted(frames[0]) == sorted(frames[1]) and len(frames[0]) > 10
    for key in frames[0]:
        a, b = frames[0][key], frames[1][key]
        if isinstance(a, dict):
            assert sorted(a) == sorted(b), key
            for sub in a:
                pd.testing.assert_frame_equal(a[sub], b[sub], check_exact=True, obj=f"{key}/{sub}")
        else:
            pd.testing.assert_frame_equal(a, b, check_exact=True, obj=str(key))

    for name, outputs in (("python", python), ("rebuilt", rebuilt)):
        write_outputs(outputs, tmp_path / name, config, grid=grid)
    files = [
        sorted(p.relative_to(tmp_path / n) for p in (tmp_path / n).rglob("*") if p.is_file())
        for n in ("python", "rebuilt")
    ]
    assert files[0] == files[1] and len(files[0]) > 10
    for rel in files[0]:
        assert (tmp_path / "python" / rel).read_bytes() == (
            tmp_path / "rebuilt" / rel
        ).read_bytes(), rel


# ------------------------------------------------------------------------ B. windows

UNFED_SOME, UNFED_WINDOW, NO_SURVIVOR = (2, 3), (5, 6, 7, 8, 9), 6


def _edit(states: list, config) -> dict:
    """Edits the states and says where (see the module docstring, B)."""
    edges, n_sp, ndt = size_edges(config), config.n_species, config.n_dt_per_year
    st = states[0]
    age_years = st.age_dt / ndt
    fed = (
        (st.species_id < n_sp)
        & (st.trophic_level > 0)
        & (st.abundance > 0)
        & (age_years >= config.output_cutoff_age[st.species_id])
    )
    order = np.argsort(-np.bincount(st.species_id[fed], minlength=n_sp))
    some, window, none = (int(sp) for sp in order[:3])
    assert np.bincount(st.species_id[fed], minlength=n_sp)[none] > 0
    for t, s in enumerate(states):
        focal = np.flatnonzero(s.species_id < n_sp)
        s.length[focal[0::7]] = edges[1]  # exactly on an edge
        s.length[focal[1::7]] = edges[-1]  # exactly on the last edge
        s.length[focal[2::7]] = edges[0] - 0.5  # below the first class
        s.length[focal[3::7]] = edges[-1] + 1000.0  # far beyond the last
        old = focal[4::7]  # ages beyond the species' last class
        s.age_dt[old] = config.lifespan_dt[s.species_id[old]] + 3 * ndt
        if t in UNFED_SOME:
            s.trophic_level[s.species_id == some] = 0.0
        if t in UNFED_WINDOW:
            s.trophic_level[s.species_id == window] = 0.0
        if t == NO_SURVIVOR:
            s.abundance[s.species_id == none] = 0.0
        states[t] = s.replace(biomass=s.abundance * s.weight)
    return {"some": some, "window": window, "none": none}


@pytest.fixture(scope="module")
def family_states(tmp_path_factory) -> dict:
    out = {}
    for demo, step in (("tutorial", 129), ("eec_full", 30)):
        d = tmp_path_factory.mktemp(demo)
        config, grid, raw, st = _post_reproduction_state(_raw(demo, d, step // 24 + 1, ALL), step)
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
    family_states,
    demo,
    freq,
    step0,
    tmp_path,
):
    config, grid, raw, states, where = family_states[demo]
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = freq, step0
    assert output_families(cfg) == 511
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(states, tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    want = _python_records(states, cfg, freq, step0)
    nf = cfg.n_species
    assert np.array_equal(res.get("out.record_step"), [r.step for r in want])
    for name in NAMES:
        got = res.get(f"out.{name}")
        assert got.shape[0] == len(want), name
        for i, r in enumerate(want):
            assert _same(got[i], _row(r, name, nf)), (name, i)
    for cs_name, py_name in FAMILIES:  # the minimal families are collected as before
        got = res.get(f"out.{cs_name}")
        for i, r in enumerate(want):
            assert got[i].tobytes() == np.asarray(getattr(r, py_name)).tobytes(), (cs_name, i)

    # The edits are inside the compared records.
    edges = size_edges(cfg)
    st = states[1]
    focal = st.species_id < nf
    assert (st.length[focal] == edges[1]).any() and (st.length[focal] == edges[-1]).any()
    assert (st.length[focal] < edges[0]).any() and (st.length[focal] > edges[-1] + 1).any()
    assert (st.age_dt[focal] > config.lifespan_dt[st.species_id[focal]]).any()
    tl, size = res.get("out.mean_tl"), res.get("out.mean_size")
    if (freq, step0) == (5, False):  # windows are states 0-4, 5-9 and 10-11
        assert not np.isnan(tl[0, where["some"]])  # present in three of the five steps
        assert np.isnan(tl[1, where["window"]])  # absent from the whole record
        assert not np.isnan(tl[2, where["window"]])
        assert not np.isnan(size[1, where["none"]])  # one step without survivors: still present
        # The mean is over the steps that have the value, not over the window.
        steps = [_collect_outputs(s, cfg, 0).mean_tl for s in states[:5]]
        assert [where["some"] in d for d in steps] == [True, True, False, False, True]
        present = [d[where["some"]] for d in steps if where["some"] in d]
        assert tl[0, where["some"]] == float(np.mean(present))
        assert tl[0, where["some"]] != pytest.approx(sum(present) / 5, rel=0.01)
    if freq == 1:
        assert np.isnan(tl[list(UNFED_SOME), where["some"]]).all()
        assert np.isnan(tl[list(UNFED_WINDOW), where["window"]]).all()
        assert np.isnan(size[NO_SURVIVOR, where["none"]])
        assert np.isnan(tl[NO_SURVIVOR, where["none"]])
        assert not np.isnan(size[NO_SURVIVOR + 1, where["none"]])


def test_only_the_families_that_are_on_are_written(engine_ok, family_states, tmp_path):  # noqa: F811
    config, grid, raw, states, _ = family_states["eec_full"]
    cfg = copy.copy(config)
    cfg.output_record_frequency = 4
    on = ("abundance_by_age", "mean_size", "ssb")
    cfg.output_biomass_byage = cfg.output_biomass_bysize = cfg.output_abundance_bysize = False
    cfg.output_meantl = cfg.output_yield_abundance = False
    assert set(family_shapes(cfg)) == {*build.STAGE_BUFFERS, *on}
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(states, tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    want = _python_records(states, cfg, 4, False)
    for name in NAMES:
        if name in on:
            for i, r in enumerate(want):
                assert _same(res.get(f"out.{name}")[i], _row(r, name, cfg.n_species)), (name, i)
        else:
            assert f"out.{name}" not in res.names() and getattr(want[0], name) is None, name


@pytest.mark.parametrize(
    ("name", "key"), [(name, key) for name, keys in SWITCHES.items() for key in keys]
)
def test_each_switch_alone_gives_the_field_the_reference_gives(
    engine_ok,  # noqa: F811
    family_states,
    name,
    key,
    tmp_path,
):
    """One config key on, from the raw config through both engines. The Python engine returns
    exactly one of the eight fields, and the C# engine writes exactly that family: the two
    sides' bit numbers are tied through what they produce, not through a table in this file."""
    _, grid, raw, states, _ = family_states["eec_full"]
    raw = raw | OFF | {key: "true", "output.recordfrequency.ndt": "2"}
    cfg = EngineConfig.from_dict(raw)
    reference = _collect_outputs(states[0], cfg, 0)
    assert {n for n in NAMES if getattr(reference, n) is not None} == {name}
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(states[:2], tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    minimal = {"record_step", "window_steps", *(cs_name for cs_name, _ in FAMILIES)}
    written = {n.removeprefix("out.") for n in res.names()} - minimal
    assert written == {*build.STAGE_BUFFERS, name}
    want = _python_records(states[:2], cfg, 2, False)
    assert len(want) == 1 and _same(res.get(f"out.{name}")[0], _row(want[0], name, cfg.n_species))


# ------------------------------------------------------------------- C. one whole step


@pytest.mark.parametrize(
    ("demo", "edge_ages"), [("bay_of_biscay", False), ("eec_full", False), ("eec_full", True)]
)
def test_one_production_step_equals_the_python_engine(
    engine_ok,  # noqa: F811
    demo,
    edge_ages,
    tmp_path,
    record_property,
):
    step = STEP_CASES[demo]
    got = _capture_step(_raw(demo, tmp_path, step // 24 + 1, ALL), step, edge_ages)
    config, out = got["config"], got["output"]
    build_bundle(config, got["raw"], got["grid"], seed=11).save(tmp_path / "model.bin")
    _write_states([got["entry"]], tmp_path / "entry.bin")
    _write_draws(got["record"], tmp_path / "draws.bin")
    res = _probe(
        "Step", tmp_path / "model.bin", tmp_path / "entry.bin", tmp_path / "draws.bin", step,
        tmp_path / "out.bin",
    )  # fmt: skip
    worst = {}
    for name in NAMES:
        mine, want = res.get(f"out.{name}")[0], _row(out, name, config.n_species)
        assert np.array_equal(np.isnan(mine), np.isnan(want)), name
        live = ~np.isnan(want)
        worst[name] = _worst(mine[live], want[live])
    record_property("worst_relative_error", worst)
    assert all(v <= TIER_L for v in worst.values()), worst
    # The compared values are not trivial: eggs were laid this step, and they are in the
    # distributions (class 0) although the cutoff or their size keeps them out elsewhere.
    by_age = _row(out, "abundance_by_age", config.n_species)
    assert by_age[age_bin_offsets(config)[:-1]].sum() > 0 and by_age.sum() > 0
    assert _row(out, "biomass_by_size", config.n_species)[:, 0].sum() > 0
    assert out.mean_size and out.yield_n is not None and out.yield_n.sum() > 0
    if demo == "eec_full":
        assert out.ssb is not None and out.ssb.sum() > 0 and out.mean_tl


# --------------------------------------------------------------------- D. whole runs


def _run(raw: dict, tmp_path, name: str, threads: int = 1):
    config, grid, *_ = PythonEngine()._prepare_run(raw, 11)
    d = tmp_path / name
    d.mkdir()
    build_bundle(config, raw, grid, seed=11, threads=threads).save(d / "model.bin")
    return _probe("Run", d / "model.bin", d / "out.bin"), config


@pytest.mark.parametrize(("demo", "leading"), [("eec_full", 2), ("bay_of_biscay", 4)])
def test_whole_run_leading_records_equal_python(engine_ok, demo, leading, tmp_path):  # noqa: F811
    raw = _raw(demo, tmp_path, 1, ALL | {"output.recordfrequency.ndt": "1"})
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    python = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    res, _ = _run(raw, tmp_path, "full")
    assert np.array_equal(res.get("out.record_step"), [r.step for r in python])
    for i in range(leading):  # no draw has acted yet: seeding, then larva deaths on unfed eggs
        for name in NAMES:
            mine, want = res.get(f"out.{name}")[i], _row(python[i], name, config.n_species)
            assert np.array_equal(np.isnan(mine), np.isnan(want)), (name, i)
            live = ~np.isnan(want)
            assert _worst(mine[live], want[live]) <= 1e-9, (name, i)
    last = leading - 1
    assert _row(python[last], "abundance_by_age", config.n_species).sum() > 0  # the first eggs
    if demo == "bay_of_biscay":
        assert python[last].mean_size  # no cutoff age: the eggs have a mean length
    else:
        assert python[last].mean_size == {}  # EEC's 0.5 yr cutoff: every species still absent


@pytest.mark.parametrize("demo", ["eec_full", "tutorial"])
def test_a_full_run_leaves_the_minimal_families_as_they_were(engine_ok, demo, tmp_path):  # noqa: F811
    extra = {"output.recordfrequency.ndt": "6", "output.step0.include": "true"}
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    minimal, _ = _run(_raw(demo, tmp_path / "a", 2, extra), tmp_path, "minimal")
    full, config = _run(_raw(demo, tmp_path / "b", 2, ALL | extra), tmp_path, "full")
    nf = config.n_species
    assert not any(f"out.{name}" in minimal.names() for name in NAMES)
    for name in ("record_step", "window_steps", "biomass", "abundance", "mortality", "yield"):
        assert minimal.get(f"out.{name}").tobytes() == full.get(f"out.{name}").tobytes(), name

    # By age and by size bin the same schools: the totals agree, class by class they do not.
    offsets = age_bin_offsets(config)
    by_age = np.add.reduceat(full.get("out.abundance_by_age"), offsets[:-1], axis=1)
    by_size = full.get("out.abundance_by_size").sum(axis=2)
    assert by_age.shape == by_size.shape == (full.get("out.abundance").shape[0], nf)
    assert np.allclose(by_age, by_size, rtol=1e-9, atol=0.0) and by_age[1:].sum() > 0
    biomass_age = np.add.reduceat(full.get("out.biomass_by_age"), offsets[:-1], axis=1)
    assert np.allclose(biomass_age, full.get("out.biomass_by_size").sum(axis=2), rtol=1e-9)

    # The distributions bin every school, eggs included; the abundance output applies the
    # cutoff age. So they legitimately disagree (CLAUDE.md), and only one way.
    abundance = full.get("out.abundance")[:, :nf]
    assert (by_age >= abundance * (1 - 1e-12)).all()
    if demo == "tutorial":  # a 0.5 yr cutoff: the young of the year are most of the count
        assert (by_age[1:] > 2 * abundance[1:]).any()
    # SSB is part of the biomass of ALL schools. It is not bounded by the biomass output, which
    # applies the cutoff age: a school can be long enough to spawn before it is old enough to
    # be counted (both demos have such schools at this seed).
    assert (full.get("out.ssb") <= biomass_age * (1 + 1e-12)).all()
    assert full.get("out.ssb")[1:].sum() > 0
    yield_n, deaths = full.get("out.yield_n"), full.get("out.mortality")[:, :, 3]
    assert np.allclose(yield_n, deaths, rtol=1e-12, atol=0.0)  # the same sum, as a family
    for name in PRESENT:
        v = full.get(f"out.{name}")
        assert np.isnan(v[0]).all()  # the initial record has no school
        assert not np.isnan(v[1:]).all() and (v[~np.isnan(v)] > 0).all()


@pytest.mark.parametrize("demo", ["eec_full", "tutorial"])
def test_the_family_records_do_not_depend_on_the_thread_count(engine_ok, demo, tmp_path):  # noqa: F811
    raw = _raw(demo, tmp_path, 2, ALL)
    runs = [_run(raw, tmp_path, f"t{t}", threads=t)[0] for t in (1, 2, 14)]
    for name in NAMES:
        one = runs[0].get(f"out.{name}").tobytes()
        assert all(r.get(f"out.{name}").tobytes() == one for r in runs[1:]), name
    assert runs[0].get("out.abundance_by_age").sum() > 0
