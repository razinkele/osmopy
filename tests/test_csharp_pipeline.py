"""Plan task 1.8: the C# step pipeline against the Python engine's own sequence and arithmetic.

A. One whole step from mortality entry to the step's outputs. A real run is spied at the target
   step: its ``_mortality`` runs with the batched kernel replaced by the per-cell oracle on
   recorded draws, ``simulate`` itself then strips, floors, grows, ages, reproduces and collects
   (so the ORDER is production's), and the ``_collect_outputs`` call yields the reference state
   and step output. NumPy's ``exp`` is swapped for the C library's in ``natural`` and ``growth``
   for that step. ``PipelineProbe.Step`` replays the step in C# (regroup, mortality, floor,
   growth, aging, reproduction, collection). The C# store is regrouped, so its sums run in a
   different order: tier L, worst error recorded.
B. Record windows: K states are collected as K steps and folded; the reference is
   ``_collect_outputs`` per state and ``_average_step_outputs`` with ``simulate``'s flush rule.
   Same school order on both sides, so bit for bit. Baltic's 0.5 yr cutoff, partial windows and
   ``output.step0.include`` are covered.
C. Whole runs (``PipelineProbe.Run``): the record steps and window lengths equal a Python run's,
   and the leading records, which no draw influences yet (verified on the Python side with two
   seeds), equal Python's.

The test needs the built engine. It skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import copy
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import osmose.engine.processes.growth as growth_mod
import osmose.engine.processes.mortality as mort
import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.processes import natural
from osmose.engine.simulate import _average_step_outputs, _collect_outputs
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from tests._tutorial_config import build_config
from tests.test_csharp_mortality_step import _LibmNumpy, _replay_kernel, _write_draws

ROOT = Path(__file__).resolve().parents[1]
STEP_CASES = {"eec_full": 30, "bay_of_biscay": 9}  # tutorial: its SR curve and gates are phase 2
TIER_L = 1e-12
FAMILIES = (("biomass", "biomass"), ("abundance", "abundance"), ("mortality", "mortality_by_cause"), ("yield", "yield_by_species"))  # fmt: skip


class _Stop(Exception):
    pass


def _raw(demo: str, d: Path, nyear: int, extra: dict | None = None) -> dict:
    if demo == "tutorial":
        raw = dict(build_config(d, n_year=nyear))
    else:
        raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, d)["config_file"])))
    return (
        raw
        | {"output.minimal.enabled": "true", "simulation.time.nyear": str(nyear)}
        | (extra or {})
    )


def _probe(method: str, *args) -> Bundle:
    out = Path(args[-1])
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        f"f = bridge.load().Diagnostics.PipelineProbe.{method}; "
        "a = [int(x) if x.lstrip('-').isdigit() else x for x in sys.argv[2:]]; f(*a)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return Bundle.load(out)


def _write_states(states: list, path: Path, extra: dict | None = None) -> None:
    """K states concatenated, with ``state.offsets`` (K + 1). ``extra`` adds arrays by name
    (``state.diet``: the schools' diet rows, concatenated like the states)."""
    b = Bundle(1)
    for name, array in (extra or {}).items():
        b.set_array(name, np.ascontiguousarray(array))
    for name, dt in [
        ("species_id", np.int32), ("age_dt", np.int32), ("length", np.float64), ("weight", np.float64),
        ("abundance", np.float64), ("trophic_level", np.float64), ("cell_x", np.int32),
        ("cell_y", np.int32), ("is_background", np.uint8), ("is_egg", np.uint8), ("is_out", np.uint8),
        ("from_seeding", np.uint8), ("first_feeding_age_dt", np.int32), ("starvation_rate", np.float64),
    ]:  # fmt: skip
        b.set_array(
            f"state.{name}",
            np.ascontiguousarray(np.concatenate([getattr(s, name) for s in states]), dtype=dt),
        )
    b.set_array(
        "state.n_dead",
        np.ascontiguousarray(np.concatenate([s.n_dead for s in states]), dtype=np.float64),
    )
    b.set_array("state.offsets", np.cumsum([0] + [len(s) for s in states]).astype(np.int64))
    b.save(path)


@pytest.fixture(scope="module")
def engine_ok():
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")


# ------------------------------------------------------------------- A. one whole step


def _capture_step(raw: dict, step: int, edge_ages: bool = False, edit=None) -> dict:
    """One production step, captured (see the module docstring, A). ``edit(state, config)``
    changes the state at mortality entry, before it is recorded. With the diet output on, the
    capture also holds the kernel's per-school diet rows (``diet``) and the state as mortality
    left it, background schools included (``after_mortality``)."""
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    got: dict = {}
    real_mort, real_collect = sim._mortality, sim._collect_outputs
    collect_sig = inspect.signature(real_collect)
    saved_np = (natural.np, growth_mod.np)
    kernels = ("_mortality_all_cells_parallel", "_mortality_all_cells_numba")
    saved_kernels = {k: getattr(mort, k) for k in kernels}

    def mort_spy(state, resources, config_, rng_, grid_, step=0, species_rngs=None, ctx=None):
        if step != target:
            return real_mort(
                state,
                resources,
                config_,
                rng_,
                grid_,
                step=step,
                species_rngs=species_rngs,
                ctx=ctx,
            )
        assert (state.abundance > 0).all() and not state.n_dead.any()
        if edge_ages:  # schools at the lifespan edge: aging must run BEFORE the age increment
            adults = np.flatnonzero(~state.is_egg)
            lifespan = config_.lifespan_dt[state.species_id[adults]]
            state.age_dt[adults[0::3]] = lifespan[0::3] - 2  # survive this step
            state.age_dt[adults[1::3]] = lifespan[1::3] - 1  # die of old age this step
        if edit is not None:
            edit(state, config_)
        got["entry"] = copy.deepcopy(state)
        record: list = []
        got["record"] = record
        natural.np = growth_mod.np = _LibmNumpy()  # restored when the step's outputs are collected
        replay = _replay_kernel(record)
        for k in kernels:
            setattr(mort, k, replay)
        try:
            after = real_mort(
                state,
                resources,
                config_,
                rng_,
                grid_,
                step=step,
                species_rngs=species_rngs,
                ctx=ctx,
            )
        finally:
            for k, fn in saved_kernels.items():
                setattr(mort, k, fn)
        if ctx is not None and ctx.diet_tracking_enabled and ctx.diet_matrix is not None:
            got["diet"] = ctx.diet_matrix[: ctx._diet_active_rows].copy()
            got["after_mortality"] = copy.deepcopy(after)
        return after

    def collect_spy(*args, **kwargs):
        out = real_collect(*args, **kwargs)
        a = collect_sig.bind(*args, **kwargs).arguments
        if a["step"] == target:
            got["final"], got["output"] = copy.deepcopy(a["state"]), out
            raise _Stop
        return out

    target = step
    sim._mortality, sim._collect_outputs = mort_spy, collect_spy
    try:
        sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    except _Stop:
        pass
    finally:
        sim._mortality, sim._collect_outputs = real_mort, real_collect
        natural.np, growth_mod.np = saved_np
        for k, fn in saved_kernels.items():
            setattr(mort, k, fn)
    assert len(got["record"]) == config.mortality_subdt
    got.update(config=config, grid=grid, raw=raw, step=step)
    return got


def _worst(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(diff == 0, 0.0, diff / np.abs(b))
    return float(rel.max()) if rel.size else 0.0


def _cap_at_the_median(state, config) -> None:
    """Set each focal species' ``lmax`` to the median length of its schools, so that the cap
    binds on half of them in this step's growth (no shipped config sets ``species.lmax``)."""
    lmax = np.array(config.lmax, dtype=np.float64)
    for sp in range(config.n_species):
        mine = state.species_id == sp
        if mine.any():
            lmax[sp] = float(np.median(state.length[mine]))
    config.lmax = lmax


IMMOBILE = 3  # the EEC species given movement method "none" in the "immobile" variant
VARIANTS = {
    None: ({}, None),
    "lmax": ({}, _cap_at_the_median),
    "immobile": ({f"movement.distribution.method.sp{IMMOBILE}": "none"}, None),
}


@pytest.mark.parametrize(
    ("demo", "edge_ages", "variant"),
    [
        ("bay_of_biscay", False, None),
        ("eec_full", False, None),
        ("eec_full", True, None),
        ("eec_full", False, "lmax"),
        ("eec_full", False, "immobile"),
    ],
)
def test_one_step_from_mortality_entry_equals_production(
    engine_ok, demo, edge_ages, variant, tmp_path, record_property
):
    step = STEP_CASES[demo]
    extra, edit = VARIANTS[variant]
    got = _capture_step(_raw(demo, tmp_path, step // 24 + 1, extra), step, edge_ages, edit=edit)
    build_bundle(got["config"], got["raw"], got["grid"], seed=11).save(tmp_path / "model.bin")
    _write_states([got["entry"]], tmp_path / "entry.bin")
    _write_draws(got["record"], tmp_path / "draws.bin")
    res = _probe(
        "Step",
        tmp_path / "model.bin",
        tmp_path / "entry.bin",
        tmp_path / "draws.bin",
        step,
        tmp_path / "out.bin",
    )
    final, out = got["final"], got["output"]
    assert len(res.get("s.species_id")) == len(final)  # same schools, new eggs included
    for name in ("species_id", "age_dt", "cell_x", "cell_y", "is_egg", "is_out"):
        assert np.array_equal(
            res.get(f"s.{name}"),
            np.asarray(getattr(final, name)).astype(res.get(f"s.{name}").dtype),
        ), name
    worst = {}
    for name in ("abundance", "length", "weight", "trophic_level", "starvation_rate", "n_dead"):
        worst[name] = _worst(res.get(f"s.{name}"), getattr(final, name))
    for cs_name, py_name in FAMILIES:
        worst[f"out.{cs_name}"] = _worst(res.get(f"out.{cs_name}")[0], getattr(out, py_name))
    record_property("worst_relative_error", worst)
    print(f"{demo}: {worst}")
    assert all(v <= TIER_L for v in worst.values()), worst
    new = len(final) - len(got["entry"])
    assert new > 0 and (final.abundance == 0).any()  # eggs laid; someone floored or aged
    if edge_ages:
        aged = final.n_dead[: len(got["entry"]), 7] > 0
        assert aged.any() and not aged[np.flatnonzero(~got["entry"].is_egg)[0::3]].any()
    assert out.mortality_by_cause.sum() > 0 and out.biomass.sum() > 0
    entry, config = got["entry"], got["config"]
    if variant == "lmax":  # the cap binds, and nothing is longer than it
        old = final.species_id[: len(entry)]
        assert np.isfinite(config.lmax[: config.n_species]).all()
        assert (final.length[: len(entry)] == config.lmax[old]).sum() > 100
        assert (final.length[: len(entry)] <= config.lmax[old]).all()
    if variant == "immobile":  # the species is there, never placed, and not flagged out
        assert list(config.movement_method)[IMMOBILE] == "none"
        mine = entry.species_id == IMMOBILE
        assert mine.sum() > 10 and (entry.cell_x[mine] < 0).all() and not entry.is_out[mine].any()
        assert (entry.cell_x[~mine] >= 0).any()


# ---------------------------------------------------------------------- B. windows


def _post_reproduction_state(raw: dict, step: int):
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    got: dict = {}
    real = sim._collect_outputs
    sig = inspect.signature(real)

    def spy(*args, **kwargs):
        a = sig.bind(*args, **kwargs).arguments
        if a["step"] == step:
            got["state"] = copy.deepcopy(a["state"])
            raise _Stop
        return real(*args, **kwargs)

    sim._collect_outputs = spy
    try:
        sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    except _Stop:
        pass
    finally:
        sim._collect_outputs = real
    return config, grid, raw, got["state"]


def _python_records(states, config, freq: int, step0: bool) -> list:
    """``simulate``'s recording rule over a list of per-step states."""
    records, acc = [], []
    first = 1 if step0 else 0
    if step0:
        records.append(_collect_outputs(states[0], config, step=-1))
    n_steps = len(states) - first
    for step in range(n_steps):
        acc.append(_collect_outputs(states[first + step], config, step))
        if (step + 1) % freq == 0:
            records.append(_average_step_outputs(acc, freq, step))
            acc = []
    if acc:
        records.append(_average_step_outputs(acc, len(acc), n_steps - 1))
    return records


@pytest.fixture(scope="module")
def window_states(tmp_path_factory) -> dict:
    out = {}
    for demo, step in (("tutorial", 129), ("eec_full", 30)):
        d = tmp_path_factory.mktemp(demo)
        config, grid, raw, st = _post_reproduction_state(_raw(demo, d, step // 24 + 1), step)
        rng = np.random.default_rng(7)
        states = []
        for _ in range(12):  # twelve "steps": the same schools, perturbed abundance and deaths
            s = copy.deepcopy(st)
            s.abundance[:] *= rng.uniform(0.5, 1.5, len(s))
            s.n_dead[:] *= rng.uniform(0.5, 1.5, s.n_dead.shape)
            states.append(s.replace(biomass=s.abundance * s.weight))
        out[demo] = (config, grid, raw, states, d)
    return out


@pytest.mark.parametrize("demo", ["tutorial", "eec_full"])
@pytest.mark.parametrize(("freq", "step0"), [(5, False), (5, True), (1, False), (12, False)])
def test_windows_equal_average_step_outputs(engine_ok, window_states, demo, freq, step0, tmp_path):
    config, grid, raw, states, _ = window_states[demo]
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = freq, step0
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(states, tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    want = _python_records(states, cfg, freq, step0)
    assert np.array_equal(res.get("out.record_step"), [r.step for r in want])
    assert np.array_equal(res.get("out.window_steps"), [r.window_steps for r in want])
    for cs_name, py_name in FAMILIES:
        got_v = res.get(f"out.{cs_name}")
        for i, r in enumerate(want):
            assert (
                got_v[i].tobytes() == np.asarray(getattr(r, py_name), dtype=np.float64).tobytes()
            ), (cs_name, i)
    if demo == "tutorial":  # the 0.5 yr cutoff binds: some schools are counted in deaths only
        st = states[0]
        young = st.age_dt / config.n_dt_per_year < config.output_cutoff_age[st.species_id]
        assert young.any() and (~young).any()


# --------------------------------------------------------------------- C. whole runs


@pytest.mark.parametrize(("demo", "leading"), [("eec_full", 2), ("bay_of_biscay", 4)])
def test_whole_run_records_and_leading_values_equal_python(
    engine_ok, demo, leading, tmp_path, record_property
):
    raw = _raw(demo, tmp_path, 1, {"output.recordfrequency.ndt": "1"})
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    python = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    build_bundle(config, raw, grid, seed=11).save(tmp_path / "model.bin")
    res = _probe("Run", tmp_path / "model.bin", tmp_path / "out.bin")
    status, steps, records, peak = res.get("out.summary")
    assert (status, steps, records) == (0, config.n_steps, len(python)) and peak > 0
    assert np.array_equal(res.get("out.record_step"), [r.step for r in python])
    assert np.array_equal(res.get("out.window_steps"), [r.window_steps for r in python])
    worst = {}
    for cs_name, py_name in FAMILIES:
        for i in range(leading):  # no draw has acted yet: seeding, then larva deaths on unfed eggs
            worst[f"{cs_name}[{i}]"] = _worst(
                res.get(f"out.{cs_name}")[i], getattr(python[i], py_name)
            )
    record_property("worst_relative_error", worst)
    assert all(v <= 1e-9 for v in worst.values()), worst  # tier L: 1 - exp(-D) terms
    # Positive witnesses inside the compared records. EEC's 0.5 yr cutoff hides the new eggs from
    # biomass, but their larva deaths show on step 1; Bay of Biscay has no cutoff and lays its
    # first (seeded) eggs at step 2, when its spawning season opens.
    if demo == "eec_full":
        assert python[1].mortality_by_cause[:, 2].sum() > 0
    else:
        assert python[0].biomass.sum() == 0 and python[3].biomass.sum() > 0


@pytest.mark.parametrize("freq", [5, 7])
def test_whole_run_windows_with_a_partial_tail(engine_ok, freq, tmp_path):
    raw = _raw(
        "bay_of_biscay",
        tmp_path,
        1,
        {"output.recordfrequency.ndt": str(freq), "output.step0.include": "true"},
    )
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    assert config.output_step0_include and config.n_steps % freq != 0
    python = sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    build_bundle(config, raw, grid, seed=11).save(tmp_path / "model.bin")
    res = _probe("Run", tmp_path / "model.bin", tmp_path / "out.bin")
    assert np.array_equal(res.get("out.record_step"), [r.step for r in python])
    assert np.array_equal(res.get("out.window_steps"), [r.window_steps for r in python])
    assert python[0].step == -1 and python[-1].window_steps == config.n_steps % freq
