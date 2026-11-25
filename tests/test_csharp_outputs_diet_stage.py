"""The staged diet outputs on the C# engine equal the Python engine's (PR-E, 2026-11-25).

Java's DietOutput / PredatorPressureOutput with ``output.diet.stage.*`` and
BiomassDietStageOutput (osmose/engine/diet_stage.py):

A. One production step through both engines (the kernels book each meal under the prey's stage
   at predation) at tier L, as test_csharp_outputs_families C does for the other families.
B. The collector bit for bit: Baltic states with background predators appended and given staged
   rows go through the C# "Windows" probe and through the Python snapshot/collect pair.

Needs the built engine; skips without it unless OSMOSE_REQUIRE_CSHARP=1.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from osmose.engine.diet_stage import (
    biomass_at_step_start,
    collect,
    pred_rows,
    snapshot_after_mortality,
    stages_for,
)
from osmose.engine.output_regions import subset
from osmose.engine_cs.build import build_bundle
from tests.test_csharp_mortality_step import _write_draws
from tests.test_csharp_outputs_families import ALL
from tests.test_csharp_pipeline import (
    STEP_CASES,
    TIER_L,
    _capture_step,
    _post_reproduction_state,
    _probe,
    _raw,
    _worst,
    _write_states,
    engine_ok,  # noqa: F401  (a fixture)
)

STAGED = ALL | {"output.diet.composition.enabled": "true", "output.diet.pressure.enabled": "true"}
FIELDS = ("diet_stage_num", "diet_stage_den", "predpressure_stage", "biomass_diet_stage")


# ------------------------------------------------------------------ A. one production step


@pytest.mark.parametrize("demo", sorted(STEP_CASES))
def test_one_production_step_equals_the_python_engine(engine_ok, demo, tmp_path):  # noqa: F811
    step = STEP_CASES[demo]
    got = _capture_step(_raw(demo, tmp_path, step // 24 + 1, STAGED), step)
    config, out = got["config"], got["output"]
    assert config.diet_staged_output and config.output_biomass_diet_stage
    build_bundle(config, got["raw"], got["grid"], seed=11).save(tmp_path / "model.bin")
    _write_states([got["entry"]], tmp_path / "entry.bin")
    _write_draws(got["record"], tmp_path / "draws.bin")
    res = _probe(
        "Step", tmp_path / "model.bin", tmp_path / "entry.bin", tmp_path / "draws.bin", step,
        tmp_path / "out.bin",
    )  # fmt: skip
    worst = {}
    for name in FIELDS:
        mine, want = res.get(f"out.{name}")[0], getattr(out, name)
        assert want is not None, name
        assert mine.shape == want.shape, name
        worst[name] = _worst(mine.ravel(), want.ravel())
    assert all(v <= TIER_L for v in worst.values()), worst

    assert out.biomass_diet_stage.sum() > 0
    if demo != "eec_full":
        return  # Bay of Biscay barely feeds (CLAUDE.md: its accessibility matrix is transposed)
    # Not vacuous: a prey species eaten in two of its stages by one predator stage (the prey
    # axis is really staged), predators of several stages that ate, and resource biomass.
    ds = stages_for(config)
    num, den = out.diet_stage_num, out.diet_stage_den
    off = ds.col_offset
    two_stages = any(
        ((num[r, off[s] : off[s + 1]] > 0).sum() >= 2)
        for r in range(num.shape[0])
        for s in range(len(off) - 1)
        if ds.n_stages[s] > 1
    )
    assert two_stages
    assert (den > 0).sum() >= 3
    assert out.biomass_diet_stage[-ds.n_resources :].sum() > 0


# --------------------------------------------------------------------- B. the collector


def _with_background(state, config, rng):
    """The state plus a few schools of each background species (copies of focal schools with
    the background species id, flag and a length that spans its stages)."""
    n_sp = config.n_species
    extra = []
    for k in range(config.n_background):
        idx = rng.choice(np.flatnonzero(state.species_id < n_sp), 6, replace=False)
        b = copy.deepcopy(subset(state, np.isin(np.arange(len(state)), idx)))
        b.species_id[:] = n_sp + k
        b.is_background[:] = True
        b.length[:] = np.linspace(20.0, 180.0, len(b))
        extra.append(b)
    out = state
    for b in extra:
        out = out.append(b)
    return out


@pytest.fixture(scope="module")
def baltic_states(tmp_path_factory):
    d = tmp_path_factory.mktemp("baltic")
    config, grid, raw, st = _post_reproduction_state(_raw("baltic", d, 2, STAGED), 30)
    assert config.n_background > 0 and config.diet_staged_output
    ds = stages_for(config)
    rng = np.random.default_rng(5)
    states, rows = [], []
    for _ in range(6):
        s = _with_background(copy.deepcopy(st), config, rng)
        s.abundance[:] *= rng.uniform(0.5, 1.5, len(s))
        r = rng.uniform(0.0, 3.0, (len(s), ds.width)) * (rng.random((len(s), ds.width)) < 0.3)
        r[rng.random(len(s)) < 0.2] = 0.0  # schools that ate nothing
        focal = np.flatnonzero(s.species_id < config.n_species)
        s.is_out[focal[0::11]] = True  # out of the domain: no diet row, still pressure
        s.age_dt[focal[1::13]] = config.lifespan_dt[s.species_id[focal[1::13]]]  # not alive
        states.append(s.replace(biomass=s.abundance * s.weight))
        rows.append(r)
    return config, grid, raw, states, rows


def test_the_collector_equals_python_bit_for_bit(engine_ok, baltic_states, tmp_path):  # noqa: F811
    config, grid, raw, states, rows = baltic_states
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = 1, False
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(
        states, tmp_path / "states.bin", extra={"state.staged_diet": np.concatenate(rows)}
    )
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    for i, (s, r) in enumerate(zip(states, rows, strict=True)):
        want = collect(s, cfg, snapshot_after_mortality(s, cfg, r))
        want["biomass_diet_stage"] = biomass_at_step_start(s, cfg, None, None)
        for name in FIELDS:
            got = res.get(f"out.{name}")[i]
            assert got.shape == want[name].shape and got.tobytes() == want[name].tobytes(), (
                name,
                i,
            )


def test_the_collector_fixture_is_not_trivial(baltic_states):
    config, _, _, states, rows = baltic_states
    s, r = states[0], rows[0]
    out = collect(s, config, snapshot_after_mortality(s, config, r))
    ds = stages_for(config)
    at = pred_rows(ds, config.n_species + config.n_background)
    bkg_rows = slice(at[config.n_species], at[-1])
    assert out["diet_stage_den"][bkg_rows].sum() > 0  # background predators have rows
    assert (out["diet_stage_den"][: at[config.n_species]] > 0).sum() >= 5
    assert s.is_out.any() and (out["predpressure_stage"].sum() > 0)
