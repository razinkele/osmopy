"""Yield by fishery (Java FisheryOutputYield) on the C# engine equals the Python engine's.

Two configs through the C# "Windows" probe and ``simulate``'s own recording rule:

- EEC on the PER-FISHERY path (tests/test_csharp_per_fishery.py's overrides: a species in two
  fisheries at catchability 0.5, a gradient on every map, a fishery off from step 6), plus a
  discard matrix, so each school's landings and discards are split by the step's catch shares;
- EEC as shipped, on the REDUCED path, where each species' catch goes to its one fishery.

Twelve perturbed copies of a real state carry fishing and discard deaths. Needs the built
engine; skips without it unless OSMOSE_REQUIRE_CSHARP=1.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from osmose.engine.state import MortalityCause
from osmose.engine_cs.build import FAMILY_YIELD_BY_FISHERY, build_bundle, output_families
from tests.test_csharp_outputs_families import ALL
from tests.test_csharp_per_fishery import _overrides
from tests.test_csharp_pipeline import (
    _post_reproduction_state,
    _probe,
    _python_records,
    _raw,
    _write_states,
    engine_ok,  # noqa: F401  (a fixture)
)

ON = ALL | {"output.fisheries.yield.biomass.enabled": "true"}
F, D = int(MortalityCause.FISHING), int(MortalityCause.DISCARDS)
NAMES = ("yield_by_fishery_landed", "yield_by_fishery_discarded")


def _states(config, st, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(12):
        s = copy.deepcopy(st)
        s.abundance[:] *= rng.uniform(0.5, 1.5, len(s))
        focal = np.flatnonzero(s.species_id < config.n_species)
        # older fish on every third school: EEC's first fishery of species 0 takes only fish
        # of 4+ years, so without this no species is ever split over two fisheries
        s.age_dt[focal[0::3]] += 5 * config.n_dt_per_year
        s.n_dead[focal, F] = s.abundance[focal] * rng.uniform(0.0, 0.1, len(focal))
        s.n_dead[focal, D] = s.abundance[focal] * rng.uniform(0.0, 0.05, len(focal))
        out.append(s.replace(biomass=s.abundance * s.weight))
    return out


@pytest.fixture(scope="module")
def fishery_states(tmp_path_factory):
    out = {}
    d = tmp_path_factory.mktemp("per_fishery_path")
    extra = _overrides(d)
    q = pd.read_csv(extra["fisheries.catchability.file"], index_col=0)
    disc = q.copy() * 0.0
    disc.iloc[:, :] = np.where(q.to_numpy() > 0, 0.3, 0.0)  # every caught pair discards 30 %
    disc.iloc[0, 0] = 0.1  # a different discard rate on the split species' first fishery
    disc.to_csv(d / "disc.csv")
    extra = {k: v for k, v in extra.items() if not k.startswith("_test.")}
    extra["fisheries.discards.file"] = str(d / "disc.csv")
    config, grid, raw, st = _post_reproduction_state(_raw("eec_full", d, 2, ON | extra), 30)
    assert config.fishery_model is not None  # the per-fishery path
    out["per_fishery"] = (config, grid, raw, _states(config, st, 3))
    d2 = tmp_path_factory.mktemp("reduced_path")
    config, grid, raw, st = _post_reproduction_state(_raw("eec_full", d2, 2, ON), 30)
    assert config.fishery_model is None and config.fishery_of_species is not None
    out["reduced"] = (config, grid, raw, _states(config, st, 4))
    return out


def _same(a, b) -> bool:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("path", ["per_fishery", "reduced"])
@pytest.mark.parametrize(("freq", "step0"), [(5, False), (1, False), (12, True)])
def test_yield_by_fishery_equals_python_bit_for_bit(
    engine_ok,  # noqa: F811
    fishery_states,
    path,
    freq,
    step0,
    tmp_path,
):
    config, grid, raw, states = fishery_states[path]
    cfg = copy.copy(config)
    cfg.output_record_frequency, cfg.output_step0_include = freq, step0
    assert output_families(cfg) & FAMILY_YIELD_BY_FISHERY
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    _write_states(states, tmp_path / "states.bin")
    res = _probe("Windows", tmp_path / "model.bin", tmp_path / "states.bin", tmp_path / "out.bin")
    want = _python_records(states, cfg, freq, step0)
    for name in NAMES:
        got = res.get(f"out.{name}")
        assert got.shape[0] == len(want), name
        for i, r in enumerate(want):
            assert _same(got[i], getattr(r, name)), (name, i)


def test_per_fishery_split_is_real_and_adds_up(fishery_states):
    """Non-vacuity and conservation: on the per-fishery path at least one species is split over
    two fisheries, the gap fishery catches on step 0 and not on step 11, and landings (discards)
    summed over fisheries equal the species' FISHING (DISCARDS) deaths x weight."""
    from osmose.engine.simulate import _collect_outputs, _fishery_catch_shares

    config, _, _, states = fishery_states["per_fishery"]
    nf = config.n_species
    for step, s in ((0, states[0]), (11, states[11])):
        out = _collect_outputs(s, config, step)
        land, disc = out.yield_by_fishery_landed, out.yield_by_fishery_discarded
        # Only schools some fishery fishes can hold fishing deaths in a run: the perturbed
        # states also put deaths on the others, which have no share to be split by.
        sid, s_land, s_disc = _fishery_catch_shares(s, config, step)
        ok_l = (sid < nf) & (s_land.sum(axis=1) > 0)
        ok_d = (sid < nf) & (s_disc.sum(axis=1) > 0)
        want_l = np.bincount(sid[ok_l], s.n_dead[ok_l, F] * s.weight[ok_l], nf)
        want_d = np.bincount(sid[ok_d], s.n_dead[ok_d, D] * s.weight[ok_d], nf)
        assert ok_l.sum() > 100  # many caught schools
        np.testing.assert_allclose(land.sum(axis=1), want_l, rtol=1e-12)
        np.testing.assert_allclose(disc.sum(axis=1), want_d, rtol=1e-12)
        assert ((land > 0).sum(axis=1) >= 2).any()  # a species split over two fisheries
        assert disc.sum() > 0
    early = _collect_outputs(states[0], config, 0).yield_by_fishery_landed
    late = _collect_outputs(states[0], config, 11).yield_by_fishery_landed
    assert not np.array_equal(early, late)  # the scheduled gap moves catch between fisheries
