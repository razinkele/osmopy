"""`_collect_by_life_stage` must stay bit-identical to its np.add.at definition.

The collector was rewritten from tuple-indexed ``np.add.at`` to flattened ``np.bincount`` for speed
(perf plan item 4). Both sum each (species, stage) group in array order starting from zero, so the
results are identical to the bit. The reference below is the original implementation. The data
span 32 orders of magnitude so that any change in summation order would show in the low bits.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from osmose.engine.simulate import _collect_by_life_stage
from osmose.engine.state import MortalityCause, SchoolState

N_SPECIES = 5


def _reference(state, config):
    n_causes = len(MortalityCause)
    abundance = np.zeros((config.n_species, 3), dtype=np.float64)
    deaths = np.zeros((config.n_species, n_causes, 3), dtype=np.float64)
    if len(state) == 0:
        return abundance, deaths
    focal = state.species_id < config.n_species
    mature = (state.length >= config.maturity_size[state.species_id]) & (
        state.age_dt >= config.maturity_age_dt[state.species_id]
    )
    stage = np.where(state.is_egg, 0, np.where(mature, 2, 1))
    sp, st = state.species_id[focal], stage[focal]
    np.add.at(abundance, (sp, st), state.abundance[focal])
    for cause in range(n_causes):
        np.add.at(deaths, (sp, cause, st), state.n_dead[focal, cause])  # type: ignore[arg-type]
    return abundance, deaths


def _config(n_background: int = 2):
    n_all = N_SPECIES + n_background  # background ids index the per-species arrays too
    return SimpleNamespace(
        n_species=N_SPECIES,
        maturity_size=np.linspace(5.0, 40.0, n_all),
        maturity_age_dt=np.arange(n_all, dtype=np.int32) * 6,
    )


def _state(n: int, seed: int, n_background: int = 2) -> SchoolState:
    rng = np.random.default_rng(seed)
    n_causes = len(MortalityCause)
    species = rng.integers(0, N_SPECIES + n_background, n).astype(np.int32)
    magnitude = 10.0 ** rng.uniform(-20.0, 12.0, n)
    n_dead = 10.0 ** rng.uniform(-20.0, 12.0, (n, n_causes))
    n_dead[rng.random((n, n_causes)) < 0.2] = 0.0
    n_dead[rng.random((n, n_causes)) < 0.05] = -0.0
    return SchoolState.create(n_schools=n, species_id=species).replace(
        abundance=magnitude,
        length=rng.uniform(0.1, 60.0, n),
        age_dt=rng.integers(0, 120, n).astype(np.int32),
        is_egg=rng.random(n) < 0.2,
        n_dead=n_dead,
    )


def _assert_bit_identical(got, want):
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        assert g.dtype == w.dtype
        assert np.ascontiguousarray(g).tobytes() == w.tobytes()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_the_add_at_definition_to_the_bit(seed):
    state, config = _state(4000, seed), _config()
    _assert_bit_identical(_collect_by_life_stage(state, config), _reference(state, config))


def test_every_stage_and_every_cause_is_populated_by_the_fixture():
    """Guard on the fixture: an empty stage or cause would compare zeros with zeros."""
    abundance, deaths = _reference(_state(4000, 0), _config())
    assert (abundance > 0).all()
    assert (deaths > 0).all()


def test_background_schools_are_dropped():
    state, config = _state(500, 3), _config()
    only_background = state.replace(species_id=np.full(len(state), N_SPECIES + 1, dtype=np.int32))
    abundance, deaths = _collect_by_life_stage(only_background, config)
    assert not abundance.any() and not deaths.any()
    _assert_bit_identical((abundance, deaths), _reference(only_background, config))


def test_empty_state():
    config = _config()
    empty = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
    _assert_bit_identical(_collect_by_life_stage(empty, config), _reference(empty, config))
