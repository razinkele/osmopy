"""A school holding fewer than one fish is dead, as in Java (review T3).

Java floors a school's instantaneous abundance below one fish to zero
(`School.updateBiomAndAbd`, `School.java:583-586`), and every later read in the step uses the
floored value: `isAlive()`, growth, the spawning stock that decides the `ssb == 0` seeding
rescue, and outputs. The Python engine kept any school with `abundance > 0`. On the Baltic
config, a median of 23,389 such residues stayed alive each step, about 62% of all live focal
schools. They also blocked the seeding rescue whenever they were a stock's only spawners.

The floor is applied after mortality, so growth, reproduction and outputs see what Java sees.
Newborn egg schools (age_dt == 0), created later in the same step, are exempt until their first
mortality, as in Java, where an egg batch is floored at its next read.
"""

from pathlib import Path

import numpy as np

from osmose.engine import PythonEngine
from osmose.engine.state import SchoolState
from tests._tutorial_config import build_config


def test_floor_zeroes_schools_under_one_fish_and_keeps_biomass_consistent():
    from osmose.engine.simulate import _floor_sub_one_fish

    state = SchoolState.create(n_schools=5, species_id=np.zeros(5, dtype=np.int32))
    state = state.replace(
        abundance=np.array([0.5, 2.0, 0.0, 0.999999, 1.0]),
        weight=np.array([1.0, 3.0, 1.0, 2.0, 4.0]),
        biomass=np.array([0.5, 6.0, 0.0, 1.999998, 4.0]),
    )
    out = _floor_sub_one_fish(state)
    # strictly below one fish is dead; exactly one fish survives (Java: instantaneousAbundance < 1)
    np.testing.assert_array_equal(out.abundance, [0.0, 2.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(out.biomass, out.abundance * out.weight)


def test_no_school_ends_a_step_with_under_one_fish(tmp_path: Path):
    cfg = build_config(tmp_path, n_year=1)
    residues: list[int] = []

    def observe(step, state, grid, config, map_sets=None):
        focal = state.species_id < config.n_species
        residue = focal & (state.abundance > 0) & (state.abundance < 1) & (state.age_dt > 0)
        residues.append(int(residue.sum()))

    PythonEngine().run(cfg, tmp_path / "out", seed=42, step_observer=observe)
    assert residues and max(residues) == 0, f"steps with residues: {sum(r > 0 for r in residues)}"
