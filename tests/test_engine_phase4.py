"""Tests for Phase 4: Movement refinement."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.movement import movement
from osmose.engine.state import SchoolState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(n_sp: int = 1, n_dt: int = 12) -> dict[str, str]:
    """Minimal config dict for phase 4 tests."""
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": str(n_dt),
        "simulation.time.nyear": "5",
        "simulation.nspecies": str(n_sp),
        "mortality.subdt": "10",
    }
    names = ["FishA", "FishB", "FishC"]
    for i in range(n_sp):
        cfg.update(
            {
                f"simulation.nschool.sp{i}": "5",
                f"species.name.sp{i}": names[i],
                f"species.linf.sp{i}": "30.0",
                f"species.k.sp{i}": "0.3",
                f"species.t0.sp{i}": "-0.1",
                f"species.egg.size.sp{i}": "0.1",
                f"species.length2weight.condition.factor.sp{i}": "0.006",
                f"species.length2weight.allometric.power.sp{i}": "3.0",
                f"species.lifespan.sp{i}": "5",
                f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
                f"predation.ingestion.rate.max.sp{i}": "3.5",
                f"predation.efficiency.critical.sp{i}": "0.57",
                f"species.sexratio.sp{i}": "0.5",
                f"species.relativefecundity.sp{i}": "800",
                f"species.maturity.size.sp{i}": "12.0",
                f"population.seeding.biomass.sp{i}": "50000",
                f"movement.distribution.method.sp{i}": "random",
                f"movement.randomwalk.range.sp{i}": "1",
            }
        )
    return cfg


def _make_school(
    n: int = 1,
    sp: int = 0,
    abundance: float = 1000.0,
    length: float = 15.0,
    age_dt: int = 48,
    cell_x: int = -1,
    cell_y: int = -1,
) -> SchoolState:
    state = SchoolState.create(n_schools=n, species_id=np.full(n, sp, dtype=np.int32))
    weight = 0.006 * length**3.0
    return state.replace(
        abundance=np.full(n, abundance),
        weight=np.full(n, weight),
        length=np.full(n, length),
        age_dt=np.full(n, age_dt, dtype=np.int32),
        cell_x=np.full(n, cell_x, dtype=np.int32),
        cell_y=np.full(n, cell_y, dtype=np.int32),
    )


# ===========================================================================
# 4.1 — Random Distribution Patch Constraint
# ===========================================================================


class TestRandomDistributionPatch:
    def test_unlocated_school_placed_within_patch(self):
        """When ncell is set and school is unlocated, it should be placed on a patch cell."""
        cfg = _base_config(n_sp=1)
        # Grid: 5x5, all ocean
        grid = Grid.from_dimensions(5, 5)
        # Set ncell to 4 — patch should be a BFS-connected set of 4 cells
        cfg["movement.distribution.ncell.sp0"] = "4"
        config = EngineConfig.from_dict(cfg)

        # Create unlocated school (cell_x=-1, cell_y=-1)
        state = _make_school(n=1, sp=0, cell_x=-1, cell_y=-1)
        rng = np.random.default_rng(42)

        # Build random patches
        from osmose.engine.processes.movement import build_random_patches

        patches = build_random_patches(config, grid, rng)
        assert 0 in patches
        assert len(patches[0]) == 4

        # Movement should place the school on a patch cell
        result = movement(state, grid, config, step=0, rng=rng, random_patches=patches)
        assert result.cell_x[0] >= 0
        assert result.cell_y[0] >= 0
        assert (int(result.cell_x[0]), int(result.cell_y[0])) in patches[0]

    def test_subsequent_walk_not_constrained_to_patch(self):
        """After initial placement, random walk should not be constrained to the patch."""
        cfg = _base_config(n_sp=1)
        # Grid: 10x10, all ocean
        grid = Grid.from_dimensions(10, 10)
        cfg["movement.distribution.ncell.sp0"] = "4"
        cfg["movement.randomwalk.range.sp0"] = "3"
        config = EngineConfig.from_dict(cfg)

        rng = np.random.default_rng(123)
        from osmose.engine.processes.movement import build_random_patches

        patches = build_random_patches(config, grid, rng)

        # Create a school already placed at (5,5) — a located school
        state = _make_school(n=1, sp=0, cell_x=5, cell_y=5)
        rng2 = np.random.default_rng(999)

        # Run movement many times; some should land outside the patch
        positions = set()
        for _ in range(100):
            result = movement(state, grid, config, step=1, rng=rng2, random_patches=patches)
            positions.add((int(result.cell_x[0]), int(result.cell_y[0])))

        # With walk_range=3 from (5,5) on a 10x10 grid, we should reach cells outside patch
        outside_patch = positions - patches[0]
        assert len(outside_patch) > 0, "Walk should reach cells outside the patch"


# ===========================================================================
# 4.2 — Deterministic Random Seeds
# ===========================================================================


class TestDeterministicRandomSeeds:
    def test_random_seed_config_parsed(self):
        """Config keys movement.randomseed.fixed and stochastic.mortality.randomseed.fixed
        should be parsed into EngineConfig."""
        cfg = _base_config(n_sp=1)
        cfg["movement.randomseed.fixed"] = "true"
        cfg["stochastic.mortality.randomseed.fixed"] = "true"
        config = EngineConfig.from_dict(cfg)
        assert config.movement_seed_fixed is True
        assert config.mortality_seed_fixed is True

    def test_random_seed_defaults_false(self):
        """Without config keys, seed flags should default to False."""
        cfg = _base_config(n_sp=1)
        config = EngineConfig.from_dict(cfg)
        assert config.movement_seed_fixed is False
        assert config.mortality_seed_fixed is False
