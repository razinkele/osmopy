"""Tests for Phase 4: Movement refinement."""

from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.movement import build_random_patches, movement
from tests.helpers import _make_school

BOB_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cells(cells) -> set[tuple[int, int]]:
    """The (x, y) cells of a patch as a set, whatever container holds them."""
    return {(int(x), int(y)) for x, y in cells}


def _occupied(state, sp: int | None = None) -> set[tuple[int, int]]:
    """The (x, y) cells holding at least one school (of species ``sp``, if given)."""
    sel = np.ones(len(state), dtype=bool) if sp is None else state.species_id == sp
    return set(zip(state.cell_x[sel].tolist(), state.cell_y[sel].tolist(), strict=True))


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

        patches = build_random_patches(config, grid, rng)
        assert 0 in patches
        assert len(patches[0]) == 4

        # Movement should place the school on a patch cell
        result = movement(state, grid, config, step=0, rng=rng, random_patches=patches)
        assert result.cell_x[0] >= 0
        assert result.cell_y[0] >= 0
        assert (int(result.cell_x[0]), int(result.cell_y[0])) in _cells(patches[0])

    def test_subsequent_walk_not_constrained_to_patch(self):
        """After initial placement, random walk should not be constrained to the patch."""
        cfg = _base_config(n_sp=1)
        # Grid: 10x10, all ocean
        grid = Grid.from_dimensions(10, 10)
        cfg["movement.distribution.ncell.sp0"] = "4"
        cfg["movement.randomwalk.range.sp0"] = "3"
        config = EngineConfig.from_dict(cfg)

        rng = np.random.default_rng(123)

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
        outside_patch = positions - _cells(patches[0])
        assert len(outside_patch) > 0, "Walk should reach cells outside the patch"


class TestRandomDistributionDomain:
    """Where UNLOCATED schools of a "random"-distribution species are placed.

    Java's RandomDistribution (4.3.3 :94-99, :110-116, :122-135; the ODD's movement submodel
    says the same) deals an unlocated school onto a uniformly random cell of its species'
    domain, and the domain is the WHOLE ocean unless ``movement.distribution.ncell`` restricts
    it. Regression for review finding S1 (2026-11-07): with ``ncell`` unset, unlocated schools
    were never placed, and the random walk clipped them from (-1, -1) into the grid's corner.
    """

    @staticmethod
    def _move_once(cfg: dict[str, str], grid: Grid, state, seed: int = 7):
        config = EngineConfig.from_dict(cfg)
        rng = np.random.default_rng(seed)
        patches = build_random_patches(config, grid, rng)
        result = movement(state, grid, config, step=0, rng=rng, random_patches=patches)
        return result, patches

    def test_unset_ncell_places_unlocated_schools_over_every_ocean_cell(self):
        # 3x3 grid with a land centre: 8 ocean cells. Dealt uniformly, 400 schools miss a
        # given cell with probability (7/8)**400 ~ 6e-24.
        mask = np.ones((3, 3), dtype=np.bool_)
        mask[1, 1] = False
        grid = Grid(ny=3, nx=3, ocean_mask=mask)
        state = _make_school(n=400, cell_x=-1, cell_y=-1)
        result, _ = self._move_once(_base_config(n_sp=1), grid, state)
        ocean = {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
        assert _occupied(result) == ocean

    def test_unset_ncell_places_schools_when_the_corner_is_land(self):
        # The walk from (-1, -1) can only reach the corner; with the corner on land the land
        # check sent every school back to (-1, -1), leaving it unlocated for good.
        mask = np.ones((4, 4), dtype=np.bool_)
        mask[0, 0] = False
        grid = Grid(ny=4, nx=4, ocean_mask=mask)
        state = _make_school(n=50, cell_x=-1, cell_y=-1)
        result, _ = self._move_once(_base_config(n_sp=1), grid, state)
        assert (result.cell_x >= 0).all() and (result.cell_y >= 0).all()
        assert mask[result.cell_y, result.cell_x].all()

    def test_species_without_ncell_uses_whole_domain_when_another_species_sets_it(self):
        # sp0 sets ncell, sp1 does not, so sp1's parsed ncell is 0 (Java: unset -> whole
        # domain). 400 schools on 25 cells miss one with probability 25 * (24/25)**400 ~ 2e-6.
        cfg = _base_config(n_sp=2)
        cfg["movement.distribution.ncell.sp0"] = "4"
        state = _make_school(n=400, sp=1, cell_x=-1, cell_y=-1)
        result, _ = self._move_once(cfg, Grid.from_dimensions(5, 5), state)
        assert len(_occupied(result, sp=1)) == 25

    def test_ncell_covering_every_ocean_cell_means_whole_domain(self):
        # Java: areaSize >= nOceanCell -> the whole grid, the boundary case included. A land
        # column splits the sea into two 10-cell regions that no BFS patch can bridge, so
        # treating ncell == n_ocean as a patch would fill only one of them. 400 schools miss
        # one of 20 cells with probability 20 * (19/20)**400 ~ 3e-8.
        mask = np.ones((5, 5), dtype=np.bool_)
        mask[:, 2] = False
        grid = Grid(ny=5, nx=5, ocean_mask=mask)
        cfg = _base_config(n_sp=1)
        cfg["movement.distribution.ncell.sp0"] = "20"
        state = _make_school(n=400, cell_x=-1, cell_y=-1)
        result, _ = self._move_once(cfg, grid, state)
        assert len(_occupied(result)) == 20

    def test_school_placed_this_step_does_not_also_walk(self):
        # Java's move() is if/else: an unlocated school is placed, a located one walks. With a
        # one-cell patch every placed school must still be on that cell after movement().
        cfg = _base_config(n_sp=1)
        cfg["movement.distribution.ncell.sp0"] = "1"
        cfg["movement.randomwalk.range.sp0"] = "3"
        state = _make_school(n=200, cell_x=-1, cell_y=-1)
        result, patches = self._move_once(cfg, Grid.from_dimensions(7, 7), state)
        assert len(patches[0]) == 1
        assert _occupied(result) == _cells(patches[0])

    def test_placement_draws_from_the_species_rng_when_movement_seed_is_fixed(self):
        # movement.randomseed.fixed gives each species its own stream (Java: one Random per
        # RandomDistribution). Placement must draw from it: with the global rng held fixed,
        # a different species stream must move the schools, and the same one must not.
        cfg = _base_config(n_sp=1)
        cfg["movement.randomseed.fixed"] = "true"
        config = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(10, 10)
        patches = build_random_patches(config, grid, np.random.default_rng(0))
        state = _make_school(n=50, cell_x=-1, cell_y=-1)

        def place(species_seed: int):
            return movement(
                state,
                grid,
                config,
                step=0,
                rng=np.random.default_rng(1),
                random_patches=patches,
                species_rngs=[np.random.default_rng(species_seed)],
            )

        a, b, again = place(5), place(6), place(5)
        assert not np.array_equal(a.cell_x * 10 + a.cell_y, b.cell_x * 10 + b.cell_y)
        assert np.array_equal(a.cell_x, again.cell_x) and np.array_equal(a.cell_y, again.cell_y)

    def test_bay_of_biscay_cohorts_are_not_placed_in_the_corner(self):
        # The review's falsifier, on the bundled config it was measured on (all 8 species use
        # "random" with ncell unset; ranges 1-3). Before the fix every range-1 species put
        # 200/200 schools in cell (0, 0). Uniform over the ~250 ocean cells, 200 schools occupy
        # ~138 distinct cells; 100 is several standard deviations below that.
        raw = dict(OsmoseConfigReader().read(BOB_CONFIG))
        config = EngineConfig.from_dict(raw)
        grid = Grid.from_netcdf(
            BOB_CONFIG.parent / raw["grid.netcdf.file"],
            mask_var=raw.get("grid.var.mask", "mask"),
        )
        random_sp = [s for s in range(config.n_species) if config.movement_method[s] == "random"]
        assert len(random_sp) == 8
        state = _make_school(n=200, sp=random_sp[0], cell_x=-1, cell_y=-1)
        for s in random_sp[1:]:
            state = state.append(_make_school(n=200, sp=s, cell_x=-1, cell_y=-1))
        rng = np.random.default_rng(0)
        patches = build_random_patches(config, grid, rng)
        result = movement(state, grid, config, step=0, rng=rng, random_patches=patches)
        # Check located first: a leftover (-1, -1) would index the mask's LAST cell silently.
        assert (result.cell_x >= 0).all() and (result.cell_y >= 0).all()
        assert grid.ocean_mask[result.cell_y, result.cell_x].all()
        for s in random_sp:
            assert len(_occupied(result, sp=s)) >= 100, f"species {s}"


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
