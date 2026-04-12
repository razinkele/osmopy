"""Tests for Phase 3: Reproduction & mortality refinements."""

import numpy as np
import pytest

from tests.helpers import _make_school
from osmose.engine.config import EngineConfig
from osmose.engine.processes.mortality import _apply_additional_for_school
from osmose.engine.processes.natural import additional_mortality
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import MortalityCause, SchoolState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(n_sp: int = 1, n_dt: int = 12) -> dict[str, str]:
    """Minimal config dict for phase 3 tests."""
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
            }
        )
    return cfg


# ===========================================================================
# 3.1 — Spawning Season Normalization
# ===========================================================================


class TestSpawningSeasonNormalization:
    def test_normalized_season_sums_to_one(self, tmp_path):
        """When normalisation is enabled, season values should sum to 1."""
        cfg = _base_config(n_sp=1, n_dt=4)
        csv_path = tmp_path / "season_sp0.csv"
        csv_path.write_text("step;season\n0;2\n1;4\n2;6\n3;8\n")
        cfg["reproduction.season.file.sp0"] = str(csv_path)
        cfg["reproduction.normalisation.enabled"] = "true"
        cfg["_osmose.config.dir"] = str(tmp_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.spawning_season is not None
        np.testing.assert_allclose(ec.spawning_season[0].sum(), 1.0, rtol=1e-10)
        # Values proportional: 2/20, 4/20, 6/20, 8/20
        np.testing.assert_allclose(ec.spawning_season[0], [0.1, 0.2, 0.3, 0.4])

    def test_unnormalized_season_preserves_raw_values(self, tmp_path):
        """When normalisation is disabled, raw CSV values are kept."""
        cfg = _base_config(n_sp=1, n_dt=4)
        csv_path = tmp_path / "season_sp0.csv"
        csv_path.write_text("step;season\n0;2\n1;4\n2;6\n3;8\n")
        cfg["reproduction.season.file.sp0"] = str(csv_path)
        cfg["reproduction.normalisation.enabled"] = "false"
        cfg["_osmose.config.dir"] = str(tmp_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.spawning_season is not None
        np.testing.assert_allclose(ec.spawning_season[0], [2, 4, 6, 8])


# ===========================================================================
# 3.2 — Multi-year Spawning Season Time Series
# ===========================================================================


class TestMultiYearSpawningSeason:
    def test_multiyear_csv_wraps_by_column_count(self, tmp_path):
        """CSV with more rows than n_dt_per_year stores full array; reproduction wraps."""
        cfg = _base_config(n_sp=1, n_dt=4)
        # 8 rows = 2 years of 4 steps each, different values per year
        csv_path = tmp_path / "season_sp0.csv"
        csv_path.write_text(
            "step;season\n"
            "0;0.1\n1;0.2\n2;0.3\n3;0.4\n"  # year 0
            "4;0.5\n5;0.6\n6;0.7\n7;0.8\n"  # year 1
        )
        cfg["reproduction.season.file.sp0"] = str(csv_path)
        cfg["_osmose.config.dir"] = str(tmp_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.spawning_season is not None
        # Array should have 8 columns (not truncated to 4)
        assert ec.spawning_season.shape[1] == 8
        # At step=0 -> index 0 -> 0.1; at step=5 -> index 5 -> 0.6
        assert ec.spawning_season[0, 0] == pytest.approx(0.1)
        assert ec.spawning_season[0, 5] == pytest.approx(0.6)

        # Test reproduction uses step % n_cols
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            length=np.array([15.0]),
            weight=np.array([20.25]),
            biomass=np.array([20250.0]),
            age_dt=np.array([24], dtype=np.int32),
        )
        rng = np.random.default_rng(42)

        # step=1 -> index 1 -> 0.2
        s1 = reproduction(state, ec, step=1, rng=rng.spawn(1)[0])
        eggs_step1 = s1.abundance[1:].sum()

        # step=5 -> index 5 -> 0.6 (3x of step 1)
        s5 = reproduction(state, ec, step=5, rng=rng.spawn(1)[0])
        eggs_step5 = s5.abundance[1:].sum()

        np.testing.assert_allclose(eggs_step5 / eggs_step1, 0.6 / 0.2, rtol=1e-6)


# ===========================================================================
# 3.3 — Time-varying Additional Mortality (BY_DT)
# ===========================================================================


class TestTimeVaryingAdditionalMortality:
    def test_bydt_rate_changes_between_steps(self, tmp_path):
        """BY_DT additional mortality uses step-indexed rate from CSV."""
        cfg = _base_config(n_sp=1, n_dt=4)
        # 4 rates: step 0 has zero, step 1 has high rate
        csv_path = tmp_path / "add_mort_sp0.csv"
        csv_path.write_text("0.0\n2.0\n0.0\n0.0\n")
        cfg["mortality.additional.rate.bytdt.file.sp0"] = str(csv_path)
        cfg["_osmose.config.dir"] = str(tmp_path)
        ec = EngineConfig.from_dict(cfg)

        assert ec.additional_mortality_by_dt is not None
        assert ec.additional_mortality_by_dt[0] is not None

        # At step=0 (rate=0.0) -> no additional mortality
        state0 = _make_school(n=1, sp=0, abundance=1000.0, age_dt=10)
        n_subdt = ec.mortality_subdt
        _apply_additional_for_school(0, state0, ec, n_subdt, state0.abundance.copy(), step=0)
        dead_step0 = state0.n_dead[0, int(MortalityCause.ADDITIONAL)]
        assert dead_step0 == 0.0

        # At step=1 (rate=2.0) -> positive mortality
        state1 = _make_school(n=1, sp=0, abundance=1000.0, age_dt=10)
        _apply_additional_for_school(0, state1, ec, n_subdt, state1.abundance.copy(), step=1)
        dead_step1 = state1.n_dead[0, int(MortalityCause.ADDITIONAL)]
        assert dead_step1 > 0.0

    def test_bydt_batch_function_uses_step(self, tmp_path):
        """The batch additional_mortality function also respects by-dt rates."""
        cfg = _base_config(n_sp=1, n_dt=4)
        csv_path = tmp_path / "add_mort_sp0.csv"
        csv_path.write_text("0.0\n2.0\n0.0\n0.0\n")
        cfg["mortality.additional.rate.bytdt.file.sp0"] = str(csv_path)
        cfg["_osmose.config.dir"] = str(tmp_path)
        ec = EngineConfig.from_dict(cfg)

        state = _make_school(n=1, sp=0, abundance=1000.0, age_dt=10)
        n_subdt = ec.mortality_subdt

        # step=0 (rate=0.0) -> no deaths
        result0 = additional_mortality(state, ec, n_subdt, step=0)
        dead0 = result0.n_dead[0, int(MortalityCause.ADDITIONAL)]
        assert dead0 == 0.0

        # step=1 (rate=2.0) -> deaths
        result1 = additional_mortality(state, ec, n_subdt, step=1)
        dead1 = result1.n_dead[0, int(MortalityCause.ADDITIONAL)]
        assert dead1 > 0.0


# ===========================================================================
# 3.4 — Spatial Additional Mortality
# ===========================================================================


class TestSpatialAdditionalMortality:
    def test_zero_cell_has_no_additional_mortality(self, tmp_path):
        """Cell with spatial factor 0 should produce no additional mortality."""
        cfg = _base_config(n_sp=1, n_dt=4)
        cfg["mortality.additional.rate.sp0"] = "1.0"
        # 2x2 spatial grid: cell (0,0) = 0.0, cell (0,1) = 2.0
        csv_path = tmp_path / "spatial_sp0.csv"
        csv_path.write_text("0.0;2.0\n1.0;1.0\n")
        cfg["mortality.additional.spatial.distrib.file.sp0"] = str(csv_path)
        cfg["_osmose.config.dir"] = str(tmp_path)
        ec = EngineConfig.from_dict(cfg)

        assert ec.additional_mortality_spatial is not None
        assert ec.additional_mortality_spatial[0] is not None

        n_subdt = ec.mortality_subdt

        # School at cell (1, 0) -> spatial grid row 0, col 0 (flipud) = 1.0
        # School at cell (0, 0) -> spatial grid row 1, col 0 (flipud) = 0.0
        # Wait, _load_spatial_csv does flipud. Grid "0.0;2.0\n1.0;1.0" becomes:
        # row0: [1.0, 1.0], row1: [0.0, 2.0]
        # So cell_y=0, cell_x=0 -> grid[0,0] = 1.0
        # And cell_y=1, cell_x=0 -> grid[1,0] = 0.0

        # School at cell_y=1, cell_x=0 -> factor 0.0 -> no mortality
        state_zero = _make_school(n=1, sp=0, abundance=1000.0, age_dt=10, cell_y=1, cell_x=0)
        _apply_additional_for_school(
            0, state_zero, ec, n_subdt, state_zero.abundance.copy(), step=0
        )
        dead_zero = state_zero.n_dead[0, int(MortalityCause.ADDITIONAL)]
        assert dead_zero == 0.0

    def test_high_spatial_factor_increases_mortality(self, tmp_path):
        """Cell with spatial factor > 1 should have more mortality than factor = 1."""
        cfg = _base_config(n_sp=1, n_dt=4)
        cfg["mortality.additional.rate.sp0"] = "0.5"
        # 2x2: after flipud row0=[1.0,1.0], row1=[1.0,3.0]
        csv_path = tmp_path / "spatial_sp0.csv"
        csv_path.write_text("1.0;3.0\n1.0;1.0\n")
        cfg["mortality.additional.spatial.distrib.file.sp0"] = str(csv_path)
        cfg["_osmose.config.dir"] = str(tmp_path)
        ec = EngineConfig.from_dict(cfg)

        n_subdt = ec.mortality_subdt

        # cell_y=0, cell_x=0 -> factor 1.0
        state_normal = _make_school(n=1, sp=0, abundance=1000.0, age_dt=10, cell_y=0, cell_x=0)
        _apply_additional_for_school(
            0, state_normal, ec, n_subdt, state_normal.abundance.copy(), step=0
        )
        dead_normal = state_normal.n_dead[0, int(MortalityCause.ADDITIONAL)]

        # cell_y=0, cell_x=1 -> factor 1.0 (same row)
        # cell_y=1, cell_x=1 -> factor 3.0 (after flipud)
        state_high = _make_school(n=1, sp=0, abundance=1000.0, age_dt=10, cell_y=1, cell_x=1)
        _apply_additional_for_school(
            0, state_high, ec, n_subdt, state_high.abundance.copy(), step=0
        )
        dead_high = state_high.n_dead[0, int(MortalityCause.ADDITIONAL)]

        assert dead_high > dead_normal


# ===========================================================================
# 3.5 — Egg Placement Timing
# ===========================================================================


class TestEggPlacementTiming:
    def test_new_eggs_have_unlocated_cells(self):
        """New eggs should have cell_x = cell_y = -1 (unlocated)."""
        cfg = _base_config(n_sp=1, n_dt=12)
        ec = EngineConfig.from_dict(cfg)

        # Mature school
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            length=np.array([15.0]),
            weight=np.array([20.25]),
            biomass=np.array([20250.0]),
            age_dt=np.array([24], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, ec, step=0, rng=rng)

        # New egg schools (indices 1:) should have cell_x = cell_y = -1
        assert len(new_state) > 1
        assert np.all(new_state.cell_x[1:] == -1)
        assert np.all(new_state.cell_y[1:] == -1)


# ===========================================================================
# 3.6 — nEgg < nSchool Edge Case
# ===========================================================================


class TestNEggLessThanNSchool:
    def test_few_eggs_creates_one_school(self):
        """When n_eggs < n_schools, only 1 school should be created."""
        cfg = _base_config(n_sp=1, n_dt=12)
        # Very low fecundity so very few eggs
        cfg["species.relativefecundity.sp0"] = "0.0001"
        cfg["population.seeding.biomass.sp0"] = "0.001"
        ec = EngineConfig.from_dict(cfg)

        # State with SSB=0 so seeding kicks in with tiny biomass
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1.0]),
            length=np.array([5.0]),
            weight=np.array([0.001]),
            biomass=np.array([0.001]),
            age_dt=np.array([0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = reproduction(state, ec, step=0, rng=rng)

        # Seeding uses 0.001 tonnes, fecundity 0.0001 eggs/g
        # n_eggs = 0.5 * 0.0001 * 0.001 * (1/12) * 1e6 ≈ 4.17
        # n_schools = 5, so n_eggs < n_schools -> 1 school created
        n_new = len(new_state) - 1  # subtract original
        assert n_new == 1


# ===========================================================================
# 3.7 — population.seeding.year.max Config Key
# ===========================================================================


class TestSeedingYearMax:
    def test_seeding_stops_at_configured_year(self):
        """Seeding should stop when step exceeds seeding_max_step."""
        cfg = _base_config(n_sp=1, n_dt=12)
        cfg["population.seeding.year.max"] = "2"  # Stop seeding after 2 years
        ec = EngineConfig.from_dict(cfg)

        assert ec.seeding_max_step[0] == 24  # 2 years * 12 dt/year

        # State with no mature fish -> SSB=0 -> seeding needed
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1.0]),
            length=np.array([5.0]),
            weight=np.array([0.001]),
            biomass=np.array([0.001]),
            age_dt=np.array([0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)

        # step=12 (year 1) -> within seeding period -> eggs produced
        s1 = reproduction(state, ec, step=12, rng=rng.spawn(1)[0])
        assert len(s1) > 1, "Should produce eggs within seeding period"

        # step=24 (year 2) -> at boundary -> no more seeding (step_year >= max)
        s2 = reproduction(state, ec, step=24, rng=rng.spawn(1)[0])
        assert len(s2) == 1, "Should NOT produce eggs after seeding period"
