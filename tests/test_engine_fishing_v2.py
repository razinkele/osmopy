"""Tests for Phase 2 fishing features: seasonality, selectivity types, v3 scenarios, MPA, discards."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.fishing import fishing_mortality
from osmose.engine.state import MortalityCause, SchoolState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _base_config(n_sp: int = 1, n_dt: int = 24) -> dict[str, str]:
    """Minimal config for fishing tests."""
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": str(n_dt),
        "simulation.time.nyear": "2",
        "simulation.nspecies": str(n_sp),
        "mortality.subdt": "10",
        "simulation.fishing.mortality.enabled": "true",
    }
    names = ["FishA", "FishB", "FishC", "FishD"]
    for i in range(n_sp):
        cfg.update(
            {
                f"simulation.nschool.sp{i}": "5",
                f"species.name.sp{i}": names[i],
                f"species.linf.sp{i}": "20.0",
                f"species.k.sp{i}": "0.3",
                f"species.t0.sp{i}": "-0.1",
                f"species.egg.size.sp{i}": "0.1",
                f"species.length2weight.condition.factor.sp{i}": "0.006",
                f"species.length2weight.allometric.power.sp{i}": "3.0",
                f"species.lifespan.sp{i}": "5",
                f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
                f"predation.ingestion.rate.max.sp{i}": "3.5",
                f"predation.efficiency.critical.sp{i}": "0.57",
                f"fishing.rate.sp{i}": "0.5",
            }
        )
    return cfg


def _make_school(
    n: int = 1,
    sp: int = 0,
    abundance: float = 1000.0,
    length: float = 15.0,
    age_dt: int = 48,
    cell_x: int = 0,
    cell_y: int = 0,
) -> SchoolState:
    """Create a simple school state for testing."""
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
# 2.1 — Fishery Seasonality
# ===========================================================================


class TestFisherySeasonality:
    def test_seasonality_parsed_from_csv(self, tmp_path):
        """Seasonality CSV is loaded and normalized to sum=1."""
        cfg = _base_config(n_sp=1, n_dt=4)
        # Write seasonality CSV: 4 rows (one per dt), values 1,2,3,4
        csv_path = tmp_path / "season_sp0.csv"
        csv_path.write_text("step;season\n0;1\n1;2\n2;3\n3;4\n")
        cfg["fisheries.seasonality.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.fishing_seasonality is not None
        assert ec.fishing_seasonality.shape == (1, 4)
        # Normalized: sum should be 1.0
        np.testing.assert_allclose(ec.fishing_seasonality[0].sum(), 1.0)
        # Values should be proportional: 1/10, 2/10, 3/10, 4/10
        np.testing.assert_allclose(ec.fishing_seasonality[0], [0.1, 0.2, 0.3, 0.4])

    def test_seasonality_modulates_fishing_rate(self, tmp_path):
        """Fishing rate in a high-season step should exceed uniform rate."""
        cfg = _base_config(n_sp=1, n_dt=4)
        csv_path = tmp_path / "season_sp0.csv"
        # Season: all fishing in step 0, zero in steps 1-3
        csv_path.write_text("step;season\n0;1\n1;0\n2;0\n3;0\n")
        cfg["fisheries.seasonality.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)

        state = _make_school()
        n_subdt = 10

        # Step 0 (in-season): seasonality = 1.0, rate = F * 1.0
        result0 = fishing_mortality(state, ec, n_subdt, step=0)
        dead_step0 = result0.n_dead[0, MortalityCause.FISHING]

        # Step 1 (off-season): seasonality = 0.0, no fishing
        result1 = fishing_mortality(state, ec, n_subdt, step=1)
        dead_step1 = result1.n_dead[0, MortalityCause.FISHING]

        assert dead_step0 > 0
        np.testing.assert_allclose(dead_step1, 0.0)


# ===========================================================================
# 2.3 — v3 Fishing Scenarios: RATE_BY_YEAR
# ===========================================================================


class TestFishingRateByYear:
    def test_rate_by_year_parsed(self, tmp_path):
        """Time-varying annual rate CSV is loaded correctly."""
        cfg = _base_config(n_sp=1, n_dt=4)
        csv_path = tmp_path / "rate_byyear_sp0.csv"
        csv_path.write_text("0.3\n0.6\n")  # year 0 → 0.3, year 1 → 0.6
        cfg["mortality.fishing.rate.byyear.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.fishing_rate_by_year is not None
        assert ec.fishing_rate_by_year[0] is not None
        np.testing.assert_allclose(ec.fishing_rate_by_year[0], [0.3, 0.6])

    def test_rate_changes_between_years(self, tmp_path):
        """Fishing mortality differs between years with by-year rates."""
        cfg = _base_config(n_sp=1, n_dt=4)
        csv_path = tmp_path / "rate_byyear_sp0.csv"
        csv_path.write_text("0.2\n0.8\n")
        cfg["mortality.fishing.rate.byyear.file.sp0"] = str(csv_path)
        ec = EngineConfig.from_dict(cfg)

        state = _make_school()
        n_subdt = 10

        # Year 0, step 0: rate should be 0.2
        result_y0 = fishing_mortality(state, ec, n_subdt, step=0)
        dead_y0 = result_y0.n_dead[0, MortalityCause.FISHING]

        # Year 1, step 4 (first step of year 1): rate should be 0.8
        result_y1 = fishing_mortality(state, ec, n_subdt, step=4)
        dead_y1 = result_y1.n_dead[0, MortalityCause.FISHING]

        # Year 1 rate is 4x year 0 rate, so deaths should be roughly 4x
        assert dead_y1 > dead_y0
        np.testing.assert_allclose(dead_y1 / dead_y0, 4.0, rtol=0.05)


# ===========================================================================
# 2.4 — Fishing Selectivity Types (sigmoid)
# ===========================================================================


class TestSigmoidSelectivity:
    def test_sigmoid_selectivity_at_l50(self, tmp_path):
        """At L50, sigmoid selectivity is 0.5 — half the full rate."""
        cfg = _base_config(n_sp=1, n_dt=24)
        # Set up fisheries-based config with sigmoid selectivity
        cfg.update(
            {
                "fisheries.enabled": "true",
                "simulation.nfisheries": "1",
                "fisheries.name.fsh0": "trawl",
                "fisheries.rate.base.fsh0": "0.5",
                "fisheries.selectivity.type.fsh0": "1",
                "fisheries.selectivity.l50.fsh0": "15.0",
                "fisheries.selectivity.slope.fsh0": "3.0",
            }
        )
        # Write catchability CSV
        catch_path = tmp_path / "catch.csv"
        catch_path.write_text(",trawl\nFishA,1\n")
        cfg["fisheries.catchability.file"] = str(catch_path)
        # Remove legacy rate
        cfg.pop("fishing.rate.sp0", None)

        ec = EngineConfig.from_dict(cfg)
        assert ec.fishing_selectivity_type[0] == 1
        assert ec.fishing_selectivity_l50[0] == pytest.approx(15.0)
        assert ec.fishing_selectivity_slope[0] == pytest.approx(3.0)

        # School at exactly L50
        state = _make_school(length=15.0)
        result = fishing_mortality(state, ec, n_subdt=10)
        dead_at_l50 = result.n_dead[0, MortalityCause.FISHING]

        # School well above L50 (selectivity ~ 1.0)
        state_big = _make_school(length=25.0)
        result_big = fishing_mortality(state_big, ec, n_subdt=10)
        dead_big = result_big.n_dead[0, MortalityCause.FISHING]

        # At L50, selectivity=0.5, so deaths should be roughly half
        np.testing.assert_allclose(dead_at_l50 / dead_big, 0.5, rtol=0.05)

    def test_sigmoid_small_fish_low_mortality(self, tmp_path):
        """Fish well below L50 experience very low fishing mortality."""
        cfg = _base_config(n_sp=1, n_dt=24)
        cfg.update(
            {
                "fisheries.enabled": "true",
                "simulation.nfisheries": "1",
                "fisheries.name.fsh0": "trawl",
                "fisheries.rate.base.fsh0": "0.5",
                "fisheries.selectivity.type.fsh0": "1",
                "fisheries.selectivity.l50.fsh0": "15.0",
                "fisheries.selectivity.slope.fsh0": "3.0",
            }
        )
        catch_path = tmp_path / "catch.csv"
        catch_path.write_text(",trawl\nFishA,1\n")
        cfg["fisheries.catchability.file"] = str(catch_path)
        cfg.pop("fishing.rate.sp0", None)

        ec = EngineConfig.from_dict(cfg)

        # Very small fish — sigmoid selectivity near 0
        state = _make_school(length=5.0)
        result = fishing_mortality(state, ec, n_subdt=10)
        dead = result.n_dead[0, MortalityCause.FISHING]

        # Full-size fish
        state_big = _make_school(length=25.0)
        result_big = fishing_mortality(state_big, ec, n_subdt=10)
        dead_big = result_big.n_dead[0, MortalityCause.FISHING]

        # Very small fish should have negligible fishing relative to big
        assert dead / dead_big < 0.01


# ===========================================================================
# 2.5 — MPA (Marine Protected Areas)
# ===========================================================================


class TestMPA:
    def _make_mpa_config(self, tmp_path, n_dt=4):
        """Config with a single MPA covering cell (0,0)."""
        cfg = _base_config(n_sp=1, n_dt=n_dt)
        # MPA grid: 2x2, only cell (0,0) is protected
        mpa_path = tmp_path / "mpa0.csv"
        # Note: spatial CSVs are flipped (south→north), so row 0 in file = top row
        # For a 2x2 grid: row0=[0,0], row1=[1,0] → after flipud: row0=[1,0], row1=[0,0]
        mpa_path.write_text("0;0\n1;0\n")
        cfg.update(
            {
                "mpa.file.mpa0": str(mpa_path),
                "mpa.start.year.mpa0": "0",
                "mpa.end.year.mpa0": "2",
                "mpa.percentage.mpa0": "0.8",
            }
        )
        return cfg

    def test_mpa_reduces_fishing_inside(self, tmp_path):
        """Inside MPA, fishing is reduced by percentage."""
        cfg = self._make_mpa_config(tmp_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.mpa_zones is not None
        assert len(ec.mpa_zones) == 1

        state = _make_school(cell_x=0, cell_y=0)  # inside MPA
        result = fishing_mortality(state, ec, n_subdt=10, step=0)
        dead_inside = result.n_dead[0, MortalityCause.FISHING]

        state_out = _make_school(cell_x=1, cell_y=1)  # outside MPA
        result_out = fishing_mortality(state_out, ec, n_subdt=10, step=0)
        dead_outside = result_out.n_dead[0, MortalityCause.FISHING]

        # Inside MPA: rate * (1 - 0.8) = 0.2 * normal
        assert dead_inside > 0  # still some fishing
        np.testing.assert_allclose(dead_inside / dead_outside, 0.2, rtol=0.05)

    def test_mpa_no_effect_outside(self, tmp_path):
        """Outside MPA grid, fishing is unaffected."""
        cfg = self._make_mpa_config(tmp_path)
        ec = EngineConfig.from_dict(cfg)

        # Cell (1,0): outside MPA (value=0)
        state = _make_school(cell_x=1, cell_y=1)
        result = fishing_mortality(state, ec, n_subdt=10, step=0)
        dead = result.n_dead[0, MortalityCause.FISHING]

        # Compare with no-MPA config
        cfg_no_mpa = _base_config(n_sp=1, n_dt=4)
        ec_no_mpa = EngineConfig.from_dict(cfg_no_mpa)
        result_no_mpa = fishing_mortality(state, ec_no_mpa, n_subdt=10, step=0)
        dead_no_mpa = result_no_mpa.n_dead[0, MortalityCause.FISHING]

        np.testing.assert_allclose(dead, dead_no_mpa, rtol=1e-10)

    def test_mpa_inactive_outside_period(self, tmp_path):
        """MPA has no effect before start or after end year."""
        cfg = _base_config(n_sp=1, n_dt=4)
        mpa_path = tmp_path / "mpa0.csv"
        mpa_path.write_text("0;0\n1;0\n")
        cfg.update(
            {
                "mpa.file.mpa0": str(mpa_path),
                "mpa.start.year.mpa0": "1",  # starts at year 1
                "mpa.end.year.mpa0": "2",
                "mpa.percentage.mpa0": "1.0",  # full closure
            }
        )
        ec = EngineConfig.from_dict(cfg)

        state = _make_school(cell_x=0, cell_y=0)  # inside MPA grid

        # Step 0 (year 0): MPA not yet active
        result_before = fishing_mortality(state, ec, n_subdt=10, step=0)
        dead_before = result_before.n_dead[0, MortalityCause.FISHING]

        # Step 4 (year 1): MPA active, full closure
        result_during = fishing_mortality(state, ec, n_subdt=10, step=4)
        dead_during = result_during.n_dead[0, MortalityCause.FISHING]

        assert dead_before > 0
        np.testing.assert_allclose(dead_during, 0.0)


# ===========================================================================
# 2.6 — Fishery Discards
# ===========================================================================


class TestDiscards:
    def test_discards_parsed_from_csv(self, tmp_path):
        """Discard rates loaded from CSV."""
        cfg = _base_config(n_sp=2, n_dt=4)
        discard_path = tmp_path / "discards.csv"
        # CSV: species × fishery discard rates (same format as catchability)
        discard_path.write_text(",fsh0,fsh1\nFishA,0.3,0.0\nFishB,0.0,0.5\n")
        cfg["fisheries.discards.file"] = str(discard_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.fishing_discard_rate is not None
        np.testing.assert_allclose(ec.fishing_discard_rate[0], 0.3)
        np.testing.assert_allclose(ec.fishing_discard_rate[1], 0.5)

    def test_discards_split_mortality(self, tmp_path):
        """Discard rate splits fishing deaths into FISHING and DISCARDS causes."""
        cfg = _base_config(n_sp=1, n_dt=4)
        discard_path = tmp_path / "discards.csv"
        discard_path.write_text(",fsh0\nFishA,0.4\n")
        cfg["fisheries.discards.file"] = str(discard_path)
        ec = EngineConfig.from_dict(cfg)

        state = _make_school()
        result = fishing_mortality(state, ec, n_subdt=10)

        fishing_dead = result.n_dead[0, MortalityCause.FISHING]
        discards_dead = result.n_dead[0, MortalityCause.DISCARDS]
        total = fishing_dead + discards_dead

        assert total > 0
        # 40% of total should be discards
        np.testing.assert_allclose(discards_dead / total, 0.4, rtol=1e-10)
        np.testing.assert_allclose(fishing_dead / total, 0.6, rtol=1e-10)
