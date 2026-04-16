"""Tests for additional mortality variants (SP-2)."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import additional_mortality, larva_mortality
from osmose.engine.state import MortalityCause, SchoolState
from osmose.engine.timeseries import ByClassTimeSeries


def _write_csv(path: Path, header: list[str], rows: list[list[str]], sep: str = ";") -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _make_config(**overrides) -> MagicMock:
    """Create minimal EngineConfig mock for additional mortality tests."""
    config = MagicMock(spec=EngineConfig)
    config.n_species = 1
    config.n_dt_per_year = 24
    config.additional_mortality_rate = np.array([0.1])
    config.additional_mortality_by_dt = None
    config.additional_mortality_by_dt_by_class = None
    config.additional_mortality_spatial = None
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_state(
    n: int = 1, sp: int = 0, abundance: float = 1000.0, age_dt: int = 48, length: float = 20.0
) -> SchoolState:
    state = SchoolState.create(n_schools=n, species_id=np.full(n, sp, dtype=np.int32))
    return state.replace(
        abundance=np.full(n, abundance),
        weight=np.full(n, 0.01),
        length=np.full(n, length),
        age_dt=np.full(n, age_dt, dtype=np.int32),
        cell_x=np.zeros(n, dtype=np.int32),
        cell_y=np.zeros(n, dtype=np.int32),
    )


class TestByDtByClassAdditionalMortality:
    """Rate varies per (dt, age/size class) from ByClassTimeSeries."""

    def test_young_gets_low_rate(self, tmp_path: Path) -> None:
        """Young school (age class 0) gets rate from first column.

        CSV thresholds are in dt units (already converted by config loader
        for byAge: years * ndt_per_year). For testing, we use dt directly.
        """
        csv_file = tmp_path / "mort.csv"
        # Thresholds already in dt (as if config loader converted from years)
        _write_csv(
            csv_file,
            ["step", "0", "48"],
            [
                ["0", "0.1", "0.5"],
            ],
        )
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=1)
        config = _make_config(additional_mortality_by_dt_by_class=[ts])

        state = _make_state(age_dt=10)  # age_dt=10 → class 0 (< 48)
        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead = result.n_dead[0, MortalityCause.ADDITIONAL]
        assert dead > 0
        assert dead < 10

    def test_old_gets_high_rate(self, tmp_path: Path) -> None:
        """Old school (age class 1) gets rate from second column."""
        csv_file = tmp_path / "mort.csv"
        _write_csv(
            csv_file,
            ["step", "0", "48"],
            [
                ["0", "0.1", "2.0"],
            ],
        )
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=1)
        config = _make_config(additional_mortality_by_dt_by_class=[ts])

        state = _make_state(age_dt=100)  # age_dt=100 → class 1 (≥ 48)
        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead = result.n_dead[0, MortalityCause.ADDITIONAL]
        assert dead > 50

    def test_below_first_threshold_gets_zero(self, tmp_path: Path) -> None:
        """Age below first threshold → rate 0 (Java: return 0)."""
        csv_file = tmp_path / "mort.csv"
        _write_csv(
            csv_file,
            ["step", "24", "48"],
            [
                ["0", "0.5", "1.0"],
            ],
        )
        ts = ByClassTimeSeries.from_csv(csv_file, ndt_per_year=1, ndt_simu=1)
        config = _make_config(additional_mortality_by_dt_by_class=[ts])

        state = _make_state(age_dt=10)  # below first threshold (24)
        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead = result.n_dead[0, MortalityCause.ADDITIONAL]
        assert dead == 0.0


class TestByDtLarvaMortality:
    """Larval rate varies per time step."""

    def test_time_varying_larva_rate(self, tmp_path: Path) -> None:
        """Larval rate changes per step from CSV."""
        csv_file = tmp_path / "larva.csv"
        # Step 0: rate 0.5, Step 1: rate 2.0
        _write_csv(csv_file, ["step", "value"], [["0", "0.5"], ["1", "2.0"]])

        from osmose.engine.timeseries import SingleTimeSeries

        ts = SingleTimeSeries.from_csv(csv_file, ndt_per_year=2, ndt_simu=2)
        config = _make_config(
            larva_mortality_rate=np.array([0.1]),  # base rate (overridden)
            larva_mortality_by_dt=[ts],
        )

        # Create egg state
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([10000.0]),
            weight=np.array([0.0001]),
            length=np.array([0.1]),
            age_dt=np.array([0], dtype=np.int32),
            cell_x=np.zeros(1, dtype=np.int32),
            cell_y=np.zeros(1, dtype=np.int32),
            is_egg=np.array([True]),
        )

        # Step 0: rate 0.5
        result0 = larva_mortality(state, config, step=0)
        dead0 = result0.n_dead[0, MortalityCause.ADDITIONAL]

        # Step 1: rate 2.0 → higher mortality
        result1 = larva_mortality(state, config, step=1)
        dead1 = result1.n_dead[0, MortalityCause.ADDITIONAL]

        assert dead1 > dead0

    def test_bytdt_typo_key_detected(self) -> None:
        """Config detection supports both bytDt (Java typo) and byDt."""
        from osmose.engine.config import _detect_larva_by_dt_key

        cfg1 = {"mortality.additional.larva.rate.bytDt.file.sp0": "/path.csv"}
        assert _detect_larva_by_dt_key(cfg1, 0) == "/path.csv"

        cfg2 = {"mortality.additional.larva.rate.byDt.file.sp0": "/other.csv"}
        assert _detect_larva_by_dt_key(cfg2, 0) == "/other.csv"


class TestSpatialAdditionalMortality:
    """Spatial factor multiplies additional mortality per cell."""

    def test_spatial_factor_applied(self) -> None:
        """Schools in high-mortality cells get more deaths."""
        # Spatial map: cell (0,0)=2.0 (double rate), cell (0,1)=0.0 (no mortality)
        spatial_map = np.array([[2.0, 0.0]])  # shape (1, 2)
        config = _make_config(
            additional_mortality_rate=np.array([1.0]),  # base rate
            additional_mortality_spatial=[spatial_map],
        )

        # Two schools: one in cell (0,0), one in cell (0,1)
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([0.01, 0.01]),
            length=np.array([20.0, 20.0]),
            age_dt=np.array([48, 48], dtype=np.int32),
            cell_x=np.array([0, 1], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )

        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead_high = result.n_dead[0, MortalityCause.ADDITIONAL]
        dead_zero = result.n_dead[1, MortalityCause.ADDITIONAL]

        assert dead_high > 0
        assert dead_zero == 0.0  # factor = 0 → no mortality

    def test_no_spatial_means_uniform(self) -> None:
        """Without spatial factor, mortality is uniform across cells."""
        config = _make_config(
            additional_mortality_rate=np.array([0.5]),
            additional_mortality_spatial=None,
        )
        state = _make_state(n=2, abundance=1000.0, age_dt=48)
        state = state.replace(
            cell_x=np.array([0, 5], dtype=np.int32),
            cell_y=np.array([0, 5], dtype=np.int32),
        )
        result = additional_mortality(state, config, n_subdt=1, step=0)
        dead0 = result.n_dead[0, MortalityCause.ADDITIONAL]
        dead1 = result.n_dead[1, MortalityCause.ADDITIONAL]
        assert dead0 == pytest.approx(dead1, rel=1e-10)
