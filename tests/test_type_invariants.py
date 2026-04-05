"""Tests that data types reject invalid construction arguments."""

import numpy as np
import pytest


def test_mpa_zone_percentage_bounds():
    from osmose.engine.config import MPAZone
    with pytest.raises(ValueError, match="percentage"):
        MPAZone(percentage=1.5, start_year=0, end_year=10, grid=np.ones((2, 2)))


def test_mpa_zone_year_order():
    from osmose.engine.config import MPAZone
    with pytest.raises(ValueError, match="start_year"):
        MPAZone(percentage=0.5, start_year=10, end_year=5, grid=np.ones((2, 2)))


def test_resource_species_size_order():
    from osmose.engine.resources import ResourceSpeciesInfo
    with pytest.raises(ValueError, match="size_min"):
        ResourceSpeciesInfo(name="test", size_min=10.0, size_max=5.0,
                           trophic_level=2.0, accessibility=0.5)


def test_resource_species_accessibility_bounds():
    from osmose.engine.resources import ResourceSpeciesInfo
    with pytest.raises(ValueError, match="accessibility"):
        ResourceSpeciesInfo(name="test", size_min=1.0, size_max=10.0,
                           trophic_level=2.0, accessibility=1.5)


def test_background_species_proportion_sum():
    from osmose.engine.background import BackgroundSpeciesInfo
    with pytest.raises(ValueError, match="proportions"):
        BackgroundSpeciesInfo(
            name="test", species_index=0, file_index=0, n_class=2,
            lengths=[5.0, 10.0], trophic_levels=[2.0, 2.5],
            ages_dt=[0, 12], condition_factor=0.01, allometric_power=3.0,
            size_ratio_min=[0.5], size_ratio_max=[2.0],
            ingestion_rate=3.5, multiplier=1.0, offset=0.0, forcing_nsteps_year=24,
            proportions=[0.3, 0.3],  # sums to 0.6, not ~1.0
        )


def test_background_species_length_count_mismatch():
    from osmose.engine.background import BackgroundSpeciesInfo
    with pytest.raises(ValueError, match="n_class"):
        BackgroundSpeciesInfo(
            name="test", species_index=0, file_index=0, n_class=3,
            lengths=[5.0, 10.0], trophic_levels=[2.0, 2.5],  # only 2, not 3
            ages_dt=[0, 12], condition_factor=0.01, allometric_power=3.0,
            size_ratio_min=[0.5], size_ratio_max=[2.0],
            ingestion_rate=3.5, multiplier=1.0, offset=0.0, forcing_nsteps_year=24,
            proportions=[0.5, 0.5],
        )


def test_calibration_phase_empty_params():
    from osmose.calibration.multiphase import CalibrationPhase
    with pytest.raises(ValueError, match="free_params"):
        CalibrationPhase(free_params=[], algorithm="differential_evolution", max_iter=100)
