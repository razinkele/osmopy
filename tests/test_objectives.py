import numpy as np
import pandas as pd
import pytest
from osmose.calibration.objectives import (
    abundance_rmse,
    biomass_rmse,
    diet_distance,
    normalized_rmse,
)


def test_biomass_rmse_identical():
    df = pd.DataFrame({"time": range(10), "biomass": [100.0] * 10})
    assert biomass_rmse(df, df) == 0.0


def test_biomass_rmse_different():
    sim = pd.DataFrame({"time": range(10), "biomass": [100.0] * 10})
    obs = pd.DataFrame({"time": range(10), "biomass": [110.0] * 10})
    assert biomass_rmse(sim, obs) == pytest.approx(10.0)


def test_biomass_rmse_with_species_filter():
    sim = pd.DataFrame({"time": [0, 0], "biomass": [100, 200], "species": ["A", "B"]})
    obs = pd.DataFrame({"time": [0, 0], "biomass": [110, 200], "species": ["A", "B"]})
    assert biomass_rmse(sim, obs, species="A") == pytest.approx(10.0)
    assert biomass_rmse(sim, obs, species="B") == pytest.approx(0.0)


def test_biomass_rmse_no_overlap():
    sim = pd.DataFrame({"time": [0, 1], "biomass": [100, 200]})
    obs = pd.DataFrame({"time": [5, 6], "biomass": [100, 200]})
    assert biomass_rmse(sim, obs) == float("inf")


def test_diet_distance_identical():
    df = pd.DataFrame({"prey1": [0.5, 0.3], "prey2": [0.5, 0.7]})
    assert diet_distance(df, df) == 0.0


def test_diet_distance_different():
    sim = pd.DataFrame({"prey1": [1.0, 0.0], "prey2": [0.0, 1.0]})
    obs = pd.DataFrame({"prey1": [0.0, 0.0], "prey2": [0.0, 0.0]})
    result = diet_distance(sim, obs)
    assert result == pytest.approx(np.sqrt(2.0))


def test_abundance_rmse_identical():
    df = pd.DataFrame({"time": range(5), "abundance": [500.0] * 5})
    assert abundance_rmse(df, df) == 0.0


def test_abundance_rmse_different():
    sim = pd.DataFrame({"time": range(5), "abundance": [500.0] * 5})
    obs = pd.DataFrame({"time": range(5), "abundance": [520.0] * 5})
    assert abundance_rmse(sim, obs) == pytest.approx(20.0)


def test_abundance_rmse_with_species_filter():
    sim = pd.DataFrame({"time": [0, 0], "abundance": [100, 200], "species": ["A", "B"]})
    obs = pd.DataFrame({"time": [0, 0], "abundance": [110, 200], "species": ["A", "B"]})
    assert abundance_rmse(sim, obs, species="A") == pytest.approx(10.0)
    assert abundance_rmse(sim, obs, species="B") == pytest.approx(0.0)


def test_abundance_rmse_no_overlap():
    sim = pd.DataFrame({"time": [0, 1], "abundance": [100, 200]})
    obs = pd.DataFrame({"time": [5, 6], "abundance": [100, 200]})
    assert abundance_rmse(sim, obs) == float("inf")


def test_diet_distance_shape_mismatch():
    sim = pd.DataFrame({"prey1": [0.5, 0.3], "prey2": [0.5, 0.7]})
    obs = pd.DataFrame({"prey1": [0.5]})
    assert diet_distance(sim, obs) == float("inf")


def test_normalized_rmse():
    sim = np.array([100, 110, 90])
    obs = np.array([100, 100, 100])
    result = normalized_rmse(sim, obs)
    expected = np.sqrt(np.mean([0, 100, 100])) / 100
    assert result == pytest.approx(expected)


def test_normalized_rmse_zero_mean():
    sim = np.array([1.0, 2.0])
    obs = np.array([0.0, 0.0])
    assert normalized_rmse(sim, obs) == float("inf")


# --- yield_rmse tests ---


def test_yield_rmse_identical():
    from osmose.calibration.objectives import yield_rmse

    df = pd.DataFrame({"time": range(5), "yield": [1000.0] * 5})
    assert yield_rmse(df, df) == 0.0


def test_yield_rmse_different():
    from osmose.calibration.objectives import yield_rmse

    sim = pd.DataFrame({"time": range(5), "yield": [100.0] * 5})
    obs = pd.DataFrame({"time": range(5), "yield": [120.0] * 5})
    assert yield_rmse(sim, obs) == pytest.approx(20.0)


def test_yield_rmse_with_species_filter():
    from osmose.calibration.objectives import yield_rmse

    sim = pd.DataFrame({"time": [0, 0], "yield": [100, 200], "species": ["A", "B"]})
    obs = pd.DataFrame({"time": [0, 0], "yield": [110, 200], "species": ["A", "B"]})
    assert yield_rmse(sim, obs, species="A") == pytest.approx(10.0)
    assert yield_rmse(sim, obs, species="B") == pytest.approx(0.0)


def test_yield_rmse_no_overlap():
    from osmose.calibration.objectives import yield_rmse

    sim = pd.DataFrame({"time": [0, 1], "yield": [100, 200]})
    obs = pd.DataFrame({"time": [5, 6], "yield": [100, 200]})
    assert yield_rmse(sim, obs) == float("inf")


# --- catch_at_size_distance tests ---


def test_catch_at_size_distance_identical():
    from osmose.calibration.objectives import catch_at_size_distance

    df = pd.DataFrame({"time": [0, 0, 1, 1], "bin": [0, 1, 0, 1], "value": [10, 20, 30, 40]})
    assert catch_at_size_distance(df, df) == 0.0


def test_catch_at_size_distance_different():
    from osmose.calibration.objectives import catch_at_size_distance

    sim = pd.DataFrame({"time": [0, 1], "bin": [0, 0], "value": [10.0, 20.0]})
    obs = pd.DataFrame({"time": [0, 1], "bin": [0, 0], "value": [15.0, 25.0]})
    assert catch_at_size_distance(sim, obs) == pytest.approx(5.0)


def test_catch_at_size_distance_no_overlap():
    from osmose.calibration.objectives import catch_at_size_distance

    sim = pd.DataFrame({"time": [0], "bin": [0], "value": [10.0]})
    obs = pd.DataFrame({"time": [5], "bin": [5], "value": [10.0]})
    assert catch_at_size_distance(sim, obs) == float("inf")


# --- size_at_age_rmse tests ---


def test_size_at_age_rmse_identical():
    from osmose.calibration.objectives import size_at_age_rmse

    df = pd.DataFrame({"time": [0, 1], "bin": [0, 1], "value": [5.0, 10.0]})
    assert size_at_age_rmse(df, df) == 0.0


def test_size_at_age_rmse_different():
    from osmose.calibration.objectives import size_at_age_rmse

    sim = pd.DataFrame({"time": [0, 1], "bin": [0, 1], "value": [5.0, 10.0]})
    obs = pd.DataFrame({"time": [0, 1], "bin": [0, 1], "value": [8.0, 13.0]})
    assert size_at_age_rmse(sim, obs) == pytest.approx(3.0)


def test_size_at_age_rmse_no_overlap():
    from osmose.calibration.objectives import size_at_age_rmse

    sim = pd.DataFrame({"time": [0], "bin": [0], "value": [5.0]})
    obs = pd.DataFrame({"time": [9], "bin": [9], "value": [5.0]})
    assert size_at_age_rmse(sim, obs) == float("inf")


# --- weighted_multi_objective tests ---


def test_weighted_multi_objective():
    from osmose.calibration.objectives import weighted_multi_objective

    objectives = [0.5, 1.0, 2.0]
    weights = [1.0, 2.0, 0.5]
    assert weighted_multi_objective(objectives, weights) == pytest.approx(3.5)


def test_weighted_multi_objective_uniform():
    from osmose.calibration.objectives import weighted_multi_objective

    objectives = [3.0, 4.0]
    weights = [1.0, 1.0]
    assert weighted_multi_objective(objectives, weights) == pytest.approx(7.0)
