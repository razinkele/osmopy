"""Tests for the posterior-predictive diagnostic layer (Goal 2)."""

from __future__ import annotations

import dataclasses

import numpy as np

from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.predictive import (
    EmulatorPredictiveRanges,
    marginal_coverage,
    posterior_predictive,
)
from osmose.calibration.uq.sampler import SamplerResult


class _LinEmulator:
    """Injected emulator: mean = slope * theta0 + intercept, tiny fixed variance."""

    def __init__(self, slope=5.0, intercept=0.0, var=1e-4):
        self.slope, self.intercept, self.var = slope, intercept, var

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return X[:, 0] * self.slope + self.intercept, np.full(len(X), self.var)


def _result(samples, weights):
    return SamplerResult(
        samples=samples, weights=weights, logz=0.0, logz_err=0.1, ess=100.0, converged=True
    )


def test_posterior_predictive_resamples_by_weight():
    # theta uniformly spread; weights heavy on HIGH theta. A monotone-increasing
    # emulator maps high theta -> high y, so weighting must shift the predictive
    # median UP vs uniform weighting. (Uniform-weight code would ignore this.)
    n = 200
    samples = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    emus = {"cod_biomass_mean": _LinEmulator()}
    ss = {"cod_biomass_mean": 0.01}
    weighted = posterior_predictive(
        _result(samples, samples[:, 0] ** 4 + 1e-9), emus, ["cod_biomass_mean"], ss, seed=0
    )
    uniform = posterior_predictive(
        _result(samples, np.ones(n)), emus, ["cod_biomass_mean"], ss, seed=0
    )
    assert (
        weighted.log_ranges["cod_biomass_mean"][1] > uniform.log_ranges["cod_biomass_mean"][1] + 0.5
    )


def test_posterior_predictive_biomass_is_exp_of_log_and_ordered():
    n = 100
    samples = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    r = posterior_predictive(
        _result(samples, np.ones(n)), {"k": _LinEmulator()}, ["k"], {"k": 0.02}, seed=0
    )
    lo, med, hi = r.log_ranges["k"]
    assert lo < med < hi
    blo, bmed, bhi = r.biomass_ranges["k"]
    assert np.allclose([blo, bmed, bhi], np.exp([lo, med, hi]))


def test_posterior_predictive_is_marginal_only_no_joint_field():
    n = 50
    samples = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    r = posterior_predictive(
        _result(samples, np.ones(n)), {"k": _LinEmulator()}, ["k"], {"k": 0.02}, seed=0
    )
    field_names = {f.name for f in dataclasses.fields(r)}
    # Structural marginal-only guard: no joint/per-draw samples are exposed.
    assert field_names == {
        "keys",
        "log_ranges",
        "biomass_ranges",
        "cross_species_correlation",
        "level",
    }
    assert not hasattr(r, "samples") and not hasattr(r, "draws")


def test_posterior_predictive_cross_species_correlation_matrix():
    n = 300
    samples = np.column_stack([np.linspace(0, 1, n), np.linspace(1, 0, n)])
    emus = {
        "a_biomass_mean": _LinEmulator(slope=5.0),
        "b_biomass_mean": _LinEmulator(slope=-5.0, intercept=5.0),
    }
    r = posterior_predictive(
        _result(samples, np.ones(n)),
        emus,
        ["a_biomass_mean", "b_biomass_mean"],
        {"a_biomass_mean": 1e-3, "b_biomass_mean": 1e-3},
        seed=0,
    )
    c = r.cross_species_correlation
    assert c.shape == (2, 2)
    assert np.allclose(np.diag(c), 1.0)
    assert np.allclose(c, c.T)


def _ranges(biomass):
    return EmulatorPredictiveRanges(
        keys=list(biomass),
        log_ranges={k: tuple(np.log(v)) for k, v in biomass.items()},
        biomass_ranges=biomass,
        cross_species_correlation=np.array([[1.0]]),
        level=0.9,
    )


def test_marginal_coverage_target_within_range_is_covered():
    ranges = _ranges({"cod_biomass_mean": (8.0, 10.0, 13.0)})
    target = BiomassTarget("cod", 10.0, 8.0, 12.0, reference_point_type="biomass")
    assert marginal_coverage(ranges, [target]) == {"cod_biomass_mean": True}


def test_marginal_coverage_target_outside_range_is_not_covered():
    ranges = _ranges({"cod_biomass_mean": (8.0, 10.0, 13.0)})
    target = BiomassTarget("cod", 20.0, 16.0, 24.0, reference_point_type="biomass")  # 20 > 13
    assert marginal_coverage(ranges, [target]) == {"cod_biomass_mean": False}
