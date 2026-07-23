"""Synthetic acceptance for the Phase 2a posterior density (no sampler)."""

from __future__ import annotations

import numpy as np

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.posterior import make_log_posterior

SIG_SEED = 0.02
EMU_VAR = 0.01
THETA_STAR = np.array([0.3, 0.6])


class _AnalyticEmulator:
    def __init__(self, w, b, var):
        self.w = np.asarray(w, float)
        self.b = b
        self.var = var

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return X @ self.w + self.b, np.full(len(X), self.var)


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


# Identifiable synthetic: 3 targets sensitive to distinct directions (n_targets >= d).
def _emulators():
    return {
        "A_biomass_mean": _AnalyticEmulator([1.0, 0.0], 2.0, EMU_VAR),
        "B_biomass_mean": _AnalyticEmulator([0.0, 1.0], 1.0, EMU_VAR),
        "C_biomass_mean": _AnalyticEmulator([1.0, 1.0], 0.5, EMU_VAR),
    }


def _targets(emulators, band=0.2, override=None):
    targets = []
    for key, emu in emulators.items():
        species = key.split("_")[0]
        mu_star, _ = emu.predict(THETA_STAR.reshape(1, -1))
        value = float(np.exp(mu_star[0] + 0.5 * SIG_SEED))  # so r(theta*) = 0
        if override and override[0] == species:
            value *= override[1]
        targets.append(
            BiomassTarget(
                species=species,
                target=value,
                lower=value * (1 - band),
                upper=value * (1 + band),
                reference_point_type="biomass",
            )
        )
    return targets


def _grid():
    g = np.linspace(0.02, 0.98, 49)
    return g, [np.array([a, b]) for b in g for a in g]


def _seed_by_key(emulators):
    return {key: SIG_SEED for key in emulators}


def test_gaussian_posterior_recovers_theta_star():
    emus = _emulators()
    logp = make_log_posterior(
        emus, _targets(emus), _fp2(), sigma_seed_sq_by_key=_seed_by_key(emus), likelihood="gaussian"
    )
    g, points = _grid()
    vals = np.array([logp(p) for p in points])
    best = points[int(np.argmax(vals))]
    assert np.allclose(best, THETA_STAR, atol=0.05)


def test_band_faithful_flat_inside_decays_outside():
    emus = _emulators()
    logp = make_log_posterior(
        emus,
        _targets(emus, band=0.4),
        _fp2(),
        sigma_seed_sq_by_key=_seed_by_key(emus),
        likelihood="band",
    )
    inside_a = logp(THETA_STAR)
    inside_b = logp(THETA_STAR + np.array([0.15, -0.15]))  # separated, still feasible
    outside = logp(np.array([0.9, 0.1]))
    assert abs(inside_a - inside_b) < 1.0  # flat across the interior plateau
    assert inside_a - outside > 5.0  # decays outside the feasible region


def test_misspecified_target_lowers_max_log_posterior():
    emus = _emulators()
    g, points = _grid()
    seed = _seed_by_key(emus)
    well = make_log_posterior(emus, _targets(emus), _fp2(), sigma_seed_sq_by_key=seed)
    bad = make_log_posterior(
        emus, _targets(emus, override=("A", 3.0)), _fp2(), sigma_seed_sq_by_key=seed
    )
    m_well = max(well(p) for p in points)
    m_bad = max(bad(p) for p in points)
    assert m_well - m_bad > 2.0
