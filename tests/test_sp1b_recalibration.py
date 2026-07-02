import math

import numba
import pytest

from osmose.calibration import larva_recal
from osmose.calibration.larva_recal import (
    RECAL_RATE,
    e_clip_first_guess,
    mean_cod,
    solve_larva_rate,
    sp1_on_config,
    with_determinism,
)
from osmose.config import OsmoseConfigReader

GRID = [0.0, 5.0, 10.0, 15.0]


def test_solve_bisects_to_interior_root():
    # mean decreases with rate: mean(d) = 100 - 4d; baseline 70 -> root at d=7.5.
    # grid f = mean-baseline = [30, 10, -10, -30] -> exactly one crossing in (5, 10).
    r = solve_larva_rate(70.0, lambda d: 100.0 - 4.0 * d, grid_points=GRID, tol=1e-4)
    assert r.feasible and r.converged
    assert r.rate is not None and abs(r.rate - 7.5) < 0.05
    assert abs(r.mean_on - 70.0) / 70.0 <= 1e-4


def test_solve_near_zero_shortcircuit_returns_grid_point():
    # grid includes the exact root 7.5 -> short-circuit, iters=0.
    r = solve_larva_rate(70.0, lambda d: 100.0 - 4.0 * d, grid_points=[0.0, 7.5, 15.0], tol=0.02)
    assert r.feasible and r.converged and r.iters == 0
    assert r.rate == 7.5


def test_solve_d0_already_within_tol_means_no_recalibration():
    # SP1 barely moved the mean: mean(d0=15)=40 == baseline 40 -> near-zero hit at the last
    # grid point -> rate == d0, no recalibration. (mean(0)=100, mean(7.5)=70 are far off.)
    r = solve_larva_rate(40.0, lambda d: 100.0 - 4.0 * d, grid_points=[0.0, 7.5, 15.0], tol=0.02)
    assert r.feasible and r.converged and r.rate == 15.0


def test_solve_infeasible_zero_crossings():
    # every grid mean is far below baseline -> baseline unreachable -> feasible=False.
    r = solve_larva_rate(200.0, lambda d: 50.0 - d, grid_points=GRID, tol=0.02)
    assert not r.feasible and r.rate is None and "0 sign change" in r.message


def test_solve_infeasible_multiple_crossings():
    # f = 30*cos(d), baseline 100: grid values give two sign changes -> ambiguous.
    r = solve_larva_rate(
        100.0, lambda d: 100.0 + 30.0 * math.cos(d), grid_points=[1.0, 2.5, 4.0, 5.5], tol=0.02
    )
    assert not r.feasible and r.rate is None and "2 sign change" in r.message


def test_solve_max_iter_not_converged_reports_best():
    # baseline 71 -> root at d=7.25 (NON-dyadic, so bisection midpoints never hit it exactly);
    # impossibly tight tol + tiny max_iter -> feasible bracket but converged=False.
    r = solve_larva_rate(71.0, lambda d: 100.0 - 4.0 * d, grid_points=GRID, tol=1e-12, max_iter=2)
    assert r.feasible and not r.converged and r.rate is not None
    assert 5.0 < r.rate < 10.0


SP_FIELD = "data/baltic/forcing/baltic_rv_field.nc"
SPAWN = "data/baltic/maps/cod_spawning.csv"


def test_e_clip_first_guess_bounds():
    d1, e_clip = e_clip_first_guess(SP_FIELD, SPAWN, d0=15.0)
    assert 0.0 < e_clip < 1.0  # some but not all viable
    assert 0.0 <= d1 <= 15.0  # a valid rate inside the bracket
    # d1 = d0 + ln(e_clip); ln(e_clip) < 0 so d1 < d0
    assert d1 < 15.0
    assert abs(d1 - max(0.0, 15.0 + math.log(e_clip))) < 1e-9


RATE_KEY = "mortality.additional.larva.rate.sp0"
DET_KEYS = ("movement.randomseed.fixed", "stochastic.mortality.randomseed.fixed")


def test_with_determinism_sets_both_keys_without_mutating():
    base = {"a": "1"}
    out = with_determinism(base)
    assert all(out[k] == "true" for k in DET_KEYS)
    assert "a" in out and base == {"a": "1"}  # original untouched


def test_sp1_on_config_flags_and_determinism():
    cfg = sp1_on_config({"x": "y"}, SP_FIELD, larva_rate=None)
    assert cfg["reproduction.rv.spatial.enabled"] == "true"
    assert cfg["reproduction.rv.spatial.field.file"] == SP_FIELD
    assert cfg["reproduction.rv.spatial.species.enabled.sp0"] == "true"
    assert all(cfg[k] == "true" for k in DET_KEYS)


def test_sp1_on_config_none_omits_rate_key():
    cfg = sp1_on_config({}, SP_FIELD, larva_rate=None)
    assert RATE_KEY not in cfg  # infeasible path: base d0 stands


def test_sp1_on_config_value_sets_rate_key():
    cfg = sp1_on_config({}, SP_FIELD, larva_rate=12.5)
    assert float(cfg[RATE_KEY]) == 12.5


def test_sp1_on_config_default_reads_current_recal_rate(monkeypatch):
    monkeypatch.setattr(larva_recal, "RECAL_RATE", 9.0)
    cfg = sp1_on_config({}, SP_FIELD)  # default -> current module RECAL_RATE
    assert float(cfg[RATE_KEY]) == 9.0
    monkeypatch.setattr(larva_recal, "RECAL_RATE", None)
    cfg2 = sp1_on_config({}, SP_FIELD)
    assert RATE_KEY not in cfg2


BALTIC = "data/baltic/baltic_all-parameters.csv"


def _baltic_15yr():
    cfg = dict(OsmoseConfigReader().read(BALTIC))
    cfg["simulation.time.nyear"] = "15"
    return cfg


def test_sp1b_mean_neutral_drift_guard():
    numba.set_num_threads(1)  # runtime determinism pin (config keys added by the helpers)
    if RECAL_RATE is None:
        pytest.skip(
            "SP1b infeasible: RECAL_RATE is None (see docs/diagnostics/sp1b_recalibration.md)"
        )
    base = _baltic_15yr()
    baseline = mean_cod(with_determinism(base))
    on = mean_cod(sp1_on_config(base, SP_FIELD))  # default larva_rate -> RECAL_RATE
    assert abs(on - baseline) / baseline <= 0.02
