import math

from osmose.calibration.larva_recal import solve_larva_rate

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
