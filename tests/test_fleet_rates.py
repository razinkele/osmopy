"""The fleet-rate solver: coarse sweep + lower-root bisection, on synthetic catch curves."""

from __future__ import annotations

import math

import pandas as pd

from osmose.calibration.fleet_rates import solve_rate, trailing_catch


def dome(r: float) -> float:
    """Catch vs rate with a peak at r = 1 (catch ~368): the shape of yield past F_MSY."""
    return 1000.0 * r * math.exp(-r)


def test_solves_on_ascending_limb():
    s = solve_rate(dome, current=0.1, target=300.0)
    assert s.reachable
    assert abs(s.catch / 300.0 - 1) <= 0.10
    assert s.rate < 1.0


def test_takes_lower_root_when_two_exist():
    s = solve_rate(dome, current=2.0, target=300.0)
    assert s.reachable and s.rate < 1.0


def test_lowering_rate_past_peak():
    # catch at 3.0 is ~149: the target 100 has its lower root near 0.11; never assume direction
    s = solve_rate(dome, current=3.0, target=100.0)
    assert s.reachable and s.rate < 1.0
    assert abs(s.catch / 100.0 - 1) <= 0.10


def test_target_above_peak_returns_peak():
    s = solve_rate(dome, current=0.3, target=500.0)
    assert not s.reachable
    assert 0.5 < s.rate < 2.0
    assert "peak" in s.note


def test_noisy_sweep_without_crossing_returns_closest():
    # flat-ish below the target everywhere left of a spike: no adjacent pair brackets it
    def lumpy(r: float) -> float:
        return 450.0 if 0.9 < r < 1.1 else 50.0

    s = solve_rate(lumpy, current=0.2, target=300.0)
    assert not s.reachable


def test_all_zero_catch():
    s = solve_rate(lambda r: 0.0, current=0.1, target=300.0)
    assert not s.reachable


def test_trailing_catch():
    df = pd.DataFrame({"Time": range(10), "cod_west": [float(i) for i in range(10)]})
    assert trailing_catch(df, "cod_west") == 7.0


def test_evaluation_budget():
    calls = []

    def counted(r: float) -> float:
        calls.append(r)
        return dome(r)

    solve_rate(counted, current=0.1, target=300.0, sweep_n=5, max_bisect=8)
    assert len(calls) <= 5 + 1 + 8
