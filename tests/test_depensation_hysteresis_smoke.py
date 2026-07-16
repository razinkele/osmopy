"""Smoke tests for the depensation hysteresis validation harness (SP1 Unit 3). The full
F-ramp is a multi-decade real-engine run (Task 8) — the pure fold-point logic is tested here
(CI-safe); the engine path is a short CI-skipped check."""

import os
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def test_fold_points_loop_and_reachability():
    from validate_depensation_hysteresis import fold_points

    # up-leg (low->high F): collapses at f_mult=4 (ssb 3000 < 6000 collapse threshold)
    up = [
        (0.5, 120_000, True),
        (1, 100_000, True),
        (2, 60_000, True),
        (4, 3_000, True),
        (8, 500, True),
    ]
    # down-leg (high->low F): recovers at f_mult=2 (ssb 45000 > 40000 band floor); F_recover < F_collapse
    down = [
        (8, 500, True),
        (4, 1_500, True),
        (2, 45_000, True),
        (1, 90_000, True),
        (0.5, 110_000, True),
    ]
    fc, fr, unconv = fold_points(up, down, base_f=0.08)
    assert fc == pytest.approx(0.32)  # 4 x 0.08
    assert fr == pytest.approx(0.16)  # 2 x 0.08
    assert fr < fc  # a genuine hysteresis loop
    assert unconv == []


def test_fold_points_flags_unconverged():
    from validate_depensation_hysteresis import fold_points

    up = [(0.5, 100_000, True), (4, 3_000, False)]  # fold-adjacent level did not converge
    down = [(4, 1_000, True), (0.5, 90_000, True)]
    _fc, _fr, unconv = fold_points(up, down, base_f=0.08)
    assert 4 in unconv


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="multi-decade real-engine run")
def test_equilibrate_level_returns_tuple():
    from baltic_bistability_chunk0 import read_base_config, read_base_larva_rates
    from validate_depensation_hysteresis import equilibrate_level, resolve_base_f

    base = read_base_config()
    base_f = resolve_base_f(base)
    mean, conv, series = equilibrate_level(
        base, read_base_larva_rates(base), 0.85, 90_000.0, 4.0, base_f, 1.0, 0
    )
    assert mean >= 0.0 and isinstance(conv, bool) and len(series) > 0
