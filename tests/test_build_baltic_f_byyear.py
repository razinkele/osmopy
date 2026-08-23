"""Unit tests for the F1 by-year derivation (spec decisions 2, 3, 5).
Fixtures are tiny synthetic snapshot dicts — no I/O beyond tmp_path."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_baltic_f_byyear",
    Path(__file__).resolve().parent.parent / "scripts" / "build_baltic_f_byyear.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_anchor_mean_uses_available_years_only():
    f = {2018: 1.0, 2019: 1.2, 2020: 0.8, 2021: 1.0}  # 2022 missing (cod_west case)
    assert m.anchor_mean(f) == 1.0


def test_hold_last_fills_trailing_gap():
    f = {1993: 0.5, 1994: 0.7}
    assert m.hold_last(f, [1993, 1994, 1995, 1996]) == [0.5, 0.7, 0.7, 0.7]


def test_factor_series_is_anchored():
    f = {y: 2.0 for y in range(1993, 2024)}
    for y in range(2018, 2023):
        f[y] = 4.0
    fac = m.factor_series(f)
    assert fac[0] == 0.5   # 1993: 2.0 / anchor 4.0
    assert fac[27] == 1.0  # 2020 is inside the anchor window: 4.0 / 4.0


def test_factor_series_final_year():
    f = {y: 1.0 for y in range(1993, 2024)}
    assert m.factor_series(f) == [1.0] * 31


def test_herring_factors_are_scale_free():
    """An index-scaled stock (F around 1) and an absolute stock (F around 0.2)
    with identical PATTERNS and equal catches must give the same aggregate as
    either stock alone — the scale must cancel (spec decision 3)."""
    years = m.YEARS
    pattern = {y: 1.0 + 0.5 * ((y % 5) - 2) / 2 for y in years}
    f_index = dict(pattern)
    f_abs = {y: 0.2 * v for y, v in pattern.items()}
    catches = {y: 100.0 for y in years}
    agg = m.herring_factor_series([(f_index, catches), (f_abs, catches)])
    solo = m.factor_series(f_index)
    assert all(abs(a - s) < 1e-12 for a, s in zip(agg, solo))


def test_build_rows_layout_and_verbatim_spinup():
    rows = m.build_rows("0.3799687566571175", [1.0] * 31)
    assert len(rows) == 50
    assert rows[:19] == ["0.3799687566571175"] * 19   # verbatim string
    assert float(rows[19]) == 0.3799687566571175      # repr round-trips exactly


def test_herring_factors_weighting_unequal():
    """3:1 catch weights, hand-computed value (transposition/indexing canary —
    herring is a pass/fail stock)."""
    years = m.YEARS
    f_flat = {y: 1.0 for y in years}                 # factor 1.0 everywhere
    f_step = {y: 1.0 for y in years}                 # factor 0.5 outside the anchor
    for y in years:
        if not (2018 <= y <= 2022):
            f_step[y] = 0.5
    big = {y: 300.0 for y in years}
    small = {y: 100.0 for y in years}
    agg = m.herring_factor_series([(f_flat, big), (f_step, small)])
    assert agg[0] == 0.875                            # (3*1.0 + 1*0.5) / 4, exact in FP
    assert agg[years.index(2020)] == 1.0              # inside the anchor window


def test_no_flounder_in_stocks():
    assert 3 not in m.STOCKS  # spec decision 5
