from pathlib import Path

import pandas as pd
import pytest

from osmose.validation import fisheries as fz

_FIXTURE = Path(__file__).parent / "fixtures" / "mortalityRate_sample.csv"


def test_read_mortality_recruits_real_csv():
    df = fz.read_mortality_recruits(_FIXTURE)
    assert ("F", "Recruits") in df.columns
    assert ("Mpred", "Recruits") in df.columns
    assert ("Mstarv", "Recruits") in df.columns
    assert ("Madd", "Recruits") in df.columns
    assert len(df) > 0
    assert df[("F", "Recruits")].notna().all()


def test_annual_rate_steps_per_year_1():
    s = pd.Series([0.1, 0.2, 0.3])
    assert fz.annual_rate(s, steps_per_year=1, window_years=2) == pytest.approx(0.25)


def test_annual_rate_steps_per_year_2():
    # 6 rows, spy=2 → annual = [0.3, 0.7, 1.1]; window 2 → mean(0.7,1.1)=0.9
    s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert fz.annual_rate(s, steps_per_year=2, window_years=2) == pytest.approx(0.9)


def test_annual_rate_drops_trailing_partial_year():
    # 5 rows, spy=2 → 2 full years [0.3,0.7]; trailing partial (row 4) dropped
    s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    assert fz.annual_rate(s, steps_per_year=2, window_years=2) == pytest.approx(
        0.5
    )  # mean(0.3,0.7)


def test_annual_rate_raises_when_shorter_than_a_year():
    with pytest.raises(ValueError):
        fz.annual_rate(pd.Series([0.1]), steps_per_year=2, window_years=2)


def test_annual_rate_rejects_bad_steps_per_year():
    with pytest.raises(ValueError):
        fz.annual_rate(pd.Series([0.1, 0.2]), steps_per_year=0, window_years=1)
