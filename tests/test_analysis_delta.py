from pathlib import Path

import pandas as pd
import pytest

from osmose import analysis as az

_WIDE_FIXTURE = Path(__file__).parent / "fixtures" / "biomass_wide_sample.csv"


class _FakeResults:
    """Stand-in for OsmoseResults exposing the three metric accessors."""

    def __init__(self, frames):  # frames: {"biomass": df, ...}
        self._frames = frames
        self.output_dir = "<fake>"

    def biomass(self, species=None):
        return self._frames["biomass"]

    def yield_biomass(self, species=None):
        return self._frames["yield"]

    def abundance(self, species=None):
        return self._frames["abundance"]


def _wide(**species_to_series):
    n = len(next(iter(species_to_series.values())))
    d = {"Time": list(range(1, n + 1))}
    d.update(species_to_series)
    d["species"] = ["all"] * n
    return pd.DataFrame(d)


def test_window_mean_wide_format():
    df = _wide(cod=[10.0, 20.0, 30.0], herring=[100.0, 100.0, 100.0])
    res = _FakeResults({"biomass": df, "yield": df, "abundance": df})
    means = az._per_species_window_mean(res, "biomass", window_years=2)
    assert means["cod"] == pytest.approx(25.0)  # mean(20,30)
    assert means["herring"] == pytest.approx(100.0)
    assert "species" not in means and "Time" not in means


def test_window_mean_long_format():
    long = pd.DataFrame(
        {
            "time": [1, 2, 3, 1, 2, 3],
            "species": ["cod", "cod", "cod", "sprat", "sprat", "sprat"],
            "value": [10.0, 20.0, 30.0, 1.0, 1.0, 1.0],
        }
    )
    res = _FakeResults({"biomass": long, "yield": long, "abundance": long})
    means = az._per_species_window_mean(res, "biomass", window_years=2)
    assert means["cod"] == pytest.approx(25.0)
    assert means["sprat"] == pytest.approx(1.0)


def test_window_mean_real_wide_fixture():
    df = pd.read_csv(_WIDE_FIXTURE)
    res = _FakeResults({"biomass": df, "yield": df, "abundance": df})
    means = az._per_species_window_mean(res, "biomass", window_years=10)
    assert "cod" in means and means["cod"] > 0
    assert "species" not in means  # the constant 'species' artifact column is excluded


def test_window_mean_uses_years_not_row_count():
    # 3 years at 2 rows/year. window=1 must take the LAST YEAR (Time>2.0 → rows at 2.5,3.0),
    # NOT the last ROW. cod last-year rows = [30,40] → mean 35 (a row-count tail(1) would give 40).
    df = _wide(cod=[10.0, 10.0, 20.0, 20.0, 30.0, 40.0])
    df["Time"] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    res = _FakeResults({"biomass": df, "yield": df, "abundance": df})
    means = az._per_species_window_mean(res, "biomass", window_years=1)
    assert means["cod"] == pytest.approx(35.0)  # by-year window; tail(1)=40 would be WRONG


def test_window_mean_rejects_nonpositive_window():
    df = _wide(cod=[10.0, 20.0, 30.0])
    res = _FakeResults({"biomass": df, "yield": df, "abundance": df})
    with pytest.raises(ValueError):
        az._per_species_window_mean(res, "biomass", window_years=0)  # empty window → NaN guard
