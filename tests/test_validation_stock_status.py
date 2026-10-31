import csv
import math
from pathlib import Path

import pandas as pd
import pytest

from osmose.validation import fisheries as fis
from osmose.validation import stock_status as ss
from osmose.validation.fisheries_reference import ReferencePoint


def test_annual_by_year_sum_and_mean():
    # 2 saved rows in year 0 (Time 0.0, 0.5), 1 in year 1 (Time 1.0)
    time = [0.0, 0.5, 1.0]
    assert fis.annual_by_year([2.0, 3.0, 10.0], time, how="sum") == {0: 5.0, 1: 10.0}
    assert fis.annual_by_year([2.0, 4.0, 10.0], time, how="mean") == {0: 3.0, 1: 10.0}


def test_annual_by_year_one_row_per_year_identity():
    assert fis.annual_by_year([10.0, 20.0], [0.0, 1.0], how="sum") == {0: 10.0, 1: 20.0}


# ---------------------------------------------------------------------------
# Task 5: stock_status tests
# ---------------------------------------------------------------------------


class _Cfg:
    n_dt_per_year = 1
    output_record_frequency = 1


class _FakeResults:
    """Minimal results stub: ssb() wide-form (Time + species col).

    Carries production-matching ``output_dir`` and ``prefix`` so that
    ``_exploited_f_by_year`` reaches the file-read path (and fails on the
    None/missing path) rather than on a missing attribute.
    """

    output_dir = None  # no real run → mortalityRate path → TypeError → graceful WARN
    prefix = "run"

    def __init__(self, ssb_rows, time=(0.0, 1.0)):
        self._ssb = list(ssb_rows)
        self._time = list(time)

    def ssb(self, species=None):
        return pd.DataFrame({"Time": self._time, "cod": self._ssb})


def test_quadrant_and_ratios():
    refs = {"cod": ReferencePoint(species="cod", fmsy=0.3, bmsy=100.0, b_ref_kind="bmsy_user")}
    statuses = ss.compute_stock_status(
        _FakeResults([120.0, 80.0]),
        refs,
        _Cfg(),
        species_list=["cod"],
        _f_override={"cod": {0: 0.15, 1: 0.45}},
    )
    s = statuses[0]
    # year0: B/Bmsy=1.2, F/Fmsy=0.5 → green; year1: 0.8, 1.5 → red
    assert s.b_over_bmsy == [1.2, 0.8]
    assert s.f_over_fmsy == pytest.approx([0.5, 1.5])
    assert s.latest_quadrant == "red"
    assert s.takeaway is not None


def test_ssb_annual_mean_over_subannual_rows():
    # 2 saved rows in year 0 (Time 0.0, 0.5) → MEAN 110, 1 row in year 1 → 80 (NOT last-row)
    res = _FakeResults([100.0, 120.0, 80.0], time=(0.0, 0.5, 1.0))
    refs = {"cod": ReferencePoint(species="cod", bmsy=100.0, b_ref_kind="bmsy_user")}
    s = ss.compute_stock_status(res, refs, _Cfg(), species_list=["cod"])[0]
    assert s.b_over_bmsy == [1.1, 0.8]


def test_data_limited_single_axis():
    refs = {"cod": ReferencePoint(species="cod", fmsy=0.3)}  # no bmsy → no B-axis
    statuses = ss.compute_stock_status(
        _FakeResults([120.0, 80.0]),
        refs,
        _Cfg(),
        species_list=["cod"],
        _f_override={"cod": {0: 0.15, 1: 0.45}},
    )
    s = statuses[0]
    assert all(v is None for v in s.b_over_bmsy)
    assert s.latest_quadrant is None  # needs both axes
    assert any("Bmsy" in c for c in s.caveats)


_PY_FIXTURE = Path(__file__).parent / "fixtures" / "mortalityRate_python_engine_sprat.csv"


class _DirResults:
    prefix = "osm"

    def __init__(self, output_dir):
        self.output_dir = output_dir


def _stage_mortality_dir(tmp_path, text):
    (tmp_path / "Mortality").mkdir()
    (tmp_path / "Mortality" / "osm_mortalityRate-sprat_Simu0.csv").write_text(text)
    return _DirResults(tmp_path)


def test_exploited_f_reads_python_engine_labels(tmp_path):
    """Real Python-engine file (Fishing x Eggs/Juvenil/Adult). Before 2026-10-31 only the
    Java-4.3 labels (F x Pre-recruits/Recruits) matched, so F came back None and the Kobe F-axis
    was silently empty. Expected: the Adult column per year, read with the csv module."""
    rows = list(csv.reader(_PY_FIXTURE.open()))
    col = list(zip(rows[1], rows[2])).index(("Fishing", "Adult"))
    expected: dict[int, float] = {}
    for r in rows[3:]:  # SUM within a year: rows can be sub-annual (this fixture is annual)
        year = math.floor(float(r[0]))
        expected[year] = expected.get(year, 0.0) + float(r[col])
    caveats: list[str] = []
    res = _stage_mortality_dir(tmp_path, _PY_FIXTURE.read_text())
    f_map = ss._exploited_f_by_year(res, "sprat", caveats)
    assert f_map is not None
    assert f_map == pytest.approx(expected)
    assert any("'Adult'" in c for c in caveats)  # Juvenil also carries a little F


def test_exploited_f_without_fishing_column_is_a_caveat(tmp_path):
    caveats: list[str] = []
    res = _stage_mortality_dir(tmp_path, _PY_FIXTURE.read_text().replace("Fishing", "Mystery"))
    assert ss._exploited_f_by_year(res, "sprat", caveats) is None
    assert any("no fishing-mortality column" in c for c in caveats)
