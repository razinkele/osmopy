import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_baltic_thermal_sr_series",
    Path(__file__).resolve().parent.parent / "scripts" / "build_baltic_thermal_sr_series.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_quarter_means():
    monthly = {mo: float(mo) for mo in range(1, 13)}
    assert m.quarter_mean(monthly, 3) == 8.0  # (7+8+9)/3
    assert m.quarter_mean(monthly, 4) == 11.0  # (10+11+12)/3


def test_assemble_series_layout():
    hist = {1993 + i: 10.0 + i * 0.1 for i in range(31)}
    rows = m.assemble_series(hist, tref=11.5)
    assert len(rows) == 50
    assert rows[0] == (1974, 11.5) and rows[18] == (1992, 11.5)
    assert rows[19] == (1993, 10.0) and rows[-1] == (2023, 13.0)
    years = [y for y, _ in rows]
    assert years == list(range(1974, 2024))  # contiguous — the loader requires it


def test_assemble_series_rejects_gap():
    import pytest

    hist = {1993: 10.0, 1995: 10.2}
    with pytest.raises(ValueError):
        m.assemble_series(hist, tref=10.0)


def test_write_series_csv_no_comments(tmp_path):
    p = tmp_path / "s.csv"
    rows = [(1974 + i, 7.0) for i in range(3)]
    m.write_series_csv(p, {0: rows, 1: [(y, 8.0) for y, _ in rows]})
    text = p.read_text()
    assert "#" not in text  # comments crash the loader
    assert text.splitlines()[0] == "year,temp_sp0,temp_sp1"
    assert text.splitlines()[1] == "1974,7.0,8.0"


def test_assemble_series_rejects_mismatched_first_year():
    import pytest

    hist = {1995: 10.0, 1996: 10.1, 1997: 10.2}  # first year is 1995, not 1993
    with pytest.raises(ValueError, match="does not match expected"):
        m.assemble_series(hist, tref=10.0, first_hist_year=1993)
