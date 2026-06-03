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


def test_compute_balance_real_fixture(tmp_path):
    mort_dir = tmp_path / "Mortality"
    mort_dir.mkdir(parents=True)
    (mort_dir / "osm_mortalityRate-cod_Simu0.csv").write_bytes(_FIXTURE.read_bytes())
    out = fz.compute_mortality_balance(
        tmp_path, prefix="osm", species_list=["cod"], steps_per_year=1, window_years=5
    )
    b = {x.species: x for x in out}["cod"]
    assert b.fishing_mortality >= 0.0 and b.natural_mortality >= 0.0
    if b.natural_mortality > 0:
        assert b.f_over_m == pytest.approx(b.fishing_mortality / b.natural_mortality)
        assert b.overexploited == (b.f_over_m > 1.0)


def test_compute_balance_m_zero_gives_none(tmp_path, monkeypatch):
    def fake_reader(path):
        cols = pd.MultiIndex.from_tuples(
            [("F", "Recruits"), ("Mpred", "Recruits"), ("Mstarv", "Recruits"), ("Madd", "Recruits")]
        )
        return pd.DataFrame([[0.4, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]], columns=cols)

    monkeypatch.setattr(fz, "read_mortality_recruits", fake_reader)
    (tmp_path / "Mortality").mkdir()
    (tmp_path / "Mortality" / "osm_mortalityRate-x_Simu0.csv").write_text("stub")
    b = fz.compute_mortality_balance(
        tmp_path, prefix="osm", species_list=["x"], steps_per_year=1, window_years=2
    )[0]
    assert b.natural_mortality == 0.0 and b.f_over_m is None and b.overexploited is False


def test_compute_balance_skips_missing_species(tmp_path):
    (tmp_path / "Mortality").mkdir()
    out = fz.compute_mortality_balance(
        tmp_path, prefix="osm", species_list=["ghost"], steps_per_year=1, window_years=2
    )
    assert out == []


def test_compute_balance_skips_malformed_file(tmp_path):
    # A 2-line malformed file makes read_mortality_recruits raise ParserError;
    # compute must catch it and skip, not crash.
    (tmp_path / "Mortality").mkdir()
    (tmp_path / "Mortality" / "osm_mortalityRate-bad_Simu0.csv").write_text("preamble\nh1,h2\n")
    out = fz.compute_mortality_balance(
        tmp_path, prefix="osm", species_list=["bad"], steps_per_year=1, window_years=2
    )
    assert out == []  # malformed → WARN-skip, not crash


def test_discover_species_from_mortality_dir(tmp_path):
    d = tmp_path / "Mortality"
    d.mkdir()
    for sp in ("cod", "sprat"):
        (d / f"osm_mortalityRate-{sp}_Simu0.csv").write_text("stub")
    assert sorted(fz.discover_species(tmp_path, prefix="osm")) == ["cod", "sprat"]


def test_format_report_renders_with_none_fm():
    bals = [
        fz.MortalityBalance("cod", 0.4, 0.2, 2.0, True),
        fz.MortalityBalance("x", 0.4, 0.0, None, False),  # M=0 → "—"
    ]
    md = fz.format_mortality_report(bals)
    assert "cod" in md and "x" in md
    assert "F/M" in md
    assert "2.00" in md and "—" in md
    assert "Recruits-stage" in md
    assert "1 overexploited" in md


def test_results_mortality_reads_real_csv(tmp_path):
    from osmose.results import OsmoseResults

    mdir = tmp_path / "Mortality"
    mdir.mkdir()
    (mdir / "osm_mortalityRate-cod_Simu0.csv").write_bytes(_FIXTURE.read_bytes())
    r = OsmoseResults(tmp_path, prefix="osm", strict=False)
    df = r.mortality("cod")  # must NOT raise ParserError
    assert df is not None and len(df) > 0


def test_cli_runs_on_empty_dir(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("cmb", "scripts/compute_mortality_balance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "Mortality").mkdir()
    rc = mod.main(["--results-dir", str(tmp_path), "--prefix", "osm", "--steps-per-year", "1"])
    assert rc == 0


def test_fm_bar_chart_builds():
    from osmose import plotting

    bals = [
        fz.MortalityBalance("cod", 0.4, 0.2, 2.0, True),
        fz.MortalityBalance("x", 0.1, 0.5, 0.2, False),
    ]
    fig = plotting.make_fm_ratio_bars(bals)
    assert fig is not None
    assert sum(len(t.x) for t in fig.data if hasattr(t, "x") and t.x is not None) >= 1
    # a reference line at y=1 exists (add_hline adds a shape with y0==y1==1.0)
    assert any(getattr(s, "y0", None) == 1.0 for s in fig.layout.shapes)
