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


def test_run_delta_ranks_by_pct():
    base = _FakeResults(
        {
            "biomass": _wide(cod=[100.0, 100.0], herring=[50.0, 50.0]),
            "yield": _wide(cod=[1.0, 1.0]),
            "abundance": _wide(cod=[1.0, 1.0]),
        }
    )
    var = _FakeResults(
        {
            "biomass": _wide(cod=[110.0, 110.0], herring=[100.0, 100.0]),
            "yield": _wide(cod=[1.0, 1.0]),
            "abundance": _wide(cod=[1.0, 1.0]),
        }
    )
    deltas = az.run_delta(base, var, metric="biomass", window_years=2)
    by = {d.species: d for d in deltas}
    assert by["cod"].abs_delta == pytest.approx(10.0)
    assert by["cod"].pct_delta == pytest.approx(0.10)
    assert by["herring"].pct_delta == pytest.approx(1.0)  # 50→100 = +100%
    # ranked by |pct_delta| desc → herring (1.0) before cod (0.10)
    assert [d.species for d in deltas][:2] == ["herring", "cod"]


def test_run_delta_top_n():
    # Names chosen so ALPHABETICAL order (a,b,c) DISAGREES with the pct ranking — a broken
    # sort that just preserves alpha order would fail this. a=+10%, b=+100%, c=+50%.
    base = _FakeResults(
        {
            "biomass": _wide(a=[1.0], b=[1.0], c=[1.0]),
            "yield": _wide(a=[1.0]),
            "abundance": _wide(a=[1.0]),
        }
    )
    var = _FakeResults(
        {
            "biomass": _wide(a=[1.1], b=[2.0], c=[1.5]),
            "yield": _wide(a=[1.0]),
            "abundance": _wide(a=[1.0]),
        }
    )
    deltas = az.run_delta(base, var, metric="biomass", window_years=1, top_n=2)
    assert len(deltas) == 2
    assert [d.species for d in deltas] == ["b", "c"]  # +100%, +50% — NOT alphabetical


def test_run_delta_from_zero_ranks_above_finite():
    # from-zero recovery must outrank a finite +200% mover.
    base = _FakeResults(
        {
            "biomass": _wide(cod=[0.0], herring=[1.0]),
            "yield": _wide(cod=[0.0]),
            "abundance": _wide(cod=[0.0]),
        }
    )
    var = _FakeResults(
        {
            "biomass": _wide(cod=[10.0], herring=[3.0]),
            "yield": _wide(cod=[0.0]),
            "abundance": _wide(cod=[0.0]),
        }
    )
    deltas = az.run_delta(base, var, metric="biomass", window_years=1)
    assert deltas[0].species == "cod" and deltas[0].from_zero is True
    assert deltas[1].species == "herring"


def test_run_delta_both_zero_ranks_last():
    # A 0->0 "dead" species (pct None but NOT from_zero) must rank LAST, never as a top mover.
    base = _FakeResults(
        {
            "biomass": _wide(cod=[1.0], ghost=[0.0]),
            "yield": _wide(cod=[0.0]),
            "abundance": _wide(cod=[0.0]),
        }
    )
    var = _FakeResults(
        {
            "biomass": _wide(cod=[2.0], ghost=[0.0]),
            "yield": _wide(cod=[0.0]),
            "abundance": _wide(cod=[0.0]),
        }
    )
    deltas = az.run_delta(base, var, metric="biomass", window_years=1)
    assert deltas[0].species == "cod"  # +100% mover on top
    assert deltas[-1].species == "ghost"  # 0->0 dead species last
    ghost = deltas[-1]
    assert ghost.pct_delta is None and ghost.from_zero is False and ghost.abs_delta == 0.0


def test_run_delta_from_zero():
    base = _FakeResults(
        {"biomass": _wide(cod=[0.0, 0.0]), "yield": _wide(cod=[0.0]), "abundance": _wide(cod=[0.0])}
    )
    var = _FakeResults(
        {"biomass": _wide(cod=[5.0, 5.0]), "yield": _wide(cod=[0.0]), "abundance": _wide(cod=[0.0])}
    )
    d = az.run_delta(base, var, metric="biomass", window_years=2)[0]
    assert d.species == "cod"
    assert d.baseline_mean == 0.0 and d.variant_mean == pytest.approx(5.0)
    assert d.pct_delta is None and d.from_zero is True
    assert d.abs_delta == pytest.approx(5.0)


def test_run_delta_union_species_present_in_one_run():
    base = _FakeResults(
        {"biomass": _wide(cod=[10.0]), "yield": _wide(cod=[1.0]), "abundance": _wide(cod=[1.0])}
    )
    var = _FakeResults(
        {
            "biomass": _wide(cod=[10.0], newsp=[7.0]),
            "yield": _wide(cod=[1.0]),
            "abundance": _wide(cod=[1.0]),
        }
    )
    by = {d.species: d for d in az.run_delta(base, var, metric="biomass", window_years=1)}
    assert by["newsp"].baseline_mean == 0.0 and by["newsp"].variant_mean == pytest.approx(7.0)
    assert by["newsp"].from_zero is True


def test_run_delta_metric_switch():
    # yield differs while biomass is identical → metric="yield" must pick up the change
    base = _FakeResults(
        {"biomass": _wide(cod=[10.0]), "yield": _wide(cod=[2.0]), "abundance": _wide(cod=[1.0])}
    )
    var = _FakeResults(
        {"biomass": _wide(cod=[10.0]), "yield": _wide(cod=[4.0]), "abundance": _wide(cod=[1.0])}
    )
    d = {x.species: x for x in az.run_delta(base, var, metric="yield", window_years=1)}["cod"]
    assert d.pct_delta == pytest.approx(1.0)


def test_format_delta_report():
    deltas = [
        az.SpeciesDelta("herring", 50.0, 100.0, 50.0, 1.0, False),
        az.SpeciesDelta("cod", 0.0, 5.0, 5.0, None, True),
    ]
    md = az.format_delta_report(deltas, metric="biomass", window_years=10)
    assert "herring" in md and "cod" in md
    assert "biomass" in md
    assert "+100.0%" in md or "100.0%" in md  # herring pct
    assert "from 0" in md  # cod from-zero note
    assert "B/Bmsy" not in md  # sanity: not the fisheries report


def test_delta_chart_builds():
    from osmose import plotting
    from osmose import analysis as az

    deltas = [
        az.SpeciesDelta("herring", 50.0, 100.0, 50.0, 1.0, False),
        az.SpeciesDelta("cod", 100.0, 90.0, -10.0, -0.10, False),
        az.SpeciesDelta("sprat", 0.0, 5.0, 5.0, None, True),  # from-zero: no finite bar
    ]
    fig = plotting.make_run_delta_chart(deltas)
    assert fig is not None
    # EXACTLY the 2 finite-pct species are barred (herring, cod); the from-zero sprat is NOT.
    assert sum(len(t.x) for t in fig.data if hasattr(t, "x") and t.x is not None) == 2
    assert "sprat" not in [s for t in fig.data if t.y is not None for s in t.y]


def test_cli_self_comparison_is_all_zero(tmp_path):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("cr", "scripts/compare_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "delta.json"
    # compare a real run against ITSELF → every delta must be exactly 0
    rc = mod.main(
        [
            "--baseline",
            "data/eec_full/output",
            "--variant",
            "data/eec_full/output",
            "--prefix",
            "eec",
            "--metric",
            "biomass",
            "--window-years",
            "10",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    rows = json.loads(out.read_text())
    assert len(rows) > 0  # species were actually compared
    assert all(r["abs_delta"] == 0.0 for r in rows)  # genuine self-comparison → zero deltas
    # pct is 0.0 for nonzero-baseline species; None for any zero-baseline species (robust either way)
    assert all(r["pct_delta"] in (0.0, None) for r in rows)
