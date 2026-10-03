"""WGSAM SMS cod-predation-mortality (M2) comparison (issue #136).

SMS M2 is cod-only, at age, numbers basis. OSMOSE reports a lumped predation rate, so the
comparable model quantity is cod-ATTRIBUTED M2 on a biomass basis: annual tonnes of the prey
eaten by the cod stocks (predatorPressure) over the prey's mean biomass. The SMS side is M2 at
age weighted by biomass at age from the same key run. Report-only, no gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from osmose.validation.ices import (
    PredationM2Comparison,
    SmsM2Snapshot,
    compare_predation_m2_to_sms,
    format_markdown_report,
    load_sms_m2,
    model_cod_attributed_m2,
    model_predation_rate_by_stage,
    sms_m2_weighted_by_year,
)

REAL_SNAPSHOTS = Path("data/baltic/reference/ices_snapshots")


# ── synthetic snapshot ────────────────────────────────────────────────────────


def _make_sms_snapshot(tmp_path: Path) -> Path:
    d = tmp_path / "snap"
    d.mkdir()
    (d / "index.json").write_text(
        json.dumps(
            {
                "model_species_to_ices_stocks": {"herring": [], "sprat": []},
                "sms_m2": {
                    "files": {"m2_annual": "m2.csv", "weights": "w.csv"},
                    "scenario_used": "2025 key run",
                    "sms_species_to_model_species": {"Herring": "herring", "Sprat": "sprat"},
                    "sms_species_to_ices_stocks": {
                        "Herring": "her.27.25-2932",
                        "Sprat": "spr.27.22-32",
                    },
                },
            }
        )
    )
    rows = [
        # Year, scenario, Species, variable, Age, value
        (2020, "2025 key run", "Herring", "M2", 0, 0.40),
        (2020, "2025 key run", "Herring", "M2", 1, 0.20),
        (2020, "2025 key run", "Herring", "M2", 2, 0.10),
        (2021, "2025 key run", "Herring", "M2", 0, 0.60),
        (2021, "2025 key run", "Herring", "M2", 1, 0.30),
        (2021, "2025 key run", "Herring", "M2", 2, 0.10),
        (2022, "2025 key run", "Herring", "M2", 1, -1.0),  # projection sentinel
        (2020, "2022 Key run", "Herring", "M2", 1, 0.99),  # other scenario, ignored
        (2020, "2025 key run", "Sprat", "M2", 1, 0.05),
        (2020, "2025 key run", "Sprat", "M2", 2, 0.15),
    ]
    pd.DataFrame(rows, columns=["Year", "scenario", "Species", "variable", "Age", "value"]).to_csv(
        d / "m2.csv", index=False
    )
    w = [
        (2020, "Herring", 0, 0.0, 0.01, 0.0),
        (2020, "Herring", 1, 100.0, 0.02, 20.0),
        (2020, "Herring", 2, 100.0, 0.04, 80.0),
        (2021, "Herring", 1, 100.0, 0.02, 50.0),
        (2021, "Herring", 2, 100.0, 0.04, 50.0),
        (2020, "Sprat", 1, 10.0, 0.01, 1.0),
        (2020, "Sprat", 2, 10.0, 0.01, 3.0),
    ]
    with (d / "w.csv").open("w") as f:
        f.write("# provenance comment line\n")
        pd.DataFrame(w, columns=["Year", "Species", "Age", "N", "west", "BIO"]).to_csv(
            f, index=False
        )
    return d


def test_load_sms_m2_selects_scenario_and_drops_sentinel(tmp_path):
    snap = load_sms_m2(_make_sms_snapshot(tmp_path))
    assert isinstance(snap, SmsM2Snapshot)
    assert set(snap.m2["scenario"]) == {"2025 key run"}
    assert (snap.m2["value"] >= 0).all()  # the -1 projection row is gone
    assert 2022 not in set(snap.m2[snap.m2.Species == "Herring"]["Year"])
    assert list(snap.weights.columns) == ["Year", "Species", "Age", "N", "west", "BIO"]
    assert snap.meta["scenario_used"] == "2025 key run"


def test_sms_m2_weighted_by_year_biomass_basis_hand_calc(tmp_path):
    snap = load_sms_m2(_make_sms_snapshot(tmp_path))
    by_year = sms_m2_weighted_by_year(snap, "Herring", basis="biomass")
    # 2020: ages 1,2 with BIO 20, 80 -> (0.2*20 + 0.1*80)/100 = 0.12 ; age 0 has BIO 0 -> no weight
    # 2021: BIO 50, 50 -> (0.3*50 + 0.1*50)/100 = 0.20
    assert by_year == {2020: pytest.approx(0.12), 2021: pytest.approx(0.20)}
    by_n = sms_m2_weighted_by_year(snap, "Herring", basis="numbers")
    assert by_n[2020] == pytest.approx((0.2 * 100 + 0.1 * 100) / 200)


def test_sms_m2_weighted_rejects_unknown_basis(tmp_path):
    snap = load_sms_m2(_make_sms_snapshot(tmp_path))
    with pytest.raises(ValueError, match="basis"):
        sms_m2_weighted_by_year(snap, "Herring", basis="volume")


# ── model side ────────────────────────────────────────────────────────────────


def _fake_results(
    pressure: pd.DataFrame, biomass_wide: pd.DataFrame, mort: dict[str, pd.DataFrame]
) -> MagicMock:
    mock = MagicMock()
    mock.predator_pressure = lambda: pressure
    mock.biomass = lambda species=None: biomass_wide if species is None else None
    mock.mortality_rate = lambda species=None: mort[species]
    mock.output_dir = Path("/tmp/fake-results")
    return mock


def _pressure_frame() -> pd.DataFrame:
    # Yearly records (Time 0.958 ...), per-step MEAN tonnes eaten (ndt = 24 per year).
    rows = []
    for yr in range(5):
        t = yr + 23 / 24
        # cod_west eats 1 t/step of sprat, cod_east 2 t/step; perch 0.5 t/step (not cod)
        rows.append([t, "sprat", 1.0, 0.5, 2.0])
        rows.append([t, "herring", 0.25, 0.0, 0.25])
    return pd.DataFrame(rows, columns=["Time", "Prey", "cod_west", "perch", "cod_east"])


def _biomass_wide() -> pd.DataFrame:
    t = [yr + 23 / 24 for yr in range(5)]
    return pd.DataFrame(
        {"Time": t, "sprat": [1000.0] * 5, "herring": [600.0] * 5, "species": ["all"] * 5}
    )


def test_model_cod_attributed_m2_is_annual_cod_consumption_over_prey_biomass():
    res = _fake_results(_pressure_frame(), _biomass_wide(), {})
    m2 = model_cod_attributed_m2(
        res, "sprat", predators=("cod_west", "cod_east"), window_years=3, n_dt_per_year=24
    )
    # (1 + 2) t/step * 24 steps = 72 t/yr eaten by cods; / 1000 t = 0.072 per year, each year
    assert m2.mean == pytest.approx(0.072)
    assert m2.min == pytest.approx(0.072) and m2.max == pytest.approx(0.072)
    assert m2.n_years == 3
    assert m2.by_predator == {"cod_west": pytest.approx(0.024), "cod_east": pytest.approx(0.048)}


def test_model_cod_attributed_m2_ignores_non_cod_predators_and_other_prey():
    res = _fake_results(_pressure_frame(), _biomass_wide(), {})
    m2 = model_cod_attributed_m2(res, "herring", predators=("cod_west",), window_years=5)
    assert m2.mean == pytest.approx(0.25 * 24 / 600.0)


def test_model_cod_attributed_m2_raises_when_predator_column_missing():
    res = _fake_results(_pressure_frame(), _biomass_wide(), {})
    with pytest.raises(KeyError, match="cod_south"):
        model_cod_attributed_m2(res, "sprat", predators=("cod_south",), window_years=2)


def _mortality_two_row(pred_adult: float, pred_juv: float) -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples(
        [
            ("Time", ""),
            ("Predation", "Eggs"),
            ("Predation", "Juvenil"),
            ("Predation", "Adult"),
            ("Fishing", "Adult"),
        ]
    )
    rows = [[yr + 23 / 24, 5.0, pred_juv, pred_adult, 0.3] for yr in range(5)]
    return pd.DataFrame(rows, columns=cols)


def test_model_predation_rate_by_stage_reads_two_row_header_window_mean():
    res = _fake_results(_pressure_frame(), _biomass_wide(), {"sprat": _mortality_two_row(0.5, 1.2)})
    rates = model_predation_rate_by_stage(res, "sprat", window_years=3)
    assert rates == {"Juvenil": pytest.approx(1.2), "Adult": pytest.approx(0.5)}


# ── comparison + report ───────────────────────────────────────────────────────


def test_compare_predation_m2_builds_one_row_per_sms_species(tmp_path):
    snap = load_sms_m2(_make_sms_snapshot(tmp_path))
    res = _fake_results(
        _pressure_frame(),
        _biomass_wide(),
        {"sprat": _mortality_two_row(0.5, 1.2), "herring": _mortality_two_row(0.1, 0.2)},
    )
    rows = compare_predation_m2_to_sms(
        res, snap, predators=("cod_west", "cod_east"), window_years=5, ices_window=range(2020, 2022)
    )
    assert [r.model_species for r in rows] == ["herring", "sprat"]
    h = rows[0]
    assert isinstance(h, PredationM2Comparison)
    assert h.sms_stock == "her.27.25-2932"
    assert h.sms_m2_mean == pytest.approx((0.12 + 0.20) / 2)
    assert h.sms_m2_min == pytest.approx(0.12) and h.sms_m2_max == pytest.approx(0.20)
    assert h.sms_n_years == 2
    assert h.model_cod_m2_mean == pytest.approx((0.25 + 0.25) * 24 / 600.0)  # both cods
    assert h.ratio == pytest.approx(h.model_cod_m2_mean / h.sms_m2_mean)
    assert h.model_predation_rate_by_stage["Adult"] == pytest.approx(0.1)
    s = rows[1]
    assert s.sms_n_years == 1 and s.sms_m2_mean == pytest.approx((0.05 * 1 + 0.15 * 3) / 4)


def test_compare_predation_m2_tolerates_missing_model_output(tmp_path):
    snap = load_sms_m2(_make_sms_snapshot(tmp_path))
    res = MagicMock()
    res.predator_pressure = MagicMock(side_effect=FileNotFoundError("no predatorPressure"))
    res.output_dir = Path("/tmp/fake")
    rows = compare_predation_m2_to_sms(res, snap, window_years=5, ices_window=range(2020, 2022))
    assert [r.model_species for r in rows] == ["herring", "sprat"]
    assert all(r.model_cod_m2_mean is None and r.ratio is None for r in rows)
    assert all(r.sms_m2_mean is not None for r in rows)  # the SMS side still reports


def test_report_has_m2_section_only_when_given(tmp_path):
    snap = load_sms_m2(_make_sms_snapshot(tmp_path))
    res = _fake_results(
        _pressure_frame(),
        _biomass_wide(),
        {"sprat": _mortality_two_row(0.5, 1.2), "herring": _mortality_two_row(0.1, 0.2)},
    )
    rows = compare_predation_m2_to_sms(res, snap, window_years=5, ices_window=range(2020, 2022))
    plain = format_markdown_report([], window_years=5, ices_window=range(2020, 2022))
    assert "SMS" not in plain
    md = format_markdown_report(
        [], window_years=5, ices_window=range(2020, 2022), m2_comparisons=rows
    )
    assert "WGSAM SMS cod-predation mortality" in md
    assert "her.27.25-2932" in md and "spr.27.22-32" in md
    assert "biomass basis" in md  # the basis caveat is stated, not hidden
    assert "Juvenil" in md and "Adult" in md  # total predation rate shown beside the cod share
    assert "report-only" in md.lower() or "not gated" in md.lower()


# ── the committed snapshot ────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (REAL_SNAPSHOTS / "wgsam_sms_baltic_2025.m2_annual.csv").exists(),
    reason="committed WGSAM SMS snapshot missing",
)
def test_committed_sms_snapshot_loads_and_recent_m2_is_plausible():
    snap = load_sms_m2(REAL_SNAPSHOTS)
    assert snap.meta["commit"] == "f690d4ff"
    assert set(snap.m2["Species"]) == {"Herring", "Sprat"}
    assert snap.m2["Year"].max() == 2024  # 2025 is the sentinel projection year, dropped
    for sp in ("Herring", "Sprat"):
        by_year = sms_m2_weighted_by_year(snap, sp, basis="biomass")
        recent = [by_year[y] for y in range(2018, 2023)]
        # cod-predation M2 on the clupeids is low in the 2018-2022 window (eastern cod collapsed):
        # ~0.08-0.11 per year; historical peaks were ~0.5 (herring) / ~0.85 (sprat) in the 1980s.
        assert all(0.03 < v < 0.3 for v in recent), (sp, recent)
        assert max(by_year.values()) > 0.3, sp
