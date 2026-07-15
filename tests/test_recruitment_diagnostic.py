"""Recruitment diagnostic helpers (Spec 2)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# scripts/ modules use BARE sibling imports (e.g. `from calibrate_baltic import ...`), so scripts/
# must be on sys.path and we import UNQUALIFIED — mirrors tests/test_fr_diagnostic.py. A dotted
# `from scripts.evaluate_calibration_vs_ices import ...` fails at collection (ModuleNotFoundError:
# calibrate_baltic), because a package-qualified import never puts scripts/ itself on sys.path.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluate_calibration_vs_ices import (  # noqa: E402
    _ices_recruitment_geomean,
    _recruitment_verdict,
    _species_recruitment_age,
)


def test_recruitment_age_sprat_herring_clean_cod_flounder_none():
    assert _species_recruitment_age("sprat") == "1"
    assert _species_recruitment_age("herring") == "0"
    assert _species_recruitment_age("cod") is None  # stocks disagree (age 0 vs 1)
    assert _species_recruitment_age("flounder") is None  # no recruitment_age


def test_sprat_geomean_matches_independent_computation():
    from validate_baltic_vs_ices_sag import WINDOW_YEARS, _load_assessment, _series_by_year

    rec = _series_by_year(_load_assessment("spr.27.22-32"), "recruitment")
    vals = [rec[y] for y in WINDOW_YEARS if y in rec]
    expected_geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
    geo, lo, hi = _ices_recruitment_geomean("sprat")
    assert geo == pytest.approx(expected_geo, rel=1e-9)
    assert lo == pytest.approx(min(vals)) and hi == pytest.approx(max(vals))


def test_herring_geomean_sums_four_stocks():
    from validate_baltic_vs_ices_sag import (
        WINDOW_YEARS,
        _load_assessment,
        _load_manifest,
        _series_by_year,
    )

    stocks = _load_manifest()["model_species_to_ices_stocks"]["herring"]
    series = [_series_by_year(_load_assessment(s), "recruitment") for s in stocks]
    per_year = [sum(s[y] for s in series) for y in WINDOW_YEARS if all(y in s for s in series)]
    expected_geo = math.exp(sum(math.log(v) for v in per_year) / len(per_year))
    geo, _, _ = _ices_recruitment_geomean("herring")
    assert geo == pytest.approx(expected_geo, rel=1e-9)
    assert geo > 100_000  # central stock included -> not the western-only undercount


def test_no_clean_r_species_return_none():
    assert _ices_recruitment_geomean("cod") is None
    assert _ices_recruitment_geomean("flounder") is None


def test_verdict_thresholds_inclusive():
    assert _recruitment_verdict(1.0, 1.0) == (1.0, "OK")
    assert _recruitment_verdict(1.0, 3.0)[1] == "OK"  # ratio 1/3 -> OK (inclusive)
    assert _recruitment_verdict(3.0, 1.0)[1] == "OK"  # ratio 3 -> OK (inclusive)
    assert _recruitment_verdict(0.33, 1.0)[1] == "FLAG"  # just below 1/3
    assert _recruitment_verdict(5.0, 1.0)[1] == "FLAG"


def test_format_recruitment_section_is_pure():
    from evaluate_calibration_vs_ices import (
        _format_recruitment_section,
    )  # scripts/ on path (top of file)

    rows = [
        {
            "species": "sprat",
            "age": "1",
            "model_R": 6.0e7,
            "ices_geomean": 7.0e7,
            "ices_min": 2.4e7,
            "ices_max": 1.1e8,
            "ratio": 0.86,
            "verdict": "OK",
            "reason": None,
        },
        {
            "species": "herring",
            "age": "0",
            "model_R": 2.0e7,
            "ices_geomean": 4.5e7,
            "ices_min": 2.7e7,
            "ices_max": 7.3e7,
            "ratio": 0.44,
            "verdict": "OK",
            "reason": None,
        },
        {
            "species": "cod",
            "age": None,
            "model_R": None,
            "ices_geomean": None,
            "ices_min": None,
            "ices_max": None,
            "ratio": None,
            "verdict": None,
            "reason": "no clean ICES R (eastern index + age mismatch 0 vs 1)",
        },
        {
            "species": "flounder",
            "age": None,
            "model_R": None,
            "ices_geomean": None,
            "ices_min": None,
            "ices_max": None,
            "ratio": None,
            "verdict": None,
            "reason": "no clean ICES R (none reported)",
        },
    ]
    out = _format_recruitment_section(rows)
    assert "Recruitment" in out
    assert "sprat" in out and "0.86" in out
    assert "no clean ICES R" in out
    assert "age-0" in out.lower()  # the herring caveat text


def test_evaluate_adds_recruitment_rows(monkeypatch):
    import evaluate_calibration_vs_ices as ev  # scripts/ on path (top of file)

    # Stub the sim so no engine runs; return biomass + recruitment stats.
    def _fake_run(base_config, overrides, n_years, seed, recruitment_ages=None):
        assert base_config.get("output.abundance.byage.enabled") == "true"
        assert recruitment_ages == {"sprat": "1", "herring": "0"}
        stats = {f"{sp}_mean": 1000.0 for sp in ev.SPECIES_NAMES}
        stats["sprat_recruitment_mean"] = 6.0e7
        stats["herring_recruitment_mean"] = 2.0e7
        return stats

    monkeypatch.setattr(ev, "run_simulation", _fake_run)
    # Minimal params file
    import json
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkstemp(suffix=".json")[1])
    p.write_text(json.dumps({"parameters": {}}))
    result = ev.evaluate(p, mode="bh", n_years=1, seed=0)
    rec = {r["species"]: r for r in result["recruitment"]}
    assert set(rec) == {"cod", "herring", "sprat", "flounder"}
    assert rec["sprat"]["verdict"] in ("OK", "FLAG") and rec["sprat"]["ices_geomean"] is not None
    assert rec["cod"]["ices_geomean"] is None and "no clean ICES R" in rec["cod"]["reason"]
