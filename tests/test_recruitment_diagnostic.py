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
