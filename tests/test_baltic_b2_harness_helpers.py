"""B2 harness helper tests (spec 2026-08-29, Task 3): pure functions only --
`expected_knob_factor`, `arm_overlays`, `hill_ordering_ok` -- on synthetic fixtures
(task-3-brief.md, verbatim). `run_b2` itself (the 6-arm x 5-seed x 50yr run) is NOT invoked
here -- see the module docstring of scripts/baltic_b2_scenario_ab.py."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "baltic_b2_scenario_ab",
    Path(__file__).resolve().parent.parent / "scripts" / "baltic_b2_scenario_ab.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# Task 2's synthetic O2 field (tests/test_build_baltic_b2_forcing.py::_field), reused
# verbatim so the Hill-ordering check is exercised against the same fixture the offset/
# predicted-dK functions were proven against.
def _field():
    o2 = np.full((24, 4, 4), 200.0)
    o2[:, 0, 0] = 30.0  # hypoxic wet cell
    o2[:, 3, 3] = np.nan  # land
    wet = np.ones((4, 4), dtype=bool)
    wet[3, 3] = False
    return o2, wet


# ---------------------------------------------------------------------------
# expected_knob_factor (spec §4d)
# ---------------------------------------------------------------------------


def test_expected_knob_factor_non_dyadic_3ulp_trap():
    # The review-pinned non-dyadic case: beta=-0.51, tref full precision, dT=2.9.
    # (tref + dT) - tref != dT exactly in float64 -- so the loader's actual float path
    # differs from the naive exp(beta*dT) by a few ULP. Pin both facts.
    beta, tref, dT = -0.51, 9.670314810741907, 2.9
    got = m.expected_knob_factor(beta, tref, dT)
    naive = float(np.exp(beta * dT))
    assert got != naive, "expected_knob_factor must NOT collapse to the naive exp(beta*dT)"

    want = float(np.exp(beta * (float(str(tref + dT)) - tref)))
    assert got == want


def test_expected_knob_factor_dyadic_zero_is_exactly_one():
    beta, tref, dT = -0.51, 9.670314810741907, 0.0
    got = m.expected_knob_factor(beta, tref, dT)
    assert got == 1.0


def test_expected_knob_factor_dyadic_matches_naive():
    # C1's dyadic dT values (2.0, 4.0) round-trip losslessly through str/float, so here
    # (unlike the 2.9 case) the loader path and the naive exp(beta*dT) do agree exactly --
    # recorded so the 3-ULP trap isn't mistaken for a general property of the function.
    beta, tref, dT = -0.51, 9.670314810741907, 2.0
    got = m.expected_knob_factor(beta, tref, dT)
    naive = float(np.exp(beta * dT))
    assert got == naive


# ---------------------------------------------------------------------------
# arm_overlays (spec §Design 3)
# ---------------------------------------------------------------------------


def test_arm_overlays_baseline_has_no_knob_or_oxygen_keys():
    artifacts = {"series_csv": Path("/tmp/unused.csv"), "o2_nc": None, "predicted_dK": 0.0}
    overlays = m.arm_overlays("baseline", artifacts, m.TREFS, m.BETAS)
    assert overlays == {}


def test_arm_overlays_scenario_arm_carries_c1_knob_keys_full_precision_tref():
    artifacts = {
        "series_csv": Path("/tmp/rcp45_bsap/knob_series.csv"),
        "o2_nc": Path("/tmp/rcp45_bsap/oxygen_offset.nc"),
        "predicted_dK": 0.021,
    }
    overlays = m.arm_overlays("rcp45_bsap", artifacts, m.TREFS, m.BETAS)

    assert overlays["reproduction.thermal.gate.enabled"] == "true"
    assert overlays["reproduction.thermal.gate.response"] == "exponential"
    assert overlays["reproduction.thermal.gate.series.file"] == str(artifacts["series_csv"])
    for sp in m.ENABLED:
        assert overlays[f"reproduction.thermal.gate.species.enabled.sp{sp}"] == "true"
        assert overlays[f"reproduction.thermal.gate.beta.sp{sp}"] == str(m.BETAS[sp])
        tref_str = overlays[f"reproduction.thermal.gate.tref.sp{sp}"]
        assert tref_str == str(m.TREFS[sp])
        # full precision, not a rounded display value
        assert float(tref_str) == m.TREFS[sp]
        assert len(tref_str.split(".")[-1]) > 2

    assert overlays["oxygen.filename"] == str(artifacts["o2_nc"])


def test_arm_overlays_zero_arm_carries_knob_keys_but_oxygen_only_if_artifact_present():
    # zero has both an O2 artifact (the sourced-zero copy) and thermal keys.
    artifacts_with_o2 = {
        "series_csv": Path("/tmp/zero/knob_series.csv"),
        "o2_nc": Path("/tmp/zero/oxygen_offset.nc"),
        "predicted_dK": 0.0,
    }
    overlays = m.arm_overlays("zero", artifacts_with_o2, m.TREFS, m.BETAS)
    assert overlays["reproduction.thermal.gate.enabled"] == "true"
    assert "oxygen.filename" in overlays

    # oxygen.filename must be ABSENT (not merely None) when the arm has no O2 artifact --
    # a present-but-None value would still shadow the base config's key on dict-merge.
    artifacts_no_o2 = {
        "series_csv": Path("/tmp/zero/knob_series.csv"),
        "o2_nc": None,
        "predicted_dK": 0.0,
    }
    overlays2 = m.arm_overlays("zero", artifacts_no_o2, m.TREFS, m.BETAS)
    assert "oxygen.filename" not in overlays2


# ---------------------------------------------------------------------------
# hill_ordering_ok (spec §4c), on Task 2's synthetic field
# ---------------------------------------------------------------------------


def test_hill_ordering_ok_positive_delta_everywhere_ge():
    o2, wet = _field()
    # import the builder's offset_o2 to build a realistic arm field (same idiom the builder
    # tests use to derive an "arm" field from the base field).
    b2 = m._b2
    arm_o2 = b2.offset_o2(o2, wet, 26.8)
    assert m.hill_ordering_ok(arm_o2, o2, wet, delta_sign=1) is True


def test_hill_ordering_ok_negative_delta_everywhere_le():
    o2, wet = _field()
    b2 = m._b2
    arm_o2 = b2.offset_o2(o2, wet, -8.9)
    assert m.hill_ordering_ok(arm_o2, o2, wet, delta_sign=-1) is True


def test_hill_ordering_ok_zero_delta_everywhere_equal():
    o2, wet = _field()
    b2 = m._b2
    arm_o2 = b2.offset_o2(o2, wet, 0.0)
    assert m.hill_ordering_ok(arm_o2, o2, wet, delta_sign=0) is True


def test_hill_ordering_ok_detects_violation():
    # Sign says "positive delta" but the arm field is actually lower everywhere wet --
    # a real wiring bug (e.g. the wrong file loaded) must be caught, not rubber-stamped.
    o2, wet = _field()
    b2 = m._b2
    arm_o2 = b2.offset_o2(o2, wet, -8.9)
    assert m.hill_ordering_ok(arm_o2, o2, wet, delta_sign=1) is False


def test_hill_ordering_ok_ignores_land_cells():
    # Land cells are NaN in both fields; f_o2_hill(NaN) must never enter the comparison
    # (the wet mask must exclude them, not merely tolerate them).
    o2, wet = _field()
    b2 = m._b2
    arm_o2 = b2.offset_o2(o2, wet, 26.8)
    assert np.isnan(arm_o2[0, 3, 3])
    assert np.isnan(o2[0, 3, 3])
    assert m.hill_ordering_ok(arm_o2, o2, wet, delta_sign=1) is True
