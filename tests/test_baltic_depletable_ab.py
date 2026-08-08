"""A/B harness for depletable LTL (spec 2026-08-08 Phase 1): keys, arms, gate, report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baltic_depletable_ab as ab  # noqa: E402

FITTED_ZOO = "0.911553421016705"


def _row(persists=True, in_env=True, mean=1000.0):
    return {
        "persists": persists,
        "in_envelope": in_env,
        "min_biomass": mean / 10,
        "late_mean_range": [mean * 0.95, mean * 1.05],
    }


def _table(**overrides):
    species = [
        "cod_west",
        "cod_east",
        "herring",
        "sprat",
        "flounder",
        "perch",
        "pikeperch",
        "smelt",
        "stickleback",
    ]
    t = {sp: _row() for sp in species}
    t.update(overrides)
    return t


def test_depletion_keys_exact():
    assert ab.DEPLETION_KEYS == {
        "ltl.depletable.enabled": "true",
        "ltl.depletable.floor": "0.05",
        "species.regrowth.rate.sp9": "5.0",
        "species.regrowth.rate.sp10": "5.0",
        "species.regrowth.rate.sp11": FITTED_ZOO,
        "species.regrowth.rate.sp12": FITTED_ZOO,
        "species.regrowth.rate.sp13": FITTED_ZOO,
        "species.regrowth.rate.sp14": FITTED_ZOO,
    }


def test_arm_off_is_explicit():
    # 'off' must stay off even after Task 4 flips the repo default — an empty override
    # would silently measure on-vs-on after adoption (review finding).
    assert ab.ARM_OFF == {"ltl.depletable.enabled": "false"}


def test_benthoslit_arm_rate():
    assert ab.BENTHOS_LIT_RATE == "0.03"


def test_required_pass_is_identity_pinned():
    assert ab.REQUIRED_PASS == (
        "cod_west",
        "cod_east",
        "herring",
        "sprat",
        "flounder",
        "perch",
        "stickleback",
    )


def test_identity_gate_passes_clean_table():
    ok, failures = ab.identity_gate(_table())
    assert ok and failures == []


def test_identity_gate_fails_on_required_species():
    ok, failures = ab.identity_gate(_table(perch=_row(in_env=False)))
    assert not ok and failures == ["perch"]


def test_identity_gate_ignores_indicative_failures():
    ok, failures = ab.identity_gate(
        _table(pikeperch=_row(in_env=False), smelt=_row(persists=False))
    )
    assert ok and failures == []


def test_make_report_two_arms_gate_and_delta():
    tables = {"off": _table(), "on": _table(herring=_row(mean=800.0))}
    rep = ab.make_report(tables, years=50, seeds=[42, 123])
    assert "herring" in rep and "GATE" in rep and "off" in rep and "on" in rep
    assert "-20.0%" in rep  # 800 vs 1000 midpoint delta


def test_make_report_three_arms_keeps_primary_delta():
    # With >2 arms every non-baseline arm gets its own delta-vs-off column (review finding:
    # a single last-vs-first column would drop the primary on-vs-off delta).
    tables = {
        "off": _table(),
        "on": _table(herring=_row(mean=800.0)),
        "on-benthoslit": _table(herring=_row(mean=900.0)),
    }
    rep = ab.make_report(tables, years=50, seeds=[42])
    assert "-20.0%" in rep and "-10.0%" in rep
