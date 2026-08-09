"""--java certification arms pin Python-only depletion off (spec 2026-08-08 §4 Phase 1 parity).

Tests go through _prepare_java_cfg, the seam certify_java actually uses, so the wiring is
exercised — not just the standalone helper (review finding). No jar required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baltic_stability_certify as cert  # noqa: E402
from osmose.runner import java_engine_block_reason  # noqa: E402


def test_pin_flips_enabled_true():
    cfg, pinned = cert.pin_java_incompatible({"ltl.depletable.enabled": "true", "x": "1"})
    assert cfg["ltl.depletable.enabled"] == "false"
    assert pinned == ["ltl.depletable.enabled"]
    assert cfg["x"] == "1"


def test_pin_flips_oxygen_benthos_enabled_true():
    # Phase 2a adoption: the O2->benthos K coupling is Python-only (Java reads oxygen.* only as
    # bioenergetics forcing, with no benthos-K coupling to apply it to) — same pinning pattern
    # as depletable plankton.
    cfg, pinned = cert.pin_java_incompatible({"ltl.oxygen.benthos.enabled": "true", "x": "1"})
    assert cfg["ltl.oxygen.benthos.enabled"] == "false"
    assert pinned == ["ltl.oxygen.benthos.enabled"]
    assert cfg["x"] == "1"
    # The oxygen.* forcing keys themselves are NOT pinned — Java legitimately reads them.
    assert "oxygen.filename" not in cert.JAVA_INCOMPATIBLE_PINS
    assert "oxygen.varname" not in cert.JAVA_INCOMPATIBLE_PINS


def test_pin_noop_when_absent_or_false():
    for base in (
        {},
        {"ltl.depletable.enabled": "false"},
        {"ltl.oxygen.benthos.enabled": "false"},
    ):
        cfg, pinned = cert.pin_java_incompatible(dict(base))
        assert pinned == []
        assert cfg.get("ltl.depletable.enabled", "false") == "false"
        assert cfg.get("ltl.oxygen.benthos.enabled", "false") == "false"


def test_prepare_java_cfg_passes_runner_guard():
    # The seam loads the real demo config, applies params, then pins — exactly what
    # certify_java stages. Must pass the runner guard even with depletion forced on.
    #
    # The real Baltic demo config now also bakes in ltl.oxygen.benthos.enabled=true (Phase 2a
    # adoption), so `pinned` must carry BOTH Python-only flags, not just the one this call's
    # `params` override explicitly sets.
    cfg, pinned = cert._prepare_java_cfg({"ltl.depletable.enabled": "true"})
    assert set(pinned) == {"ltl.depletable.enabled", "ltl.oxygen.benthos.enabled"}
    # jar_version is required: the Baltic config declares 2 background species, which the
    # guard conservatively blocks when the jar version is unknown — unrelated to depletion.
    assert java_engine_block_reason(cfg, jar_version="4.4.1") is None
