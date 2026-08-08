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


def test_pin_noop_when_absent_or_false():
    for base in ({}, {"ltl.depletable.enabled": "false"}):
        cfg, pinned = cert.pin_java_incompatible(dict(base))
        assert pinned == []
        assert cfg.get("ltl.depletable.enabled", "false") == "false"


def test_prepare_java_cfg_passes_runner_guard():
    # The seam loads the real demo config, applies params, then pins — exactly what
    # certify_java stages. Must pass the runner guard even with depletion forced on.
    cfg, pinned = cert._prepare_java_cfg({"ltl.depletable.enabled": "true"})
    assert "ltl.depletable.enabled" in [k for k in pinned]
    assert java_engine_block_reason(cfg, jar_version="4.4.1") is None
