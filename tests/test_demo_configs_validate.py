"""Every demo the UI offers must load without schema validation errors.

A schema bound that contradicts a shipped config shows up as an error on load, and the form's
numeric input carries the same bound, so a nudge can silently rewrite the value. Found
2026-11-15: ``reproduction.rv.gate.ref`` had ``max_val=1.0`` while the Baltic demos run 150 (the
RV series is in km^3, 48-380), so every Baltic load reported an error and the gate's reference
was one scroll away from 1.0, which disables the gate cod_east depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.config.validator import validate_config
from osmose.demo import list_demos, osmose_demo
from osmose.schema import build_registry

# Known, separate gap: the validator does not parse comma-separated per-stage values
# ("15, 5"), which benguela's predation size ratios use. Strict, so fixing it fails this marker.
_KNOWN = {
    "benguela": pytest.mark.xfail(
        strict=True, reason="validator does not parse comma-separated multi-value entries"
    ),
}


@pytest.mark.parametrize(
    "demo", [pytest.param(name, marks=_KNOWN.get(name, ())) for name in list_demos()]
)
def test_demo_loads_without_validation_errors(demo, tmp_path):
    info = osmose_demo(demo, tmp_path)
    cfg = OsmoseConfigReader().read(Path(info["config_file"]))
    errors, _warnings = validate_config(cfg, build_registry())
    assert errors == [], f"{demo}: {errors}"
