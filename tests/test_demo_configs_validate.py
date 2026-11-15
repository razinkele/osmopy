"""Every demo the UI offers must pass the validation the UI runs on load and before a run.

The Run tab blocks on any validation error (``ui/pages/run.py``: "VALIDATION ERRORS (run
blocked)"), so a schema bound that contradicts a shipped config stops the demo from running at
all. Found 2026-11-15: ``reproduction.rv.gate.ref`` had ``max_val=1.0`` while the Baltic demos run
150 (the RV series is in km^3, 48-380), so all three Baltic demos were blocked from running in the
UI since that value went in (2026-07-28).

The check is the UI's own: ``summarize_config_validation`` with the demo's config directory, which
adds file-reference checks to the schema checks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.config.validator import summarize_config_validation
from osmose.demo import list_demos, osmose_demo
from osmose.schema import build_registry

# Known, separate gap: the validator does not parse comma-separated per-stage values ("15, 5"),
# which benguela's predation size ratios use. Only errors of exactly that shape are excused, so
# any OTHER benguela error still fails the test.
_COMMA_MULTI_VALUE = re.compile(r"expected number, got '[^']*,[^']*'")
_KNOWN_GAPS = {"benguela": _COMMA_MULTI_VALUE}


def _ui_validation(demo: str, tmp_path: Path) -> list[str]:
    info = osmose_demo(demo, tmp_path)
    master = Path(info["config_file"])
    cfg = OsmoseConfigReader().read(master)
    errors, _warnings = summarize_config_validation(cfg, build_registry(), master.parent)
    return errors


@pytest.mark.parametrize("demo", list_demos())
def test_demo_passes_the_ui_validation(demo, tmp_path):
    errors = _ui_validation(demo, tmp_path)
    known = _KNOWN_GAPS.get(demo)
    unexpected = [e for e in errors if not (known and known.search(e))]
    assert unexpected == [], f"{demo}: {unexpected}"


@pytest.mark.xfail(
    strict=True, reason="validator does not parse comma-separated multi-value entries"
)
def test_benguela_passes_unfiltered(tmp_path):
    """Goes XPASS (and fails, being strict) once the validator handles comma multi-values: then
    drop this test and benguela's entry in _KNOWN_GAPS."""
    assert _ui_validation("benguela", tmp_path) == []
