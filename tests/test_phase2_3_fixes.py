"""Regression tests for Phase 2-3 fixes.

Covers:
- CORR-1: sync_n_maps_from_config old-style key regex
- SEC-1:  runner.py override key validation
- SEC-3:  calibration_handlers._clamp_int bounds
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# CORR-1: sync_n_maps_from_config uses old-style keys
# ---------------------------------------------------------------------------


def test_sync_n_maps_counts_old_style_keys():
    """Old-style movement.file.mapN keys must be counted."""
    import re

    pattern = re.compile(r"movement\.file\.map\d+$")
    cfg = {
        "movement.file.map1": "/path/to/map1.csv",
        "movement.file.map2": "/path/to/map2.csv",
        "movement.file.map3": "",          # empty — should NOT count
        "movement.file.map4": "NULL",      # null sentinel — should NOT count
        "movement.species.map1": "anchovy",
    }
    count = sum(
        1 for k in cfg
        if pattern.match(k)
        and isinstance(cfg[k], str)
        and cfg[k].strip()
        and cfg[k].strip().lower() not in ("null", "none")
    )
    assert count == 2  # map1, map2 only (empty and NULL are excluded)


def test_sync_n_maps_ignores_new_style_keys():
    """New-style movement.mapN.file keys must NOT be counted by the old-style regex."""
    import re

    pattern = re.compile(r"movement\.file\.map\d+$")
    cfg = {
        "movement.map1.file": "/path/to/map1.csv",
        "movement.map2.file": "/path/to/map2.csv",
    }
    count = sum(1 for k in cfg if pattern.match(k) and cfg[k])
    assert count == 0


def test_sync_n_maps_ignores_empty_values():
    """Keys with empty, None-like values must not be counted."""
    import re

    pattern = re.compile(r"movement\.file\.map\d+$")
    cfg = {
        "movement.file.map1": "",
        "movement.file.map2": None,
    }
    count = sum(1 for k in cfg if pattern.match(k) and cfg[k])
    assert count == 0


def test_sync_n_maps_regex_anchored():
    """Regex must not match keys with suffixes (e.g. movement.file.map1.extra)."""
    import re

    pattern = re.compile(r"movement\.file\.map\d+$")
    cfg = {
        "movement.file.map1.extra": "/path/to/map1.csv",
        "movement.file.mapXYZ": "/path/to/map2.csv",
    }
    count = sum(1 for k in cfg if pattern.match(k) and cfg[k])
    assert count == 0


# ---------------------------------------------------------------------------
# SEC-1: runner.py override key validation
# ---------------------------------------------------------------------------


def test_runner_build_cmd_rejects_invalid_key():
    """_build_cmd must raise ValueError for non-alphanumeric-dot-underscore keys."""
    from osmose.runner import OsmoseRunner

    runner = OsmoseRunner(jar_path=Path("fake.jar"))
    with pytest.raises(ValueError, match="Invalid override key"):
        runner._build_cmd(
            Path("config.csv"),
            overrides={"bad key!": "value"},
        )


def test_runner_build_cmd_rejects_uppercase_key():
    """Override keys must be lowercase (OSMOSE config convention)."""
    from osmose.runner import OsmoseRunner

    runner = OsmoseRunner(jar_path=Path("fake.jar"))
    with pytest.raises(ValueError, match="Invalid override key"):
        runner._build_cmd(
            Path("config.csv"),
            overrides={"Species.Name": "anchovy"},
        )


def test_runner_build_cmd_accepts_valid_keys():
    """Dotted lowercase alphanumeric keys must pass validation."""
    from osmose.runner import OsmoseRunner

    runner = OsmoseRunner(jar_path=Path("fake.jar"))
    cmd = runner._build_cmd(
        Path("config.csv"),
        overrides={
            "simulation.time.nyear": "10",
            "species.name.sp0": "anchovy",
            "output.dir.path": "/some/path",
        },
    )
    assert any("-Psimulation.time.nyear=10" in a for a in cmd)
    assert any("-Pspecies.name.sp0=anchovy" in a for a in cmd)


def test_runner_build_cmd_accepts_values_with_special_chars():
    """Override values (not keys) may contain special characters — list exec is safe."""
    from osmose.runner import OsmoseRunner

    runner = OsmoseRunner(jar_path=Path("fake.jar"))
    # file paths, scientific notation, comma-separated values — all valid values
    cmd = runner._build_cmd(
        Path("config.csv"),
        overrides={"output.dir.path": "/path/with spaces/and; semicolons"},
    )
    assert any("/path/with spaces" in a for a in cmd)


# ---------------------------------------------------------------------------
# SEC-3: calibration_handlers._clamp_int bounds validation
# ---------------------------------------------------------------------------


def test_clamp_int_accepts_within_bounds():
    from ui.pages.calibration_handlers import _clamp_int

    assert _clamp_int(1, 1, 32, "n_parallel") == 1
    assert _clamp_int(32, 1, 32, "n_parallel") == 32
    assert _clamp_int(16, 1, 32, "n_parallel") == 16


def test_clamp_int_rejects_below_min():
    from ui.pages.calibration_handlers import _clamp_int

    with pytest.raises(ValueError, match="n_parallel"):
        _clamp_int(0, 1, 32, "n_parallel")


def test_clamp_int_rejects_above_max():
    from ui.pages.calibration_handlers import _clamp_int

    with pytest.raises(ValueError, match="pop_size"):
        _clamp_int(501, 10, 500, "pop_size")


def test_clamp_int_pop_size_bounds():
    from ui.pages.calibration_handlers import _clamp_int

    assert _clamp_int(10, 10, 500, "pop_size") == 10
    assert _clamp_int(500, 10, 500, "pop_size") == 500
    with pytest.raises(ValueError):
        _clamp_int(9, 10, 500, "pop_size")


def test_clamp_int_generations_bounds():
    from ui.pages.calibration_handlers import _clamp_int

    assert _clamp_int(1000, 10, 1000, "generations") == 1000
    with pytest.raises(ValueError):
        _clamp_int(1001, 10, 1000, "generations")
