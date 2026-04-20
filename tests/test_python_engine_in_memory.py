"""Tests for PythonEngine.run_in_memory().

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults

EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _short_config() -> dict[str, str]:
    raw = OsmoseConfigReader().read(EXAMPLE_CONFIG)
    raw["simulation.time.nyear"] = "1"
    return raw


def test_seed_determinism():
    """Same seed twice produces equal biomass."""
    cfg = _short_config()
    r1 = PythonEngine().run_in_memory(cfg, seed=42)
    r2 = PythonEngine().run_in_memory(cfg, seed=42)
    pd.testing.assert_frame_equal(
        r1.biomass().reset_index(drop=True),
        r2.biomass().reset_index(drop=True),
    )


def test_disk_vs_memory_same_biomass():
    """run() + OsmoseResults(dir) and run_in_memory() produce equal biomass
    within rtol=1e-12 (same RNG stream, same engine)."""
    cfg = _short_config()
    with tempfile.TemporaryDirectory() as d:
        PythonEngine().run(cfg, Path(d), seed=42)
        disk = OsmoseResults(Path(d)).biomass()
    memory = PythonEngine().run_in_memory(cfg, seed=42).biomass()
    disk_sorted = disk.reset_index(drop=True)
    memory_sorted = memory.reset_index(drop=True)
    pd.testing.assert_frame_equal(disk_sorted, memory_sorted, rtol=1e-12)


def test_missing_grid_file_raises_FileNotFoundError():
    """Same error contract as run() — config with a non-existent grid
    file raises FileNotFoundError from _resolve_grid."""
    cfg = _short_config()
    cfg["grid.netcdf.file"] = "no_such_file_exists.nc"
    with pytest.raises(FileNotFoundError, match="not found in search paths"):
        PythonEngine().run_in_memory(cfg, seed=0)


def test_no_disk_writes(tmp_path, monkeypatch):
    """run_in_memory must not leak CSV/properties/restart artefacts into cwd.

    Numba / Python __pycache__ lives with the package install, not in cwd,
    so it doesn't count here. We're checking that the engine's own output
    pipeline doesn't write to the current directory when output_dir is None.
    """
    cfg = _short_config()
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    PythonEngine().run_in_memory(cfg, seed=42)
    after = set(tmp_path.iterdir())
    new_entries = after - before
    suspicious = [
        p for p in new_entries if not p.name.startswith(".") and "__pycache__" not in p.name
    ]
    assert not suspicious, (
        f"run_in_memory leaked files into cwd: {[str(p) for p in suspicious]}"
    )
