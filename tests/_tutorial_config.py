"""Canonical source of truth for the 3-species tutorial.

The tutorial markdown at docs/tutorials/30-minute-ecosystem.md transcribes
the contents of `build_config`, `ACCESSIBILITY_CSV`, and `build_ltl` into
its main code block. The regression test at tests/test_tutorial_3species.py
imports them directly.

If anything here changes, update docs/tutorials/30-minute-ecosystem.md to
match. Drift is caught at PR review time + by the markdown-parses test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: F401  (used by build_ltl in Task 4)
import xarray as xr  # noqa: F401  (used by build_ltl in Task 4)

# The exact substring the tutorial tells the reader to find in accessibility.csv,
# and the exact substring they replace it with. Used by the perturbation test
# to confirm the instruction is robust to CSV-format edits.
BASELINE_PERTURBATION: tuple[str, str] = ("Forager;0.8;0;0", "Forager;0.1;0;0")

# Canonical accessibility CSV. ;-separated; rows = prey labels, cols = predator
# labels (focal species only). Loaded by osmose/engine/accessibility.py:72-131.
ACCESSIBILITY_CSV: str = """;Predator;Forager;PlanktonEater
Predator;0;0;0
Forager;0.8;0;0
PlanktonEater;0;0.8;0
Plankton;0;0.2;0.8
"""


def build_ltl(work_dir: Path) -> Path:
    """Write the constant-Plankton LTL forcing NetCDF to work_dir.

    Returns the path to the written file. Filled in by Task 4.
    """
    raise NotImplementedError("Filled in by Task 4")


def build_config(work_dir: Path) -> dict:
    """Return the engine config dict with paths resolved against work_dir.

    `species.file.sp3` and `predation.accessibility.file` get resolved to
    `work_dir / "ltl.nc"` and `work_dir / "accessibility.csv"` respectively
    via `.as_posix()`. Filled in by Task 4.
    """
    raise NotImplementedError("Filled in by Task 4")
