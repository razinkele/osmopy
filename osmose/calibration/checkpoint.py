"""Calibration checkpoint module — atomic on-disk progress snapshots.

Read by the Shiny dashboard at 1 Hz; written by every optimizer
(DE / CMA-ES / surrogate-DE / NSGA-II) every N generations.

See docs/superpowers/specs/2026-05-12-calibration-dashboard-design.md for the
full contract and 14 invariants enforced in CalibrationCheckpoint.__post_init__.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

MAX_CHECKPOINT_BYTES: Final[int] = 1_048_576  # 1 MiB; real checkpoints are ~10 KB

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def default_results_dir() -> Path:
    """Baltic default — package-root-resolved data/baltic/calibration_results/.

    Callers may pass a different directory to write_checkpoint / read_checkpoint
    to support non-Baltic configurations.
    """
    return _PACKAGE_ROOT / "data" / "baltic" / "calibration_results"


# Single source-of-truth for the Baltic results directory. Both
# scripts/calibrate_baltic.py and ui/pages/calibration_handlers.py import this
# instead of redeclaring their own copy — keeps tmp_results_dir's monkeypatch
# to ONE target. See Task 8 fixture notes.
RESULTS_DIR: Final[Path] = default_results_dir()
