from __future__ import annotations

from pathlib import Path

from osmose.calibration.checkpoint import MAX_CHECKPOINT_BYTES, default_results_dir


def test_default_results_dir_resolves_to_baltic_calibration_results():
    """default_results_dir points at the Baltic results dir, package-root-resolved."""
    p = default_results_dir()
    assert isinstance(p, Path)
    assert p.parts[-3:] == ("data", "baltic", "calibration_results")


def test_max_checkpoint_bytes_is_1mib():
    """1 MiB ceiling for read_checkpoint's size guard."""
    assert MAX_CHECKPOINT_BYTES == 1_048_576
