from __future__ import annotations

import os
import time

import pytest


def test_scan_results_dir_returns_no_run_when_empty(tmp_results_dir):
    from ui.pages.calibration_handlers import _scan_results_dir

    snap = _scan_results_dir()
    assert snap.active.kind == "no_run"
    assert snap.other_live_paths == ()


def test_scan_results_dir_picks_newest_live(tmp_results_dir):
    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint
    from tests.test_calibration_checkpoint import _valid_checkpoint_kwargs
    from ui.pages.calibration_handlers import _scan_results_dir

    kwargs_old = _valid_checkpoint_kwargs()
    kwargs_old["phase"] = "old"
    kwargs_new = _valid_checkpoint_kwargs()
    kwargs_new["phase"] = "new"

    p_old = tmp_results_dir / "phase_old_checkpoint.json"
    p_new = tmp_results_dir / "phase_new_checkpoint.json"
    write_checkpoint(p_old, CalibrationCheckpoint(**kwargs_old))
    old_mtime = time.time() - 10
    os.utime(p_old, (old_mtime, old_mtime))
    write_checkpoint(p_new, CalibrationCheckpoint(**kwargs_new))

    snap = _scan_results_dir()
    assert snap.active.kind == "ok"
    assert snap.active.checkpoint.phase == "new"
    assert len(snap.other_live_paths) == 1


def test_scan_skips_symlinks(tmp_results_dir):
    """Security: symlink in RESULTS_DIR is skipped, not followed.
    Symlink target is a VALID checkpoint outside the results dir; if followed,
    kind would be 'ok' with phase='should_not_be_visible'."""
    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint
    from tests.test_calibration_checkpoint import _valid_checkpoint_kwargs
    from ui.pages.calibration_handlers import _scan_results_dir

    kwargs = _valid_checkpoint_kwargs()
    kwargs["phase"] = "should_not_be_visible"
    outside = tmp_results_dir.parent / "outside_checkpoint.json"
    write_checkpoint(outside, CalibrationCheckpoint(**kwargs))

    link = tmp_results_dir / "phase_evil_checkpoint.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported on this filesystem")

    snap = _scan_results_dir()
    assert snap.active.kind == "no_run"
    assert snap.active.checkpoint is None


def test_scan_signature_invalidates_on_persistent_oserror(monkeypatch):
    """Persistent failures produce a strictly-increasing tick so poll keeps invalidating."""
    from ui.pages import calibration_handlers as ch_mod
    from ui.pages.calibration_handlers import _scan_signature

    monkeypatch.setattr(ch_mod, "_signature_tick", 0)
    monkeypatch.setattr(ch_mod, "_seen_scan_errors", set())

    class _RaisingPath:
        def glob(self, pattern):
            raise OSError("ENOENT: simulated mount failure")

    monkeypatch.setattr(ch_mod, "RESULTS_DIR", _RaisingPath())
    sig1 = _scan_signature()
    sig2 = _scan_signature()
    sig3 = _scan_signature()
    assert sig1[2] < sig2[2] < sig3[2]
