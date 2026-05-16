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


def test_format_elapsed_seconds():
    from ui.pages.calibration_handlers import _format_elapsed
    assert _format_elapsed(0) == "0s"
    assert _format_elapsed(45) == "45s"
    assert _format_elapsed(59.4) == "59s"


def test_format_elapsed_minutes():
    from ui.pages.calibration_handlers import _format_elapsed
    assert _format_elapsed(60) == "1m 0s"
    assert _format_elapsed(125) == "2m 5s"
    assert _format_elapsed(3599) == "59m 59s"


def test_format_elapsed_hours():
    from ui.pages.calibration_handlers import _format_elapsed
    assert _format_elapsed(3600) == "1h 0m"
    assert _format_elapsed(4980) == "1h 23m"
    assert _format_elapsed(86400) == "24h 0m"


def test_ckpt_mtime_for_returns_timestamp_iso_when_ok(tmp_results_dir):
    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint
    from tests.test_calibration_checkpoint import _valid_checkpoint_kwargs
    from ui.pages.calibration_handlers import _ckpt_mtime_for, _scan_results_dir

    kwargs = _valid_checkpoint_kwargs()
    kwargs["timestamp_iso"] = "2026-05-12T10:00:00+00:00"
    write_checkpoint(tmp_results_dir / "phase_x_checkpoint.json",
                     CalibrationCheckpoint(**kwargs))
    snap = _scan_results_dir()
    from datetime import datetime, timezone
    expected = datetime(2026, 5, 12, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    assert _ckpt_mtime_for(snap) == expected


def test_ckpt_mtime_for_falls_back_when_no_active_checkpoint():
    from osmose.calibration.checkpoint import CheckpointReadResult, LiveSnapshot
    from ui.pages.calibration_handlers import _ckpt_mtime_for
    import time
    snap = LiveSnapshot(
        active=CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None),
        other_live_paths=(),
        snapshot_monotonic=0.0,
    )
    before = time.time()
    result = _ckpt_mtime_for(snap)
    after = time.time()
    assert before <= result <= after


def test_other_live_runs_returns_all_but_newest(tmp_results_dir):
    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint
    from tests.test_calibration_checkpoint import _valid_checkpoint_kwargs
    from ui.pages.calibration_handlers import _scan_results_dir

    for phase in ["a", "b", "c"]:
        kwargs = _valid_checkpoint_kwargs()
        kwargs["phase"] = phase
        write_checkpoint(
            tmp_results_dir / f"phase_{phase}_checkpoint.json",
            CalibrationCheckpoint(**kwargs),
        )

    snap = _scan_results_dir()
    assert len(snap.other_live_paths) == 2
