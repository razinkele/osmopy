"""Tests for osmose.cleanup — temp directory management."""

import os
import tempfile


from osmose.cleanup import cleanup_old_temp_dirs, register_cleanup, _OSMOSE_PREFIXES


def test_cleanup_removes_old_dirs(tmp_path, monkeypatch):
    """Old osmose temp dirs (beyond threshold) are removed; others are kept."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    # Create an old osmose dir (mtime = epoch → extremely old)
    old_dir = tmp_path / "osmose_run_old"
    old_dir.mkdir()
    os.utime(old_dir, (0, 0))

    # Create a fresh osmose dir (mtime = now)
    new_dir = tmp_path / "osmose_run_new"
    new_dir.mkdir()

    # Create a non-osmose dir that must never be touched
    non_osmose = tmp_path / "other_dir"
    non_osmose.mkdir()

    removed = cleanup_old_temp_dirs(max_age_hours=1)

    assert removed == 1
    assert not old_dir.exists(), "Old osmose dir should have been removed"
    assert new_dir.exists(), "New osmose dir should be kept"
    assert non_osmose.exists(), "Non-osmose dir must not be removed"


def test_cleanup_zero_max_age_removes_all_osmose_dirs(tmp_path, monkeypatch):
    """max_age_hours=0 removes all osmose dirs regardless of age."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    dirs = []
    for prefix in _OSMOSE_PREFIXES:
        d = tmp_path / f"{prefix}test"
        d.mkdir()
        dirs.append(d)

    # Non-osmose dir should survive
    other = tmp_path / "some_other_dir"
    other.mkdir()

    removed = cleanup_old_temp_dirs(max_age_hours=0)

    assert removed == len(_OSMOSE_PREFIXES)
    for d in dirs:
        assert not d.exists(), f"{d.name} should have been removed"
    assert other.exists()


def test_cleanup_ignores_files(tmp_path, monkeypatch):
    """Files (not directories) with osmose prefixes are ignored."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    osmose_file = tmp_path / "osmose_run_file.txt"
    osmose_file.write_text("not a dir")
    os.utime(osmose_file, (0, 0))

    removed = cleanup_old_temp_dirs(max_age_hours=0)

    assert removed == 0
    assert osmose_file.exists()


def test_cleanup_returns_zero_when_nothing_to_clean(tmp_path, monkeypatch):
    """Returns 0 when there are no osmose dirs to remove."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    (tmp_path / "unrelated").mkdir()

    removed = cleanup_old_temp_dirs(max_age_hours=24)
    assert removed == 0


def test_cleanup_all_prefixes_recognised(tmp_path, monkeypatch):
    """All documented osmose prefixes are recognised and cleaned."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    expected_prefixes = {
        "osmose_run_",
        "osmose_cal_",
        "osmose_sens_",
        "osmose_export_",
        "osmose_demo_",
    }
    assert set(_OSMOSE_PREFIXES) == expected_prefixes

    for prefix in expected_prefixes:
        d = tmp_path / f"{prefix}abc"
        d.mkdir()
        os.utime(d, (0, 0))

    removed = cleanup_old_temp_dirs(max_age_hours=1)
    assert removed == len(expected_prefixes)


def test_register_cleanup_registers_atexit(monkeypatch):
    """register_cleanup() calls atexit.register without raising."""
    registered = []

    import atexit

    monkeypatch.setattr(atexit, "register", lambda fn, **kw: registered.append((fn, kw)))

    register_cleanup()

    assert len(registered) == 1
    fn, kw = registered[0]
    assert fn is cleanup_old_temp_dirs
    assert kw.get("max_age_hours") == 0
