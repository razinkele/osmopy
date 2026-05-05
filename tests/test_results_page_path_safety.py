"""Path-safety tests for ui.pages.results._safe_output_dir (closes C3).

The previous `..`-substring check accepted absolute paths like `/etc`.
The new helper resolves the path and rejects anything outside `cwd`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ui.pages.results import _safe_output_dir


@pytest.fixture
def chdir_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make `tmp_path` the working directory for the test."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_rejects_absolute_path_outside_cwd(chdir_tmp: Path) -> None:
    assert _safe_output_dir("/etc") is None


def test_rejects_path_outside_cwd_via_traversal(chdir_tmp: Path) -> None:
    # ".." chains resolve outside cwd; the helper must reject.
    assert _safe_output_dir("../../etc/passwd") is None


def test_rejects_absolute_tmp_outside_cwd(tmp_path_factory: pytest.TempPathFactory,
                                          chdir_tmp: Path) -> None:
    other_root = tmp_path_factory.mktemp("not-cwd")
    assert _safe_output_dir(str(other_root)) is None


def test_accepts_directory_inside_cwd(chdir_tmp: Path) -> None:
    output = chdir_tmp / "output" / "run123"
    output.mkdir(parents=True)
    result = _safe_output_dir(str(output))
    assert result is not None
    assert result == output.resolve()


def test_rejects_empty_string(chdir_tmp: Path) -> None:
    assert _safe_output_dir("") is None


def test_rejects_nonexistent_directory(chdir_tmp: Path) -> None:
    assert _safe_output_dir("output/never-existed") is None


def test_rejects_file_path(chdir_tmp: Path) -> None:
    f = chdir_tmp / "file.txt"
    f.write_text("hi")
    assert _safe_output_dir(str(f)) is None


def test_accepts_symlink_to_directory_inside_cwd(chdir_tmp: Path) -> None:
    """Symlinks pointing inside cwd must be accepted (calibration scenario forks)."""
    real = chdir_tmp / "output" / "real"
    real.mkdir(parents=True)
    link = chdir_tmp / "output" / "linked"
    link.symlink_to(real)
    result = _safe_output_dir(str(link))
    assert result is not None
    # The helper resolves symlinks before checking; the resolved path is `real`.
    assert result == real.resolve()


def test_rejects_symlink_pointing_outside_cwd(chdir_tmp: Path,
                                              tmp_path_factory: pytest.TempPathFactory) -> None:
    """Symlinks that escape cwd must be rejected (this is the security-relevant case)."""
    outside = tmp_path_factory.mktemp("outside")
    link = chdir_tmp / "escape-link"
    link.symlink_to(outside)
    assert _safe_output_dir(str(link)) is None


def test_accepts_cwd_itself(chdir_tmp: Path) -> None:
    """Edge case: the working directory itself is inside cwd."""
    result = _safe_output_dir(str(chdir_tmp))
    assert result == chdir_tmp.resolve()


@pytest.mark.skipif(os.name == "nt", reason="POSIX path semantics")
def test_rejects_root(chdir_tmp: Path) -> None:
    assert _safe_output_dir("/") is None
