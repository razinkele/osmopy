"""Tests for resolve_data_path path traversal guard."""

from pathlib import Path

from osmose.engine.path_resolution import resolve_data_path as _resolve_file


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    """File keys with '..' should be rejected."""
    inner = tmp_path / "subdir"
    inner.mkdir()
    secret = tmp_path / "secret.csv"
    secret.write_text("data")
    result = _resolve_file("subdir/../secret.csv", config_dir=str(tmp_path))
    assert result is None


def test_rejects_bare_parent_traversal(tmp_path: Path) -> None:
    """Bare '../file' should be rejected."""
    (tmp_path / "subdir").mkdir(exist_ok=True)
    secret = tmp_path / "secret.csv"
    secret.write_text("data")
    result = _resolve_file("../secret.csv", config_dir=str(tmp_path / "subdir"))
    assert result is None


def test_accepts_absolute_path_that_exists(tmp_path: Path) -> None:
    """Absolute paths that exist are accepted (security is via '..' guard)."""
    data_file = tmp_path / "data.csv"
    data_file.write_text("x")
    result = _resolve_file(str(data_file), config_dir=str(tmp_path))
    assert result == data_file


def test_rejects_absolute_path_with_traversal(tmp_path: Path) -> None:
    """Absolute paths containing '..' should be rejected."""
    result = _resolve_file(str(tmp_path / "sub" / ".." / "secret.csv"), config_dir=str(tmp_path))
    assert result is None


def test_allows_valid_relative_path(tmp_path: Path) -> None:
    """Valid relative paths within search dirs should resolve normally."""
    data_file = tmp_path / "grid.csv"
    data_file.write_text("1;2;3")
    result = _resolve_file("grid.csv", config_dir=str(tmp_path))
    assert result is not None
    assert result.name == "grid.csv"


def test_allows_subdirectory_path(tmp_path: Path) -> None:
    """Paths in subdirectories within config dir should work."""
    subdir = tmp_path / "maps"
    subdir.mkdir()
    data_file = subdir / "movement.csv"
    data_file.write_text("1;2;3")
    result = _resolve_file("maps/movement.csv", config_dir=str(tmp_path))
    assert result is not None
    assert result.name == "movement.csv"


def test_empty_key_returns_none() -> None:
    """Empty string should return None (not raise)."""
    result = _resolve_file("")
    assert result is None
