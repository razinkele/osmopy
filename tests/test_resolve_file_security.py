"""Tests for _resolve_file path traversal guard."""

from pathlib import Path

from osmose.engine.config import _resolve_file, _set_config_dir


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    """File keys with '..' should be rejected."""
    _set_config_dir(str(tmp_path))
    inner = tmp_path / "subdir"
    inner.mkdir()
    secret = tmp_path / "secret.csv"
    secret.write_text("data")
    result = _resolve_file("subdir/../secret.csv")
    assert result is None


def test_rejects_bare_parent_traversal(tmp_path: Path) -> None:
    """Bare '../file' should be rejected."""
    _set_config_dir(str(tmp_path / "subdir"))
    (tmp_path / "subdir").mkdir(exist_ok=True)
    secret = tmp_path / "secret.csv"
    secret.write_text("data")
    result = _resolve_file("../secret.csv")
    assert result is None


def test_rejects_absolute_path_outside_config_dir(tmp_path: Path) -> None:
    """Absolute paths not under any search dir should be rejected."""
    _set_config_dir(str(tmp_path))
    result = _resolve_file("/etc/hosts")
    assert result is None


def test_allows_valid_relative_path(tmp_path: Path) -> None:
    """Valid relative paths within search dirs should resolve normally."""
    _set_config_dir(str(tmp_path))
    data_file = tmp_path / "grid.csv"
    data_file.write_text("1;2;3")
    result = _resolve_file("grid.csv")
    assert result is not None
    assert result.name == "grid.csv"


def test_allows_subdirectory_path(tmp_path: Path) -> None:
    """Paths in subdirectories within config dir should work."""
    _set_config_dir(str(tmp_path))
    subdir = tmp_path / "maps"
    subdir.mkdir()
    data_file = subdir / "movement.csv"
    data_file.write_text("1;2;3")
    result = _resolve_file("maps/movement.csv")
    assert result is not None
    assert result.name == "movement.csv"


def test_empty_key_returns_none() -> None:
    """Empty string should return None (not raise)."""
    result = _resolve_file("")
    assert result is None
