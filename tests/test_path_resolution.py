"""Tests for the consolidated path resolution module."""



from osmose.engine.path_resolution import resolve_data_path


def test_absolute_path_under_config_dir(tmp_path):
    data_file = tmp_path / "forcing.csv"
    data_file.write_text("x")
    result = resolve_data_path(str(data_file), config_dir=str(tmp_path))
    assert result == data_file


def test_relative_path_found_in_config_dir(tmp_path):
    data_file = tmp_path / "maps" / "map0.csv"
    data_file.parent.mkdir()
    data_file.write_text("x")
    result = resolve_data_path("maps/map0.csv", config_dir=str(tmp_path))
    assert result == data_file


def test_path_traversal_rejected(tmp_path):
    result = resolve_data_path("../../etc/passwd", config_dir=str(tmp_path))
    assert result is None


def test_empty_key_returns_none():
    result = resolve_data_path("", config_dir="/tmp")
    assert result is None


def test_not_found_returns_none(tmp_path):
    result = resolve_data_path("nonexistent.csv", config_dir=str(tmp_path))
    assert result is None
