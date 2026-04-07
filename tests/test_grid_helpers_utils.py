import numpy as np

from ui.pages.grid_helpers import _compute_half_extents, _find_config_file


def test_find_config_file_returns_existing(tmp_path):
    f = tmp_path / "grid" / "mask.csv"
    f.parent.mkdir()
    f.write_text("1;2;3")
    result = _find_config_file("grid/mask.csv", config_dir=tmp_path)
    assert result == f.resolve()


def test_find_config_file_returns_none_for_missing(tmp_path):
    result = _find_config_file("nonexistent.csv", config_dir=tmp_path)
    assert result is None


def test_find_config_file_rejects_traversal(tmp_path):
    result = _find_config_file("../../etc/passwd", config_dir=tmp_path)
    assert result is None


def test_find_config_file_falls_back_to_examples(tmp_path):
    result = _find_config_file("nonexistent_in_both.csv", config_dir=tmp_path)
    assert result is None


def test_half_extents_regular_grid():
    lat = np.array([[48.0, 48.0], [47.0, 47.0]])
    lon = np.array([[1.0, 2.0], [1.0, 2.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert hlat.shape == (2, 2)
    assert hlon.shape == (2, 2)
    assert np.all(hlat > 0)
    assert np.all(hlon > 0)


def test_half_extents_single_row():
    lat = np.array([[48.0, 48.0, 48.0]])
    lon = np.array([[1.0, 2.0, 3.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert hlat.shape == (1, 3)
    assert np.all(hlon > 0)


def test_half_extents_single_column():
    lat = np.array([[48.0], [47.0], [46.0]])
    lon = np.array([[1.0], [1.0], [1.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert hlat.shape == (3, 1)
    assert np.all(hlat > 0)
    assert np.all(hlon > 0)


def test_half_extents_single_cell():
    lat = np.array([[48.0]])
    lon = np.array([[1.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert hlat.shape == (1, 1)
    assert hlat[0, 0] > 0
    assert hlon[0, 0] > 0


def test_half_extents_zero_step_fallback():
    lat = np.array([[48.0, 48.0], [48.0, 48.0]])
    lon = np.array([[1.0, 2.0], [1.0, 2.0]])
    hlat, hlon = _compute_half_extents(lat, lon)
    assert np.all(hlat > 0), "Zero-step dlat should fall back to lon_step"
