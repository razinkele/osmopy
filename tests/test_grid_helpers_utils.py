from ui.pages.grid_helpers import _find_config_file


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
