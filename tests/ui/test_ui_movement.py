"""Tests for pure helpers used by ui/pages/movement.py."""

from ui.pages._helpers import count_map_entries


def test_count_map_entries_one_map():
    cfg = {"movement.file.map0": "map0.csv"}
    assert count_map_entries(cfg) == 1


def test_count_map_entries_multiple():
    cfg = {
        "movement.file.map0": "map0.csv",
        "movement.file.map1": "map1.csv",
        "movement.file.map3": "map3.csv",
    }
    assert count_map_entries(cfg) == 3


def test_count_map_entries_excludes_null():
    cfg = {"movement.file.map0": "null", "movement.file.map1": "real.csv"}
    assert count_map_entries(cfg) == 1


def test_count_map_entries_excludes_empty():
    cfg = {"movement.file.map0": "", "movement.file.map1": "real.csv"}
    assert count_map_entries(cfg) == 1


def test_count_map_entries_empty_config():
    assert count_map_entries({}) == 0
