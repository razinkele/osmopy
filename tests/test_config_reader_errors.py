"""Tests for OsmoseConfigReader error handling paths."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from osmose.config.reader import OsmoseConfigReader


# ---------------------------------------------------------------------------
# Circular reference
# ---------------------------------------------------------------------------


def test_circular_reference_terminates(tmp_path):
    """Two config files referencing each other must not infinite-loop."""
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    file_a.write_text(f"osmose.configuration.b ; {file_b.name}\nkey.a ; val_a\n")
    file_b.write_text(f"osmose.configuration.a ; {file_a.name}\nkey.b ; val_b\n")

    reader = OsmoseConfigReader()
    result = reader.read(file_a)

    # Both keys should be present — the cycle is broken, not silently dropped
    assert result["key.a"] == "val_a"
    assert result["key.b"] == "val_b"


def test_circular_reference_logs_warning(tmp_path, caplog):
    """The circular reference skip must emit a WARNING log entry."""
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    file_a.write_text(f"osmose.configuration.b ; {file_b.name}\nkey.a ; val_a\n")
    file_b.write_text(f"osmose.configuration.a ; {file_a.name}\nkey.b ; val_b\n")

    reader = OsmoseConfigReader()
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        reader.read(file_a)

    assert "ircular" in caplog.text or "circular" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Path escape protection
# ---------------------------------------------------------------------------


def test_path_escape_blocked(tmp_path):
    """Sub-config paths that escape the config directory must be skipped."""
    master = tmp_path / "master.csv"
    # Use ../../etc/passwd — should be skipped, not crash
    master.write_text("osmose.configuration.escape ; ../../etc/passwd\nvalid.key ; ok\n")

    reader = OsmoseConfigReader()
    result = reader.read(master)

    # The valid key is still loaded; no crash
    assert result["valid.key"] == "ok"
    # The escaped path is not loaded as a key
    assert "root" not in str(result)


def test_path_escape_logs_warning(tmp_path, caplog):
    """A path-escape attempt must emit a WARNING log entry."""
    master = tmp_path / "master.csv"
    master.write_text("osmose.configuration.escape ; ../../etc/passwd\nvalid.key ; ok\n")

    reader = OsmoseConfigReader()
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        reader.read(master)

    warning_text = caplog.text.lower()
    assert "escapes" in warning_text or "skip" in warning_text or "escape" in warning_text


# ---------------------------------------------------------------------------
# Oversized file
# ---------------------------------------------------------------------------


def test_oversized_file_raises(tmp_path):
    """A config file larger than 10 MB must raise ValueError."""
    big_file = tmp_path / "big.csv"
    big_file.write_text("key ; value\n")

    # Patch stat to return a size above the 10MB threshold
    original_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        st = original_stat(self, *args, **kwargs)
        # Return a stat_result-like object with inflated st_size
        import os
        fields = list(st)
        # st_size is index 6 in the tuple
        fields[6] = 10_000_001
        return os.stat_result(fields)

    with patch.object(Path, "stat", fake_stat):
        reader = OsmoseConfigReader()
        with pytest.raises(ValueError, match="too large"):
            reader.read_file(big_file)


# ---------------------------------------------------------------------------
# Missing sub-config
# ---------------------------------------------------------------------------


def test_missing_subconfig_continues(tmp_path):
    """A missing referenced sub-config must not crash; other keys still load."""
    master = tmp_path / "master.csv"
    master.write_text(
        "simulation.nspecies ; 3\n"
        "osmose.configuration.missing ; does_not_exist.csv\n"
        "extra.key ; present\n"
    )

    reader = OsmoseConfigReader()
    result = reader.read(master)

    assert result["simulation.nspecies"] == "3"
    assert result["extra.key"] == "present"


def test_missing_subconfig_logs_warning(tmp_path, caplog):
    """A missing sub-config must emit a WARNING log entry."""
    master = tmp_path / "master.csv"
    master.write_text("osmose.configuration.ghost ; ghost.csv\nfoo ; bar\n")

    reader = OsmoseConfigReader()
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        reader.read(master)

    assert "ghost" in caplog.text or "not found" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Unparseable lines
# ---------------------------------------------------------------------------


def test_unparseable_lines_skipped(tmp_path):
    """Lines without a recognised key-value separator are silently skipped."""
    config_file = tmp_path / "config.csv"
    config_file.write_text(
        "thisisnotavalidline\n"
        "another_bad_line\n"
        "good.key ; good_value\n"
    )

    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)

    assert "good.key" in result
    assert result["good.key"] == "good_value"
    assert "thisisnotavalidline" not in result
    assert "another_bad_line" not in result


def test_unparseable_lines_counted(tmp_path):
    """skipped_lines counter increments for each unparseable line."""
    config_file = tmp_path / "config.csv"
    config_file.write_text(
        "bad_line_one\n"
        "bad_line_two\n"
        "good.key ; value\n"
    )

    reader = OsmoseConfigReader()
    reader.read_file(config_file)

    assert reader.skipped_lines == 2


def test_skipped_lines_reset_between_reads(tmp_path):
    """Calling read() resets the skipped_lines counter from a prior call."""
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("not_parseable\nvalid.key ; v\n")

    good_file = tmp_path / "good.csv"
    good_file.write_text("only.valid ; key\n")

    reader = OsmoseConfigReader()
    reader.read(bad_file)
    assert reader.skipped_lines == 1

    reader.read(good_file)
    assert reader.skipped_lines == 0


def test_unparseable_line_logged_at_warning(tmp_path, caplog):
    """Each skipped line must be logged at WARNING level with its content."""
    config_file = tmp_path / "config.csv"
    config_file.write_text("noseparatorhere\nvalid.key ; ok\n")

    reader = OsmoseConfigReader()
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        reader.read_file(config_file)

    assert "noseparatorhere" in caplog.text
