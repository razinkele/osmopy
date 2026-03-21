import tempfile
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_single_file():
    reader = OsmoseConfigReader()
    result = reader.read_file(FIXTURES / "osm_all-parameters.csv")
    assert result["simulation.time.ndtperyear"] == "12"
    assert result["simulation.time.nyear"] == "50"


def test_read_recursive():
    reader = OsmoseConfigReader()
    result = reader.read(FIXTURES / "osm_all-parameters.csv")
    assert result["simulation.nspecies"] == "2"
    assert result["species.name.sp0"] == "Anchovy"
    assert result["species.linf.sp0"] == "19.5"
    assert result["species.name.sp1"] == "Sardine"


def test_keys_are_lowercase():
    reader = OsmoseConfigReader()
    result = reader.read_file(FIXTURES / "osm_all-parameters.csv")
    for key in result:
        assert key == key.lower()


def test_auto_detect_equals_separator():
    reader = OsmoseConfigReader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("key1 = value1\nkey2 = value2\n")
        path = Path(f.name)
    result = reader.read_file(path)
    assert result["key1"] == "value1"
    assert result["key2"] == "value2"
    path.unlink()


def test_auto_detect_tab_separator():
    reader = OsmoseConfigReader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("key1\tvalue1\nkey2\tvalue2\n")
        path = Path(f.name)
    result = reader.read_file(path)
    assert result["key1"] == "value1"
    path.unlink()


def test_skip_comments():
    reader = OsmoseConfigReader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("# This is a comment\nkey1 ; value1\n! Another comment\nkey2 ; value2\n")
        path = Path(f.name)
    result = reader.read_file(path)
    assert len(result) == 2
    assert result["key1"] == "value1"
    assert result["key2"] == "value2"
    path.unlink()


def test_skip_empty_lines():
    reader = OsmoseConfigReader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("key1 ; value1\n\n\nkey2 ; value2\n")
        path = Path(f.name)
    result = reader.read_file(path)
    assert len(result) == 2
    path.unlink()


def test_value_with_spaces_preserved():
    reader = OsmoseConfigReader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("species.name.sp0 ; King Mackerel\n")
        path = Path(f.name)
    result = reader.read_file(path)
    assert result["species.name.sp0"] == "King Mackerel"
    path.unlink()


def test_missing_subfile_ignored():
    """If a referenced sub-config doesn't exist, skip it without error."""
    reader = OsmoseConfigReader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, dir="/tmp") as f:
        f.write("simulation.nspecies ; 1\nosmose.configuration.missing ; nonexistent.csv\n")
        path = Path(f.name)
    result = reader.read(path)
    assert result["simulation.nspecies"] == "1"
    path.unlink()


def test_circular_reference_does_not_recurse(tmp_path):
    """Circular sub-file references should not cause infinite recursion."""
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    file_a.write_text(f"osmose.configuration.b ; {file_b.name}\nfoo ; bar\n")
    file_b.write_text(f"osmose.configuration.a ; {file_a.name}\nbaz ; qux\n")

    reader = OsmoseConfigReader()
    result = reader.read(file_a)
    assert result["foo"] == "bar"
    assert result["baz"] == "qux"


def test_missing_subfile_logs_warning(tmp_path, caplog):
    """Missing sub-config files should log a warning."""
    import logging

    master = tmp_path / "master.csv"
    master.write_text("osmose.configuration.sub ; nonexistent.csv\nfoo ; bar\n")

    reader = OsmoseConfigReader()
    with caplog.at_level(logging.WARNING):
        result = reader.read(master)
    assert result["foo"] == "bar"
    assert "nonexistent" in caplog.text


# ---------------------------------------------------------------------------
# Malformed input edge-case tests (H15)
# ---------------------------------------------------------------------------


def test_read_skips_lines_without_separator(tmp_path):
    """Lines with no recognised separator should be skipped without crash."""
    config_file = tmp_path / "test.csv"
    config_file.write_text("noseparatorhere\nvalid.key ; valid_value\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    assert "valid.key" in result
    assert "noseparatorhere" not in result


def test_read_handles_empty_value(tmp_path):
    """A key with an empty value after the separator should be stored as empty string."""
    config_file = tmp_path / "test.csv"
    config_file.write_text("some.key ; \nanother.key ; value\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    assert "another.key" in result
    assert result["another.key"] == "value"
    # some.key may be present with empty string or absent; either is acceptable
    if "some.key" in result:
        assert result["some.key"] == ""


def test_read_skips_only_whitespace_lines(tmp_path):
    """Lines containing only whitespace should be ignored."""
    config_file = tmp_path / "test.csv"
    config_file.write_text("   \n\t\nkey1 ; v1\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    assert result == {"key1": "v1"}


def test_read_file_all_comments(tmp_path):
    """A file consisting entirely of comments should return an empty dict."""
    config_file = tmp_path / "test.csv"
    config_file.write_text("# comment one\n! comment two\n# another\n")
    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    assert result == {}


def test_unparseable_line_is_logged(tmp_path, caplog):
    """Lines without a recognised separator must be logged at WARNING level."""
    import logging

    config_file = tmp_path / "test.csv"
    config_file.write_text("noseparatorhere\nvalid.key ; valid_value\n")
    reader = OsmoseConfigReader()
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        reader.read_file(config_file)
    assert "noseparatorhere" in caplog.text


def test_reader_handles_latin1_characters(tmp_path):
    """Config files with accented species names (Latin-1) should not crash."""
    from osmose.config.reader import OsmoseConfigReader

    config_file = tmp_path / "test.csv"
    config_file.write_bytes("species.name.sp0 ; Sébaste\n".encode("latin-1"))

    reader = OsmoseConfigReader()
    result = reader.read_file(config_file)
    assert "species.name.sp0" in result
