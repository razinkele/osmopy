"""Tests for copy_data_files helper."""
from ui.pages.run import copy_data_files


def test_copies_referenced_csv(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "data.csv").write_text("a,b\n1,2\n")

    config = {"some.key": "data.csv"}
    skipped = copy_data_files(config, source, dest)

    assert skipped == []
    assert (dest / "data.csv").exists()
    assert (dest / "data.csv").read_text() == "a,b\n1,2\n"


def test_copies_nested_file(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "sub").mkdir()
    (source / "sub" / "file.nc").write_bytes(b"fake")

    config = {"grid.file": "sub/file.nc"}
    skipped = copy_data_files(config, source, dest)

    assert skipped == []
    assert (dest / "sub" / "file.nc").exists()


def test_skips_configuration_keys(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    config = {"osmose.configuration.output": "output/file.csv"}
    skipped = copy_data_files(config, source, dest)
    assert skipped == []  # skipped silently (not counted as missing)


def test_rejects_source_path_traversal(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    config = {"evil.key": "../../etc/passwd"}
    skipped = copy_data_files(config, source, dest)
    assert "../../etc/passwd" in skipped


def test_reports_missing_files(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    config = {"data.key": "nonexistent.csv"}
    skipped = copy_data_files(config, source, dest)
    assert "nonexistent.csv" in skipped


def test_skips_non_data_values(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    config = {"simulation.time.step": "24", "species.name.sp0": "Anchovy"}
    skipped = copy_data_files(config, source, dest)
    assert skipped == []
    assert not list(dest.iterdir())  # nothing copied


def test_copies_txt_file(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "params.txt").write_text("param=value\n")

    config = {"some.param": "params.txt"}
    skipped = copy_data_files(config, source, dest)

    assert skipped == []
    assert (dest / "params.txt").exists()


def test_copies_properties_file(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "config.properties").write_text("key=value\n")

    config = {"cfg.file": "config.properties"}
    skipped = copy_data_files(config, source, dest)

    assert skipped == []
    assert (dest / "config.properties").exists()


def test_copies_json_file(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "data.json").write_text('{"key": "value"}\n')

    config = {"json.file": "data.json"}
    skipped = copy_data_files(config, source, dest)

    assert skipped == []
    assert (dest / "data.json").exists()


def test_copies_dat_file(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "data.dat").write_bytes(b"\x00\x01\x02")

    config = {"dat.file": "data.dat"}
    skipped = copy_data_files(config, source, dest)

    assert skipped == []
    assert (dest / "data.dat").exists()


def test_creates_parent_directories(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "a" / "b").mkdir(parents=True)
    (source / "a" / "b" / "deep.csv").write_text("deep\n")

    config = {"deep.key": "a/b/deep.csv"}
    skipped = copy_data_files(config, source, dest)

    assert skipped == []
    assert (dest / "a" / "b" / "deep.csv").exists()


def test_skips_slash_value_pointing_outside(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    # A value with "/" but resolves outside source_dir
    config = {"evil.slash": "../outside/file.csv"}
    skipped = copy_data_files(config, source, dest)
    assert "../outside/file.csv" in skipped


def test_empty_config(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    skipped = copy_data_files({}, source, dest)
    assert skipped == []
