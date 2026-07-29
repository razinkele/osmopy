"""Unit tests for osmose.java_config_reconcile — the Java-4.4.1 name/matrix reconciliation
pass that lets the disaggregated (underscored-name) Baltic config load and run on the Java jar.

Root cause it addresses: Java 4.4.1 Species.java strips '_'/'-' from species.name.spN
(cod_west -> codwest) but leaves NAME-BASED references (movement.species.mapN, matrix
headers/rows) untouched, so name resolution fails; plus the disaggregation left the discards
matrix stale (aggregate 'cod', missing the cod_east fishery column)."""

from pathlib import Path

import pytest

from osmose.java_config_reconcile import reconcile_config_for_java, sanitize_java_name


def test_sanitize_java_name_mirrors_java_stripping():
    assert sanitize_java_name("cod_west") == "codwest"
    assert sanitize_java_name("cod_east") == "codeast"
    assert sanitize_java_name("trawlcod_east") == "trawlcodeast"
    assert sanitize_java_name("cod-east") == "codeast"  # hyphen too
    assert sanitize_java_name("herring") == "herring"  # no-op on clean names


def _write_disagg_stage(stage: Path) -> None:
    """Minimal staged config mirroring the disaggregated structure that crashes Java:
    underscored names, a duplicate predator column, and a stale discards matrix."""
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "osm_all-parameters.csv").write_text(
        "species.name.sp0 ; cod_west\n"
        "species.name.sp1 ; cod_east\n"
        "species.type.sp0 ; focal\n"
        "species.type.sp1 ; focal\n"
        "movement.species.map0 ; cod_west\n"
        "movement.species.map1 ; cod_east\n"
        "movement.file.map0 ; maps/cod_west_juvenile.csv\n"  # file PATH — must NOT be mangled
        "fisheries.name.fsh0 ; trawlcod_east\n"
        "predation.accessibility.file ; predation-accessibility.csv\n"
        "fisheries.catchability.file ; fishery-catchability.csv\n"
        "fisheries.discards.file ; fishery-discards.csv\n"
    )
    # accessibility: duplicate 'Cormorant' predator column (background staging re-added it)
    (stage / "predation-accessibility.csv").write_text(
        "v Prey / Predator >;cod_west;cod_east;Cormorant;Cormorant\n"
        "cod_west;0;0;0.05;0.05\n"
        "cod_east;0;0;0.05;0.05\n"
    )
    # catchability: disaggregated (cod_west/cod_east rows, trawlcod_east column)
    (stage / "fishery-catchability.csv").write_text(",trawlcod_east\ncod_west,0\ncod_east,1\n")
    # discards: STALE — aggregate 'cod', missing cod_east row and the trawlcod_east column
    (stage / "fishery-discards.csv").write_text(",trawlcod\ncod,0\n")


def test_reconcile_sanitizes_master_name_keys_but_not_file_paths(tmp_path):
    stage = tmp_path / "stage"
    _write_disagg_stage(stage)
    reconcile_config_for_java(stage)
    master = (stage / "osm_all-parameters.csv").read_text()
    assert "movement.species.map0 ; codwest" in master
    assert "movement.species.map1 ; codeast" in master
    assert "species.name.sp0 ; codwest" in master
    assert "fisheries.name.fsh0 ; trawlcodeast" in master
    # file-path value untouched (mangling it would break the map file lookup)
    assert "movement.file.map0 ; maps/cod_west_juvenile.csv" in master


def test_reconcile_dedups_duplicate_predator_columns(tmp_path):
    stage = tmp_path / "stage"
    _write_disagg_stage(stage)
    reconcile_config_for_java(stage)
    header = (stage / "predation-accessibility.csv").read_text().splitlines()[0].split(";")
    assert header.count("Cormorant") == 1  # duplicate collapsed
    # prey rows sanitized
    prey = [
        ln.split(";")[0]
        for ln in (stage / "predation-accessibility.csv").read_text().splitlines()[1:]
    ]
    assert prey == ["codwest", "codeast"]


def test_reconcile_makes_discards_structurally_match_catchability(tmp_path):
    stage = tmp_path / "stage"
    _write_disagg_stage(stage)
    reconcile_config_for_java(stage)
    cat = (stage / "fishery-catchability.csv").read_text().splitlines()
    disc = (stage / "fishery-discards.csv").read_text().splitlines()
    # discards fishery columns == catchability's (sanitized), incl. trawlcodeast
    assert disc[0] == cat[0]
    assert "trawlcodeast" in disc[0]
    # discards has a row for every prey in the accessibility universe (codwest, codeast)
    disc_rows = {ln.split(",")[0] for ln in disc[1:]}
    assert {"codwest", "codeast"} <= disc_rows
    # all-zero preserved (source discards were all zero)
    assert all(float(c) == 0.0 for ln in disc[1:] for c in ln.split(",")[1:])


def test_reconcile_preserves_catchability_values(tmp_path):
    stage = tmp_path / "stage"
    _write_disagg_stage(stage)
    reconcile_config_for_java(stage)
    rows = {
        ln.split(",")[0]: ln.split(",")[1:]
        for ln in (stage / "fishery-catchability.csv").read_text().splitlines()[1:]
    }
    assert rows["codeast"] == ["1"]  # identity catchability preserved through sanitize/reconcile
    assert rows["codwest"] == ["0"]


def test_reconcile_raises_on_name_collision(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "osm_all-parameters.csv").write_text(
        "species.name.sp0 ; cod_west\n"
        "species.name.sp1 ; cod-west\n"  # both strip to 'codwest' — ambiguous
        "species.type.sp0 ; focal\n"
        "species.type.sp1 ; focal\n"
    )
    with pytest.raises(ValueError, match="collide"):
        reconcile_config_for_java(stage)
