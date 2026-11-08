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


def _write_stage_structured(stage: Path) -> None:
    """A staged config whose predation-accessibility matrix is STAGE-STRUCTURED, as EEC's is:
    each prey has a juvenile row (`name < threshold`) and an adult row (`name`)."""
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "osm_all-parameters.csv").write_text(
        "species.name.sp0 ; cod\n"
        "species.name.sp1 ; sole\n"
        "predation.accessibility.file ; predation-accessibility.csv\n"
        "fisheries.catchability.file ; fishery-catchability.csv\n"
        "fisheries.discards.file ; fishery-discards.csv\n"
    )
    (stage / "predation-accessibility.csv").write_text(
        "v Prey / Predator >;cod < 0.45;cod;sole < 0.25;sole\n"
        "cod < 0.45 ;0.8;0.8;0;0\n"
        "cod;0.1;0.1;0;0\n"
        "sole < 0.25 ;0.8;0.8;0;0\n"
        "sole;0.1;0.1;0;0\n"
    )
    (stage / "fishery-catchability.csv").write_text(",trawl,net\ncod,1,0\nsole,0,1\n")
    (stage / "fishery-discards.csv").write_text(",trawl,net\ncod,0,0\nsole,0,0.2\n")


def test_reconcile_gives_fishery_matrices_one_row_per_species_not_per_stage(tmp_path):
    """Fishery matrices are per species. Taking their rows from a stage-structured accessibility
    matrix added zero-filled `cod < 0.45`-style rows ahead of each species' real row. Since #223
    put every Java path through this reconcile, that cut EEC's Java yields to about 1e-4 of
    Python's while biomass still agreed (2026-11-08 parity re-run). Rows must be the species
    names, once each, with their values preserved."""
    stage = tmp_path / "stage"
    _write_stage_structured(stage)
    reconcile_config_for_java(stage)
    for name, want in (
        ("fishery-catchability.csv", {"cod": ["1", "0"], "sole": ["0", "1"]}),
        ("fishery-discards.csv", {"cod": ["0", "0"], "sole": ["0", "0.2"]}),
    ):
        lines = (stage / name).read_text().splitlines()
        rows = [ln.split(",") for ln in lines[1:]]
        labels = [r[0] for r in rows]
        assert labels == ["cod", "sole"], (name, labels)
        assert {r[0]: r[1:] for r in rows} == want, name


def test_reconcile_matches_fishery_matrix_labels_despite_surrounding_whitespace(tmp_path):
    """EEC's fishery-catchability.csv labels a row `pouting ` (trailing space). Java trims it, and
    pouting's yield matched Python before #223. The reconcile looked values up by the untrimmed
    label, missed it and zero-filled the row, leaving pouting's Java yield at about 1e-4 of
    Python's. Labels must match after trimming."""
    stage = tmp_path / "stage"
    _write_stage_structured(stage)
    (stage / "fishery-catchability.csv").write_text(",trawl , net\ncod ,1,0\n sole,0,1\n")
    reconcile_config_for_java(stage)
    rows = {
        ln.split(",")[0]: ln.split(",")[1:]
        for ln in (stage / "fishery-catchability.csv").read_text().splitlines()[1:]
    }
    assert rows == {"cod": ["1", "0"], "sole": ["0", "1"]}
