"""Unit tests for osmose.java_background_staging (the Baltic Java-4.4.1 staging recipe, C2)."""

import hashlib
import shutil
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.java_background_staging import (
    augment_accessibility,
    background_staging_supported,
    inline_biomass_series,
)


def test_inline_biomass_from_nc():
    # domain-total per-step series, length = ndt
    series = inline_biomass_series("data/baltic/baltic_predator_biomass.nc", "GreySeal")
    assert len(series) == 24
    assert abs(series[0] - 4500.0) < 1.0  # documented standing biomass


def test_augment_accessibility_adds_predator_columns(tmp_path):
    src = tmp_path / "predation-accessibility.csv"
    src.write_text("v Prey / Predator >;cod;herring\ncod;0.05;0\nherring;0.4;0\n")
    augment_accessibility(
        src, {"GreySeal": {"herring": 0.4, "cod": 0.3}, "Cormorant": {"herring": 0.3}}
    )
    lines = src.read_text().splitlines()
    header = lines[0].split(";")
    assert "GreySeal" in header and "Cormorant" in header  # predator columns added
    assert any(ln.startswith("GreySeal;") for ln in lines)  # apex prey rows added
    gs_col = header.index("GreySeal")
    herring_row = next(ln.split(";") for ln in lines if ln.startswith("herring;"))
    assert float(herring_row[gs_col]) == 0.4  # authored value present


def test_augment_accessibility_does_not_touch_source(tmp_path):
    """augment_accessibility writes ONLY to its target path; the canonical source is byte-identical."""
    source_path = Path("data/baltic/predation-accessibility.csv")
    assert source_path.exists()
    before = source_path.read_bytes()
    before_hash = hashlib.sha256(before).hexdigest()

    tmp_copy = tmp_path / "predation-accessibility.csv"
    shutil.copy(source_path, tmp_copy)
    augment_accessibility(tmp_copy, {"GreySeal": {"herring": 0.4}, "Cormorant": {"sprat": 0.3}})

    assert "GreySeal" in tmp_copy.read_text().splitlines()[0]  # copy changed
    after = source_path.read_bytes()
    assert hashlib.sha256(after).hexdigest() == before_hash and after == before  # source untouched


def test_background_staging_supported():
    baltic = dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))
    assert background_staging_supported(baltic) is True  # GreySeal + Cormorant -> known
    # an unknown background species -> not supported
    unknown = {"species.type.sp9": "background", "species.name.sp9": "Yeti"}
    assert background_staging_supported(unknown) is False
    # no background species -> not "supported" (nothing to stage)
    assert background_staging_supported({"species.type.sp0": "focal"}) is False


def test_stage_background_for_java_emits_keys_and_returns_cutoff_override(tmp_path):
    """End-to-end staging on a copied Baltic config dir: emits the per-background keys + matrices,
    returns the cutoff -P override, and never touches data/."""
    from ui.pages.run import write_temp_config

    src = Path("data/baltic")
    raw = dict(OsmoseConfigReader().read(str(src / "baltic_all-parameters.csv")))
    stage = tmp_path / "stage"
    write_temp_config(raw, stage, source_dir=src, target_version="4.4.1")

    from osmose.java_background_staging import stage_background_for_java

    overrides = stage_background_for_java(stage, raw)
    assert overrides == {"output.cutoff.enabled": "false"}
    master = (stage / "osm_all-parameters.csv").read_text()
    assert "species.biomass.sp14 ;" in master  # GreySeal inline biomass
    assert "output.diet.stage.threshold.sp14 ; 90" in master  # diet-stage threshold
    assert "simulation.nschool.sp14 ; 10" in master
    # accessibility matrix got the predator columns (staged copy)
    assert "GreySeal" in (stage / "predation-accessibility.csv").read_text().splitlines()[0]
    # source untouched
    assert (src / "predation-accessibility.csv").read_text().splitlines()[0].count("GreySeal") == 0
