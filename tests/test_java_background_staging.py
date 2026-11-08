"""Unit tests for osmose.java_background_staging (the Baltic Java-4.4.1 staging recipe, C2)."""

import csv
import hashlib
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.java_background_staging import (
    BG_ACCESS,
    _rescale_background_weights,
    augment_accessibility,
    background_staging_supported,
    inline_biomass_series,
)

_JAR = os.environ.get("OSMOSE_JAR")


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
    source_matrix = (src / "predation-accessibility.csv").read_text()
    raw = dict(OsmoseConfigReader().read(str(src / "baltic_all-parameters.csv")))
    stage = tmp_path / "stage"
    write_temp_config(raw, stage, source_dir=src, target_version="4.4.1")

    from osmose.java_background_staging import stage_background_for_java

    overrides = stage_background_for_java(stage, raw)
    assert overrides == {"output.cutoff.enabled": "false"}
    master = (stage / "osm_all-parameters.csv").read_text()
    # Disaggregation shifted the background block +1: GreySeal is sp15 (Cormorant sp16).
    assert "species.biomass.sp15 ;" in master  # GreySeal inline biomass
    assert "output.diet.stage.threshold.sp15 ; 90" in master  # diet-stage threshold
    assert "simulation.nschool.sp15 ; 10" in master
    # the staged matrix has exactly one GreySeal column: the source already carries it (since
    # 2026-11-08), so staging must not append a duplicate
    staged_header = (stage / "predation-accessibility.csv").read_text().splitlines()[0]
    assert staged_header.split(";").count("GreySeal") == 1
    # source untouched
    assert (src / "predation-accessibility.csv").read_text() == source_matrix


def test_stage_background_raises_clear_error_when_predator_nc_missing(tmp_path):
    """A missing predator NetCDF -> a clear FileNotFoundError, not a silent StopIteration."""
    import pytest

    from osmose.java_background_staging import stage_background_for_java

    (tmp_path / "osm_all-parameters.csv").write_text("simulation.time.ndtperyear ; 24\n")
    raw = {"species.type.sp14": "background", "species.name.sp14": "GreySeal"}
    with pytest.raises(FileNotFoundError, match="predator-biomass NetCDF"):
        stage_background_for_java(tmp_path, raw)


def _stage_production_matrix(tmp_path):
    """Stage the REAL production matrix with the REAL BG_ACCESS — not a synthetic fixture.

    The cod E/W regression these tests guard was invisible to synthetic-prey-name tests: `BG_ACCESS`
    kept the pre-split `cod` key while the production matrix had `cod_west`/`cod_east`, and
    `augment_accessibility` defaults an unmatched prey key to 0.0. Only staging the real file can
    catch that class.
    """
    src = Path("data/baltic/predation-accessibility.csv")
    dst = tmp_path / "predation-accessibility.csv"
    shutil.copy(src, dst)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the unused pre-split `cod` key warns by design
        augment_accessibility(dst, {k: dict(v) for k, v in BG_ACCESS.items()})
    lines = [ln for ln in dst.read_text().splitlines() if ln.strip()]
    header = [h.strip() for h in lines[0].split(";")]
    rows = {ln.split(";")[0].strip(): [c.strip() for c in ln.split(";")] for ln in lines[1:]}
    return header, rows


def test_staged_matrix_gives_greyseal_access_to_both_split_cod_stocks(tmp_path):
    """REGRESSION: BG_ACCESS's pre-split `cod` key silently zeroed BOTH cod stocks on Java arms.

    Python gave GreySeal 1.0 on cod (no column at all in the matrix); Java gave 0.0. The two engines
    were not simulating the same food web, on the species at the centre of the C3 investigation.
    """
    header, rows = _stage_production_matrix(tmp_path)
    gs = header.index("GreySeal")
    for prey in ("cod_west", "cod_east"):
        assert prey in rows, f"{prey} must be a prey row of the production matrix"
        assert float(rows[prey][gs]) > 0.0, (
            f"GreySeal has 0.0 access to {prey} after staging — the authored intent is 0.3. "
            "A prey row was renamed and BG_ACCESS was not updated."
        )


def test_staged_matrix_does_not_duplicate_an_existing_predator_column(tmp_path):
    """Cormorant already has a column in data/baltic; staging must not append a second one."""
    header, _rows = _stage_production_matrix(tmp_path)
    for name in BG_ACCESS:
        assert header.count(name) == 1, (
            f"{name} appears {header.count(name)}x in the staged header — Java would have to pick "
            "between duplicate columns for one predator."
        )


def test_staged_matrix_gives_every_background_predator_an_apex_prey_row(tmp_path):
    """A predator with no PREY row resolves to -1, which the kernel reads as FULL accessibility."""
    _header, rows = _stage_production_matrix(tmp_path)
    for name in BG_ACCESS:
        assert name in rows, f"{name} has no prey row; it would be fully edible via the -1 default"


def test_augment_accessibility_raises_when_no_authored_prey_key_matches(tmp_path):
    """Total miss is an error, not a silent 0.0 — the whole point of the regression."""
    src = tmp_path / "predation-accessibility.csv"
    src.write_text("v Prey / Predator >;cod_west\ncod_west;0.05\n")
    with pytest.raises(ValueError, match="match any prey row"):
        augment_accessibility(src, {"GreySeal": {"cod": 0.3}})  # stale pre-split key only


def test_augment_accessibility_warns_on_partially_unmatched_prey_keys(tmp_path):
    """Partial miss warns rather than raising: BG_ACCESS carries split AND unsplit cod names."""
    src = tmp_path / "predation-accessibility.csv"
    src.write_text("v Prey / Predator >;cod_west\ncod_west;0.05\nherring;0.1\n")
    with pytest.warns(UserWarning, match="match no prey row"):
        augment_accessibility(src, {"GreySeal": {"herring": 0.4, "cod": 0.3}})


def _background_indices(cfg: dict) -> set[str]:
    return {
        k[len("species.type.sp") :]
        for k, v in cfg.items()
        if k.startswith("species.type.sp") and str(v).strip().lower() == "background"
    }


def _condition_factors(master: Path) -> dict[str, list[str]]:
    """sp index -> every condition-factor value the master defines for it, in file order."""
    out: dict[str, list[str]] = {}
    prefix = "species.length2weight.condition.factor.sp"
    for line in master.read_text().splitlines():
        key, sep, value = line.partition(";")
        if sep and key.strip().startswith(prefix):
            out.setdefault(key.strip()[len(prefix) :], []).append(value.strip())
    return out


def test_java_staging_rescales_only_background_weights_to_tonnes(tmp_path):
    """REGRESSION: Java 4.4.x keeps background-school weights in GRAMS, so every seal and cormorant
    school was emptied the first time any mortality cause touched it, and its predation with it.

    Staged through `stage_config_for_java`, the path every Java run takes (reconcile included).
    Each background factor must appear ONCE — Java reads only the first definition of a key — at
    1e-6 of its plain-write value. The control arm is a plain `write_temp_config` of the same
    config: every focal factor must match it byte for byte, which is what catches a rewrite that
    reaches past the background species.
    """
    from ui.pages.run import stage_config_for_java, write_temp_config

    src = Path("data/baltic")
    raw = dict(OsmoseConfigReader().read(str(src / "baltic_all-parameters.csv")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the unused pre-split `cod` key warns by design
        master, _ = stage_config_for_java(raw, tmp_path / "stage", src, target_version="4.4.1")
    plain = write_temp_config(raw, tmp_path / "plain", source_dir=src, target_version="4.4.1")

    staged, control = _condition_factors(master), _condition_factors(plain)
    background = _background_indices(raw)
    assert len(background) == 2 and background <= set(staged)  # GreySeal + Cormorant
    for idx, values in staged.items():
        if idx in background:
            assert len(values) == 1, f"sp{idx} is defined {len(values)}x; Java reads only the first"
            assert float(values[0]) == pytest.approx(float(control[idx][0]) * 1e-6, rel=1e-9)
        else:
            assert values == control[idx], f"focal sp{idx} condition factor was rewritten"


def test_rescale_raises_when_a_background_species_has_no_condition_factor():
    """Java's BackgroundSpecies cannot run without the factor; say which species lacks it."""
    with pytest.raises(ValueError, match=r"background species \['16'\]"):
        _rescale_background_weights(
            "species.length2weight.condition.factor.sp15 ; 0.04\n", ["15", "16"]
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not _JAR,
    reason="opt-in: set OSMOSE_JAR to a 4.4.x jar (a 2-yr Java run). Run it when changing the Java "
    "background staging or switching jars: only a real run shows what the jar does with it.",
)
def test_java_holds_background_biomass_at_its_forcing(tmp_path):
    """END-TO-END: on the jar itself, seals and cormorants stay at their forcing every step.

    Before the weight rescale, Java's end-of-step biomass for both fell to zero within a year
    while the start-of-step biomass still showed the forcing. Nothing eats them (their prey rows
    are zero), so the per-step output must equal forcing x multiplier exactly.
    """
    from osmose.config.aliases import _numeric_version, target_version_for_jar
    from osmose.demo import osmose_demo
    from osmose.java_config_reconcile import sanitize_java_name
    from ui.pages.run import stage_config_for_java

    version = target_version_for_jar(_JAR)
    if _numeric_version(version) < _numeric_version("4.4.0"):
        pytest.skip(f"OSMOSE_JAR is a {version} jar; background staging targets 4.4.x")
    res = osmose_demo("baltic", tmp_path / "demo")
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the unused pre-split `cod` key warns by design
        master, overrides = stage_config_for_java(
            cfg, tmp_path / "stage", res["config_file"].parent, target_version=version
        )
    out = tmp_path / "out"
    cmd = [
        "java",
        "-Xmx2g",
        "-jar",
        _JAR,
        str(master),
        f"-Poutput.dir.path={out}",
        "-Psimulation.time.nyear=2",
        "-Poutput.start.year=0",
        "-Poutput.recordfrequency.ndt=1",
        *[f"-P{k}={v}" for k, v in overrides.items()],
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stdout[-3000:]

    rows = [
        ln
        for ln in next(out.rglob("*biomass_Simu*.csv")).read_text().splitlines()
        if not ln.lstrip().startswith('"Mean')
    ]
    header = [h.strip().strip('"') for h in next(csv.reader([rows[0]]))]
    data = np.array([[float(c) for c in next(csv.reader([ln]))] for ln in rows[1:]])
    nc = min(master.parent.glob("*predator*biomass*.nc"))
    for idx in sorted(_background_indices(cfg)):
        name = cfg[f"species.name.sp{idx}"]
        forcing = inline_biomass_series(nc, name)
        multiplier = float(cfg.get(f"species.biomass.multiplier.sp{idx}", "1"))
        expected = [forcing[step % len(forcing)] * multiplier for step in range(len(data))]
        got = data[:, header.index(sanitize_java_name(name))]
        assert np.allclose(got, expected, rtol=1e-4), (
            f"{name}: Java's per-step biomass {got[:6]}... departs from its forcing "
            f"{expected[:6]}... — background schools are being emptied mid-step again"
        )
