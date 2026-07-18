"""#121 Layer B: the fixed bundled configs actually produce the output they request."""

from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.config_validation import validate

ROOT = Path(__file__).parent.parent


def _cfg(rel: str) -> EngineConfig:
    return EngineConfig.from_dict(OsmoseConfigReader().read(ROOT / rel))


def test_examples_requests_byage_bysize_meansize_tl():
    """data/examples is the new-user starting point; it must not silently drop output."""
    ec = _cfg("data/examples/osm_all-parameters.csv")
    assert ec.output_biomass_byage and ec.output_abundance_byage
    assert ec.output_biomass_bysize and ec.output_abundance_bysize
    assert ec.output_mean_size
    assert ec.output_meantl


def test_eec_requests_byage_bysize_meansize_tl():
    # data/eec's top-level is osm_all-parameters.csv (NOT eec_all-parameters.csv — that name
    # exists only under the unrelated data/eec_full/).
    ec = _cfg("data/eec/osm_all-parameters.csv")
    assert ec.output_biomass_byage and ec.output_abundance_byage
    assert ec.output_biomass_bysize and ec.output_abundance_bysize
    assert ec.output_mean_size
    assert ec.output_meantl


def test_no_dead_output_keys_remain_in_live_configs():
    """The 5 removed invented keys must be gone from ALL THREE live configs (not 433_orig)."""
    dead = (
        "output.byage.enabled",
        "output.bysize.enabled",
        "output.meansize.enabled",
        "output.trophiclevel.enabled",
        "output.frequency.ndtperyear",
    )
    for rel in (
        "data/examples/osm_param-output.csv",
        "data/eec/osm_param-output.csv",
        "data/minimal/osm_param-output.csv",
    ):
        text = (ROOT / rel).read_text()
        for k in dead:
            assert k not in text, f"{k} still in {rel}"


def test_examples_actually_produces_meantl_output(tmp_path):
    """Spec requires proving OUTPUT, not just the flag (#121's whole thesis: run it).

    A True flag under-proves — the CSV writers gate on flag AND data presence. Run a short sim
    to disk and assert the mean-TL CSV is materialized and non-empty. `mean_trophic_level()` is
    the real OsmoseResults accessor (results.py:451) — NOT `meantl()`, which does not exist.

    Note: `PythonEngine.run()` calls `write_outputs()` without a `prefix` kwarg, so the on-disk
    files use the default prefix "osm" regardless of the config's `output.file.prefix` — read
    back with `OsmoseResults(tmp_path)` (default prefix="osm"), not `OsmoseResults.from_outputs`
    (that classmethod is the in-memory constructor taking `(outputs, engine_config, grid)`, not
    a directory path).
    """
    from osmose.engine import PythonEngine
    from osmose.results import OsmoseResults

    cfg = OsmoseConfigReader().read(ROOT / "data/examples/osm_all-parameters.csv")
    cfg["simulation.time.nyear"] = "1"  # keep it fast
    PythonEngine().run(config=cfg, output_dir=tmp_path, seed=0)
    results = OsmoseResults(tmp_path)
    tl = results.mean_trophic_level()
    assert not tl.empty, "meanTL output empty — output.tl.enabled not honored end-to-end"


def test_dead_keys_now_flagged_unknown_by_strict_validation():
    """After de-allowlisting, the 5 invented keys are reported unknown (they should be)."""
    dead = (
        "output.byage.enabled",
        "output.bysize.enabled",
        "output.meansize.enabled",
        "output.trophiclevel.enabled",
        "output.frequency.ndtperyear",
    )
    unknown = {u.key for u in validate({k: "true" for k in dead}, "warn")}
    assert set(dead) <= unknown, f"still allowlisted: {set(dead) - unknown}"


def test_working_replacement_keys_are_recognized():
    """The keys the engine actually reads must NOT be flagged unknown."""
    working = (
        "output.tl.enabled",
        "output.size.enabled",
        "output.recordfrequency.ndt",
        "output.biomass.byage.enabled",
        "output.abundance.byage.enabled",
        "output.biomass.bysize.enabled",
        "output.abundance.bysize.enabled",
    )
    unknown = {u.key for u in validate({k: "true" for k in working}, "warn")}
    assert not (set(working) & unknown), f"wrongly flagged: {set(working) & unknown}"


def test_keys_with_real_lineage_stay_allowlisted():
    """Keys with real Java lineage must NOT be flagged (would be a false 'unknown')."""
    keep = (
        "output.diet.stage.threshold.sp0",
        "output.diet.stage.structure",
        "species.conversion2tons.sp0",
        "ltl.conversion2tons.rsc0",
    )
    unknown = {u.key for u in validate({k: "true" for k in keep}, "warn")}
    assert not (set(keep) & unknown), f"wrongly de-allowlisted: {set(keep) & unknown}"
