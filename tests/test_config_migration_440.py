from osmose.demo import migrate_config


def test_440_clean_renames():
    cfg = {
        "osmose.version": "4.3.3",
        "simulation.bioen.enabled": "true",
        "fisheries.enabled": "false",
        "output.restart.spinup": "5",
        "output.fishery.enabled": "true",
        "output.spatial.fishery.enabled": "true",
        "species.bioen.maturity.eta.sp0": "1.2",
        "species.bioen.maturity.m0.sp3": "0.5",
    }
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["module.multispecies.fisheries.enabled"] == "false"
    assert out["simulation.restart.spinup.nyear"] == "5"
    assert out["output.fisheries.enabled"] == "true"
    assert out["output.spatial.fisheries.enabled"] == "true"
    assert out["species.maturity.eta.sp0"] == "1.2"
    assert out["species.maturity.m0.sp3"] == "0.5"
    assert "simulation.bioen.enabled" not in out
    assert "species.bioen.maturity.eta.sp0" not in out
    assert out["osmose.version"] == "4.4.0"


def test_440_ingestion_merge_skip_if_target_exists():
    cfg = {
        "osmose.version": "4.3.3",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.bioen.sp0": "3.0",
        "predation.ingestion.rate.max.bioen.sp1": "4.0",
    }
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"  # base kept (skip-if-exists)
    assert "predation.ingestion.rate.max.bioen.sp0" not in out  # bioen dropped
    assert out["predation.ingestion.rate.max.sp1"] == "4.0"  # bioen-only -> renamed


def test_440_idempotent_on_new_keys():
    cfg = {"osmose.version": "4.4.0", "module.bioenergetics.enabled": "true"}
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert "simulation.bioen.enabled" not in out
