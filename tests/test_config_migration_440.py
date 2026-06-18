from osmose.config.aliases import canonicalize_config, to_target_keys
from osmose.demo import migrate_config
from osmose.engine.config import EngineConfig


def _min_bioen_cfg(extra: dict) -> dict[str, str]:
    base = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "19.5",
        "species.k.sp0": "0.364",
        "species.t0.sp0": "-0.70",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.06",
        "species.lifespan.sp0": "4",
        "species.vonbertalanffy.threshold.age.sp0": "0",
        "mortality.subdt": "10",
        "predation.efficiency.critical.sp0": "0.57",
    }
    base.update(extra)
    return base


def test_from_dict_accepts_old_module_toggle_key():
    cfg_old = _min_bioen_cfg(
        {"simulation.bioen.enabled": "false", "predation.ingestion.rate.max.sp0": "3.5"}
    )
    config = EngineConfig.from_dict(cfg_old)
    assert config.bioen_enabled is False


def test_from_dict_unified_ingestion_read():
    cfg = _min_bioen_cfg(
        {
            "simulation.bioen.enabled": "false",
            "predation.ingestion.rate.max.bioen.sp0": "4.2",
        }
    )
    config = EngineConfig.from_dict(cfg)
    assert config.ingestion_rate[0] == 4.2


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


def test_canonicalize_reports_deprecated_and_canonicalizes():
    cfg = {"osmose.version": "4.3.3", "simulation.bioen.enabled": "true"}
    out, deprecated = canonicalize_config(cfg)
    assert out["module.bioenergetics.enabled"] == "true"
    assert "simulation.bioen.enabled" in deprecated


def test_canonicalize_missing_version_does_not_corrupt_new_keys():
    # No osmose.version, but config already uses NEW keys -> must stay new.
    cfg = {"module.bioenergetics.enabled": "true", "predation.ingestion.rate.max.sp0": "3.5"}
    out, _ = canonicalize_config(cfg)
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"


def test_canonicalize_snapshot_version_handled():
    cfg = {"osmose.version": "4.4.0-SNAPSHOT", "simulation.bioen.enabled": "true"}
    out, _ = canonicalize_config(cfg)
    assert out["module.bioenergetics.enabled"] == "true"


def test_to_target_keys_reverses_to_4_3_3():
    cfg = {
        "osmose.version": "4.4.0",
        "module.bioenergetics.enabled": "true",
        "simulation.restart.spinup.nyear": "5",
        "species.maturity.eta.sp0": "1.2",
        "predation.ingestion.rate.max.sp0": "3.5",
        # pre-existing GROWTH maturity keys that must NOT be touched by the inverse:
        "species.maturity.size.sp0": "20.0",
        "species.maturity.age.sp0": "2.0",
    }
    out = to_target_keys(cfg, target_version="4.3.3")
    assert out["simulation.bioen.enabled"] == "true"
    assert out["output.restart.spinup"] == "5"
    assert out["species.bioen.maturity.eta.sp0"] == "1.2"
    assert out["species.maturity.size.sp0"] == "20.0"  # untouched
    assert out["species.maturity.age.sp0"] == "2.0"  # untouched
    assert out["osmose.version"] == "4.3.3"


def test_to_target_keys_4_4_0_is_identity_plus_stamp():
    cfg = {"osmose.version": "4.3.3", "module.bioenergetics.enabled": "true"}
    out = to_target_keys(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["osmose.version"] == "4.4.0"


def test_to_target_keys_ingestion_merge_is_lossy_to_legacy_key():
    cfg = {"osmose.version": "4.4.0", "predation.ingestion.rate.max.sp0": "3.5"}
    out = to_target_keys(cfg, target_version="4.3.3")
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"  # stays the legacy key
    assert not any(".bioen." in k for k in out)  # no .bioen fabricated


def test_inverse_is_faithful_inverse_of_renames():
    # DRIFT GUARD: _INVERSE_440 hand-maintained alongside RENAMES_440. Every forward rename
    # — except the lossy merge and the maturity prefix that fans out to leaves — must have a
    # faithful inverse; no inverse may be orphaned.
    from osmose.config.aliases import _INVERSE_440, RENAMES_440

    LOSSY = {"predation.ingestion.rate.max.bioen"}  # merge -> not invertible
    LEAF_FANOUT = {"species.bioen.maturity"}  # prefix -> 4 leaves
    for old, new in RENAMES_440.items():
        if old in LOSSY or old in LEAF_FANOUT:
            continue
        assert _INVERSE_440.get(new) == old, f"{new} has no faithful inverse"
    for leaf in ("eta", "r", "m0", "m1"):
        assert _INVERSE_440[f"species.maturity.{leaf}"] == f"species.bioen.maturity.{leaf}"
    forward_new = set(RENAMES_440.values()) | {
        f"species.maturity.{leaf}" for leaf in ("eta", "r", "m0", "m1")
    }
    for new in _INVERSE_440:
        assert any(new == f or new.startswith(f) for f in forward_new), f"orphan inverse {new}"


def test_config_validation_clean_on_old_keys():
    # An old-key config validates warning-free (old keys canonicalized before the
    # unknown-key check; new keys recognized via the allowlist).
    from osmose.engine.config_validation import validate

    cfg = {
        "osmose.version": "4.3.3",
        "simulation.bioen.enabled": "true",
        "output.restart.enabled": "false",
        "economy.enabled": "false",
    }
    unknowns = validate(cfg, mode="warn")  # list[UnknownKey], each with .key
    flagged = {u.key for u in unknowns}
    assert "simulation.bioen.enabled" not in flagged
    assert "economy.enabled" not in flagged
    assert "output.restart.enabled" not in flagged
