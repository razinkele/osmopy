from osmose.config.aliases import to_target_keys

_RK = "mortality.additional.larva.rate.sp0"


def test_larva_rate_scaled_for_440_target_not_433():
    # to_target_keys receives a CANONICAL (per-cohort) rate from the reader (which is its inverse)
    # and ×ndt for a 4.4.x target -> native rate/year for the jar; leaves it per-cohort for 4.3.x.
    # Reader(÷ndt) -> writer(×ndt) round-trips a native config; there is NO source-awareness.
    cfg = {"osmose.version": "4.4.0", "simulation.time.ndtperyear": "24", _RK: "15"}
    assert float(to_target_keys(dict(cfg), "4.4.1")[_RK]) == 360.0  # 15 × 24 = rate/year
    assert float(to_target_keys(dict(cfg), "4.3.3")[_RK]) == 15.0  # 4.3.3 keeps per-cohort


def test_background_species_keys_emitted_for_440():
    cfg = {
        "osmose.version": "4.3.3",
        "simulation.time.ndtperyear": "24",
        "species.type.sp14": "background",
        "species.name.sp14": "GreySeal",
        "species.type.sp0": "focal",
        "species.name.sp0": "cod",
    }
    out = to_target_keys(dict(cfg), target_version="4.4.1")
    assert out["species.multiplier.sp14"] == "1"  # scalar, load-critical
    assert out["species.beta.sp14"] == "1"
    # NOT the resource NetCDF keys, and NOT for focal species
    assert "species.biomass.mode.sp14" not in out
    assert "species.multiplier.sp0" not in out
    # author-provided value preserved
    cfg2 = dict(cfg)
    cfg2["species.multiplier.sp14"] = "2.5"
    assert to_target_keys(cfg2, target_version="4.4.1")["species.multiplier.sp14"] == "2.5"


def test_background_multiplier_carried_to_the_java_key():
    """The staged inline background biomass is the raw NetCDF series, so Java's multiplier must
    carry the Python engine's species.biomass.multiplier; defaulting it to 1 halved Baltic's
    Cormorant on every Java arm. An explicit species.multiplier still wins."""
    cfg = {
        "osmose.version": "4.3.3",
        "species.type.sp16": "background",
        "species.name.sp16": "Cormorant",
        "species.biomass.multiplier.sp16": "2.0",
    }
    assert to_target_keys(dict(cfg), target_version="4.4.1")["species.multiplier.sp16"] == "2.0"
    cfg["species.multiplier.sp16"] = "1.5"
    assert to_target_keys(dict(cfg), target_version="4.4.1")["species.multiplier.sp16"] == "1.5"


def test_production_baltic_background_multipliers_reach_java():
    """Against the real config, not a fixture: every background species' Java multiplier equals
    the one the Python engine uses."""
    from pathlib import Path

    from osmose.config.reader import OsmoseConfigReader

    root = Path(__file__).resolve().parents[1]
    cfg = dict(OsmoseConfigReader().read(str(root / "data/baltic/baltic_all-parameters.csv")))
    out = to_target_keys(dict(cfg), target_version="4.4.1")
    bkg = [
        k.rsplit("sp", 1)[1]
        for k, v in cfg.items()
        if k.startswith("species.type.sp") and v == "background"
    ]
    assert bkg, "no background species in the Baltic config"
    for i in bkg:
        expected = float(cfg.get(f"species.biomass.multiplier.sp{i}", "1"))
        assert float(out[f"species.multiplier.sp{i}"]) == expected, cfg[f"species.name.sp{i}"]
    assert any(float(cfg.get(f"species.biomass.multiplier.sp{i}", "1")) != 1.0 for i in bkg)


def test_background_keys_not_emitted_on_433_path():
    cfg = {
        "osmose.version": "4.3.3",
        "species.type.sp14": "background",
        "species.name.sp14": "GreySeal",
    }
    out = to_target_keys(dict(cfg), target_version="4.3.3")
    assert "species.multiplier.sp14" not in out


def test_target_version_for_jar():
    from osmose.config.aliases import DEFAULT_TARGET_VERSION, target_version_for_jar

    assert DEFAULT_TARGET_VERSION == "4.4.1"
    assert target_version_for_jar("osmose-java/osmose-4.4.1-jar-with-dependencies.jar") == "4.4.1"
    assert target_version_for_jar("osmose-java/osmose_4.3.3-jar-with-dependencies.jar") == "4.3.3"
    assert target_version_for_jar("weird.jar") == DEFAULT_TARGET_VERSION  # unparseable -> default
