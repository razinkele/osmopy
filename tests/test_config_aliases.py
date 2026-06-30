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
