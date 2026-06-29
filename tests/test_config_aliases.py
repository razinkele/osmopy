from osmose.config.aliases import to_target_keys


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
