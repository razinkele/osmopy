import pytest

from osmose.config.aliases import to_target_keys

_RK = "mortality.additional.larva.rate.sp0"


@pytest.mark.parametrize(
    "src,tgt,factor",
    [
        ("4.3.3", "4.4.1", 24.0),  # write native: x ndt
        ("4.4.1", "4.4.1", 1.0),  # native->native: NO double-scale
        ("4.4.1", "4.3.3", 1 / 24.0),  # native->legacy jar: / ndt
        ("4.3.3", "4.3.3", 1.0),  # legacy->legacy: unchanged
    ],
)
def test_larva_rate_source_aware(src, tgt, factor):
    cfg = {"osmose.version": src, "simulation.time.ndtperyear": "24", _RK: "2.4"}
    out = to_target_keys(dict(cfg), target_version=tgt)
    assert float(out[_RK]) == pytest.approx(2.4 * factor, rel=1e-12)


def test_native_source_keeps_emit_idempotent_no_double_scale():
    # a native-4.4.0 config with a background species, staged for 4.4.1, must not double-scale
    cfg = {
        "osmose.version": "4.4.1",
        "simulation.time.ndtperyear": "24",
        _RK: "2.4",
        "species.type.sp9": "background",
        "species.name.sp9": "Seal",
    }
    out = to_target_keys(dict(cfg), target_version="4.4.1")
    assert float(out[_RK]) == pytest.approx(2.4, rel=1e-12)  # unchanged
    assert out["species.multiplier.sp9"] == "1"  # emit still idempotent


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
