import numpy as np
import pytest
from pathlib import Path

from osmose.engine.config import _load_thermal_gate

FIX = Path(__file__).parent / "data" / "percid_thermal_ok.csv"


def _cfg(**over):
    base = {
        "reproduction.thermal.gate.enabled": "true",
        "reproduction.thermal.gate.series.file": str(FIX),
        "reproduction.thermal.gate.species.enabled.sp4": "true",
        "reproduction.thermal.gate.species.enabled.sp5": "true",
        "reproduction.thermal.gate.mode": "thermal_cap",
    }
    base.update(over)
    return base


def test_off_returns_none():
    f, e, o = _load_thermal_gate({"reproduction.thermal.gate.enabled": "false"}, 6, 24, 4)
    assert f is None and e is None and o == 0


def test_thermal_cap_shapes_and_enabled_mask():
    f, e, o = _load_thermal_gate(_cfg(), n_species=6, n_dt_per_year=24, n_year=4)
    assert f.shape == (4, 6)
    assert list(np.where(e)[0]) == [4, 5]
    assert np.allclose(f[:, 0], 1.0)  # disabled species column stays 1.0
    assert f[3, 4] == pytest.approx(1.0, abs=0.05)  # warm final year -> ~1
    assert f[0, 4] < f[3, 4]  # cold first year < warm final year


def test_mean_preserving_unit_mean():
    f, e, o = _load_thermal_gate(
        _cfg(**{"reproduction.thermal.gate.mode": "mean_preserving"}), 6, 24, 4
    )
    assert np.mean(f[:, 4]) == pytest.approx(1.0)


def test_missing_file_raises():
    with pytest.raises(ValueError, match="series.file is empty"):
        _load_thermal_gate(_cfg(**{"reproduction.thermal.gate.series.file": ""}), 6, 24, 4)


def test_no_species_enabled_raises():
    cfg = _cfg(
        **{
            "reproduction.thermal.gate.species.enabled.sp4": "false",
            "reproduction.thermal.gate.species.enabled.sp5": "false",
        }
    )
    with pytest.raises(ValueError, match="no species enabled"):
        _load_thermal_gate(cfg, 6, 24, 4)


def test_missing_species_column_raises():
    cfg = _cfg(**{"reproduction.thermal.gate.species.enabled.sp3": "true"})
    with pytest.raises(ValueError, match="temp_sp3"):
        _load_thermal_gate(cfg, 6, 24, 4)


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        _load_thermal_gate(_cfg(**{"reproduction.thermal.gate.mode": "bogus"}), 6, 24, 4)


def test_bad_floor_raises():
    with pytest.raises(ValueError, match="floor"):
        _load_thermal_gate(_cfg(**{"reproduction.thermal.gate.floor": "1.5"}), 6, 24, 4)


def test_mean_preserving_rejects_floor(tmp_path):
    # review finding 4: floor>0 under mean_preserving must fail-fast in the loader.
    with pytest.raises(ValueError, match="floor"):
        _load_thermal_gate(
            _cfg(
                **{
                    "reproduction.thermal.gate.mode": "mean_preserving",
                    "reproduction.thermal.gate.floor": "0.1",
                }
            ),
            6,
            24,
            4,
        )


def test_thermal_keys_are_recognized_by_config_validation():
    """Once registered in the schema (Step 7), config_validation reports zero
    unknown keys for the thermal keys. Real API:
    config_validation.validate(cfg, mode) -> list[UnknownKey]."""
    from osmose.engine import config_validation as cv

    keys = {
        "reproduction.thermal.gate.enabled": "true",
        "reproduction.thermal.gate.series.file": str(FIX),
        "reproduction.thermal.gate.mode": "thermal_cap",
        "reproduction.thermal.gate.floor": "0.0",
        "reproduction.thermal.gate.start.year": "2000",
        "reproduction.thermal.gate.species.enabled.sp4": "true",
        "reproduction.thermal.gate.species.enabled.sp5": "true",
        "reproduction.thermal.gate.t50.sp4": "18.5",
        "reproduction.thermal.gate.slope.sp4": "1.5",
        "reproduction.thermal.gate.tref.sp4": "20.0",
    }
    assert cv.validate(keys, mode="error") == []
