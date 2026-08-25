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


class TestExponentialResponse:
    """C1 spec decisions 2, 8, 9 — Voss & Quaas exponential response."""

    def _series_csv(self, tmp_path, temps_sp0, first_year=1974):
        rows = ["year,temp_sp0"]
        for i, t in enumerate(temps_sp0):
            rows.append(f"{first_year + i},{t}")
        p = tmp_path / "thermal.csv"
        p.write_text("\n".join(rows) + "\n")
        return p

    def _cfg(self, tmp_path, temps, **over):
        cfg = {
            "simulation.nspecies": "1",
            "simulation.time.ndtperyear": "4",
            "simulation.time.nyear": str(len(temps)),
            "_osmose.config.dir": str(tmp_path),
            "reproduction.thermal.gate.enabled": "true",
            "reproduction.thermal.gate.series.file": str(self._series_csv(tmp_path, temps)),
            "reproduction.thermal.gate.species.enabled.sp0": "true",
            "reproduction.thermal.gate.response": "exponential",
            "reproduction.thermal.gate.beta.sp0": "-0.51",
            "reproduction.thermal.gate.tref.sp0": "7.0",
        }
        cfg.update(over)
        return cfg

    def test_factor_is_exactly_one_at_tref(self, tmp_path):
        from osmose.engine.config import _load_thermal_gate

        factor, enabled, offset = _load_thermal_gate(self._cfg(tmp_path, [7.0] * 5), 1, 4, 5)
        assert (factor[:, 0] == 1.0).all()  # exp(0) == 1.0 exactly — bit-identity rests on this

    def test_exponential_scaling(self, tmp_path):
        import numpy as np
        from osmose.engine.config import _load_thermal_gate

        factor, _, _ = _load_thermal_gate(self._cfg(tmp_path, [9.0] * 3), 1, 4, 3)
        assert np.allclose(factor[:, 0], np.exp(-0.51 * 2.0))

    def test_missing_beta_raises(self, tmp_path):
        import pytest
        from osmose.engine.config import _load_thermal_gate

        cfg = self._cfg(tmp_path, [7.0] * 3)
        del cfg["reproduction.thermal.gate.beta.sp0"]
        with pytest.raises(ValueError, match="beta.sp0"):
            _load_thermal_gate(cfg, 1, 4, 3)

    def test_missing_tref_raises_not_defaults(self, tmp_path):
        """The key has a silent 20.0 thermal_cap default the exponential path must refuse."""
        import pytest
        from osmose.engine.config import _load_thermal_gate

        cfg = self._cfg(tmp_path, [7.0] * 3)
        del cfg["reproduction.thermal.gate.tref.sp0"]
        with pytest.raises(ValueError, match="tref.sp0"):
            _load_thermal_gate(cfg, 1, 4, 3)

    def test_mode_matrix(self, tmp_path):
        import pytest
        from osmose.engine.config import _load_thermal_gate

        for bad in ("thermal_cap", "mean_preserving"):
            cfg = self._cfg(tmp_path, [7.0] * 3)
            cfg["reproduction.thermal.gate.mode"] = bad
            with pytest.raises(ValueError, match="raw"):
                _load_thermal_gate(cfg, 1, 4, 3)
        cfg = self._cfg(tmp_path, [7.0] * 3)
        cfg["reproduction.thermal.gate.mode"] = "raw"
        _load_thermal_gate(cfg, 1, 4, 3)  # explicit raw OK
        cfg = self._cfg(tmp_path, [7.0] * 3)
        cfg["reproduction.thermal.gate.response"] = "logistic"
        cfg["reproduction.thermal.gate.mode"] = "raw"
        cfg["reproduction.thermal.gate.t50.sp0"] = "18.5"
        with pytest.raises(ValueError, match="raw"):
            _load_thermal_gate(cfg, 1, 4, 3)

    def test_negative_offset_raises(self, tmp_path):
        import pytest
        from osmose.engine.config import _load_thermal_gate

        cfg = self._cfg(tmp_path, [7.0] * 3)
        cfg["reproduction.thermal.gate.start.year"] = "1960"  # < series first year 1974
        with pytest.raises(ValueError, match="negative|predates"):
            _load_thermal_gate(cfg, 1, 4, 3)

    def test_floor_applies_under_raw(self, tmp_path):
        from osmose.engine.config import _load_thermal_gate

        cfg = self._cfg(tmp_path, [27.0] * 3)  # exp(-0.51*20) ~ 4e-5
        cfg["reproduction.thermal.gate.floor"] = "0.05"
        factor, _, _ = _load_thermal_gate(cfg, 1, 4, 3)
        assert (factor[:, 0] == 0.05).all()

    def test_csv_float_parsing_round_trips_at_tref(self, tmp_path):
        """Harness-probe finding: pandas' default (non-round-trip) float parser
        mis-rounds full-precision decimals by up to 2 ULP, so a constant series at
        T==tref could yield factor != 1.0 exactly even though exp(0.0) == 1.0 is
        exact. This value round-trips under Python's own float parser but NOT
        under pandas' default engine, unlike 7.0 (used elsewhere in this class),
        which round-trips under either — so it actually exercises the bug."""
        from osmose.engine.config import _load_thermal_gate

        t = 9.670314810741907
        cfg = self._cfg(tmp_path, [t] * 3, **{"reproduction.thermal.gate.tref.sp0": str(t)})
        factor, _, _ = _load_thermal_gate(cfg, 1, 4, 3)
        assert (factor[:, 0] == 1.0).all()
