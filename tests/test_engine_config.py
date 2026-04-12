"""Tests for EngineConfig — typed parameter extraction from flat config dicts."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig


@pytest.fixture
def minimal_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "10",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "20",
        "simulation.nschool.sp1": "15",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Sardine",
        "species.linf.sp0": "15.0",
        "species.linf.sp1": "25.0",
        "species.k.sp0": "0.4",
        "species.k.sp1": "0.3",
        "species.t0.sp0": "-0.1",
        "species.t0.sp1": "-0.2",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.15",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.008",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.length2weight.allometric.power.sp1": "3.1",
        "species.lifespan.sp0": "3",
        "species.lifespan.sp1": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.0",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
    }


class TestEngineConfig:
    def test_from_dict_basic(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.n_species == 2
        assert cfg.n_dt_per_year == 24
        assert cfg.n_year == 10
        assert cfg.n_steps == 240

    def test_species_names(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.species_names == ["Anchovy", "Sardine"]

    def test_growth_params_arrays(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.linf[0] == pytest.approx(15.0)
        assert cfg.linf[1] == pytest.approx(25.0)
        assert cfg.k[0] == pytest.approx(0.4)
        assert cfg.t0[1] == pytest.approx(-0.2)

    def test_lifespan_in_dt(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.lifespan_dt[0] == 3 * 24
        assert cfg.lifespan_dt[1] == 5 * 24

    def test_mortality_subdt(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.mortality_subdt == 10

    def test_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            EngineConfig.from_dict({})

    def test_delta_lmax_factor(self, minimal_config):
        minimal_config["species.delta.lmax.factor.sp0"] = "2.0"
        minimal_config["species.delta.lmax.factor.sp1"] = "1.8"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.delta_lmax_factor[0] == pytest.approx(2.0)
        assert cfg.delta_lmax_factor[1] == pytest.approx(1.8)

    def test_delta_lmax_factor_default(self, minimal_config):
        """delta_lmax_factor defaults to 2.0 when not specified."""
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.delta_lmax_factor[0] == pytest.approx(2.0)
        assert cfg.delta_lmax_factor[1] == pytest.approx(2.0)

    def test_additional_mortality_rate(self, minimal_config):
        minimal_config["mortality.additional.rate.sp0"] = "0.2"
        minimal_config["mortality.additional.rate.sp1"] = "0.15"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.additional_mortality_rate[0] == pytest.approx(0.2)
        assert cfg.additional_mortality_rate[1] == pytest.approx(0.15)

    def test_sex_ratio_default(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.sex_ratio[0] == pytest.approx(0.5)

    def test_relative_fecundity(self, minimal_config):
        minimal_config["species.relativefecundity.sp0"] = "800"
        minimal_config["species.relativefecundity.sp1"] = "200"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.relative_fecundity[0] == pytest.approx(800.0)
        assert cfg.relative_fecundity[1] == pytest.approx(200.0)

    def test_maturity_size(self, minimal_config):
        minimal_config["species.maturity.size.sp0"] = "12.0"
        minimal_config["species.maturity.size.sp1"] = "40.0"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.maturity_size[0] == pytest.approx(12.0)
        assert cfg.maturity_size[1] == pytest.approx(40.0)

    def test_movement_method(self, minimal_config):
        minimal_config["movement.distribution.method.sp0"] = "random"
        minimal_config["movement.distribution.method.sp1"] = "maps"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.movement_method == ["random", "maps"]

    def test_random_walk_range(self, minimal_config):
        minimal_config["movement.randomwalk.range.sp0"] = "1"
        minimal_config["movement.randomwalk.range.sp1"] = "2"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.random_walk_range[0] == 1
        assert cfg.random_walk_range[1] == 2

    def test_size_ratio_params(self, minimal_config):
        minimal_config["predation.predprey.sizeratio.min.sp0"] = "3.5"
        minimal_config["predation.predprey.sizeratio.min.sp1"] = "2.0"
        minimal_config["predation.predprey.sizeratio.max.sp0"] = "1.0"
        minimal_config["predation.predprey.sizeratio.max.sp1"] = "0.5"
        cfg = EngineConfig.from_dict(minimal_config)
        # Java convention: min > max gets swapped → min becomes the smaller value
        assert cfg.size_ratio_min[0, 0] == pytest.approx(1.0)
        assert cfg.size_ratio_min[1, 0] == pytest.approx(0.5)
        assert cfg.size_ratio_max[0, 0] == pytest.approx(3.5)
        assert cfg.size_ratio_max[1, 0] == pytest.approx(2.0)

    def test_size_ratio_defaults(self, minimal_config):
        """size_ratio_min defaults to 1.0, size_ratio_max defaults to 3.5."""
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.size_ratio_min[0, 0] == pytest.approx(1.0)
        assert cfg.size_ratio_max[0, 0] == pytest.approx(3.5)

    def test_out_mortality_rate(self, minimal_config):
        minimal_config["mortality.out.rate.sp0"] = "0.1"
        minimal_config["mortality.out.rate.sp1"] = "0.05"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.out_mortality_rate[0] == pytest.approx(0.1)

    def test_starvation_rate_max(self, minimal_config):
        minimal_config["mortality.starvation.rate.max.sp0"] = "3.0"
        minimal_config["mortality.starvation.rate.max.sp1"] = "2.0"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.starvation_rate_max[0] == pytest.approx(3.0)

    def test_fishing_enabled(self, minimal_config):
        minimal_config["simulation.fishing.mortality.enabled"] = "false"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.fishing_enabled is False

    def test_engine_config_from_empty_dict_raises(self):
        """EngineConfig.from_dict({}) must raise (missing required keys)."""
        with pytest.raises((KeyError, ValueError)):
            EngineConfig.from_dict({})

    def test_engine_config_non_numeric_ndtperyear_raises(self, minimal_config):
        """Non-numeric ndtperyear must raise ValueError."""
        minimal_config["simulation.time.ndtperyear"] = "not_a_number"
        with pytest.raises((ValueError, KeyError)):
            EngineConfig.from_dict(minimal_config)


def test_spawning_season_normalization_per_year(tmp_path):
    """Normalization must divide each per-year chunk independently, not the total sum."""
    from osmose.engine.config import _load_spawning_seasons

    csv_path = tmp_path / "season_sp0.csv"
    csv_path.write_text("step;value\n0;1\n1;2\n2;3\n3;4\n4;5\n5;6\n6;7\n7;8\n")
    cfg = {
        "_osmose.config.dir": str(tmp_path),
        "reproduction.season.file.sp0": "season_sp0.csv",
        "reproduction.normalisation.enabled": "true",
    }
    seasons = _load_spawning_seasons(cfg, n_species=1, n_dt_per_year=4)
    assert seasons is not None
    # Year 1 chunk [1,2,3,4] must sum to 1.0 after per-year normalization
    year1_sum = seasons[0, 0:4].sum()
    np.testing.assert_allclose(year1_sum, 1.0, atol=1e-10)
    # Year 2 chunk [5,6,7,8] must also sum to 1.0
    year2_sum = seasons[0, 4:8].sum()
    np.testing.assert_allclose(year2_sum, 1.0, atol=1e-10)


def test_spawning_season_normalization_partial_year(tmp_path, caplog):
    """A non-whole-year file normalizes every chunk including the partial tail, with a warning."""
    import logging

    from osmose.engine.config import _load_spawning_seasons

    # 10-row CSV with n_dt_per_year=4: two full years (rows 0-3, 4-7) plus a 2-row tail (8-9)
    csv_path = tmp_path / "season_sp0.csv"
    csv_path.write_text(
        "step;value\n0;1\n1;2\n2;3\n3;4\n4;5\n5;6\n6;7\n7;8\n8;9\n9;11\n"
    )
    cfg = {
        "_osmose.config.dir": str(tmp_path),
        "reproduction.season.file.sp0": "season_sp0.csv",
        "reproduction.normalisation.enabled": "true",
    }
    with caplog.at_level(logging.WARNING, logger="osmose.engine.config"):
        seasons = _load_spawning_seasons(cfg, n_species=1, n_dt_per_year=4)

    assert seasons is not None
    # Year 1 [1,2,3,4] normalized
    np.testing.assert_allclose(seasons[0, 0:4].sum(), 1.0, atol=1e-10)
    # Year 2 [5,6,7,8] normalized
    np.testing.assert_allclose(seasons[0, 4:8].sum(), 1.0, atol=1e-10)
    # Partial tail [9, 11] normalized so the 2-row chunk sums to 1.0
    np.testing.assert_allclose(seasons[0, 8:10].sum(), 1.0, atol=1e-10)
    # And a warning was emitted
    assert any(
        "not a multiple of n_dt_per_year" in rec.message for rec in caplog.records
    ), "Expected a partial-year warning"


# ---------------------------------------------------------------------------
# Bioen coupling invariant (deep review v3 I-2)
#
# When bioen_enabled=True, from_dict() populates all 18 bioen_* per-species
# arrays with defaults for any missing config keys.  The coupling is therefore
# implicit (no runtime check in __post_init__ is needed because no construction
# path can produce None arrays with bioen_enabled=True).
# ---------------------------------------------------------------------------


class TestBioenCoupling:
    """I-2: bioen_enabled=True guarantees all bioen_* fields non-None (implicit coupling)."""

    _BIOEN_FIELDS = [
        "bioen_beta",
        "bioen_zlayer",
        "bioen_assimilation",
        "bioen_c_m",
        "bioen_eta",
        "bioen_r",
        "bioen_m0",
        "bioen_m1",
        "bioen_e_mobi",
        "bioen_e_d",
        "bioen_tp",
        "bioen_e_maint",
        "bioen_o2_c1",
        "bioen_o2_c2",
        "bioen_i_max",
        "bioen_theta",
        "bioen_c_rate",
        "bioen_k_for",
    ]

    def test_bioen_enabled_populates_all_fields(self, minimal_config):
        """All 18 bioen_* arrays are non-None after from_dict with bioen_enabled=True.

        This documents the implicit coupling: from_dict() applies hard-coded defaults
        for every missing bioen key, so __post_init__ does not need a redundant check.
        """
        cfg_dict = dict(minimal_config)
        cfg_dict["simulation.bioen.enabled"] = "true"
        ec = EngineConfig.from_dict(cfg_dict)
        assert ec.bioen_enabled is True
        for field in self._BIOEN_FIELDS:
            arr = getattr(ec, field)
            assert arr is not None, f"{field} must not be None when bioen_enabled=True"
            assert len(arr) == ec.n_species, (
                f"{field} has length {len(arr)}, expected n_species={ec.n_species}"
            )

    def test_bioen_disabled_leaves_fields_none(self, minimal_config):
        """All 18 bioen_* arrays remain None when bioen_enabled=False (default)."""
        ec = EngineConfig.from_dict(minimal_config)
        assert ec.bioen_enabled is False
        for field in self._BIOEN_FIELDS:
            assert getattr(ec, field) is None, (
                f"{field} should be None when bioen_enabled=False"
            )


# ---------------------------------------------------------------------------
# Regression: file-resolution helpers must raise when a non-empty config key
# points at a missing file, instead of silently disabling the feature.
# Deep review v3 C-3 through C-7 — the "silent feature removal" anti-pattern.
# ---------------------------------------------------------------------------


class TestRequireFileRaisesOnMissing:
    """Non-empty file keys with missing files must raise FileNotFoundError."""

    def test_require_file_helper_raises_for_missing_file(self, tmp_path):
        """_require_file raises FileNotFoundError with the file_key + context."""
        from osmose.engine.config import _require_file

        with pytest.raises(FileNotFoundError, match="doesnt_exist.csv"):
            _require_file("doesnt_exist.csv", str(tmp_path), "test.context")

    def test_require_file_helper_returns_path_for_existing_file(self, tmp_path):
        """_require_file returns the resolved Path for a file that exists."""
        from osmose.engine.config import _require_file

        (tmp_path / "real.csv").write_text("x")
        path = _require_file("real.csv", str(tmp_path), "test.context")
        assert path.exists()
        assert path.name == "real.csv"

    def test_require_file_error_message_includes_context_and_config_dir(self, tmp_path):
        """Error message must help the user find the bad config key."""
        from osmose.engine.config import _require_file

        with pytest.raises(FileNotFoundError) as exc_info:
            _require_file("typo.csv", str(tmp_path), "mortality.fishing.rate.byyear.file.sp3")
        msg = str(exc_info.value)
        assert "typo.csv" in msg
        assert "mortality.fishing.rate.byyear.file.sp3" in msg
        assert str(tmp_path) in msg

    def test_fisheries_catchability_file_missing_raises(self, tmp_path):
        """C-3: simulation.nfisheries>0 with missing catchability file raises, not silently zeros."""
        from osmose.engine.config import _parse_fisheries

        # Minimal fishing v4 config with a catchability file that doesn't exist
        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "simulation.nfisheries": "1",
            "fisheries.catchability.file": "typo_catch.csv",  # doesn't exist
        }
        with pytest.raises(FileNotFoundError, match="typo_catch.csv"):
            _parse_fisheries(cfg, ["Anchovy"], n_species=1)

    def test_fisheries_catchability_empty_key_with_n_fisheries_raises(self, tmp_path):
        """C-3: simulation.nfisheries>0 but catchability.file unset raises ValueError."""
        from osmose.engine.config import _parse_fisheries

        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "simulation.nfisheries": "1",
            # fisheries.catchability.file intentionally missing
        }
        with pytest.raises(ValueError, match="fisheries.catchability.file is not set"):
            _parse_fisheries(cfg, ["Anchovy"], n_species=1)

    def test_fishing_rate_by_year_missing_file_raises(self, tmp_path):
        """C-7: non-empty mortality.fishing.rate.byyear.file.sp{i} with missing file raises."""
        from osmose.engine.config import _load_fishing_rate_by_year

        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "mortality.fishing.rate.byyear.file.sp0": "typo.csv",
        }
        with pytest.raises(FileNotFoundError, match="typo.csv"):
            _load_fishing_rate_by_year(cfg, n_species=1)

    def test_fishing_rate_by_year_empty_key_does_not_raise(self, tmp_path):
        """C-7: empty mortality.fishing.rate.byyear.file.sp{i} still means 'not configured'."""
        from osmose.engine.config import _load_fishing_rate_by_year

        cfg = {"_osmose.config.dir": str(tmp_path)}
        # No keys set → None, no raise
        result = _load_fishing_rate_by_year(cfg, n_species=2)
        assert result is None

    def test_additional_mortality_by_dt_missing_file_raises(self, tmp_path):
        """C-5: non-empty mortality.additional.rate.bytdt.file.sp{i} with missing file raises."""
        from osmose.engine.config import _load_additional_mortality_by_dt

        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "mortality.additional.rate.bytdt.file.sp0": "ghost.csv",
        }
        with pytest.raises(FileNotFoundError, match="ghost.csv"):
            _load_additional_mortality_by_dt(cfg, n_species=1)

    def test_additional_mortality_spatial_missing_file_raises(self, tmp_path):
        """C-6: non-empty mortality.additional.spatial.distrib.file.sp{i} with missing file raises."""
        from osmose.engine.config import _load_additional_mortality_spatial

        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "mortality.additional.spatial.distrib.file.sp0": "missing.csv",
        }
        with pytest.raises(FileNotFoundError, match="missing.csv"):
            _load_additional_mortality_spatial(cfg, n_species=1)

    def test_per_species_fishing_spatial_map_missing_raises(self, tmp_path):
        """C-4: non-empty mortality.fishing.spatial.distrib.file.sp{i} with missing file raises.

        This one has to go through EngineConfig.from_dict because the per-species
        spatial map is processed inline in from_dict, not in a standalone helper.
        """
        # Minimal working config with the problematic key
        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "5",
            "species.name.sp0": "Anchovy",
            "species.linf.sp0": "20.0",
            "species.k.sp0": "0.3",
            "species.t0.sp0": "-0.1",
            "species.egg.size.sp0": "0.1",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.lifespan.sp0": "3",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
            # The key under test: points at a non-existent file
            "mortality.fishing.spatial.distrib.file.sp0": "ghost_map.csv",
        }
        with pytest.raises(FileNotFoundError, match="ghost_map.csv"):
            EngineConfig.from_dict(cfg)

    def test_fisheries_discards_file_missing_raises(self, tmp_path):
        """Task-16: non-empty fisheries.discards.file with missing file raises, not silently zeroes."""
        from osmose.engine.config import _load_discard_rates

        cfg = {"_osmose.config.dir": str(tmp_path), "fisheries.discards.file": "typo.csv"}
        with pytest.raises(FileNotFoundError, match="typo.csv"):
            _load_discard_rates(cfg, ["Anchovy"], n_species=1)

    def test_mpa_file_missing_raises(self, tmp_path):
        """Task-16: non-empty mpa.file.mpa{i} with missing file raises, not silently skips."""
        from osmose.engine.config import _parse_mpa_zones

        cfg = {"_osmose.config.dir": str(tmp_path), "mpa.file.mpa0": "typo.csv"}
        with pytest.raises(FileNotFoundError, match="typo.csv"):
            _parse_mpa_zones(cfg)

    def test_shared_fishing_map_missing_raises(self, tmp_path):
        """Task-16: non-empty fisheries.movement.file.map0 with missing file raises."""
        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "5",
            "species.name.sp0": "Anchovy",
            "species.linf.sp0": "20.0",
            "species.k.sp0": "0.3",
            "species.t0.sp0": "-0.1",
            "species.egg.size.sp0": "0.1",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.lifespan.sp0": "3",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
            "fisheries.movement.file.map0": "ghost_shared_map.csv",
        }
        with pytest.raises(FileNotFoundError, match="ghost_shared_map.csv"):
            EngineConfig.from_dict(cfg)

    def test_predation_accessibility_file_missing_raises(self, tmp_path):
        """Task-16: non-empty predation.accessibility.file with missing file raises."""
        from osmose.engine.config import _load_accessibility

        cfg = {"_osmose.config.dir": str(tmp_path), "predation.accessibility.file": "typo_access.csv"}
        with pytest.raises(FileNotFoundError, match="typo_access.csv"):
            _load_accessibility(cfg, n_species=2)


class TestMPAZoneValidation:
    """Deep review v3 I-8: MPAZone must validate grid shape and value range."""

    def _base_kwargs(self):
        import numpy as np
        return {
            "grid": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            "start_year": 0,
            "end_year": 10,
            "percentage": 0.5,
        }

    def test_valid_mpa_zone(self):
        from osmose.engine.config import MPAZone
        MPAZone(**self._base_kwargs())

    def test_1d_grid_rejected(self):
        import numpy as np
        import pytest
        from osmose.engine.config import MPAZone
        kwargs = self._base_kwargs()
        kwargs["grid"] = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="grid must be 2D"):
            MPAZone(**kwargs)

    def test_3d_grid_rejected(self):
        import numpy as np
        import pytest
        from osmose.engine.config import MPAZone
        kwargs = self._base_kwargs()
        kwargs["grid"] = np.zeros((2, 2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="grid must be 2D"):
            MPAZone(**kwargs)

    def test_continuous_grid_rejected(self):
        import numpy as np
        import pytest
        from osmose.engine.config import MPAZone
        kwargs = self._base_kwargs()
        kwargs["grid"] = np.array([[0.0, 0.5], [1.0, 0.2]])
        with pytest.raises(ValueError, match="grid values must be 0 or 1"):
            MPAZone(**kwargs)

    def test_negative_start_year_rejected(self):
        import pytest
        from osmose.engine.config import MPAZone
        kwargs = self._base_kwargs()
        kwargs["start_year"] = -1
        with pytest.raises(ValueError, match="start_year must be non-negative"):
            MPAZone(**kwargs)
