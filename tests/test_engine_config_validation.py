"""Tests for EngineConfig __post_init__ validation."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig


def _minimal_config(n_species: int = 2, n_background: int = 0, **overrides) -> dict:
    """Build a minimal kwargs dict for EngineConfig with valid defaults."""
    n_total = n_species + n_background
    defaults = dict(
        n_species=n_species,
        n_dt_per_year=24,
        n_year=1,
        n_steps=24,
        n_schools=np.array([100] * n_total, dtype=np.int32),
        species_names=[f"sp{i}" for i in range(n_species)],
        n_background=n_background,
        background_file_indices=[],
        all_species_names=[f"sp{i}" for i in range(n_total)],
        linf=np.array([30.0] * n_total),
        k=np.array([0.3] * n_total),
        t0=np.array([-0.5] * n_total),
        egg_size=np.array([0.1] * n_total),
        condition_factor=np.array([0.006] * n_total),
        allometric_power=np.array([3.0] * n_total),
        vb_threshold_age=np.array([0.0] * n_total),
        lifespan_dt=np.array([120] * n_total, dtype=np.int32),
        mortality_subdt=10,
        ingestion_rate=np.array([3.5] * n_total),
        critical_success_rate=np.array([0.57] * n_total),
        delta_lmax_factor=np.array([2.0] * n_total),
        additional_mortality_rate=np.array([0.0] * n_total),
        additional_mortality_by_dt=None,
        additional_mortality_by_dt_by_class=None,
        additional_mortality_spatial=None,
        sex_ratio=np.array([0.5] * n_total),
        relative_fecundity=np.array([500.0] * n_total),
        maturity_size=np.array([15.0] * n_total),
        seeding_biomass=np.array([1000.0] * n_total),
        seeding_max_step=np.array([120] * n_total, dtype=np.int32),
        larva_mortality_rate=np.array([0.0] * n_total),
        larva_mortality_by_dt=None,
        size_ratio_min=np.zeros((n_total, 1)),
        size_ratio_max=np.ones((n_total, 1)),
        feeding_stage_thresholds=[[] for _ in range(n_total)],
        feeding_stage_metric=["size"] * n_total,
        n_feeding_stages=np.array([1] * n_total, dtype=np.int32),
        starvation_rate_max=np.array([3.0] * n_total),
        fishing_enabled=False,
        fishing_rate=np.array([0.0] * n_total),
        fishing_selectivity_l50=np.array([0.0] * n_total),
        fishing_selectivity_a50=np.full(n_total, np.nan),
        fishing_selectivity_type=np.zeros(n_total, dtype=np.int32),
        fishing_selectivity_slope=np.ones(n_total),
        fishing_seasonality=None,
        fishing_rate_by_year=None,
        fishing_rate_by_dt_by_class=None,
        fishing_catches=None,
        fishing_catches_by_year=None,
        fishing_catches_season=None,
        fishing_selectivity_l75=np.array([0.0] * n_total),
        mpa_zones=None,
        fishing_discard_rate=None,
        accessibility_matrix=None,
        stage_accessibility=None,
        dynamic_accessibility_enabled=False,
        dynamic_accessibility_exponent=1.0,
        dynamic_accessibility_floor=0.05,
        spawning_season=None,
        movement_method=["maps"] * n_total,
        random_walk_range=np.array([3] * n_total, dtype=np.int32),
        out_mortality_rate=np.array([0.0] * n_total),
        maturity_age_dt=np.zeros(n_total, dtype=np.int32),
        lmax=np.array([30.0] * n_total),
        fishing_spatial_maps=[None] * n_total,
        egg_weight_override=None,
        output_cutoff_age=None,
        output_record_frequency=1,
        diet_output_enabled=False,
        output_step0_include=False,
        movement_seed_fixed=False,
        mortality_seed_fixed=False,
        java_compat_rng=False,
        random_distribution_ncell=None,
        growth_class=["VB"] * n_total,
        recruitment_type=["none"] * n_total,
        recruitment_ssb_half=np.zeros(n_total),
        raw_config={},
    )
    defaults.update(overrides)
    return defaults


def test_valid_config_passes() -> None:
    """A minimal valid config should construct without error."""
    cfg = _minimal_config()
    ec = EngineConfig(**cfg)
    # Check stored scalar and that per-species arrays have the expected shape,
    # confirming __post_init__ ran to completion without trimming or rejecting data.
    assert ec.n_species == 2
    assert ec.linf.shape == (2,)
    assert ec.k.shape == (2,)
    assert ec.fishing_rate.shape == (2,)


def test_mismatched_array_length_raises() -> None:
    """Per-species arrays with wrong length should raise ValueError."""
    cfg = _minimal_config(n_species=2)
    cfg["linf"] = np.array([30.0, 30.0, 30.0])  # length 3 != n_total=2
    with pytest.raises(ValueError, match="linf.*length 3.*expected 2"):
        EngineConfig(**cfg)


def test_zero_linf_raises() -> None:
    """Zero linf for a focal species should raise ValueError."""
    cfg = _minimal_config(n_species=2)
    cfg["linf"] = np.array([0.0, 30.0])
    with pytest.raises(ValueError, match="linf.*positive"):
        EngineConfig(**cfg)


def test_negative_k_raises() -> None:
    """Negative k should raise ValueError."""
    cfg = _minimal_config(n_species=2)
    cfg["k"] = np.array([-0.3, 0.3])
    with pytest.raises(ValueError, match="k.*positive"):
        EngineConfig(**cfg)


def test_nsteps_consistency_raises() -> None:
    """n_steps != n_dt_per_year * n_year should raise ValueError."""
    cfg = _minimal_config()
    cfg["n_steps"] = 999
    with pytest.raises(ValueError, match="n_steps"):
        EngineConfig(**cfg)


def test_background_species_zero_linf_allowed() -> None:
    """Background species with zero linf should NOT raise (they don't grow)."""
    cfg = _minimal_config(n_species=2, n_background=1)
    cfg["linf"] = np.array([30.0, 30.0, 0.0])
    cfg["k"] = np.array([0.3, 0.3, 0.0])
    ec = EngineConfig(**cfg)
    # Confirm the background species zero linf was accepted as-is (not clamped or rejected)
    # and that focal species linf remains positive.
    assert ec.n_background == 1
    assert ec.linf[2] == 0.0, "background species linf must be stored unchanged"
    assert (ec.linf[:2] > 0).all(), "focal species linf must remain positive"


def test_from_dict_still_works() -> None:
    """Existing from_dict path should still produce valid configs."""
    minimal = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "20",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "15.0",
        "species.k.sp0": "0.4",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
    }
    cfg = EngineConfig.from_dict(minimal)
    assert cfg.n_species == 1
    assert cfg.linf[0] == 15.0


# --- Phase 7.3: unknown-key validation integration tests ----------------------
import logging as _logging  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

import pytest as _pytest  # noqa: E402

from osmose.engine.config import EngineConfig as _EngineConfig  # noqa: E402


def _load_example_config(example_name: str) -> dict:
    """Load an example all-parameters.csv into a dict via OsmoseConfigReader.

    Globs ``*_all-parameters.csv`` rather than enumerating candidates so all
    three shipped filename conventions (``osm_all-parameters.csv``,
    ``<name>_all-parameters.csv``, ``eec_all-parameters.csv`` in eec_full)
    work without special-casing.
    """
    from osmose.config import OsmoseConfigReader

    base = _Path(__file__).resolve().parent.parent
    matches = sorted((base / "data" / example_name).glob("*_all-parameters.csv"))
    if not matches:
        _pytest.skip(f"example config not found: {example_name}")
    cfg = OsmoseConfigReader().read(matches[0])
    # Strip empty-string keys: artefacts from CSV rows like ",," where the
    # separator regex splits to an empty key.  Not a real config key; exclude
    # so the validator only sees genuine keys.
    return {k: v for k, v in cfg.items() if k != ""}


@_pytest.mark.parametrize(
    "example_name",
    ["eec", "baltic", "eec_full"],
)
def test_from_dict_warn_mode_clean_on_example_configs(example_name, caplog):
    """Load each reference example in warn mode; assert zero WARNING records."""
    cfg = _load_example_config(example_name)
    cfg["validation.strict.enabled"] = "warn"
    with caplog.at_level(_logging.WARNING, logger="osmose.config"):
        _EngineConfig.from_dict(cfg)
    warn_records = [r for r in caplog.records if r.levelno >= _logging.WARNING]
    assert warn_records == [], (
        f"{example_name} config has {len(warn_records)} unknown keys -- "
        f"first 5: {[r.getMessage() for r in warn_records[:5]]}"
    )


def test_from_dict_warn_mode_catches_known_typo(caplog):
    """Inject a single-char typo; assert the warning includes the suggestion."""
    cfg = _load_example_config("eec")
    cfg["species.liinf.sp0"] = "30.0"
    cfg["validation.strict.enabled"] = "warn"
    with caplog.at_level(_logging.WARNING, logger="osmose.config"):
        _EngineConfig.from_dict(cfg)
    warn_records = [r for r in caplog.records if r.levelno >= _logging.WARNING]
    assert any("species.liinf.sp0" in r.message for r in warn_records)
    assert any("species.linf.sp{idx}" in r.message for r in warn_records)


def test_from_dict_error_mode_raises_with_typo():
    """Same injection; mode=error; assert from_dict raises."""
    cfg = _load_example_config("eec")
    cfg["species.liinf.sp0"] = "30.0"
    cfg["validation.strict.enabled"] = "error"
    with _pytest.raises(ValueError, match="species.liinf.sp0"):
        _EngineConfig.from_dict(cfg)
