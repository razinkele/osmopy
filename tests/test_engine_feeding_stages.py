"""Tests for feeding stage config parsing and compute_feeding_stages()."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.feeding_stage import compute_feeding_stages
from osmose.engine.processes.predation import predation
from osmose.engine.state import SchoolState


def _make_base_config() -> dict[str, str]:
    """Minimal valid config for 1 focal species (sp0)."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "15.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "4",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.predprey.sizeratio.min.sp0": "1.0",
        "predation.predprey.sizeratio.max.sp0": "3.5",
    }


def _make_2stage_config() -> dict[str, str]:
    """Config for 2 focal species, sp1 (Hake) has 2 feeding stages."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "5",
        "simulation.nschool.sp1": "5",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Hake",
        "species.linf.sp0": "15.0",
        "species.linf.sp1": "80.0",
        "species.k.sp0": "0.3",
        "species.k.sp1": "0.15",
        "species.t0.sp0": "-0.1",
        "species.t0.sp1": "-0.2",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.005",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.length2weight.allometric.power.sp1": "3.0",
        "species.lifespan.sp0": "4",
        "species.lifespan.sp1": "10",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
        "predation.predprey.sizeratio.min.sp0": "1.0",
        "predation.predprey.sizeratio.max.sp0": "3.5",
        "predation.predprey.sizeratio.min.sp1": "1.0",
        "predation.predprey.sizeratio.max.sp1": "10.0",
        # sp1 has 2 feeding stages with threshold at length=12
        "predation.predprey.stage.threshold.sp1": "12.0",
    }


@pytest.fixture
def base_config() -> dict[str, str]:
    """Minimal valid config for 2 focal species, no background."""
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


class TestConfigParsing:
    """Tests for feeding-stage related config parsing in EngineConfig."""

    def test_single_stage_default(self, base_config):
        """No threshold keys → 1 stage, size_ratio arrays are (n, 1)."""
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.size_ratio_min.shape == (2, 1)
        assert cfg.size_ratio_max.shape == (2, 1)
        np.testing.assert_array_equal(cfg.n_feeding_stages, [1, 1])

    def test_multi_stage_ratios(self, base_config):
        """Multiple stages with semicolon-separated ratios."""
        base_config["predation.predprey.stage.threshold.sp0"] = "5.0"
        base_config["predation.predprey.sizeratio.min.sp0"] = "1.0;2.0"
        base_config["predation.predprey.sizeratio.max.sp0"] = "3.5;5.0"
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.n_feeding_stages[0] == 2
        assert cfg.size_ratio_min.shape[1] == 2  # max_stages == 2
        assert cfg.size_ratio_min[0, 0] == pytest.approx(1.0)
        assert cfg.size_ratio_min[0, 1] == pytest.approx(2.0)
        assert cfg.size_ratio_max[0, 0] == pytest.approx(3.5)
        assert cfg.size_ratio_max[0, 1] == pytest.approx(5.0)
        # sp1 has 1 stage, padded to max_stages=2
        assert cfg.size_ratio_min[1, 0] == cfg.size_ratio_min[1, 1]

    def test_null_threshold_means_one_stage(self, base_config):
        """threshold = "null" → 1 stage."""
        base_config["predation.predprey.stage.threshold.sp0"] = "null"
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.n_feeding_stages[0] == 1

    def test_absent_threshold_means_one_stage(self, base_config):
        """No threshold key at all → 1 stage."""
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.n_feeding_stages[0] == 1

    def test_global_metric_default_size(self, base_config):
        """Absent global structure key defaults to 'size'."""
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.feeding_stage_metric[0] == "size"
        assert cfg.feeding_stage_metric[1] == "size"

    def test_global_metric_set(self, base_config):
        """Global structure key is used for all species."""
        base_config["predation.predprey.stage.structure"] = "age"
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.feeding_stage_metric[0] == "age"
        assert cfg.feeding_stage_metric[1] == "age"

    def test_per_species_metric_override(self, base_config):
        """Per-species structure overrides global."""
        base_config["predation.predprey.stage.structure"] = "size"
        base_config["predation.predprey.stage.structure.sp1"] = "age"
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.feeding_stage_metric[0] == "size"
        assert cfg.feeding_stage_metric[1] == "age"

    def test_2d_shape_single_stage(self, base_config):
        """With no thresholds, shape is (n_species, 1)."""
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.size_ratio_min.ndim == 2
        assert cfg.size_ratio_min.shape == (2, 1)

    def test_2d_shape_multi_stage(self, base_config):
        """Shape expands to (n_species, max_stages)."""
        base_config["predation.predprey.stage.threshold.sp0"] = "5.0;10.0"
        base_config["predation.predprey.sizeratio.min.sp0"] = "1.0;2.0;3.0"
        base_config["predation.predprey.sizeratio.max.sp0"] = "3.5;5.0;6.0"
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.size_ratio_min.shape == (2, 3)
        assert cfg.n_feeding_stages[0] == 3
        assert cfg.n_feeding_stages[1] == 1

    def test_swap_validation(self, base_config):
        """When max > min (Java convention), they get swapped."""
        # Java convention: sizeratio.min=50 (actually the larger value)
        base_config["predation.predprey.sizeratio.min.sp0"] = "50.0"
        base_config["predation.predprey.sizeratio.max.sp0"] = "3.0"
        cfg = EngineConfig.from_dict(base_config)
        # After swap: min should be 3.0, max should be 50.0
        assert cfg.size_ratio_min[0, 0] == pytest.approx(3.0)
        assert cfg.size_ratio_max[0, 0] == pytest.approx(50.0)

    def test_stage_count_mismatch_raises(self, base_config):
        """If ratio count != n_stages, raise ValueError."""
        base_config["predation.predprey.stage.threshold.sp0"] = "5.0"
        base_config["predation.predprey.sizeratio.min.sp0"] = "1.0;2.0;3.0"  # 3, expect 2
        base_config["predation.predprey.sizeratio.max.sp0"] = "3.5;5.0"
        with pytest.raises(ValueError, match="mismatch"):
            EngineConfig.from_dict(base_config)

    def test_trailing_semicolon_handled(self, base_config):
        """Trailing semicolons don't add extra empty values."""
        base_config["predation.predprey.stage.threshold.sp0"] = "5.0"
        base_config["predation.predprey.sizeratio.min.sp0"] = "1.0;2.0;"
        base_config["predation.predprey.sizeratio.max.sp0"] = "3.5;5.0;"
        cfg = EngineConfig.from_dict(base_config)
        assert cfg.n_feeding_stages[0] == 2
        assert cfg.size_ratio_min[0, 1] == pytest.approx(2.0)

    def test_unrecognized_metric_raises(self, base_config):
        """Unrecognized metric raises ValueError."""
        base_config["predation.predprey.stage.structure"] = "garbage"
        with pytest.raises(ValueError, match="Unrecognized"):
            EngineConfig.from_dict(base_config)


class TestComputeFeedingStages:
    """Tests for compute_feeding_stages() function."""

    def _make_config(self, n_species=2, n_dt_per_year=24, thresholds=None, metrics=None):
        """Helper to create a minimal EngineConfig-like object for testing."""
        if thresholds is None:
            thresholds = [[] for _ in range(n_species)]
        if metrics is None:
            metrics = ["size"] * n_species
        return type(
            "FakeConfig",
            (),
            {
                "n_species": n_species,
                "n_dt_per_year": n_dt_per_year,
                "feeding_stage_thresholds": thresholds,
                "feeding_stage_metric": metrics,
                "n_background": 0,
            },
        )()

    def _make_state(self, n, species_id, age_dt=None, length=None, weight=None, tl=None):
        """Create a SchoolState with given values."""
        state = SchoolState.create(n_schools=n, species_id=np.array(species_id, dtype=np.int32))
        if age_dt is not None:
            state = state.replace(age_dt=np.array(age_dt, dtype=np.int32))
        if length is not None:
            state = state.replace(length=np.array(length, dtype=np.float64))
        if weight is not None:
            state = state.replace(weight=np.array(weight, dtype=np.float64))
        if tl is not None:
            state = state.replace(trophic_level=np.array(tl, dtype=np.float64))
        return state

    def test_single_stage_all_zeros(self):
        """No thresholds → all schools in stage 0."""
        config = self._make_config(n_species=2)
        state = self._make_state(4, [0, 0, 1, 1], length=[5.0, 10.0, 8.0, 20.0])
        stages = compute_feeding_stages(state, config)
        np.testing.assert_array_equal(stages, [0, 0, 0, 0])

    def test_size_metric_two_stages(self):
        """Size metric: schools above threshold → stage 1."""
        config = self._make_config(n_species=1, thresholds=[[10.0]], metrics=["size"])
        state = self._make_state(3, [0, 0, 0], length=[5.0, 10.0, 15.0])
        stages = compute_feeding_stages(state, config)
        # 5 < 10 → 0, 10 >= 10 → 1, 15 >= 10 → 1
        np.testing.assert_array_equal(stages, [0, 1, 1])

    def test_age_metric_converts_to_years(self):
        """Age metric: age_dt is converted to years via / n_dt_per_year."""
        config = self._make_config(
            n_species=1, n_dt_per_year=24, thresholds=[[2.0]], metrics=["age"]
        )
        # age_dt=47 → 47/24=1.958 years (< 2.0 → stage 0)
        # age_dt=48 → 48/24=2.0 years (>= 2.0 → stage 1)
        state = self._make_state(2, [0, 0], age_dt=[47, 48])
        stages = compute_feeding_stages(state, config)
        np.testing.assert_array_equal(stages, [0, 1])

    def test_weight_metric_converts_tonnes_to_grams(self):
        """Weight metric: weight (tonnes) * 1e6 → grams."""
        config = self._make_config(n_species=1, thresholds=[[500.0]], metrics=["weight"])
        # weight=0.0004 tonnes = 400g < 500 → stage 0
        # weight=0.0006 tonnes = 600g >= 500 → stage 1
        state = self._make_state(2, [0, 0], weight=[0.0004, 0.0006])
        stages = compute_feeding_stages(state, config)
        np.testing.assert_array_equal(stages, [0, 1])

    def test_tl_metric(self):
        """TL metric: uses trophic_level directly."""
        config = self._make_config(n_species=1, thresholds=[[3.5]], metrics=["tl"])
        state = self._make_state(2, [0, 0], tl=[3.0, 4.0])
        stages = compute_feeding_stages(state, config)
        np.testing.assert_array_equal(stages, [0, 1])

    def test_multiple_thresholds_three_stages(self):
        """Three stages with two thresholds."""
        config = self._make_config(n_species=1, thresholds=[[5.0, 15.0]], metrics=["size"])
        state = self._make_state(4, [0, 0, 0, 0], length=[3.0, 5.0, 14.9, 15.0])
        stages = compute_feeding_stages(state, config)
        np.testing.assert_array_equal(stages, [0, 1, 1, 2])

    def test_exact_threshold_next_stage(self):
        """Value exactly at threshold → next stage (>= comparison)."""
        config = self._make_config(n_species=1, thresholds=[[10.0]], metrics=["size"])
        state = self._make_state(1, [0], length=[10.0])
        stages = compute_feeding_stages(state, config)
        np.testing.assert_array_equal(stages, [1])

    def test_mixed_species(self):
        """Different species can have different metrics and thresholds."""
        config = self._make_config(
            n_species=2,
            n_dt_per_year=24,
            thresholds=[[10.0], [2.0]],
            metrics=["size", "age"],
        )
        # sp0: size-based, threshold 10
        # sp1: age-based, threshold 2 years
        state = self._make_state(
            4,
            [0, 0, 1, 1],
            length=[5.0, 15.0, 0.0, 0.0],
            age_dt=[0, 0, 24, 48],
        )
        stages = compute_feeding_stages(state, config)
        # sp0: 5<10→0, 15>=10→1; sp1: 24/24=1.0<2→0, 48/24=2.0>=2→1
        np.testing.assert_array_equal(stages, [0, 1, 0, 1])

    def test_unrecognized_metric_raises(self):
        """Unrecognized metric in compute raises ValueError."""
        config = self._make_config(n_species=1, thresholds=[[5.0]], metrics=["garbage"])
        state = self._make_state(1, [0], length=[10.0])
        with pytest.raises(ValueError, match="Unrecognized"):
            compute_feeding_stages(state, config)


class TestPredationWithStages:
    """Tests for predation with 2D size ratios and feeding stages."""

    def test_juvenile_uses_stage0_ratios(self):
        """Juvenile Hake (length=8, stage 0) uses wide ratio window -- eats prey."""
        cfg = _make_2stage_config()
        # Stage 0: accept ratios in [1.0, 10.0) -- wide window
        # Stage 1: accept ratios in [2.0, 5.0) -- narrow window
        cfg["predation.predprey.sizeratio.min.sp1"] = "1.0;2.0"
        cfg["predation.predprey.sizeratio.max.sp1"] = "10.0;5.0"
        ec = EngineConfig.from_dict(cfg)

        # Juvenile Hake (sp1, length=8 < threshold 12 -> stage 0)
        # Prey (sp0, length=2), ratio=8/2=4.0 -> stage 0 window [1.0, 10.0) -> ACCEPTED
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([8.0, 2.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, ec, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        assert new_state.abundance[1] < 1000.0  # Prey eaten

    def test_adult_uses_stage1_ratios(self):
        """Adult Hake (length=30, stage 1) uses narrow window -- rejects distant prey."""
        cfg = _make_2stage_config()
        # Stage 0: [1.0, 100.0) wide, Stage 1: [5.0, 8.0) narrow
        cfg["predation.predprey.sizeratio.min.sp1"] = "1.0;5.0"
        cfg["predation.predprey.sizeratio.max.sp1"] = "100.0;8.0"
        ec = EngineConfig.from_dict(cfg)

        # Adult Hake (sp1, length=30 >= threshold 12 -> stage 1)
        # Prey (sp0, length=2), ratio=30/2=15 -> stage 1 window [5.0, 8.0) -> 15 >= 8 -> SKIP
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([30.0, 2.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, ec, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        np.testing.assert_allclose(new_state.abundance[1], 1000.0)  # Prey NOT eaten

    def test_backward_compat_single_stage(self):
        """Single-stage config works identically to pre-B2 behavior."""
        cfg = _make_base_config()
        # Use Python convention: r_min=1.0, r_max=3.5
        cfg["predation.predprey.sizeratio.min.sp0"] = "1.0"
        cfg["predation.predprey.sizeratio.max.sp0"] = "3.5"
        ec = EngineConfig.from_dict(cfg)
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([15.0, 7.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, ec, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        # ratio=15/7~2.14, r_min=1.0, r_max=3.5 -> in [1.0, 3.5) -> ACCEPTED
        assert new_state.abundance[1] < 1000.0

    def test_java_convention_swap_works(self):
        """Java-convention ratios (sizeratio.min=50, max=3) work after swap."""
        cfg = _make_base_config()
        cfg["predation.predprey.sizeratio.min.sp0"] = "50"
        cfg["predation.predprey.sizeratio.max.sp0"] = "3"
        ec = EngineConfig.from_dict(cfg)
        # After swap: r_min=3, r_max=50 -> accept [3, 50)
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 1000.0]),
            weight=np.array([10.0, 1.0]),
            biomass=np.array([1000.0, 1000.0]),
            length=np.array([30.0, 5.0]),
            age_dt=np.array([10, 10], dtype=np.int32),
            first_feeding_age_dt=np.array([0, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, ec, rng, n_subdt=10, grid_ny=1, grid_nx=1)
        # ratio=30/5=6 -> after swap: in [3, 50) -> ACCEPTED
        assert new_state.abundance[1] < 1000.0
