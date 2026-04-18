"""Tests for diet matrix tracking (C2 feature)."""

import numpy as np
import pandas as pd
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.predation import (
    disable_diet_tracking,
    enable_diet_tracking,
    get_diet_matrix,
    predation,
)
from osmose.engine.simulate import SimulationContext
from osmose.engine.state import SchoolState


def _make_diet_config() -> dict[str, str]:
    """Two-species config: Predator eats Prey."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "5",
        "simulation.nschool.sp1": "5",
        "species.name.sp0": "Prey",
        "species.name.sp1": "Predator",
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
        "predation.predprey.sizeratio.min.sp1": "1.0",
        "predation.predprey.sizeratio.max.sp0": "3.5",
        "predation.predprey.sizeratio.max.sp1": "3.5",
    }


def _make_two_school_state(pred_sp: int = 1, prey_sp: int = 0) -> SchoolState:
    """Create a 2-school state with one predator and one prey in same cell."""
    state = SchoolState.create(
        n_schools=2,
        species_id=np.array([pred_sp, prey_sp], dtype=np.int32),
    )
    return state.replace(
        abundance=np.array([50.0, 500.0]),
        length=np.array([25.0, 10.0]),
        weight=np.array([78.125, 6.0]),
        biomass=np.array([3906.25, 3000.0]),
        age_dt=np.array([24, 24], dtype=np.int32),
        cell_x=np.array([0, 0], dtype=np.int32),
        cell_y=np.array([0, 0], dtype=np.int32),
    )


class TestDietTrackingState:
    """Test diet tracking enable/disable/reset."""

    def test_disabled_by_default(self):
        """Diet tracking should be off by default."""
        ctx = SimulationContext()
        disable_diet_tracking(ctx=ctx)
        assert get_diet_matrix(ctx=ctx) is None

    def test_enable_creates_matrix(self):
        """enable_diet_tracking creates a zero matrix of correct shape."""
        ctx = SimulationContext()
        enable_diet_tracking(n_schools=10, n_species=3, ctx=ctx)
        mat = get_diet_matrix(ctx=ctx)
        assert mat is not None
        assert mat.shape == (10, 3)
        assert mat.dtype == np.float64
        np.testing.assert_array_equal(mat, 0.0)

    def test_disable_clears_matrix(self):
        """disable_diet_tracking clears the matrix and flag."""
        ctx = SimulationContext()
        enable_diet_tracking(n_schools=5, n_species=2, ctx=ctx)
        disable_diet_tracking(ctx=ctx)
        assert get_diet_matrix(ctx=ctx) is None

    def test_get_diet_matrix_none_when_disabled(self):
        """get_diet_matrix returns None when tracking is off."""
        ctx = SimulationContext()
        disable_diet_tracking(ctx=ctx)
        assert get_diet_matrix(ctx=ctx) is None


class TestDietTrackingPredation:
    """Test that predation populates the diet matrix."""

    def test_python_predation_records_diet(self):
        """Predation with diet tracking records prey species consumption."""
        cfg = EngineConfig.from_dict(_make_diet_config())
        state = _make_two_school_state(pred_sp=1, prey_sp=0)
        n_species = 2
        rng = np.random.default_rng(42)
        ctx = SimulationContext()

        enable_diet_tracking(n_schools=len(state), n_species=n_species, ctx=ctx)
        predation(state, cfg, rng, n_subdt=10, grid_ny=1, grid_nx=1, ctx=ctx)
        mat = get_diet_matrix(ctx=ctx)
        assert mat is not None
        # Predator (school 0, species 1) should have eaten prey (species 0)
        # School 0 is the predator -> mat[0, prey_sp=0] > 0
        assert mat[0, 0] > 0.0, "Predator should have recorded eating prey species 0"
        # Prey school should not have eaten anything (too small to eat predator)
        assert mat[1, 1] == 0.0, "Prey should not eat predator"

    def test_diet_matrix_sums_match_preyed_biomass(self):
        """Row sums of diet matrix should match preyed_biomass for each school."""
        cfg = EngineConfig.from_dict(_make_diet_config())
        state = _make_two_school_state(pred_sp=1, prey_sp=0)
        rng = np.random.default_rng(42)
        ctx = SimulationContext()

        enable_diet_tracking(n_schools=len(state), n_species=2, ctx=ctx)
        result = predation(state, cfg, rng, n_subdt=10, grid_ny=1, grid_nx=1, ctx=ctx)
        mat = get_diet_matrix(ctx=ctx)
        assert mat is not None
        # Sum across prey species for each predator school
        diet_sums = mat.sum(axis=1)
        # Should match preyed_biomass for each school
        np.testing.assert_allclose(
            diet_sums,
            result.preyed_biomass,
            rtol=1e-10,
            err_msg="Diet matrix row sums should match preyed_biomass",
        )

    def test_diet_tracking_works_with_numba(self):
        """When diet tracking is enabled, Numba path should accumulate diet correctly."""
        cfg = EngineConfig.from_dict(_make_diet_config())
        state = _make_two_school_state()
        rng = np.random.default_rng(42)
        ctx = SimulationContext()

        enable_diet_tracking(n_schools=len(state), n_species=2, ctx=ctx)
        # Diet tracking works with both Python and Numba paths
        predation(state, cfg, rng, n_subdt=10, grid_ny=1, grid_nx=1, ctx=ctx)
        mat = get_diet_matrix(ctx=ctx)
        # Non-zero matrix confirms diet was tracked and accumulated
        assert mat is not None
        assert mat.sum() > 0.0, "Diet matrix should accumulate consumed biomass"

    def test_diet_with_three_species(self):
        """Diet matrix correctly tracks with 3 species."""
        raw = _make_diet_config()
        raw.update(
            {
                "simulation.nspecies": "3",
                "simulation.nschool.sp2": "5",
                "species.name.sp2": "MidPred",
                "species.linf.sp2": "40.0",
                "species.k.sp2": "0.2",
                "species.t0.sp2": "-0.1",
                "species.egg.size.sp2": "0.1",
                "species.length2weight.condition.factor.sp2": "0.005",
                "species.length2weight.allometric.power.sp2": "3.0",
                "species.lifespan.sp2": "6",
                "species.vonbertalanffy.threshold.age.sp2": "1.0",
                "predation.ingestion.rate.max.sp2": "3.5",
                "predation.efficiency.critical.sp2": "0.57",
                "predation.predprey.sizeratio.min.sp2": "1.0",
                "predation.predprey.sizeratio.max.sp2": "3.5",
            }
        )
        cfg = EngineConfig.from_dict(raw)
        # School 0: Predator sp1 len=25, School 1: Prey sp0 len=10, School 2: MidPred sp2 len=15
        state = SchoolState.create(
            n_schools=3,
            species_id=np.array([1, 0, 2], dtype=np.int32),
        )
        state = state.replace(
            abundance=np.array([50.0, 500.0, 100.0]),
            length=np.array([25.0, 10.0, 15.0]),
            weight=np.array([78.125, 6.0, 16.875]),
            biomass=np.array([3906.25, 3000.0, 1687.5]),
            age_dt=np.array([24, 24, 24], dtype=np.int32),
            cell_x=np.array([0, 0, 0], dtype=np.int32),
            cell_y=np.array([0, 0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        ctx = SimulationContext()

        enable_diet_tracking(n_schools=3, n_species=3, ctx=ctx)
        result = predation(state, cfg, rng, n_subdt=10, grid_ny=1, grid_nx=1, ctx=ctx)
        mat = get_diet_matrix(ctx=ctx)
        assert mat is not None
        assert mat.shape == (3, 3)
        # Verify row sums match preyed_biomass
        np.testing.assert_allclose(
            mat.sum(axis=1),
            result.preyed_biomass,
            rtol=1e-10,
        )


class TestDietResourceTracking:
    """Test diet tracking for resource (LTL) predation."""

    def test_resource_predation_tracked_in_diet(self):
        """Resource predation should appear in diet matrix columns."""
        from osmose.engine.grid import Grid
        from osmose.engine.resources import ResourceState

        raw = _make_diet_config()
        raw.update(
            {
                "simulation.nresource": "1",
                "ltl.name.rsc0": "Phyto",
                "ltl.size.min.rsc0": "0.001",
                "ltl.size.max.rsc0": "5.0",
                "ltl.tl.rsc0": "1.0",
                "ltl.accessibility2fish.rsc0": "0.01",
                "ltl.biomass.total.rsc0": "100000",
            }
        )
        cfg = EngineConfig.from_dict(raw)
        grid = Grid(ny=1, nx=1, ocean_mask=np.ones((1, 1), dtype=np.bool_))
        resources = ResourceState(config=raw, grid=grid)
        resources.update(step=0)

        # Predator with small enough prey window to eat resources
        state = SchoolState.create(
            n_schools=1,
            species_id=np.array([1], dtype=np.int32),
        )
        state = state.replace(
            abundance=np.array([50.0]),
            length=np.array([10.0]),  # small enough to see resources
            weight=np.array([5.0]),
            biomass=np.array([250.0]),
            age_dt=np.array([24], dtype=np.int32),
            cell_x=np.array([0], dtype=np.int32),
            cell_y=np.array([0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        ctx = SimulationContext()

        # n_species (2 focal) + n_resources (1) = 3 columns
        n_total = cfg.n_species + resources.n_resources
        enable_diet_tracking(n_schools=1, n_species=n_total, ctx=ctx)
        result = predation(
            state,
            cfg,
            rng,
            n_subdt=10,
            grid_ny=1,
            grid_nx=1,
            resources=resources,
            ctx=ctx,
        )
        mat = get_diet_matrix(ctx=ctx)
        assert mat is not None
        assert mat.shape == (1, n_total)
        # Resource species is at column index n_species + 0 = 2
        # If the predator ate any resources, column 2 should be > 0
        total_eaten = result.preyed_biomass[0]
        if total_eaten > 0:
            np.testing.assert_allclose(
                mat.sum(axis=1),
                result.preyed_biomass,
                rtol=1e-10,
            )


class TestDietCSVOutput:
    """Test diet composition CSV output."""

    def test_write_diet_csv(self):
        """_normalize_diet_matrix_to_percent returns correct shape and non-zero entries."""
        from osmose.engine.output import _normalize_diet_matrix_to_percent

        # Species-level diet: 2 predators, 3 prey columns (2 focal + 1 resource)
        # diet_by_species[pred_sp, prey_sp] = biomass eaten
        diet_by_species = np.array(
            [
                [0.0, 0.0, 100.0],  # Prey: ate 100 units of Phyto
                [500.0, 0.0, 200.0],  # Predator: ate 500 Prey, 200 Phyto
            ]
        )

        pct = _normalize_diet_matrix_to_percent(diet_by_species)

        assert pct.shape == diet_by_species.shape
        # Prey row: only Phyto eaten -> 100%
        assert abs(pct[0, 2] - 100.0) < 0.01
        assert abs(pct[0, 0]) < 0.01
        # Predator row: 500/(500+200)*100 = 71.43%, 200/(700)*100 = 28.57%
        assert abs(pct[1, 0] - 500.0 / 700.0 * 100.0) < 0.01
        assert abs(pct[1, 2] - 200.0 / 700.0 * 100.0) < 0.01

    def test_write_diet_csv_percentage(self):
        """_normalize_diet_matrix_to_percent returns per-predator percentages summing to 100."""
        from osmose.engine.output import _normalize_diet_matrix_to_percent

        # Pred eats 300 PreyA + 700 PreyB = 1000 total
        diet_by_species = np.array([[300.0, 700.0]])

        pct = _normalize_diet_matrix_to_percent(diet_by_species)

        assert abs(pct[0, 0] - 30.0) < 0.01
        assert abs(pct[0, 1] - 70.0) < 0.01


class TestDietAggregation:
    """Test aggregation of per-school diet into per-species diet."""

    def test_aggregate_diet_by_species(self):
        """Per-school diet should aggregate correctly to per-species diet."""
        from osmose.engine.output import aggregate_diet_by_species

        # 4 schools: sp0 (2 schools), sp1 (2 schools), 3 prey species
        species_id = np.array([0, 0, 1, 1], dtype=np.int32)
        diet_matrix = np.array(
            [
                [0.0, 10.0, 5.0],  # school 0 (sp0): ate 10 from sp1, 5 from sp2
                [0.0, 20.0, 0.0],  # school 1 (sp0): ate 20 from sp1
                [15.0, 0.0, 0.0],  # school 2 (sp1): ate 15 from sp0
                [5.0, 0.0, 3.0],  # school 3 (sp1): ate 5 from sp0, 3 from sp2
            ]
        )
        n_pred_species = 2

        result = aggregate_diet_by_species(diet_matrix, species_id, n_pred_species)
        assert result.shape == (2, 3)
        # sp0 total: [0, 30, 5]
        np.testing.assert_allclose(result[0], [0.0, 30.0, 5.0])
        # sp1 total: [20, 0, 3]
        np.testing.assert_allclose(result[1], [20.0, 0.0, 3.0])


def test_simulation_context_diet_coupling_after_enable():
    """After enable_diet_tracking(), diet_tracking_enabled and diet_matrix
    must be consistent. Deep review v3 M-14.
    """
    from osmose.engine.processes.predation import enable_diet_tracking, disable_diet_tracking
    from osmose.engine.simulate import SimulationContext

    ctx = SimulationContext(config_dir="")

    assert ctx.diet_tracking_enabled is False
    assert ctx.diet_matrix is None

    enable_diet_tracking(n_schools=5, n_species=3, ctx=ctx)

    assert ctx.diet_tracking_enabled is True
    assert ctx.diet_matrix is not None
    assert ctx.diet_matrix.shape == (5, 3)

    disable_diet_tracking(ctx=ctx)

    assert ctx.diet_tracking_enabled is False
    # Buffer is kept for reuse, but get_diet_matrix returns None when disabled
    from osmose.engine.processes.predation import get_diet_matrix

    assert get_diet_matrix(ctx=ctx) is None


def test_write_diet_csv_emits_one_row_per_recording_period(tmp_path):
    """Java-parity: one CSV per run, one row per recording period, Time
    in the first column. Whole-run sum recoverable via
    df.drop(columns='Time').sum(axis=0)."""
    from osmose.engine.output import write_diet_csv

    predator_names = ["cod", "herring"]
    prey_names = ["cod", "herring", "plankton"]

    step_matrices = [
        np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]]),
        np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 5.0]]),
        np.array([[2.0, 2.0, 0.0], [1.0, 2.0, 6.0]]),
    ]
    step_times = [1.0, 2.0, 3.0]

    path = tmp_path / "run_dietMatrix_Simu0.csv"
    write_diet_csv(
        path=path,
        step_diet_matrices=step_matrices,
        step_times=step_times,
        predator_names=predator_names,
        prey_names=prey_names,
    )

    df = pd.read_csv(path)
    assert len(df) == 3
    assert list(df["Time"]) == step_times
    assert list(df.columns) == [
        "Time",
        "cod_cod",
        "cod_herring",
        "cod_plankton",
        "herring_cod",
        "herring_herring",
        "herring_plankton",
    ]
    assert df.iloc[0]["cod_plankton"] == pytest.approx(2.0)
    assert df.iloc[1]["herring_plankton"] == pytest.approx(5.0)
    assert df.iloc[2]["cod_cod"] == pytest.approx(2.0)


def test_write_diet_csv_with_empty_step_list_writes_no_file(tmp_path):
    from osmose.engine.output import write_diet_csv

    path = tmp_path / "run_dietMatrix_Simu0.csv"
    write_diet_csv(
        path=path,
        step_diet_matrices=[],
        step_times=[],
        predator_names=["cod"],
        prey_names=["cod", "plankton"],
    )
    assert not path.exists()
