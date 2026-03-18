"""Tests for the engine protocol and state foundation."""

from pathlib import Path

import numpy as np

from osmose.engine import JavaEngine, PythonEngine
from osmose.engine.state import MortalityCause, SchoolState


def test_python_engine_satisfies_protocol():
    """PythonEngine must have run() and run_ensemble() methods."""
    engine = PythonEngine()
    assert hasattr(engine, "run")
    assert hasattr(engine, "run_ensemble")


def test_java_engine_satisfies_protocol():
    """JavaEngine must have run() and run_ensemble() methods."""
    engine = JavaEngine(jar_path=Path("/fake.jar"))
    assert hasattr(engine, "run")
    assert hasattr(engine, "run_ensemble")


class TestSchoolState:
    """Tests for SchoolState creation, replace, append, and compact."""

    def _make_state(self, n: int = 5) -> SchoolState:
        """Helper: create a minimal SchoolState with n schools."""
        return SchoolState.create(
            n_schools=n,
            species_id=np.arange(n, dtype=np.int32) % 3,
        )

    def test_create_default(self):
        state = self._make_state(5)
        assert len(state) == 5
        assert state.species_id.shape == (5,)
        assert state.abundance.shape == (5,)
        assert state.n_dead.shape == (5, len(MortalityCause))

    def test_create_sets_species_id(self):
        state = self._make_state(3)
        np.testing.assert_array_equal(state.species_id, [0, 1, 2])

    def test_replace_returns_new_state(self):
        state = self._make_state(3)
        new_abundance = np.array([100.0, 200.0, 300.0])
        new_state = state.replace(abundance=new_abundance)
        np.testing.assert_array_equal(new_state.abundance, [100.0, 200.0, 300.0])
        # Original unchanged
        np.testing.assert_array_equal(state.abundance, np.zeros(3))

    def test_append_adds_schools(self):
        state = self._make_state(3)
        extra = self._make_state(2)
        merged = state.append(extra)
        assert len(merged) == 5

    def test_compact_removes_dead(self):
        state = self._make_state(5)
        state = state.replace(abundance=np.array([100.0, 0.0, 50.0, 0.0, 25.0]))
        compacted = state.compact()
        assert len(compacted) == 3
        np.testing.assert_array_equal(compacted.abundance, [100.0, 50.0, 25.0])

    def test_mortality_cause_enum(self):
        assert MortalityCause.PREDATION.value == 0
        assert MortalityCause.AGING.value == 7
        assert len(MortalityCause) == 8


class TestPythonEngineIntegration:
    def test_run_with_minimal_config(self, tmp_path):
        """PythonEngine.run() should complete with a minimal config."""
        config = {
            "simulation.time.ndtperyear": "12",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "5",
            "species.name.sp0": "TestFish",
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
        }
        engine = PythonEngine()
        result = engine.run(config=config, output_dir=tmp_path, seed=42)
        assert result.returncode == 0
        assert result.output_dir == tmp_path
