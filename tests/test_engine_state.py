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
