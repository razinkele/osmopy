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


class TestSchoolStateValidate:
    """Opt-in validation of biological invariants on SchoolState.
    Deep review v3 I-1.
    """

    def _minimal_valid_state(self, n: int = 2):
        import numpy as np
        from osmose.engine.state import SchoolState

        s = SchoolState.create(
            n_schools=n, species_id=np.zeros(n, dtype=np.int32)
        )
        return s.replace(
            abundance=np.full(n, 100.0),
            weight=np.full(n, 0.01),
            biomass=np.full(n, 1.0),
            length=np.full(n, 10.0),
            age_dt=np.zeros(n, dtype=np.int32),
            cell_x=np.zeros(n, dtype=np.int32),
            cell_y=np.zeros(n, dtype=np.int32),
        )

    def test_validate_passes_on_clean_state(self):
        s = self._minimal_valid_state()
        s.validate()

    def test_validate_raises_on_negative_abundance(self):
        import numpy as np
        import pytest
        s = self._minimal_valid_state()
        s = s.replace(abundance=np.array([-1.0, 10.0]))
        with pytest.raises(ValueError, match="abundance must be non-negative"):
            s.validate()

    def test_validate_raises_on_negative_length(self):
        import numpy as np
        import pytest
        s = self._minimal_valid_state()
        s = s.replace(length=np.array([-5.0, 10.0]))
        with pytest.raises(ValueError, match="length must be non-negative"):
            s.validate()

    def test_validate_raises_on_biomass_mismatch(self):
        import numpy as np
        import pytest
        s = self._minimal_valid_state()
        s = s.replace(biomass=np.array([2.0, 1.0]))
        with pytest.raises(ValueError, match="biomass .* abundance \\* weight"):
            s.validate()

    def test_validate_raises_on_negative_cell(self):
        import numpy as np
        import pytest
        s = self._minimal_valid_state()
        s = s.replace(cell_x=np.array([-1, 0], dtype=np.int32))
        with pytest.raises(ValueError, match="cell_x must be non-negative"):
            s.validate()

    def test_validate_skip_dead_schools(self):
        import numpy as np
        s = self._minimal_valid_state()
        s = s.replace(
            abundance=np.array([100.0, 0.0]),
            weight=np.array([0.01, 0.0]),
            biomass=np.array([1.0, 0.0]),
        )
        s.validate()


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
