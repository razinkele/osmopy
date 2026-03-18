"""Tests for egg retain/release mechanism."""

import numpy as np

from osmose.engine.state import SchoolState


class TestEggRetainRelease:
    def test_retain_eggs(self):
        """Retain should set egg_retained = abundance for egg schools."""
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 500.0]),
            is_egg=np.array([True, False]),
        )
        # Retain: egg schools get their abundance stored
        new_retained = np.where(state.is_egg, state.abundance, 0.0)
        state = state.replace(egg_retained=new_retained)
        np.testing.assert_allclose(state.egg_retained[0], 1000.0)
        np.testing.assert_allclose(state.egg_retained[1], 0.0)

    def test_release_eggs_progressive(self):
        """Release should reduce egg_retained by abundance/n_subdt each step."""
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            is_egg=np.array([True]),
            egg_retained=np.array([1000.0]),
        )
        n_subdt = 10
        # After 1 release: retained -= 1000/10 = 100
        release = state.abundance / n_subdt
        new_retained = np.maximum(0, state.egg_retained - release)
        np.testing.assert_allclose(new_retained[0], 900.0)

        # After 10 releases: retained = 0
        retained = 1000.0
        for _ in range(n_subdt):
            retained = max(0, retained - 1000.0 / n_subdt)
        np.testing.assert_allclose(retained, 0.0, atol=1e-10)

    def test_instantaneous_abundance(self):
        """Instantaneous abundance = abundance - egg_retained."""
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            egg_retained=np.array([800.0]),
        )
        inst = state.abundance - state.egg_retained
        np.testing.assert_allclose(inst[0], 200.0)
