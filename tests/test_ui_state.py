"""Tests for AppState and sync_inputs — UI state management edge cases."""

import pytest

pytest.importorskip("shiny")

from shiny import reactive

from ui.state import AppState, sync_inputs


def test_appstate_initializes_with_empty_config():
    """AppState starts with an empty config dict."""
    state = AppState()
    with reactive.isolate():
        assert state.config.get() == {}


def test_appstate_initially_not_dirty():
    """AppState dirty flag starts False."""
    state = AppState()
    with reactive.isolate():
        assert state.dirty.get() is False


def test_update_config_marks_dirty():
    """Updating a config key that didn't exist sets dirty=True."""
    state = AppState()
    with reactive.isolate():
        state.update_config("simulation.nspecies", "4")
        assert state.dirty.get() is True


def test_update_config_same_value_does_not_mark_dirty():
    """Setting the same value twice should not re-set dirty on second call."""
    state = AppState()
    with reactive.isolate():
        # Pre-populate the key
        state.config.set({"simulation.nspecies": "3"})
        state.dirty.set(False)
        # Now update with the same value
        state.update_config("simulation.nspecies", "3")
        assert state.dirty.get() is False


def test_update_config_different_value_marks_dirty():
    """Changing a key to a new value sets dirty=True."""
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "3"})
        state.dirty.set(False)
        state.update_config("simulation.nspecies", "5")
        assert state.dirty.get() is True


def test_update_config_persists_value():
    """update_config stores the new value in the config dict."""
    state = AppState()
    with reactive.isolate():
        state.update_config("grid.nlon", "100")
        assert state.config.get()["grid.nlon"] == "100"


def test_reset_to_defaults_clears_dirty():
    """reset_to_defaults populates config and resets dirty to False."""
    state = AppState()
    with reactive.isolate():
        state.update_config("simulation.nspecies", "9")
        assert state.dirty.get() is True
        state.reset_to_defaults()
        assert state.dirty.get() is False


def test_sync_inputs_skips_loading_state():
    """sync_inputs returns empty dict when loading flag is True."""
    state = AppState()
    with reactive.isolate():
        state.loading.set(True)

        class FakeInput:
            def __getattr__(self, name):
                return lambda: "42"

        changed = sync_inputs(FakeInput(), state, ["simulation.nspecies"])
        assert changed == {}


def test_sync_inputs_skips_missing_input_attr():
    """sync_inputs silently skips keys whose input attribute doesn't exist."""
    state = AppState()
    with reactive.isolate():

        class EmptyInput:
            pass

        changed = sync_inputs(EmptyInput(), state, ["simulation.nspecies"])
        assert changed == {}


def test_sync_inputs_does_not_set_dirty_when_value_unchanged():
    """sync_inputs leaves dirty=False when input value matches existing config."""
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "3"})
        state.dirty.set(False)

        class FakeInput:
            def __getattr__(self, name):
                return lambda: 3  # same as stored "3"

        sync_inputs(FakeInput(), state, ["simulation.nspecies"])
        assert state.dirty.get() is False
