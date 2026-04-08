"""Tests for engine_mode reactive state."""

from shiny import reactive

from ui.state import AppState


def test_engine_mode_default_is_java():
    state = AppState()
    with reactive.isolate():
        assert state.engine_mode.get() == "java"


def test_engine_mode_can_be_set_to_python():
    state = AppState()
    with reactive.isolate():
        state.engine_mode.set("python")
        assert state.engine_mode.get() == "python"
