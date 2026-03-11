"""Tests for ui.state -- shared application state."""

from pathlib import Path

from shiny import reactive

from ui.state import AppState


def test_appstate_initial_config_is_empty():
    state = AppState()
    with reactive.isolate():
        assert state.config.get() == {}


def test_appstate_initial_output_dir_is_none():
    state = AppState()
    with reactive.isolate():
        assert state.output_dir.get() is None


def test_appstate_initial_run_result_is_none():
    state = AppState()
    with reactive.isolate():
        assert state.run_result.get() is None


def test_appstate_scenarios_dir_default():
    state = AppState()
    assert state.scenarios_dir == Path("data/scenarios")


def test_appstate_custom_scenarios_dir():
    state = AppState(scenarios_dir=Path("/tmp/my_scenarios"))
    assert state.scenarios_dir == Path("/tmp/my_scenarios")


def test_appstate_config_update():
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "3"})
        assert state.config.get() == {"simulation.nspecies": "3"}


def test_appstate_update_config_key():
    state = AppState()
    with reactive.isolate():
        state.config.set({"a": "1"})
        state.update_config("b", "2")
        assert state.config.get() == {"a": "1", "b": "2"}


def test_appstate_update_config_key_overwrites():
    state = AppState()
    with reactive.isolate():
        state.config.set({"a": "1"})
        state.update_config("a", "99")
        assert state.config.get() == {"a": "99"}


def test_appstate_reset_to_defaults():
    state = AppState()
    with reactive.isolate():
        state.reset_to_defaults()
        cfg = state.config.get()
        # Should have simulation params
        assert "simulation.nspecies" in cfg
        assert cfg["simulation.nspecies"] == "3"
        # Should have grid params
        assert "grid.ncolumn" in cfg
        # Should have species-indexed params expanded for 3 species
        assert "species.linf.sp0" in cfg


def test_appstate_jar_path_default():
    state = AppState()
    with reactive.isolate():
        assert state.jar_path.get() == "osmose-java/osmose_4.3.3-jar-with-dependencies.jar"


def test_appstate_jar_path_set():
    state = AppState()
    with reactive.isolate():
        state.jar_path.set("/path/to/osmose.jar")
        assert state.jar_path.get() == "/path/to/osmose.jar"


def test_sync_inputs_updates_config():
    """sync_inputs should update state.config for non-indexed fields with matching input values."""
    from ui.state import sync_inputs

    state = AppState()
    with reactive.isolate():
        state.reset_to_defaults()

        class FakeInput:
            def __getattr__(self, name):
                def getter():
                    if name == "simulation_nspecies":
                        return 5
                    if name == "simulation_time_ndtperyear":
                        return 12
                    return None

                return getter

        changed = sync_inputs(
            FakeInput(), state, ["simulation.nspecies", "simulation.time.ndtperyear"]
        )
        assert changed["simulation.nspecies"] == "5"
        assert changed["simulation.time.ndtperyear"] == "12"
        assert state.config.get()["simulation.nspecies"] == "5"


def test_appstate_has_busy_field():
    state = AppState()
    with reactive.isolate():
        assert state.busy.get() is None


def test_sync_inputs_skips_none():
    """sync_inputs should skip keys where the input value is None."""
    from ui.state import sync_inputs

    state = AppState()
    with reactive.isolate():
        state.reset_to_defaults()

        class FakeInput:
            def __getattr__(self, name):
                return lambda: None

        changed = sync_inputs(FakeInput(), state, ["simulation.nspecies"])
        assert changed == {}


def test_appstate_has_dirty_field():
    state = AppState()
    with reactive.isolate():
        assert state.dirty.get() is False


def test_update_config_sets_dirty():
    state = AppState()
    with reactive.isolate():
        state.update_config("simulation.nspecies", "5")
        assert state.dirty.get() is True
