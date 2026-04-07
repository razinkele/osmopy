"""Tests for ui.state -- shared application state."""

from pathlib import Path

from shiny import reactive

from tests.helpers import make_catch_all_input, make_multi_input
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
        assert "grid.nlon" in cfg
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

        changed = sync_inputs(
            make_multi_input(simulation_nspecies=5, simulation_time_ndtperyear=12, default=None),
            state,
            ["simulation.nspecies", "simulation.time.ndtperyear"],
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

        changed = sync_inputs(make_catch_all_input(None), state, ["simulation.nspecies"])
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


def test_appstate_has_config_name():
    state = AppState()
    with reactive.isolate():
        assert state.config_name.get() == ""


def test_appstate_has_species_names():
    state = AppState()
    with reactive.isolate():
        assert state.species_names.get() == []


def test_appstate_has_results_loaded():
    state = AppState()
    with reactive.isolate():
        assert state.results_loaded.get() is False


def test_config_header_shows_name_and_count():
    state = AppState()
    with reactive.isolate():
        state.config_name.set("Eec Full")
        state.config.set({"a": "1", "b": "2"})
        assert state.config_name.get() == "Eec Full"
        assert len(state.config.get()) == 2


def test_config_header_empty_when_no_config():
    state = AppState()
    with reactive.isolate():
        assert state.config_name.get() == ""


def test_species_names_extracted_from_config():
    state = AppState()
    with reactive.isolate():
        state.config.set(
            {
                "simulation.nspecies": "3",
                "species.name.sp0": "Anchovy",
                "species.name.sp1": "Sardine",
                "species.name.sp2": "Hake",
            }
        )
        cfg = state.config.get()
        n = int(cfg.get("simulation.nspecies", "0"))
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        assert names == ["Anchovy", "Sardine", "Hake"]


def test_species_names_fallback_for_missing():
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "2"})
        cfg = state.config.get()
        n = int(cfg.get("simulation.nspecies", "0"))
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        assert names == ["Species 0", "Species 1"]


def test_species_names_zero_species():
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "0"})
        cfg = state.config.get()
        n = int(cfg.get("simulation.nspecies", "0"))
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        assert names == []


def test_results_loaded_flag():
    state = AppState()
    with reactive.isolate():
        assert state.results_loaded.get() is False
        state.results_loaded.set(True)
        assert state.results_loaded.get() is True
        state.results_loaded.set(False)
        assert state.results_loaded.get() is False
