"""Tests for loading demo scenarios into the UI state."""

import pytest
from pathlib import Path

from shiny import reactive

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import list_demos, osmose_demo, migrate_config
from ui.state import AppState


def _load_scenario_into_state(scenario: str, tmp_path: Path) -> tuple[AppState, dict[str, str]]:
    """Simulate what handle_load_example() does: generate demo, read, migrate, set state."""
    result = osmose_demo(scenario, tmp_path)
    reader = OsmoseConfigReader()
    cfg = migrate_config(reader.read(result["config_file"]))
    state = AppState()
    with reactive.isolate():
        state.loading.set(True)
        state.config.set(cfg)
        state.config_dir.set(result["config_file"].parent)
        state.load_trigger.set(state.load_trigger.get() + 1)
        state.dirty.set(False)
        state.loading.set(False)
    return state, cfg


@pytest.mark.parametrize("scenario", list_demos())
def test_load_scenario_populates_config(tmp_path, scenario):
    """Each demo scenario should produce a non-empty config dict."""
    state, cfg = _load_scenario_into_state(scenario, tmp_path)
    with reactive.isolate():
        loaded = state.config.get()
    assert len(loaded) > 0
    assert "simulation.nspecies" in loaded or "simulation.nresource" in loaded


@pytest.mark.parametrize("scenario", list_demos())
def test_load_scenario_sets_config_dir(tmp_path, scenario):
    """Config dir should point to the generated config directory."""
    state, _ = _load_scenario_into_state(scenario, tmp_path)
    with reactive.isolate():
        config_dir = state.config_dir.get()
    assert config_dir is not None
    assert config_dir.exists()


@pytest.mark.parametrize("scenario", list_demos())
def test_load_scenario_not_dirty(tmp_path, scenario):
    """Freshly loaded config should not be marked dirty."""
    state, _ = _load_scenario_into_state(scenario, tmp_path)
    with reactive.isolate():
        assert state.dirty.get() is False


@pytest.mark.parametrize("scenario", list_demos())
def test_load_scenario_increments_trigger(tmp_path, scenario):
    """Load trigger should be incremented to force UI re-render."""
    state, _ = _load_scenario_into_state(scenario, tmp_path)
    with reactive.isolate():
        assert state.load_trigger.get() == 1


@pytest.mark.parametrize("scenario", list_demos())
def test_load_scenario_has_version(tmp_path, scenario):
    """Migrated config should have osmose.version set."""
    _, cfg = _load_scenario_into_state(scenario, tmp_path)
    assert "osmose.version" in cfg


def test_load_bay_of_biscay_species(tmp_path):
    """Bay of Biscay should load 8 species with correct names."""
    _, cfg = _load_scenario_into_state("bay_of_biscay", tmp_path)
    assert cfg["simulation.nspecies"] == "8"
    assert cfg["species.name.sp0"] == "Anchovy"
    assert cfg["species.name.sp7"] == "BlueWhiting"
    # Should have species biology params
    assert "species.linf.sp0" in cfg
    assert "species.k.sp0" in cfg
    assert "population.seeding.biomass.sp0" in cfg


def test_load_bay_of_biscay_resources(tmp_path):
    """Bay of Biscay should include 6 LTL resource species."""
    _, cfg = _load_scenario_into_state("bay_of_biscay", tmp_path)
    assert cfg["simulation.nresource"] == "6"
    assert cfg["species.name.sp8"] == "SmallPhyto"
    assert cfg["species.type.sp8"] == "resource"


def test_load_bay_of_biscay_grid(tmp_path):
    """Bay of Biscay grid should use post-migration key names."""
    _, cfg = _load_scenario_into_state("bay_of_biscay", tmp_path)
    assert cfg["grid.nlon"] == "20"
    assert cfg["grid.nlat"] == "20"
    # Should NOT have pre-migration keys
    assert "grid.ncolumn" not in cfg
    assert "grid.nline" not in cfg


def test_load_bay_of_biscay_mortality(tmp_path):
    """Bay of Biscay should use post-migration mortality keys."""
    _, cfg = _load_scenario_into_state("bay_of_biscay", tmp_path)
    assert "mortality.additional.rate.sp0" in cfg
    # Should NOT have pre-migration keys
    assert "mortality.natural.rate.sp0" not in cfg


def test_load_bay_of_biscay_fishing(tmp_path):
    """Bay of Biscay should include fishing rates."""
    _, cfg = _load_scenario_into_state("bay_of_biscay", tmp_path)
    assert cfg["simulation.fishing.mortality.enabled"] == "true"
    assert "mortality.fishing.rate.sp0" in cfg


def test_load_eec_species(tmp_path):
    """Simplified EEC should load 6 species."""
    _, cfg = _load_scenario_into_state("eec", tmp_path)
    assert cfg["simulation.nspecies"] == "6"
    assert cfg["species.name.sp0"] == "Herring"
    assert cfg["species.name.sp5"] == "Cod"


def test_load_eec_full_species(tmp_path):
    """Full EEC should load 14 focal species + 10 LTL resources."""
    _, cfg = _load_scenario_into_state("eec_full", tmp_path)
    assert cfg["simulation.nspecies"] == "14"
    assert cfg["simulation.nresource"] == "10"
    assert cfg["species.name.sp0"] == "lesserSpottedDogfish"
    assert cfg["species.name.sp13"] == "squids"
    # LTL resources
    assert cfg["species.name.sp14"] == "Dinoflagellates"
    assert cfg["species.type.sp14"] == "resource"


def test_load_eec_full_has_movement_maps(tmp_path):
    """Full EEC config should reference movement distribution maps."""
    _, cfg = _load_scenario_into_state("eec_full", tmp_path)
    # Movement maps are referenced via movement.map* keys
    movement_keys = [k for k in cfg if k.startswith("movement.")]
    assert len(movement_keys) > 0


def test_load_minimal_species(tmp_path):
    """Minimal should load 2 species (Anchovy + Hake)."""
    _, cfg = _load_scenario_into_state("minimal", tmp_path)
    assert cfg["simulation.nspecies"] == "2"
    assert cfg["species.name.sp0"] == "Anchovy"
    assert cfg["species.name.sp1"] == "Hake"


def test_load_minimal_no_fishing(tmp_path):
    """Minimal demo should have fishing disabled."""
    _, cfg = _load_scenario_into_state("minimal", tmp_path)
    assert cfg["simulation.fishing.mortality.enabled"] == "false"


def test_load_minimal_no_resources(tmp_path):
    """Minimal demo should have 0 LTL resources."""
    _, cfg = _load_scenario_into_state("minimal", tmp_path)
    assert cfg["simulation.nresource"] == "0"


def test_load_scenario_edit_marks_dirty(tmp_path):
    """Editing a loaded config should mark it dirty."""
    state, _ = _load_scenario_into_state("minimal", tmp_path)
    with reactive.isolate():
        assert state.dirty.get() is False
        state.update_config("simulation.nspecies", "5")
        assert state.dirty.get() is True


def test_load_scenario_loading_guard(tmp_path):
    """During loading, sync_inputs should be blocked by the loading guard."""
    from ui.state import sync_inputs

    result = osmose_demo("minimal", tmp_path)
    reader = OsmoseConfigReader()
    cfg = migrate_config(reader.read(result["config_file"]))

    state = AppState()
    with reactive.isolate():
        state.loading.set(True)
        state.config.set(cfg)

        # sync_inputs should return empty while loading is True
        class FakeInput:
            def __getattr__(self, name):
                return lambda: "overwritten"

        changed = sync_inputs(FakeInput(), state, ["simulation.nspecies"])
        assert changed == {}

        state.loading.set(False)


def test_all_demos_produce_unique_configs(tmp_path):
    """Each demo should produce a distinct configuration."""
    configs = {}
    for scenario in list_demos():
        sub = tmp_path / scenario
        _, cfg = _load_scenario_into_state(scenario, sub)
        configs[scenario] = cfg

    # Each pair of demos should differ in species count or species names
    demo_list = list_demos()
    for i, a in enumerate(demo_list):
        for b in demo_list[i + 1 :]:
            assert configs[a].get("simulation.nspecies") != configs[b].get(
                "simulation.nspecies"
            ) or configs[a].get("species.name.sp0") != configs[b].get("species.name.sp0"), (
                f"Demos {a} and {b} produced identical configs"
            )
