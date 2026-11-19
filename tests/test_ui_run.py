"""Tests for run page logic -- config writing, override parsing, status flow."""

from pathlib import Path

from osmose.config.validator import validate_config
from osmose.schema import build_registry
from ui.pages.run import _inject_random_movement_ncell, parse_overrides, write_temp_config


def test_parse_overrides_empty():
    assert parse_overrides("") == {}


def test_parse_overrides_single():
    assert parse_overrides("simulation.nspecies=5") == {"simulation.nspecies": "5"}


def test_parse_overrides_multiple():
    text = "simulation.nspecies=5\nspecies.k.sp0=0.3"
    result = parse_overrides(text)
    assert result == {"simulation.nspecies": "5", "species.k.sp0": "0.3"}


def test_parse_overrides_skips_blank_lines():
    text = "a=1\n\nb=2\n"
    assert parse_overrides(text) == {"a": "1", "b": "2"}


def test_parse_overrides_strips_whitespace():
    text = "  a = 1  \n  b=2"
    assert parse_overrides(text) == {"a": "1", "b": "2"}


def test_write_temp_config(tmp_path):
    config = {"simulation.nspecies": "3", "species.k.sp0": "0.2"}
    config_path = write_temp_config(config, tmp_path)
    assert config_path.exists()
    assert config_path.name == "osm_all-parameters.csv"
    content = config_path.read_text()
    assert "simulation.nspecies" in content


def test_inject_ncell_for_random_movement():
    config = {
        "grid.nlon": "15",
        "grid.nlat": "12",
        "movement.distribution.method.sp0": "random",
        "movement.distribution.method.sp1": "maps",
        "movement.distribution.method.sp2": "random",
    }
    out = _inject_random_movement_ncell(config)
    assert out["movement.distribution.ncell.sp0"] == "180"
    assert "movement.distribution.ncell.sp1" not in out
    assert out["movement.distribution.ncell.sp2"] == "180"


def test_inject_ncell_preserves_existing():
    config = {
        "grid.nlon": "15",
        "grid.nlat": "12",
        "movement.distribution.method.sp0": "random",
        "movement.distribution.ncell.sp0": "50",
    }
    out = _inject_random_movement_ncell(config)
    assert out["movement.distribution.ncell.sp0"] == "50"


def test_inject_ncell_does_not_mutate_caller():
    """H3: callers of _inject_random_movement_ncell rely on the input not mutating
    (e.g. when the input is a reactive value's dict-of-record). Pin the contract."""
    config = {
        "grid.nlon": "10",
        "grid.nlat": "10",
        "movement.distribution.method.sp0": "random",
    }
    snapshot = dict(config)
    out = _inject_random_movement_ncell(config)
    assert config == snapshot, "input dict was mutated"
    assert out is not config, "output must be a new dict, not the input"
    assert out["movement.distribution.ncell.sp0"] == "100"


def test_inject_ncell_invalid_grid_returns_unchanged_copy():
    """When grid dims are invalid, the return must still be a NEW dict (copy)."""
    config = {"grid.nlon": "not-a-number", "grid.nlat": "10"}
    out = _inject_random_movement_ncell(config)
    assert out == config
    assert out is not config


def test_prerun_validation_blocks_on_errors():
    registry = build_registry()
    config = {"species.linf.sp0": "not_a_number", "simulation.nspecies": "2"}
    errors, _warnings = validate_config(config, registry)
    assert len(errors) > 0


def test_prerun_validation_passes_valid_config():
    registry = build_registry()
    config = {"simulation.nspecies": "3", "species.linf.sp0": "50.0"}
    errors, _warnings = validate_config(config, registry)
    assert len(errors) == 0


def test_live_movement_wired_into_run_page():
    """The live-movement view (toggle, map, queue, poll, observer) is wired into run.py."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "run.py").read_text()
    assert "make_step_observer" in src
    assert "live_view_expanded" in src
    assert "choose_live_layer" in src  # the layer chooser (dots/heatmap) the live map now uses
    assert "partial_update" in src


def test_species_choices_from_config():
    from ui.pages.run import _species_choices

    cfg = {"simulation.nspecies": "2", "species.name.sp0": "Cod", "species.name.sp1": "Sprat"}
    assert _species_choices(cfg) == {"__all__": "All species", "Cod": "Cod", "Sprat": "Sprat"}
    # empty / missing config -> just the all-option
    assert _species_choices({}) == {"__all__": "All species"}


def test_overrides_reach_the_in_process_engines_whatever_their_case():
    """The config reader lowercases every key and the Python and C# engines read the lowercase
    form. An override typed the way config files spell keys (``species.lInf.sp0``) must replace
    the loaded value, not sit beside it under another spelling, where nothing reads it."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from ui.pages.run import effective_run_config

    root = Path(__file__).resolve().parents[1]
    config = dict(OsmoseConfigReader().read(root / "data" / "minimal" / "osm_all-parameters.csv"))
    years = int(config["simulation.time.nyear"])
    linf = config["species.linf.sp0"]
    merged = effective_run_config(
        config, f"simulation.time.nYear = {years + 3}\nSpecies.lInf.sp0={float(linf) + 1.5}\n"
    )
    assert merged["simulation.time.nyear"] == str(years + 3)
    assert merged["species.linf.sp0"] == str(float(linf) + 1.5)
    assert len(merged) == len(config), "the override was added as a second key"
    engine_config = EngineConfig.from_dict(merged)
    # what the engine runs
    assert engine_config.n_steps == (years + 3) * engine_config.n_dt_per_year
    assert float(engine_config.linf[0]) == float(linf) + 1.5
    assert config["simulation.time.nyear"] == str(years)  # the loaded config is not modified


def test_java_overrides_keep_their_case():
    """The Java engine is case-sensitive: its overrides go to the jar as typed. Its call site
    parses them directly and does not go through the in-process merge, which lowercases."""
    assert parse_overrides("species.lInf.sp0 = 120") == {"species.lInf.sp0": "120"}
    source = (Path(__file__).resolve().parents[1] / "ui" / "pages" / "run.py").read_text(
        encoding="utf-8"
    )
    assert 'parse_overrides(input.param_overrides() or "")' in source
    assert "effective_run_config(config, input.param_overrides()" not in source
