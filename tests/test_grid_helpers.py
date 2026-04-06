"""Tests for grid helper functions."""

import numpy as np


def test_load_mask_valid_csv(tmp_path):
    from ui.pages.grid_helpers import load_mask

    mask_file = tmp_path / "mask.csv"
    mask_file.write_text("0,1,1\n1,0,1\n1,1,0\n")
    config = {"grid.mask.file": "mask.csv"}
    result = load_mask(config, config_dir=tmp_path)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)


def test_load_mask_missing_file(tmp_path):
    from ui.pages.grid_helpers import load_mask

    config = {"grid.mask.file": "nonexistent.csv"}
    result = load_mask(config, config_dir=tmp_path)
    assert result is None


def test_load_mask_empty_string():
    from ui.pages.grid_helpers import load_mask

    config = {"grid.mask.file": ""}
    result = load_mask(config, config_dir=None)
    assert result is None


def test_load_mask_missing_key():
    from ui.pages.grid_helpers import load_mask

    # No grid.mask.file key at all
    result = load_mask({}, config_dir=None)
    assert result is None


def test_load_mask_no_config_dir(tmp_path):
    from ui.pages.grid_helpers import load_mask

    config = {"grid.mask.file": "somefile.csv"}
    # config_dir=None and file not in examples dir -> None
    result = load_mask(config, config_dir=None)
    assert result is None


def test_build_grid_layers_basic():
    from ui.pages.grid_helpers import build_grid_layers

    layers = build_grid_layers(
        ul_lat=45.0,
        ul_lon=-2.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=2,
        ny=2,
        mask=None,
    )
    assert isinstance(layers, list)
    assert len(layers) > 0


def test_build_grid_layers_zero_coords_returns_empty():
    from ui.pages.grid_helpers import build_grid_layers

    layers = build_grid_layers(
        ul_lat=0.0,
        ul_lon=0.0,
        lr_lat=0.0,
        lr_lon=0.0,
        nx=2,
        ny=2,
        mask=None,
    )
    assert layers == []


def test_build_grid_layers_with_mask():
    from ui.pages.grid_helpers import build_grid_layers

    mask = np.array([[1, 0], [0, 1]])
    layers = build_grid_layers(
        ul_lat=45.0,
        ul_lon=-2.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=2,
        ny=2,
        mask=mask,
    )
    assert isinstance(layers, list)
    # Should have boundary + ocean cells + land cells layers
    assert len(layers) >= 1


def test_build_grid_layers_dark_mode():
    from ui.pages.grid_helpers import build_grid_layers

    layers_dark = build_grid_layers(
        ul_lat=45.0,
        ul_lon=-2.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=2,
        ny=2,
        is_dark=True,
        mask=None,
    )
    layers_light = build_grid_layers(
        ul_lat=45.0,
        ul_lon=-2.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=2,
        ny=2,
        is_dark=False,
        mask=None,
    )
    # Both should produce layers
    assert len(layers_dark) > 0
    assert len(layers_light) > 0


def test_build_grid_layers_zero_nx_ny():
    from ui.pages.grid_helpers import build_grid_layers

    layers = build_grid_layers(
        ul_lat=45.0,
        ul_lon=-2.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=0,
        ny=0,
        mask=None,
    )
    # Only boundary layer, no cell layers
    assert isinstance(layers, list)
    assert len(layers) == 1


def test_build_netcdf_grid_layers_1d():
    from ui.pages.grid_helpers import build_netcdf_grid_layers

    lat = np.array([40.0, 41.0, 42.0])
    lon = np.array([1.0, 2.0, 3.0])
    mask = np.ones((3, 3))
    result = build_netcdf_grid_layers(lat, lon, mask=mask)
    assert isinstance(result, tuple)
    layers, view_state = result
    assert isinstance(layers, list)
    assert len(layers) > 0
    assert "latitude" in view_state
    assert "longitude" in view_state
    assert "zoom" in view_state


def test_build_netcdf_grid_layers_2d():
    from ui.pages.grid_helpers import build_netcdf_grid_layers

    lon2d, lat2d = np.meshgrid([1.0, 2.0], [40.0, 41.0])
    mask = np.ones((2, 2))
    result = build_netcdf_grid_layers(lat2d, lon2d, mask=mask)
    assert isinstance(result, tuple)
    layers, view_state = result
    assert isinstance(layers, list)
    assert len(layers) > 0


def test_build_netcdf_grid_layers_with_land():
    from ui.pages.grid_helpers import build_netcdf_grid_layers

    lat = np.array([40.0, 41.0])
    lon = np.array([1.0, 2.0])
    # mask with zeros = land cells
    mask = np.array([[1, 0], [0, 1]])
    result = build_netcdf_grid_layers(lat, lon, mask=mask)
    layers, view_state = result
    # Should have boundary + ocean + land layers
    assert len(layers) >= 2


def test_build_netcdf_grid_layers_all_land():
    from ui.pages.grid_helpers import build_netcdf_grid_layers

    lat = np.array([40.0, 41.0])
    lon = np.array([1.0, 2.0])
    mask = np.zeros((2, 2))  # all land
    result = build_netcdf_grid_layers(lat, lon, mask=mask)
    layers, view_state = result
    # boundary + land layer
    assert isinstance(layers, list)
    assert len(layers) >= 1


def test_build_netcdf_grid_layers_view_state_values():
    from ui.pages.grid_helpers import build_netcdf_grid_layers

    lat = np.array([40.0, 42.0])
    lon = np.array([1.0, 3.0])
    mask = np.ones((2, 2))
    _, view_state = build_netcdf_grid_layers(lat, lon, mask=mask)
    # Center should be roughly in the middle
    assert 40.0 <= view_state["latitude"] <= 42.0
    assert 1.0 <= view_state["longitude"] <= 3.0
    assert view_state["zoom"] >= 1


def test_derive_map_label_spawning():
    from ui.pages.grid_helpers import derive_map_label

    assert derive_map_label("maps/6cod_spawning.csv", 17) == "Spawning"


def test_derive_map_label_multiword():
    from ui.pages.grid_helpers import derive_map_label

    assert derive_map_label("maps/3tacaud_spawners_printemps.csv", 5) == "Spawners Printemps"


def test_derive_map_label_numeric_fallback():
    from ui.pages.grid_helpers import derive_map_label

    assert derive_map_label("maps/1Roussette_01.csv", 0) == "Map 0"


def test_derive_map_label_no_underscore():
    from ui.pages.grid_helpers import derive_map_label

    assert derive_map_label("maps/empty.csv", 3) == "Map 3"


def test_derive_map_label_1plus():
    from ui.pages.grid_helpers import derive_map_label

    assert derive_map_label("maps/6cod_1plus.csv", 16) == "1Plus"


def test_parse_movement_steps_basic():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps("0;1;2;3") == {0, 1, 2, 3}


def test_parse_movement_steps_trailing_semicolon():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps("6;7;8;") == {6, 7, 8}


def test_parse_movement_steps_whitespace():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps(" 0 ; 1 ; 2 ") == {0, 1, 2}


def test_parse_movement_steps_empty():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps("") == set()


def test_parse_movement_steps_none():
    from ui.pages.grid_helpers import parse_movement_steps

    assert parse_movement_steps(None) == set()


def test_build_movement_cache_basic(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "1sp_nurseries.csv").write_text("0,1,0\n1,0,1\n0,1,0\n")
    (tmp_path / "maps" / "1sp_spawning.csv").write_text("1,0,1\n0,1,0\n1,0,1\n")

    cfg = {
        "movement.species.map0": "speciesA",
        "movement.file.map0": "maps/1sp_nurseries.csv",
        "movement.steps.map0": "0;1;2;3",
        "movement.initialAge.map0": "0",
        "movement.lastAge.map0": "1",
        "movement.species.map1": "speciesA",
        "movement.file.map1": "maps/1sp_spawning.csv",
        "movement.steps.map1": "4;5;6;7",
        "movement.initialAge.map1": "1",
        "movement.lastAge.map1": "5",
    }
    grid_params = (48.0, -6.0, 43.0, -1.0, 3, 3)

    cache = build_movement_cache(cfg, tmp_path, grid_params, species="speciesA")
    assert len(cache) == 2
    assert "map0" in cache
    assert "map1" in cache
    assert cache["map0"]["steps"] == {0, 1, 2, 3}
    assert cache["map1"]["steps"] == {4, 5, 6, 7}
    assert cache["map0"]["label"] == "Nurseries"
    assert cache["map1"]["label"] == "Spawning"
    assert cache["map0"]["age_range"] == "0-1 yr"
    assert cache["map1"]["age_range"] == "1-5 yr"
    assert cache["map0"]["cells"] is not None
    assert cache["map1"]["cells"] is not None
    assert cache["map0"]["color"] != cache["map1"]["color"]


def test_build_movement_cache_no_maps():
    from ui.pages.grid_helpers import build_movement_cache

    cfg = {"simulation.nspecies": "3"}
    cache = build_movement_cache(cfg, None, (0, 0, 0, 0, 10, 10), species="cod")
    assert cache == {}


def test_build_movement_cache_missing_file(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    cfg = {
        "movement.species.map0": "cod",
        "movement.file.map0": "maps/nonexistent.csv",
        "movement.steps.map0": "0;1",
        "movement.initialAge.map0": "0",
        "movement.lastAge.map0": "2",
    }
    cache = build_movement_cache(cfg, tmp_path, (48.0, -6.0, 43.0, -1.0, 3, 3), species="cod")
    assert cache == {}


def test_build_movement_cache_null_file(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    cfg = {
        "movement.species.map0": "cod",
        "movement.file.map0": "null",
        "movement.steps.map0": "0;1",
        "movement.initialAge.map0": "0",
        "movement.lastAge.map0": "2",
    }
    cache = build_movement_cache(cfg, tmp_path, (48.0, -6.0, 43.0, -1.0, 3, 3), species="cod")
    assert cache == {}


def test_build_movement_cache_color_cycling(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    (tmp_path / "maps").mkdir()
    cfg = {}
    for i in range(9):
        fname = f"maps/1sp_map{i}.csv"
        (tmp_path / fname).write_text("0,1,0\n1,0,1\n0,1,0\n")
        cfg[f"movement.species.map{i}"] = "cod"
        cfg[f"movement.file.map{i}"] = fname
        cfg[f"movement.steps.map{i}"] = str(i)
        cfg[f"movement.initialAge.map{i}"] = "0"
        cfg[f"movement.lastAge.map{i}"] = "10"

    cache = build_movement_cache(cfg, tmp_path, (48.0, -6.0, 43.0, -1.0, 3, 3), species="cod")
    assert len(cache) == 9
    assert cache["map0"]["color"][:3] == cache["map8"]["color"][:3]


def test_build_movement_cache_filters_species(tmp_path):
    from ui.pages.grid_helpers import build_movement_cache

    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "a.csv").write_text("0,1,0\n1,0,1\n0,1,0\n")
    (tmp_path / "maps" / "b.csv").write_text("1,0,1\n0,1,0\n1,0,1\n")

    cfg = {
        "movement.species.map0": "cod",
        "movement.file.map0": "maps/a.csv",
        "movement.steps.map0": "0;1",
        "movement.initialAge.map0": "0",
        "movement.lastAge.map0": "2",
        "movement.species.map1": "herring",
        "movement.file.map1": "maps/b.csv",
        "movement.steps.map1": "2;3",
        "movement.initialAge.map1": "0",
        "movement.lastAge.map1": "5",
    }
    cache = build_movement_cache(cfg, tmp_path, (48.0, -6.0, 43.0, -1.0, 3, 3), species="cod")
    assert len(cache) == 1
    assert "map0" in cache


def test_list_movement_species():
    from ui.pages.grid_helpers import list_movement_species

    cfg = {
        "movement.species.map0": "cod",
        "movement.species.map1": "cod",
        "movement.species.map2": "herring",
        "movement.species.map3": "sole",
    }
    result = list_movement_species(cfg)
    assert result == ["cod", "herring", "sole"]


def test_list_movement_species_empty():
    from ui.pages.grid_helpers import list_movement_species

    assert list_movement_species({}) == []


def test_make_spatial_map_importable():
    from ui.pages.grid_helpers import make_spatial_map
    assert callable(make_spatial_map)


def test_make_legend_importable():
    from ui.pages.grid_helpers import make_legend
    assert callable(make_legend)
