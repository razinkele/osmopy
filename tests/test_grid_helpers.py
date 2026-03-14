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
        ul_lat=45.0, ul_lon=-2.0,
        lr_lat=43.0, lr_lon=0.0,
        nx=2, ny=2,
        mask=None,
    )
    assert isinstance(layers, list)
    assert len(layers) > 0


def test_build_grid_layers_zero_coords_returns_empty():
    from ui.pages.grid_helpers import build_grid_layers

    layers = build_grid_layers(
        ul_lat=0.0, ul_lon=0.0,
        lr_lat=0.0, lr_lon=0.0,
        nx=2, ny=2,
        mask=None,
    )
    assert layers == []


def test_build_grid_layers_with_mask():
    from ui.pages.grid_helpers import build_grid_layers

    mask = np.array([[1, 0], [0, 1]])
    layers = build_grid_layers(
        ul_lat=45.0, ul_lon=-2.0,
        lr_lat=43.0, lr_lon=0.0,
        nx=2, ny=2,
        mask=mask,
    )
    assert isinstance(layers, list)
    # Should have boundary + ocean cells + land cells layers
    assert len(layers) >= 1


def test_build_grid_layers_dark_mode():
    from ui.pages.grid_helpers import build_grid_layers

    layers_dark = build_grid_layers(
        ul_lat=45.0, ul_lon=-2.0,
        lr_lat=43.0, lr_lon=0.0,
        nx=2, ny=2,
        is_dark=True,
        mask=None,
    )
    layers_light = build_grid_layers(
        ul_lat=45.0, ul_lon=-2.0,
        lr_lat=43.0, lr_lon=0.0,
        nx=2, ny=2,
        is_dark=False,
        mask=None,
    )
    # Both should produce layers
    assert len(layers_dark) > 0
    assert len(layers_light) > 0


def test_build_grid_layers_zero_nx_ny():
    from ui.pages.grid_helpers import build_grid_layers

    layers = build_grid_layers(
        ul_lat=45.0, ul_lon=-2.0,
        lr_lat=43.0, lr_lon=0.0,
        nx=0, ny=0,
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
