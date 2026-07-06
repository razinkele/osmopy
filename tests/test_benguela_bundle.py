from pathlib import Path
import numpy as np
import xarray as xr

from osmose.engine.movement_maps import _load_csv_grid
from osmose.engine.grid import Grid

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "benguela"
MERGED = BUNDLE / "input" / "roms_climatological_merged.nc"
RES_VARS = ["sphy", "lphy", "szoo", "lzoo"]

MAPS_DIR = BUNDLE / "maps"
MOVE_KEYS = BUNDLE / "_movement_keys.txt"
GRID_NC = BUNDLE / "input" / "grid-mask.nc"
BEN_SPECIES = [
    "euphausiids",
    "anchovy",
    "sardine",
    "redeye",
    "horsemackerel",
    "mesopelagic",
    "silverkob",
    "snoek",
    "shallowwaterhake",
    "deepwaterhake",
]


def test_merged_forcing_has_all_four_resources():
    assert MERGED.exists(), "run scripts/merge_benguela_forcing.py <SRC>"
    ds = xr.open_dataset(MERGED)
    try:
        for v in RES_VARS:
            assert v in ds.data_vars, f"{v} missing from merged forcing"
            assert ds[v].shape == (24, 62, 56), f"{v} wrong dims {ds[v].shape}"
            assert float(np.nansum(ds[v].values)) > 0, f"{v} is all-zero/NaN"
    finally:
        ds.close()


def _load_movement_keys():
    d = {}
    for ln in MOVE_KEYS.read_text().splitlines():
        if ";" in ln:
            k, v = ln.split(";", 1)
            d[k.strip().lower()] = v.strip()
    return d


def test_movement_maps_converted_and_wired():
    assert MOVE_KEYS.exists(), "run scripts/convert_benguela_maps.py <SRC>"
    keys = _load_movement_keys()
    idxs = sorted({int(k.split(".map")[1]) for k in keys if k.startswith("movement.species.map")})
    assert idxs, "no movement.species.mapN emitted (species would be orphaned -> is_out)"
    for n in idxs:
        assert keys[f"movement.species.map{n}"] in BEN_SPECIES
        assert (BUNDLE / keys[f"movement.file.map{n}"]).exists()
        assert f"movement.steps.map{n}" in keys
        assert f"movement.initialage.map{n}" in keys
        assert f"movement.lastage.map{n}" in keys


def test_movement_csv_loads_ocean_within_grid_via_real_loader():
    # Load via the PRODUCTION loader (which flips rows) + the real grid ocean mask.
    assert GRID_NC.exists(), "run scripts/build_benguela_config.py to copy grid-mask.nc"
    ocean = Grid.from_netcdf(str(GRID_NC)).ocean_mask  # (62,56) bool, engine convention
    for csv in sorted(MAPS_DIR.glob("*.csv")):
        grid = _load_csv_grid(str(csv), 62, 56)
        present = (grid > 0) & (grid != -99)
        assert present[~ocean].sum() == 0, f"{csv.name} places presence on land (flip/orientation)"
