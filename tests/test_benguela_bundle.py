from pathlib import Path
import numpy as np
import xarray as xr

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine import PythonEngine
from osmose.engine.movement_maps import _load_csv_grid
from osmose.engine.grid import Grid

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "benguela"
MERGED = BUNDLE / "input" / "roms_climatological_merged.nc"
RES_VARS = ["sphy", "lphy", "szoo", "lzoo"]

MAPS_DIR = BUNDLE / "maps"
MOVE_KEYS = BUNDLE / "_movement_keys.txt"
GRID_NC = BUNDLE / "input" / "grid-mask.nc"
SEED_KEYS = BUNDLE / "_seeding_keys.txt"
MASTER = BUNDLE / "benguela_all-parameters.csv"
EXPECTED_SEED = {
    0: 3129213,
    1: 3888750,
    2: 3029155,
    3: 1286364,
    4: 1138339,
    5: 1439984,
    6: 198865,
    7: 81054,
    8: 575361,
    9: 591907,
}
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


def test_seeding_block_matches_authors_values():
    assert SEED_KEYS.exists(), "run scripts/derive_benguela_seeding.py <SRC>"
    got = {}
    for ln in SEED_KEYS.read_text().splitlines():
        if "seeding.biomass.sp" in ln:
            k, v = ln.split(";", 1)
            got[int(k.strip().split(".sp")[1])] = float(v)
    assert set(got) == set(range(10))
    for sp, exp in EXPECTED_SEED.items():
        assert abs(got[sp] - exp) < 1.0, f"sp{sp} seed {got[sp]} != {exp}"
        assert got[sp] > 0


def test_master_loads_and_is_wired():
    assert MASTER.exists(), "run scripts/build_benguela_config.py <SRC>"
    raw = dict(OsmoseConfigReader().read(str(MASTER)))
    for sp in range(300, 304):
        assert raw[f"species.file.sp{sp}"].endswith("roms_climatological_merged.nc")
    assert "fisheries.catchability.file" not in raw
    assert "fisheries.discards.file" not in raw
    assert not any(k.startswith("fisheries.movement.") for k in raw)
    assert raw["simulation.nfisheries"] == "0"
    assert "population.initialization.file" not in raw
    assert float(raw["population.seeding.biomass.sp1"]) == 3888750
    assert raw["output.file.prefix"] == "benguela"


def test_master_loads_without_fisheries_dir():
    raw = dict(OsmoseConfigReader().read(str(MASTER)))
    cfg = EngineConfig.from_dict(raw)  # raises if any _require_file is unmet
    assert cfg.n_species == 10
    assert not (BUNDLE / "input" / "fisheries").exists()


def test_master_runs_without_nan_integration_smoke():
    # nyear=1 engine run: catches the class of integration failure (mis-oriented maps -> NaN) at
    # Task 4's own gate, not only at Task 5. Spike-confirmed to be NaN-free with correct maps.
    raw = dict(OsmoseConfigReader().read(str(MASTER)))
    raw["simulation.time.nyear"] = "1"
    res = PythonEngine().run_in_memory(raw, seed=42)
    b = res.biomass()
    cols = [c for c in b.columns if c not in ("Time", "time", "species")]
    v = b[cols].to_numpy(dtype=float)
    assert not np.isnan(v).any(), "integration produced NaN biomass (check map orientation flip)"
