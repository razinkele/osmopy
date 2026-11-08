"""Baltic spawning inputs match the literature decisions of 2026-11-08 (review C1, C2).

* Eastern cod spawn only in SD 25 (Eero et al. 2012): the committed map is exactly what
  scripts/build_cod_east_sd25_spawning.py derives, and every cell is in the SD 25 box.
* Herring have no autumn spawning component: the season file carries no Sep-Nov mass and still
  sums to 1, and no herring movement map points at the autumn spawning map.
"""

import importlib.util
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader

ROOT = Path(__file__).resolve().parent.parent
BALTIC = ROOT / "data" / "baltic"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_cod_east_sd25_spawning", ROOT / "scripts" / "build_cod_east_sd25_spawning.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cod_east_spawning_map_is_the_builder_output():
    b = _builder()
    committed = b.read_map(b.SPAWN) > 0
    assert np.array_equal(committed, b.spawning_cells())
    assert committed.sum() > 0


def test_cod_east_spawns_only_in_sd25():
    b = _builder()
    rows, cols = np.nonzero(b.read_map(b.SPAWN) > 0)
    lat = b.TOP - (rows + 0.5) * b.DLAT
    lon = b.LEFT + (cols + 0.5) * b.DLON
    assert ((lat >= 54.5) & (lat <= 56.2) & (lon >= 14.0) & (lon <= 18.0)).all()


def test_every_baltic_map_is_oriented_to_the_grid():
    """Each map, read by the engine's loader, has its land (-99) exactly where the engine's grid
    has land, and no positive cell on land. baltic_grid.nc is north-first while the loader
    follows the OSMOSE convention (CSV row 0 -> grid row ny-1), so Baltic maps are stored
    south-first; a map written the other way up fails here rather than putting fish on land."""
    from osmose.engine.grid import Grid
    from osmose.engine.movement_maps import _load_csv_grid

    grid = Grid.from_netcdf(BALTIC / "baltic_grid.nc")
    maps = sorted((BALTIC / "maps").glob("*.csv"))
    assert len(maps) > 20
    for path in maps:
        m = _load_csv_grid(path, grid.ny, grid.nx)
        assert np.array_equal(m == -99, ~grid.ocean_mask), path.name
        assert not ((m > 0) & ~grid.ocean_mask).any(), path.name


def test_herring_has_no_autumn_spawning():
    lines = (BALTIC / "reproduction" / "reproduction-seasonality-sp1.csv").read_text().splitlines()
    rows = [ln.replace('"', "").split(";") for ln in lines[1:] if ln.strip()]
    t = np.array([float(r[0]) for r in rows])
    v = np.array([float(r[1]) for r in rows])
    assert abs(v.sum() - 1.0) < 1e-6
    assert (v[t >= 0.5] == 0).all()
    cfg = dict(OsmoseConfigReader().read(str(BALTIC / "baltic_all-parameters.csv")))
    herring_maps = [
        k[len("movement.species.") :]
        for k, val in cfg.items()
        if k.startswith("movement.species.map") and val.strip().lower() == "herring"
    ]
    files = [cfg.get(f"movement.file.{m}", "") for m in herring_maps]
    assert herring_maps and not any("autumn" in f for f in files), files
