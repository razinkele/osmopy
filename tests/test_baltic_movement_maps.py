"""Baltic movement maps: orientation, and herring's autumn packing stays removed (2026-11-08).

* Every map, read by the engine's loader, sits on the engine grid's ocean. baltic_grid.nc is
  north-first while the loader follows the OSMOSE convention (CSV row 0 -> grid row ny-1, which is
  north on a south-first grid such as EEC's), so Baltic maps are stored south-first.
* Herring adults are no longer moved onto the 10-cell autumn map for Sep-Nov (review C1), and
  every herring (age, step) slot still has a map, so removing it cannot strand fish out of domain.
"""

from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader

ROOT = Path(__file__).resolve().parent.parent
BALTIC = ROOT / "data" / "baltic"


def _config() -> dict[str, str]:
    return dict(OsmoseConfigReader().read(str(BALTIC / "baltic_all-parameters.csv")))


def test_every_baltic_map_is_oriented_to_the_grid():
    """Land (-99) exactly where the engine's grid has land, and no positive cell on land: a map
    written the other way up fails here, by name, rather than putting fish on land."""
    from osmose.engine.grid import Grid
    from osmose.engine.movement_maps import _load_csv_grid

    grid = Grid.from_netcdf(BALTIC / "baltic_grid.nc")
    maps = sorted((BALTIC / "maps").glob("*.csv"))
    assert len(maps) > 20
    for path in maps:
        m = _load_csv_grid(path, grid.ny, grid.nx)
        assert np.array_equal(m == -99, ~grid.ocean_mask), path.name
        assert not ((m > 0) & ~grid.ocean_mask).any(), path.name


def test_herring_adults_are_not_packed_onto_the_autumn_map():
    cfg = _config()
    herring_maps = [
        k[len("movement.species.") :]
        for k, v in cfg.items()
        if k.startswith("movement.species.map") and v.strip().lower() == "herring"
    ]
    files = [cfg.get(f"movement.file.{m}", "") for m in herring_maps]
    assert herring_maps and not any("autumn" in f for f in files), files


def test_every_herring_slot_has_a_map():
    """strict=True raises on any (age_dt, step) slot without a map."""
    from osmose.engine.config import EngineConfig
    from osmose.engine.movement_maps import MovementMapSet

    ec = EngineConfig.from_dict(_config())
    sp = ec.species_names.index("herring")
    MovementMapSet(
        config=ec.raw_config,
        species_name="herring",
        n_dt_per_year=ec.n_dt_per_year,
        n_years=1,
        lifespan_dt=int(ec.lifespan_dt[sp]),
        ny=40,
        nx=50,
        config_dir=ec.raw_config.get("_osmose.config.dir", ""),
        strict=True,
    )


def test_herring_season_has_no_autumn_eggs_and_sums_to_one():
    """The autumn egg share is folded into spring (sourced: autumn spawners are about 1% or less of
    Baltic landings today; docs/baltic_spawning_fixes_2026-11-08.md). Annual egg output is
    preserved: the season still sums to 1."""
    lines = (BALTIC / "reproduction" / "reproduction-seasonality-sp1.csv").read_text().splitlines()
    rows = [ln.replace('"', "").split(";") for ln in lines[1:] if ln.strip()]
    t_yr = np.array([float(r[0]) for r in rows])
    share = np.array([float(r[1]) for r in rows])
    assert abs(share.sum() - 1.0) < 1e-6
    assert (share[t_yr >= 0.5] == 0).all()
