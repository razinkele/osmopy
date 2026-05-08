"""Rebuild Baltic pikeperch maps and salinity provenance layers.

Pikeperch in this coarse Baltic setup is treated as a coastal brackish-water
perciform: adults remain coastal, juveniles/recruits use sheltered low-salinity
bay habitat, and spawning is an April-June inner-bay event. The HELCOM
Pan Baltic Scope layer available for pikeperch is a recruitment layer rather
than an adult abundance layer, so these maps combine the existing coastal perch
proxy with a simple surface-salinity constraint and keep salinity as explicit
environment/provenance CSVs.

Run from repo root:
    .venv/bin/python scripts/rebuild_baltic_pikeperch_salinity_maps.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/baltic")
GRID_MASK = DATA_DIR / "grid" / "baltic_mask.csv"
MAPS_DIR = DATA_DIR / "maps"
ENV_DIR = DATA_DIR / "environment"

N_LAT = 40
N_LON = 50
MIN_LAT = 54.0
MAX_LAT = 66.0
MIN_LON = 10.0
MAX_LON = 30.0

PIKEPERCH_SURFACE_SALINITY_MAX = 6.5
SOUTHERN_LAGOON_SURFACE_SALINITY = 4.5
SOUTHERN_LAGOON_BOXES = (
    # name, min_lat, max_lat, min_lon, max_lon
    ("Oder/Szczecin Lagoon", 53.9, 54.8, 13.5, 14.7),
    ("Vistula Lagoon/Gulf of Gdansk", 54.2, 54.8, 18.8, 20.2),
    ("Curonian Lagoon", 54.9, 56.0, 20.8, 21.8),
)


def _read_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=";", header=None).values


def _write_csv(path: Path, arr: np.ndarray) -> None:
    if np.issubdtype(arr.dtype, np.integer):
        df = pd.DataFrame(arr.astype(int))
    else:
        df = pd.DataFrame(np.round(arr.astype(float), 1))
    df.to_csv(path, sep=";", header=False, index=False)


def _lat_lon_centres() -> tuple[np.ndarray, np.ndarray]:
    lat_edges = np.linspace(MIN_LAT, MAX_LAT, N_LAT + 1)
    lon_edges = np.linspace(MIN_LON, MAX_LON, N_LON + 1)
    return (lat_edges[:-1] + lat_edges[1:]) / 2, (lon_edges[:-1] + lon_edges[1:]) / 2


def _surface_salinity(lat: float, lon: float) -> float:
    """Coarse Baltic surface salinity proxy in practical-salinity units.

    The gradient follows the expected southwest-to-northeast freshening and is
    only used to classify broad low-salinity coastal habitat on this 0.3 x 0.4
    degree grid; it is not intended as an observation-derived hydrographic field.
    """
    salinity = 8.5 - 0.28 * (lat - MIN_LAT) - 0.10 * (lon - MIN_LON)

    if lat >= 59.0 and lon >= 24.0:  # Gulf of Finland
        salinity -= 1.5
    if 56.5 <= lat < 59.0 and 22.0 <= lon < 25.5:  # Gulf of Riga
        salinity -= 1.0
    if lat >= 61.5:  # Bothnian Sea and northward
        salinity -= 1.2
    if lat >= 64.0:  # Bothnian Bay
        salinity -= 1.5
    if 59.0 <= lat < 61.5 and 20.0 <= lon < 24.0:  # Archipelago Sea
        salinity -= 0.8

    return float(np.clip(salinity, 1.0, 9.0))


def _southern_lagoon_cell(lat: float, lon: float) -> bool:
    return any(
        min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
        for _, min_lat, max_lat, min_lon, max_lon in SOUTHERN_LAGOON_BOXES
    )


def southern_lagoon_mask(ocean: np.ndarray) -> np.ndarray:
    lat_centres, lon_centres = _lat_lon_centres()
    mask = np.zeros((N_LAT, N_LON), dtype=bool)
    for row, lat in enumerate(lat_centres):
        for col, lon in enumerate(lon_centres):
            mask[row, col] = ocean[row, col] and _southern_lagoon_cell(float(lat), float(lon))
    return mask


def build_salinity_layers(ocean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_centres, lon_centres = _lat_lon_centres()
    surface = np.full((N_LAT, N_LON), -99.0)
    bottom = np.full((N_LAT, N_LON), -99.0)

    for row, lat in enumerate(lat_centres):
        for col, lon in enumerate(lon_centres):
            if not ocean[row, col]:
                continue
            surface_value = _surface_salinity(float(lat), float(lon))
            if _southern_lagoon_cell(float(lat), float(lon)):
                surface_value = min(surface_value, SOUTHERN_LAGOON_SURFACE_SALINITY)
            deepening = 0.7
            if lat < 56.5 and lon < 18.5:
                deepening = 2.0
            elif lat < 59.0 and lon < 23.0:
                deepening = 1.3
            elif lat >= 61.5:
                deepening = 0.3
            surface[row, col] = surface_value
            bottom[row, col] = min(13.0, surface_value + deepening)

    return surface, bottom


def build_pikeperch_maps(surface_salinity: np.ndarray, ocean: np.ndarray) -> dict[str, np.ndarray]:
    low_salinity = (surface_salinity > -90.0) & (surface_salinity <= PIKEPERCH_SURFACE_SALINITY_MAX)
    southern_lagoons = southern_lagoon_mask(ocean)

    stage_sources = {
        "juvenile": _read_csv(MAPS_DIR / "perch_juvenile.csv") > 0,
        "adult": _read_csv(MAPS_DIR / "perch_adult.csv") > 0,
        "spawning": _read_csv(MAPS_DIR / "perch_spawning.csv") > 0,
    }

    maps: dict[str, np.ndarray] = {}
    for stage, source in stage_sources.items():
        active = ((source & low_salinity) | southern_lagoons) & ocean
        maps[stage] = np.where(ocean, active.astype(int), -99)
    return maps


def main() -> None:
    ENV_DIR.mkdir(parents=True, exist_ok=True)

    ocean = _read_csv(GRID_MASK) == 0
    surface, bottom = build_salinity_layers(ocean)
    pikeperch_maps = build_pikeperch_maps(surface, ocean)

    _write_csv(ENV_DIR / "surface_salinity_mean.csv", surface)
    _write_csv(ENV_DIR / "bottom_salinity_mean.csv", bottom)

    for stage, arr in pikeperch_maps.items():
        _write_csv(MAPS_DIR / f"pikeperch_{stage}.csv", arr)

    print("Wrote salinity layers:")
    print(f"  {ENV_DIR / 'surface_salinity_mean.csv'}")
    print(f"  {ENV_DIR / 'bottom_salinity_mean.csv'}")
    print("Wrote pikeperch maps:")
    for stage, arr in pikeperch_maps.items():
        print(f"  pikeperch_{stage}.csv active cells: {int((arr > 0).sum())}")


if __name__ == "__main__":
    main()
