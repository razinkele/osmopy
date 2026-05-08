"""Static checks for Baltic pikeperch salinity-constrained maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALTIC = PROJECT_ROOT / "data" / "baltic"
MAPS = BALTIC / "maps"
ENV = BALTIC / "environment"
MAX_PIKEPERCH_SALINITY = 6.5


def _read_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=";", header=None).values


def test_salinity_layers_exist_and_match_grid_mask():
    mask = _read_csv(BALTIC / "grid" / "baltic_mask.csv")

    for name in ("surface_salinity_mean.csv", "bottom_salinity_mean.csv"):
        salinity = _read_csv(ENV / name)
        assert salinity.shape == mask.shape == (40, 50)
        np.testing.assert_array_equal(salinity == -99, mask == -99)
        assert np.nanmin(salinity[salinity > -90]) > 0


def test_pikeperch_maps_are_low_salinity_coastal_subsets():
    surface_salinity = _read_csv(ENV / "surface_salinity_mean.csv")
    stage_counts = {
        "juvenile": 67,
        "adult": 82,
        "spawning": 62,
    }

    for stage, expected_count in stage_counts.items():
        pikeperch = _read_csv(MAPS / f"pikeperch_{stage}.csv") > 0
        perch = _read_csv(MAPS / f"perch_{stage}.csv") > 0
        southern_lagoons = _southern_lagoon_mask()

        assert int(pikeperch.sum()) == expected_count
        assert np.all((perch | southern_lagoons)[pikeperch])
        assert float(surface_salinity[pikeperch].max()) <= MAX_PIKEPERCH_SALINITY


def _southern_lagoon_mask() -> np.ndarray:
    lat_edges = np.linspace(54.0, 66.0, 41)
    lon_edges = np.linspace(10.0, 30.0, 51)
    lat_centres = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centres = (lon_edges[:-1] + lon_edges[1:]) / 2
    mask = np.zeros((40, 50), dtype=bool)

    boxes = (
        (53.9, 54.8, 13.5, 14.7),  # Oder/Szczecin Lagoon
        (54.2, 54.8, 18.8, 20.2),  # Vistula Lagoon/Gulf of Gdansk
        (54.9, 56.0, 20.8, 21.8),  # Curonian Lagoon
    )
    for row, lat in enumerate(lat_centres):
        for col, lon in enumerate(lon_centres):
            mask[row, col] = any(
                min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
                for min_lat, max_lat, min_lon, max_lon in boxes
            )
    return mask


def test_pikeperch_maps_keep_southern_lagoons_but_not_southern_open_baltic():
    lat_edges = np.linspace(54.0, 66.0, 41)
    lat_centres = (lat_edges[:-1] + lat_edges[1:]) / 2
    southern_rows = lat_centres < 56.5
    southern_lagoons = _southern_lagoon_mask()

    for stage in ("juvenile", "adult", "spawning"):
        pikeperch = _read_csv(MAPS / f"pikeperch_{stage}.csv") > 0
        assert int((pikeperch & southern_lagoons).sum()) == 20
        assert not (pikeperch & southern_rows[:, None] & ~southern_lagoons).any()
