#!/usr/bin/env python
"""Build salinity-niched distribution maps for cod_west and cod_east (Phase 1 Task 3).

Splits the aggregate cod distribution along longitude into a western stock
(SD22-24 Belt/Arkona, cols <= 14, ~lon <= 15.6E) and an eastern stock (SD25-32
deep basins Bornholm/Gdansk/Gotland, cols >= 13, ~lon >= 15.2E). SD24 (Arkona,
cols 13-14) is a deliberate shared transition — the documented mixing zone.

  cod_west_{juvenile,adult}   = aggregate cod maps masked to western cols
  cod_west_spawning           = western adult footprint (western cod spawn in
                                SD22-24; the aggregate spawning map is eastern-biased)
  cod_east_{juvenile,adult}   = aggregate cod maps masked to eastern cols
  cod_east_spawning           = aggregate cod SPAWNING map masked east — the
                                deep-basin reproduction grounds the RV gate acts on

Movement wiring: cod's map0-3 -> cod_west (spring spawning window, steps 2-9);
new map26-29 -> cod_east (summer spawning window, steps 10-19; lifespan 15).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
CONFIG_DIR = _REPO / "data" / "baltic"
MAPS_DIR = CONFIG_DIR / "maps"

spec = importlib.util.spec_from_file_location("apply_calibration", _HERE / "apply_calibration.py")
apply_calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_calibration)
set_key = apply_calibration.set_key

WEST_COLS = set(range(0, 15))  # cols <= 14  (SD22-24, incl. Arkona transition)
EAST_COLS = set(range(13, 50))  # cols >= 13  (SD24 transition + deep basins east)


def _load(name: str) -> np.ndarray:
    return np.genfromtxt(MAPS_DIR / name, delimiter=";")


def _mask_cols(m: np.ndarray, keep_cols: set[int]) -> np.ndarray:
    """Keep presence (1) only in kept columns; preserve land (-99) everywhere,
    zero presence elsewhere."""
    out = m.copy()
    for c in range(m.shape[1]):
        if c not in keep_cols:
            col = out[:, c]
            col[col == 1] = 0  # drop presence; land (-99) and absent (0) unchanged
    return out


def _write_map(name: str, m: np.ndarray) -> None:
    np.savetxt(MAPS_DIR / name, m.astype(int), fmt="%d", delimiter=";")


def build_maps() -> None:
    juv = _load("cod_juvenile.csv")
    adult = _load("cod_adult.csv")
    spawn = _load("cod_spawning.csv")

    _write_map("cod_west_juvenile.csv", _mask_cols(juv, WEST_COLS))
    _write_map("cod_west_adult.csv", _mask_cols(adult, WEST_COLS))
    _write_map("cod_west_spawning.csv", _mask_cols(adult, WEST_COLS))  # west spawn in west habitat
    _write_map("cod_east_juvenile.csv", _mask_cols(juv, EAST_COLS))
    _write_map("cod_east_adult.csv", _mask_cols(adult, EAST_COLS))
    _write_map("cod_east_spawning.csv", _mask_cols(spawn, EAST_COLS))  # deep-basin RV grounds
    print("wrote 6 cod_west / cod_east map files")


_ALL_STEPS = ";".join(str(i) for i in range(24))
_WEST_SPAWN_STEPS = ";".join(str(i) for i in range(2, 10))  # Feb-May western spring
_WEST_REST_STEPS = ";".join(str(i) for i in [0, 1] + list(range(10, 24)))
_EAST_SPAWN_STEPS = ";".join(str(i) for i in range(10, 20))  # Jun-Oct eastern summer
_EAST_REST_STEPS = ";".join(str(i) for i in list(range(0, 10)) + list(range(20, 24)))


def wire_movement() -> None:
    move = CONFIG_DIR / "baltic_param-movement.csv"

    # cod_west: repoint the existing cod maps (map0-3)
    for i in range(4):
        set_key(move, f"movement.species.map{i}", "cod_west")
    set_key(move, "movement.file.map0", "maps/cod_west_juvenile.csv")
    set_key(move, "movement.file.map1", "maps/cod_west_adult.csv")
    set_key(move, "movement.file.map2", "maps/cod_west_spawning.csv")
    set_key(move, "movement.steps.map2", _WEST_SPAWN_STEPS)
    set_key(move, "movement.file.map3", "maps/cod_west_adult.csv")
    set_key(move, "movement.steps.map3", _WEST_REST_STEPS)

    # cod_east: new maps 26-29 (juvenile, sub-adult, spawning-summer, adult-rest)
    east = [
        (26, 0, 1, "cod_east_juvenile.csv", _ALL_STEPS),
        (27, 1, 4, "cod_east_adult.csv", _ALL_STEPS),
        (28, 4, 15, "cod_east_spawning.csv", _EAST_SPAWN_STEPS),
        (29, 4, 15, "cod_east_adult.csv", _EAST_REST_STEPS),
    ]
    for idx, a0, a1, fname, steps in east:
        set_key(move, f"movement.species.map{idx}", "cod_east")
        set_key(move, f"movement.initialage.map{idx}", str(a0))
        set_key(move, f"movement.lastage.map{idx}", str(a1))
        set_key(move, f"movement.file.map{idx}", f"maps/{fname}")
        set_key(move, f"movement.steps.map{idx}", steps)
    print("wired movement: map0-3 -> cod_west; map26-29 -> cod_east")


def report_disjointness() -> None:
    west = _load("cod_west_adult.csv") == 1
    east = _load("cod_east_adult.csv") == 1
    overlap = int((west & east).sum())
    smaller = int(min(west.sum(), east.sum()))
    wc = np.argwhere(west)[:, 1].mean()
    ec = np.argwhere(east)[:, 1].mean()
    print(
        f"go/no-go: west={int(west.sum())} east={int(east.sum())} overlap={overlap} "
        f"({overlap / smaller:.1%} of smaller); west col {wc:.1f} vs east col {ec:.1f}"
    )


if __name__ == "__main__":
    build_maps()
    wire_movement()
    report_disjointness()
    print("cod E/W maps (Task 3) complete.")
