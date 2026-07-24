"""cod_west (western SD22-24) and cod_east (eastern deep basins SD25-32) occupy
largely disjoint distribution maps — the spatial niche separation that makes the
disaggregation real rather than cosmetic (Phase 1 Task 3 go/no-go).
"""

from pathlib import Path

import numpy as np

MAPS = Path("data/baltic/maps")
MOVE = Path("data/baltic/baltic_param-movement.csv")


def _load_map(name: str) -> np.ndarray:
    return np.genfromtxt(MAPS / name, delimiter=";")


def _present(m: np.ndarray) -> np.ndarray:
    return m == 1


def test_cod_ew_map_files_exist():
    for stem in ("cod_west", "cod_east"):
        for stage in ("juvenile", "adult", "spawning"):
            assert (MAPS / f"{stem}_{stage}.csv").exists(), f"missing {stem}_{stage}.csv"


def test_west_and_east_footprints_largely_disjoint():
    west = _present(_load_map("cod_west_adult.csv"))
    east = _present(_load_map("cod_east_adult.csv"))
    assert west.any() and east.any()
    overlap = (west & east).sum()
    smaller = min(west.sum(), east.sum())
    # SD24 (Arkona) is a legitimate shared transition; overlap must stay small
    assert overlap / smaller < 0.25, f"footprints overlap {overlap}/{smaller} — niche not separated"


def test_west_is_western_east_is_eastern():
    """Centres of mass separate along longitude (west of east)."""
    west = _present(_load_map("cod_west_adult.csv"))
    east = _present(_load_map("cod_east_adult.csv"))
    west_col = np.argwhere(west)[:, 1].mean()
    east_col = np.argwhere(east)[:, 1].mean()
    assert west_col < east_col - 5, f"west col {west_col:.1f} not clearly west of east {east_col:.1f}"


def test_movement_config_wires_both_cod_species():
    text = MOVE.read_text()
    assert "movement.species.map0;cod_west" in text
    assert any(f"movement.species.map{i};cod_east" in text for i in range(26, 40))


def test_salinity_gate_enabled_for_both_cod():
    text = MOVE.read_text()
    assert "movement.salinity.gate.species.enabled.sp0;true" in text
    assert "movement.salinity.gate.species.enabled.sp8;true" in text
