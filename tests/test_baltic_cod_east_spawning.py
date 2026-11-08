"""Eastern cod spawn only in SD 25 (Eero et al. 2012): the committed map is exactly what
scripts/build_cod_east_sd25_spawning.py derives, and every cell is in the SD 25 box."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


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
