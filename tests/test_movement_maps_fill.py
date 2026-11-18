"""MovementMapSet's index table equals the original per-cell loop, bit for bit.

The fill was vectorised (2026-11-18, C# engine task 1.11: ~1.2M Python iterations on EEC at
10 years, paid by simulate() and by the C# bundle build). Both engines read this table, so a
C#-vs-Python comparison cannot catch an error in it; this test compares against the loop the
vectorised fill replaced, on real configs and on synthetic edge cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine.config import EngineConfig
from osmose.engine.movement_maps import MovementMapSet, _parse_semicolon_ints


def _reference_index(
    config: dict, species: str, ndt: int, n_years: int, lifespan_dt: int
) -> np.ndarray:
    """The original fill, verbatim in its logic (map order; later maps overwrite earlier ones)."""
    n_total = max(n_years * ndt, ndt)
    index = np.full((lifespan_dt, n_total), -1, dtype=np.int32)
    numbers = sorted(
        int(k[len("movement.species.map") :])
        for k, v in config.items()
        if k.startswith("movement.species.map")
        and k[len("movement.species.map") :].isdigit()
        and v.strip().lower() == species.strip().lower()
    )
    canonical: dict[str, int] = {}
    for map_idx, n in enumerate(numbers):
        init_age = round(float(config.get(f"movement.initialage.map{n}", "0")) * ndt)
        last_age = min(
            round(float(config.get(f"movement.lastage.map{n}", str(lifespan_dt / ndt))) * ndt),
            lifespan_dt - 1,
        )
        steps_raw = config.get(f"movement.steps.map{n}", "")
        steps = (
            _parse_semicolon_ints(steps_raw, ndt)
            if steps_raw and steps_raw.strip().lower() not in ("null", "")
            else list(range(ndt))
        )
        years_raw = config.get(f"movement.years.map{n}", "")
        if years_raw and years_raw.strip().lower() not in ("null", ""):
            years = _parse_semicolon_ints(years_raw, n_years)
        else:
            y0 = int(config.get(f"movement.initialyear.map{n}", "0"))
            y1 = int(config.get(f"movement.lastyear.map{n}", str(n_years - 1)))
            years = [y for y in range(y0, y1 + 1) if y < n_years]
        file_val = config.get(f"movement.file.map{n}", "null")
        key = None if file_val.strip().lower() == "null" else file_val
        value = canonical.setdefault(key, map_idx) if key is not None else map_idx
        for age in range(init_age, last_age + 1):
            if age >= lifespan_dt:
                continue
            for year in years:
                for season in steps:
                    g = year * ndt + season
                    if g >= n_total:
                        continue
                    index[age, g] = value
    return index


def _canonical(index: np.ndarray) -> np.ndarray:
    """Renumber map indices by first appearance, so the reference's pre-dedup numbering and the
    class's final (deduplicated, compacted) numbering compare as partitions of the table."""
    out = np.full_like(index, -1)
    seen: dict[int, int] = {}
    flat, res = index.ravel(), out.ravel()
    for i, v in enumerate(flat):
        if v >= 0:
            res[i] = seen.setdefault(int(v), len(seen))
    return out


def _build(cfg, sp: int, ny: int, nx: int) -> MovementMapSet:
    raw = cfg.raw_config
    return MovementMapSet(
        config=raw, species_name=cfg.species_names[sp], n_dt_per_year=cfg.n_dt_per_year,
        n_years=cfg.n_year, lifespan_dt=int(cfg.lifespan_dt[sp]), ny=ny, nx=nx,
        config_dir=raw.get("_osmose.config.dir", ""),
    )  # fmt: skip


@pytest.mark.parametrize(("demo", "nyear"), [("eec_full", 10), ("baltic", 3)])
def test_index_table_equals_the_original_loop_on_shipped_configs(demo, nyear, tmp_path):
    from osmose.engine import PythonEngine

    raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, tmp_path)["config_file"])))
    raw["simulation.time.nyear"] = str(nyear)
    cfg = EngineConfig.from_dict(raw)
    grid = PythonEngine()._resolve_grid(raw)
    checked = 0
    for sp in range(cfg.n_species):
        if cfg.movement_method[sp] != "maps":
            continue
        ms = _build(cfg, sp, grid.ny, grid.nx)
        ref = _reference_index(
            cfg.raw_config,
            cfg.species_names[sp],
            cfg.n_dt_per_year,
            cfg.n_year,
            int(cfg.lifespan_dt[sp]),
        )
        assert np.array_equal(_canonical(ms.index_maps), _canonical(ref)), cfg.species_names[sp]
        assert (ms.index_maps >= 0).any()
        checked += 1
    assert checked >= 5


def test_index_table_equals_the_original_loop_on_edge_cases():
    """Year lists, a step list with an out-of-range season, a negative initial age (it wraps, as
    the loop's indexing did), overlapping maps (the later one wins) and a case-insensitive name.
    Null maps only, so no map file is needed."""
    ndt, n_years, lifespan = 4, 3, 10
    config = {
        "movement.species.map0": "fish", "movement.file.map0": "null",
        "movement.initialage.map0": "-0.25", "movement.lastage.map0": "1.0",
        "movement.species.map1": "fish", "movement.file.map1": "null",
        "movement.initialage.map1": "0.5", "movement.lastage.map1": "3.0",
        "movement.steps.map1": "0;2;5", "movement.years.map1": "0;2",
        "movement.species.map2": "Fish", "movement.file.map2": "null",
        "movement.initialage.map2": "1.0", "movement.lastyear.map2": "1",
        "movement.species.map3": "other", "movement.file.map3": "null",
    }  # fmt: skip
    ref = _reference_index(config, "fish", ndt, n_years, lifespan)
    got = MovementMapSet(config, "fish", ndt, n_years, lifespan, ny=2, nx=2).index_maps
    assert np.array_equal(_canonical(got), _canonical(ref))
    assert (ref[-1] >= 0).any()  # the negative initial age wrapped to the last row, as before
