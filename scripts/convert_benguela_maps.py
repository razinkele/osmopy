"""Convert Benguela's per-species NetCDF movement maps into osmopy CSV maps.

Each source movement.*.mapN references a stage variable (24,62,56) in input/maps/<species>.nc. osmopy
reads static CSV grids; time-variation is expressed by multiple indices with different
movement.steps.mapN. Each source index -> one osmopy index per DISTINCT 62x56 time-slice.
CSV format: semicolon-delimited, -99=land, ocean value = slice value. The grid is written np.flipud'd
because _load_csv_grid reverses rows on load (mirrors osmose/maps/builder.py::to_csv_text). Pass the
external source clone dir as argv[1].
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from osmose.config.reader import OsmoseConfigReader


def _grid_ocean(src_dir: Path) -> np.ndarray:
    ds = xr.open_dataset(src_dir / "input" / "grid-mask.nc")
    ocean = np.nan_to_num(ds["mask"].values.astype(float)) > 0
    ds.close()
    return ocean  # (62,56), True=ocean, engine (unflipped) convention


def _write_csv(path: Path, grid: np.ndarray) -> None:
    # flip so CSV row 0 = grid row ny-1, matching _load_csv_grid's row-reversal on read
    lines = [";".join(f"{v:g}" for v in row) + ";" for row in np.flipud(grid)]
    path.write_text("\n".join(lines) + "\n")


def convert_maps(src_dir: Path, maps_out: Path, keys_out: Path) -> list[dict]:
    maps_out.mkdir(parents=True, exist_ok=True)
    raw = dict(OsmoseConfigReader().read(str(src_dir / "osmose-ben.R")))
    ocean = _grid_ocean(src_dir)
    src_idxs = sorted({int(k.split(".map")[1]) for k in raw if k.startswith("movement.species.map")})
    out_rows: list[dict] = []
    seen: dict[bytes, str] = {}
    out_n = 0
    for si in src_idxs:
        sp = raw[f"movement.species.map{si}"]
        stage = raw[f"movement.variable.map{si}"]
        a0 = raw[f"movement.initialage.map{si}"]
        a1 = raw[f"movement.lastage.map{si}"]
        da = xr.open_dataset(src_dir / raw[f"movement.file.map{si}"])[stage].values  # (24,62,56)

        # Decide orientation from ALL time slices, not just t==0: an empty/land-only first slice must
        # not lock the whole series into the wrong flip. Pick the orientation with fewer presence-on-
        # land cells summed across every step (the real-loader test still guards the emitted CSVs).
        def _land_hits(arr: np.ndarray) -> int:
            return int((((arr > 0) & ~np.isnan(arr)) & ~ocean).sum())

        land_asis = sum(_land_hits(da[t]) for t in range(da.shape[0]))
        land_flip = sum(_land_hits(np.flipud(da[t])) for t in range(da.shape[0]))
        flip = land_flip < land_asis

        groups: dict[bytes, list[int]] = {}
        oriented = []
        for t in range(da.shape[0]):
            s = np.flipud(da[t]) if flip else da[t]
            g = np.where(ocean, np.nan_to_num(s), -99.0)
            oriented.append(g)
            groups.setdefault(g.tobytes(), []).append(t)
        for gb, steps in groups.items():
            g = oriented[steps[0]]
            if gb in seen:
                rel = seen[gb]
            else:
                fn = f"{sp}_{stage}_g{out_n}.csv"
                _write_csv(maps_out / fn, g)
                rel = f"maps/{fn}"
                seen[gb] = rel
                out_n += 1
            out_rows.append({"species": sp, "file": rel, "steps": steps, "a0": a0, "a1": a1})
    lines = []
    for n, r in enumerate(out_rows):
        lines += [
            f"movement.species.map{n} ; {r['species']}",
            f"movement.file.map{n} ; {r['file']}",
            f"movement.steps.map{n} ; {';'.join(str(s) for s in r['steps'])}",
            f"movement.initialage.map{n} ; {r['a0']}",
            f"movement.lastage.map{n} ; {r['a1']}",
        ]
    keys_out.write_text("\n".join(lines) + "\n")
    return out_rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1])
    convert_maps(src, root / "data" / "benguela" / "maps",
                 root / "data" / "benguela" / "_movement_keys.txt")
    print("maps converted")
