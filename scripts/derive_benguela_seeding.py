"""Derive Benguela's analytic seeding block from osmose-ben_seeding.R (authors' values),
cross-checked against the restart file's per-species standing stock. Pass source clone as argv[1]."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from osmose.config.reader import OsmoseConfigReader


def derive_seeding(src_dir: Path, out_path: Path) -> dict[int, float]:
    raw = dict(OsmoseConfigReader().read(str(src_dir / "osmose-ben_seeding.R")))
    seed = {sp: float(raw[f"population.seeding.biomass.sp{sp}"]) for sp in range(10)}
    ds = xr.open_dataset(src_dir / "input" / "ben-initial_conditions.nc")
    spid = ds["species"].values; ab = ds["abundance"].values; w = ds["weight"].values
    for sp in range(10):
        m = spid == sp
        restart_t = float(np.nansum(ab[m] * w[m])) / 1e6
        if restart_t > 0 and not (0.2 < seed[sp] / restart_t < 5):
            print(f"WARN sp{sp}: authors={seed[sp]:.0f} vs restart={restart_t:.0f} (>5x apart)")
    ds.close()
    out_path.write_text("\n".join(
        f"population.seeding.biomass.sp{sp} ; {seed[sp]:.0f}" for sp in range(10)) + "\n")
    return seed


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1])
    derive_seeding(src, root / "data" / "benguela" / "_seeding_keys.txt")
    print("seeding derived")
