from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from osmose.forcing.percid_habitat import percid_stage_map, vacuity_ok
from scripts.build_baltic_fine_grid import build_shallow_fraction, OUT

# (stage_file, depth_max_m, sal_ceiling, sal_gate)
STAGES = {
    "perch_adult.csv": (12.0, 12.0, None),
    "perch_juvenile.csv": (8.0, 12.0, None),
    "perch_spawning.csv": (6.0, None, 6.0),
    "pikeperch_adult.csv": (18.0, 14.0, None),
    "pikeperch_juvenile.csv": (12.0, 14.0, None),
    "pikeperch_spawning.csv": (8.0, None, 5.0),
}


def _annual_mean_salinity():
    ds = xr.open_dataset(OUT / "baltic_salinity_bottom_climatology.nc")
    v = "salinity" if "salinity" in ds else list(ds.data_vars)[0]
    return np.asarray(ds[v].values).mean(axis=0)  # (160,200); gap-filled in Task 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--max-ratio", type=float, default=0.4)
    args = ap.parse_args()
    maps_dir = OUT / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    sal = _annual_mean_salinity()
    for fname, (dmax, ceil, gate) in STAGES.items():
        frac, ocean = build_shallow_fraction(depth_max_m=dmax)
        m = percid_stage_map(frac, ocean, sal, tau=args.tau, sal_ceiling=ceil, sal_gate=gate)
        up = pd.read_csv(
            maps_dir / (fname[:-4] + "_upsampled.csv"), sep=";", header=None
        ).values.astype(float)
        if not vacuity_ok(m, up, max_ratio=args.max_ratio):
            raise ValueError(
                f"{fname}: vacuity guard failed (empty or > {args.max_ratio} of upsampled footprint)"
            )
        np.savetxt(maps_dir / fname, np.flipud(m), fmt="%d", delimiter=";")
        print(
            f"{fname}: {int(np.sum(m == 1))} habitat cells (upsampled footprint {int(np.sum(up > 0))})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
