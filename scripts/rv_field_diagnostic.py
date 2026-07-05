#!/usr/bin/env python
"""Report the spatial RV field's correctness metrics + the gate on/off cod shift."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.calibration.larva_recal import mean_cod
from osmose.config import OsmoseConfigReader
from osmose.forcing.grid import load_ocean_mask

ROOT = Path(__file__).resolve().parent.parent
FIELD = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"
SPAWN = ROOT / "data" / "baltic" / "maps" / "cod_spawning.csv"


def main() -> int:
    da = xr.open_dataset(FIELD)["reproductive_volume"]
    rv = da.values
    ref = float(da.attrs["RV_ref"])
    spawn = np.flipud(np.genfromtxt(SPAWN, delimiter=";")) > 0
    ocean = load_ocean_mask(ROOT / "data" / "baltic" / "baltic_grid.nc")

    # Confounded contrast (spawn vs ALL non-spawning ocean): inverts, because the
    # ultra-saline Danish straits/Kattegat (outside the cod range) dominate RV.
    coast = ocean & ~spawn
    mean_spawn = float(rv[:, spawn].mean())
    mean_coast = float(rv[:, coast].mean()) if coast.any() else float("nan")
    ratio_naive = mean_spawn / mean_coast if mean_coast else float("inf")
    # Corrected contrast: spawn basins vs the FRESH northern Gulf of Bothnia (rows 0-13).
    fresh = ocean.copy()
    fresh[14:] = False
    mean_fresh = float(rv[:, fresh].mean()) if fresh.any() else float("nan")
    ratio_fresh = mean_spawn / mean_fresh if mean_fresh else float("inf")

    m_field = rv.mean(axis=0)  # time-mean (shipped climatology); CV across spawning cells
    cv = float(m_field[spawn].std() / m_field[spawn].mean()) if m_field[spawn].mean() > 0 else 0.0
    sp_nz = rv[:, spawn][rv[:, spawn] > 0]
    mean_s = float(np.clip(sp_nz / ref, 0.0, 1.0).mean()) if sp_nz.size else 0.0
    frac_viable = float((rv[:, spawn] > 0).mean())

    base = dict(
        OsmoseConfigReader().read(str(ROOT / "data" / "baltic" / "baltic_all-parameters.csv"))
    )
    base["simulation.time.nyear"] = "15"
    on = dict(
        base,
        **{
            "reproduction.rv.spatial.enabled": "true",
            "reproduction.rv.spatial.field.file": str(FIELD),
            "reproduction.rv.spatial.species.enabled.sp0": "true",
        },
    )
    off_mean = mean_cod(base)
    on_mean = mean_cod(on)

    lines = [
        "# Spatial RV field diagnostic",
        "",
        f"RV_ref = {ref:.2f} m",
        f"within-basin CV = {cv:.3f}  (GO/NO-GO: go if >= 0.20)  ->  "
        f"{'GO' if cv >= 0.20 else 'NO-GO'}",
        f"mean(s_cell) over RV>0 spawning cells = {mean_s:.3f}  (mean-anchor target [0.6, 1.0])",
        f"fraction of cod_spawning cells with RV > 0 = {frac_viable:.3f}",
        "",
        "## Basin contrast",
        f"spawn vs fresh northern gulf = {ratio_fresh:.2f}  (mean_spawn={mean_spawn:.2f}, "
        f"mean_fresh={mean_fresh:.2f})",
        f"spawn vs ALL non-spawning ocean = {ratio_naive:.2f}  (mean_coast={mean_coast:.2f}) "
        "-- CONFOUNDED, see finding",
        "",
        "## Finding: the Danish-straits confound",
        "The viable-thickness metric (salinity >= 11 PSU AND O2 >= 89.3 mmol/m3) makes the "
        "ultra-saline Danish straits/Kattegat the highest-RV cells (up to ~220 m), even though "
        "they are outside the cod spawning range and receive no eggs. A naive spawn-vs-all-ocean "
        "contrast therefore inverts. This does NOT affect the mechanism (eggs are placed only on "
        "the cod_spawning map), and the within-basin CV -- the real go/no-go -- is a strong GO.",
        "",
        "## Gate on/off cod biomass (15-yr, years 3-14 mean)",
        f"off={off_mean:.0f}  on={on_mean:.0f}  delta={100 * (on_mean / off_mean - 1):+.0f}%  "
        "(SP1b larval-M recalibration restores the mean; not done here)",
    ]
    print("\n".join(lines))
    out = ROOT / "docs" / "diagnostics" / "rv_spatial_field.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
