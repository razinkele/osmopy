from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.calibration.targets import load_targets

DET = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}
PERCIDS, HIGHW = ["perch", "pikeperch"], ["cod", "herring", "sprat"]
RUNGS = {
    "coarse": "data/baltic/baltic_all-parameters.csv",
    "4x-upsampled": "data/baltic-fine/baltic_fine_upsampled_all-parameters.csv",
    "4x-real": "data/baltic-fine/baltic_fine_real_all-parameters.csv",
}


def late_mean(series, frac=1 / 3):
    b = np.asarray(series, float)
    return float(np.mean(b[int(len(b) * (1 - frac)) :]))


def percid_area_ratio():
    """real habitat cells / upsampled footprint cells, averaged over the 6 percid maps."""
    m = Path("data/baltic-fine/maps")
    rs, us = 0, 0
    for f in [
        "perch_adult",
        "perch_juvenile",
        "perch_spawning",
        "pikeperch_adult",
        "pikeperch_juvenile",
        "pikeperch_spawning",
    ]:
        real = pd.read_csv(m / f"{f}.csv", sep=";", header=None).values
        up = pd.read_csv(m / f"{f}_upsampled.csv", sep=";", header=None).values
        rs += int(np.sum(real == 1))
        us += int(np.sum(up > 0))
    return rs / us if us else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyear", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    tlist, _ = load_targets(Path("data/baltic/reference/biomass_targets.csv"))
    targets = {t.species: t.target for t in tlist}
    results = {}
    for rung, path in RUNGS.items():
        base = dict(OsmoseConfigReader().read(path))
        base.update(DET)
        base["simulation.time.nyear"] = str(args.nyear)
        acc = {sp: [] for sp in PERCIDS + HIGHW}
        for s in range(args.seeds):
            bio = PythonEngine().run_in_memory(dict(base), seed=s).biomass()  # WIDE frame
            for sp in acc:
                acc[sp].append(late_mean(bio[sp]) / targets[sp])
        results[rung] = {sp: (float(np.mean(v)), float(np.std(v))) for sp, v in acc.items()}
    area = percid_area_ratio()
    print("species     " + "  ".join(f"{r:>16}" for r in RUNGS) + "   role")
    for sp in PERCIDS + HIGHW:
        row = "  ".join(f"{results[r][sp][0]:8.1f}±{results[r][sp][1]:4.1f}" for r in RUNGS)
        print(f"{sp:11} {row}   {'PERCID' if sp in PERCIDS else 'high-weight'}")
    print(f"real/upsampled percid area ratio = {area:.3f}")
    up, real = results["4x-upsampled"], results["4x-real"]

    # GO: real percids drop toward single digits, STABLY, AND by MORE than the pure area cut
    def dropped(sp):
        rel = (up[sp][0] - real[sp][0]) / up[sp][0] if up[sp][0] else 0.0
        return rel > (1 - area) and real[sp][0] < 10 and real[sp][1] < 0.5 * max(real[sp][0], 1e-9)

    go = all(dropped(sp) for sp in PERCIDS)
    print(
        "VERDICT:",
        "GO — real habitat damps percids beyond the area cut, stably"
        if go
        else "NO-GO — real ~= upsampled / only-area / unstable (structural)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
