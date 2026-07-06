"""Run the wired Benguela config over long horizons and report per-species stability + seeding
diagnostics, to pin a safe simulation.time.nyear. Diagnostics attribute instability (seeding
re-injection vs food-web) per the spec."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

SEED = {0: 3129213, 1: 3888750, 2: 3029155, 3: 1286364, 4: 1138339,
        5: 1439984, 6: 198865, 7: 81054, 8: 575361, 9: 591907}


def run(master: Path, nyear: int):
    raw = dict(OsmoseConfigReader().read(str(master)))
    raw["simulation.time.nyear"] = str(nyear)
    raw["output.ssb.enabled"] = "true"
    res = PythonEngine().run_in_memory(raw, seed=42)
    b = res.biomass()
    cols = [c for c in b.columns if c not in ("Time", "time", "species")]
    bio = {c: b[c].to_numpy(dtype=float) for c in cols}
    try:
        s = res.ssb()
        ssb = {c: s[c].to_numpy(dtype=float) for c in cols if c in s.columns}
    except Exception:
        ssb = {}
    return cols, bio, ssb


def bounded(cols, bio) -> dict[str, bool]:
    v = {}
    for i, c in enumerate(cols):
        x = bio[c]
        cap = 1000.0 * SEED.get(i, max(SEED.values()))
        v[c] = bool(np.all(np.isfinite(x)) and np.all(x <= cap) and np.all(x <= 1e9) and x[-1] > 0)
    return v


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    master = root / "data" / "benguela" / "benguela_all-parameters.csv"
    sweep = (int(sys.argv[1]),) if len(sys.argv) > 1 else (100, 50, 30, 15)
    for ny in sweep:
        cols, bio, ssb = run(master, ny)
        v = bounded(cols, bio)
        print(f"nyear={ny}: bounded={sum(v.values())}/{len(v)}  fails={[k for k, ok in v.items() if not ok]}")
        # attribution: for each species, first step natural SSB exceeds its seed (seeding no longer needed)
        for i, c in enumerate(cols):
            if c in ssb:
                over = np.where(ssb[c] > SEED[i])[0]
                first = int(over[0]) if len(over) else -1
                print(f"    {c:18s} bio[-1]={bio[c][-1]:.3g} ssb>seed@step={first}")
