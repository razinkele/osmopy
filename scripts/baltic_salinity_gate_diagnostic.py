"""A/B diagnostic: Baltic salinity gate OFF vs ON (real bottom-salinity field).

Reports percid (perch, pikeperch) late-window biomass off vs on, and cod's
biomass, so the spatial-realism effect (cod excluded from low-salinity coastal
cells -> less cod predation on percids there -> percid biomass UP) is visible.
NOT an overshoot fix: raising percids would if anything worsen the overshoot.
See docs/superpowers/specs/2026-07-04-baltic-salinity-forcing-design.md.
"""

from __future__ import annotations

import argparse

import numpy as np


def late_mean(series, frac: float = 1.0 / 3.0) -> float:
    b = np.asarray(series, dtype=np.float64)
    n = len(b)
    return float(np.mean(b[int(n * (1.0 - frac)) :]))


def run_ab(config_path: str, cod_index: int = 0) -> dict:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    base = OsmoseConfigReader().read(config_path)
    species = ("cod", "perch", "pikeperch")

    def _run_late(cfg):
        # ONE Baltic run per condition; read all species from the same result.
        bm = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
        return {sp: late_mean(bm[sp].to_numpy()) for sp in species}

    off = _run_late(base)
    on_cfg = dict(base)
    on_cfg["movement.salinity.gate.enabled"] = "true"
    on_cfg[f"movement.salinity.gate.species.enabled.sp{cod_index}"] = "true"
    on = _run_late(on_cfg)
    return {"off": off, "on": on}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baltic salinity-gate A/B (real field).")
    ap.add_argument("--config", default="data/baltic/baltic_all-parameters.csv")
    ap.add_argument("--cod-index", type=int, default=0)
    args = ap.parse_args(argv)
    r = run_ab(args.config, args.cod_index)
    print("late-window biomass (t)   OFF        ON        delta%")
    for sp in ("cod", "perch", "pikeperch"):
        o, n = r["off"][sp], r["on"][sp]
        d = (n - o) / o * 100.0 if o else float("nan")
        print(f"  {sp:10s} {o:12.1f} {n:12.1f} {d:+7.1f}%")
    print("\nNOTE: gating cod out of low-salinity coastal cells is a SPATIAL-REALISM")
    print("correction. Higher percid biomass here means less cod predation in the")
    print("refuge — it is NOT an overshoot fix (raising percids worsens overshoot).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
