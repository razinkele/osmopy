"""A/B: does the percid thermal gate (thermal_cap) damp percid overshoot?

Two deterministic Baltic runs (fixed movement + mortality seeds), gate off vs on
for perch (sp4) + pikeperch (sp5). Uses the committed example sidecar until the
real CMEMS series is built. Honest-negative permitted: if percids do not fall,
say so plainly.

Review finding 2: the percid overshoot the feature targets is ABSOLUTE biomass
vs the ICES/HELCOM envelope (memory: x38-96), NOT the temporal peak/late boom-
bust ratio. So the primary verdict here is the change in absolute mean biomass
(down = toward the envelope = the intended direction); the peak/late ratio is
reported separately and labelled "stability", not "overshoot".

Review finding 3: perch is near-collapse, so thermal_cap can drive it to ~0.
overshoot_ratio guards late-mean==0 (returns inf) and the report flags a collapse
rather than presenting a spurious number; pikeperch is the load-bearing signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

SERIES = (Path(__file__).resolve().parent.parent
          / "data" / "baltic" / "forcing" / "baltic_percid_thermal_series_example.csv")
DET = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}
PERCIDS = (("perch", 4), ("pikeperch", 5))


def overshoot_ratio(series: np.ndarray, late_frac: float = 1.0 / 3.0) -> float:
    """Temporal boom/bust: peak biomass / late-window mean.

    Guards the collapse case (late-window mean == 0 -> inf) so a percid driven to
    ~0 does not masquerade as a finite ratio (review finding 3).
    """
    b = np.asarray(series, dtype=np.float64)
    late = b[int(len(b) * (1.0 - late_frac)):]
    lm = float(np.mean(late))
    return float("inf") if lm <= 0.0 else float(np.max(b)) / lm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyear", type=int, default=None,
                    help="override simulation.time.nyear (small = cheap smoke run; omit for full A/B)")
    args = ap.parse_args()

    cfg_path = sorted((Path("data") / "baltic").glob("*all-parameters*.csv"))[0]
    base = dict(OsmoseConfigReader().read(str(cfg_path)))
    if args.nyear is not None:
        base["simulation.time.nyear"] = str(args.nyear)
    off_cfg = {**base, **DET}
    on_cfg = {**off_cfg,
              "reproduction.thermal.gate.enabled": "true",
              "reproduction.thermal.gate.series.file": str(SERIES),
              "reproduction.thermal.gate.mode": "thermal_cap",
              "reproduction.thermal.gate.species.enabled.sp4": "true",
              "reproduction.thermal.gate.species.enabled.sp5": "true"}

    off = PythonEngine().run_in_memory(off_cfg, seed=0).biomass()
    on = PythonEngine().run_in_memory(on_cfg, seed=0).biomass()

    print("PRIMARY axis (finding 2) = absolute mean biomass; lower = toward the ICES envelope.")
    print(f"{'species':<11} {'mean_off':>11} {'mean_on':>11} {'abs_change':>11}  "
          f"{'stability_off':>13} {'stability_on':>12}  verdict")
    for name, _sp in PERCIDS:
        so, sn = off[name].to_numpy(), on[name].to_numpy()
        mo, mn = float(np.mean(so)), float(np.mean(sn))
        pct = (mn - mo) / mo * 100.0 if mo else float("nan")
        oo, on_r = overshoot_ratio(so), overshoot_ratio(sn)
        collapsed = (mn <= 1e-9) or not np.isfinite(on_r)
        if collapsed:
            verdict = "COLLAPSED (finding 3)"
        elif mn < mo * 0.98:
            verdict = f"overshoot damped ({pct:+.0f}% biomass)"
        elif mn > mo * 1.02:
            verdict = f"overshoot worse ({pct:+.0f}% biomass)"
        else:
            verdict = "no change"
        so_s = f"{oo:.3f}" if np.isfinite(oo) else "collapsed"
        on_s = f"{on_r:.3f}" if np.isfinite(on_r) else "collapsed"
        print(f"{name:<11} {mo:>11.4g} {mn:>11.4g} {pct:>+10.1f}%  "
              f"{so_s:>13} {on_s:>12}  {verdict}")
    print("\nNOTE: 'stability' = peak/late-window boom-bust ratio (temporal), NOT the "
          "absolute overshoot the feature targets. pikeperch is the load-bearing "
          "signal (perch near-collapse; finding 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
