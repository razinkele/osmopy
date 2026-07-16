#!/usr/bin/env python
"""Unit 3 validation (SP1): at a chosen depensation operating point,
  (1) a same-scale no-Allee warm-start control (gate OFF must be MONOSTABLE — proves the gate,
      not the larval-M window, creates the split); and
  (2) a quasi-static fishing-hysteresis F-ramp: ~10 F levels up to >=30x base F, each held until
      SSB EQUILIBRATES (|decade-over-decade rel change| < 5%, cap >=3tau ~75yr) because relaxation
      time inflates near the fold points (critical slowing down). A gate-OFF control ramp must show
      NO comparable loop. Reports F_collapse / F_recover in REAL (absolute) F for the reachability
      check: F_collapse <= historical peak (~2.3) AND F_recover > present-day reduced F (~0.16).

Base F = fisheries.rate.base.fsh0 (=0.08 for Baltic cod), resolved mode-agnostically via
fmsy_sweep.fishing_override; F applied via byyear-F (mortality.fishing.rate.byyear.file.sp0).
NOT a CI gate (multi-decade real-engine run, ~9,000-15,000 sim-years).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from baltic_bistability_chunk0 import (  # noqa: E402
    cod_rich_seeding,
    larva_scale_override,
    read_base_config,
    read_base_larva_rates,
    warmstart_override,
)
from calibrate_depensation_bistability import (  # noqa: E402
    COLLAPSE_T,
    GO_BAND,
    classify_point,
    gate_overrides,
)

LEVELS = [
    0.5,
    1,
    2,
    4,
    8,
    12,
    16,
    20,
    25,
    30,
]  # x base F; >=30x brackets historical peak ~2.3 (28.6x base)
DWELL_CAP_Y = 75
CONV_TOL = 0.05
HIST_PEAK_F = 2.3  # ICES cod.27.24-32, 1999
PRESENT_F = 0.16  # 2018-2022 mean


def resolve_base_f(base: dict) -> float:
    """cod base F via the mode-agnostic fisheries/legacy dispatch (=0.08 for the Baltic config)."""
    from osmose.engine.config import EngineConfig
    from osmose.validation.fmsy_sweep import fishing_override

    cfg = EngineConfig.from_dict(base)
    _, base_f = fishing_override(base, cfg, 0)
    return float(base_f)


def equilibrate_level(base, base_rates, scale, s50, theta, base_f, f_mult, seed, gate=True):
    """Hold one constant F level (= f_mult * base_f) from a cod-rich warm-start until SSB
    equilibrates (cap DWELL_CAP_Y). Returns (final_decade_mean, converged, series)."""
    from osmose.engine import PythonEngine

    tmp = Path(tempfile.mkdtemp())
    f_csv = tmp / "cod_f_byyear.csv"
    np.savetxt(f_csv, np.full(DWELL_CAP_Y, f_mult * base_f))
    raw = {
        **base,
        "simulation.time.nyear": str(DWELL_CAP_Y),
        "output.ssb.enabled": "true",
        "mortality.fishing.rate.byyear.file.sp0": str(f_csv),
    }
    raw.update(warmstart_override(True))
    raw.update(cod_rich_seeding())
    raw.update(larva_scale_override(scale, base_rates))
    if gate:
        raw.update(gate_overrides(s50, theta))
    s = PythonEngine().run_in_memory(raw, seed=seed).ssb()["cod"].to_numpy(dtype=float)
    dm = [float(np.mean(s[i : i + 10])) for i in range(0, max(1, len(s) - 9), 10)]
    converged = len(dm) >= 2 and abs(dm[-1] - dm[-2]) / max(dm[-2], 1.0) < CONV_TOL
    return float(np.mean(s[-10:])), converged, s


def ramp(base, base_rates, scale, s50, theta, base_f, seed, gate=True):
    """Up-leg then down-leg; each entry = (f_mult, final_decade_mean, converged)."""
    up = [
        (m, *equilibrate_level(base, base_rates, scale, s50, theta, base_f, m, seed, gate)[:2])
        for m in LEVELS
    ]
    down = [
        (m, *equilibrate_level(base, base_rates, scale, s50, theta, base_f, m, seed, gate)[:2])
        for m in reversed(LEVELS)
    ]
    return up, down


def fold_points(up, down, base_f):
    """F_collapse = lowest up-leg f_mult where SSB drops below COLLAPSE_T; F_recover = highest
    down-leg f_mult where SSB climbs back above the GO band. Returns absolute F (x base_f) or None,
    plus a flag if any fold-adjacent level failed to converge (-> withhold verdict as AMBIGUOUS)."""
    # up-leg is low->high F: first collapse = F_collapse (lowest F that collapses).
    f_collapse = next((m for m, ssb, _ in up if ssb < COLLAPSE_T), None)
    # down-leg is high->low F: first recovery = F_recover (highest F still healthy on the way down).
    f_recover = next((m for m, ssb, _ in down if ssb > GO_BAND[0]), None)
    # any level adjacent to a fold that hit the cap without converging?
    unconverged = [m for m, _ssb, conv in (up + down) if not conv]
    return (
        None if f_collapse is None else f_collapse * base_f,
        None if f_recover is None else f_recover * base_f,
        unconverged,
    )


def main(scale, s50, theta):
    base = read_base_config()
    base_rates = read_base_larva_rates(base)
    base_f = resolve_base_f(base)
    print(f"base F (cod) = {base_f}; scale={scale} s50={s50} theta={theta}", flush=True)

    print("\n=== (1) same-scale NO-ALLEE control (gate off must be MONOSTABLE) ===", flush=True)
    ctl = classify_point(base, base_rates, scale, s50, theta)  # gate-on reference (for contrast)
    # gate-off rich vs poor at the same scale via the placement harness with gate=False:
    from calibrate_depensation_bistability import cod_ssb_series

    rich0 = float(
        np.mean(cod_ssb_series(base, base_rates, scale, True, s50, theta, 0, 50, gate=False)[-10:])
    )
    poor0 = float(
        np.mean(cod_ssb_series(base, base_rates, scale, False, s50, theta, 0, 50, gate=False)[-10:])
    )
    gap0 = abs(rich0 - poor0) / max(rich0, poor0, 1.0)
    print(
        f"gate-off: rich={rich0:,.0f} poor={poor0:,.0f} gap={gap0:.2f} "
        f"({'MONOSTABLE (good)' if gap0 <= 0.5 else 'SPLIT (bad — not gate-attributable!)'})",
        flush=True,
    )
    print(f"gate-on reference verdict: {ctl.get('verdict')}", flush=True)

    print("\n=== (2) depensation F-ramp (quasi-static, per-level equilibration) ===", flush=True)
    up, down = ramp(base, base_rates, scale, s50, theta, base_f, seed=0, gate=True)
    print("up:  ", up, flush=True)
    print("down:", down, flush=True)
    fc, fr, unconv = fold_points(up, down, base_f)

    print("\n=== control F-ramp (gate off — expect NO loop) ===", flush=True)
    cup, cdown = ramp(base, base_rates, scale, s50, theta, base_f, seed=0, gate=False)
    print("control up:  ", cup, flush=True)
    print("control down:", cdown, flush=True)

    print("\n=== hysteresis + reachability ===", flush=True)
    if unconv:
        print(
            f"AMBIGUOUS: fold-adjacent level(s) hit the {DWELL_CAP_Y}yr cap without converging: {unconv}",
            flush=True,
        )
    print(
        f"F_collapse={fc} (abs), F_recover={fr} (abs) [x base = {fc and fc / base_f}, {fr and fr / base_f}]",
        flush=True,
    )
    reach = (fc is not None and fc <= HIST_PEAK_F) and (fr is not None and fr > PRESENT_F)
    print(
        f"reachability (F_collapse<={HIST_PEAK_F} AND F_recover>{PRESENT_F}): "
        f"{'PASS' if reach else 'FAIL/AMBIGUOUS'}",
        flush=True,
    )
    return {"f_collapse": fc, "f_recover": fr, "unconverged": unconv, "reachable": reach}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=0.85)
    p.add_argument("--s50", type=float, default=90_000.0)
    p.add_argument("--theta", type=float, default=4.0)
    a = p.parse_args()
    main(a.scale, a.s50, a.theta)
