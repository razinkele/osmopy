#!/usr/bin/env python
"""DE-RISK SPIKE (throwaway): can a recruitment depensation/Allee term create BISTABILITY
in the Baltic model? Monkeypatches a cod (sp0) Allee factor A(SSB)=SSB^θ/(S50^θ+SSB^θ) onto
apply_stock_recruitment, then runs the VALIDATED warm-start reciprocal-invasion contrast
(cod-rich 300kt vs cod-poor 1kt standing-stock ICs) across a larval-mortality-scale × S50 grid.

BISTABLE (GO) if any (scale,S50) makes the two ICs land in DIFFERENT cod basins (one persists,
one collapses) — the depensation trap the compensatory SR forms cannot produce. θ=4 (sharpest
trap = most favorable to bistability) for this first pass; refine only if a split appears.
Baseline (no Allee) control must reproduce the known MONOSTABLE result (rich≈poor).
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

from baltic_bistability_chunk0 import (  # noqa: E402
    cod_poor_seeding,
    cod_rich_seeding,
    larva_scale_override,
    read_base_config,
    read_base_larva_rates,
    warmstart_override,
)
from calibrate_baltic import run_simulation  # noqa: E402

COD = 0
N_YEAR = 15
SEEDS = (0, 1)
SCALES = [0.3, 0.5, 0.7, 1.0]  # larval-M scale: low=cod-viable/overshoot, 1.0=deployed(collapse)
S50_GRID = [30_000.0, 90_000.0]  # Allee half-SSB (tonnes), around/below cod Bpa 120kt
THETA = 4.0  # sharpest trap; most favorable to bistability


def set_allee(s50, theta=THETA):
    """Monkeypatch (s50=None restores original). Multiplies cod recruits by the Allee factor
    using the real per-step cod SSB. Warm-start disables egg-rescue, so the trap is genuine."""
    import osmose.engine.processes.reproduction as repro

    if not hasattr(repro, "_ORIG_ASR"):
        repro._ORIG_ASR = repro.apply_stock_recruitment
    orig = repro._ORIG_ASR
    if s50 is None:
        repro.apply_stock_recruitment = orig
        return

    def patched(linear_eggs, ssb, ssb_half, recruitment_type, shepherd_beta=None):
        out = orig(linear_eggs, ssb, ssb_half, recruitment_type, shepherd_beta)
        s = ssb[COD]
        if s > 0:
            out[COD] *= s**theta / (s50**theta + s**theta)
        return out

    repro.apply_stock_recruitment = patched


def cod_mean(base_config, base_rates, scale, rich, s50, seed):
    set_allee(s50)
    ov = {}
    ov.update(warmstart_override(True))
    ov.update(cod_rich_seeding() if rich else cod_poor_seeding())
    ov.update(larva_scale_override(scale, base_rates))
    try:
        stats = run_simulation(base_config, ov, n_years=N_YEAR, seed=seed)
    except Exception as exc:  # noqa: BLE001
        print(f"      ! run failed (scale={scale} rich={rich} s50={s50} seed={seed}): {exc!r}")
        return float("nan")
    finally:
        set_allee(None)
    return float(stats.get("cod_mean", 0.0))


def avg(base_config, base_rates, scale, rich, s50):
    import numpy as np

    return float(
        np.nanmean([cod_mean(base_config, base_rates, scale, rich, s50, s) for s in SEEDS])
    )


def gap(rich, poor):
    import numpy as np

    m = max(rich, poor, 1.0)
    return abs(rich - poor) / m if np.isfinite(rich) and np.isfinite(poor) else float("nan")


if __name__ == "__main__":
    base_config = read_base_config()
    base_rates = read_base_larva_rates(base_config)
    print(f"base larva rates (per-dt): {base_rates}")
    print(f"grid: scales={SCALES} S50={S50_GRID} theta={THETA} seeds={SEEDS} nyear={N_YEAR}\n")

    print("=== BASELINE control (no Allee): expect MONOSTABLE (rich≈poor) ===")
    for scale in [1.0, 0.5]:
        r = avg(base_config, base_rates, scale, True, None)
        p = avg(base_config, base_rates, scale, False, None)
        print(f"  scale={scale}: cod_rich={r:,.0f}  cod_poor={p:,.0f}  gap={gap(r, p):.2f}")

    print("\n=== ALLEE test (theta=4): BISTABLE if rich persists & poor collapses (gap large) ===")
    hits = []
    for scale in SCALES:
        for s50 in S50_GRID:
            r = avg(base_config, base_rates, scale, True, s50)
            p = avg(base_config, base_rates, scale, False, s50)
            g = gap(r, p)
            flag = "  <-- BASIN SPLIT" if g > 0.5 else ""
            if g > 0.5:
                hits.append((scale, s50, r, p, g))
            print(
                f"  scale={scale} S50={s50:>7,.0f}: cod_rich={r:>12,.0f}  cod_poor={p:>12,.0f}  gap={g:.2f}{flag}"
            )

    print("\n=== VERDICT ===")
    if hits:
        print(
            f"GO (tentative): {len(hits)} basin-split point(s) — depensation CAN create bistability:"
        )
        for scale, s50, r, p, g in hits:
            print(f"  scale={scale} S50={s50:,.0f}: rich={r:,.0f} poor={p:,.0f} gap={g:.2f}")
        print("  -> confirm with more seeds + finer grid before building the feature.")
    else:
        print("NO-GO: no basin split at any (scale,S50) with the sharpest trap (theta=4).")
        print(
            "  -> depensation-via-recruitment-Allee does not manufacture bistability here either;"
        )
        print("     model structurally resists a second basin. Strong negative.")
