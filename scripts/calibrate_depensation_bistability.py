#!/usr/bin/env python
"""Unit 2 placement harness (SP1): grid-sweep (larval-scale x S50 x theta) with the REAL
config-plumbed depensation gate, using warm-start reciprocal-invasion (cod-rich 300kt vs
cod-poor 1kt) + a two-tier SSB stability discriminator:

  - coarse 50yr screen: shortlist a point if the rich basin is in the GO magnitude band
    [40k, 300k] t SSB AND its final decade is decelerating (decline <= 10% vs the prior decade).
  - 175yr arbiter (the arbiter, not the screen): the rich basin is genuinely stable iff, after a
    ~30yr burn-in, it makes no new low below its post-burn-in running minimum, its tail passes an
    is_stationary check, AND its arbiter final-decade mean is in [40k, 300k]. A slow ghost-attractor
    slide eventually collapses over 175yr and is rejected.

"Bistable" split = rich (in-band, stable) vs poor (collapsed) differ by gap > 0.5. Measures SSB
(output.ssb.enabled + .ssb()), NOT run_simulation's biomass cod_mean (Bpa is an SSB reference).
Three-way verdict per point: GO / no-split / arbiter-fail. NOT a CI gate (long compute).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from baltic_bistability_chunk0 import (  # noqa: E402
    cod_poor_seeding,
    cod_rich_seeding,
    is_stationary,
    larva_scale_override,
    read_base_config,
    read_base_larva_rates,
    warmstart_override,
)

COD = 0
SCALES = [0.6, 0.75, 0.85, 0.90, 0.95, 1.0]
S50_GRID = [30_000.0, 60_000.0, 90_000.0, 120_000.0]
THETA_GRID = [2.0, 4.0]
SEEDS = (0, 1, 2)
SCREEN_YEARS = 50
ARBITER_YEARS = 175
GO_BAND = (40_000.0, 300_000.0)
COLLAPSE_T = 6_000.0  # classify_state collapse_frac(0.05) x Bpa(120k)
GAP_THRESH = 0.5
BURN_IN_DECADES = 3
DECLINE_TOL = 0.10


def gate_overrides(s50: float, theta: float) -> dict:
    return {
        "reproduction.depensation.gate.enabled": "true",
        "reproduction.depensation.gate.species.enabled.sp0": "true",
        "reproduction.depensation.gate.s50.sp0": str(s50),
        "reproduction.depensation.gate.theta.sp0": str(theta),
        "output.ssb.enabled": "true",
    }


def cod_ssb_series(base, base_rates, scale, rich, s50, theta, seed, n_year, gate=True):
    """Annual cod SSB (tonnes) over an n_year warm-start run at the given IC + gate params."""
    from osmose.engine import PythonEngine

    raw = {**base, "simulation.time.nyear": str(n_year), "output.ssb.enabled": "true"}
    raw.update(warmstart_override(True))
    raw.update(cod_rich_seeding() if rich else cod_poor_seeding())
    raw.update(larva_scale_override(scale, base_rates))
    if gate:
        raw.update(gate_overrides(s50, theta))
    return PythonEngine().run_in_memory(raw, seed=seed).ssb()["cod"].to_numpy(dtype=float)


def _decade_means(series: np.ndarray) -> list[float]:
    return [float(np.mean(series[i : i + 10])) for i in range(0, max(1, len(series) - 9), 10)] or [
        float(np.mean(series))
    ]


def passes_coarse_screen(series: np.ndarray) -> bool:
    dm = _decade_means(series)
    if not (GO_BAND[0] <= dm[-1] <= GO_BAND[1]):
        return False
    if len(dm) < 2:
        return True
    decline = (dm[-2] - dm[-1]) / max(dm[-2], 1.0)
    return decline <= DECLINE_TOL


def arbiter_stable(series: np.ndarray) -> tuple[bool, float]:
    """Genuinely stable iff: post-burn-in no-new-lows + tail stationarity + arbiter-final-decade
    magnitude in the GO band. Returns (stable, arbiter_final_decade_mean)."""
    dm = _decade_means(series)
    final_mean = dm[-1]
    if not (GO_BAND[0] <= final_mean <= GO_BAND[1]):
        return False, final_mean
    # no new low below the post-burn-in running minimum (converging-down-to-plateau is OK)
    post = series[BURN_IN_DECADES * 10 :]
    if len(post) < 20:
        return False, final_mean
    running_min = post[0]
    for v in post[1:]:
        if v < running_min * (1.0 - DECLINE_TOL):
            return False, final_mean  # still making new lows -> sliding to collapse
        running_min = min(running_min, v)
    # tail stationarity
    tail = series[-20:]
    tmean = float(np.mean(tail))
    cv = float(np.std(tail) / (tmean + 1.0))
    trend = abs(float(np.polyfit(range(len(tail)), tail, 1)[0])) / (tmean + 1.0)
    return is_stationary(cv, trend), final_mean


def _median_series(runs: list[np.ndarray]) -> np.ndarray:
    n = min(len(r) for r in runs)
    return np.median(np.stack([r[:n] for r in runs]), axis=0)


def classify_point(base, base_rates, scale, s50, theta) -> dict:
    rich = _median_series(
        [cod_ssb_series(base, base_rates, scale, True, s50, theta, s, SCREEN_YEARS) for s in SEEDS]
    )
    poor = _median_series(
        [cod_ssb_series(base, base_rates, scale, False, s50, theta, s, SCREEN_YEARS) for s in SEEDS]
    )
    rich_end, poor_end = float(np.mean(rich[-10:])), float(np.mean(poor[-10:]))
    gap = abs(rich_end - poor_end) / max(rich_end, poor_end, 1.0)
    split = gap > GAP_THRESH and poor_end < COLLAPSE_T
    out = {
        "scale": scale,
        "s50": s50,
        "theta": theta,
        "rich_screen": rich_end,
        "poor_screen": poor_end,
        "gap": gap,
    }
    if not (split and passes_coarse_screen(rich)):
        out["verdict"] = "no-split"
        return out
    arb = _median_series(
        [cod_ssb_series(base, base_rates, scale, True, s50, theta, s, ARBITER_YEARS) for s in SEEDS]
    )
    stable, mean = arbiter_stable(arb)
    out["verdict"] = "GO" if stable else "arbiter-fail"
    out["healthy_ssb_mean"] = mean
    return out


def main() -> list[dict]:
    base = read_base_config()
    base_rates = read_base_larva_rates(base)
    print(f"base larva rates (per-dt): {base_rates}", flush=True)
    results = []
    for scale in SCALES:
        for s50 in S50_GRID:
            for theta in THETA_GRID:
                r = classify_point(base, base_rates, scale, s50, theta)
                results.append(r)
                print(r, flush=True)
    gos = [r for r in results if r["verdict"] == "GO"]
    print("\n=== GO points (closest to Bpa 120kt, then scale->1.0) ===", flush=True)
    for r in sorted(gos, key=lambda r: (abs(r["healthy_ssb_mean"] - 120_000.0), 1.0 - r["scale"])):
        print(r, flush=True)
    if not gos:
        print(
            "NO GO points at any grid node — AMBIGUOUS (under-resolved) or NO-GO; see spec.",
            flush=True,
        )
    return results


if __name__ == "__main__":
    main()
