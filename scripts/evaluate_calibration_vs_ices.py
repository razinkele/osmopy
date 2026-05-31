#!/usr/bin/env python
"""Evaluate a calibrated Baltic parameter set against ICES biomass envelopes.

Runs ONE full simulation with a given parameter JSON (plus the per-phase
fixed-config the calibrator injects) and reports, per species, the last-10-year
mean biomass against the ``biomass_targets.csv`` ICES envelope: in-range /
overshoot / undershoot and a magnitude factor.

Used to compare the Beverton-Holt (phase 12) baseline against the Shepherd
(phase 13) result on equal footing — same simulation length, same seed — which
the bare objective number cannot do when the two runs were calibrated at
different sim lengths.

Example
-------
    .venv/bin/python scripts/evaluate_calibration_vs_ices.py \
        --params data/baltic/calibration_results/phase12_results.json \
        --mode bh --years 40 --seed 42

    .venv/bin/python scripts/evaluate_calibration_vs_ices.py \
        --params data/baltic/calibration_results/phase13_results.json \
        --mode shepherd --years 40 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_baltic import (
    BALTIC_CONFIG,
    SPECIES_NAMES,
    load_targets,
    run_simulation,
)


def _apply_mode(base_config: dict[str, str], mode: str) -> None:
    """Inject the fixed-config that the calibrator's ``main()`` applies per phase.

    ``bh``       — base config unchanged (B-H types already live in
                   baltic_param-reproduction.csv for sp0/3/4/5).
    ``shepherd`` — every species switched to Shepherd SR; cod sp0 ssb_half
                   pinned at the Bpa (120 kt), matching ``phase == "13"`` setup.
    """
    if mode == "bh":
        return
    if mode == "shepherd":
        for sp_idx in range(len(SPECIES_NAMES)):
            base_config[f"stock.recruitment.type.sp{sp_idx}"] = "shepherd"
        base_config["stock.recruitment.ssbhalf.sp0"] = "120000"
        return
    raise ValueError(f"unknown mode {mode!r}; expected 'bh' or 'shepherd'")


def evaluate(params_path: Path, mode: str, n_years: int, seed: int) -> dict:
    from osmose.config.reader import OsmoseConfigReader

    with open(params_path) as f:
        params = json.load(f)["parameters"]

    reader = OsmoseConfigReader()
    base_config = reader.read(BALTIC_CONFIG)
    _apply_mode(base_config, mode)

    # The JSON stores real-space (de-log10'd) values; apply directly as overrides.
    overrides = {k.lower(): str(v) for k, v in params.items()}

    stats = run_simulation(base_config, overrides, n_years=n_years, seed=seed)
    if not stats:
        raise RuntimeError("simulation failed (run_simulation returned {})")

    targets = {t.species: t for t in load_targets()}
    rows = []
    in_range = 0
    for sp in SPECIES_NAMES:
        mean = stats.get(f"{sp}_mean", 0.0)
        t = targets[sp]
        if mean <= 0:
            status, factor = "EXTINCT", 0.0
        elif mean < t.lower:
            status, factor = "under", mean / t.target
        elif mean > t.upper:
            status, factor = "OVER", mean / t.target
        else:
            status, factor = "in-range", mean / t.target
            in_range += 1
        rows.append(
            {
                "species": sp,
                "mean_tonnes": mean,
                "lower": t.lower,
                "upper": t.upper,
                "target": t.target,
                "weight": t.weight,
                "status": status,
                "factor_vs_target": factor,
                "cv": stats.get(f"{sp}_cv", 0.0),
            }
        )
    return {
        "params_path": str(params_path),
        "mode": mode,
        "n_years": n_years,
        "seed": seed,
        "in_range_count": in_range,
        "species": rows,
    }


def _print_report(result: dict) -> None:
    print(
        f"\n=== {result['mode'].upper()} | {Path(result['params_path']).name} "
        f"| {result['n_years']}y seed={result['seed']} ==="
    )
    hdr = f"{'species':12s} {'mean (t)':>14s} {'range (t)':>22s} {'status':>9s} {'×target':>8s} {'CV':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for r in result["species"]:
        rng = f"{r['lower']:,.0f}–{r['upper']:,.0f}"
        print(
            f"{r['species']:12s} {r['mean_tonnes']:>14,.0f} {rng:>22s} "
            f"{r['status']:>9s} {r['factor_vs_target']:>8.2f} {r['cv']:>5.2f}"
        )
    print(f"\nIN ICES RANGE: {result['in_range_count']} / {len(result['species'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, required=True, help="calibration result JSON")
    ap.add_argument("--mode", choices=["bh", "shepherd"], required=True)
    ap.add_argument("--years", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, default=None, help="optional output JSON path")
    args = ap.parse_args()

    result = evaluate(args.params, args.mode, args.years, args.seed)
    _print_report(result)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
