#!/usr/bin/env python3
"""Compare two phase-12 calibration results side-by-side.

Designed for the 2026-04-25 engine-fix follow-up: contrasts the prior
phase 12 run (background-species pathway broken, predators inactive) with
the post-fix run (predators active, same parameter bounds).

Usage:
    .venv/bin/python scripts/compare_phase12_runs.py \\
        --before data/baltic/calibration_results/phase12_results.no-predators.json \\
        --after data/baltic/calibration_results/phase12_results.json

Both files must have the same parameter keys; only optimized values may
differ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    args = ap.parse_args()

    if not args.before.exists():
        raise SystemExit(f"missing: {args.before}")
    if not args.after.exists():
        raise SystemExit(f"missing: {args.after}")

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())

    print(f"=== Phase 12 comparison ===")
    print(f"BEFORE: {args.before.name}")
    print(f"  obj single-seed: {before['objective_single_seed']:.4f}")
    print(f"  obj multi-seed mean: {before.get('objective_multiseed_mean', 'n/a')}")
    print(f"  evals: {before.get('n_evaluations', 'n/a')}")
    print(f"  runtime: {before.get('elapsed_seconds', 0)/3600:.2f} h")
    print()
    print(f"AFTER:  {args.after.name}")
    print(f"  obj single-seed: {after['objective_single_seed']:.4f}")
    print(f"  obj multi-seed mean: {after.get('objective_multiseed_mean', 'n/a')}")
    print(f"  evals: {after.get('n_evaluations', 'n/a')}")
    print(f"  runtime: {after.get('elapsed_seconds', 0)/3600:.2f} h")
    print()

    delta = before["objective_single_seed"] - after["objective_single_seed"]
    pct = 100 * delta / before["objective_single_seed"]
    arrow = "↓" if delta > 0 else "↑"
    print(f"OBJECTIVE Δ: {arrow} {abs(delta):.4f}  ({pct:+.1f}%)")
    print()

    # Per-parameter comparison
    print(f"{'parameter':45s} {'before':>10s}  {'after':>10s}  {'Δ%':>8s}")
    print("-" * 80)
    keys = sorted(before["parameters"].keys())
    for k in keys:
        b = before["parameters"].get(k)
        a = after["parameters"].get(k)
        if b is None or a is None:
            continue
        if b == 0:
            d_pct = float("nan")
        else:
            d_pct = 100 * (a - b) / b
        print(f"  {k:45s} {b:10.4g}  {a:10.4g}  {d_pct:+7.1f}%")

    # Highlight which params moved most
    print()
    print("Largest parameter changes (top 5 by abs Δ%):")
    changes = []
    for k in keys:
        b = before["parameters"].get(k)
        a = after["parameters"].get(k)
        if b is None or a is None or b == 0:
            continue
        changes.append((k, b, a, 100 * abs(a - b) / b))
    changes.sort(key=lambda x: -x[3])
    for k, b, a, d in changes[:5]:
        direction = "raised" if a > b else "lowered"
        print(f"  {k}: {direction} {b:.4g} → {a:.4g} ({d:+.0f}%)")


if __name__ == "__main__":
    main()
