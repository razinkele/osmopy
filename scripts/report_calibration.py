#!/usr/bin/env python3
"""Report on the most recent Baltic calibration run.

Produces a side-by-side comparison: pre-calibration 50-y equilibrium (if
available in /tmp/osmose_baltic_50y) vs. post-calibration 50-y equilibrium
(freshly simulated here) vs. ICES biomass targets.

Usage:
    .venv/bin/python scripts/report_calibration.py [--phase 1]
                                                    [--baseline DIR]
                                                    [--seeds 3]

The --baseline flag points at a pre-existing output directory (e.g. the
/tmp/osmose_baltic_50y folder produced during the 2026-04-22 session).
If the baseline is missing, only post-calibration is shown.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_baltic import (  # noqa: E402
    get_phase1_params,
    get_phase1b_params,
    get_phase1c_params,
    get_phase1e_params,
    get_phase1f_params,
    get_phase1g_params,
    get_phase2_params,
    get_phase12_params,
)

BALTIC_CONFIG = PROJECT_ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
TARGETS_CSV = PROJECT_ROOT / "data" / "baltic" / "reference" / "biomass_targets.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "baltic" / "calibration_results"

SPECIES = ["cod", "herring", "sprat", "flounder", "perch", "pikeperch", "smelt", "stickleback"]


def _expected_param_keys(phase: str) -> list[str] | None:
    phase_getters = {
        "1": get_phase1_params,
        "1b": get_phase1b_params,
        "1c": get_phase1c_params,
        "1d": get_phase1b_params,
        "1e": get_phase1e_params,
        "1f": get_phase1f_params,
        "1g": get_phase1g_params,
        "2": get_phase2_params,
        "12": get_phase12_params,
    }
    getter = phase_getters.get(phase)
    if getter is None:
        return None
    keys, _, _ = getter()
    return keys


def warn_if_result_schema_stale(phase: str, data: dict) -> None:
    expected = _expected_param_keys(phase)
    if expected is None:
        return
    actual = set(data.get("parameters", {}))
    expected_set = set(expected)
    missing = [key for key in expected if key not in actual]
    extra = sorted(actual - expected_set)
    if missing or extra:
        print(
            "\nWARNING: saved calibration result does not match the current "
            f"phase {phase} parameter definition."
        )
        print(f"  Expected {len(expected)} params; JSON has {len(actual)}.")
        if missing:
            print(f"  Missing: {', '.join(missing)}")
        if extra:
            print(f"  Extra: {', '.join(extra)}")
        print("  Treat this report as a legacy-result check, not a fresh calibration.")


def load_targets():
    t = {}
    for ln in TARGETS_CSV.read_text().splitlines():
        if ln.startswith("#") or ln.startswith("species,") or not ln.strip():
            continue
        p = ln.split(",", 4)
        if len(p) >= 4:
            try:
                t[p[0].strip()] = (float(p[1]), float(p[2]), float(p[3]))
            except ValueError:
                pass
    return t


def load_biomass(output_dir: Path) -> pd.DataFrame:
    bio_file = output_dir / "osm_biomass_Simu0.csv"
    if not bio_file.exists():
        raise FileNotFoundError(bio_file)
    return pd.read_csv(bio_file, skiprows=1)


def run_calibrated_sim(
    overrides: dict[str, str], n_years: int, seed: int, out_root: Path
) -> pd.DataFrame:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    reader = OsmoseConfigReader()
    cfg = reader.read(BALTIC_CONFIG)
    cfg["_osmose.config.dir"] = str((PROJECT_ROOT / "data" / "baltic").resolve())
    cfg["simulation.time.nyear"] = str(n_years)
    cfg["output.spatial.enabled"] = "false"
    cfg["output.recordfrequency.ndt"] = "24"
    cfg.update(overrides)

    seed_dir = out_root / f"seed{seed}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True)
    PythonEngine().run(cfg, seed_dir, seed=seed)
    return load_biomass(seed_dir)


def summarise(bio: pd.DataFrame, n_last: int = 5) -> pd.Series:
    return bio.iloc[-n_last:][SPECIES].mean()


def verdict(sim: float, lo: float, hi: float) -> str:
    if sim <= 0:
        return "EXTINCT"
    if sim < lo:
        return f"LOW ×{sim / lo:.2f}"
    if sim > hi:
        return f"HIGH ×{sim / hi:.1f}"
    return "in range ✓"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="1")
    ap.add_argument(
        "--baseline",
        default="/tmp/osmose_baltic_50y",
        help="Pre-calibration output dir (contains osm_biomass_Simu0.csv)",
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--years", type=int, default=50)
    args = ap.parse_args()

    results_file = RESULTS_DIR / f"phase{args.phase}_results.json"
    if not results_file.exists():
        raise SystemExit(f"No calibration results at {results_file}")

    with open(results_file) as f:
        data = json.load(f)
    warn_if_result_schema_stale(args.phase, data)

    # Stack phase 1 under phase 2 (but not under phase 12 — its JSON already has both).
    stacked_overrides: dict[str, str] = {}
    if args.phase == "2":
        p1_file = RESULTS_DIR / "phase1_results.json"
        if p1_file.exists():
            with open(p1_file) as f:
                p1 = json.load(f)
            for k, v in p1.get("parameters", {}).items():
                stacked_overrides[k] = str(v)
            print(f"Stacked: phase1 ({len(stacked_overrides)} params) + phase2")
    elif args.phase == "12":
        print("Phase 12 results are expected to be self-contained — no stacking needed")

    print(f"=== Calibration report — phase {args.phase} ===")
    print(f"Evaluations: {data['n_evaluations']}")
    print(f"Runtime:     {data['elapsed_seconds']:.0f}s ({data['elapsed_seconds'] / 60:.1f} min)")
    print(f"Objective (single seed): {data['objective_single_seed']:.4f}")
    mo = data.get("objective_multiseed_mean")
    if mo is not None:
        print(
            f"Objective (multi-seed mean): {mo:.4f} "
            f"(std {data.get('objective_multiseed_std', 0):.4f})"
        )

    print("\nOptimized parameters:")
    for k, v in sorted(data["parameters"].items()):
        log10 = data["log10_parameters"][k]
        lo, hi = data["bounds_log10"][k]
        frac = (log10 - lo) / (hi - lo) if hi > lo else 0.5
        bar = "·" * int(frac * 20) + "●" + "·" * (19 - int(frac * 20))
        print(f"  {k:45s} {v:10.4g}   log10={log10:+.3f}  |{bar}|")

    targets = load_targets()
    overrides = dict(stacked_overrides)
    overrides.update({k: str(v) for k, v in data["parameters"].items()})

    # Pre-calibration reference
    baseline = None
    b_path = Path(args.baseline)
    if b_path.is_dir() and (b_path / "osm_biomass_Simu0.csv").exists():
        try:
            baseline = summarise(load_biomass(b_path))
            print(f"\nBaseline biomass loaded from {b_path}")
        except Exception as e:
            print(f"Baseline read failed: {e}")

    # Run post-calibration sims
    print(f"\nRunning {args.seeds} × {args.years}-year validation...")
    out_root = Path(tempfile.mkdtemp(prefix="osmose_postcal_"))
    post_runs = []
    t0 = time.time()
    for s in range(args.seeds):
        bio = run_calibrated_sim(overrides, args.years, seed=s, out_root=out_root)
        post_runs.append(summarise(bio))
    print(f"  Done in {time.time() - t0:.1f}s  (outputs in {out_root})")

    post_df = pd.DataFrame(post_runs)
    post_mean = post_df.mean()
    post_cv = post_df.std() / post_mean.replace(0, np.nan)

    # Report table
    print(
        f"\n{'species':12s} {'baseline':>14s}  {'post-cal':>14s}  "
        f"{'CV':>6s}  {'target':>11s}  {'range':>22s}  verdict"
    )
    n_in = 0
    for sp in SPECIES:
        base = baseline[sp] if baseline is not None else None
        post = post_mean[sp]
        cv = post_cv.get(sp, float("nan"))
        tgt, lo, hi = targets[sp]
        v = verdict(post, lo, hi)
        if "in range" in v:
            n_in += 1
        base_str = f"{base:14,.0f}" if base is not None else f"{'—':>14s}"
        print(
            f"{sp:12s} {base_str}  {post:14,.0f}  {cv:6.3f}  {tgt:11,.0f}  "
            f"{f'{lo:,.0f}-{hi:,.0f}':>22s}  {v}"
        )
    print(
        f"\n{n_in}/{len(SPECIES)} species in ICES biomass range "
        f"after calibration (pre-cal was 1/8 per 2026-04-22 memory)"
    )


if __name__ == "__main__":
    main()
