#!/usr/bin/env python3
"""Sobol sensitivity analysis for Baltic phase 12 calibration parameters.

Quantifies each parameter's first-order (S1) and total (ST) contribution to
variance of the phase 12 objective. Output: a ranked CSV that identifies
parameters worth tuning vs. parameters that can be fixed at literature
defaults — permanently shrinking the DE search dimensionality.

Pipeline:
1. Pull the 27 phase 12 params + log10 bounds from `get_phase12_params()`.
2. Generate Saltelli samples — total = n_base * (2*D + 2). For D=27, n_base=256
   → 14,336 evaluations.
3. Evaluate each sample in parallel via ProcessPoolExecutor. Each worker
   initialises one `_ObjectiveWrapper` (picklable) and reuses it.
4. Run SALib Sobol decomposition. Output ranked CSV + JSON + raw Y values.

Cost: ~25 evals/min on 24 workers ≈ 9.5h for n_base=256. Use n_base=128 for
~4.7h diagnostic at slightly looser confidence intervals.

Resume support: writes Y values incrementally to a CSV; pass --resume to
continue from where a prior interrupted run left off.

Usage:
    .venv/bin/python scripts/sensitivity_phase12.py \\
        --n-base 256 --seed 42 --n-years 50 --workers 24

The single-seed limitation is deliberate: SALib's variance attribution
assumes a deterministic objective, so multi-seed averaging would muddy
the interaction-vs-noise distinction. The seed=42 result is a structural
signal about the *deterministic* component of the objective; per-seed
variance is a separate (orthogonal) concern handled by multi-seed
re-ranking in calibrate_baltic.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osmose.calibration.sensitivity import SensitivityAnalyzer  # noqa: E402
from osmose.config.reader import OsmoseConfigReader  # noqa: E402
from scripts.calibrate_baltic import (  # noqa: E402
    BALTIC_CONFIG,
    get_phase12_params,
    load_targets,
    make_objective,
)

DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "data" / "baltic" / "calibration_results" / "sensitivity"
)
DEFAULT_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Worker-side state (initialised once per worker, reused across evals)
# ---------------------------------------------------------------------------
_OBJECTIVE = None


def _pool_init(base_config, targets, param_keys, n_years, seed):
    """Initialiser run once per worker process — builds the objective wrapper."""
    global _OBJECTIVE
    _OBJECTIVE = make_objective(
        base_config, targets, param_keys,
        n_years=n_years, seed=seed, use_log_space=True,
    )


def _eval_one(args):
    """Evaluate one Sobol sample. Returns (idx, objective). NaN on failure.

    Exceptions are logged to stderr (type, message, traceback) before the NaN
    return so genuine programming bugs in workers don't silently masquerade
    as scientific NaN — a 14k-eval job with a systematic bug would otherwise
    surface only at the analysis step after hours of compute.
    """
    import sys as _sys
    import traceback as _tb

    idx, x = args
    try:
        val = float(_OBJECTIVE(np.asarray(x, dtype=float)))
        if not np.isfinite(val):
            val = float("nan")
    except Exception as exc:
        print(
            f"[sensitivity worker] idx={idx} FAILED: "
            f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}",
            file=_sys.stderr,
            flush=True,
        )
        val = float("nan")
    return idx, val


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------
def _load_existing_y(y_csv: Path, n_samples: int) -> tuple[np.ndarray, set[int]]:
    """Load Y values from a previous run.

    Only finite values are added to `done`; rows with NaN are loaded into Y
    (so the analysis step can flag them) but excluded from `done`, which means
    --resume will re-evaluate them. Without this, a NaN'd index would loop
    forever between "ERROR: NaN samples — re-run via --resume" and a resume
    that skips it as already-done.
    """
    Y = np.full(n_samples, np.nan)
    done: set[int] = set()
    if not y_csv.exists():
        return Y, done
    with open(y_csv) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            idx = int(row[0])
            if 0 <= idx < n_samples:
                val = float(row[1])
                Y[idx] = val
                if np.isfinite(val):
                    done.add(idx)
    return Y, done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sobol sensitivity analysis for Baltic phase 12 parameters",
    )
    parser.add_argument("--n-base", type=int, default=256,
                        help="Saltelli base count; total evals = n_base * (2*D + 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for both Sobol sampling and OSMOSE simulations")
    parser.add_argument("--n-years", type=int, default=50,
                        help="Simulation years per evaluation. Note: calibrate_baltic.py "
                             "phase 12 default is 40; using 50 here gives a longer "
                             "equilibrium window for cleaner sensitivity signal but "
                             "introduces a methodological gap with the DE search it "
                             "informs. Match to 40 for strict consistency.")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("OSMOSE_DE_WORKERS", "24")),
                        help="Parallel workers (default: $OSMOSE_DE_WORKERS or 24)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help=f"Output directory (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"ST threshold for TUNE/FIX recommendation (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing y_*.csv in output-dir; NaN'd rows "
                             "are auto-retried.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing y_*.csv. DESTROYS prior work — "
                             "prefer --resume unless intentionally restarting.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only — print expected eval count + ETA, don't run")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Param definitions
    param_keys, bounds, _ = get_phase12_params()
    n_params = len(param_keys)

    # 2. Saltelli samples
    analyzer = SensitivityAnalyzer(param_keys, bounds)
    X = analyzer.generate_samples(n_base=args.n_base)
    n_samples = X.shape[0]
    expected = args.n_base * (2 * n_params + 2)

    print("=== Sobol Sensitivity — Baltic Phase 12 ===", flush=True)
    print(f"Params: {n_params} (log10-space)")
    print(f"n_base: {args.n_base}; total samples: {n_samples} (expected {expected})")
    print(f"Workers: {args.workers}; n_years: {args.n_years}; seed: {args.seed}")
    print(f"Output: {args.output_dir}")

    if args.dry_run:
        # Empirical rate from the 2026-04-28 phase 12 run: ~3 evals/min overall
        # at 8 workers ≈ 0.4 evals/worker/min in steady state on a 28-core box.
        # Per-worker rate degrades slightly past 16 workers due to memory-bandwidth
        # contention, so use a conservative 0.35/worker/min for high counts.
        per_worker = 0.4 if args.workers <= 16 else 0.35
        rate = max(1.0, args.workers * per_worker)
        eta_h = n_samples / rate / 60.0
        print(f"\nDry run: estimated wall-clock ≈ {eta_h:.1f}h at {rate:.1f} evals/min "
              f"(assumes {per_worker} evals/worker/min)")
        return 0

    # 3. Resume support — refuse to truncate prior work without explicit consent.
    # A 14-28h job's Y values must not be silently destroyed by a re-launch
    # that forgot --resume.
    y_csv = args.output_dir / f"y_n{args.n_base}_seed{args.seed}.csv"
    if y_csv.exists() and not args.resume and not args.force:
        print(
            f"ERROR: {y_csv} already exists from a prior run.\n"
            f"  → Pass --resume to continue evaluating remaining samples.\n"
            f"  → Pass --force to overwrite (DESTROYS prior work).\n"
            f"  → Or delete the file manually to start fresh.",
            file=sys.stderr,
        )
        return 1

    if args.resume and y_csv.exists():
        Y, done = _load_existing_y(y_csv, n_samples)
        if done:
            print(f"Resuming with {len(done)}/{n_samples} samples already evaluated")
        # Open in append mode; existing rows (including any NaN-marked ones,
        # which are NOT in `done` and will be re-evaluated) stay in the file.
    else:
        Y = np.full(n_samples, np.nan)
        done = set()
        with open(y_csv, "w") as f:
            f.write("idx,objective\n")

    todo = [(i, X[i].tolist()) for i in range(n_samples) if i not in done]
    if not todo:
        print("All samples already evaluated; skipping to analysis.")
    else:
        # 4. Load shared state once (workers reuse via _pool_init)
        reader = OsmoseConfigReader()
        base_config = reader.read(BALTIC_CONFIG)
        targets = load_targets()

        print(f"\nEvaluating {len(todo)} samples...", flush=True)
        t0 = time.time()
        completed = len(done)
        nan_count = 0

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_pool_init,
            initargs=(base_config, targets, param_keys, args.n_years, args.seed),
        ) as pool, open(y_csv, "a", buffering=1) as ylog:
            futures = {pool.submit(_eval_one, t): t[0] for t in todo}
            for fut in as_completed(futures):
                idx, val = fut.result()
                Y[idx] = val
                ylog.write(f"{idx},{val:.6f}\n")
                completed += 1
                if not np.isfinite(val):
                    nan_count += 1
                if completed % 50 == 0 or completed == n_samples:
                    elapsed = time.time() - t0
                    new_done = completed - len(done)
                    rate_per_min = (new_done / max(elapsed, 1.0)) * 60.0
                    remaining = n_samples - completed
                    eta_h = remaining / max(rate_per_min, 0.1) / 60.0
                    print(
                        f"  [{completed}/{n_samples}] "
                        f"rate={rate_per_min:.1f}/min  eta={eta_h:.2f}h  "
                        f"nan={nan_count}",
                        flush=True,
                    )

    # 5. Sobol decomposition. SALib raises on NaN — fail-loud rather than mask.
    nan_mask = ~np.isfinite(Y)
    if nan_mask.any():
        print(
            f"\nERROR: {int(nan_mask.sum())} NaN/inf samples — Sobol analysis "
            f"requires complete Y. Re-run those indices via --resume.",
            file=sys.stderr,
        )
        nan_idx = np.flatnonzero(nan_mask)[:10]
        print(f"  First failing indices: {nan_idx.tolist()}", file=sys.stderr)
        return 2

    print("\nRunning Sobol analysis...", flush=True)
    Si = analyzer.analyze(Y)

    # 6. Ranked output (sorted by ST descending)
    rows = []
    for i, key in enumerate(param_keys):
        st = float(Si["ST"][i])
        rows.append({
            "param": key,
            "S1": round(float(Si["S1"][i]), 6),
            "S1_conf": round(float(Si["S1_conf"][i]), 6),
            "ST": round(st, 6),
            "ST_conf": round(float(Si["ST_conf"][i]), 6),
            "recommend": "TUNE" if st >= args.threshold else "FIX",
        })
    rows.sort(key=lambda r: r["ST"], reverse=True)

    summary_csv = args.output_dir / f"sobol_n{args.n_base}_seed{args.seed}.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_tune = sum(1 for r in rows if r["recommend"] == "TUNE")
    n_fix = len(rows) - n_tune

    summary_json = args.output_dir / f"sobol_n{args.n_base}_seed{args.seed}.json"
    with open(summary_json, "w") as f:
        json.dump({
            "n_base": args.n_base,
            "n_samples": n_samples,
            "seed": args.seed,
            "n_years": args.n_years,
            "threshold": args.threshold,
            "n_tune": n_tune,
            "n_fix": n_fix,
            "ranked_by_ST": rows,
        }, f, indent=2)

    # 7. Print summary
    print("\n=== Ranked by Total Sobol Index (ST) ===")
    print(f"{'Rank':<4} {'Param':<48} {'S1':>10} {'ST':>10} {'Action':>8}")
    print("-" * 84)
    for rank, r in enumerate(rows, 1):
        print(f"{rank:<4} {r['param']:<48} {r['S1']:>10.4f} {r['ST']:>10.4f} {r['recommend']:>8}")

    print(f"\n→ TUNE: {n_tune} params (ST ≥ {args.threshold})")
    print(f"→ FIX:  {n_fix} params (ST < {args.threshold})")
    if n_fix > 0:
        fix_keys = [r["param"] for r in rows if r["recommend"] == "FIX"]
        print("\nNext-run --skip-warm-start-keys candidate set "
              "(or hardcode these to literature defaults):")
        print(f"  {','.join(fix_keys)}")

    print("\nResults written:")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")
    print(f"  {y_csv}  (raw Y values, for re-analysis)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
