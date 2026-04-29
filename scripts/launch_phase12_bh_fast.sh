#!/usr/bin/env bash
# Launch wrapper for phase 12 B-H calibration with Tier A+B speedups.
#
# Combined effect vs. the 2026-04-28 baseline run (~22h for 3 seeds):
#   A1: workers 24    (was 8)        ~3×    CPU saturation on 28-core box
#   A2: popsize_mult 5 (was 10)      ~2×    half the evals per generation
#   A3: tol 0.005     (was 0.001)    ~1.3×  earlier convergence
#   B1: warm-start from prior JSON   ~1.7×  skip known-bad regions
# Net projection: ~3h wall-clock for the same 3-seed multi-seed run.
#
# Also: PYTHONUNBUFFERED=1 fixes the stdout buffering issue that hid scipy DE
# per-generation output from logs in the prior run.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/phase12_bh_fast_${TS}.log"
PIDFILE="logs/phase12_bh_fast_${TS}.pid"

# Cod sp0 adult mortality is excluded from warm-start: the prior optimum
# (log10≈0.57) sat against the cod-floor ceiling (-0.523, 0.7), but the new
# search uses (-3.0, 0.7) and B-H now closes the high-SSB compensation
# pathway — that warm start would bias the search toward an artefact.
SKIP_KEYS="mortality.additional.rate.sp0"
WARM_FROM="data/baltic/calibration_results/phase12_results.json"

OSMOSE_DE_WORKERS=24 \
PYTHONUNBUFFERED=1 \
setsid .venv/bin/python scripts/calibrate_baltic.py \
    --phase 12 \
    --seeds 3 \
    --years 50 \
    --maxiter 200 \
    --popsize 15 \
    --popsize-mult 5 \
    --tol 0.005 \
    --warm-start "$WARM_FROM" \
    --skip-warm-start-keys "$SKIP_KEYS" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PIDFILE"
disown "$PID" || true

SHA=$(git rev-parse HEAD)

echo "LAUNCHED pid=$PID"
echo "LOG=$LOG"
echo "PIDFILE=$PIDFILE"
echo "SHA=$SHA"
echo ""
echo "Tier A+B speedups active. Expected wall-clock: ~3h (vs ~22h baseline)."
