---
name: OSMOSE_DE_WORKERS=16 is the safe default on a 28-core box
description: OSMOSE_DE_WORKERS=24 produces severe memory-bandwidth contention. Per-worker eval rate halves and total throughput stalls at ~3 evals/min vs the projected 8+. 16 workers is the sweet spot — leaves headroom for OS + forkserver overhead and stays below the bandwidth knee.
type: feedback
originSessionId: af3b28b2-0438-47e9-8b63-2b06b1debe34
---
**Rule:** Set `OSMOSE_DE_WORKERS=16` (not 24, not 28) when launching DE-based calibration on the 28-core OSMOSE workstation.

**Why:** On 2026-04-30 a phase 12 calibration was launched with `OSMOSE_DE_WORKERS=24` to maximise CPU saturation. Empirical eval rate over 75h+ was **3 evals/min sustained** — not the projected 8+ evals/min that 3× more workers should have given vs the 8-worker baseline (175 evals/h ≈ 3 evals/min). Per-worker rate dropped from 0.4 evals/min (at 8 workers) to 0.13 evals/min (at 24 workers).

Memory bandwidth on the 28-core machine is the binding constraint, not CPU cycles. Each 50-y OSMOSE simulation touches a ~400 MB working set; 24 workers × 400 MB = 9.6 GB of active hot data competing for the same memory subsystem. The CPU cores idle waiting for memory loads. Linux `load average` confirmed: 165–200 on a 28-core box (5.9–7.1× cores) with 24 workers, but 24% wall-clock CPU on the parent process — workers were stalled in iowait/D states.

16 workers gives a ~7 GB working set that fits more comfortably in memory bandwidth, with ~50% CPU saturation, which empirically delivers higher TOTAL throughput than the 24-worker oversubscription.

**How to apply:**
- The launch wrapper at `scripts/launch_phase12_bh_fast.sh` was updated 2026-05-03 (commit cf5cb8e) to default to `OSMOSE_DE_WORKERS=16`. Use it.
- For ad-hoc launches, prefer `OSMOSE_DE_WORKERS=16` unless you have evidence that the new workload is CPU-bound rather than memory-bandwidth-bound.
- If the box hardware changes (more cores OR much more memory bandwidth), revisit. Specifically: a machine with ≥64 GB RAM and DDR5 / 8-channel memory could likely sustain 24+ workers at full per-worker rate.
- Diagnostic: if `top` shows CPU idle > 30% with high load average, you're bandwidth-bound and over-subscribed; reduce workers.

**Reference incident:** 75h+ phase 12 run 2026-04-30 → 2026-05-03 (PID 1029206 on SHA fe0c04c). Killed at step 104 with f=1.7735 (best phase 12 result on record) lost because the SHA pre-dated checkpointing.
