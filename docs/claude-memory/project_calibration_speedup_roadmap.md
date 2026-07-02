---
name: Calibration speedup roadmap (state)
description: 2026-04-29 — Tier A+B+C1+C2+D4 of the calibration speedup roadmap shipped on master; C3/D2/D3 deferred pending profiling. Combined effect: ~22h baseline → ~3h projected for phase 12 multi-seed.
type: project
originSessionId: af3b28b2-0438-47e9-8b63-2b06b1debe34
---
**Shipped (2026-04-29, commits 1ae7630 → 135aa8f):**

| Tier | What | Where |
|---|---|---|
| A1 | `OSMOSE_DE_WORKERS` env var | already existed, recommend 24 on 28-core box |
| A2 | `--popsize-mult` knob (default 10, recommend 5) | calibrate_baltic.py CLI |
| A3 | `--tol` knob (default 0.005; was hardcoded 0.001) | 1ae7630 |
| B1 | `--warm-start <json>` + `--skip-warm-start-keys` | 1ae7630 |
| — | `scripts/launch_phase12_bh_fast.sh` (combined-knob wrapper, PYTHONUNBUFFERED=1) | f775df3 |
| C1 | GP surrogate-assisted DE (`osmose/calibration/surrogate_de.py`) | 8649e55 + review-fix 8edffa4 |
| C2 | CMA-ES wrapper (`osmose/calibration/cmaes_runner.py`) | 6a90cf6 + review-fix 54b61d1 |
| D4 | Sobol sensitivity script (`scripts/sensitivity_phase12.py`) | 15e610f + review-fix b7a5708 |
| — | `--optimizer {de,cmaes,surrogate-de}` dispatcher in calibrate_baltic.py | 135aa8f |

**Deferred (pending profiling):**
- C3 (persistent worker engines): wait until profiling shows import + EngineConfig.from_dict are >20% of per-eval cost.
- D2 (ProcessPoolExecutor migration): wait until profiling confirms scipy DE's pool is GIL-bound (memory's v0.10.0 note flagged this for the NSGA-II path; unclear if scipy DE is similarly affected).
- D3 (Numba parallel expansion): one `parallel=True` already on `mortality.py:1307`; expanding requires per-loop profiling to identify safe candidates.
- D1 (BoTorch BO): subsumed by C1 — same GP machinery, but C1 wraps it around DE's robust global search instead of trusting BO's acquisition alone.

**Why:** B-H added density-dependent recruitment (v0.11.0) which made phase 12 DE search land in a flatter, multi-modal landscape. Prior phase 12 with 24 params converged in ~2 generations × 240 popsize = 768 evals; the 27-param B-H landscape took 11+ generations and was still grinding at 21h+ in the 2026-04-28 run. Tier A+B+C1+C2+D4 are the response.

**How to apply:**
- For each new calibration run, decide which optimizer empirically. Default recommendation: **CMA-ES**. Synthetic benchmark on 2026-04-29 (commit fe0c04c, `data/benchmarks/optimizer_comparison.json`) showed CMA-ES dominates DE and surrogate-DE by 3–380× at matched eval budgets on sphere/rosenbrock 5D/10D. Reserve vanilla DE for global search of unfamiliar landscapes where CMA-ES might prematurely commit to a local basin; reserve surrogate-DE for explicit budget-constrained runs (≤100 real evals) where the GP overhead amortizes against minute-scale evals — on synthetic problems where evals are nanoseconds, surrogate-DE's machinery overhead swamps its sample-efficiency benefit.
- Empirical eval rate: **175 evals/h with 8 workers** in steady state on the 28-core box (0.4 evals/worker/min); use this for ETA estimation. Above 16 workers degrade to ~0.35/worker/min from memory bandwidth contention.
- The launch wrapper (`launch_phase12_bh_fast.sh`) bundles A1+A2+A3+B1 with `PYTHONUNBUFFERED=1` for live DE per-generation output. Use it; the bug it fixes is that block-buffered stdout under file redirection kept ~17h of DE generation messages invisible.
- Sensitivity analysis cost: n_base=128 → ~14h on 24 workers; n_base=256 → ~28h. Run once per major landscape change (e.g., when adding/removing parameters), then reuse the recommendation across many calibration runs.
- Surrogate-DE iteration cap: keep `n_iterations ≤ 6` regardless of budget. The benchmark's first version computed `n_iterations = remaining // n_topk` naively and hung at iter≈25 because GP fit cost is O(n_train³). The dispatcher in `calibrate_baltic.py` already hardcodes 6, but custom callers must respect this.
- **DE bounded-runtime guards (added 2026-05-03, commit cf5cb8e):** every DE invocation must specify `--patience N --wall-clock-cap-h H --checkpoint-every K`. Defaults (20 / 12 / 5) make a 75h+ marathon structurally impossible — see `feedback_de_bounded_runtime.md` for the incident motivation. The `_make_checkpoint_callback` helper combines all three; bypass only with explicit reason.
- **OSMOSE_DE_WORKERS=16 default (changed 2026-05-03, commit cf5cb8e):** the launch wrapper now defaults to 16 workers, not 24. The 24-worker config is memory-bandwidth-oversubscribed on the 28-core box and produces lower total throughput than 16. See `feedback_de_workers_default.md`.
