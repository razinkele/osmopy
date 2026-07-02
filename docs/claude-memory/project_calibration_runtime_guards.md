---
name: Calibration runtime guards (bounded DE + checkpointing)
description: 2026-05-03 — scripts/calibrate_baltic.py refactored so every scipy DE invocation is bounded by construction. _make_checkpoint_callback now combines checkpointing (every N gens) + patience-based early-stop (after M stale gens) + wall-clock cap (after H hours). Defaults 5/20/12 in calibrate_baltic.py CLI; launch_phase12_bh_fast.sh enforces them. OSMOSE_DE_WORKERS lowered 24→16.
type: project
originSessionId: af3b28b2-0438-47e9-8b63-2b06b1debe34
---
**Status:** SHIPPED on master 2026-05-03. Commits a6c596b (checkpointing) → cf5cb8e (patience + wall-clock + workers default).

**What landed:**

- `scripts/calibrate_baltic.py`:
  - `_make_checkpoint_callback(checkpoint_path, every_n, param_keys, bounds, *, patience, wall_clock_max_seconds, rel_improvement_threshold)` — single callback combining 3 concerns.
  - New `run_calibration` parameters: `checkpoint_every` (5), `patience` (20), `wall_clock_cap_h` (12).
  - New CLI flags: `--checkpoint-every N`, `--patience N`, `--wall-clock-cap-h H` (each `0` disables).
  - The callback writes `data/baltic/calibration_results/phase{N}_checkpoint.json` atomically (tmp + rename) every N gens. Snapshot includes `best_fun`, `best_x_log10`, decoded `best_parameters`, `gens_since_improvement`, `elapsed_seconds`, `timestamp_iso`.
  - Patience-based early stop returns `True` from the callback when M consecutive generations pass without best-fun improvement (relative threshold 1e-6 guards against float noise resetting the counter).
  - Wall-clock cap returns `True` when DE has run for H hours regardless of convergence.

- `scripts/launch_phase12_bh_fast.sh`:
  - `OSMOSE_DE_WORKERS=24` → **16** (memory-bandwidth contention; see `feedback_de_workers_default.md`).
  - All three guards explicit on the command line: `--checkpoint-every 5 --patience 20 --wall-clock-cap-h 12`.

- 18 tests in `tests/test_calibrate_baltic_parallelism.py`, including the original 10 dispatcher/checkpoint tests plus 8 new for patience trigger/disable/reset/oscillation, wall-clock trigger/disable, snapshot diagnostics, CLI exposure.

**Why the refactor (incident, 2026-04-30→05-03):** A phase 12 calibration ran for **75h+** before being killed manually. DE found f=1.7735 at step 65 (~42h) and plateaued for 40 more generations / ~33h with no improvement. scipy's `tol=0.005` never triggered because population diversity stayed high in the multi-modal landscape — `tol` checks population-std, not best-fun-progress. The result was **completely lost** on kill because the running process was on SHA fe0c04c, one commit before checkpointing landed.

**How to apply:**
- For any new DE-based optimization call site in this repo, follow the pattern: build the callback via `_make_checkpoint_callback`, pass it via the dispatcher's `de_callback=` arg. Don't roll your own.
- The launch wrapper is the canonical recipe — always start there for phase 12 work. To override defaults, edit it; don't bypass it.
- If you intentionally want an unbounded run (rare — mostly for science experiments where you trust DE to converge naturally), pass `--patience 0 --wall-clock-cap-h 0`. This is the "I know what I'm doing" mode.
- The `_make_checkpoint_callback` is reusable for non-Baltic calibrations too. Import it from `scripts.calibrate_baltic`.
- If you change the relative improvement threshold, prefer 1e-3 (0.1%) for very noisy multi-seed objectives over the default 1e-6.

**Scope limit:** the guards apply only to the DE branch of `_dispatch_optimizer`. CMA-ES (`run_cmaes`) has its own `maxiter` + `tolfun` and behaves well; surrogate-DE has fixed `n_iterations × n_topk` budget by construction. Both ignore `de_callback`.

**Related memory:**
- `feedback_de_bounded_runtime.md` — the rule + the incident motivating it
- `feedback_de_workers_default.md` — the workers=16 empirical evidence
- `project_calibration_speedup_roadmap.md` — broader Tier A/B/C/D state
