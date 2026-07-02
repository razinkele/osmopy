---
name: scipy DE needs bounded-runtime guards on multi-modal landscapes
description: scipy DE's tol-based termination fails on multi-modal landscapes — population stays diverse while best-fun has clearly converged. Use patience-based early-stop + wall-clock cap. Incident 2026-04-30→05-03 lost a 75h+ result this way before fix landed at cf5cb8e.
type: feedback
originSessionId: af3b28b2-0438-47e9-8b63-2b06b1debe34
---
**Rule:** Every scipy `differential_evolution` call on an OSMOSE-like multi-modal objective must specify both a patience-based early-stop and a wall-clock cap, AND must checkpoint best-x to disk so SIGTERM is non-destructive. Defaults in `scripts/calibrate_baltic.py`: `--patience 20 --wall-clock-cap-h 12 --checkpoint-every 5`.

**Why:** A phase 12 calibration started 2026-04-30 09:48 ran for **75h+** before being killed manually on 2026-05-03. DE found f=1.7735 at step 65 (~42h), then plateaued at that value for 40 more generations / ~33h with no further improvement. scipy's `tol=0.005` never triggered convergence because population diversity stayed high in the multi-modal landscape (the population had members exploring multiple basins; the *best* member was stable). The result was completely lost on kill because the running process was launched on a SHA that pre-dated the checkpointing commit by one day.

The fundamental issue is that scipy DE's `tol` checks `population_std < tol × |mean_fun|` — a *population-convergence* criterion. On smooth unimodal landscapes (sphere, Rosenbrock) the population collapses around the optimum and tol fires. On multi-modal landscapes (OSMOSE phase 12 with B-H), the population stays scattered across basins precisely BECAUSE the landscape has multiple basins; tol never fires. The right termination criterion is "best-objective hasn't improved for N generations" — what every ML training loop uses.

**How to apply:**
- Default to `--patience 20` for any new DE-based optimizer in this repo. Tune lower (5-10) for fast iteration, higher (40+) for production runs you want to be very thorough.
- Always set `--wall-clock-cap-h H` for any run that might run unattended. Even 24h is better than no cap; cap-of-zero is `--wall-clock-cap-h 0`.
- Always set `--checkpoint-every 5` (or similar) for any DE run longer than ~1h. The atomic-snapshot JSON at `data/baltic/calibration_results/phase{N}_checkpoint.json` is what makes a SIGTERM survivable.
- Patience `rel_improvement_threshold` defaults to 1e-6 — guards against floating-point noise resetting the counter. For very noisy objectives (multi-seed DE), bump to 1e-3 (0.1% required).
- The wall-clock cap is wall-clock-of-DE-only, not multi-seed validation. Multi-seed validation happens after DE returns and is bounded by `n_seeds`.

**Scope:** scipy `differential_evolution` only. CMA-ES (`run_cmaes`) has its own `maxiter` and `tolfun` and behaves well in practice. Surrogate-DE has fixed `n_iterations × n_topk` budget by construction. The patience+cap problem is DE-specific.

**Reference commits:**
- `cf5cb8e` (2026-05-03) — bounded-runtime DE: patience + wall-clock cap + bug fix
- `a6c596b` (2026-05-02) — DE intermediate checkpointing
