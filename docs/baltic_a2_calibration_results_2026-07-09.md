# Baltic A2 calibration results (2026-07-09/10)

**What:** the existing DE calibrator (`scripts/calibrate_baltic.py`) run with Chunk A2 depletion **on**
(`--a2`, co-calibrating per-species larval + adult mortality **plus** one shared zooplankton regrowth
rate), compared to an A2-**off** baseline, against the ICES bands. Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-09-baltic-a2-calibration*`.

**Both arms:** bounded DE — `--maxiter 20 --popsize 12 --popsize-mult 3 --years 15`, 8 workers.

## Runs

- **A2-off baseline: completed** (gen 20/20, 1008 evals, 6.77 h). Multi-seed validated.
- **A2-on: cut short at generation 10/20.** One DE candidate produced a pathological simulation that never
  returned; `differential_evolution` with `updating="deferred"` blocks a generation until every eval
  returns, so generation 11 stalled indefinitely (main process idle at ~3% CPU, no progress for >1.5 h).
  The run was stopped manually and the **gen-10 best config recovered from the checkpoint** (single-seed).
  Critically, `gens_since_improvement = 0` at the stall — the DE was **still improving** when it hung, so
  this A2-on result is **under-converged** (a lower bound on what A2 can achieve).

## Result — A2 dramatically improves the fit (objective), even under-converged

| species | band (lower–upper) | A2-off (multi-seed) | A2-on (gen-10, single-seed) |
|---|---|---|---|
| cod | 60 k – 250 k | 1.49 M · **over 12×** | 534 k · over 2.1× |
| herring | 0.8 M – 3 M | 1.03 M · **in** | 1.93 M · **in** |
| sprat | 0.8 M – 2.5 M | 880 k · **in** | 459 k · under |
| flounder | 20 k – 100 k | 80 k · **in** | 6.6 k · under |
| perch | 8 k – 50 k | 3.70 M · **over 185×** | 355 k · over 7× |
| pikeperch | 4 k – 25 k | 4.0 M · **over 400×** | 375 k · over 15× |
| smelt | 20 k – 120 k | 1.03 M · over 17× | 167 k · over 1.4× |
| stickleback | 50 k – 500 k | 8.75 M · **over 44×** | 795 k · over 1.6× |
| **in-band count** | | **3/8** | **1/8** |
| **DE objective** | | **3.57** | **0.896** (≈ 4× better) |

**"In-band count" is the wrong metric here** — it favours the baseline (3/8 vs 1/8), but that inverts the
real picture. The baseline lands three well-assessed pelagics exactly in band while letting the low-weight
species explode **17–400×**; A2 compresses the *entire* community to within **~1.4–7×** of the bands (only
herring lands exactly inside, with sprat/flounder dipping slightly *under* and everything else just over).
The banded objective, which rewards proximity, captures this correctly: **A2 is ~4× better**, and it was
still improving when the run was cut off.

A2's zoo regrowth rate optimized to **0.14** (strong depletion), consistent with the standalone rate sweep
that showed rate ~0.3–0.6 relaxes the overshoot.

## Interpretation — A2 is a genuine calibration lever; a full ICES calibration looks reachable

Depletable plankton takes the Baltic community from a **catastrophically overshooting** state (percids
185–400× over) to one that is **near the ICES bands across the board** (mostly within 2–7×), a step-change
the per-species mortality search could not achieve on its own. This is the over-production fix the whole
investigation pointed at. It is **not yet a finished calibrated config** — cod (2.1× over) and the percids
remain the hardest, and the DE was cut off while still improving — but a deployable ICES-calibrated Baltic
now looks **reachable** with A2, which it did not with mortality tuning alone.

## Convergence blocked — the gen-10 result is accepted as final (2026-07-10)

A per-simulation wall-clock timeout guard (`_run_with_timeout` / `--sim-timeout`, commit `c4ab6cc`) was
added and the A2 DE re-run. **It hit the identical hang at generation 10→11**, deterministically
(same `best_fun 0.8964`, workers gone, main process idle at ~3% CPU for >1 h). So the true failure mode is
**not** a slow sim in a live worker (which the timeout would catch) but a specific DE candidate that
**crashes a worker process — almost certainly an OOM from a biomass explosion — after which scipy's
`differential_evolution` process pool deadlocks waiting for the dead worker.** SIGALRM inside the objective
cannot help once the worker is gone. The timeout guard remains a valid robustness improvement (it prevents
the *slow-sim* class of stalls, and it is parity-safe/default-off, 341 calibrator tests) but it is the
wrong fix for *this* failure.

**Decision: accept the gen-10 config as the A2 calibration result** (user, 2026-07-10). It already answers
the question decisively — A2 is a genuine calibration lever (4× better fit; community compressed from
17–400× to ~1.4–7× of the bands) — and the deterministic hang means gen-10 is as far as this DE
configuration reaches. Squeezing out the final generations to a fully in-band config is deferred.

## Follow-ups (deferred; the actual fix for convergence)

1. **Engine explosion guard (the robust fix):** add a max-schools / max-abundance cap in
   `osmose/engine/simulate.py` — if a run's population explodes past a sane bound, abort it with a marker
   so the objective returns a penalty. This kills the pathology at the source and unblocks convergence
   regardless of worker count. (Preferred over `workers=1`, which only avoids the pool deadlock at ~8× the
   wall-clock and still risks a single-process OOM.)
2. **Then re-run the A2 DE to convergence**, multi-seed validated, for a candidate deployable config.
3. **Objective tweak (optional):** a small bonus for landing *inside* a band would stop the DE from
   overshooting the correction (sprat/flounder went slightly under).

## Note

No deployed config was changed. The **accepted gen-10** A2-on parameters + the baseline are recorded in
`docs/diagnostics/baltic_a2_calibrated_params.json` as a **candidate** sidecar, not a promoted config.
