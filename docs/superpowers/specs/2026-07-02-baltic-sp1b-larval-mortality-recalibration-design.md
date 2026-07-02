# SP1b — Baltic cod larval-mortality recalibration (mean-neutral for SP1)

**Date:** 2026-07-02
**Status:** design (awaiting user review)
**Depends on:** SP1 spatial cod egg-survival (branch `baltic-rv-spatial-egg-survival`, PR #97). SP1b builds on that branch.

## Problem

Enabling SP1 multiplies each cod egg-school's larval survival by `clip(RV(cell)/RV_ref, 0, 1)`
(`osmose/engine/processes/natural.py:larva_mortality`). Because ~72% of the 145 `cod_spawning`
cells have little or no reproductive volume, the spatial term removes egg survivors in the
low-RV cells and drops the mean cod biomass ~16% (2054 → 1728 t, 15-yr run, years 3–14 mean;
recorded in `docs/diagnostics/rv_spatial_field.md`).

That mean shift confounds interpretation: we cannot tell whether SP1's spatial structure changes
cod *dynamics* (stability/overshoot) or merely lowers the *level*. SP1b removes the level change
so SP1 becomes a mean-neutral, purely spatial/variance intervention.

## Goal (target metric)

**Mean-neutral vs the SP1-off baseline.** Recalibrate a single cod parameter so that, with SP1
enabled, the long-term mean cod biomass returns to the SP1-off Python-engine baseline within a
tolerance. Explicitly **not** in scope: matching ICES/observed absolute biomass (a larger
multi-parameter calibration), and gating on overshoot reduction (the overshoot ratio is *measured
and recorded*, not a pass/fail here).

- **Baseline** = mean cod biomass over years 3–14 of the 15-yr Baltic run with SP1 disabled,
  seed 0 (same window/seed the SP1 diagnostic already uses). Computed at solve time, not hard-coded.
- **Target tolerance** = |mean_on_recal − baseline| / baseline ≤ 0.02 (±2%).

## The knob

`mortality.additional.larva.rate.sp0` (cod). After the 4.4.0 larval-rate migration
(`osmose/config/aliases.py:_migrate_larva_rate`, which divides the stored per-year rate by
`ndtperyear` on read), the value the engine actually consumes is the **resolved per-cohort rate**
(base file stores `360.0` per year → engine reads `15.0`). SP1b operates on this **resolved**
value in the in-memory config dict (the same dict `PythonEngine.run_in_memory` receives), so it is
migration-agnostic: lowering it raises baseline egg survival to offset the average spatial kill.

Only cod (sp0) is touched — the only SP1-enabled species.

## Approach — 1-D empirical solve

A single-parameter root-find, chosen over the two alternatives:

- *Analytical offset only* — `d1 = d0 + ln(E[clip])` from the field. A good first guess but will
  miss the target: Beverton-Holt density dependence makes the survival→biomass map nonlinear, so a
  ~19% survival boost is not a ~16% biomass gain. **Used as the initial bracket, not the answer.**
- *Full DE/CMA-ES optimizer* over cod params — overkill for one parameter; reserved for the
  (out-of-scope) ICES-absolute target.
- **1-D secant/bisection solve (chosen)** — evaluate mean cod (SP1-on) at a few candidate rates and
  converge to the rate that restores the baseline mean within tolerance.

### Solver contract

Pure function, no file I/O, engine injected for testability:

```
solve_larva_rate(
    cfg_off: dict[str, str],        # SP1-off baseline config (resolved)
    cfg_on: dict[str, str],         # SP1-on config (resolved), same except spatial flags
    *,
    run_mean: Callable[[dict], float],   # cfg -> mean cod biomass (years 3-14, seed 0)
    tol: float = 0.02,
    max_iter: int = 10,
) -> RecalResult
```

- `RecalResult` = `{rate: float, baseline: float, mean_on: float, rel_err: float, iters: int,
  history: list[(rate, mean)]}`.
- Algorithm: compute `baseline = run_mean(cfg_off)`. Bracket the cod rate: lower bound = analytical
  first guess `max(0, d0 + ln(E[clip]))`, upper bound = `d0` (current rate; SP1-on at `d0` gives the
  known low mean). Secant iteration on `f(d) = run_mean(cfg_on_with_rate(d)) − baseline`, falling
  back to bisection if secant steps leave the bracket. Monotonic assumption: lower rate → higher
  survival → higher mean (verified by the two bracket endpoints; if not monotone, raise a clear
  error rather than return a bad root).
- Fail-fast: if the bracket does not straddle zero (e.g. even rate 0 cannot reach baseline), raise
  with the measured endpoints — do not silently clamp.

`E[clip]` (the egg-placement-weighted mean spatial multiplier) is computed from the RV field + the
`cod_spawning` egg-distribution map, reusing the field already generated in SP1.

### Where the recalibrated value lives

SP1 is off by default (parity bit-identical when disabled), and the recalibrated rate is only
meaningful **with SP1 on**. So SP1b ships the value as an **SP1-on overlay**, never touching the
Baltic default:

- A **production** helper (in the SP1b module, not test-local) —
  `sp1_on_config(base_cfg, field_path, *, larva_rate=RECAL_RATE) -> dict[str, str]` — returns the
  SP1-on config dict: the SP1 flags (`reproduction.rv.spatial.enabled`, the field file, the sp0
  species-enable) plus the recalibrated `mortality.additional.larva.rate.sp0`. Both the CLI, the
  diagnostic, and the tests import this one helper (the test-local `_baltic_gate_cfg` is replaced by
  it). `RECAL_RATE` is a module constant set to the solved value and cross-checked by the
  mean-neutrality test, so a drift in the field or engine that moves the true root is caught.
- The default `data/baltic/*` config files are **unchanged**. Enabling SP1 without the overlay
  still uses the un-recalibrated rate (the −16% case), which stays the documented "raw SP1" result.

## Deliverables

1. **Solver** — `osmose/calibration/larva_recal.py` (or a focused module): `solve_larva_rate` +
   the `E[clip]` helper. Pure; unit-tested with a synthetic monotone `run_mean` stub (fast, no
   engine) plus one real end-to-end Baltic solve.
2. **CLI** — `scripts/recalibrate_sp1b.py`: builds `cfg_off`/`cfg_on` from the Baltic config + the
   SP1 field, runs the solver against the real engine, prints and records the recalibrated rate.
3. **SP1-on overlay** — the recalibrated rate wired into the SP1-on config helper so the diagnostic
   and tests use it.
4. **Test** — asserts mean-neutrality within ±2% when the overlay is applied (real Baltic run,
   foreground); plus fast unit tests of the solver on the synthetic stub (monotone convergence,
   non-straddling-bracket error, non-monotone error).
5. **Diagnostic update** — `docs/diagnostics/rv_spatial_field.md` (or a sibling) records the
   recalibrated rate, the achieved rel-err, **and** the cod overshoot ratio SP1-on-recalibrated vs
   SP1-off (measured, not gated) — the actual scientific payoff: does mean-neutral spatial
   egg-survival damp the boom/bust?

## Isolation / units

- `solve_larva_rate` + `E[clip]`: pure calibration logic, engine injected, independently testable.
- CLI: thin wrapper (config assembly + I/O).
- Overlay: data (the recalibrated rate).
- Test / diagnostic: consumers of the overlay.

Each unit is understandable and testable without the others; the solver is verifiable in
milliseconds via the stub, decoupled from the slow Baltic engine.

## Risks / open questions

- **Nonlinearity / non-monotonicity.** If mean cod is not monotone in the larva rate over the
  bracket (density-dependent feedbacks), the solver raises rather than returns a wrong root; the
  fallback is to widen/relax and report. Low risk over a single-parameter survival knob.
- **Tolerance vs run noise.** Seed-0 single-run mean may carry stochastic noise comparable to ±2%.
  If so, the solve target uses the same seed throughout (deterministic), so the *relative*
  comparison to baseline is consistent; multi-seed averaging is a possible extension, out of scope
  for v1.
- **Which mean window.** Years 3–14 (post spin-up) matches the SP1 diagnostic; kept identical for
  comparability.

## Out of scope

ICES-absolute calibration; overshoot *gating*; recalibrating non-cod species; changing the Baltic
default config; multi-seed calibration.
