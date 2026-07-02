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

- **Baseline** = mean cod biomass over the SP1-off run, using the **exact same window/seed helper the
  SP1 diagnostic already uses** (`scripts/rv_field_diagnostic.py:_mean_cod` at line 21 — years index
  `[3:15]`, finite & >0, seed 0). SP1b imports/extracts that helper rather than re-deriving the window, so the
  baseline and the on-mean are computed identically. The solver receives `baseline` precomputed by
  the caller; the neutrality test recomputes both `cfg_off` and overlay-on means and compares (so it
  is not hard-coded either).
- **Target tolerance** = |mean_on − baseline| / baseline ≤ `tol` (default 0.02). The ±2% is only
  well-defined under the determinism pinning above; if a one-off noise measurement (re-run `cfg_off`
  twice, same seed, confirm bit-identical means) shows residual variance, raise `tol` above it rather
  than shipping a flaky test.

## The knob

`mortality.additional.larva.rate.sp0` (cod). After the 4.4.0 larval-rate migration (orchestrated
in `osmose/config/reader.py:100-103`, which calls the generic `_migrate_larva_rate` in
`osmose/config/aliases.py` with `factor = 1.0/ndtperyear` on read, gated on source version ≥4.4.0),
the value the engine actually consumes is the **resolved per-cohort rate** (base file stores `360.0`
per year → engine reads `15.0`; the ÷ndt is in `reader.py`, not `aliases.py`). SP1b operates on this
**resolved** value in the in-memory config dict (the same dict `PythonEngine.run_in_memory`
receives), so it is migration-agnostic: lowering it raises baseline egg survival to offset the
average spatial kill.

Only cod (sp0) is touched — the only SP1-enabled species.

## Approach — coarse grid scan + bracketed bisection

A single-parameter root-find on `f(d) = mean_cod_on(d) − baseline`, chosen over the two alternatives:

- *Analytical offset only* — `d1 = d0 + ln(E[clip])`. Mathematically it restores the *instantaneous*
  egg-weighted average survival (`exp(-d1)·E[clip] = exp(-d0)`), and its direction is correct
  (`E[clip] ∈ (0,1]` ⇒ `ln E[clip] ≤ 0` ⇒ `d1 ≤ d0`). But the equilibrium years-3–14 *mean* is the
  fixed point of the nonlinear SSB→Beverton-Holt→eggs→larval-clip→recruit feedback loop, so the true
  root generally differs. **Used only as a diagnostic first-guess / initial probe, never as a bracket
  endpoint.**
- *Full DE/CMA-ES optimizer* over cod params — overkill for one parameter; reserved for the
  (out-of-scope) ICES-absolute target.
- **Coarse grid scan then bisection (chosen).** Two reasons the naive secant-in-a-narrow-bracket was
  rejected: (1) **the bracket must be `[0, d0]`**, not `[d1_analytical, d0]` — rate 0 (no larval
  mortality ⇒ maximal survival) guarantees `f(0) ≫ 0` and thus a straddle whenever the baseline is
  reachable at all, whereas the razor-thin analytical bracket (`≈[14.83, 15]`) can have `f<0` at both
  ends and fail. (2) **The mean is NOT reliably monotone in `d`** for this boom-bust system — the
  project's own RV-gate result showed *more* cod survival lowering the mean and worsening the
  overshoot, because extra recruits can deepen a later crash. A two-endpoint monotonicity check
  cannot detect interior non-monotonicity, and secant can then converge to a spurious root.

  So the algorithm is: (a) evaluate `f` on a **coarse grid** of 4–6 points across `[0, d0]` (seeded
  to include `d1_analytical`); (b) **near-zero short-circuit FIRST** — if any grid point already has
  `|f|/baseline ≤ tol`, return it immediately (`feasible=True, converged=True`); when that point is
  `d0` itself this reads as "SP1 barely moved the mean — no recalibration needed, `rate=d0`". This
  also makes the sign of the remaining points well-posed (an exact/near hit is a *hit*, not a sign);
  (c) **feasibility gate** — count sign changes of `f` among the non-hit grid points: exactly one →
  bracket it and refine by **bisection** (robust to local non-monotonicity within the bracket); zero
  (baseline unreachable) or ≥two (ambiguous / multi-root) → STOP and report the grid honestly as a
  negative/ambiguous result (`feasible=False`) rather than returning a forced number; (d) refine to
  `tol` or `max_iter`, reporting the achieved rel-err.

### Determinism (required for a well-defined `f`)

`f(d)` must be exactly reproducible across the ~10–15 engine evaluations or bisection chases noise.
The solve therefore **requires**, in both `cfg_off` and `cfg_on`, the two real fixed-seed keys
(`config.py:2054-2056`): `movement.randomseed.fixed = true` and
`stochastic.mortality.randomseed.fixed = true` (these drive `build_rng(..., fixed=True)` per species,
`osmose/engine/__init__.py:81-83`). Plus a **single engine thread** via `numba.set_num_threads(1)`
(precedent: `osmose/validation/fmsy_sweep.py:244`) — thread-order FP reductions in the Numba mortality
kernel are otherwise non-deterministic. The CLI sets all three before solving; the ±2% tolerance is
only meaningful under this pinning. (Note: `simulation.rng.fixed` as written in CLAUDE.md is stale
shorthand — no `simulation.rng*` key is read anywhere in `osmose/` (grep-confirmed); the engine reads
only the two `*.randomseed.fixed` keys above. Worth correcting CLAUDE.md as a side fix.)

### Solver contract

Pure function, no file I/O, engine + config-builder injected for testability:

```
solve_larva_rate(
    baseline: float,                      # mean cod biomass, SP1-off (precomputed by caller)
    run_mean_on: Callable[[float], float],# larva_rate -> mean cod biomass, SP1-on (deterministic)
    *,
    grid_points: Sequence[float],         # 4-6 rates spanning [0, d0], incl. d1_analytical; max == d0
    tol: float = 0.02,
    max_iter: int = 20,
) -> RecalResult
```

- `run_mean_on(rate)` is built by the caller (CLI/test) as a closure over the SP1-on config
  assembled once via `sp1_on_config(...placeholder...)`; the closure sets
  `mortality.additional.larva.rate.sp0=rate` on that dict before invoking the engine and returns the
  years-3–14 mean. The solver only ever *calls* `run_mean_on(rate)` — it never touches files, SP1
  flags, or config dicts, so `sp1_on_config` stays the single owner of assembly. (`d0` is not a solver
  parameter; it is simply `max(grid_points)`.)
- `RecalResult` = `{rate: float | None, baseline, mean_on, rel_err, converged: bool, feasible: bool,
  grid: list[(rate, mean)], iters, message: str}`.
  - near-zero hit (any grid point `|f|/baseline ≤ tol`) ⇒ `feasible=True, converged=True`, `rate` =
    that point (`rate==max(grid_points)` means "no recalibration needed").
  - `feasible=False` (zero or ≥two sign changes among non-hit points) ⇒ `rate=None`, `message`
    explains; the CLI/test report the grid, do NOT fabricate a rate.
  - `feasible=True, converged=False` (hit `max_iter` still outside `tol`) ⇒ `rate` = midpoint of the
    final (single) bisection bracket, `converged=False`; caller decides whether to accept.
  - `feasible=True, converged=True` ⇒ `rate` within `tol`.
- Algorithm: near-zero short-circuit → count sign changes of `f = mean − baseline` on the grid →
  feasibility gate → bisect the single sign-changing sub-interval to `tol`/`max_iter`. No monotonicity
  *assumption*; the grid *measures* the shape instead.

`E[clip]` (egg-placement-weighted mean spatial multiplier, used only to seed `d1_analytical =
d0 + ln E[clip]` in `grid_points`) is computed from the RV field + the `cod_spawning`
egg-distribution map by a small helper, reusing the field generated in SP1. It is a first-guess only;
the empirical solve makes the final rate weighting-agnostic, and a static egg-distribution weighting
is acceptable precisely because it is not a bracket endpoint.

### Where the recalibrated value lives

SP1 is off by default (parity bit-identical when disabled), and the recalibrated rate is only
meaningful **with SP1 on**. So SP1b ships the value as an **SP1-on overlay**, never touching the
Baltic default:

- A **production** helper (in the SP1b module, not test-local) —
  `sp1_on_config(base_cfg, field_path, *, larva_rate=RECAL_RATE) -> dict[str, str]` — returns the
  SP1-on config dict: the SP1 flags (`reproduction.rv.spatial.enabled`, the field file, the sp0
  species-enable) plus the recalibrated `mortality.additional.larva.rate.sp0` (when `larva_rate is
  not None`; if `None`, the key is **omitted** so the base d0=15 stands — the infeasible path), and
  the config-side determinism keys `movement.randomseed.fixed=true` +
  `stochastic.mortality.randomseed.fixed=true`. Both the CLI (via a placeholder rate whose closure
  overrides it per call), the diagnostic, and the tests import this one helper — the test-local
  `_baltic_gate_cfg` is deleted and replaced by it (single owner of SP1-on assembly). The runtime
  pin `numba.set_num_threads(1)` is NOT a config key, so **every** caller that runs the engine (CLI
  *and* the neutrality test) must call it itself before running.
- **`RECAL_RATE` is a hand-set module constant, and the CLI does NOT write it back automatically.**
  Closing the loop is an explicit manual step: run the CLI, read the printed solved rate, paste it
  into the constant. The CLI emits a ready-to-paste line (`RECAL_RATE = <value>  # solved ...`) to
  make that copy trivial. The mean-neutrality test is the **drift guard**: it re-runs `cfg_off` and
  the overlay-on config and asserts neutrality within `tol`, so if the field/engine later moves the
  true root away from the frozen constant, the test fails. (No circularity: the solve produces the
  value offline; the constant freezes it; the test independently re-verifies it.)
- **If the solve is infeasible** (`feasible=False` — baseline unreachable even at rate 0, or multiple
  roots), SP1b ships **no** `RECAL_RATE`: the diagnostic records the grid and the negative/ambiguous
  finding (mean-neutrality not achievable via the cod larva rate alone), and `sp1_on_config` keeps the
  un-recalibrated rate. That is a legitimate terminal outcome, consistent with the project's
  negative-result honesty (cf. the RV-gate).
- The default `data/baltic/*` config files are **unchanged**. Enabling SP1 without the overlay
  still uses the un-recalibrated rate (the −16% case), which stays the documented "raw SP1" result.

## Deliverables

1. **Solver + overlay module** — `osmose/calibration/larva_recal.py`: `solve_larva_rate` (grid scan +
   bisection, `RecalResult`), the `E[clip]` first-guess helper, `sp1_on_config`, and the `RECAL_RATE`
   constant. Pure/deterministic (engine injected via `run_mean_on`). Unit-tested with synthetic
   `run_mean_on` stubs (fast, no engine): monotone-→-converges, single-sign-change bisection,
   zero-sign-change → `feasible=False`, multiple-sign-change → `feasible=False`, `max_iter` →
   `converged=False`.
2. **CLI** — `scripts/recalibrate_sp1b.py`: calls `numba.set_num_threads(1)`, precomputes `baseline`
   via the shared `_mean_cod` on a `cfg_off` carrying the two `*.randomseed.fixed` keys, builds
   `run_mean_on` from `sp1_on_config(...placeholder...)`, forms `grid_points = sorted({0.0, d0} ∪
   linspace(0, d0, 5) ∪ {clip(d1_analytical, 0, d0)})` (so both endpoints and the analytical guess are
   present), runs the solver against the real engine, prints the grid + result + a ready-to-paste
   `RECAL_RATE` line. Does **not** edit the module. (`d0` = the resolved cod rate, 15.0.)
3. **SP1-on overlay** — `RECAL_RATE` (hand-set from the CLI output) consumed by `sp1_on_config`; the
   diagnostic and tests use that one helper.
4. **Test** — `tests/test_sp1b_recalibration.py`: (a) fast solver unit tests on stubs (above); (b) one
   real-engine mean-neutrality test — calls `numba.set_num_threads(1)`, then runs `cfg_off` and the
   overlay-on config (foreground, generous timeout), asserts `|mean_on − baseline|/baseline ≤ tol`; if
   `RECAL_RATE is None` (infeasible), the
   test asserts that state instead (xfail/skip with the recorded reason) rather than a numeric bound.
5. **Diagnostic update** — `docs/diagnostics/rv_spatial_field.md` (or a sibling) records the
   recalibrated rate (or the infeasible grid), the achieved rel-err, **and** the cod overshoot ratio
   SP1-on-recalibrated vs SP1-off (measured, not gated) — the actual scientific payoff: does
   mean-neutral spatial egg-survival damp the boom/bust?

## Isolation / units

- `solve_larva_rate` + `E[clip]`: pure calibration logic, engine injected via `run_mean_on`,
  independently testable.
- `sp1_on_config` + `RECAL_RATE`: single owner of SP1-on assembly and the recalibrated value (data).
- CLI: thin wrapper (determinism pinning, baseline, grid, I/O).
- Test / diagnostic: consumers of the overlay.

Each unit is understandable and testable without the others; the solver is verifiable in
milliseconds via the stub, decoupled from the slow Baltic engine.

## Risks / open questions

- **Non-monotonicity (primary risk).** The years-3–14 mean is not guaranteed monotone in the larva
  rate for this boom-bust system (RV-gate precedent). Mitigated by *measuring* the shape with the
  coarse grid and gating on the sign-change count: single crossing → bisect; zero/multiple → report
  as infeasible/ambiguous, never a forced number. Bisection converges on any single-sign-change
  bracket regardless of interior wiggles.
- **Baseline reachability.** If even rate 0 (maximal survival) cannot lift the SP1-on mean to the
  baseline, the field's spatial kill is simply too deep to offset with this one knob — reported as
  `feasible=False`, a legitimate negative result (points to SP1's egg loss being structural, not a
  level knob can fix).
- **Determinism.** ±2% is only meaningful with `movement.randomseed.fixed` +
  `stochastic.mortality.randomseed.fixed` + `numba.set_num_threads(1)` (see the Determinism section);
  both the CLI and the neutrality test must set all three, and the noise-measurement step confirms
  `f(d)` is reproducible before trusting the solve. Multi-seed averaging is a possible extension, out
  of scope.
- **Which mean window.** Years 3–14 (post spin-up) matches the SP1 diagnostic; kept identical for
  comparability.

## Out of scope

ICES-absolute calibration; overshoot *gating*; recalibrating non-cod species; changing the Baltic
default config; multi-seed calibration.
