# Surrogate-Bayesian UQ — Real-Data Validation Log

> Running log for validating the surrogate-Bayesian UQ layer against a real Baltic
> config. Started 2026-07-23. Entry point: `run_surrogate_bayes()` in
> `osmose/calibration/uq/run.py`.

## Prerequisite 1 — `k` coverage-multiplier vs the real ICES targets (DONE)

**Watch-point (from Phase 2a):** the split-normal likelihood
(`likelihood.gaussian_log_biomass`) sets its log-space width as
`sigma = half_width / k`, where `half_width = ln(target/lower)` (lower side) or
`ln(upper/target)` (upper side). The default `DEFAULT_K_BY_TYPE =
{"biomass": 1.0, "ssb": 1.0, "catch": 1.0}` therefore treats each band edge as a
**1σ** deviation. If the real bands mean something else, posterior widths are
miscalibrated. `k=1.0` gives the *widest* likelihood, so it is the most
permissive / least-confident choice.

**Finding — the real bands are heterogeneous** (source:
`data/baltic/reference/biomass_targets.csv` v1.2; geometry via
`scratchpad/analyze_bands.py`):

| type | evidence | band meaning | log half-widths |
|------|----------|--------------|-----------------|
| biomass / ssb | CSV header: "order-of-magnitude targets with **acceptable ranges** for calibration"; near-symmetric in *log* space (≈ target ÷2…×2, up to ÷2.5 for perch/pikeperch) | **expert tolerance ranges — NOT σ-intervals** | 0.51 – 1.39 |
| catch | notes: "**mean +/- 1.5σ**; floored at window min (5 yr)"; confirmed numerically (sprat implied-lower 273,467 ≈ stated 274,100; others floored) | arithmetic-space **±1.5σ**, lower floored | 0.08 – 2.20 |

**Decision (reviewed w/ advisor):**
- **catch → k = 1.5** is the one data-defensible value (documented + numerically
  confirmed). Caveat: the bands are *arithmetic*-space and lower-floored, which a
  single log-space scalar `k` cannot faithfully encode — but this is low-impact
  (see below), so the nuance is deferred, not fixed.
- **biomass / ssb → k is a free modeling knob, NOT data-derived.** There is no
  "true" σ to recover from a tolerance range. Keep **k = 1.0** as the honest
  current-default baseline for the *first* real run; do **not** hardcode k=2.0 as
  if derived. Let the **held-out OSMOSE coverage check adjudicate empirically**:
  - k too small → posteriors too wide → held-out points **over-cover** (coverage > nominal)
  - Sweep **k ∈ {1.0, 1.5, 2.0}** for biomass/ssb and pick the value landing
    held-out coverage ≈ nominal.
- Where `k` bites: the likelihood variance is `sigma² + emulator_var`
  (`sigma_seed_sq` enters only the Jensen **mean** bump `mu = mu_emu +
  0.5·sigma_seed_sq`, not the variance). `k` therefore matters **most** on the
  wide biomass/ssb bands (`sigma²` ≈ 0.3–1.3) and **least** on the tight catch
  bands (sprat/herring `sigma²` ≈ 0.006–0.09, where `emulator_var` dominates).
  → don't over-invest in the catch arithmetic/floored nuance.

**First-run `k_by_type` = `{"biomass": 1.0, "ssb": 1.0, "catch": 1.5}`.**

## Prerequisite 2 — yield output emitted (DONE)

`make_engine_evaluator` force-enables SSB but has **no** `enable_yield` flag — it
relies on the base config for yield. If the 4 catch targets are in the run and
yield isn't emitted, every catch design point censors → `gate_failed` or a
`KeyError` in `make_log_posterior` **after** the expensive design.

**Verified clear:** `data/baltic/baltic_all-parameters.csv:8` includes
`baltic_param-output.csv`, which sets `output.yield.biomass.enabled;true`
(line 32). Focal species sp0–sp7 (cod, herring, sprat, flounder, perch,
pikeperch, smelt, stickleback) match the biomass targets; catch targets
(cod, herring, sprat, flounder) are a subset. Catch keys will be produced.

## Small-scale dry-run (DONE — PASS)

Real Python engine (not synthetic), 2 log10 larval-mortality params (cod sp0,
herring sp1), 2 real targets (cod ssb, herring biomass), `n_seeds=2`, `nyear=10`,
`n0=10`. A **pass-through `gate_fn`** forced calibration at `n0` so the full
pipeline ran on real GP-fit data (the sampler/predictive consume emulators, which
the synthetic tests already cover; the novel real surface is
engine → `compute_uq_stats` → `run_design` → GP fit). Script:
`scratchpad/dry_run.py`.

**Result:** `status="ok"`, wall 152.7 s / 20 runs (~7.6 s/run @ nyear=10), no
crash. Full pipeline executed end-to-end on real engine output.
- No censoring (cod/herring survived the whole box @ nyear=10).
- **Natural gate failed both keys** (cod cov 0.80 / MSSR 3.09; herring cov 0.80 /
  MSSR 9.07, r² −0.11) — as expected for `n_seeds=2`, n=10 (1-DOF noise, LOO).
  Confirms the real gate correctly rejects an under-seeded design.
- Sampler converged (ess 744); cod larval rate tightly identified
  (90% CI [1.02, 1.11]), herring wide/prior-dominated (matches its r² −0.11).
- `predictive_ranges` populated (log + biomass ranges, cross-species corr).

Plumbing is validated. Not scientific: `nyear=10` ≠ equilibrium; the box was not
verified to bracket the targets (cod ssb / herring biomass came back 20–40× above
target at the mid-box point).

## Full run — plan + cost (AWAITING USER SIGN-OFF)

Per-run timing (measured, this machine, `ncpu=1`, serial in-process):
`nyear=10` → 8.4 s; **`nyear=40` → 53.4 s**; nyear=50 ≈ 68 s (~1.5 s/yr + overhead).

Engine cost = `n_points × n_seeds` runs (design). The k-sweep and held-out
*re-scoring* are FREE (sampler + coverage run on emulators, no engine calls); the
held-out *set itself* needs fresh engine runs.

Design-point count is gate-driven (unknown a priori); rule-of-thumb ~15–25 pts/dim.

| param set | ~design pts | n_seeds | design runs | serial @53 s | 8-way parallel |
|-----------|-------------|---------|-------------|--------------|----------------|
| Phase 1b (9 params) | ~150 | 3 | ~450 | ~6.6 h | ~50 min |
| Phase 1b (9 params) | ~150 | 10 | ~1500 | ~22 h | ~2.8 h |
| Phase 1 (16 params) | ~300 | 10 | ~3000 | ~44 h | ~5.5 h |

+ held-out coverage set: +30–50% runs. Phase 12/13 (27+ params) are **barred** by
the `check_dimension` cap of 20 — the validation param set must be ≤20.

**Open decisions (user's call — cost/scope):**
1. Param set: Phase 1b (9, faster) vs Phase 1 (16, fuller). Rec: start 1b.
2. `n_seeds`: 3 (calibrate_baltic validation default, ~3× cheaper) vs 10 (handoff).
   Rec: 3–5 first.
3. `nyear` = 40 (calibration default) + confirm near-stationarity (Phase 1
   non-equilibrium watch-point). Verify the param box makes target biomass
   interior-achievable before committing.
4. Parallelize the design loop (process pool; points are embarrassingly parallel)
   → hours instead of days. Engineering task; decide whether to build first.
5. `k_by_type` = `{"biomass":1.0,"ssb":1.0,"catch":1.5}`; sweep biomass/ssb
   k∈{1,1.5,2} against held-out coverage to pick the value landing coverage ≈ nominal.
