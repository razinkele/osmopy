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

## Parallelization (DONE — merged 8b870a7)

`run_design` now dispatches through `evaluator.evaluate_batch` when present;
`make_engine_evaluator(n_workers>1)` returns a spawn-based process-pool evaluator,
bit-identical to serial. 6 new tests. Measured: nyear=40 ≈ 53 s/run serial; the
pool amortizes spawn+numba-JIT startup over the run. **Caller must guard scripts
with `if __name__ == "__main__":`** (spawn re-imports the entry module).

## Prerequisite 3 — box brackets the targets (FAIL — validation is NOT well-posed as scoped)

Pre-launch probe (`scratchpad/prelaunch_check.py`): 9 Phase-1b params on the R18
baseline box, nyear=40, driven targets, x0 + both box corners.

| target | band (t) | x0 (R18) | all-low-mort | all-high-mort |
|--------|----------|----------|--------------|---------------|
| cod.ssb | 60k–250k | **≈0** | ≈0 | ≈0 |
| herring.biomass | 0.8–3M | 18.9M (6×) | **0** | 0 |
| sprat.biomass | 0.8–2.5M | 6.1M | **0** | 0 |
| flounder.biomass | 20k–100k | **≈0** | ≈0 | ≈0 |
| stickleback.biomass | 50k–500k | 3.85M | 3.9e7 | 0 |

Two independent problems:
1. **The box is garbage parameter space.** Centered on the R18 baseline the model
   sits 6–8× above its own planktivore targets; cod/flounder extinct. Extinction
   at *both* corners (all-low = competitive exclusion: stickleback 3.9e7 while
   herring=sprat=0; all-high = over-mortality) ⇒ interaction-dominated, non-monotone
   surface, pervasive censoring. A 9-D GP over this is what the gate is built to
   **refuse**. `mortality` params only push biomass down, so cod/flounder targets
   are unreachable from here regardless.
2. **Even a *calibrated* Baltic config fits only 2/8 ICES envelopes.** Best
   documented run (Shepherd, obj 2.1, `docs/baltic_shepherd_calibration_2026-05-30.md`):
   herring + smelt in-range; cod ×10.9, sprat ×2.45, flounder ×5.42, stickleback
   ×7.44, perch ×79.7, pikeperch ×167 — a spatial-resolution/aggregation limit, not
   a calibration gap. There is **no** Baltic parameter regime where the model fits
   the ICES bands well enough for an across-species "uncertainty around a fit".

**Consequence:** launching `run_surrogate_bayes` on the R18 Phase-1b box → cod &
flounder censor at every point → `gate_failed`, no posterior, after hours. Do not
launch. Narrowing the R18 bounds to force a pass would *manufacture* the result.

## Reframing options (AWAITING USER DIRECTION)

- **A — Method validation via self-consistency (rec first).** Use a calibrated
  config's OWN engine outputs as the "targets"; center the params in a tight box
  around their calibrated values. Validates that the UQ layer recovers known params
  with correct held-out coverage on REAL engine dynamics — without needing the model
  to match ICES. Well-posed, low-censoring, the gate can certify.
- **B — Data-realism validation on a better-fitting config.** EEC (14/14) or Bay of
  Biscay (8/8) parity — the model fits observations far better, so an ICES/data-band
  validation is well-posed there. Switch base config off Baltic.
- **C — Narrow ICES validation to Baltic's fittable species** (herring; maybe smelt/
  sprat), re-centered on the Shepherd-calibrated config with tight bounds. Limited to
  1–3 targets and requires aligning the varied params to those species.
- **D — Proceed on Baltic-R18 anyway** — expect `gate_failed`; informative-but-negative,
  wastes hours. Not recommended.

Once a framing is chosen: cheap ~15-min LHS censoring/gate probe at the settled
center BEFORE the full run; confirm nyear=40 near-stationarity (last-10 vs
prior-10-yr drift); then Phase B (held-out coverage + k-sweep, not yet built).

### Carried-forward knobs (once framing settled)
- `n_seeds` 3–5 first; `nyear`=40; `n_workers`≈12 (28 cores / 24 GB); gate n0≈40,
  increment 20, n_max ~200; `k_by_type`={"biomass":1.0,"ssb":1.0,"catch":1.5}
  with biomass/ssb k∈{1,1.5,2} swept against held-out coverage (Phase B).
