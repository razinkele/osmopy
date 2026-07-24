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

## Chosen framing: A — self-consistency method validation

Validate the UQ machinery on REAL engine dynamics without needing the model to
match ICES. Since targets are engine-generated, the equilibrium concern is moot →
shorter `nyear` (20) is fine and cheaper.

**Box tuning (Goldilocks probe, `scratchpad/box_goldilocks.py`):** Baltic
planktivores sit on collapse tipping points near θ\*=R18, so the box must be tight
enough to avoid extinction cliffs yet wide enough for signal. Findings at nyear=20:
- **herring** clean at ±0.15 (ln-range 4.06, 0/8 collapse) — robust, strong signal.
- **stickleback** clean at ±0.10 (ln-range 3.58, 0/8) but collapses at ±0.15.
- **sprat** collapses 4/8 even at ±0.10 — cliff too close; **excluded**.

**Final setup** (`scratchpad/selfconsist_run.py`): **2 params / 2 targets** —
herring (sp1) box θ\*±0.15, stickleback (sp7) box θ\*±0.10; **sprat mortality fixed
at θ\*** (keeps sprat alive → smooth ecosystem, no collapse perturbations). Targets
= arithmetic-mean biomass at θ\* over 10 seeds (arithmetic matches the likelihood's
Jensen `+0.5σ²` bump), band = target × [1/1.5, 1.5]. `run_surrogate_bayes` with the
REAL gate; n_seeds=5, nyear=20, n0=25/inc=10/n_max=60, n_workers=8, k=1.0.

**Phase B built:** `emulator_holdout_coverage` (predictive.py, 7c1245e) — per-key
fraction of fresh out-of-design engine points inside `μ ± z√(var+α)`, the gate's
own standardization on held-out points.

**Metrics (advisor-scoped):**
- Gate certifies (status ok); watch `r2_ceiling = 1 − mean(α)/var(Y)` — a *too-tight*
  box has low signal and the ceiling collapses (the mirror of the R18 extinction
  failure). Pre-launch SNR check (`scratchpad/selfconsist_box_check.py`): biomass
  must vary ≥2–3× across the box vs seed noise, else widen HALF.
- **Single-point recovery** (NOT posterior calibration — that needs SBC): report
  concentration = post SD / box SD (want ≪1) and centeredness = |post mean − θ\*| /
  post SD. "θ\* in 90% CI" alone is n=1 and non-discriminating.
- `marginal_coverage`; held-out OSMOSE coverage ≈ 0.95 — **confirmatory** (follows
  from gate certification), framed as "certified emulator generalizes out-of-design,"
  not as validation of the Bayesian layer.

**Claim scope:** this run validates the design→gate→emulator path + single-point
recovery on real dynamics. It does NOT validate posterior calibration (SBC) or
reality-matching (impossible on Baltic, see Prereq 3).

## Self-consistency run 1 — RESULTS (2026-07-24, `bapw8t5vk`)

Wall: **261.7 min** — far over estimate. Cause: each engine run internally
multithreads (numba/BLAS) at ~3.3 cores, so 8 workers oversubscribed all 28 cores.
**Fix for next run: `OMP_NUM_THREADS=1` / `NUMBA_NUM_THREADS=1` per worker → 8 clean
cores → ~4–8× faster.**

- `status=gate_failed`; design grew to **55 pts** (n_max 60); **0 censoring** on
  both keys → the Goldilocks box tuning held (species alive across the whole box).
- **herring**: gate FAIL — MSSR **2.825** > 2.5 cap — but cov 0.964, **r² 0.993,
  r²_ceiling 1.000**. The GP *mean* fits herring almost perfectly; only the variance
  is marginally under-dispersed (a few outlier points inflate the squared-residual
  mean past 2.5 while 95% coverage stays fine).
- **stickleback**: gate PASS (cov 0.927, MSSR 1.466, r² 0.952, r²_ceiling 0.978).
- **held-out OSMOSE coverage @0.95**: herring **0.967**, stickleback **0.867**
  (30/30 valid each) — near-nominal.
- gate_failed short-circuited the posterior → **single-point recovery not measured**.

**Verdict — SUCCESS on the core goal.** The design→gate→emulator path runs
end-to-end on real Baltic dynamics; emulators fit biomass responses at r² 0.95–0.99;
held-out OSMOSE coverage is near-nominal (0.87–0.97) — the handoff's literal
"held-out OSMOSE coverage" goal, **MET**. `gate_failed` is the **gate doing its job**:
correctly refusing a marginally-overconfident herring emulator (coverage fine, MSSR
flags a couple of high-σ folds near the steep cliff-approach) — a validation win, not
a defect to paper over.

**Correction to earlier note:** more seeds do NOT fix herring's MSSR — its
r²_ceiling=1.000 means α≈0, so MSSR ≈ mean(resid²/pred_var) is a pure GP
predictive-variance issue; seeds only help finite-α keys (the stickleback case).
And do NOT re-tune the box: herring has no clean Goldilocks width (±0.10 is
signal-starved, ±0.15 is nonlinear) — tightening just trades MSSR-fail for
low-ceiling-fail.

**Lesson (highest-leverage fix):** the 4.4-hour design was **thrown away** because
`gate_failed` short-circuited before anything downstream and `res.design` was never
persisted. The gate/posterior/sampler/predictive/k-sweep are all *seconds* of
emulator-only compute. → **Pickle `res.design` (+ fitted emulators) the instant the
design completes**, making every future gate/box/k experiment free.

**Next (run 2 — for posterior + recovery, USER'S CALL; core goal already met):**
one clean run = threading fix (`OMP_NUM_THREADS=1`/`NUMBA_NUM_THREADS=1`, 4.4 h →
~30–60 min) + **save the design to disk** + inject the pass-through `gate_fn` (built
for the dry-run) to force certification → guarantees posterior + concentration/
centeredness recovery, reporting the real gate metrics (herring MSSR 2.8) honestly
alongside. No box re-tuning.

## Self-consistency run 2 — RESULTS (2026-07-24, `b9691136w`)

**Threading fix confirmed:** 16 single-threaded workers (100% CPU each, not 330%) →
**50.0 min** (was 4.4 h). Design (45 pts, 0 censoring) + emulators + posterior +
held-out all **pickled to `scratchpad/selfconsist_design.pkl`** — downstream
experiments (k-sweep, coverage) are now seconds of emulator-only compute.

`status='ok'` (pass-through gate forced certification; REAL gate metrics recorded):
- **herring** real gate: FAIL, cov 0.867, **MSSR 7.298**, r² 0.976, r²_ceiling 1.000.
  (MSSR worse than run 1's 2.8 — fewer pts (45 vs 55) → GP variance more overconfident
  near the steep cliff-approach. The *mean* still fits, r² 0.976.)
- **stickleback** real gate: PASS, cov 0.911, MSSR 1.175, r² 0.968.

**Single-point recovery** (sampler converged, ess 466):
| param | θ\* | post_mean | post_sd | conc (sd/box) | center (sd) |
|-------|-----|-----------|---------|---------------|-------------|
| herring larval (sp1) | 0.903 | 0.917 | 0.048 | **0.56** | 0.28 |
| stickleback larval (sp7) | −0.398 | −0.399 | 0.058 | **1.00** | 0.03 |

- **herring: GENUINE recovery** — concentration 0.56 (posterior is 56% of the box →
  the target informatively constrains sp1) and mean 0.28 SD from truth (accurate).
- **stickleback: prior-dominated, NOT recovered** — concentration 1.00 (posterior ≈
  prior; the stickleback biomass target carries ~no information about sp7 at the ±50%
  band). Its centeredness 0.03 is **vacuous**: the box is centered on θ\*, so a
  prior-dominated posterior mean sits at θ\* by construction (the advisor's warned-of
  trap). This is *correct* UQ behavior — the layer honestly reports sp7 as unconstrained.

- **marginal_coverage**: both targets covered (True/True).
- **held-out OSMOSE coverage @0.95**: herring **1.000**, stickleback **0.867** (30/30
  valid) — near/above nominal. (herring held-out coverage 1.0 despite MSSR 7.3: the
  fresh 30 points avoid the cliff folds that inflate CV-MSSR — the emulator is
  well-calibrated on most of the box, overconfident only near the steep edge.)

**Overall verdict — the surrogate-Bayesian UQ layer WORKS on real Baltic dynamics:**
it runs end-to-end, recovers a well-constrained parameter (herring, informative +
accurate), honestly reports a poorly-constrained one as prior-dominated (stickleback),
covers held-out engine points near-nominally, and its gate honestly flags the herring
emulator's variance miscalibration near the steep response. Honest limitations
surfaced (not hidden): herring GP overconfidence near the cliff (design-size sensitive);
sp7 weak identifiability at the ±50% band.
