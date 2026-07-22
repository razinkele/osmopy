# Surrogate-based Bayesian UQ for OSMOSE calibration — design

- **Date:** 2026-07-22
- **Status:** Approved design, revised after two in-loop review rounds (pre-implementation)
- **Owner:** Arturas Razinkovas-Baziukas
- **Related:** `osmose/calibration/` (existing NSGA-II / CMA-ES / surrogate stack)

## Context & motivation

The existing calibration stack finds best-fit / Pareto-optimal parameter points but
does **not** quantify parameter identifiability or uncertainty. A prior assessment
(PEST/PEST++ vs. the current system) concluded that at this problem's scale
(the live Baltic chunks are 9–17 params; EEC 14, BoB 8 — all inside the trustworthy
envelope below) **surrogate-based Bayesian inference is the right first tool**.

Its value is **not a lower run count than PEST++** — a variance-calibrated emulator at
~17 dims × S≥10 seeds is plausibly 1,600–3,200 OSMOSE runs, the same order as PESTPP-IES.
Its value is **not a lower run count**: the initial design alone is `N₀ = 20·d·S` runs
(≈ 3,400 at the a2 chunk's d=17, S=10, before any densification), the same order as
PESTPP-IES. The value is **amortization + parameter-space structure**: one design is
reused across both likelihoods, what-if scenarios, and posterior re-queries, and it yields
the joint parameter posterior — the identifiability/equifinality structure — which is
**not** recoverable from Sobol effects (marginal) or NSGA-II Pareto fronts
(objective-space). PEST++ remains a heavyweight fallback for the 30–50-param frontier this
work does not reach.

## Goals (Goal 1 is the product; Goal 2 is a diagnostic)

1. **Parameter identifiability / equifinality (primary).** Joint posterior over the
   calibrated `FreeParameter`s: credible intervals, parameter correlations, and which
   parameters the ICES targets constrain vs. leave at prior width. Robust deliverable:
   the **qualitative identifiability structure** — which θ-directions the targets inform
   vs. leave at prior width. Honest bound: θ-dependent structural discrepancy can
   *rotate/bias* the ridge orientation and parameter correlations (not merely shrink
   widths), and that bias is unmeasurable in v1 — so claim only the informed-vs-uninformed
   direction structure, never that the ridge orientation itself is unbiased.
2. **Diagnostic predictive ranges (secondary, labeled).** Per-species **marginal**
   emulator-predictive ranges on biomass/SSB/yield, produced for posterior-predictive
   checks. Labeled everywhere as: *emulator-coverage-measured, not calibrated against
   reality, marginal-only, and invalid for derived/joint quantities* (ratios, total
   biomass, P(stock>Blim)). Not to be presented as calibrated "credible intervals."

## Trustworthy envelope & known limitations (read first)

Valid UQ only inside a bounded regime; outside it the output is confident-but-wrong.

- **Dimensionality: reliable at ≤ ~15–20 *effective* parameters** (concentration of
  measure + ensemble-sampler mixing ceiling). The largest live chunk (Baltic a2 = 17) sits
  *at* the edge — which is exactly why the coverage/PIT gate (not dimension reduction) is
  the first thing to build: it tells you whether a chunk has fallen out of the regime.
  Above ~20, reduce dimension first (Sobol screening exists via `SensitivityAnalyzer`;
  note **Morris does not exist here**, and Sobol on the full simulator is ~10k+ runs, so
  dimension reduction is a v2 concern, not a v1 front-end).
- **v1 under-coverage against reality is real *and largely unmeasurable within v1*.**
  There is no discrepancy function δ(θ) (structural OSMOSE error is absorbed into θ) and
  species targets are treated as conditionally independent though trophically coupled;
  both shrink interval widths (not ridge orientation). Crucially, v1's real-data checks
  (fresh OSMOSE at posterior draws; coverage on held-out OSMOSE) measure only
  **emulator-vs-simulator** fidelity — the structural flaw is **simulator-vs-observation**,
  which held-out OSMOSE points cannot see, and θ is fit to the same ~8 coupled targets so
  there is almost no held-out power to calibrate δ. Do **not** let emulator-calibration
  diagnostics stand in as reassurance about structural coverage.
- **δ(θ) does not cleanly fix this even in v2:** separating structural bias from parameter
  error is ill-posed without strong priors on δ (Brynjarsdóttir & O'Hagan, 2014, *Inverse
  Problems* 30(11):114007). The v1 σ_disc² floor is a width-inflation knob, not a correction.
- **Extinction/collapse regime is out of scope**, and **censoring it biases the retained
  posterior toward stability** (design points at collapse are dropped → posterior pushed
  away from collapse boundaries → system looks more stable than it is). Report this.

## Non-goals (explicitly out of v1)

Active-learning beyond the coverage-driven (uniform-densification) stopping rule;
PPL/NUTS joint hyperparameter inference (Approach B); a full δ(θ) function (only a variance
floor); time-series / non-equilibrium outputs; correlated multi-output GP; dimension-
reduction front-end; UI integration.

## Architecture — `osmose/calibration/uq/` (new subpackage)

`emulator.py` builds its **own** sklearn GP directly — it does **not** mutate the shared
`SurrogateCalibrator` (which the UI and `find_optimum` depend on; making it ARD/`alpha` by
default would silently drift those untested paths). All uq deps are lazy-imported.

| Module | Responsibility |
|---|---|
| `output_stats.py` | One `OsmoseResults` → per-species stat dict with **distinct UQ keys** `{sp}_biomass_mean`, `{sp}_ssb_mean`, `{sp}_yield_mean` (calling `results.biomass()`/`ssb()`/`yield_biomass()`, indexing the **wide** frames per species). SSB extraction is **net-new** (the script computes only mean/yield/cv/trend). UQ-scoped — does not touch `losses.quantity_key`, so no NSGA/DE backward-compat break. |
| `keying.py` (or within `output_stats.py`) | `target_to_output_key(BiomassTarget) -> str` mapping `reference_point_type` {biomass,ssb,catch} to the distinct stat key. The one interface the whole chain hinges on. |
| `design.py` | Execute LHS design × **S seeds** (independent seed axis) via the Python engine **with SSB output enabled**; return `(X, Y=log-seed-mean, alpha=s²/S)` per output. New plumbing: no existing harness returns this (`problem.py:441` pins `seed=run_id`; all return scalar losses). |
| `emulator.py` | One GP per output stat: **ARD** `Matern(2.5)` (vector length scales), per-point `alpha` **co-scaled by `1/Var_design(Y)`** (sklearn adds `alpha` to the kernel diagonal *unscaled* while `normalize_y` standardizes `y`, so a raw `s²/S` is mis-scaled by `Var_design(Y)` — verified ~3× length-scale / 1.27× predictive-std distortion; equivalently use `normalize_y=False` + manual mean-centering), fit on **natural-log** outputs; `predict(theta) -> (mean, var)`; a CV routine that **returns per-fold predictive variance** (for the gate), not just RMSE/R². Unit-test: predictive variance invariant to arbitrary y-rescaling when `alpha` is co-scaled. |
| `prior.py` | Uniform prior in the transformed sampling space. Apply the existing **per-parameter** transform (`10**` for `Transform.LOG`, pass-through for `LINEAR`, per `problem.py:263`) once, at the **simulator-input and reporting** boundaries — never blanket, never at the emulator input (the emulator trains on sampling-space X). |
| `likelihood.py` | `GaussianLogBiomass` (default) and `BandFaithful` (ABC kernel). Both keep the θ-dependent log-normalizer. See likelihood section. |
| `posterior.py` | `log_prior + Σ_targets likelihood(emulator[key(target)].predict(θ), target)`. Documents cross-target independence as an overconfidence source. |
| `sampler.py` | `DynestySampler` (default) and `EmceeSampler` (with mandatory over-dispersed multi-start + cross-run mode agreement). |
| `predictive.py` | Genuine per-θ mixture; per-species **marginal** ranges; cross-species-correlation PPC. |
| `run.py` | Orchestrator `run_surrogate_bayes(...) -> UQResult` (wiring). |

## Statistical model (all quantities in **natural log** of biomass)

- **Emulator target** = **mean-of-logs** over seeds, `Y = mean_s log(x_s)`; training noise
  `alpha = s²/S` where `s²` = variance *of log single-run* over seeds (variance of the
  mean). Emulate log so `alpha`, `emulator_var`, `σ_band²`, `σ_disc²` are all log-space
  and additive. Pin the base as natural log everywhere (targets `σ_band` uses `ln`, not
  `log10`, to avoid an (ln10)²≈5.3× units bug). **Declare the estimand:** mean-of-logs is
  the **log-geometric-mean** `E[ln x]`, biased low of `ln E[x]` (arithmetic / management
  biomass) by the θ-dependent lognormal Jensen term `½·σ_seed²(θ)`. The ICES targets are
  arithmetic-scale, so add `+½·σ_seed²(θ)` to μ_emu before forming the residual in **both**
  likelihood paths; otherwise this bias compounds same-direction with the stability-
  censoring bias.
- **Default `GaussianLogBiomass`** per-target log-likelihood, **keeping the full
  θ-dependent normalizer** `−½ log σ_eff²(θ) − ½ r²/σ_eff²(θ)` (the normalizer *is* the
  emulator-variance self-penalty — dropping it lets mass flood high-variance regions):
  - fitting variance `σ_eff²(θ) = σ_band² + emulator_var(θ) + σ_disc²` (no seed term).
  - `σ_band` from the band via a **configurable coverage multiplier** `band = k·σ`
    (default per `reference_point_type`, documented; not hardwired 1.96 — ICES bands are
    often reference points, not 95% CIs). Asymmetric bands → split-normal
    `σ_lo=(ln target−ln l)/k`, `σ_hi=(ln u−ln target)/k` with the **coupled** two-piece
    normalizer `−log(σ_eff_lo(θ)+σ_eff_hi(θ))` (not a per-branch `−½ log σ_eff²`, which is
    mode-discontinuous); additive-with-emulator_var is a documented approximation, not a true
    convolution. Unit-test: integrate-to-1 and continuity at r=0.
  - `σ_disc²` = per-output discrepancy floor, small, **optionally** tuned so the
    posterior-*predictive* covers the ICES *observations* — documented as weak (in-sample,
    ~8 coupled targets, few DoF) and as width-inflation, **not** a structural correction.
    Included in the fitting likelihood **and** in the predictive PPC comparison.
- **`BandFaithful`** = ABC-style tolerance kernel: score `P(y∈band)` under
  `N(μ_emu, emulator_var)` (this *is* the convolution; the separate "flat-inside/decay-
  outside" construction is dropped to avoid a second, unidentified decay-scale knob).
  Prior-dominated on the plateau (its width is a prior artifact); posterior proper under
  the bounded prior. Reporting width under **both** likelihoods is a **required** diagnostic.
- **Predictive layer** (Goal-2 diagnostic): genuine mixture — for each posterior θ draw,
  `y ~ N(μ_emu(θ), emulator_var(θ) + σ_seed²(θ))`, where `σ_seed²(θ)` comes from a separate
  positivity-guaranteed log-space noise model (or pooled constant) fit to the design's
  per-point `s²`. Convolve `σ_band²`+`σ_disc²` for the band PPC.

## Trust gate & envelope enforcement (probabilistic; drives + caps the design)

- **Emulator gate = predictive-variance calibration** per output (Bastos & O'Hagan 2009):
  k-fold (LOO for small N) 95% coverage ≈ 0.95, mean standardized-squared residual ≈ 1,
  PIT ~ uniform / CRPS. Standardize held-out residuals by `emulator_var(θᵢ) + sᵢ²/S`
  (latent GP variance + held-out seed-mean noise). Raw R² only as a secondary screen against
  its noise-adjusted ceiling `1 − (σ_seed²/S)/Var(Y)` (uses the **mean's** noise `s²/S`).
  **Abort UQ** if variance calibration fails. This gate certifies **emulator fidelity only**;
  it does **not** enforce the ≤~20-param envelope, whose failure mode is *sampler mixing*
  (a2=17 can pass every emulator gate yet yield wrong intervals from an under-explored
  posterior). Enforce the envelope separately (next bullets): a sampler-adequacy diagnostic
  plus a hard nominal-dimension cap that aborts regardless of gate pass.
- **Design-growth loop is bounded:** start `N₀ = 20·d`; on gate failure `design.py` appends
  a fresh seeded-LHS batch (size = increment) and refits on the union; **hard ceiling**
  `N_max` with an explicit abort (never an unbounded "grow until passes"). `S` is **fixed**
  at the design's seed count (no `S_max` growth axis); Phase 1 pins the increment / `N_max`.
- **Convergence gate (per sampler mode):** dynesty (default) — `dlogz` stopping tolerance +
  live-point / posterior-mass checks; emcee — R̂ < 1.01 + adequate ESS + cross-run mode
  agreement. R̂/ESS are necessary-not-sufficient (can't detect a *missed* mode); label which
  terms apply to which sampler.
- **Sampler-adequacy + dimension cap:** independent-chain (or independent-run) posterior
  agreement on held-out summaries must pass, and a hard nominal-dimension cap aborts above
  the ~20-effective-param envelope regardless of the emulator gate.
- **Diagnostics:** boundary-pileup (mass on a design-box wall ⇒ bounds too tight);
  local-emulator-variance (posterior mass where predictive std ≫ design-mean).

## Extinction / regime handling

- Log-emulation excludes any design point with a target species at biomass → 0. **Pin the
  rule to the seed-mean**: exclude a point if the mean-of-logs is undefined (any seed at 0)
  — record and report censored points (their count and location bias the posterior).
- Reject/flag non-equilibrated points (reuse existing CV/trend stability signals).
- Non-stationarity diagnostic: flag if `s²` varies strongly across the design or residuals
  correlate with location.

## Testing strategy

- **Unit:** `keying` (biomass≠ssb≠yield distinctness, reference_point_type routing);
  `output_stats` (wide-frame per-species indexing); `prior` (per-param transform, single
  application); each `Likelihood` vs hand values incl. the retained normalizer and
  split-normal; `emulator` `alpha` plumbing, natural-log fit, `predict→var`, and
  variance-returning CV.
- **Integration — well-specified synthetic** (CI, zero OSMOSE runs): recover known θ*
  within CI; predictive ranges cover held-out points.
- **Integration — misspecified synthetic:** data from a non-GP threshold function; assert
  the pipeline **fails the calibration gate loudly or widens** — never silent confident-wrong.
- **Real-data validation (required before trusting output; explicitly labeled as
  emulator-vs-simulator fidelity, NOT structural coverage):** held-out OSMOSE coverage;
  emulator-in-the-loop at posterior mode + tail draws; cross-species correlation PPC.
- **Determinism:** fixed seeds → reproducible summaries.

## Dependencies & packaging

`[uq]` extra (`emcee`, `arviz`, `dynesty`) per existing convention. Keep `uq/` out of
`calibration/__init__.py` eager imports; lazy-import the extras inside their modules. CI
installs `[uq]` for the synthetic tests (or skip-guard when absent).

## Phasing (too large for one plan — build as 4)

- **Phase 0 — emulator GP substrate (standalone).** `output_stats.py` + `keying.py` +
  `emulator.py` (own ARD/`alpha`/log/var GP + variance-returning CV). No change to
  `SurrogateCalibrator`. Milestone: distinct-keyed, natural-log emulator with a
  variance-returning CV, unit-tested.
- **Phase 1 — design + calibration gate.** `design.py` (seed loop, SSB enabled) + the
  probabilistic coverage/PIT gate + bounded design-growth loop. Milestone: "can we build a
  *calibrated* emulator," validated on the well-specified synthetic.
- **Phase 2 — inference.** `prior.py` + `likelihood.py` (Gaussian + BandFaithful, σ_disc²)
  + `posterior.py` + `sampler.py` (dynesty + emcee/multistart) + `run.py` (returns a
  **posterior-only partial `UQResult`**; the predictive layer completes it in Phase 3).
  Validated on synthetic posterior recovery **and** the misspecified-synthetic guard. Pins
  the knobs (`k` defaults, σ_disc², σ_seed²(θ) model, design-growth increment / `N_max`).
- **Phase 3 — predictive diagnostic + real-data validation.** `predictive.py` + cross-
  species PPC + held-out/emulator-in-the-loop validation + both-likelihood width report +
  `UQResult`.

## Open questions / future work

Promote correlated multi-output GP to required if the cross-species PPC fails materially;
full δ(θ) + target-covariance (with the caveat that neither cleanly buys honest coverage);
persist emulator + design matrix for reuse; dimension-reduction front-end for the 30–50-
param frontier.
