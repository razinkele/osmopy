# Baltic C1 — temperature-dependent recruitment as a scenario knob (Voss & Quaas form)

**Date:** 2026-08-25
**Status:** approved (design), **revised same day after adversarial review** (15-agent, 5-lens
workflow; 10 confirmed findings — two root clusters — plus ~24 minors, all folded in; 0
refutations). Headline correction: the original knob+0 arm was arithmetically incapable of being
an identity arm (Jensen inflation 9–18% at β=−0.51 with real Baltic σ_T, plus the warming trend
putting the scored decade 0.5–1 °C above T_ref → 20–35% herring suppression with no bug present).
The A/B is redesigned around **constant-temperature arms**, which make the identity claim exact.
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` scenario
track (C1). **Scoping decision (user, 2026-08-25):** C1 is a *scenario knob* — a labelled
encoding of the Voss & Quaas aggregate productivity finding for scenario runs — not a validated
mechanism and not a hindcast device (post-F1 doctrine, `docs/baltic_f_hindcast_2026-08-23.md`).
**Related:** `docs/baltic_thermal_recruitment_shape_2026-08-10.md` (the three blocks),
`docs/baltic_recruitment_pathway_2026-08-10.md` (the 7-stage chain),
`docs/baltic_herring_phenology_a0_2026-08-12.md` (the pathway this design does NOT take).

## The anchor citation, verified (2026-08-25, scite + full text)

Voss, R. & Quaas, M. F. (2026). *Future fishing potential of cod and herring under climate change
in the Western Baltic Sea.* ICES JMS 83(4), doi:10.1093/icesjms/fsag033 — real, gold OA (CC-BY),
no editorial notices, **no supplement** (the review checked: Conradt's cod coefficient is
published nowhere accessible — our self-fit is the sole source for cod_west's β, decision 4).

* **Form (Methods, quoted):** `ln(R) = ln(e^(−β₀+β₁·T)·SSB/(1+β₃·SSB)) + ε` — Beverton–Holt with
  an **exp(β₁·T)** productivity term = a per-year multiplicative factor in this engine's terms.
* **Herring:** β₁ = **−0.51 /°C**, driver **bottom temperature Q4** (adj. R² 0.817). Stock
  **her.27.20-24**. **Cod:** driver **SST Q3**, coefficient in Conradt (2023, Univ. Hamburg
  dissertation). Stock **cod.27.22-24** = this model's **cod_west**.
* Paper temperature source: BSIO reconstructions; scenarios RCP4.5/8.5.

**Scoping corrections the verification forced:** targets are **cod_west** (no RV gate — no
conflation) and the **herring complex**, where the western stock the coefficient belongs to is a
**minor member (~9% of complex catch)** — the transplant is a *pattern-only, scenario-grade*
approximation, labelled as such everywhere (review considered catch-share-scaling β and rejected
it: it would fake precision the transplant does not have).

## The three blocks this design routes around (2026-08-10/12)

1. The existing gate's logistic **rises** with T — wrong sign at any parameterisation.
2. Herring's literature mechanism is phenological, and A0 measured the model's phenology response
   **opposite** to the Polte prediction — this design encodes the aggregate productivity effect
   (a scalar), which A0 did not test.
3. cod_east's drivers are already represented and RV-dominated — **no cod_east knob**.

## Decisions (recorded; 6–9 added/rewritten in the post-review revision)

1. **Scope: scenario knob for cod_west (sp0) and herring (sp1)** (user). Existing files in
   `data/baltic/` stay byte-identical; this stage ADDS a forcing CSV and an overlay JSON; the
   knob is enabled only in scenario overlays; certification stays climatological.
2. **Response: `factor(y) = exp(beta · (T(y) − tref))`** added to the existing thermal gate.
   No cap (the paper's form has none); factor > 1 in cold years is legitimate. **Honesty note
   (review):** tref-anchoring makes the *geometric* mean 1, not the arithmetic mean — over a
   variable series the arithmetic mean factor is exp(β²σ²/2) > 1 (≈1.09–1.18 for the real
   series). The design does not claim otherwise anywhere; the A/B (§4) uses constant-T arms
   precisely so no such claim is needed.
3. **Herring β = −0.51 /°C** (quoted), with the ~9%-share transplant label of the scoping note.
4. **cod_west β: fit ourselves, pre-registered — alignment included** (review cluster 2):
   recruitment for cod.27.22-24 is **age-1**, so the fit pairs **R_{y+1} (assessment rows
   1994–2022) with SSB_y and SST-Q3_y for hatch years y = 1993–2021** (≈29 points; the cached
   snapshot is the advice-2022 assessment — rows end 2022, and the stock is category-3 since
   2024). Primary fit includes the terminal pair; report leave-one-out-terminal sensitivity.
   **Enable cod_west iff** (a) fitted β₁ < 0 with p < 0.1 **and** (b) the sign survives fitting
   against linearly detrended T (guards the 22–27% false-positive rate a trending non-causal T
   produces at this gate — review measurement). Otherwise cod_west ships disabled and the knob
   is herring-only; no sign-forcing, no tuning. Cross-check vs the paper is impossible (no
   supplement) — say so in the fit doc.
5. **Drivers mirror the paper** for the *historical* series: cod_west ← SST (`thetao` surface)
   Q3 mean, herring ← bottom T (`bottomT`) Q4 mean, both over SD22–24, CMEMS Baltic PHY
   multi-year reanalysis (labelled substitution for BSIO). **Data reality (review):** the local
   cache holds `thetao` only 1993–2010 and **no `bottomT`** — the builder must download the
   remainder (credentialed via `.env`; the MY product's live end is ≥2021). The window is
   **1993 → the product's actual end**, stated in provenance; `tref` = each series' mean over
   that window. **The A/B (§4) does not depend on these downloads** — constant-T arms need only
   the `tref` numbers, and the cod fit needs `thetao` only (partially cached). If `bottomT` is
   unobtainable, herring's tref falls back to a documented literature constant and the
   historical herring series is deferred to B2 — recorded, not blocking.
6. **A/B arms use constant-temperature series** (review cluster 1 — the design change):
   * **off** — production config.
   * **knob+0** — knob on, series T(y) ≡ tref for all 50 rows → factor ≡ exp(0) = 1.0 exactly →
     **pre-registered as BIT-IDENTICAL to off per seed** (multiplying eggs by exactly 1.0
     preserves the RNG stream; `simulation.rng.fixed` semantics). No tolerance concept needed —
     the criterion the review proved unimplementable is gone.
   * **knob+2 / knob+4** — series ≡ tref+2 / tref+4 → every year's factor is exactly
     exp(2β) / exp(4β) (herring: 0.360 / 0.130).
   * The **historical-series arm is dropped from pass/fail** (it conflates Jensen inflation,
     the warming trend, and the scenario signal); it may be run later as a labelled extra.
7. **Series-file format (review):** the thermal-gate loader reads a `year,temp_sp{N},...` CSV
   with **contiguous ascending years and no `#` comment lines** (comments crash it — verified).
   Layout: 50 rows, years **1974–2023** where 1974–1992 are synthetic spin-up years at tref and
   1993–2023 carry the values (historical file) or tref+ΔT (arm files, generated by the harness
   into a temp dir at run time — they are derived artifacts, not committed data). Provenance
   lives in a sidecar `*.README.md`, not in the CSV. `reproduction.thermal.gate.start.year` is
   left unset (defaults to the file's first year; offset 0).
8. **Config surface (review — corrected key inventory):** `reproduction.thermal.gate.response`
   (`logistic` default | `exponential`) and `reproduction.thermal.gate.beta.sp{N}` are **new**;
   `reproduction.thermal.gate.tref.sp{N}` **already exists with a silent 20.0 °C default and
   thermal_cap semantics** — under `response=exponential` the loader must REQUIRE an explicit
   tref (raise if defaulted; inheriting 20.0 silently would be a wrong-anchor bug). Mode
   interaction: under `exponential`, `mode` must be absent or the new value `raw` (factor
   applied as computed, floored at `.floor`, default 0); `thermal_cap`/`mean_preserving` are
   rejected (rationale, corrected per review: per-arm renormalisation would eat the +ΔT offsets
   — not the original 'tref already normalises' claim, which was false); `raw` is invalid under
   `logistic`.
9. **Negative-offset guard, correctly described (review):** `offset = start_year − first_year`
   negative silently misbehaves in **both** gates but by different mechanisms — the RV gate's
   `min(offset+year, n−1)` produces Python negative indexing (reads from the series END); the
   thermal gate's `(offset+year) % n` wraps to a wrong-but-in-range year. Add `offset >= 0`
   validation to both loaders. The other silent class (run longer than the series under the
   thermal gate's modulo) is Stage-2 time-policy scope — noted open, not fixed here.

## Non-goals (YAGNI)

* No phenology coupling; no cod_east knob; no new gate — one response shape in `thermal_gate.py`.
* No hindcast-skill claim ever. No RCP series in this stage (B2 supplies future series through
  the same CSV interface). No recalibration.
* No claim that the transplanted β yields the paper's *net recruitment* response: the knob lands
  on eggs, upstream of the engine's emergent early-life density dependence, which the paper's β
  (fitted to total recruitment with DD inside the S–R form) already partially contains — so the
  realized biomass elasticity is expected to be **damped** relative to exp(βΔT), and the A/B
  reports that ratio without a threshold. (Review: the original criterion-(c) wording
  "recruitment-level suppression by construction" mislabelled this; corrected.)

## Design

### 1. Engine: `exponential` response + guards in the thermal gate

Per decisions 2, 8, 9. Schema fields for the two genuinely new keys (`response`, `beta.sp{idx}`)
— both are read via literal `cfg.get` patterns in `config.py`, so the AST walker captures them
and **no allowlist/frozen-snapshot edits are needed** (review-verified; unlike F1's keys). Run
`tests/test_schema_engine_key_parity.py` and `tests/test_issue_123_known_but_unread_keys.py`
anyway as the guard. Update CLAUDE.md's registry count (264 → 266). **Java:** add
`reproduction.thermal.gate.enabled` to `osmose/runner.py:java_engine_block_reason` — Java has no
thermal gate and currently ignores the keys silently (review); blocking a direct Java run of a
knob-enabled config mirrors the oxygen-coupling precedent.

### 2. Data: `scripts/build_baltic_thermal_sr_series.py`

Per decisions 5 & 7. Emits `data/baltic/forcing/baltic_thermal_sr_series.csv`
(`year,temp_sp0,temp_sp1`, 50 rows, no comments) + sidecar README with product IDs, bbox,
months, window, tref values, generation date. Downloads what the cache lacks (thetao 2011→end,
bottomT full window) via the credentialed CMEMS path; degrades explicitly (herring tref
fallback, decision 5) rather than silently.

### 3. Fit: `scripts/fit_codwest_thermal_sr.py`

Per decision 4: lagged pairing, ~29 hatch-year points, log-scale fit of the paper's form
(`scipy.optimize`), reports β₁ ± CI, p, the detrended-T sensitivity, leave-one-out-terminal
sensitivity, and the enable/disable verdict. Writes a dated results doc.

### 4. Validation A/B — constant-T arms, pre-registered (decision 6)

5 house seeds × 50 yr via `scripts/baltic_c1_knob_ab.py`. Arm series files generated at run
time from tref (+ΔT). Pass criteria:

* **(a) Identity (blocking):** knob+0 trajectories **bit-identical** to off, per seed (compare
  final-decade series exactly; any deviation = wiring bug, stop).
* **(b) Monotonicity:** each enabled species' final-decade mean declines monotonically across
  +0 → +2 → +4.
* **(c) Instrument:** the applied factor is deterministic by construction (constant series);
  the harness independently recomputes exp(βΔT) from the arm's series file + config and asserts
  the loader's factor trajectory equals it (loader-level check — no engine output exposes the
  gate factor (review), and none is added; the biomass response in (b) is the run-level
  evidence the forcing engaged, since exp(2β)=0.36 for herring is far outside seed noise).
* **(d) Reported without threshold:** realized biomass elasticity vs exp(βΔT) (expected damped
  — see Non-goals), and every labelled approximation restated in the results doc.

### 5. Deliverables

Engine change + tests; builder + series CSV + README; fit script + fit doc; A/B harness + dated
results doc; scenario-overlay JSON `data/baltic/calibration_results/c1_thermal_knob_arm.json`
(knob keys only — the horizon is the harness's business; the convention directory exists and
holds the depletable-arm JSONs). The B2 interface is the CSV format: swapping the series file is
the entire future hookup.

## Testing

* CI-safe: exponential response math (factor 1.0 exactly at T=tref; exp(βΔT) scaling; floor);
  loader — explicit-tref requirement under exponential, mode rejection matrix (cap/mp ×
  exponential; raw × logistic), negative-offset guard on BOTH gates (each gate's own mechanism
  exercised), contiguous-years acceptance of the 1974-labelled spin-up block; schema parity +
  frozen snapshot runs; builder Q3/Q4 selection + layout on synthetic fixtures; fit script
  recovers a known β from synthetic data, applies the lag correctly (synthetic age-1 fixture),
  and the detrend sensitivity flips a synthetic trend-only β.
* NOT CI: the A/B arms (local, documented).

## Success criteria

1. Engine + guards land with tests; production certification unchanged; Java block-reason
   covers the knob.
2. Series + trefs derived (or the herring fallback documented); fit verdict documented either
   way, with both sensitivity checks.
3. A/B: (a) bit-identity holds; (b) monotone declines; (c) loader-factor instrument exact;
   (d) elasticities reported. A non-monotone response is a FAIL and a finding.
4. The overlay JSON + CSV interface is the complete B2 hookup, stated in the results doc.
