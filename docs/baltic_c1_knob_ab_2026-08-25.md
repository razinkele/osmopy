# Baltic C1 thermal-recruitment knob — A/B validation (2026-08-25)

**Verdict: PASS — all pre-registered criteria met.** Identity: zero violations
(bit-identical knob+0 vs off, 5 seeds × 9 species). Instrument: exact in every
knob arm. Monotonicity: herring declines monotonically 2,539,645 t → 1,448,408 t
(+2°C) → 368,297 t (+4°C). Elasticity is reported without a threshold, per
spec decision/Non-goals (below).

Spec: `docs/superpowers/specs/2026-08-25-baltic-c1-temperature-recruitment-scenario-knob-design.md`
(binding — decisions 1-9, §4, Non-goals). Plan:
`docs/superpowers/plans/2026-08-25-baltic-c1-thermal-knob.md`.

## Scope: the knob is HERRING-ONLY

Decision 4 pre-registered the enable rule for cod_west (sp0) — `beta1 < 0 and
p < 0.1 and beta1(detrended) < 0` — before the fit was run, with no
sign-forcing or tuning allowed. The fit
(`docs/baltic_c1_codwest_fit_2026-08-25.md`) returned `beta1 = -0.0276,
p = 0.887` (primary) and `beta1 = +0.0995` (detrended-T sensitivity, sign
flip). Both legs of the rule fail, so **cod_west ships with the gate
disabled** — no `reproduction.thermal.gate.*` keys are set for sp0 anywhere
in this A/B or in the scenario overlay. Every result below is herring (sp1)
only; the knob has no effect on cod_west or any other Baltic species.

## Run provenance

- Branch `c1-thermal-knob`, harness `scripts/baltic_c1_knob_ab.py`.
- 4 arms × 5 seeds × 50 yr: **off** (production config, no thermal-gate
  keys), **knob0** (series ≡ tref, factor ≡ exp(0) = 1.0), **knob2** (series
  ≡ tref+2°C, factor ≡ exp(2β)), **knob4** (series ≡ tref+4°C, factor ≡
  exp(4β)). Arm series files are generated at run time from `tref`, not
  committed data.
- Seeds: `[42, 123, 7, 999, 2024]` — the house set (`simulation.rng.fixed=true`).
- Herring constants: `beta.sp1 = -0.51` (Voss & Quaas 2026, quoted), `tref.sp1
  = 9.670314810741907` (full CSV precision, not the README's rounded display
  value of 9.67).
- Raw report: `docs/diagnostics/baltic_c1_knob_report.json` (copied verbatim
  from the run at `/tmp/c1_knob_report.json`).
- **This is a local validation run, not a CI gate.** Production certification
  is unaffected — the knob ships disabled by default and is only turned on
  through a scenario overlay (see below).

## (a) Identity — blocking, PASS

knob+0 (series ≡ tref everywhere, factor ≡ 1.0 exactly) vs off, per seed, per
species, raw (non-annualized) `.biomass()` arrays compared with
`np.array_equal`. Multiplying eggs by exactly `1.0` preserves the RNG stream
under `simulation.rng.fixed` semantics, so this is a bit-identity claim, not
a tolerance-band one.

| comparison | seeds | species | violations | verdict |
|---|---|---|---|---|
| knob0 vs off | 5 (42, 123, 7, 999, 2024) | 9 (all Baltic species) | 0 | PASS |

## (b) Instrument — PASS

The A/B harness recomputes `exp(β·ΔT)` independently from each arm's series
file + config and asserts the loader's (`_load_thermal_gate`) factor
trajectory equals it exactly. This is a loader-level check — no engine
output exposes the gate factor directly, and none was added for this task;
the biomass response in (b)/monotonicity is the run-level evidence the
forcing actually engaged the simulation.

| arm | sp1 (herring) factor == exp(βΔT) exactly |
|---|---|
| knob0 (ΔT=0) | yes |
| knob2 (ΔT=+2°C) | yes |
| knob4 (ΔT=+4°C) | yes |

(No row for sp0/cod_west — the species is not enabled in any arm.)

## (c) Monotonicity — PASS

Final-decade mean biomass (years 41-50 of 50, matching
`baltic_stability_certify.py`'s convention), averaged over the 5 seeds.

| arm | ΔT | herring final-decade mean (t) | vs knob0 |
|---|---|---|---|
| knob0 | +0°C | 2,539,645.16 | — |
| knob2 | +2°C | 1,448,407.65 | -43.0% |
| knob4 | +4°C | 368,297.19 | -85.5% |

Strictly decreasing across knob0 → knob2 → knob4: PASS. (A non-monotone
response was pre-registered as a FAIL and a finding per spec success
criterion 3 — not triggered.)

## (d) Elasticity — reported, no threshold

Per the Non-goals section: the knob multiplies eggs, upstream of the
engine's emergent early-life density dependence. The paper's β was fitted to
*total* recruitment with density-dependence already inside the S-R form, so
the realized biomass response is expected to be **damped** relative to the
naive `exp(βΔT)` prediction — that is stated as an expectation, not a pass
criterion, and the A/B reports the ratio without a threshold.

| arm | realized ratio (biomass) | expected ratio (exp(βΔT)) | elasticity (realized/expected) |
|---|---|---|---|
| knob2 (+2°C) | 0.5703 | 0.3606 | 1.582 |
| knob4 (+4°C) | 0.1450 | 0.1300 | 1.115 |

Reading: at +2°C the realized decline (0.570×) is substantially smaller than
the naive exponential prediction (0.361×) — density dependence damps the
biomass response relative to the raw recruitment-level forcing, as the
Non-goals section anticipated. At +4°C the stock has collapsed far enough
(368 kt of a ~2.5M t knob0 baseline) that density-dependent buffering has
largely run out of room, and the realized ratio (0.145×) sits much closer to
the naive expectation (0.130×) — near pass-through as the stock collapses.
Elasticity > 1 in both cases means "damped relative to the naive
prediction," consistent across both offsets.

## Labelled approximations (restated)

These carry through from the spec and fit doc; restating them here so anyone
reading only the A/B result does not miss them:

- **Herring complex ≈9% catch share.** The β = -0.51/°C coefficient is fitted
  to the western Baltic herring stock (her.27.20-24) alone in Voss & Quaas
  (2026), which is a minor member (~9% of catch) of the herring complex this
  model represents. Applying that coefficient to the whole complex is a
  **pattern-only, scenario-grade transplant**, not a validated per-stock
  fit — the review explicitly rejected catch-share-scaling the β as fake
  precision.
- **CMEMS-for-BSIO substitution.** The paper's temperature source is BSIO
  reconstructions; this implementation drives the series from CMEMS Baltic
  PHY multi-year reanalysis (`thetao`/`bottomT`) over SD22-24, a labelled
  substitution, not the paper's own product.
- **Window 1974-2021**, not 1974-2023. The spec's original text described a
  fixed 50-row 1974-2023 layout; the CMEMS product's actual data end
  (2021) is earlier, so the historical file — and the constant-T arms
  derived from its `tref`s — cover 1974-2021 (19 synthetic spin-up years at
  `tref`, 1993-2021 real years). This is a controller ruling made during the
  build (Task 2, spec-defect resolution), not a silent truncation: no run in
  this stage consumes the historical series past its real extent, and the
  fit script's lagged pairing depends only on the real (1993-2021) portion.
- **tref sourced at full CSV precision** (`9.670314810741907` for herring),
  not the README's rounded display value (`9.67`). Using the rounded value
  would have broken the knob+0 bit-identity criterion — see the round-trip
  CSV-parsing fix below.
- The engine change also fixed a related latent bug: `_load_thermal_gate`'s
  `pd.read_csv` call used pandas' default (non-round-trip) float parser,
  which is not correctly-rounding for `tref.sp1`'s mantissa (2-ULP error,
  `9.670314810741907` → `9.670314810741909`). This broke identity at engine
  scale (measurable divergence by year 2, not just a cosmetic bit flip) for
  every arm, not only knob+0. Fixed by adding `float_precision="round_trip"`
  to both the thermal-gate and RV-gate CSV loaders in `osmose/engine/config.py`
  (commit `ff1c421`, landed before this run).

## Deferred item

The spec (design §1) calls for adding `reproduction.thermal.gate.enabled` to
`osmose/runner.py:java_engine_block_reason`, mirroring the existing
oxygen-benthos-coupling block reason, so a direct Java run of a knob-enabled
config fails loudly instead of Java silently ignoring the keys (Java has no
thermal gate). **This edit is deferred**: `osmose/runner.py` carries
unrelated user-dirty changes (JVM-option-validation rework, unstaged, not
part of this branch's work) at the time of this run, and this task's file
list must not touch or stage user-dirty files. Follow-up: add the block-reason
entry once `osmose/runner.py` is committed by its owner or the dirty state is
otherwise resolved.

## B2 interface

The series-file format is the entire future scenario hookup: swapping
`reproduction.thermal.gate.series.file` to point at a different CSV (e.g. an
RCP4.5/8.5 projection series) is sufficient to drive the knob with future
climate data through the same loader, with no further engine changes. No
RCP series is built in this stage (Non-goals).

## Scenario status — not a certification statement

At +4°C herring's final-decade mean (368,297 t) sits below its ~800 kt ICES
envelope floor used elsewhere in Baltic certification. **This is a scenario
trajectory under an explicit +4°C constant-temperature offset, not a
certification result.** Production certification (`scripts/baltic_stability_certify.py`)
is climatological (uses the historical/production forcing, not this knob)
and is untouched by this work — the knob ships disabled in the production
config and is only active when a scenario overlay
(`data/baltic/calibration_results/c1_thermal_knob_arm.json`) is explicitly
merged in.

## Deliverables

- Engine: `exponential` response + guards in `osmose/engine/config.py` /
  `osmose/engine/processes/thermal_gate.py` (commits `b04d32d`, `849865c`,
  `ff1c421`).
- Data: `scripts/build_baltic_thermal_sr_series.py`,
  `data/baltic/forcing/baltic_thermal_sr_series.csv` + `.README.md`
  (commits `30a8d15`..`11e348c`).
- Fit: `scripts/fit_codwest_thermal_sr.py`,
  `docs/baltic_c1_codwest_fit_2026-08-25.md` (commits `11fc458`, `fbd02fc`).
- A/B: `scripts/baltic_c1_knob_ab.py`, `tests/test_baltic_c1_knob_helpers.py`
  (commit `3f577ee`); this results doc + copied report +
  `data/baltic/calibration_results/c1_thermal_knob_arm.json` (this task).
