# Baltic B2 — RCP × load scenario arms from literature deltas (evidence-adjusted)

**Date:** 2026-08-29
**Status:** approved (design), **revised same day after a deep 6-lens adversarial review**
(19 agents incl. a live literature-mining lens; 13 confirmed findings, 0 refuted, ~17 minors —
all folded in). The review resolved the spec's one pre-registered unknown itself: the ΔO₂
numbers are tabulated in the anchor paper (Table 10), making the planned companion-paper
sourcing pass unnecessary; it also corrected a miscopied ΔT, exposed a reference-period
double-count, redesigned the confounded check (c), and quantified the O₂ axis's thinness on the
actual production field.
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` scenario
track (B2). **Sourcing decision (user, 2026-08-29): literature deltas** — not raw ERGOM field
swaps (inter-model bias vs the calibrated baseline) and not field acquisition (no clean CDS
dataset; C2(b)-class stall risk). Field-derived deltas later = a JSON edit only.
**Related:** `docs/baltic_c1_knob_ab_2026-08-25.md` (temperature machinery + exactness chain),
`docs/baltic_hypoxia_benthos_ab_2026-08-09.md` / `_certification_2026-08-09.md` (the live
O₂→benthos-K coupling; measured −21.4 % cod_east / −18.7 % flounder for coupling-on at baseline
O₂; cod_east across-seed spread ±1.9 % — the noise floor the O₂ axis must be read against),
`docs/baltic_f_hindcast_2026-08-23.md` (F1 null: transient tracking is unusable; equilibrium
time-slice response is the honest scenario mode — stated as doctrine, not as a validated
strength).

## The anchor numbers, verified and corrected (review, 2026-08-29)

Meier, H.E.M., et al. (2022). ESD 13, 159–199, doi:10.5194/esd-13-159-2022 (open access; no
editorial notices). All values are CLIMSEA (RCO-SCOBI, four ESMs), **1976–2005 → 2069–2098**:

* **ΔT (annual-mean SST, Table 7):** **RCP4.5 +1.9 °C; RCP8.5 +2.9 °C.** (The originally drafted
  +2.0 was the RCP4.5 *summer* cell; the "+2.3…+4.7" spread belongs to the coupled RCSM's RCP8.5
  ensemble, mean +3.5 °C, Gröger et al. 2019 — reported as context only. Secondary anchor if a
  volume-mean is preferred: Saraiva et al. 2019b, +1.6 ± 0.5 / +2.7 ± 0.4 °C.)
* **ΔO₂ (summer-mean BOTTOM oxygen, Table 10, CLIMSEA ensemble-mean-SLR column, mL/L →
  mmol m⁻³ at 44.66):**
  | | RCP4.5 | RCP8.5 |
  |---|---|---|
  | **BSAP** | +0.6 → **+26.8** | +0.4 → **+17.9** |
  | **REF** | 0.0 → **0.0** | −0.2 → **−8.9** |
  Labels that travel with every use: *summer-only* values applied year-round; *SLR-variant*
  (ensemble-mean sea-level rise); referent = **bottom O₂** — the same referent as our forcing
  variable `o2b`, which is what makes these numbers portable (the review showed a deep-water
  referent would differ by ~10×; the schema forbids mixing referents).
  **RCP4.5×REF is a sourced ZERO** — that arm is temperature-only *by the table*, a designed
  null O₂ contrast, not a degenerate case.
* **Load-dominance caveat** (verified verbatim) and **salinity-not-robust**: unchanged from v1.
  "Phyto/zoo not quantified" is softened to: not quantified *in a form portable to our LTL
  forcing* (the paper discusses phytoplankton qualitatively); LTL stays at baseline, labelled.

## The O₂ axis, quantified on our own field (review computation — design-shaping)

On `data/baltic/baltic_oxygen_bottom.nc` (verified: 24 frames, `o2b`, mmol m⁻³, 2024 CMEMS
analysis) masked to the 616 ocean cells and weighted by benthos base K: baseline K-weighted Hill
factor **0.866**; the field is bimodal (14.8 % of ocean cell-frames below c50=60; 75.5 % ≥ 200
where the curve is flat; 8.6 % near-anoxic). Consequences, pre-registered:

* Effective-K response to the Table-10 deltas is **thin and asymmetric**: roughly +2–3 %
  (BSAP×RCP4.5, +26.8), +1.5–2 % (BSAP×RCP8.5), 0 (REF×RCP4.5), ~−1 % (REF×RCP8.5) — the last
  **below the ±1.9 % cod_east seed spread**. Stock-level O₂ contrasts may legitimately drown in
  seed noise; the design therefore separates *wiring* checks (deterministic, blocking) from
  *ecological* contrasts (reported against a printed noise floor).
* The **builder computes and prints the predicted effective-K change per arm** (field + Hill,
  no engine run) and the results doc prints it beside each arm's stock deltas.

## Decisions (recorded; 4–7 rewritten in the post-review revision)

1. **Sourcing: literature deltas** (user). 2. **Matrix: baseline + {RCP4.5, RCP8.5} ×
   {BSAP, REF}, end-century** — now with all four ΔO₂ cells sourced from Table 10 (above).
3. **Drivers/application** (unchanged in structure): ΔT via the C1 knob (herring-only,
   constant-T series at tref+ΔT); ΔO₂ as an additive offset on the O₂ NetCDF — **wet cells
   only** (land/NaN conventions preserved), floored at 0, 24 frames asserted in and out. The
   existing `oxygen.offset` config key is deliberately NOT used: it lacks the wet-mask/floor
   semantics and a per-arm file is auditable; this choice is now stated rather than implicit.
4. **ΔO₂ = Table 10 CLIMSEA ensemble-mean-SLR cells** (the sourcing pass is dissolved). The
   delta-spec JSON schema requires per-number: value, unit, referent (`summer_bottom_o2` —
   the only accepted value this stage), source string (table/column/variant), and the
   conversion factor. Mixed referents are schema-invalid.
5. **Reference-period honesty (the double-count finding):** the literature deltas are
   end-century **vs 1976–2005**, but our baselines already sit decades later (O₂ file = 2024
   analysis; knob tref = 1993–2021 mean). Ruled: **apply the raw deltas, relabelled** — every
   arm is "Meier end-century delta applied on a present-day baseline; overstates end-century
   forcing by the realized 1976-2005→present component" — recorded in the JSON, the results
   table caption, and §4(d)'s label list, with a cited context estimate of the realized SST
   component reported (not subtracted). Subtracting would stack a second, uncertain literature
   layer for a labelled scenario device; the honest relabel is cheaper and clearer.
6. **No envelope pass/fail for scenario arms** (unchanged). 7. **Timing honesty:** 6 arms ×
   5 seeds × 50 yr ≈ **2.5–3 h** at C1's measured pace (not "~2 h").

## Design

### 1. Delta-spec JSON — `data/baltic/scenarios/b2_literature_deltas.json`

Per arm: `{name, rcp, load, dT_C, dO2: {value_mmol_m3, referent, source, conversion} | null}`.
The dead knobs of v1 (`ltl_scale`, `salinity`, `time_slice`) are **removed** — the JSON carries
only what machinery consumes; the upgrade path re-adds fields when their machinery exists.
Schema validation (CI): citations present for every number; referent fixed; RCP4.5×REF's
sourced zero carries its citation like any other value.

### 2. Builder — `scripts/build_baltic_b2_forcing.py`

Reads delta spec + production forcing; emits per-arm artifacts into a caller-supplied run dir:
knob constant-T series (C1 conventions) and, for arms with non-null dO₂ (incl. the sourced
zero, which emits a value-identical copy), the offset O₂ NetCDF (wet-only, floor 0, frames
asserted). Prints the predicted effective-K change per arm. **Zero-delta self-check (blocking):
the re-read written zero-arm files are value-identical (NaN-aware comparison) to the production
inputs.**

### 3. Harness — `scripts/baltic_b2_scenario_ab.py`

Arms: `baseline`, `zero` (all machinery engaged, zero deltas — **bit-identical to baseline per
seed, blocking**), `rcp45_bsap`, `rcp45_ref` (temperature-only by source), `rcp85_bsap`,
`rcp85_ref`. Overlays: C1 knob keys + `oxygen.filename` at the arm's generated file (absolute
path). 5 house seeds × 50 yr. Results JSON **committed** to
`docs/diagnostics/baltic_b2_scenario_report.json` (not /tmp-only — the C1 precedent's gap).

### 4. Pre-registered checks — blocking vs reported, made crisp

**BLOCKING (any failure = wiring bug, stop, no interpretation):**
* (a) zero-arm bit-identity to baseline, per seed;
* (b) **O₂ load-through assert**: per arm, the harness loads the arm's `EngineConfig` and
  asserts the engine-held O₂ array equals the builder's written array (kills the verified
  silent-fallback trap where a non-resolving `oxygen.filename` reverts to coupling-defaults);
* (c) **deterministic Hill ordering**: per arm, f_o2_hill over the arm's field vs the baseline
  field obeys the delta's sign per cell (guaranteed by monotonicity — a violation is wiring);
* (d) **knob factor instrument through the loader's float path**: expected factor =
  `exp(beta · (float(str(tref+dT)) − tref))` — exact `array_equal` (the review proved plain
  `exp(beta·dT)` fails by 3 ULP at the non-dyadic ΔT=2.9; C1's exact recovery at 2.0/4.0 was
  dyadic luck, recorded here so the JSON upgrade path doesn't trip on it).

**REPORTED (no pass/fail):**
* herring's decline per arm (direction context from C1's A/B — +1.9/+2.9 interpolate its
  tested 0/+2/+4 range);
* the **within-RCP load contrast** (BSAP vs REF at identical ΔT — the only clean ecological
  read of the O₂ axis) for cod_east and flounder, printed beside the predicted effective-K
  change and the ±1.9 % seed-noise floor;
* all labels, restated in §4(d)'s list: SST-for-bottom-T proxy (annual SST applied to a Q4
  bottom-T knob; deep warming runs *higher* than SST in ventilated basins, so the herring
  decline is likely **understated** — direction now stated); summer-only + SLR-variant ΔO₂
  applied year-round; uniform-offset spatial blindness + floor asymmetry; LTL-at-baseline
  (the BSAP cells are a partial-load world: the load cut's O₂ benefit enters but its
  plankton/nutrient pathways do not — the omitted pathways plausibly oppose the included one);
  reference-period overstatement (decision 5); **cod_east's trajectory partly prescribed by
  the RV narrative series** (gate factor 0.32–0.87 across the scored decade) — its scenario
  deltas are conditioned on that prescription.

### 5. Deliverables

Delta-spec JSON (all numbers cited), builder + tests, harness + tests, one 6-arm run, dated
results doc whose headline is the 2×2 scenario table with the predicted-ΔK and noise-floor
columns, memory update. Upgrade path restated: field-derived deltas = JSON edit.

## Non-goals (YAGNI)

No field acquisition; no LTL/salinity deltas; no mid-century slices; no recalibration; no new
engine mechanisms; no envelope claims; no cod-recruitment temperature response (C1's verdicts
stand); no claim that arms represent calibrated end-century states (decision 5's relabel).

## Testing

CI-safe: schema validation (citations/referent/no-dead-knobs); builder wet-mask/floor/frames/
zero-identity (NaN-aware) on synthetic fixtures; predicted-ΔK computation on a synthetic field;
harness helpers (overlay construction; the §4(d) float-path expected-factor math incl. a
non-dyadic ΔT case; Hill-ordering check on synthetic fields). NOT CI: the 6-arm run.

## Success criteria

1. Delta spec ships with every number cited to table/column/variant; RCP4.5×REF's sourced zero
   documented as a designed null.
2. All four BLOCKING checks pass; any failure stops interpretation.
3. The 2×2 results table exists with predicted-ΔK and noise-floor columns; the load-dominance
   caveat and decision-5 relabel lead the O₂ commentary; every §4 label present.
4. Field-derived-delta upgrade = JSON edit, stated in the results doc.
