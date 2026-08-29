# Baltic B2 — RCP scenario arms from literature deltas (evidence-adjusted)

**Date:** 2026-08-29
**Status:** approved (design), pending adversarial review, then implementation plan.
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` scenario track
(B2, "Scenario forcing from ERGOM RCP output (offline, one-way)").
**Sourcing decision (user, 2026-08-29): literature deltas** — ensemble-mean changes from the
open-access Baltic Earth assessment applied to OUR production forcing, not raw ERGOM field swaps
(inter-model bias against the calibrated baseline would confound every scenario delta — the
C2(b)-withdrawal trap class) and not field acquisition (no clean CDS dataset found; acquisition
risk graded higher than the spec's original "data only"). Upgrading to field-derived deltas later
touches only the delta-spec JSON (§2) — the same interface discipline as C1's CSV swap.
**Related:** `docs/baltic_c1_knob_ab_2026-08-25.md` (the temperature machinery and its verified
exactness chain), `docs/baltic_hypoxia_certification_2026-08-09.md` (the live O₂→benthos-K
coupling and its measured response signs), `docs/baltic_f_hindcast_2026-08-23.md` (the F1-null
doctrine: this model's scenario value is equilibrium response to sustained forcing change —
time-slice deltas are its honest use).

## The anchor literature, verified (2026-08-29, full-text fetch)

Meier, H.E.M., et al. (2022). *Oceanographic regional climate projections for the Baltic Sea
until 2100.* Earth Syst. Dynam. 13, 159–199, doi:10.5194/esd-13-159-2022 — open access; the
CLIMSEA ensemble (RCO-SCOBI 3.7 km, four ESMs: MPI-ESM-LR, EC-EARTH, IPSL-CM5A-MR, HadGEM2-ES).

Three findings that reshape B2 from the parent spec's assumption:

1. **Temperature is the robust axis.** Annual-mean SST change 2069–2098 vs 1976–2005:
   **RCP4.5 +2.0 °C; RCP8.5 +2.9 °C** (RCO-SCOBI; ESM spread +2.3…+4.7). Bottom-water warming
   is noted (elevated in sporadically ventilated basins via warmer inflows) but **not separately
   tabulated** — applying the SST delta to the herring bottom-T knob is a labelled proxy.
2. **Bottom-O₂'s future is nutrient-load-dominated, not climate-dominated** (the paper's own
   caveat: climate impact on biogeochemistry "still smaller than that of plausible nutrient
   input changes"). Sign depends on the load trajectory: BSAP abatement → O₂ *improves* even
   under warming; reference (REF) loads → slight decline under RCP8.5. **A climate-only O₂
   delta is ill-defined; the scenario axis must be RCP × load.**
3. **Phytoplankton/zooplankton changes are not quantified** in the assessment, and **salinity
   change is not robust** (freshening ≈ offset by sea-level-rise-enhanced inflows; ensemble
   spread exceeds signal — CLIMSEA vs the older BalticAPP/ECOSUPPORT freshening). Consequences:
   LTL forcing stays at baseline (labelled), salinity delta is zero (labelled; also a citable
   datum lowering C4's urgency relative to the parent spec's "first-order driver" framing).

## Decisions (recorded)

1. **Sourcing: literature deltas** (user, 2026-08-29; rationale above).
2. **Scenario matrix: baseline + {RCP4.5, RCP8.5} × {BSAP, REF}, end-century (2069–2098 vs
   1976–2005).** Four scenario arms + baseline. Mid-century slices are out of scope this stage
   (the delta-spec format carries a time-slice field so they can be added by JSON edit alone).
3. **Drivers and application:**
   * **Temperature** → the C1 thermal knob (herring-only, per its shipped verdict): constant-T
     arm series at `tref + ΔT_RCP`. ΔT is the same for BSAP and REF cells of one RCP (loads do
     not set temperature). SST-for-bottom-T proxy labelled.
   * **Bottom-O₂** → the production `baltic_oxygen_bottom.nc` (24 frames — the frame-count trap
     in CLAUDE.md; the builder asserts 24 in and 24 out): additive uniform offset in mmol m⁻³,
     floored at 0. The Hill response (c50=60, n=3) translates the offset nonlinearly per cell —
     that is the mechanism working, not a bug.
   * **LTL biomass: no change** (finding 3); **salinity: no change** (finding 3); both restated
     in every results table.
4. **ΔO₂ numbers: to be sourced in the spec-finalisation pass, pre-registered procedure.** The
   assessment gives signs but not portable numbers. The implementation's first task is a
   scientific-validation pass over the BalticAPP/companion literature (Saraiva et al. 2019a/b,
   Meier et al. 2018/2019) for ensemble-mean bottom-O₂ or deep-water-O₂ changes per
   (RCP × load) cell, in mmol m⁻³ (1 mL/L = 44.66 mmol m⁻³). **Acceptance per cell:** a number
   usable as a basin-scale delta with a quotable source. **Any cell without one ships
   temperature-only and the results doc says so** — no invented numbers, no digitised-figure
   guesses without labelling them as such.
5. **No envelope pass/fail for scenario arms.** Scenarios legitimately exit calibration
   envelopes. Pre-registered checks instead (§4). Certification stays climatological and
   untouched; production `data/baltic/` existing files byte-identical (this stage ADDS one
   delta-spec JSON and one results doc; scenario forcing files are generated at run time, never
   committed).

## Design

### 1. Delta-spec JSON — the single source of scenario numbers

`data/baltic/scenarios/b2_literature_deltas.json`: per arm
`{name, rcp, load, time_slice, dT_C, dO2_mmol_m3, ltl_scale: 1.0, salinity: 0.0,
citations: {dT: "...", dO2: "..."}}` — every number carries its citation string; `null` for an
unsourced dO2 (→ temperature-only arm). A `reference` block records the baseline periods. This
file is the entire upgrade surface for later field-derived deltas.

### 2. Builder — `scripts/build_baltic_b2_forcing.py`

Reads the delta spec + production forcing; emits per-arm artifacts into a caller-supplied run
dir: the knob's constant-T series CSV (reusing `write_arm_series`'s conventions from
`scripts/baltic_c1_knob_ab.py`) and, for arms with `dO2`, a modified copy of
`baltic_oxygen_bottom.nc` (additive offset, floor 0, **assert 24 frames in and out**, same
dtype/attrs). **Zero-delta self-check (blocking, builder-level):** applying the all-zero delta
must yield value-identical arrays to the production inputs (`np.array_equal` per variable) —
the C1 identity discipline at the file level.

### 3. Harness — `scripts/baltic_b2_scenario_ab.py`

Arms: `baseline` (production config), `zero` (all machinery engaged, zero deltas —
**pre-registered bit-identical to baseline per seed**, the C1 trick), plus the four scenario
arms. Overlays per arm: C1 knob keys (herring beta/tref, tref+ΔT series file) +
`oxygen.filename` pointed at the arm's generated NetCDF. 5 house seeds × 50 yr. Report:
final-decade mean per species per arm, deltas vs baseline, written to a JSON + dated results
doc.

### 4. Pre-registered checks

(a) **zero-arm bit-identity** to baseline (blocking — any deviation is wiring, stop);
(b) **herring declines in every arm** (ΔT>0 in all four; direction established by C1's A/B);
(c) **O₂-delta arms move cod_east and flounder in the coupling's known direction** (the
2026-08-09 gate measured −21.4 % / −18.7 % for the coupling switching ON at baseline O₂; a
positive ΔO₂ (BSAP cells) must not *decrease* their benthos-mediated food, and vice versa —
sign check only, no magnitude threshold);
(d) everything else is **reported, not gated** — per-species deltas with all labels (SST proxy,
uniform-offset spatial blindness, LTL-at-baseline, load-dominated-O₂ caveat) restated.

### 5. Deliverables

Delta-spec JSON (with the sourcing-pass results baked in), builder + CI-safe tests
(offset/floor/frame-assert/zero-identity on synthetic fixtures), harness + helper tests, one
scenario run (6 arms × 5 seeds × 50 yr ≈ 2 h engine), dated results doc, memory update. The
results doc's headline is the four-cell scenario table for the assessed stocks.

## Non-goals (YAGNI)

No field acquisition/regridding (upgrade path only). No LTL or salinity deltas. No mid-century
slices this stage. No recalibration; no envelope claims for scenario arms. No new engine code —
the C1 knob and the O₂ coupling are the only mechanisms, both shipped and verified. No
projection of cod_west/cod_east recruitment via temperature (C1's fit refused cod_west;
cod_east's RV narrative is prescribed — both restated as scenario-scope limits in the results
doc).

## Testing

CI-safe: delta-spec schema validation (citations present for every non-null number); builder
offset/floor/frames/zero-identity on synthetic NetCDF fixtures; harness helpers (arm-overlay
construction, expected knob factors reusing C1's `expected_factors`). NOT CI: the scenario run.

## Success criteria

1. Every number in the shipped delta spec carries a verified citation; unsourced dO2 cells
   ship temperature-only and are named in the results doc.
2. Builder zero-identity + frame asserts pass; harness zero-arm bit-identity passes.
3. The four-cell scenario table exists with directions consistent with checks (b)/(c); the
   results doc restates every label and the load-dominance caveat as the FIRST interpretive
   sentence of the O₂ commentary.
4. Upgrade path stated: field-derived deltas = a JSON edit, nothing else.
