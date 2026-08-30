# Baltic C4 — salinity-gate sensitivity arms (mechanism characterization, not projection)

**Date:** 2026-08-30
**Status:** approved (design), **revised same day after a 5-lens adversarial review** (13 agents;
2 confirmed majors, 5 downgraded-with-substance, 1 refuted, ~20 minors — all folded in). The
review's computed numbers are now stated expectations (below); the instrument was redesigned
around what the movement sampler actually transmits; a third lever (−3 PSU) was added
pre-registration because the original two cannot reach the exclusion regime.
**Parent:** improvement-avenues spec C4. **Evidence corrections (v1, verified):** the gate is
**already live in production** (both cod stocks, ramp 3–6 PSU, since July —
`baltic_param-movement.csv:220-233`); **no ensemble generation supplies a citable mean
freshening delta** (Meier 2022 Table 8: BalticAPP −0.06, ECOSUPPORT −0.15, CLIMSEA ≈0 g kg⁻¹
SSS; bottom salinity slightly positive; only 2006-era extremes reached −45 %). **Scoping
decisions (user):** pure **sensitivity arms**, no climate-scenario claim; the projection context
is exactly one cited sentence.
**Related:** `docs/baltic_salinity_gate_percid_mechanism_2026-07-05.md` (the July OFF→ON chain:
cod exclusion+concentration → stickleback +94 % → percids −33/−35 %, pre-calibration 8-species),
`docs/baltic_b2_scenarios_2026-08-29.md` (the wiring discipline this clones),
`docs/baltic_certification_2026-08-14.md` (stickleback envelope position).

## Pre-computed expectations (review, 2026-08-30 — stated up front, not discovered post hoc)

Computed on the real field × the real production maps, engine-oriented (orientation validated
100 % against the grid mask):

* **cod_west's gate is saturated**: mean w = **1.0000** on all three maps, every month — the
  production gate is currently a **no-op for cod_west**, and at ΔS=−1/−2 its metrics stay
  ≈0 (−0.002…−0.021). **cod_west is therefore the experiment's built-in null control**, and
  this is effectively a **cod_east experiment** — stated here, not discovered later.
* **cod_east**: 93–99 % of map cells saturated; mean-Δw −0.016…−0.031 (ΔS=−1),
  −0.115…−0.140 (ΔS=−2); TV of the normalized occupancy distribution 0.028 (−1) / 0.099 (−2);
  newly-excluded cells ≤ **0.23 %**. **The July exclusion mechanism cannot meaningfully fire
  at −1/−2** — those arms characterize the redistribution (pattern) pathway.
* **ΔS=−3 (added at revision)**: the exclusion-regime lever — cells below 6 PSU at baseline
  hit w=0. The builder prints, pre-run, the per-map exclusion fractions AND any all-zero
  (map, frame) events (see gates — the engine's all-zero guard silently reverts a species to
  UNGATED movement; the builder printout turns that from a silent hazard into a visible fact,
  and any arm where it fires is reported under that label, never silently).
* **July comparison framing (review-corrected):** July measured an OFF→ON flip — a ~10×
  larger perturbation than −1/−2 on today's already-gated baseline. The comparison is
  well-posed only as *graded-lever vs flip*: the arms measure the transmission gradient; −3
  approaches the flip's exclusion regime from below.

## Decisions (recorded; 3–4 rewritten, 1 arm added at revision)

1. **Arms: baseline, zero, ΔS=−1, −2, −3 PSU** — uniform additive offsets on
   `baltic_salinity_bottom_climatology.nc`; **wet = grid.nc mask AND finite** (the field's land
   convention is **NaN, not 0.0** — the opposite of the O₂ file; 3 finite off-mask cells are
   excluded by the mask-AND-finite rule); NaN-propagating arithmetic; stored salinity floored
   at ≥ 0; 24 frames asserted in/out; dtype/encoding preserved. 5 arms × 5 house seeds ×
   50 yr (~1.6 h at B2 pace).
2. **No climate-scenario claim** (unchanged). ΔS values are chosen levers: the delta-spec JSON
   gives each a `rationale` field, NOT a citation (review: fake citations for uncitable levers
   would violate the schema's own spirit); the two context numbers (2006-era −45 %, modern ≈0)
   carry real citations.
3. **Instruments (redesigned — the sampler renormalizes, so level metrics are not what the
   engine sees):** per arm, per gated-species map, per frame, computed by the builder pre-run
   with maps obtained **via the engine's own map loader** (import-not-reimplement; the raw
   CSVs are stored upside-down relative to the field — a naive read gives a mirrored ~5×-wrong
   instrument; a CI test pins orientation via zero-map-positive-cells-on-land vs grid mask):
   * (i) **TV distance** between normalized base and arm occupancy distributions (map·w) —
     the redistribution lever the sampler transmits;
   * (ii) **predicted change in normalized cod occupancy mass over each prey species' map
     cells** (stickleback, perch, pikeperch, smelt overlap) — the direct chain lever;
   * (iii) **newly-excluded-cell fraction** (w>0 → w=0) — the July-mechanism lever;
   * (iv) mean-Δw retained as a **wiring check only** (monotone ⇒ ~0 iff nothing changed),
     never printed beside stock responses without the framing sentence: *the gate conserves
     total occupancy — it redistributes and excludes, it never removes fish*.
   Vacuity criterion applies to (i)+(iii); a ≈0 reading is a reportable finding.
4. **Blocking gates (B2 discipline + review additions), in order:** (a) builder zero-check
   (zero-arm field value-identical NaN-aware to production); (b) three-way load-through per
   arm — engine-loaded field (via `_load_salinity_gate`/`EngineConfig.salinity_field`) == disk
   == builder-recomputed offset; (c) ramp ordering per wet cell (negative ΔS ⇒ w′ ≤ w);
   (d) zero-arm run bit-identity per seed (verified achievable: no float transforms on load);
   (e) **frame-count assert (24) on every arm field, harness-side** — the salinity loader has
   NO frame validation (silent `step % frames` wrap; surfaced per success criterion 4,
   recorded as a Stage-2 time-policy item, NOT fixed here); plus the harness asserts
   `movement.salinity.field.constant` is absent from every arm config (the loader prefers
   `.constant` over `.file`; a stray key would silently discard the arms' only lever).
5. **Reported, no pass/fail:** per-arm final-decade means for **all nine species** (review: v1
   omitted smelt — coastal, overlapping the affected cells); the chain reading
   (cod_east redistribution → stickleback → perch/pikeperch/smelt) against the July signature
   under the graded-vs-flip framing; seed spreads printed; instruments (i)–(iii) beside every
   stock column. **Expected signature (stated pre-run):** stickleback UP (a release moves it
   *away* from its floor, toward a ceiling +517 % distant — its certified position is 6.9 % of
   the envelope width above the floor, headroom −38.3 %; v1 misquoted this), percids and smelt
   DOWN; a large percid decline is a sensitivity finding, and for pikeperch it points toward
   realism (its overshoot is the certified bias).
6. **Labels (all restated in the results doc):** not-a-projection; the **RV confound**
   (occupancy pathway only; cod_east recruitment RV-prescribed, gate factor 0.32–0.87);
   single-source climatology (provenance from its attrs: CMEMS PHY, deepest-valid level);
   fixed production ramp 3–6; uniform-offset spatial blindness; **cod_west = saturated null
   control**; the all-zero/un-gate guard status per arm; the Java gap (Java silently ignores
   `movement.salinity.*`; block-reason entry joins the C1 thermal item, both waiting on the
   user-dirty runner.py — recorded, deferred).

## Non-goals (YAGNI)

No ramp retuning; no percid-side gating; no reproduction-side salinity mechanism; **no engine
changes** (both loader gaps — frame-count, all-zero un-gate — are surfaced findings for the
Stage-2 time-policy work, guarded harness-side here); no recalibration; no envelope claims.

## Design

1. **Delta spec** `data/baltic/scenarios/c4_salinity_sensitivity.json`: three ΔS with
   rationales; two context citations; schema test (citations for context numbers, rationale
   for levers, no dead knobs).
2. **Builder** `scripts/build_baltic_c4_forcing.py`: B2-clone offset (mask-AND-finite wet
   rule, NaN-propagating, ≥0 floor, frames/dtype/encoding preserved); instruments (i)–(iv)
   printed per arm incl. the −3 arm's exclusion/all-zero report; zero self-check; maps via the
   engine loader with the orientation CI test.
3. **Harness** `scripts/baltic_c4_salinity_ab.py`: B2 `run_*` pattern; overlays swap
   `movement.salinity.field.file` (absolute paths); gates per decision 4; report to /tmp +
   committed `docs/diagnostics/baltic_c4_salinity_report.json`.
4. **Run + results doc** `docs/baltic_c4_salinity_YYYY-MM-DD.md`: the nine-species × five-arm
   chain table with instruments beside it; the July graded-vs-flip comparison; every label.

## Testing

CI-safe: schema; builder offset/wet-rule/floor/frames/zero-identity + instruments on synthetic
fields (incl. a saturation fixture where all cells sit at w=1 → TV=0, exclusions=0 — the
vacuity case — and an orientation-pinning test on a real map vs the grid mask); harness helpers
(overlay construction, `.constant` assert, frame assert, ramp ordering, three-way equality with
the no-op-write pathological case). NOT CI: the run.

## Success criteria

1. All gates pass; zero arm bit-identical.
2. The chain table exists with instruments (i)–(iii) beside stock responses; the
   graded-vs-flip July comparison stated either way; the −3 arm's exclusion regime
   characterized (or its all-zero guard status reported).
3. Every decision-6 label present; the framing sentence (redistributes, never removes)
   accompanies any occupancy metric.
4. Both loader gaps surfaced in the results doc as Stage-2 items.
