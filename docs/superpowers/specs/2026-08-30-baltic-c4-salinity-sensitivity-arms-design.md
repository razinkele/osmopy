# Baltic C4 — salinity-gate sensitivity arms (mechanism characterization, not projection)

**Date:** 2026-08-30
**Status:** approved (design), pending adversarial review, then implementation plan.
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` scenario
track (C4, "salinity-gated occupancy/movement — the freshening signal is a first-order Baltic
scenario driver"). **Evidence corrections that rescoped C4 (verified live, 2026-08-30):**
1. **The mechanism is already shipped.** The salinity occupancy gate
   (`osmose/engine/processes/salinity_gate.py`, ramp `w(S)=clip((S−s_low)/(s_high−s_low),0,1)`
   multiplying the movement map) has been **live in production since July** —
   `data/baltic/baltic_param-movement.csv:220-233`: enabled for **cod_west (sp0) and cod_east
   (sp8)**, `s_low=3.0`, `s_high=6.0`, field `baltic_salinity_bottom_climatology.nc`. It
   survived the E/W split and is inside every certification and every F1/C1/B2 baseline.
   "Revive the prototype" is therefore already done; C4 reduces to the scenario ambition.
2. **The freshening scenario driver does not survive verification.** Meier et al. 2022
   (doi:10.5194/esd-13-159-2022, Table 8 + §3.2.4, fetched and quoted): ensemble-MEAN salinity
   changes are near-zero in every generation — BalticAPP SSS −0.06/−0.07 g kg⁻¹ (RCP4.5/8.5),
   ECOSUPPORT −0.15, CLIMSEA +0.01/−0.01, with **bottom salinity slightly positive** in the
   older ensembles; CLIMSEA: "salinity changes are not robust; i.e. the ensemble spread is
   larger than the signal" (runoff increase "approximately compensated by the impact of larger
   saltwater inflows due to the projected SLR"). Only first-generation extremes (Meier 2006 via
   the BACC citation) reached "decreases of as much as 45 %". **No ensemble generation supplies
   a citable mean freshening delta** — an earlier draft's "bracket the older projections"
   framing was corrected mid-design when this was discovered (user re-decision below).
**Scoping decisions (user, 2026-08-30):** bracketing-arms premise corrected →
**pure sensitivity arms**: ΔS = −1 and −2 PSU as *mechanism characterization* of the production
gate, with no climate-scenario claim anywhere; the projection context (2006-era extremes,
modern near-zero consensus) appears as exactly one cited sentence.
**Related:** `docs/baltic_salinity_gradient_exploration_2026-07-24.md` (gate status + effect at
1 seed on the old config), `docs/baltic_salinity_gate_percid_mechanism_2026-07-05.md` (the
regime-shift chain: cod exclusion → stickleback +94 % → percids −33/−35 %, on the
pre-calibration 8-species config), `docs/baltic_b2_scenarios_2026-08-29.md` (the five-gate
wiring discipline this clones), `docs/baltic_certification_2026-08-14.md` (stickleback sits
6.9 % above its floor — the floor-risk context for any release response),
`docs/baltic_herring_stickleback...` via memory: stickleback is ALSO clamped by herring
egg/YOY predation (2026-08-14 finding) — the July release may be damped on today's config.

## The headline question

**Does the July coastal regime-shift chain reproduce on the certified 9-species config, and at
what gain?** July measured it pre-calibration, 8 species, 1 seed. Today's config differs in
every relevant way: E/W cod split (both gated), recalibrated food web, and the
herring–stickleback clamp. The arms quantify the transmission: salinity field → ramp occupancy
→ cod distribution → stickleback release → percid response.

## Decisions (recorded)

1. **Arms: baseline, zero, ΔS=−1, ΔS=−2 PSU** (uniform additive offsets on the bottom-salinity
   climatology, wet cells only, floored at 0; 24 frames asserted in/out). 4 arms × 5 house
   seeds × 50 yr (~1.3 h at B2's measured pace).
2. **No climate-scenario claim.** The results doc's framing sentence: "these are mechanism-
   characterization arms; every ensemble generation's mean salinity change is ≈0 (Meier et al.
   2022 Table 8), and only first-generation extremes reached −45 % (Meier 2006, cited
   therein)." Nothing in the doc may call the arms RCP-anything.
3. **Deterministic instrument — predicted ramp-occupancy change** (the predicted-ΔK analog):
   per arm, compute `w(S+ΔS)` vs `w(S)` over wet cells with the production ramp (3–6 PSU) and
   report the change weighted by the gated species' movement-map cells (cod_west and cod_east
   separately). Printed by the builder pre-run; the results doc prints it beside the stock
   responses. If the predicted occupancy change is ~0 (few reachable cells within ramp reach),
   that is a REPORTABLE vacuity finding, not a reason to enlarge ΔS post hoc.
4. **The five-gate wiring discipline, transplanted from B2** (all BLOCKING, in order):
   (a) builder zero-check — zero-arm written field value-identical (NaN-aware) to production;
   (b) three-way load-through per arm — engine-loaded salinity field == disk == offset
   recomputed via the builder's own offset function (import, not reimplement);
   (c) ramp ordering per arm — `w(S_arm) ≤ w(S_base)` per wet cell for negative ΔS (ramp
   monotone ⇒ guaranteed; violation = wiring);
   (d) zero-arm run bit-identity to baseline per seed;
   (no knob instrument — no thermal machinery is touched; arms differ only in
   `movement.salinity.field.file`).
5. **Reported, no pass/fail** (sensitivity arms make no envelope claims): per-arm final-decade
   means for cod_west, cod_east, stickleback, perch, pikeperch, herring, sprat, flounder;
   the chain reading (cod change → stickleback change → percid change) vs the July signature;
   stickleback's floor distance restated (floor-risk context); seed spread printed.
6. **Labels (every one restated in the results doc):** (i) not-a-projection (decision 2);
   (ii) **the RV confound, prominently**: real freshening would act on cod primarily through
   REPRODUCTION (the RV mechanism), but cod_east's recruitment is prescribed by the RV
   narrative series — these arms exercise the OCCUPANCY pathway only, and cod_east's response
   is conditioned on that prescription (gate factor 0.32–0.87 over the scored decade);
   (iii) the salinity field is a single-source climatology (provenance restated from its
   attrs); (iv) ramp bounds fixed at the production 3–6 PSU — the arms move the field, not the
   ramp; (v) uniform-offset spatial blindness (real freshening is spatially structured).

## Non-goals (YAGNI)

No ramp retuning; no percid-side (inverted-ramp) gating; no reproduction-side salinity
mechanism; no engine changes (the gate is shipped; if a loader gap is found, it is a finding
to surface, not to silently patch); no new scenario JSON schema (a tiny C4-local delta spec
mirroring B2's, minus the O₂ referent machinery); no recalibration; no envelope claims.

## Design

1. **Delta spec** `data/baltic/scenarios/c4_salinity_sensitivity.json`: two ΔS values with the
   decision-2 context citations; schema-validated (citations mandatory, no dead knobs).
2. **Builder** `scripts/build_baltic_c4_forcing.py`: clones B2's wet-mask/floor/frames pattern
   for the salinity NetCDF (variable per `movement.salinity.field.varname` = `salinity`);
   emits per-arm fields + the decision-3 predicted occupancy changes; zero self-check.
3. **Harness** `scripts/baltic_c4_salinity_ab.py`: B2's `run_*` pattern; overlays swap
   `movement.salinity.field.file` to the arm's absolute path; gates per decision 4; report
   JSON to /tmp + committed copy at `docs/diagnostics/baltic_c4_salinity_report.json`.
4. **Run + results doc** `docs/baltic_c4_salinity_YYYY-MM-DD.md`: the chain table
   (per-species, per-arm, deltas vs baseline), predicted occupancy changes beside them, the
   July-signature comparison, all labels.

## Testing

CI-safe: delta-spec schema; builder offset/floor/frames/zero-identity + predicted-occupancy
math on synthetic fields (incl. a vacuity fixture where no cell is within ramp reach → 0.0);
harness helpers (overlay construction; ramp-ordering check on synthetic fields; three-way
equality incl. the no-op-write pathological case — B2's proven test set). NOT CI: the run.

## Success criteria

1. All four blocking gates pass; zero arm bit-identical.
2. The chain table exists with predicted occupancy changes; the July-signature comparison is
   stated either way (reproduced / damped / absent — all are findings).
3. Every decision-6 label present; decision-2 framing sentence present; no scenario language.
4. Anything the arms reveal about the gate's loader (e.g. silent fallback analogs) is surfaced
   as a finding.
