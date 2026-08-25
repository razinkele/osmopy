# Baltic C1 — temperature-dependent recruitment as a scenario knob (Voss & Quaas form)

**Date:** 2026-08-25
**Status:** approved (design), pending adversarial review, then implementation plan.
**Parent:** `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` scenario track
(C1). **Scoping decision (user, 2026-08-25):** C1 is a *scenario knob* — a labelled encoding of
the Voss & Quaas aggregate productivity finding for scenario runs — not a validated mechanism and
not a hindcast device. This follows the F1 null
(`docs/baltic_f_hindcast_2026-08-23.md`): this model's scenario value is capability, and B1
Stages 2–3 are capability-motivated; C1 is the scenario-track item that needs no interannual
engine work.
**Related:** `docs/baltic_thermal_recruitment_shape_2026-08-10.md` (the three blocks this design
routes around), `docs/baltic_recruitment_pathway_2026-08-10.md` (the 7-stage recruitment chain),
`docs/baltic_herring_phenology_a0_2026-08-12.md` (the phenology pathway this design deliberately
does NOT take).

## The anchor citation, verified (2026-08-25, scite + full text)

Voss, R. & Quaas, M. F. (2026). *Future fishing potential of cod and herring under climate change
in the Western Baltic Sea.* ICES JMS 83(4), doi:10.1093/icesjms/fsag033 — real, gold OA (CC-BY),
no editorial notices.

* **Functional form (Methods, quoted):** `ln(R) = ln(e^(−β₀+β₁·T)·SSB/(1+β₃·SSB)) + ε` — a
  Beverton–Holt whose productivity numerator carries **exp(β₁·T)**. In this engine's terms
  (`docs/baltic_recruitment_pathway_2026-08-10.md`: all environmental gates are per-year scalar
  multipliers), that is exactly a per-year multiplicative factor on recruitment.
* **Herring:** β₁ = **−0.51 /°C**, driver = **bottom temperature, Q4** (best of 8 tested
  configurations, adj. R² 0.817). Stock: **her.27.20-24** (Western Baltic spring spawners).
* **Cod:** driver = **SST, Q3**; the coefficient is in Conradt (2023), *Management of Atlantic cod
  (Gadus morhua L.) under future climate change*, Dissertation, Univ. Hamburg — **not printed in
  the paper**. Stock: **cod.27.22-24** = this model's **cod_west**.
* Temperature source in the paper: BSIO model reconstructions; scenarios RCP4.5/8.5.

**Two corrections this verification forced on the original framing:** the paper's stocks are
*Western* Baltic — so the knob's best-supported targets are **cod_west** (not the RV-gated
cod_east, which dissolves the gate-conflation objection of the shape doc) and the **western
member of the herring complex**; and the right response shape is the paper's own **exponential
factor**, not an invented "decreasing logistic".

## The three blocks this design routes around (established 2026-08-10/12)

1. The existing thermal gate's logistic **rises** with temperature (percid provenance) — wrong
   sign for Voss & Quaas at any parameterisation.
2. Herring's literature mechanism is **phenological** (Polte: 3.5–4.5 °C onset trigger), and the
   A0 experiment measured the model's response to spawning-time shifts **opposite in sign** to
   the Polte prediction — so a temperature→phenology coupling would project effects the model
   demonstrably contradicts. This design encodes the **aggregate productivity** effect instead
   (a scalar factor), which A0 did not test and does not contradict.
3. Eastern cod's drivers (RV hydrography, egg predation) are already represented and the RV gate
   dominates — so **cod_east gets no thermal knob**. cod_west carries no RV gate; no conflation.

## Decisions (recorded)

1. **Scope: scenario knob for cod_west (sp0) and herring (sp1)** (user, 2026-08-25). Production
   `data/baltic/` stays byte-identical; certification stays climatological; the knob lives in
   scenario overlays only.
2. **Response shape: the paper's exponential factor**, added to the existing thermal gate:
   `factor(y) = exp(beta · (T(y) − T_ref))`, per-species `beta` and `T_ref`. Factor = 1 at
   `T_ref`; cold years legitimately exceed 1 (the paper's form has no cap).
3. **Herring β = −0.51 /°C** (quoted from the paper). Applying the western-stock coefficient to
   the model's 4-stock herring complex is an **approximation, labelled** in config comments and
   any results doc (the complex's other members are central/northern stocks the paper does not
   cover).
4. **cod_west β: fit ourselves, pre-registered.** Fit the paper's exact form (BH with exp(β₁T)
   numerator, log scale) to the cached ICES cod.27.22-24 recruitment & SSB series
   (`data/baltic/reference/ices_snapshots/`) against our derived SST-Q3 series. Cross-check the
   paper's supplement if it publishes Conradt's value. **Pre-registered fallback:** if the fitted
   β₁ is not negative with p < 0.1, the cod_west arm ships **disabled** and the finding is
   documented — the knob is then herring-only. No sign-forcing, no tuning.
5. **Drivers mirror the paper:** cod_west ← SST (surface `thetao`) Q3 mean over SD22–24;
   herring ← bottom temperature (`bottomT`) Q4 mean over the western Baltic (SD22–24), both from
   the CMEMS Baltic PHY multi-year reanalysis (`cmems_mod_bal_phy_my_P1M-m`, carries both
   variables). `T_ref` = each series' 1993–2023 climatological mean. (BSIO is not available to
   us; CMEMS reanalysis is the same class of product — labelled substitution.)
6. **Calendar: F1's exact layout.** The thermal series CSV carries 50 rows per species column:
   19 spin-up rows at `T_ref` (factor 1), then 1993–2023 — sim-year 19 = 1993, identical to the
   F1 by-year F convention. Scenario arms apply a uniform +ΔT offset to the historical block
   only (spin-up stays at `T_ref`).

## Non-goals (YAGNI)

* No phenology/seasonality coupling (block 2). No cod_east knob (block 3). No new gate — one new
  response shape in the existing `thermal_gate.py`.
* No hindcast-skill claim, ever, for this knob (F1 doctrine). Validation is behavioral
  (§Validation).
* No RCP series in this stage: the knob consumes the same per-year CSV format regardless; future
  scenario series arrive via B2 (ERGOM/RCP swap) through the same interface.
* No recalibration of anything against the knob's arms.

## Design

### 1. Engine: `exponential` response in the thermal gate

* `reproduction.thermal.gate.response` ∈ {`logistic` (default, unchanged), `exponential`}.
* New per-species keys: `reproduction.thermal.gate.beta.sp{N}`,
  `reproduction.thermal.gate.tref.sp{N}` (required when `response=exponential` and the species is
  enabled).
* **Mode interaction:** with `response=exponential`, the gate's normalisation mode must be the new
  `raw` (factor applied as computed, floored at `reproduction.thermal.gate.floor`, default 0);
  the loader **rejects** `thermal_cap` and `mean_preserving` under `exponential` — `T_ref`
  anchoring already provides the normalisation, and a cap would truncate the cold-side response
  the paper's form does not truncate.
* **Negative-offset guard (small hardening, both year-indexed gates):** `_load_rv_gate` and
  `_load_thermal_gate` compute `offset = start_year − first_year`; a negative offset feeds
  `idx = min(offset + year, n−1)` with a negative index — Python then silently indexes from the
  END of the series (a latent bug found in the B1 audit). Add `offset >= 0` validation to both
  loaders (raise at load). C1's own layout never needs a negative offset; the guard prevents the
  silent-wrap class.
* Schema fields + allowlist entries for the three new keys, applying the F1 lesson: check
  `tests/test_schema_engine_key_parity.py` and the frozen snapshot in
  `tests/test_issue_123_known_but_unread_keys.py` in the same change.

### 2. Data: `scripts/build_baltic_thermal_sr_series.py`

Follows the `build_percid_thermal_series.py` precedent. Downloads (or reads from
`data/cmems_cache/`) monthly `thetao` (surface) and `bottomT` from the Baltic PHY reanalysis,
1993–2023, over the SD22–24 bbox; computes per-year cod_west SST-Q3 and herring bottom-T-Q4
means; writes `data/baltic/forcing/baltic_thermal_sr_series.csv` with columns
`year,temp_sp0,temp_sp1` in the decision-6 layout (50 rows, 19 spin-up at `T_ref`), plus `#`
provenance headers (product IDs, bbox, months, `T_ref` values, generation date). The `T_ref`
values are also what the overlay sets `reproduction.thermal.gate.tref.sp{N}` to.

### 3. Fit: `scripts/fit_codwest_thermal_sr.py`

Offline, reproducible: loads cod.27.22-24 `recruitment`/`ssb` from the cached snapshots and the
SST-Q3 series; fits `ln(R) = −β₀ + β₁·T + ln(SSB) − ln(1+β₃·SSB)` (the paper's form, log scale;
`scipy.optimize`); reports β₁ with CI and p-value, writes a short dated results doc with the
fit diagnostics and the decision-4 verdict (enable cod_west or ship it disabled). The overlap
window is bounded by the snapshots (R/SSB to ~2023) and the derived series (1993–2023).

### 4. Validation A/B — behavioral, pre-registered

Arms (5 house seeds × 50 yr, Python engine, `scripts/baltic_c1_knob_ab.py` or the
`baltic_depletable_ab.py --extra-arm` harness):

* **off** — production config (identity arm).
* **knob+0** — knob enabled (herring; cod_west iff decision-4 fit passes), historical series,
  no offset. **Regression check (blocking):** the identity gate ({5 assessed}+perch+stickleback)
  must PASS, and no species' final-decade mean may leave the off-arm's seed envelope by more
  than the harness' standard tolerance — at T ≈ T_ref the knob must be a near-no-op
  (mean factor ≈ exp(β²σ_T²/2), computed and reported, expected within ~1–2% of 1).
* **knob+2, knob+4** — uniform +2 °C / +4 °C on the historical block.

**Pass criteria (pre-registered):** (a) knob+0 blocking check as above; (b) final-decade mean of
each enabled species declines **monotonically** across +0 → +2 → +4; (c) the recruitment-level
suppression in the arms matches exp(β·ΔT) by construction (assert the realized mean gate factor
per arm — the instrument check, F1's lesson institutionalized); the **biomass** elasticity is
reported against it with no threshold (density dependence and trophic feedback legitimately damp
it). A non-monotone biomass response is a FAIL and a finding, not a tuning invitation.

### 5. Deliverables

Engine change + tests; the two scripts; the series CSV + fit results doc; the A/B results doc
(dated, honest verdict); a scenario-overlay JSON (`data/baltic/calibration_results/` convention)
that turns the knob on — the artifact B2 scenario work later points at with future series.

## Testing

* CI-safe: exponential response math (factor=1 at T_ref, exp(β·ΔT) scaling, floor); loader
  rejection of cap/mean_preserving under `exponential` and of missing beta/tref; negative-offset
  guard on both gates (raises; zero/positive offsets pass); schema/allowlist parity + frozen
  snapshot; builder Q3/Q4 month selection and spin-up layout on synthetic fixtures; fit script
  recovers a known β from synthetic data.
* NOT CI: the A/B arms (emergent, local, documented).

## Success criteria

1. Exponential response + guards land with tests; production certification unchanged
   (climatological, knob absent — byte-identical config).
2. Series + `T_ref`s derived reproducibly with provenance; fit verdict documented either way.
3. knob+0 blocking regression passes; +2/+4 arms produce monotone declines with the realized
   gate factors matching exp(β·ΔT); results doc states the labelled approximations (herring
   complex, CMEMS-for-BSIO) and the scenario-only epistemic status.
4. The B2 interface is explicit: swapping the series file is the entire scenario hookup.
