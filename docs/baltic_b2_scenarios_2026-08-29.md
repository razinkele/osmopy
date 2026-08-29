# Baltic B2 — RCP × load literature-delta scenarios (2026-08-29)

**Verdict: ALL GATES PASS.** Identity: zero violations (zero arm bit-identical to
baseline, 5 seeds). Load-through (three-way assert: engine-loaded == on-disk ==
recomputed-from-production): true for all 5 non-baseline arms. Hill ordering:
true for all 5 arms. Knob-factor instrument (sp1/herring, through the loader's
float path): true for every arm. The 2×2 scenario table (RCP4.5/RCP8.5 ×
BSAP/REF) exists below, with predicted-ΔK and noise-floor columns.

Spec: `docs/superpowers/specs/2026-08-29-baltic-b2-literature-delta-scenarios-design.md`
(binding — decisions 1–7, §4, success criteria). Plan:
`docs/superpowers/plans/2026-08-29-baltic-b2-scenarios.md`.

## Run provenance

- 6 arms — `baseline`, `zero` (all machinery engaged, zero deltas), `rcp45_bsap`,
  `rcp45_ref`, `rcp85_bsap`, `rcp85_ref` — × 5 seeds `[42, 123, 7, 999, 2024]`
  (the house set, `simulation.rng.fixed=true`) × 50 yr.
- Branch `b2-scenarios`, harness `scripts/baltic_b2_scenario_ab.py`.
- Runtime: 1h52m (under decision 7's "2.5–3 h" estimate — and notably close to
  the "~2 h" figure decision 7 explicitly rejected as too optimistic; recorded
  as observed fact, not a claim the estimate was wrong).
- **This is a local validation run, not a CI gate** — same status as the C1 A/B
  precedent. Production certification is untouched.
- Raw report: `docs/diagnostics/baltic_b2_scenario_report.json` (copied verbatim
  from the run at `/tmp/b2_scenario_report.json`).
- **Upgrade path:** field-derived deltas replacing the literature numbers below
  is a JSON edit to `data/baltic/scenarios/b2_literature_deltas.json` — no
  builder, harness, or engine change (spec decision 1 rationale; §5).

## Delta spec — citations

All from Meier, H.E.M., et al. (2022), *ESD* 13, 159–199,
doi:10.5194/esd-13-159-2022 (open access; no editorial notices). CLIMSEA
(RCO-SCOBI, four-ESM ensemble mean), **1976–2005 → 2069–2098**:

| arm | RCP | load | ΔT °C (Table 7, annual-mean SST) | ΔO₂ mmol m⁻³ (Table 10, summer-mean bottom O₂, ensemble-mean-SLR) |
|---|---|---|---|---|
| `rcp45_bsap` | RCP4.5 | BSAP | +1.9 | +26.8 (+0.6 mL/L) |
| `rcp45_ref`  | RCP4.5 | REF  | +1.9 | 0.0 (+0.0 mL/L) — **sourced zero, a designed null O₂ contrast, not a degenerate case** |
| `rcp85_bsap` | RCP8.5 | BSAP | +2.9 | +17.9 (+0.4 mL/L) |
| `rcp85_ref`  | RCP8.5 | REF  | +2.9 | −8.9 (−0.2 mL/L) |
| `zero`       | —      | —    | 0.0  | 0.0 (value-identical copy of production; all machinery engaged, zero deltas) |

Conversion: 1 mL/L = 44.66 mmol m⁻³. ΔT applies via the C1 thermal-recruitment
knob (herring/sp1-only, constant-T series at `tref+ΔT`); ΔO₂ applies as a
wet-cells-only additive offset on `baltic_oxygen_bottom.nc`, floored at 0, 24
frames asserted in and out (spec decision 3).

**Caption (spec decision 5, reference-period overstatement — one of the two
mandated placements beyond the JSON and the label list):** every ΔT/ΔO₂ pair
above is Meier et al. (2022)'s end-century delta (2069–2098 vs 1976–2005)
applied raw on a present-day baseline (O₂ file: 2024 CMEMS analysis; knob
`tref`: 1993–2021 mean) — this overstates end-century forcing by whatever
warming/deoxygenation has already been realized between 1976–2005 and today.
The deltas are applied raw and relabelled, not adjusted (decision 5's ruling);
every table below inherits this caption.

## Blocking gates (spec §4, all PASS)

| gate | scope | result |
|---|---|---|
| (a) zero-arm bit-identity to baseline | 5 seeds | **PASS** — 0 violations |
| (b) O₂ load-through (three-way: engine-loaded == on-disk == recomputed-from-production) | all 5 non-baseline arms | **PASS** — true for `zero`, `rcp45_bsap`, `rcp45_ref`, `rcp85_bsap`, `rcp85_ref` |
| (c) deterministic Hill ordering | all 5 non-baseline arms | **PASS** — true for all five |
| (d) knob-factor instrument (loader float path, sp1/herring) | all 5 non-baseline arms | **PASS** — true for all five (`sp1: true` in every arm) |

## The 2×2 results

### Predicted effective-K change per arm (builder: field + Hill, no engine run)

| arm | RCP | load | ΔO₂ (mmol m⁻³) | predicted ΔK |
|---|---|---|---|---|
| `zero` | — | — | 0.0 | 0.00% |
| `rcp45_bsap` | RCP4.5 | BSAP | +26.8 | **+3.42%** |
| `rcp45_ref` | RCP4.5 | REF | 0.0 | 0.00% |
| `rcp85_bsap` | RCP8.5 | BSAP | +17.9 | **+2.04%** |
| `rcp85_ref` | RCP8.5 | REF | −8.9 | **−0.81%** |

### Herring (sp1) — final-decade mean decline vs baseline

Herring is not benthos-fed, so it does not receive the O₂/ΔK pathway above —
its response comes entirely through the C1 thermal knob.

| arm | ΔT | herring final-decade decline vs baseline |
|---|---|---|
| `zero` | 0.0°C | **0.0000%** (exact — zero arm is bit-identical to baseline) |
| `rcp45_bsap` | +1.9°C | **−40.23%** |
| `rcp45_ref` | +1.9°C | **−41.00%** |
| `rcp85_bsap` | +2.9°C | **−62.81%** |
| `rcp85_ref` | +2.9°C | **−62.27%** |

Within each RCP, `bsap` and `ref` are near-identical (−40.23% vs −41.00% at
RCP4.5; −62.81% vs −62.27% at RCP8.5) — consistent with herring's response
running through temperature only, and monotone-worse with ΔT. The two ΔT
values (+1.9, +2.9) interpolate C1's own tested range (0/+2/+4°C). (No
absolute final-decade biomass column is given — the run report carries only
the decline-vs-baseline ratio for herring, not the raw means; by design, not
an omission.)

### Within-RCP load contrast — cod_east and flounder (spec §4 REPORTED: "the only clean ecological read of the O₂ axis")

Per spec §4, per-arm deltas vs baseline are **not** reported for these two
demersal stocks in isolation — the pre-registered, confound-free read is the
BSAP-vs-REF contrast *within* a fixed RCP (temperature held constant, load
varied), printed against the predicted-ΔK and the measured seed-noise floor.

| RCP | cod_east: BSAP vs REF | flounder: BSAP vs REF | predicted ΔK spread (BSAP − REF) | cod_east seed-noise floor |
|---|---|---|---|---|
| RCP4.5 | **+17.74%** | **+15.60%** | +3.42% − 0.00% = 3.42 pp | ±1.9% |
| RCP8.5 | **+17.03%** | **+12.67%** | +2.04% − (−0.81%) = 2.85 pp | ±1.9% |

All four contrasts are roughly an order of magnitude above the recorded
cod_east seed-noise floor (±1.9%, from `docs/baltic_hypoxia_benthos_ab_2026-08-09.md`'s
cod_east across-seed spread at baseline O₂) — this is not noise. It arises
from benthos-K differences of only ~3.4 pp (RCP4.5) / ~2.9 pp (RCP8.5): strong
nonlinear food-web amplification between the O₂→benthos-K coupling and the
demersal stocks that feed on it. Flounder's noise floor was not separately
computed by this harness; cod_east's ±1.9% is used as the reference order of
magnitude for both stocks since they share the same benthos-K coupling
mechanism.

## Reading: the O₂ axis

Meier et al. (2022)'s own caveat, quoted verbatim in the binding spec: the
climate impact on Baltic biogeochemistry is "still smaller than that of
plausible nutrient input changes" — bottom-O₂'s future is nutrient-load
dominated, not climate-dominated, and a climate-only O₂ delta is ill-defined
(the scenario axis has to be RCP × load, which is exactly this run's design).
**And per decision 5's relabel:** every number below is a raw, unadjusted
end-century (2069–2098 vs 1976–2005) delta applied on a present-day baseline —
it overstates end-century forcing by the realized 1976-2005→present component,
which this run does not subtract or estimate (see caption above).

The model **endogenously reproduces the literature's load-dominance finding**:
for the demersal stocks the nutrient-load axis outweighs the RCP axis (17–18%
BSAP-vs-REF within a fixed RCP, vs only a few percent of difference in that
same contrast between RCP4.5 and RCP8.5), while for herring the temperature
axis dominates entirely (the O₂ axis barely touches it — herring is not
benthos-fed). This split was not built into the model; it falls out of the two
independently-wired forcing pathways (C1's thermal knob, the existing O₂→benthos-K
coupling) responding to literature-sourced deltas. **Caveat on the "few
percent" clause:** per spec §4 the harness extracts only the within-RCP
BSAP-vs-REF load contrast for cod_east/flounder (the pre-registered clean
read) — it does not separately measure a demersal-stock response to the RCP
axis alone (i.e., no cod_east/flounder delta-vs-baseline at fixed load). The
"few percent" is the *difference between the two RCPs' load contrasts*
(3.42→2.85 pp swing in predicted ΔK, 17.74%→17.03% / 15.60%→12.67% in the
measured contrasts) — evidence the load effect is stable across RCPs, which is
the same qualitative conclusion (load axis dominates), not a separately
measured RCP-axis magnitude for these two stocks.

## Labels (spec §4(d) list, from the run report's `labels` array, verbatim)

1. "SST-for-bottom-T proxy: annual SST applied to a Q4 bottom-T knob; deep
   warming runs higher than SST in ventilated basins, so the herring decline
   is likely UNDERSTATED."
2. "Summer-only + SLR-variant delta-O2 (Meier2022 Table 10) applied
   year-round."
3. "Uniform-offset spatial blindness + floor asymmetry: the additive O2
   offset is spatially uniform and floor-clipped on the negative side but
   uncapped on the positive side."
4. "LTL-at-baseline: the BSAP cells are a partial-load world -- the load
   cut's O2 benefit enters but its plankton/nutrient pathways do not; the
   omitted pathways plausibly oppose the included one."
5. "Reference-period overstatement (spec decision 5): literature deltas are
   end-century vs 1976-2005 but applied raw on a present-day baseline (O2:
   2024 analysis; tref: 1993-2021 mean) -- overstates end-century forcing by
   the realized 1976-2005->present component."
6. "cod_east's trajectory is partly prescribed by the RV narrative series
   (gate factor 0.32-0.87 across the scored decade) -- its scenario deltas
   are conditioned on that prescription."

## What this is not

No envelope pass/fail is claimed for any scenario arm (spec decision 6) — the
arms legitimately exit calibration envelopes and are not compared against
ICES targets here. No recalibration, no new engine mechanisms (this run
exercises the already-shipped C1 thermal knob and O₂→benthos-K coupling
exclusively), no cod-recruitment temperature response (C1's verdicts on
cod_west/cod_east stand unchanged), and no claim that any arm represents a
calibrated end-century state — decision 5's relabel above is the honest
framing for what these numbers mean.

## Deliverables

- Delta spec: `data/baltic/scenarios/b2_literature_deltas.json` (all numbers
  cited; commit `b4abd04`).
- Builder: `scripts/build_baltic_b2_forcing.py` (commit `f07c892`).
- Harness: `scripts/baltic_b2_scenario_ab.py` + tests (commits `60de419`,
  `960d077`, `a2cd5bd`).
- This results doc + copied report: `docs/baltic_b2_scenarios_2026-08-29.md`,
  `docs/diagnostics/baltic_b2_scenario_report.json` (this task).
