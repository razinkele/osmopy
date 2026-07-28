# Percid missing-removals + cod stability (formerly "fishing-forced cod") — design

**Date:** 2026-07-28 (revised post-review 2026-07-28)
**Status:** design — REVISED after a 3-reviewer in-loop review (science, config-integrity,
feasibility). The revision corrects a premise error and re-centres the effort; the original
"fishing-forced cod + top-down control" framing is superseded (see Revision note).
**Base:** the aggregate 8-species baseline at commit `646a36d`.
**Motivation docs:** `docs/baltic_findings_summary_2026-07-28.docx`,
`docs/baltic_cod_ew_phase1_report_2026-07-25.md`.

## Revision note — what the review changed

The original design assumed cod sits too high in the baseline and releases the prey field, so
it proposed forcing cod down and adding seal predation to hold the forage fish. **The reviews
showed this premise is false:**

- The baseline's **equilibrium cod is ~64 kt (RV-gate on) / ~167 kt (off) — in-envelope**
  (envelope 60–250 kt), NOT the ~150 kt "apex predator" figure, which is cod's *seeding*
  biomass (`baltic-fish-lifecycle.md:204`), not equilibrium. The `646a36d` base has the RV gate
  on → cod ~64 kt, at the low edge (`baltic_stability_certification_2026-07-01.md`:
  cod ✓ in-envelope at 61–68 kt).
- At that cod level the **forage prey are already in-envelope**: herring ✓, sprat ✓, flounder ✓.
- So forcing cod lower is a **no-op or harmful** (it would push cod, which already fails to
  persist at min 2.4 kt, into its crash and *manufacture* the prey-release it was meant to
  prevent). Strengthening seals to hold forage fish targets a **gap that does not exist**.

The real, verified baseline gaps are the **percids and smelt** (pikeperch ~90×, perch ~2×,
smelt ~5× over envelope) and **cod non-persistence** (dips to 2.4 kt). The genuinely valuable,
citable idea — the **under-reported percid removals** (recreational/coastal fishing +
seal/cormorant predation ~2× the fishery) — survives and becomes the centre of this design.
The cod work is re-scoped from "lower cod's level" to "optionally stabilise cod."

## 1. Goal

On the aggregate 8-species baseline:
- **Primary:** reduce the percid (perch, pikeperch) and smelt overshoot using
  *scientifically-grounded* removals the model currently omits — realistic percid fishing
  mortality (recreational + small-scale coastal, under-reported to ICES) and cormorant
  predation (~2× the fishery for perch; Hansson et al. 2018).
- **Secondary (optional):** improve cod *persistence* (currently min 2.4 kt) — NOT its level.

Honest framing up front (from the review): **perch (~2×) is plausibly closable** with grounded
removals; **pikeperch (~90×) is very likely NOT** — cormorants reach only juvenile pikeperch
(size window 8.75–34 cm; pikeperch matures ~40 cm), and a ~90× overshoot signals missing
*habitat density regulation* (coarse grid) that additive mortality cannot supply without
destabilising. This design tests how far grounded removals move the percids; it does not
promise envelope-membership for pikeperch.

## 2. Base configuration and the mandatory pre-flight

- **Branch** off `646a36d` (aggregate 8-species, RV gate on). `master` keeps the disaggregation
  experiment.
- **Restore + verify FIRST (blocking):** apply `phase13_equilibrium.json`, re-run
  `baltic_stability_certify.py`, and confirm the 5/8 / obj-2.33 / cod ~64 kt state before
  touching anything. `phase13_equilibrium.json` is a 39-param, 8-species-era artifact — confirm
  it applies cleanly on the 8-species `get_phase13_shepherd_params` (NOT master's 45-param
  9-species version). Without a verified clean baseline, later regressions are unattributable.

## 3. Engine feasibility (verified in the review; corrections folded in)

- **Background predators DO predate focal fish**, scaled by prescribed biomass ×
  `predation.ingestion.rate.max`, size-ratio gated; absent from the accessibility matrix they
  predate at the **default coefficient 1.0** (`predation.py:211,222`). Adding a matrix column
  lets that coefficient be tuned per prey — this is **necessary, not optional**, to raise
  cormorant predation on perch without over-cropping herring/sprat.
- **Predator indices on the `646a36d` base are sp14 = GreySeal, sp15 = Cormorant** (there is no
  sp16). The sp15/sp16 in the original draft were the *disaggregated-master* indices — using
  them would silently mis-target. All keys below use sp14/sp15.
- **`species.biomass.multiplier.sp{i}`** scales NetCDF standing biomass (`background.py:366`);
  **`predation.ingestion.rate.max.sp{i}`** caps realized predation (`predation.py:189`). Both
  are read for background species and are in the validation allowlist (AST walk of
  `background.py`).
- **Cod-F forcing loader is `_load_fishing_rate_by_year` → `np.loadtxt(path).flatten()`**
  (`config.py:455,466`), NOT the dead `ByYearTimeSeries` class. The forcing file must be a
  **headerless single column of F values, one row per simulation year, padded to the full run
  horizon** (a header row crashes; a 2-column file silently misreads; beyond the file's span F
  falls back to the scalar base rate — `fishing.py:43` — so the final decade would silently
  revert). `mortality.fishing.rate.byYear.file.sp{idx}` is **flagged UNKNOWN by config-validation**
  (variable key_pattern escapes the AST walk) — add it to `_ALLOWLIST_PY_HONORED`
  (`config_validation.py:47`) or the config is rejected under strict mode.

## 4. Step 0 — mandatory cheap feasibility gate (before any calibration)

The disaggregation burned two multi-hour DE runs to discover a structural wall; its one
decisive cheap check (a hand-built forward sim scoring 1817) was run *after* the fact. **Do not
repeat that.** Before the re-calibration:

1. On the restored baseline, crank the grounded levers to their **maximum defensible** values by
   hand — percid F to the top of the real coastal range (~0.6), cormorant biomass-multiplier +
   ingestion to their count/physiology anchors (Section 5), cormorant matrix column shaping
   predation onto perch/young pikeperch.
2. Run **one** forward sim (40–50 yr) and read: (a) how far perch, pikeperch and smelt move;
   (b) whether the high-weight prey (herring/sprat/flounder) and cod stay in-envelope; (c) the
   realized cormorant/seal consumption vs the Hansson budget.
3. **Go/no-go:** if even maxed grounded levers barely move pikeperch (expected) but *do* move
   perch/smelt without collateral damage → proceed, scoped to perch/smelt. If they move nothing
   or destabilise the high-weight stocks → the re-calibration is futile; record the finding and
   stop. This 15-minute check gates the 4–8 h run.

## 5. Components

### 5.1 Percid missing-removals (PRIMARY)

**Lever A — realistic percid fishing F (fixed).** Set perch (fsh4) and pikeperch (fsh5)
`fisheries.rate.base` to **absolute** literature-grounded values for total (commercial +
recreational) coastal removal — real exploited Baltic coastal percid F is **~0.3–0.6**, often
F > M — NOT a multiple of the model's calibrated artifact (0.0095 → 2× = 0.019 is negligible).
Fixed, not a free param, so it is not optimised away. Value + provenance set in the plan from
coastal-fishery statistics (Curonian/Baltic). Check total Z = F_commercial + F_recreational +
M_predation stays realistic so perch does not over-crash.

**Lever B — cormorant predation (predation side).** `species.biomass.multiplier.sp15` (cormorant)
and `predation.ingestion.rate.max.sp15` as free params, **each bounded to its own realism
anchor**: biomass to count-based standing stock; ingestion toward the physiological rate
(~70/yr for a 2 kg bird eating ~400–500 g/day — the current 40/yr is low; prefer raising
ingestion over inflating bird counts). Add a **cormorant column to `predation-accessibility.csv`**
to shape the perch/pikeperch fraction (necessary — otherwise coefficient 1.0 over-crops
herring/sprat). Honest: this reaches adult perch (window covers <34 cm) but only *juvenile*
pikeperch — a recruitment-side lever on pikeperch, not an adult-biomass one.

### 5.2 Cod stability (SECONDARY, optional)

Cod's issue is persistence (min 2.4 kt), not level. IF pursued: force a credible F-driven cod
trajectory (ICES cod.27.24-32 F via `mortality.fishing.rate.byYear.file.sp0`, headerless,
horizon-padded) and **pin or tightly bound cod's additional M** (do not leave it free — else M
+ seal predation are confounded and "calibrated M" is uninterpretable). Document that this
targets *eastern* SSB and pins the aggregate to it (a known bias), and that in the scored final
decade F≈0.015 so the depressed level is M/predation-governed, not F-governed. **Remove cod's
`fisheries.rate.base.fsh0` from the free-param set** (byYear overrides it) and set it to the
hold-F. If the Step-0 gate shows cod is fine as-is, **skip this component.**

### 5.3 Seal on forage fish — DROPPED

Forage prey are already in-envelope, so there is no gap to fill; strengthening seals only risks
over-cropping and confounds cod M (seal size window reaches cod spawners). Not included. (Seal
biomass may still be corrected to its ~2× real-standing-stock anchor for realism, but as a fixed
value, not a calibration lever for a non-existent target.)

### 5.4 Re-calibration (scoped)

Full 8-species DE **only if** the Step-0 gate passes, warm-started from `phase13_equilibrium.json`.
Changes vs baseline: cod fsh0 removed (or fixed); percid F fixed (Lever A); cormorant
biomass-multiplier + ingestion added as free params with **explicit x0 = the Step-0 max-grounded
values**; optional cod-F forcing + pinned cod M. Required plumbing:
- `apply_calibration.py` `_FILE_FOR` extended so `species.biomass.multiplier.` and
  `predation.ingestion.rate.max.` route to `baltic_param-background.csv` (else KeyError).
- Confirm whether percid **catch targets** are in the objective; a fixed total-removal F will
  miss a commercial-only catch target — exclude/adjust consistently (this is the tension that
  neutered prey-fishing in the disaggregation; predation, unlike fishing, does NOT fight a catch
  target — the design's one genuine structural advantage over the disaggregation).
- State run-horizon vs forcing-horizon and check the scored final decade is quasi-stationary
  (the disaggregation was bitten by an RV-series wrap mismatch between calibration and cert).

## 6. Validation, acceptance bar, and revert rule

- **Pre-registered magnitude bar:** perch overshoot reduced to ≤ (target upper × 2) or better;
  smelt toward envelope; pikeperch — record the reduction achieved, expected small. "Improved
  toward" alone is insufficient — state the numeric threshold before running.
- **No-regression (hard):** every high-weight prey (herring, sprat, flounder) stays in its
  baseline envelope; cod persistence no worse. Note this is *nearly guaranteed by construction*
  (warm-start + levers tunable to zero effect), so it does not by itself prove the mechanism
  works — also predict the *sign and magnitude* of cormorant consumption and the percid response
  and check them against the Hansson budget as an output test.
- **Insensitivity test for the structural claim:** before attributing a residual pikeperch
  overshoot to the coarse grid, push grounded removals to the top of the defensible range and
  show the residual persists — otherwise "structural" is unproven.
- **Pre-registered revert rule:** if any high-weight prey drops below its baseline envelope, or
  the objective exceeds the baseline's 2.33 by more than a set margin, revert to the baseline and
  record the finding as the structural limit (as the disaggregation did).

## 7. Risks (post-review)

- **Milder apex-release still possible** if cod is depressed at all (Component 5.2) — the Step-0
  gate + pinned cod M + the dropped seal component mitigate this; a per-species predation budget
  (baseline-cod vs depressed-cod cropping of each prey) should be computed before enabling 5.2.
- **Percid mortality may destabilise** (prior 8-lever finding) — grounded magnitudes + the
  Step-0 gate + the revert rule bound this; accept residual overshoot over over-cranking.
- **Pikeperch ~90× likely not closable** — stated as the binding constraint, not hidden.
- **SSB-vs-total-biomass** — state which metric the bar uses (cod target is ICES SSB; the harness
  scores total biomass).

## 8. Config-binding checklist (from the config review — must hold in the plan)

1. Indices sp14 = seal, sp15 = cormorant on `646a36d` (NOT sp15/sp16).
2. Cod-F forcing file: headerless single column, one row per sim year, padded to horizon.
3. `apply_calibration._FILE_FOR`: add `species.biomass.multiplier.` and
   `predation.ingestion.rate.max.` → `baltic_param-background.csv`.
4. Allowlist `mortality.fishing.rate.byYear.file.sp{idx}` in `config_validation.py`.
5. Remove/fix cod `fisheries.rate.base.fsh0` when F is forced.
6. Add the cormorant column to `predation-accessibility.csv`.

## 9. Out of scope / future

Finer coastal grid (the proper pikeperch fix); herring/flounder disaggregation; multi-year
seal/cormorant biomass trajectories (needs an engine change to the seasonal-wrap indexing).

## 10. References

`docs/baltic_findings_summary_2026-07-28.docx` §6 (full, verification-tagged). Key: Hansson et
al. (2018) *ICES JMS* 75(3):999; Baltic pikeperch status reviews (recreational ≥ commercial);
Heikinheimo et al. (2021), Östman et al. (2013) (cormorant predation on perch); ICES
cod.27.24-32.
