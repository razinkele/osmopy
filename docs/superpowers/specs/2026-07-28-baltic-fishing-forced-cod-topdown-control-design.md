# Baltic percid missing-removals — design

**Date:** 2026-07-28 (revised post-review; scoped to percid removals 2026-07-28)
**Status:** design — ready for planning.
**Base:** the aggregate 8-species baseline at commit `646a36d`.
**Motivation docs:** `docs/baltic_findings_summary_2026-07-28.docx`,
`docs/baltic_cod_ew_phase1_report_2026-07-25.md`.

## Scope note

This design was originally "fishing-forced cod + top-down control." A three-reviewer in-loop
review (science, config-integrity, feasibility) showed the cod premise was false — baseline cod
is already ~64 kt (in-envelope; the ~150 kt figure is *seeding*, not equilibrium), and the
forage prey (herring/sprat/flounder) are already in-envelope — so forcing cod down and adding
seals to hold forage fish target gaps that do not exist. Per that finding, the design is now
**scoped purely to the percid/smelt missing-removals** — the one component that addresses a real,
verified baseline gap. Cod-forcing, cod-M pinning, and the seal lever are **out of scope**.

## 1. Goal

On the aggregate 8-species baseline, reduce the **percid (perch, pikeperch) and smelt overshoot**
using *scientifically-grounded* removals the model currently omits:
- realistic **percid fishing mortality** (recreational + small-scale coastal, under-reported to
  ICES and not analytically assessed);
- **cormorant predation** on perch and young pikeperch (~2× the fishery for perch; Hansson et al.
  2018).

**Baseline gaps (verified, `baltic_stability_certification_2026-07-01.md`):** perch ~2×,
**pikeperch ~90×**, smelt ~5× over their ICES envelopes; the well-assessed stocks (cod, herring,
sprat, flounder, stickleback) are in-envelope and must stay so.

**Honest framing (from the review):** perch (~2×) is plausibly closable with grounded removals;
**pikeperch (~90×) very likely is NOT** — cormorants reach only *juvenile* pikeperch (size window
8.75–34 cm; pikeperch matures ~40 cm), and a ~90× overshoot signals missing *habitat density
regulation* (coarse grid) that additive mortality cannot supply without destabilising. This
design tests how far grounded removals move the percids; it does not promise envelope-membership
for pikeperch.

## 2. Base configuration and the mandatory pre-flight

- **Branch** off `646a36d` (aggregate 8-species). `master` keeps the disaggregation experiment.
- **Restore + verify FIRST (blocking):** apply `phase13_equilibrium.json`, re-run
  `baltic_stability_certify.py`, and confirm the 5/8 / obj-2.33 / cod ~64 kt state before touching
  anything. `phase13_equilibrium.json` is a 39-param, 8-species-era artifact — confirm it applies
  cleanly on the 8-species `get_phase13_shepherd_params` (NOT master's 45-param 9-species version).
  Without a verified clean baseline, later regressions are unattributable.

## 3. Engine facts (verified in the review)

- **Background predators DO predate focal fish**, scaled by prescribed biomass ×
  `predation.ingestion.rate.max`, size-ratio gated; absent from the accessibility matrix they
  predate at the **default coefficient 1.0** (`predation.py:211,222`). Adding a matrix column lets
  that coefficient be tuned per prey — **necessary, not optional**, to raise cormorant predation
  on perch without over-cropping herring/sprat.
- **Predator indices on the `646a36d` base are sp14 = GreySeal, sp15 = Cormorant** (no sp16). The
  sp15/sp16 in the master (disaggregated) config are shifted — using them would silently
  mis-target. All keys below use **sp15 = Cormorant**.
- **`species.biomass.multiplier.sp15`** scales cormorant NetCDF standing biomass
  (`background.py:366`); **`predation.ingestion.rate.max.sp15`** caps its realized predation
  (`predation.py:189`). Both are read for background species and are in the validation allowlist
  (AST walk of `background.py`) — no allowlist change needed.

## 4. Step 0 — mandatory cheap feasibility gate (before any calibration)

The disaggregation burned two multi-hour DE runs to discover a structural wall; its one decisive
cheap check was run *after* the fact. **Do not repeat that.**

1. On the restored baseline, crank the grounded levers to their **maximum defensible** values by
   hand — percid F to the top of the real coastal range (~0.6), cormorant biomass-multiplier +
   ingestion to their count/physiology anchors (§5.2), cormorant matrix column shaping predation
   onto perch/young pikeperch.
2. Run **one** forward sim (40–50 yr) and read: (a) how far perch, pikeperch, smelt move; (b)
   whether the well-assessed stocks (cod/herring/sprat/flounder/stickleback) stay in-envelope; (c)
   realized cormorant consumption vs the Hansson budget (~2× the fishery for perch).
3. **Go/no-go — final-decade statistics only.** *(Revised 2026-08-01; see the correction section at
   the end of this document. The original wording read an undefined "destabilise" off whole-run
   extrema, which are dominated by the seeding bootstrap and would have fired a false "futile → stop"
   on the whole effort — cod dips to 2.39e3 t on the aggregate baseline against a 61–68 kt
   final-decade mean.)*

   Evaluate **every** criterion on the final decade of the run. Never on whole-run minima or maxima.

   - **PROCEED** if perch's final-decade mean moves toward its envelope, **and** no well-assessed
     stock (cod, herring, sprat, flounder, stickleback) leaves its envelope on the final-decade mean.
   - **STOP — futile** if perch's final-decade mean is unmoved (< 5%, i.e. inside the ~1.9%
     seed-to-seed noise with margin) under maxed levers.
   - **STOP — harmful** if any well-assessed stock's final-decade mean leaves its envelope, or its
     final-decade minimum falls below `0.1 × envelope-lower` (the corrected `persists` test,
     `scripts/baltic_stability_certify.py` at `556ba3d` or later).

   **Smelt is NOT a go/no-go criterion.** Both levers are percid-directed and smelt is percid prey, so
   removing a predator releases it — smelt is expected to rise and is **monitored, not gated**.

   **Pre-flight requirement:** confirm the certify script in use is at `556ba3d` or later. The copy at
   this spec's base commit `646a36d` computes `vmin = float(v.min())` over the whole run, which is the
   discredited statistic this gate must avoid.

   This 15-minute check gates the 4–8 h run.

## 5. Components

### 5.1 Realistic percid fishing F (Lever A)

Set perch (fsh4) and pikeperch (fsh5) `fisheries.rate.base` to **absolute** literature-grounded
values for total (commercial + recreational) coastal removal — real exploited Baltic coastal
percid F is **~0.3–0.6**, often F > M — NOT a multiple of the model's calibrated artifact
(0.0095 → 2× = 0.019 is negligible). **Fixed, not a free param**, so it is not optimised away.
Value + provenance set in the plan from coastal-fishery statistics (Curonian/Baltic). Check total
Z = F + M_predation stays realistic so perch does not over-crash. If percid **catch targets** are
in the calibration objective, a fixed total-removal F will miss a commercial-only catch target —
exclude/adjust them consistently.

### 5.2 Cormorant predation (Lever B)

`species.biomass.multiplier.sp15` and `predation.ingestion.rate.max.sp15` as calibration free
params, **each bounded to its own realism anchor**: biomass to count-based standing stock;
ingestion toward the physiological rate (~70/yr for a 2 kg bird eating ~400–500 g/day — the
current 40/yr is low; prefer raising ingestion over inflating bird counts). Add a **cormorant
column to `predation-accessibility.csv`** to shape the perch/pikeperch fraction (necessary — else
coefficient 1.0 over-crops herring/sprat). Honest: reaches adult perch (window covers <34 cm) but
only *juvenile* pikeperch — a recruitment-side lever on pikeperch, not an adult-biomass one.
Validate realized consumption as an **output** against the Hansson budget, not just as a free
product of biomass × ingestion.

### 5.3 Scoped re-calibration

Full 8-species DE **only if the Step-0 gate passes**, warm-started from `phase13_equilibrium.json`.
Changes vs baseline: percid F fixed (Lever A); cormorant biomass-multiplier + ingestion added as
free params with **explicit x0 = the Step-0 max-grounded values**. Required plumbing:
- `apply_calibration.py` `_FILE_FOR` extended so `species.biomass.multiplier.` and
  `predation.ingestion.rate.max.` route to `baltic_param-background.csv` (else KeyError). Caveat:
  the bare `predation.ingestion.rate.max.` prefix also matches focal sp0–7 (which live in
  `baltic_param-predation.csv`) — safe here because only sp15 is free, but guard against misroute.
- State run-horizon vs any `byYear` series and check the scored final decade is quasi-stationary
  (the disaggregation was bitten by an RV-series wrap mismatch between calibration and cert).

## 6. Validation, acceptance bar, revert rule

- **Pre-registered magnitude bar:** perch overshoot reduced to ≤ (envelope-upper × 2) or better;
  smelt toward envelope; pikeperch — record the reduction achieved (expected small). "Improved
  toward" alone is insufficient — state the numeric threshold before running.
- **No-regression (hard):** every well-assessed stock (cod, herring, sprat, flounder, stickleback)
  stays in its baseline envelope; cod persistence no worse. Note this is *nearly guaranteed by
  construction* (warm-start + a lever tunable to zero effect), so it does not by itself prove the
  mechanism works — also predict the sign+magnitude of cormorant consumption and the percid
  response and check against the Hansson budget as an output test.
- **Insensitivity test for the structural claim:** before crediting a residual pikeperch overshoot
  to the coarse grid, push grounded removals to the top of the defensible range and show the
  residual persists.
- **Pre-registered revert rule:** if any well-assessed stock drops below its baseline envelope, or
  the objective exceeds the baseline's 2.33 by more than a set margin, revert to the baseline and
  record the finding as the structural limit (as the disaggregation did).

## 7. Risks

- **Percid mortality may destabilise** (prior 8-lever finding) — grounded magnitudes + the Step-0
  gate + the revert rule bound this; accept residual overshoot over over-cranking.
- **Pikeperch ~90× likely not closable** — stated as the binding constraint, not hidden; the
  effort's realistic wins are perch (~2×) and smelt (~5×).
- **SSB-vs-total-biomass** — state which metric the bar uses (the harness scores total biomass).

## 8. Config-binding checklist (must hold in the plan)

1. Indices: **sp15 = Cormorant** on `646a36d` (NOT sp16).
2. `apply_calibration._FILE_FOR`: add `species.biomass.multiplier.` and
   `predation.ingestion.rate.max.` → `baltic_param-background.csv`.
3. Add the cormorant column to `predation-accessibility.csv`.
4. Percid F set as fixed base rates (fsh4, fsh5); handle percid catch targets in the objective.

## 9. Out of scope / future

Cod-forcing / cod-stability (dropped — no cod gap); seal lever (no forage-fish gap); finer coastal
grid (the proper pikeperch fix); herring/flounder disaggregation.

## 10. References

`docs/baltic_findings_summary_2026-07-28.docx` §6 (full, verification-tagged). Key: Hansson et al.
(2018) *ICES JMS* 75(3):999; Baltic pikeperch status reviews (recreational ≥ commercial);
Heikinheimo et al. (2021), Östman et al. (2013) (cormorant predation on perch).

---

# ⚠ CORRECTION 2026-08-01 — gates and bars read a discredited statistic

`556ba3d` rescoped `persists` from the whole-run minimum (seeding-bootstrap dominated) to the
final-decade minimum. Audit: `docs/baltic_certification_reread_2026-08-01.md`. **The design's core
motivation survives** — the percid overshoot is an `in_envelope` failure, untouched by the fix.

1. **Smelt is not a reducible target — the levers push it the wrong way** (§1 L21, §6 L120, §7 L139).
   Both levers are percid-directed mortality and smelt is perch/pikeperch prey, so removing a predator
   *releases* it. The executed run confirms: smelt **×5.3 → ×5.8** (`baltic_percid_removals_outcome_2026-07-28.md`),
   same mechanism as raising cod_east M lifting sprat +14%. Smelt is a **monitored side-effect**, not a
   target; "smelt toward envelope" is unachievable by these levers by construction.
2. **The Step-0 go/no-go gate (§4 L76-78) would fire a false "futile → stop".** It reads an undefined
   "destabilise" off a single sim, and every stability verdict at spec-time was the whole-run minimum —
   on the aggregate baseline cod dips to 2.39e3 t against a 61–68 kt final-decade mean. This is the
   kill switch and the one gate with no multi-seed redundancy. **Redefine on final-decade statistics.**
   → **DONE**: §4 step 3 rewritten 2026-08-01 with explicit PROCEED / STOP-futile / STOP-harmful
   criteria on final-decade statistics, smelt excluded as a gate, and a `556ba3d` pre-flight check.
3. **The blocking pre-flight (§2 L42-46) reproduces the defect.** `git show 646a36d:scripts/baltic_stability_certify.py`
   has `vmin = float(v.min())`; `556ba3d` is on master only. **Cherry-pick `556ba3d` before the
   pre-flight.** Also: "5/8" in the source docs is the *in-envelope* count, while the script prints the
   combined persist∧in-envelope verdict — **2/8** pre-fix. The numbers coincide only after the fix, so
   "confirm 5/8" cannot be satisfied by the pre-fix script.
4. **"Cod persistence no worse" (§6 L123) was vacuous and its reference has flipped.** Cod read
   `persists ✗` in the cited baseline, so "no worse than ✗" was unfailable. Post-fix cod is a PASS, so
   the bar becomes materially stricter than intended. Wrong either way as written.
5. **Both "destabilising" premises need re-deriving** (§7 L136; §1 L34-36), by provenance:
   - the certification rows behind "the percid work destabilises" are **verified artifacts**
     (`baltic_percid_removals_certification_2026-07-28.md` is on the audit list, cod/sprat/flounder/perch);
   - **Lever 8**'s boom/bust 76→2591 rests on a whole-run max/min ratio (cf.
     `scripts/baltic_rv_overshoot_diagnostic.py:311`) — **unsupported pending re-derivation**, not false;
   - **Lever 1** cites an `in_envelope` result and **survives** — do not lump it with the above.

**Verified correct, do not "fix":** pikeperch ~90× / perch ~2× / smelt ~5× are right for this spec's
8-species base at `646a36d` (finding #5's "~56×" is the 9-species master — different config, both
correct in context); the scope note's cod descoping already distinguishes seeding from equilibrium.
