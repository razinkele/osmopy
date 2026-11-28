# Baltic fleet-rate recalibration to ICES 2020–2024 — design

**Date:** 2026-11-28 · **Owner decision:** approaches B (target window) and "rates, not maps"

## Intent

The ICES comparison (`docs/baltic_ices_check_2026-11-28.md`) showed the Baltic model's fishing out of line with ICES: western cod SSB 2.4× ICES with a catch a
third of ICES, flounder catch ~4×, herring catch ~1.9×, eastern cod catch ~⅓. The owner wants the
model realistic **for the ICES comparison**. Spatial fleet maps were considered and dropped: each
fleet catches exactly one species and the habitat maps already confine every species where ICES
places it, so area masks would change almost nothing (measured 2026-11-28: cod W/E already split at
SD24; herring/sprat/flounder/stickleback/smelt fleets already cover their whole range; only
perch/pikeperch edges would move). What moves the comparison is the **per-fleet base rates**.

## Goals

1. Move every ICES target to the **2020–2024** window (the latest assessments, the validator's
   default), catch and SSB together so the two describe the same years.
2. Re-solve `fisheries.rate.base.fsh{N}` for the fleets whose model catch falls outside its new
   ICES catch band, so each lands inside it.
3. Keep the ecosystem certified: no species that persists today stops persisting, and the
   in-envelope count does not fall.
4. Record the before/after against ICES in a dated report.

## Non-goals

- Fleet maps, seasonal closures, MPAs (a later, scenario-driven project).
- Fleets without an ICES catch: perch (fsh4), pikeperch (fsh5), smelt (fsh6), stickleback (fsh7)
  keep their rates.
- Re-fitting any biological parameter (growth, M, recruitment). If a fleet cannot reach its band by
  rate alone, that is reported, not compensated elsewhere.
- Changing the C# or Java engines.

## Anti-goals (a failure even if the numbers match)

- Hitting catch bands by collapsing a stock (catch inside band but SSB → 0 or `persists ✗`).
- Mixing windows (SSB targets 2018–22, catch targets 2020–24).
- Leaving SP1b's frozen cod rates solved against the old baseline (see CLAUDE.md, SP1b gotcha:
  the d0 guard cannot see a fishing change, so a cod-shifting change forces a re-solve).

## Constraints

- Catch rows in `biomass_targets.csv` are an OUTPUT of `scripts/derive_ices_targets.py --write`
  (pinned by `tests/test_derive_ices_targets.py`): change `WINDOW` there and re-run; never hand-edit
  catch rows.
- `cod.27.22-24`'s snapshot is pinned to the 2022 assessment (the only absolute one), so its window
  is 2020–2022; keep the pin. Eastern cod and flounder F are relative in ICES: their target is catch
  only. Herring gets no F verdict (central stock relative); its target is the summed 4-stock catch.
- Model catch = `yield` output (tonnes/yr), trailing-5-year mean of a 50-yr run, as the validator
  and calibration already read it.
- Seeds: the certification seeds (`CERT_SEEDS`), so before/after are paired draws; replicate a
  borderline result on other seeds (memory: shared seeds are not independent).

## Approaches considered

| | Approach | Trade-off | Verdict |
|---|---|---|---|
| 1 | Per-fleet fleet maps (ICES SD + coastal masks) | near no-op on the ICES comparison (see Intent) | dropped |
| 2 | **Per-fleet 1-D rate solve, iterated** | one parameter per fleet, transparent, each fleet catches one species so the problem is nearly separable | **chosen** |
| 3 | Joint DE/CMA-ES over all rates via `calibrate_baltic.py` | handles coupling, but hours of compute, opaque, and re-fits more than needed | fallback only if 2 does not converge |

Target-window options were A keep 2018–22, **B move all to 2020–24 (chosen)**, C catch only
(rejected: mixes windows).

## Design

### 1. Targets (2020–2024)

- `scripts/derive_ices_targets.py`: `WINDOW = (2020, 2024)`; re-run `--write`; update its docstring
  and the CSV header (`ices_advice_year`, `last_updated`, version bump) and the test's expectations.
- Hand-update `cod_west`'s SSB row from its snapshot over 2020–2022 (tonnes; 5.3–9.1 kt).
- **`cod_east` SSB stays the 2018–22 70 kt row (owner decision):** ICES publishes
  `cod.27.24-32` SSB only as a relative index with no reference points, and the 70 kt was
  hand-curated from an older absolute figure, so it cannot be re-derived. Its `source`/`notes`
  must say it is the one 2018–22 exception; the report repeats this.
- Other `biomass`-type rows are order-of-magnitude literature targets, not windowed ICES
  numbers: unchanged.
- Catch targets from the script with `WINDOW=(2020, 2024)` (dry run, 2026-11-28), t:
  cod_west 1428 [115–3256] (3 yr), cod_east 1566 [922–2654], herring 228 570 [166 998–297 546],
  sprat 273 249 [242 559–303 940], flounder 9520 [5626–13 413]. The `--write` run must reproduce
  these; the test may need adapting to a 3-year window for cod_west.

### 2. Which fleets get solved

**Owner decision:** solve exactly four fleets **to their target** (not merely into the band):
fsh0 cod_west (up; its band is so wide, floored at the 2022 low, that today's catch already sits
inside it), fsh8 cod_east (up), fsh1 herring (down), fsh3 flounder (down).

**Sprat fsh2 keeps its ICES-F anchor (0.369, owner decision):** never solved; its catch is
reported. Herring is solved on catch: it was left unchanged when its catch matched ICES, which no
longer holds (model 440 kt is above both the 2018–22 and the 2020–24 band).

### 3. Solver

- One fleet at a time. First a **coarse sweep** of 4–5 rates on a log scale spanning the current
  rate and the expected solution (one seed), to map catch vs rate. Catch vs rate is domed (past
  F_MSY more fishing gives less catch) and a catch target can have two roots: always take the
  **lower-F root**, bisecting on `log(rate)` inside the bracket the sweep found, to within ±10% of
  `target_tonnes` or 8 evaluations. If the target is above the dome's peak, take the peak's rate
  and report the maximum attainable catch.
- Lowering a rate (herring, flounder) can raise catch if the fleet is now past the peak; the sweep
  shows which side it is on, so the solver never assumes direction.
- Coupling (cod eats sprat/herring; herring eats stickleback eggs): after one pass over all solved
  fleets, re-check every fleet; repeat passes (max 3) until all are inside their bands.
- **Compute budget (my call):** a 50-yr Python seed takes ~2.7 min, so an all-Python search
  (4 fleets × ~8 evals × up to 3 passes × 5 seeds) would take 20+ h. Search on the **C# engine**
  with 2 seeds; confirm the final rates on **Python with the 5 cert seeds**. If Python puts a
  fleet outside ±15% of its target, refine that fleet on Python (2 seeds) before confirming.
- Runs use the SP1-off production base; SP1b is an overlay re-solved once at the end (step 4) and
  does not move production catches.
- Script: `scripts/recalibrate_fleet_rates.py` (new), writing a JSON log of every evaluation.

### 4. Apply and re-certify

- Write the solved rates into `data/baltic/baltic_param-fishing.csv` with a provenance comment.
- "Before" = the **current rates certified against the new 2020–24 targets** (run once after
  step 1), so a target move alone cannot fail the gate.
- `scripts/baltic_stability_certify.py` (5 seeds, 50 yr): no persisting species lost, in-envelope
  count not lower than before.
- **Certification wins (owner decision):** if the gate fails, back the offending fleet off toward
  the edge of its catch band (it re-solves to the band edge nearest its old catch); if it still
  fails, that fleet keeps its old rate and the report says so. Certification is a pass/fail check
  at the end, not inside the solver loop.
- SP1b: re-solve `RECAL_RATES` (`osmose/calibration/larva_recal.py`) against the new baseline, as
  after the GreySeal column change.
- Re-derive any fixture that pins absolute Baltic biomass (e.g. `tests/test_tutorial_3species.py`
  `_PYRAMID_BOUNDS`), run the full suite and the C# suites.
- Re-run `scripts/validate_outputs_vs_ices.py --prefix baltic --ices-window 2020-2024 --sms-m2`.

### 5. Report

`docs/baltic_fleet_rate_recal_2026-11-28.md`: targets before/after, rates before/after, catch and
SSB vs ICES per stock, certification table, SP1b re-solve, and any fleet that could not reach its
band (with the max attainable catch).

## Success criteria

- Every solved fleet's 5-seed mean catch within ±10% of its target, or inside its band where
  certification forced a back-off, or reported unsolved with the reason.
- Certification: no persists ✗ that was ✓ before; in-envelope count ≥ before.
- cod_west SSB moves toward the ICES envelope (reported, not gated: SSB is the outcome, catch the
  target).
- Full suite and C# suites green.

## Left to the builder

- Bracket factors and bisection details within the rules above.
- Whether to search on C# and confirm on Python, or search on Python throughout.
- Report layout.
