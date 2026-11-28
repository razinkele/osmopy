# Baltic fleet-rate recalibration to ICES 2020–2024 — 2026-11-28

Spec: `docs/superpowers/specs/2026-11-28-baltic-fleet-rate-recal-design.md` ·
Plan: `docs/superpowers/plans/2026-11-28-baltic-fleet-rate-recal.md` ·
Before: `docs/baltic_ices_check_2026-11-28.md`

## Summary

- Every windowed ICES target moved to **2020–2024** (catch rows re-derived; cod_west SSB
  2020–22). **cod_east SSB stays 70 kt (2018–22)**: ICES now publishes `cod.27.24-32` SSB only as a
  relative index with no reference points, so the absolute target cannot be re-derived.
- Three fleets re-solved to ICES catch: **cod_west 0.039 → 0.182, cod_east 0.010 → 0.044,
  herring 0.380 → 0.121**. Sprat keeps its ICES-F anchor (0.369).
- **Flounder is unsolved.** Its ICES catch (9.5 kt) needs F ≈ 0.025, which grows the stock 8×
  past its 20–100 kt envelope (388 kt) and pushes cod_east out of its envelope. Certification wins
  (owner decision), so flounder keeps 1.368 and its catch stays ~4× ICES.
- Certification: **assessed 4/5**, the same as the old rates scored against the new targets; all
  9 species persist. The one miss is cod_west, and the **narrower 2020–22 cod_west band
  (5.3–9.1 kt, was 4–25 kt), not the new rates, costs that pass**: against the old band the old
  config read 5/5 and cod_west at 16–17 kt would still pass. cod_west's SSB actually moved toward
  ICES (19.2k → 16.3–16.8k).
- SP1b re-solved: rates unchanged (9.029 / 8.940), baselines moved.

## Targets (2020–2024, from `scripts/derive_ices_targets.py --write`)

| species | catch target t [band] | was (2018–22) | SSB target |
|---|---:|---:|---|
| cod_west | 1 428 [115–3 256] (3 yr) | 2 951 [115–6 404] | 6 699 [5 303–9 133] (2020–22); was 10 000 [4 000–25 000] |
| cod_east | 1 566 [922–2 654] | 7 878 [1 181–19 726] | 70 000 [60 000–85 000] — **2018–22 exception** |
| herring | 228 570 [166 998–297 546] | 302 459 | unchanged (literature, order of magnitude) |
| sprat | 273 249 [242 559–303 940] | 298 062 | unchanged |
| flounder | 9 520 [5 626–13 413] | 15 683 | unchanged |

## Rates and catch (Python, 5 cert seeds, 50 yr, trailing 5-yr mean)

| fleet | rate before → after | catch before | catch after | after / target |
|---|---|---:|---:|---:|
| fsh0 cod_west | 0.039 → **0.182** | 460 t | 1 693 t | 1.19× (in band) |
| fsh8 cod_east | 0.010 → **0.044** | 578 t | 2 131 t | 1.36× (in band) |
| fsh1 herring | 0.380 → **0.121** | 452 kt | 206 kt | 0.90× |
| fsh3 flounder | 1.368 (kept) | 40.0 kt | 38.6 kt | 4.05× — unsolved |
| fsh2 sprat | 0.369 (anchor) | 289 kt | 290 kt | 1.06× |

The search (C#, 2 seeds) converged in 2 passes with all four fleets within ±10% on C#
(`docs/diagnostics/fleet_recal_search.json`). Python confirmed three within ±10% and herring at
0.899, just outside ±10% but inside the ±15% confirm rule and its ICES band (5 seeds: 1.00 /
1.00 / 0.899 / 1.01, `fleet_recal_confirm.json`). Reverting flounder enlarged both cod stocks, which raised their
catches to 1.19× and 1.36×. Both stay inside their ICES bands. They were **not** re-solved to the
exact target: that needs higher cod rates, which lower cod biomass, and cod_east already sits at its
60 kt envelope floor.

**Flounder's catch curve** (C#, 2 seeds): rate 0.17 → 38 kt, 0.33 → 44 kt, 0.67 → 42 kt,
1.37 → 39 kt, 2.74 → 36 kt; 0.025 → 9.6 kt. The curve is flat across 0.17–1.37: the model's
flounder stock is far larger than the 2020–24 ICES catch implies. ICES flounder catch fell from
23–26 kt (2015–18) to ~8 kt (2023), mostly through lower demand rather than a smaller stock.
Matching it is a biomass-calibration question, not a rate question.

## Certification (Python, cert seeds 42 123 7 999 2024, 50 yr)

| species | before (old rates, new targets) | 4 fleets solved | flounder reverted (shipped) | envelope |
|---|---|---|---|---|
| cod_west | 19.2k ✗ | 13.9k ✗ | 16.3–16.8k ✗ | 5.3–9.1k |
| cod_east | 73k ✓ | 46k ✗ | 61.3–64.1k ✓ | 60–85k |
| herring | 1.53M ✓ | 2.06M ✓ | 2.00–2.12M ✓ | 0.8–3M |
| sprat | 1.00M ✓ | 1.03M ✓ | 0.99–1.00M ✓ | 0.8–2.5M |
| flounder | 48k ✓ | 388k ✗ | 46–48k ✓ | 20–100k |
| assessed | **4/5** | 2/5 | **4/5** | |
| indicative | 2/4 | 1/4 | 2/4 | |
| persists | 9/9 | 9/9 | 9/9 | |

"Before" scores the old rates against the NEW targets (spec), so the target change alone cannot
fail the gate. That is also why both read 4/5: under the old 4–25 kt cod_west band the old config
would have read 5/5 (cod_west 19.2k inside), and so would the shipped one (16.3–16.8k).

cod_east's lowest seed sits 2% above its envelope floor: a borderline pass. A later change that
lowers cod_east by a few percent will flip it.

## Against ICES (validator, 3 seeds 1–3, last 5 of 50 yr, 2020–24)

| species | SSB before | SSB after | ICES SSB envelope | verdict |
|---|---:|---:|---:|---|
| cod_west | 15.8–17.1k (2.3–2.5×) | 14.1–14.5k (2.0–2.1×) | 5.3–9.1k | ✗, closer |
| herring | 1.38–1.43M (1.7–1.8×*) | 1.89–1.93M (2.4×*) | 0.78–0.81M* | ✗* |
| sprat | ~798k (0.95×) | ~802k (0.95×) | 635k–1.12M | ✓ |
| cod_east | 66–68k | 58k | index only | — |

\* The herring envelope covers only the tonnes-unit stocks. The index-only central Baltic stock is
excluded, so the real ICES total is larger and the factor overstates the gap. Herring SSB rises
because its catch is now cut to ICES.

**Cod predation vs WGSAM SMS M2:** herring 0.51 → **0.37** of SMS; sprat 1.64 → **1.33**. The
sprat rate moves toward SMS; herring moves away, because the herring stock it is taken on grew.

## What did not move

- **cod_west's SSB is still ~2× ICES** even with F 4.7× higher. Fishing explains only part of the
  overshoot. The next lever is biological (recruitment or the SP1 larval pathway), not fishing.
- **Flounder** (see above).

## Files

- Rates: `data/baltic/baltic_param-fishing.csv` · targets: `data/baltic/reference/biomass_targets.csv`
- Solver: `osmose/calibration/fleet_rates.py` · driver: `scripts/recalibrate_fleet_rates.py`
- Diagnostics: `docs/diagnostics/fleet_recal_{baseline,search,confirm,final_catch}.json`,
  `fleet_recal_cert_{before,after,after_flounder_reverted}.md`, `sp1b_solve.json`
- Test band re-derived (isolated A/B): `tests/test_tutorial_3species.py` stickleback.
