# Baltic recalibration on the corrected LTL forcing (2026-11-08)

**What this is:** a new calibration of the production Baltic model. It is not a fix certified
against the old calibration. Four corrections are combined, and two larval mortality rates are
re-fitted onto the biomass targets. Each step below was measured before the next was decided, and
each reading was stated before its run. Those statements are committed as written:
`docs/diagnostics/baltic_ltl_gate_prestated_2026-11-08.md` (the gate) and
`docs/diagnostics/baltic_recal_prestated_2026-11-08.md` (the solve's targets, fallback and
cod_east rule). The 'accept only if every seed keeps cod_west at or above 5 kt' rule in §3 was
stated in the working session before candidate A's certification ran.

## 1. The corrections

| change | what it corrects | source |
|---|---|---|
| LTL forcing regenerated (#234 generator) | the old file integrated 0–50 m as a level mean × a full 50 m column, about 2.5× too much food (phyto 0.52–0.54×, zoo/benthos 0.48× after the fix), and 65 coastal ocean cells had none | issue #229 (measurements and the gate); the hole-fill-only record is on branch `fix/baltic-ltl-coastal-holes` |
| C2: cod_east spawning on 14 SD 25 cells | the old map had no Bornholm Basin cell and 89 of 144 cells at ≥ 57 °N | Eero et al. 2012; `docs/baltic_spawning_fixes_2026-11-08.md` |
| herring autumn eggs folded into spring | an unsourced 25% autumn share; the sourced value is about 1% or less today | Bekkevold et al. 2016; Rechlin 1991; Atmore et al. 2022 |
| herring and cod_west larval mortality re-fitted | brings both onto their biomass targets after the corrections | this note, §3 |

The engine already carries #233 (sub-one-fish floor, no L∞ cap), #227 (GreySeal accessibility)
and #230 (no herring autumn packing).

## 2. Measure first: two readings before any fitting

Both runs are 50 yr × 5 seeds with current parameters (final-decade means).

| species | before | corrected LTL (gate) | + C2 + autumn eggs 0 (10a) | envelope |
|---|---|---|---|---|
| cod_west | 11.5–12.6 kt | 24.5–25.7 kt | 24.9–26.1 kt | 4–25 kt |
| cod_east | 63.7–65.1 kt | 85.2–87.1 kt | 80.8–82.2 kt | 60–85 kt |
| herring | 2.68–2.72 Mt | 1.16–1.17 Mt | 0.798–0.807 Mt | 0.8–3.0 Mt |
| sprat | 0.93–0.94 Mt | 0.88–0.89 Mt | 0.91–0.93 Mt | 0.8–2.5 Mt |
| flounder | 32–34 kt | 51–52 kt | 51.2–51.9 kt | 20–100 kt |
| *pikeperch* | *1.35–1.41 Mt* | *0.54–0.56 Mt* | *0.51–0.52 Mt* | *4–25 kt* |

- **Gate reading: GREEN.** The criteria were stated before the run: all assessed stocks persist,
  each within 2× of its envelope. The only misses were the two cod stocks, 2–3% over their
  ceilings. The corrected food field also cut pikeperch by 60%, the largest move any intervention
  has made on its overshoot.
- **Step 10a: 3/5, with nothing collapsing.** On the corrected food field the autumn-egg fold no
  longer collapses cod_west, as it did on the old field. C2 brings cod_east under its ceiling.
  Two marginal misses remain: cod_west 4% over, herring on its floor.

## 3. The targeted solve

`scripts/recalibrate_baltic_larva_targets.py` (record and log in `docs/diagnostics/baltic_larva_targets_solve.*`).
The targets were stated before the first evaluation: herring 1.5 Mt and cod_west 10 kt, the
biomass_targets.csv values. Each stock is fitted on the final-decade mean of 50-yr seed-42 runs,
by Gauss-Seidel bisection with herring first.

| rate | before (CSV per year / resolved per step) | after | result |
|---|---|---|---|
| herring `larva.rate.sp1` | 98.56 / 4.107 | **28.89 / 1.204** | 1.48 Mt (target 1.5) |
| cod_west `larva.rate.sp0` | 243.76 / 10.157 | **257.75 / 10.740** | 10.6 kt (target 10) |

cod_east stayed inside its envelope (77.2 kt), so its rate was left unchanged.

**cod_west sits close to a cliff.** At larval rates 10.157, 10.740 and 11.356 (per step) it holds
22.3 kt, 10.6 kt and 0 respectively. The chosen rate is 5.7% below a rate known to drive it
extinct, so the 5-seed certification (§4) is the robustness test. The rule, stated before that
run: accept only if every seed persists with cod_west ≥ 5 kt; otherwise fall back to a rate of
about 10.45 (about 16 kt).

## 4. Certification

**Candidate A (the solved rates) fails the rule.** cod_west's final-decade means span
2.3–10.9 kt across the 5 seeds, and one seed sits under its 4 kt floor (4/5). Seed 42, the one
the solve fitted, was the best case. A single-seed solve that lands near a cliff fits the lucky
seed.

**Candidate B, the fallback, passes: 5/5 assessed.** It keeps herring's solved rate and sets
cod_west's rate to the geometric midpoint of the two bracketing rates,
sqrt(10.157 × 10.740) = 10.444 per step, or **250.66 per year** in the CSV.

| species | candidate A | **candidate B (adopted)** | envelope |
|---|---|---|---|
| cod_west | 2.3–10.9 kt ✗ | **14.9–18.3 kt** | 4–25 kt |
| cod_east | 77.2–80.7 kt | 76.1–80.0 kt | 60–85 kt |
| herring | 1.48–1.54 Mt | 1.48–1.56 Mt | 0.8–3.0 Mt |
| sprat | 0.88–0.91 Mt | 0.88–0.90 Mt | 0.8–2.5 Mt |
| flounder | 48.7–51.6 kt | 47.6–49.8 kt | 20–100 kt |
| *perch* | *61.7–64.6 kt* | *60.0–62.9 kt* | *8–50 kt* |
| *pikeperch* | *0.59–0.61 Mt* | *0.60–0.62 Mt* | *4–25 kt* |
| *smelt* | *0.65–0.67 Mt* | *0.65–0.67 Mt* | *20–120 kt* |
| *stickleback* | *90–101 kt* | *90–102 kt* | *50–500 kt* |

Moving back from the cliff narrows cod_west's seed spread from about 4.7× to 1.2×. Every seed
persists and sits inside its envelope. Pikeperch is 56% below the pre-correction model (1.38 Mt)
but still over its target. Perch rises above its indicative ceiling. Both are indicative tier,
outside the verdict.

**Adopted rates (CSV, per year):** `mortality.additional.larva.rate.sp1` 98.56 → **28.89**
(herring), and `mortality.additional.larva.rate.sp0` 243.76 → **250.66** (cod_west). cod_east's
`sp8` is unchanged at 243.76.

## 5. What else moves

- **Tests:** the full suite passes on the adopted config: 5027 passed, 0 failed. No absolute-value
  fixture needed re-deriving. The pyramid bands, the thermal-gate mechanism test and the GreySeal
  FR test all hold.
- **SP1b rates re-solved**, as required: cod_west's d0 moved 10.157 → 10.444. cod_east converged.
  cod_west reached 2.4% after 20 steps, inside its documented coupling jitter and inside
  JOINT_TOL (5%), the band the drift guard enforces (`docs/diagnostics/sp1b_recalibration.md`).
- **Docs brought current:**
  - `docs/baltic_example.md`: the percid limitation figures (pikeperch about 24× its ceiling, not
    58×), and C2 in the maps paragraph;
  - `docs/baltic-fish-lifecycle.md`: herring is a single spring bell; cod_east spawns in SD 25;
  - the movement-file comments;
  - `docs/baltic_spawning_fixes_2026-11-08.md`: marked settled.
- **Stale now, kept as dated records:** every Baltic result computed on the old LTL file. That
  includes the Java cross-check (`docs/baltic_java_crosscheck_2026-11-06.md`; the Java arm reads
  the same file), the ICES validation notes, the B2/C4 scenario tables and the generated reports.
  Each states its date; none was re-run here.
- **Cross-engine parity is unverified since #233.** The T3 floor and T1 no-cap engine changes
  affect every config. `scripts/cross_engine_parity_440.py` was NOT re-run, so the published
  "14/14 EEC, 7/8 BoB within 1 OoM" figures (CLAUDE.md, README, `docs/osmose_4.4_parity.md`)
  rest on the pre-#233 engine. Both changes move Python toward Java, so parity should hold or
  improve, but that is a prediction, not a measurement. Re-running it is the first follow-up when
  a jar is available.
- **Superseded parked branches:** `fix/baltic-ltl-coastal-holes` (the hole fill alone),
  `fix/baltic-cod-east-sd25-spawning` (C2 alone) and `exp/baltic-c1-full` (the autumn fold on the
  old field).
- **Release:** this is a new calibration of the production model, so v1.9.0 rather than a patch.
  v1.8.7 is tagged and not yet deployed.
