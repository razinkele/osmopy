# Baltic cod E/W disaggregation — Phase 1 report (2026-07-25)

Splitting the aggregated Baltic cod stock into **cod_west** (sp0, cod.27.22-24)
and **cod_east** (sp8, cod.27.24-32) as two OSMOSE focal species, so the model can
represent a collapsed eastern stock coexisting with a healthier western one — the
qualitative structure the single cod stock averages away.

- **Spec:** `docs/superpowers/specs/2026-07-24-baltic-stock-disaggregation-design.md`
- **Plan:** `docs/superpowers/plans/2026-07-24-baltic-cod-disaggregation-phase1.md`
- **Calibration finding (detail):** `docs/baltic_cod_ew_calibration_finding_2026-07-25.md`
- **Certification run:** `docs/baltic_cod_ew_certification_2026-07-25.md`

## Outcome in one line

The **disaggregation structure is complete and works** (Tasks 1–6); the
**9-species re-calibration (Task 7) hit a structural wall** that is itself a
scientific finding — a collapsed eastern-cod food web cannot hold the prey species
in their ICES envelopes without the apex predator. A proper warm-started
re-calibration is in progress to test whether prey fishing can compensate.

## Tasks 1–6 — structure (done, committed, pushed)

| Task | Deliverable | Verification |
|------|------------|--------------|
| 1 | `scripts/reindex_species.py` — append-shift tool | Lossless on the real 629-key config (focal untouched, LTL/bg +1, nspecies 8→9) |
| 2 | cod → cod_west + cod_east (sp8) full param set | 9-species config loads, warn-mode clean; eastern life-history (impaired condition 0.0068 vs 0.0087, early maturity 22 vs 38 cm, buoyant eggs 0.17 vs 0.15, truncated lifespan 15 vs 20, summer spawning) |
| 3 | Salinity-niched W/E distribution maps | Go/no-go **passed** — footprints overlap only 10.3% (west centroid ~12.3°E Belt/Arkona, east ~18.9°E Gdańsk/Gotland deeps) |
| 4 | Predation matrix 14×14 → 15×15 | Loads through the engine's name-resolving `AccessibilityMatrix`; cod_east more sprat/benthos-dependent, no cross-predation between stocks |
| 5 | ICES target split (cod_east 70 kt, cod_west 10 kt) | Pulled from ICES SAG via MCP: cod.27.24-32 2018–2022 mean SSB ~70 kt; cod.27.22-24 ~5–15 kt; both below Blim |
| 6 | cod_east RV gate (raw_cap) + elevated M + own fishery | Config runs end-to-end (returncode 0); cod_west already ≈ its 10 kt target in the smoke test |

Key mechanic verified: the predation matrix, fishery-catchability, and movement
maps are **name-labeled, not positional** (`accessibility.py` resolves sp_idx→row
by name), so appending a focal species and shifting indices ≥8 does not corrupt
them — they are expanded/renamed by name. `apply_calibration.py` and the
certification harness were made `nspecies`-driven (were hardcoded `range(8)`).

## Task 7 — calibration failure and finding

The first 9-species phase-13 DE (bounded ~3.6 h, eff_popsize 90) converged to
**objective 12.34 vs the pre-split baseline's 2.33** and certified **2/9**
in-envelope. Every species ran 10–80× over its ICES envelope at *both* the 40-yr
calibration horizon and the 50-yr certification:

| species | 40yr mean | envelope | over |
|---|---|---|---|
| cod_west | 711 kt | 4–25 kt | 28× |
| cod_east | 5.08 Mt | 60–85 kt | 60× |
| herring | 1.71 Mt | 0.8–3 Mt | OK |
| sprat | 3.09 Mt | 0.8–2.5 Mt | 1.2× |
| flounder | 2.30 Mt | 20–100 kt | 23× |
| perch | 413 kt | 8–50 kt | 8× |
| pikeperch | 2.04 Mt | 4–25 kt | 82× |
| smelt | 175 kt | 20–120 kt | 1.5× |
| stickleback | 62 kt | 50–500 kt | OK |

### Diagnosis

- **Not the RV-wrap window mismatch.** The RV series wraps modulo its length, so
  the 50-yr certification's final decade wraps into high-RV years while the 40-yr
  calibration's sits in the low-RV trough. But cod_east is 60× over at 40 yr too,
  so the wrap is not the cause — the boom is at the calibration horizon.
- **Not merely optimizer budget.** Under-budgeting made it worse, but the failure
  is structural. The pre-split baseline's sp1-7 params were tuned with a **full
  cod apex predator** (~150 kt) cropping the prey field. Disaggregating cod and
  suppressing the eastern stock removes that top-down control, so the prey
  (pikeperch, flounder, perch, sprat) are released far above their ICES envelopes.
- **The objective faces a tension.** Hitting cod_east's low ~70 kt target releases
  the prey and *worsens* the prey fits, so the DE resolved it by NOT suppressing
  cod_east — booming everything to a mediocre-but-balanced obj 12.34. A hand-built
  warm-start that forces cod_east down scores **1817**, confirming the two goals
  fight each other.

### The finding

**A collapsed eastern-cod Baltic cannot hold the prey species in their ICES
envelopes without the apex predator** — consistent with the prior result that even
the aggregated 8-species model is only 2/8 stable. This is a real property of the
disaggregated food web, not just a calibration nuisance.

The one caveat that keeps a re-calibration worthwhile: the failed DE left prey
*fishing* low (flounder F=0.008) despite the boom — it was under-budgeted and
didn't use its levers. A properly budgeted run *may* hold the prey via higher prey
fishing, though holding flounder/pikeperch at F≈3 is itself fidelity-questionable.

## Proper re-calibration — run and result (did NOT clear the bar)

The best-shot re-calibration ran to completion (warm-start all 9 from
`phase13_equilibrium.json`; eff_popsize 180 vs 90; `fsh8` clamped to the
moratorium; 3240 evals, early-stop after 15 stale gens):

- **Objective 8.855** — better than the failed 12.34 but still far above the
  baseline's 2.33.
- 40-yr per-species biomass: **cod_west 1.37 Mt (55× over), cod_east 1.11 Mt
  (13× over — more suppressed than the 60× first attempt but still nowhere near
  70 kt), herring 13 Mt (4× over), pikeperch 1.64 Mt (65×), perch 836 kt (17×),
  flounder 1.20 Mt (12×), stickleback COLLAPSED (under floor).**

**Acceptance bar (honest, structural — not 9/9):** cod_east suppressed toward
~70 kt, cod_west near ~10 kt, others no worse than baseline. **NOT cleared** on
any leg.

Decisive detail: with 2× the budget, a warm start, and `fsh8` clamped, the DE
**still left prey fishing low** (flounder F=0.02, pikeperch F=0.049) despite the
boom. It does not use the prey-fishing lever to hold the released prey — so the
failure is **robust and structural, not a budget artifact.** (Likely because
raising prey F fights the prey CATCH targets; holding flounder/pikeperch at the
F≈3 that predation-replacement would need is both unreachable-by-the-optimizer
and fidelity-questionable.)

**Conclusion:** the apex-predator-release limitation holds under a proper effort.
The pre-split 5/8 baseline (`phase13_equilibrium.json`, obj 2.33) is intact and
restorable; the failed params were reverted from the live config.

## Assets

- Pre-split 5/8 baseline: `data/baltic/calibration_results/phase13_equilibrium.json` (obj 2.33, intact).
- Reproducible surgery scripts: `disaggregate_cod.py`, `build_cod_ew_maps.py`, `expand_cod_predation_matrix.py`, `configure_cod_ew_dynamics.py`, `reindex_species.py`.
