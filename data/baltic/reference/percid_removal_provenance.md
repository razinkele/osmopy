# Percid fishing-mortality provenance (2026-07-28)

Fixed elevated fishing mortality for perch (`fisheries.rate.base.fsh4`) and
pikeperch (`fisheries.rate.base.fsh5`), representing **total (commercial +
recreational) coastal removal** that the model previously under-counted.

## Values

| species | fishery | committed F | rationale |
|---|---|---|---|
| perch (sp4) | fsh4 (coastalperch) | **0.40** | total coastal exploitation F |
| pikeperch (sp5) | fsh5 (coastalpikeperch) | **0.50** | total coastal exploitation F |

These **replace the calibration artifacts** (perch 0.029, pikeperch 0.0095) that
the objective chose only because percid ICES assessment weight is 0.2. Held
**fixed** (not a DE free param) so the realistic removal is imposed, not
optimised away.

## Basis

- Baltic coastal perch and pikeperch are exploited by small-scale coastal
  commercial **and** recreational fisheries whose catch is poorly reported and
  **not analytically (category-1) assessed by ICES** — unlike cod/herring/sprat.
  Real exploited coastal percid F is **~0.3–0.6, frequently F > M** (M ≈ 0.2–0.3).
- **Recreational catch can equal or exceed commercial** in the Baltic coastal
  zone — documented for Lithuania (Curonian Lagoon) and the Archipelago Sea
  (Baltic pikeperch status reviews, 2019–2024).
- Hansson, S. et al. (2018). Competition for the fish – fish extraction from the
  Baltic Sea by humans, aquatic mammals, and birds. *ICES Journal of Marine
  Science* 75(3), 999–1008. https://doi.org/10.1093/icesjms/fsx207 — coastal-fish
  removals and the reporting gap.

## Caveat

Total Z = F + M + cormorant-predation must stay realistic so perch does not
over-crash (checked in the Task-4 feasibility gate). F = 0.40/0.50 are within
the documented coastal range but at its upper-middle; the gate + the certification
no-regression bar are the guards.
