# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** config default

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✓ | 1.14e+04 | [11948.479170111133, 13800.534609436358] |
| cod_east | ✓ | ✓ | 4.73e+04 | [63972.24554234184, 66446.06525502432] |
| herring | ✓ | ✓ | 2.25e+06 | [2489031.5165172284, 2606459.760888887] |
| sprat | ✓ | ✓ | 9.61e+05 | [1013208.2987491975, 1035926.6918703874] |
| flounder | ✓ | ✓ | 2.86e+04 | [31839.751920479448, 34034.62534292196] |
| perch | ✓ | ✓ | 3.81e+04 | [42554.35412166352, 44848.57463315508] |
| pikeperch | ✓ | ✗ | 1.21e+06 | [1375582.0819530422, 1459487.6104924665] |
| smelt | ✓ | ✗ | 6.38e+05 | [672269.4534443791, 694336.4623709053] |
| stickleback | ✓ | ✓ | 6.42e+04 | [77096.56215668567, 84954.15255095414] |

**Python verdict: 5/5 ASSESSED species persistent & in-envelope.** All 5 pass — candidate is certifiable; verify value round-trip before writing data/baltic.

*Indicative tier: 2/4 (perch w=0.2, pikeperch w=0.2, smelt w=0.3, stickleback w=0.2).* These targets are **not ICES assessments** — ICES does not assess Baltic pikeperch, perch, smelt or stickleback. `biomass_targets.csv` sources them as literature estimates at weight ≤ 0.3, noting the coarse grid under-resolves species concentrated in estuaries and lagoons. They are reported for information and are **not** part of the verdict; do not tune against them. (Legacy all-species figure, for comparison with notes written before 2026-08-04: 7/9.)
