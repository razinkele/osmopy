# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** config default  ·  **engine:** Python

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✗ | 1.75e+04 | [18947.226622716313, 19398.12230430759] |
| cod_east | ✓ | ✓ | 5.35e+04 | [72460.96614443793, 74219.57772513054] |
| herring | ✓ | ✓ | 1.47e+06 | [1520218.614770869, 1547825.5132050845] |
| sprat | ✓ | ✓ | 9.30e+05 | [996319.8038034823, 1009240.8861277911] |
| flounder | ✓ | ✓ | 4.27e+04 | [46945.94228051653, 48339.29907900714] |
| perch | ✓ | ✓ | 3.23e+04 | [34051.82885068738, 35842.43094247262] |
| pikeperch | ✓ | ✗ | 4.74e+05 | [515160.9959617901, 531838.8141153296] |
| smelt | ✓ | ✗ | 6.19e+05 | [650929.6658196303, 665946.276163927] |
| stickleback | ✓ | ✓ | 6.78e+04 | [86329.53798807246, 94748.20021353709] |

**Python verdict: 4/5 ASSESSED species persistent & in-envelope.** Not 5/5 — SP-B gate: the failing ASSESSED species are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).

*Indicative tier: 2/4 (perch w=0.2, pikeperch w=0.2, smelt w=0.3, stickleback w=0.2).* These targets are **not ICES assessments** — ICES does not assess Baltic pikeperch, perch, smelt or stickleback. `biomass_targets.csv` sources them as literature estimates at weight ≤ 0.3, noting the coarse grid under-resolves species concentrated in estuaries and lagoons. They are reported for information and are **not** part of the verdict; do not tune against them. (Legacy all-species figure, for comparison with notes written before 2026-08-04: 6/9.)
