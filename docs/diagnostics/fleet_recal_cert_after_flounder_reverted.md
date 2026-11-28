# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** config default  ·  **engine:** Python

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✗ | 1.56e+04 | [16321.021244370186, 16846.955053619087] |
| cod_east | ✓ | ✓ | 4.59e+04 | [61277.734026997256, 64148.36165963912] |
| herring | ✓ | ✓ | 1.95e+06 | [1995401.8870164403, 2117364.660440367] |
| sprat | ✓ | ✓ | 9.32e+05 | [989824.6000933412, 1001711.4027635831] |
| flounder | ✓ | ✓ | 4.35e+04 | [45537.47493982165, 47726.25286149244] |
| perch | ✓ | ✓ | 2.96e+04 | [31107.03329226974, 34159.93605201655] |
| pikeperch | ✓ | ✗ | 5.09e+05 | [543338.0410517247, 551820.984936831] |
| smelt | ✓ | ✗ | 5.90e+05 | [634699.3061618468, 650160.9892037152] |
| stickleback | ✓ | ✓ | 4.42e+04 | [57125.64547930349, 74752.62709635438] |

**Python verdict: 4/5 ASSESSED species persistent & in-envelope.** Not 5/5 — SP-B gate: the failing ASSESSED species are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).

*Indicative tier: 2/4 (perch w=0.2, pikeperch w=0.2, smelt w=0.3, stickleback w=0.2).* These targets are **not ICES assessments** — ICES does not assess Baltic pikeperch, perch, smelt or stickleback. `biomass_targets.csv` sources them as literature estimates at weight ≤ 0.3, noting the coarse grid under-resolves species concentrated in estuaries and lagoons. They are reported for information and are **not** part of the verdict; do not tune against them. (Legacy all-species figure, for comparison with notes written before 2026-08-04: 6/9.)
