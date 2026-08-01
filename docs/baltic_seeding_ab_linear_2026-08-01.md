# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** linear

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✓ | 1.88e+03 | [13386.31245376308, 14637.1586782984] |
| cod_east | ✗ | ✓ | 2.75e+02 | [81069.5835901974, 83704.12604888651] |
| herring | ✓ | ✓ | 1.21e+06 | [2517908.805464496, 2624849.2152588572] |
| sprat | ✓ | ✓ | 3.11e+05 | [1041587.5300764805, 1068706.884873026] |
| flounder | ✓ | ✓ | 2.74e+04 | [39686.644922278145, 41818.87272725209] |
| perch | ✓ | ✓ | 1.18e+04 | [46262.67752327268, 48162.064773642676] |
| pikeperch | ✓ | ✗ | 3.42e+05 | [1381272.9347100125, 1473131.2222714354] |
| smelt | ✓ | ✗ | 4.72e+05 | [679994.9618181164, 706406.4491453588] |
| stickleback | ✓ | ✓ | 4.30e+04 | [77871.13614206715, 84073.45042468209] |

**Python verdict: 6/9 persistent & in-envelope.** Not 9/9 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
