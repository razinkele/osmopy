# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✗ | 4.19e+05 | [692530.2721846086, 729691.8737346148] |
| cod_east | ✓ | ✗ | 7.78e+05 | [4980742.719341582, 5252689.708965456] |
| herring | ✓ | ✓ | 5.91e+05 | [1717457.3675118568, 1870237.0096850265] |
| sprat | ✓ | ✗ | 1.38e+06 | [3190239.758309221, 3567439.1366434684] |
| flounder | ✓ | ✗ | 9.46e+05 | [2203466.6409569383, 2410537.036324653] |
| perch | ✓ | ✗ | 6.59e+03 | [375379.9426502563, 439744.8126430282] |
| pikeperch | ✓ | ✗ | 1.79e+05 | [1795671.9794822522, 2004570.5805345438] |
| smelt | ✓ | ✗ | 8.36e+04 | [147475.37441420386, 172315.07668159326] |
| stickleback | ✓ | ✓ | 9.48e+03 | [55347.343492591914, 78906.86798621393] |

**Python verdict: 2/9 persistent & in-envelope.** Not 9/9 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
