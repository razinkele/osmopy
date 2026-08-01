# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** stock_recruitment

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✓ | 1.26e+04 | [13623.167490394462, 15063.505577609434] |
| cod_east | ✓ | ✓ | 5.98e+04 | [82635.55664288499, 83364.89294014715] |
| herring | ✓ | ✓ | 2.42e+06 | [2565436.9653721577, 2616576.243130738] |
| sprat | ✓ | ✓ | 9.73e+05 | [1051155.6795747548, 1070012.1610680171] |
| flounder | ✓ | ✓ | 3.79e+04 | [39813.85153578645, 41189.69514252036] |
| perch | ✓ | ✓ | 3.96e+04 | [43541.69155599967, 46636.13512026587] |
| pikeperch | ✓ | ✗ | 1.11e+06 | [1339137.6211135923, 1461749.51076667] |
| smelt | ✓ | ✗ | 6.46e+05 | [676793.0815886308, 688088.2102327935] |
| stickleback | ✓ | ✓ | 6.31e+04 | [74742.00553012146, 80413.6770304197] |

**Python verdict: 7/9 persistent & in-envelope.** Not 9/9 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
