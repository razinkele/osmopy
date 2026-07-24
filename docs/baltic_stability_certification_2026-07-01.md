# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod | ✗ | ✓ | 2.39e+03 | [166922.1210381022, 169818.651992529] |
| herring | ✓ | ✓ | 9.91e+05 | [2158539.3287608675, 2344137.36198984] |
| sprat | ✗ | ✓ | 6.99e+04 | [906589.366708838, 931999.6135540766] |
| flounder | ✗ | ✓ | 1.07e+03 | [38504.57757787097, 40050.46968671007] |
| perch | ✗ | ✗ | 1.88e+02 | [107396.11383016889, 113433.27818321464] |
| pikeperch | ✓ | ✗ | 3.05e+05 | [1893917.9516601022, 2217985.7953248294] |
| smelt | ✓ | ✗ | 5.46e+05 | [623558.8707051948, 643670.1556188915] |
| stickleback | ✓ | ✓ | 9.28e+03 | [72595.34500504777, 79577.92419387779] |

**Python verdict: 2/8 persistent & in-envelope.** Not 8/8 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
