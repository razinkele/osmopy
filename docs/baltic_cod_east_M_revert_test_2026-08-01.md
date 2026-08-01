# Baltic stability — SP-A certification

**Params:** /tmp/claude-1000/-home-razinka-osmopy/df645e89-e71d-44fb-920e-eff75f51187b/scratchpad/m_original.json  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** config default

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✓ | 1.28e+04 | [13146.982637366184, 14644.21407586135] |
| cod_east | ✓ | ✗ | 7.50e+03 | [14519.928934916315, 15077.0120831718] |
| herring | ✓ | ✓ | 2.39e+06 | [2518780.932668591, 2600310.6952673895] |
| sprat | ✓ | ✓ | 1.15e+06 | [1195482.6851419997, 1211963.1424261588] |
| flounder | ✓ | ✓ | 3.93e+04 | [42005.53346053111, 43803.48638216245] |
| perch | ✓ | ✓ | 4.02e+04 | [42764.49166408322, 46152.89454331895] |
| pikeperch | ✓ | ✗ | 1.35e+06 | [1425260.17735624, 1493300.211447803] |
| smelt | ✓ | ✗ | 6.37e+05 | [676657.9089599829, 695688.9728198219] |
| stickleback | ✓ | ✓ | 5.91e+04 | [72788.63874959138, 82265.13201977787] |

**Python verdict: 6/9 persistent & in-envelope.** Not 9/9 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
