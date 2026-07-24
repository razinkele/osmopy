# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod | ✗ | ✓ | 2.39e+03 | [60931.06760986075, 68364.12014491015] |
| herring | ✓ | ✓ | 9.91e+05 | [2210504.0450091357, 2300439.2797759203] |
| sprat | ✗ | ✓ | 6.99e+04 | [1046624.7253830075, 1087010.9384440961] |
| flounder | ✗ | ✓ | 1.07e+03 | [43205.75589590842, 43858.63633645097] |
| perch | ✗ | ✗ | 1.88e+02 | [108471.61151806972, 112765.67292445777] |
| pikeperch | ✓ | ✗ | 3.05e+05 | [2093201.1828933074, 2380675.652803406] |
| smelt | ✓ | ✗ | 5.41e+05 | [636017.5617750797, 646342.7859896086] |
| stickleback | ✓ | ✓ | 9.28e+03 | [80313.21355746567, 83202.80909677768] |

**Python verdict: 2/8 persistent & in-envelope.** Not 8/8 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
