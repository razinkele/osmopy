# Baltic stability — SP-A certification

**Params:** /home/razinka/osmose/osmose-python/.sp-a-t6/stability_sweep.json  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod | ✓ | ✗ | 4.82e+06 | [15749925.178955559, 15934453.453784931] |
| herring | ✓ | ✗ | 3.97e+05 | [687031.3966152864, 819441.6640698768] |
| sprat | ✓ | ✗ | 8.78e+04 | [260415.5622194665, 332141.365403243] |
| flounder | ✓ | ✗ | 8.76e+05 | [1196947.0428230104, 1410100.3635183019] |
| perch | ✓ | ✗ | 1.42e+05 | [2187562.448948998, 2240323.239749864] |
| pikeperch | ✓ | ✗ | 2.52e+05 | [1721286.456411571, 1803797.37708656] |
| smelt | ✓ | ✗ | 5.97e+05 | [1184640.4305406087, 1249273.469735099] |
| stickleback | ✓ | ✗ | 6.39e+04 | [1254826.8110701344, 1317697.1541430214] |

**Python verdict: 0/8 persistent & in-envelope.** Not 8/8 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
