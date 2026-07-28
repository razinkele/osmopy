# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✗ | ✓ | 4.76e+01 | [13064.41839380046, 14491.399005168194] |
| cod_east | ✗ | ✗ | 2.00e+01 | [57852.98459593181, 58879.38355622625] |
| herring | ✓ | ✓ | 1.00e+06 | [2528986.153591207, 2608087.0882612253] |
| sprat | ✗ | ✓ | 6.98e+04 | [1092763.232086734, 1113825.370750448] |
| flounder | ✗ | ✓ | 5.16e+02 | [41186.258384444554, 43013.569056820575] |
| perch | ✗ | ✓ | 1.64e+02 | [44148.060908581014, 46391.23955812784] |
| pikeperch | ✓ | ✗ | 2.87e+05 | [1413904.5128671075, 1508871.2870465866] |
| smelt | ✓ | ✗ | 6.16e+05 | [677733.3896146647, 691411.4318218695] |
| stickleback | ✓ | ✓ | 1.03e+04 | [75616.7066807755, 86790.31583288994] |

**Python verdict: 2/9 persistent & in-envelope.** Not 9/9 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
