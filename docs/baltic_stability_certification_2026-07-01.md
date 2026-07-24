# Baltic stability — SP-A certification

**Params:** /home/razinka/osmose/osmose-python/.sp-a-t6/stability_sweep.json  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod | ✓ | ✗ | 4.82e+06 | [15716441.434967095, 15898058.24019596] |
| herring | ✓ | ✗ | 3.85e+05 | [527081.1731505229, 648764.7370861474] |
| sprat | ✗ | ✗ | 5.43e+04 | [213656.6380716907, 351739.04335801146] |
| flounder | ✓ | ✗ | 7.41e+05 | [1019935.9954826359, 1303598.0112667894] |
| perch | ✓ | ✗ | 1.42e+05 | [2216457.536103319, 2338376.0623113634] |
| pikeperch | ✓ | ✗ | 2.41e+05 | [1596496.149761355, 1711517.4059294057] |
| smelt | ✓ | ✗ | 5.97e+05 | [1137536.0980793717, 1181386.7515492546] |
| stickleback | ✓ | ✗ | 6.39e+04 | [1010840.9065435992, 1275260.08504865] |

**Python verdict: 0/8 persistent & in-envelope.** Not 8/8 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).

**Java cross-check: 0/8 persistent (single seed).** Survivor sets AGREE with Python — Python ['cod', 'flounder', 'herring', 'perch', 'pikeperch', 'smelt', 'stickleback'], Java ['cod', 'flounder', 'herring', 'perch', 'pikeperch', 'smelt', 'stickleback']. Coarse consistency check only (Baltic is not bit-equal cross-engine); a DIFFER is a flag to inspect, not an automatic failure.
