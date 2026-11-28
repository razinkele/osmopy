# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** config default  ·  **engine:** Python

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✗ | 1.32e+04 | [13748.086251670315, 14104.55178772395] |
| cod_east | ✓ | ✗ | 3.19e+04 | [45376.976458109115, 46037.440408995244] |
| herring | ✓ | ✓ | 1.98e+06 | [2036819.2869522627, 2091616.6767473468] |
| sprat | ✓ | ✓ | 9.69e+05 | [1020258.98895564, 1033579.5988663022] |
| flounder | ✓ | ✗ | 3.60e+05 | [379740.8353002075, 396082.1559970215] |
| perch | ✓ | ✓ | 2.41e+04 | [25786.765097285315, 27171.13251216652] |
| pikeperch | ✓ | ✗ | 4.90e+05 | [519261.6095065576, 524480.1909016934] |
| smelt | ✓ | ✗ | 5.11e+05 | [544811.3761485631, 559463.5262786897] |
| stickleback | ✓ | ✗ | 1.48e+04 | [25549.705925927927, 31765.312657858805] |

**Python verdict: 2/5 ASSESSED species persistent & in-envelope.** Not 5/5 — SP-B gate: the failing ASSESSED species are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).

*Indicative tier: 1/4 (perch w=0.2, pikeperch w=0.2, smelt w=0.3, stickleback w=0.2).* These targets are **not ICES assessments** — ICES does not assess Baltic pikeperch, perch, smelt or stickleback. `biomass_targets.csv` sources them as literature estimates at weight ≤ 0.3, noting the coarse grid under-resolves species concentrated in estuaries and lagoons. They are reported for information and are **not** part of the verdict; do not tune against them. (Legacy all-species figure, for comparison with notes written before 2026-08-04: 3/9.)
