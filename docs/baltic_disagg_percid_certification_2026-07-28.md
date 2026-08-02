# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✗ | ✓ | 4.61e+01 | [13984.859404640749, 14475.543887443677] |
| cod_east | ✗ | ✗ | 8.67e-26 | [5.24575856893824e-21, 2.6311171231418165e-20] |
| herring | ✓ | ✓ | 1.01e+06 | [2504333.1722085555, 2631763.9344816348] |
| sprat | ✗ | ✓ | 6.87e+04 | [1229686.8531640605, 1248274.2257682388] |
| flounder | ✗ | ✓ | 4.98e+02 | [43634.96785788222, 46238.615072729255] |
| perch | ✗ | ✓ | 1.62e+02 | [42746.52070661611, 44628.25711275684] |
| pikeperch | ✓ | ✗ | 2.89e+05 | [1390889.6078372293, 1509987.5867181397] |
| smelt | ✓ | ✗ | 6.07e+05 | [681351.1794814431, 693122.2220963547] |
| stickleback | ✓ | ✓ | 1.04e+04 | [75796.51688623366, 78173.01058521224] |

~~**Python verdict: 2/9 persistent & in-envelope.**~~
> **Corrected 2026-08-02 — the verdict and the SP-B candidate list are both wrong as published.**
> The `persists` column above used the **whole-run** minimum, dominated by the Baltic seeding
> bootstrap; `556ba3d` rescoped it to the final decade. Every row reading `persists ✗` /
> `in-envelope ✓` is that artifact. Re-read: ****6/9****, with `cod_east` a **real** collapse (final-decade mean ~1e-20, twenty orders below its envelope floor — not an artifact) plus `pikeperch`/`smelt` over envelope.
> The SP-B candidate list shrinks from 7 species to 3. Full re-derivation:
> `docs/baltic_spb_attribution_rederivation_2026-08-02.md`. (A re-read of the committed table under
> the audit's rule, not a fresh run.)
 Not 9/9 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
