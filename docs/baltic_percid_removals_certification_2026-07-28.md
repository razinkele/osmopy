# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod | ✗ | ✓ | 1.37e+03 | [73114.11973047431, 75695.44072217045] |
| herring | ✓ | ✓ | 1.02e+06 | [2549203.373078829, 2606404.6337740975] |
| sprat | ✗ | ✓ | 7.00e+04 | [1133709.8079841468, 1159068.7776254124] |
| flounder | ✗ | ✓ | 5.91e+02 | [42472.92109559917, 44511.45887544898] |
| perch | ✗ | ✓ | 1.58e+02 | [43908.069341189366, 45944.986994788196] |
| pikeperch | ✓ | ✗ | 3.02e+05 | [1401049.9073134414, 1469230.9551866227] |
| smelt | ✓ | ✗ | 6.11e+05 | [685609.7167253016, 703025.3725883933] |
| stickleback | ✓ | ✓ | 9.30e+03 | [71709.56088369185, 78183.62555475216] |

~~**Python verdict: 2/8 persistent & in-envelope.**~~
> **Corrected 2026-08-02 — the verdict and the SP-B candidate list are both wrong as published.**
> The `persists` column above used the **whole-run** minimum, dominated by the Baltic seeding
> bootstrap; `556ba3d` rescoped it to the final decade. Every row reading `persists ✗` /
> `in-envelope ✓` is that artifact. Re-read: ****6/8****, with only `pikeperch` and `smelt` genuinely failing, both on `in_envelope` (over) rather than persistence.
> The SP-B candidate list shrinks from 6 species to 2. Full re-derivation:
> `docs/baltic_spb_attribution_rederivation_2026-08-02.md`. (A re-read of the committed table under
> the audit's rule, not a fresh run.)
 Not 8/8 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
