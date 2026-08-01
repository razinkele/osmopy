# Seeding mode A/B — `stock_recruitment` vs `linear` (GitHub #143)

**Date:** 2026-08-01 · 50 yr × 5 seeds (42, 123, 7, 999, 2024) · `--params current` · **identical
parameters in both arms**, so the seeding conversion is the only variable.

| species | SR mean | **linear mean** | mean Δ | SR min | **linear min** | min ratio |
|---|---|---|---|---|---|---|
| cod_west | 13,623–15,064 | 13,386–14,637 | −2% | 47 | **1,880** | **40×** |
| cod_east | 82,636–83,365 | 81,070–83,704 | −1% | 17 | **275** | **16×** |
| herring | 2,565–2,617 k | 2,518–2,625 k | −1% | 1.00 M | 1.21 M | 1.2× |
| sprat | 1,051–1,070 k | 1,042–1,069 k | −1% | 70,400 | **311,000** | **4.4×** |
| flounder | 39,814–41,190 | 39,687–41,819 | ~0% | 518 | **27,400** | **53×** |
| perch | 43,542–46,636 | 46,263–48,162 | +5% | 158 | **11,800** | **75×** |
| pikeperch | 1,339–1,462 k | 1,381–1,473 k | +2% | 301 k | 342 k | 1.1× |
| smelt | 677–688 k | 680–706 k | +2% | 602 k | 472 k | 0.8× |
| stickleback | 74,742–80,414 | 77,871–84,073 | +4% | 10,400 | **43,000** | **4.1×** |
| **verdict** | **2/9** | **6/9** | | | | |

## The headline is misleading; read the two columns separately

**Final-decade means are effectively unchanged** — within ±5% for every species, mostly ±2%. The
50-year equilibrium does not care which seeding conversion was used.

**Minima are 4–75× higher under `linear`.** That is the whole story. `linear` seeds more eggs, so the
seeding bootstrap never dips as close to zero. Since `persists` is defined as
`min > 0.1 × envelope-lower`, five species clear the bar under `linear` that failed under
`stock_recruitment` — on the strength of the transient, not the biology.

**So 2/9 → 6/9 is a statement about the initialisation transient, not about ecological outcome.**
Reading the pass count alone would badly overstate the effect.

## What this means for #143

* The seeding conversion has a **large effect on the seeding bootstrap** and a **negligible effect on
  the 50-year equilibrium** of this config.
* Prior `COLLAPSE` verdicts in Baltic certifications are substantially **artifacts of the
  `stock_recruitment` seeding transient**, not evidence of ecological instability. `cod_east` dipping
  to 17 t before settling at ~83 kt in-envelope is a bootstrap artifact.
* For cross-engine work, `linear` is the mode under which Python and Java are comparable at all (Java
  always seeds linearly, and the key is Python-only).

## Limits — read before citing

1. **Parameters were calibrated under `stock_recruitment`.** `linear` is scored on a fit made for its
   rival. It nonetheless scores higher, which strengthens rather than weakens the result — but this is
   *not* the converged A/B originally planned. The refit was abandoned after ~16 h of compute failed to
   complete a single measurable generation (each 40-yr evaluation costs ~189 s solo and more under
   8-way contention; 25-generation checkpoints were never reached).
2. **`persists` conflates transient and equilibrium.** The threshold tests the run minimum, which is
   dominated by the seeding bootstrap. A criterion evaluated over the final decade only would score
   these two arms almost identically.
3. Single config, single parameter set. No claim about other configs.
