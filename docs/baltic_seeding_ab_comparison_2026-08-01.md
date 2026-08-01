# Seeding mode A/B — `stock_recruitment` vs `linear` (GitHub #143)

**Date:** 2026-08-01 · 50 yr × 5 seeds (42, 123, 7, 999, 2024) · `--params current` · **identical
parameters in both arms**, so the seeding conversion is the only variable. Scored under the corrected
`persists` criterion (`556ba3d`), which measures the final decade rather than the whole run.

## Result: the arms are indistinguishable — both 7/9

| species | SR mean | linear mean | Δ | verdict (both) |
|---|---|---|---|---|
| cod_west | 13,623–15,064 | 13,386–14,637 | −2% | PASS |
| cod_east | 82,636–83,365 | 81,070–83,704 | −1% | PASS |
| herring | 2,565–2,617 k | 2,518–2,625 k | −1% | PASS |
| sprat | 1,051–1,070 k | 1,042–1,069 k | −1% | PASS |
| flounder | 39,814–41,190 | 39,687–41,819 | ~0% | PASS |
| perch | 43,542–46,636 | 46,263–48,162 | +5% | PASS |
| pikeperch | 1,339–1,462 k | 1,381–1,473 k | +2% | over envelope |
| smelt | 677–688 k | 680–706 k | +2% | over envelope |
| stickleback | 74,742–80,414 | 77,871–84,073 | +4% | PASS |
| **verdict** | **7/9** | **7/9** | | |

Every species agrees within ±5%, mostly ±2%, and the two failures are identical in both arms:
`pikeperch` and `smelt` fail **`in_envelope`** — the known percid overshoot — not persistence.

## Conclusion for #143

**The seeding conversion is immaterial to the 50-year equilibrium of this config.** Whatever the
bootstrap does, the model converges to the same place.

So #143 is a **parity-and-tooling decision, not a scientific one**. The case for keeping `linear`
available is that `population.seeding.mode` is Python-only and Java always seeds linearly, so
`--seeding-mode linear` is the only setting under which the two engines are comparable at all. The
case for `stock_recruitment` as default is that it is dynamically consistent — the seeded population
enters through the same relationship that governs it thereafter — and it costs nothing at equilibrium.

Both are defensible; the choice does not change the science of this config.

## What the A/B actually contributed

Not an answer about seeding conversions — **it exposed a defective certification criterion.**

Under the *old* `persists` (minimum over the whole run) the same two arms scored **2/9 vs 6/9**, a
difference driven entirely by seeding-transient minima: `linear` seeds more eggs, so the bootstrap dips
less deeply, and five species cleared `min > 0.1 × envelope-lower` on the transient alone. The
final-decade means were already within ±5% then — the signal was in the wrong window.

That prompted `556ba3d`, which scoped the minimum to the final decade. Re-scoring both arms:

| | old criterion | corrected |
|---|---|---|
| `stock_recruitment` | 2/9 | **7/9** |
| `linear` | 6/9 | **7/9** |

**Retrospective consequence.** Prior `COLLAPSE` verdicts in Baltic certifications were substantially
seeding-transient artifacts. `cod_east` dipping to 17 t before settling at ~83 kt *inside* its envelope
was reported as a collapse; under the corrected criterion its final-decade minimum is 58–60 kt and it
passes. **The Baltic baseline was never "2/9 collapsing" — it is 7/9, with two species over envelope
and none collapsing at equilibrium.** Stability conclusions resting on the old persistence flags should
be re-read.

## Limits

1. **Neither arm was refitted.** Parameters were calibrated under `stock_recruitment`; `linear` is
   scored on its rival's fit. It matches anyway, which strengthens the "immaterial at equilibrium"
   reading, but this is not the converged A/B originally planned. The refit was abandoned after ~16 h
   of compute failed to complete one measurable generation (~189 s per 40-yr evaluation solo, worse
   under 8-way contention).
2. Single config, single parameter set. No claim about other configs or about the transient itself,
   which genuinely does differ between the modes.
