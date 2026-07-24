# RV recruitment gate — validation on the 5/8 baseline (2026-07-25)

Task 3 of `docs/superpowers/plans/2026-07-24-baltic-rv-recruitment-gate.md`. The RV
gate is now enabled for cod (sp0, `mean_preserving`) on the committed config; this is
the gated-vs-pre-gate comparison and the go/no-go for the cod E/W disaggregation
Phase 1.

## Result — 50 yr × 5 seeds, worst-case-across-seeds (`baltic_stability_certify.py`)

| species | pre-gate final-decade mean | **RV-gated final-decade mean** | verdict |
|---------|----------------------------|-------------------------------|---------|
| cod | ~167 kt (×1.4) | **~64 kt (×0.53, low edge of envelope)** | COLLAPSE (min 2.4 kt, both) |
| herring | in-range | in-range | PASS |
| sprat | ~910 kt | ~1,047 kt | COLLAPSE (min dip) |
| flounder | ~39 kt | ~43 kt | COLLAPSE (min dip) |
| pikeperch | ~1,893 kt | ~2,093 kt | persists, over-target |
| smelt | ~630 kt | ~641 kt | persists, over-target |
| stickleback | in-range | in-range | PASS |
| **count** | **2/8** | **2/8** | unchanged |

## Interpretation

1. **The gate works and the effect is realistic.** `mean_preserving` preserves *mean
   recruitment*, but recruitment **variability lowers mean biomass** (Jensen's inequality
   on the nonlinear stock-recruitment response). Cod fell from ×1.4 (167 kt) to ~64 kt —
   right at the documented eastern-Baltic-cod **post-collapse SSB (~60–77 kt)**. The
   mechanism the SP-A branch identified is confirmed on the calibrated baseline.
2. **Trophic release** — lower cod raises sprat/flounder/pikeperch (less cod predation),
   as expected for a top predator being suppressed.
3. **Stability is NOT resolved (2/8).** cod still dips to min 2.4 kt in some seed; the RV
   variability did not dampen the boom–bust. Long-horizon stability needs more than
   recruitment variability on the aggregated cod.

## Go/no-go for Phase 1 (cod E/W disaggregation)

- **Use `raw_cap` mode (not `mean_preserving`) for eastern cod.** `mean_preserving`
  reduces cod via variability but cannot *deterministically* drive the eastern collapse;
  `raw_cap` (factor = clip(rv/ref, 0, 1)) caps recruitment in low-RV years, which — paired
  with the elevated M the fidelity review requires — is the lever for the eastern-collapse
  structure. The `mean_preserving` result here is the realism baseline; `raw_cap` is the
  collapse driver.
- **The stability gap is a disaggregation target, not an RV-gate failure.** A collapsed
  *eastern* stock + a persistent *western* stock is a different (and more stable-on-average)
  structure than the single dipping cod — Phase 1 tests whether the split itself, plus
  eastern-cod `raw_cap` + M, closes the gap.

## Status

RV gate: **built (reconciled), data + config + validated.** Left enabled
(`mean_preserving`) on the committed config as the realism baseline. Phase 1 will switch
eastern cod to `raw_cap`. See `docs/baltic_stability_certification_2026-07-01.md` for the
raw per-species table.
