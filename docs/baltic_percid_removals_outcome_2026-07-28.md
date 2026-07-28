# Baltic percid missing-removals — outcome (2026-07-28)

Part B result. The grounded percid removals — realistic percid fishing F +
cormorant predation, both within documented magnitudes — **close the perch
overshoot without regressing any well-assessed stock**, and confirm that
pikeperch and smelt are structural residuals as the design predicted.

## Method

On the aggregate 8-species 5/8 baseline (branch `baltic-percid-removals` off
`646a36d`), two levers, both bounded to realistic magnitudes:
- **Lever A (fishing):** perch F = 0.40, pikeperch F = 0.50 — total commercial +
  recreational coastal removal (real coastal percid F ~0.3–0.6; recreational
  ≈/> commercial). Fixed, not optimised (see `percid_removal_provenance.md`).
- **Lever B (predation):** cormorant (sp15) biomass multiplier 2.0 + physiological
  ingestion 70/yr (≈70 kt/yr consumption, within the Hansson et al. 2018 ~100 kt
  bird budget) + a `Cormorant` predator column in the accessibility matrix shaped
  toward percids (perch 0.6 vs herring/sprat 0.15).

A cheap three-contrast feasibility gate (both / F-only / cormorant-only) first
confirmed each lever is causally responsible for the perch reduction (neither
masks the other); then a 50 yr × 5-seed certification confirmed the result. The
full 4–8 h DE re-calibration was **skipped by design** — percid objective weight
is 0.2, so it would not improve on (and could relax) these hand-grounded levers.

## Result — 50 yr × 5-seed certification

| species | baseline final-decade mean | **percid-removals mean** | envelope | in-env? |
|---|---|---|---|---|
| **perch** | ~110 kt (×2.2 over) | **44–46 kt** | 8–50 kt | **✓ FIXED** |
| pikeperch | ~2.2 Mt (×90) | 1.40–1.47 Mt (×58) | 4–25 kt | ✗ structural |
| smelt | ~641 kt (×5.3) | 686–703 kt (×5.8) | 20–120 kt | ✗ not targeted |
| cod | ~61–68 kt | 73–76 kt | 60–250 kt | ✓ |
| herring | ~2.2 Mt | 2.55–2.61 Mt | 0.8–3 Mt | ✓ |
| sprat | ~1.05 Mt | 1.13–1.16 Mt | 0.8–2.5 Mt | ✓ |
| flounder | ~43 kt | 42–45 kt | 20–100 kt | ✓ |
| stickleback | ~80 kt | 72–78 kt | 50–500 kt | ✓ |

- **In-envelope count: 5/8 → 6/8** (perch added). The five well-assessed stocks
  (cod, herring, sprat, flounder, stickleback) all stay in-envelope — **no
  regression**.
- **Strict persist-and-in-envelope count: 2/8 → 2/8** (herring, stickleback) —
  unchanged. The min-biomass dips (cod/sprat/flounder/perch) are the pre-existing
  boom-bust of the 5/8 baseline, not caused by the percid work.

## Interpretation — the honest bar was met exactly

- **Perch (~2×) is closable, and closed.** Both grounded levers contribute
  (gate: F-only 110→51 kt, cormorant-only 110→81 kt, both →41 kt); at the
  budget-realistic combined level the 5-seed mean lands at 44–46 kt, in-envelope.
  This is the design's predicted, defensible win.
- **Pikeperch (~90×) is NOT closable by these levers, and did not close.** It
  improved ~34% (×90 → ×58) — cormorants reach only *juvenile* pikeperch
  (size window 8.75–34 cm; pikeperch matures ~40 cm), and a ~90× overshoot is the
  coarse-grid habitat-carrying-capacity limit that additive mortality cannot
  supply. Exactly as pre-registered; not hidden, not over-cranked.
- **Smelt (~5×) is unchanged** (marginally up — lower perch releases a smelt
  predator). It is not a target of either lever; a separate issue.

## Scientific finding

The perch result is direct evidence that the model's percid over-prediction is
**partly a missing-removals artifact**, not purely coarse-grid structure: adding
the documented but previously-omitted coastal/recreational fishing and cormorant
predation brings perch into its ICES envelope with realistic, budgeted magnitudes.
Pikeperch's residual, by contrast, survives the maximum defensible grounded
removal — confirming it *is* the structural coarse-grid limit. This splits the
prior "percids are purely structural" verdict: perch was reachable, pikeperch is
not.

## Status

Certified (`docs/baltic_percid_removals_certification_2026-07-28.md`). Committed on
the `baltic-percid-removals` branch; ICES targets re-checked first
(`docs/baltic_ices_refresh_2026-07-28.md`). Deep-review-hardened plan:
`docs/superpowers/plans/2026-07-28-baltic-percid-removals.md`.
