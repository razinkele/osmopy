---
name: project_percid_overshoot_diagnostic
description: "2026-06-03 — PROVEN why Baltic perch/pikeperch overshoot ×100+; recruitment partly fixes pikeperch but not perch; CC-cap fix DEFERRED (weight-0.2). Diagnostic-first, no feature built."
metadata: 
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

Diagnostic-first investigation (during a brainstorm that concluded "document, don't build") into the Baltic perch (sp4) / pikeperch (sp5) overshoot that capped the strict ICES in-range count at 1–2/8 across the Shepherd (PR #50) and FR (PR-A/PR-B) features. Finding doc: `docs/baltic_percid_overshoot_diagnostic_2026-06-03.md` (committed to master `9dad350`, pushed). **Supersedes the inferred "unfixable spatial limit" caveat in the two prior verdict docs with proof.**

## Proven (β-probe: max Shepherd β=5.0 vs baseline, phase-13 base, FR-off, seed 42)
- Footprints already confined: perch adult 62 cells (10% of 616 ocean cells), pikeperch 27 (4.4%) — NOT over-broad, so map-concentration is a dead end. Seeded near target (30kt/15kt; targets 20kt/10kt) but explode ~100× at equilibrium.
- β=5.0: pikeperch ×127→×23 (−82%), perch ×166→×107 (−35% — FLOORS, can't reach range). Inter-annual CV explodes (perch 0.03→0.42, pikeperch 0.07→0.77 = over-compensation oscillation / paradox of enrichment). Cod/herring/sprat go WORSE (×33→×37.5, ×2.3→×3.9, ×5→×6.1 — capping percids frees prey).
- **Conclusion: pikeperch is recruitment-fixable (DE just never tried — left β stuck at 0.50/under-compensation because weight-0.2); perch is carrying-capacity-limited (β floor at ×107); brute-forcing recruitment destabilizes the system AND harms the high-weight stocks.** Neither prior doc captured this mix.

## Decision: CC-cap fix DEFERRED (do NOT build now)
The clean fix for perch is an opt-in per-species **spatial carrying-capacity cap** (density-dependent local mortality scaled to habitat capacity) — gentler than β-cranking, a general engine mechanism. Deferred: medium engine effort + a follow-on multi-hour calibration, headline payoff is two **weight-0.2** species, real risk to the sound high-weight fit, and it'd be the 3rd recruitment/predation/capacity mechanism in a row chasing the same low-weight species. Poor value-per-effort.

## Actionable takeaways (for future Baltic recalibration)
1. **Free/raise pikeperch β** (stuck at 0.50) — a MODERATE increase (not max — max → CV 0.77) cuts pikeperch toward range; easy partial win, but only inside a re-weighted run that guards the high-weight species from the freed-prey side effect.
2. **Build the spatial CC-cap only when a HIGH-WEIGHT species shows the same signature** (confined habitat + recruitment-β floor + CV blow-up under hard capping). That's when it earns its keep.
3. **Don't add a strict-in-range-count objective term** — it weights grid-under-resolved weight-0.2 pikeperch == weight-1.0 cod and would force the destabilizing behavior. The weighted objective correctly tolerates the percid overshoot.

See [[project_predator_functional_response]] and [[project_density_dependent_recruitment]] (the two features whose residual this explains).
