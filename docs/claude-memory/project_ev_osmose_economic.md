---
name: Ev-OSMOSE + Economic modules — SHIPPED
description: Ev-OSMOSE genetics + DSVM fleet dynamics both shipped. Implementation location, toggle keys, state layout, pointer to original design spec.
type: project
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
Ev-OSMOSE (eco-evolutionary genetics) + DSVM fleet dynamics (bioeconomic) shipped as Front 6 of the 2026-04-18 post-session roadmap — included in v0.9.0 + v0.9.1.

**Why they exist:** these were the two remaining major Java OSMOSE features not yet ported. Ev-OSMOSE adds diploid polygenic genetics; DSVM adds spatially-explicit fleet dynamics with logit-based vessel decision-making.

**Design + plan artifacts:**

- Spec: `docs/superpowers/specs/2026-04-05-ev-osmose-economic-design.md`
- MVP plan (STATUS-COMPLETE): `docs/superpowers/plans/2026-04-06-ev-osmose-economic-mvp-plan.md`
- Genetics core plan (STATUS-COMPLETE): `docs/superpowers/plans/2026-04-06-ev-osmose-genetics-core-plan.md`
- Economics core plan (STATUS-COMPLETE): `docs/superpowers/plans/2026-04-06-ev-osmose-economics-core-plan.md`

**How to apply when modifying these modules:**

- Toggles: `simulation.genetic.enabled` / `simulation.economic.enabled`.
- State lives on `SimulationContext` (NOT `SchoolState`) — fields `genetic_state` and `fleet_state`. Adding per-school genetic fields would force None-guards at every school iteration site.
- Packages: `osmose/engine/genetics/` and `osmose/engine/economics/`.
- Economics end-of-run dispatch: `write_economic_outputs` is called from `simulate(output_dir=...)` — this is the wire-up landed in `1dcffd7` as part of v0.9.1. Don't re-wire it.
- Phase 2 scope delivered: 4 bioen traits wired, neutral loci, seeding phase, full cost model, memory, days-at-sea, CSV output.
