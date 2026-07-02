---
name: Phase 12 with B-H wiring (active configuration)
description: 2026-04-29 — phase 12 calibration is now wired with v0.11.0 Beverton-Holt active. 27 free params (was 24); cod sp0 ssb_half FIXED at 120 kt (Bpa per ICES cod.27.24-32); flounder/perch/pikeperch ssb_half DE-tunable; cod adult-mortality floor REVERTED — B-H replaces it.
type: project
originSessionId: af3b28b2-0438-47e9-8b63-2b06b1debe34
---
**Status:** Live calibration kicked off 2026-04-28 18:00 at SHA 816d5c1, still running 2026-04-29 ~16:00 (21h+ elapsed, 3,765 evals at 175 evals/h). Watcher armed; results land in `data/baltic/calibration_results/`.

**Configuration (commits 92088c9 + 816d5c1):**
- 27 DE params = 16 mortality + 8 fishing + 3 B-H ssb_half (sp3/4/5)
- Cod (sp0): `stock.recruitment.type.sp0=beverton_holt`, `ssbhalf=120000` t. Held FIXED at the literature prior — Bpa per ICES cod.27.24-32 (Blim=109, Bpa=122 kt). Not in DE param list.
- Flounder (sp3): DE-tuned, log10 bounds (3.7, 5.3) → 5–200 kt. Wide because OSMOSE's lumped flounder is a sum across 4 ICES sub-stocks; the prior 5-50 kt was too narrow at the upper bound.
- Perch (sp4) and Pikeperch (sp5): DE-tuned, log10 bounds (2.7, 4.7) → 0.5–50 kt. No ICES assessment for Curonian Lagoon stocks; fit against simulated trajectory.
- Other species (herring, sprat, smelt, stickleback): `type=none`, less recruitment-limited.

**Why:** B-H closes the structural compensation pathway DE was exploiting. The 2026-04-27 cod-floor run confirmed: forcing higher cod adult mortality just made DE drop larval mortality 24× to preserve cod recruitment. With B-H capping recruitment at high SSB, this no longer pays off.

**How to apply:**
- The cod-floor (sp0 adult mortality bound `(-0.523, 0.7)`) is REVERTED to `(-3.0, 0.7)` in master at 816d5c1. Do not re-introduce; B-H replaces it as the structural fix.
- For phase 12 calibration, the 27-param search is the canonical setup. Older 24-param results (before 816d5c1) are not directly comparable.
- Bg-predation length fix landed at 92088c9 — required for sp14/sp15 (seal/cormorant) to actually exert predation pressure. Do not run phase 12 against any SHA earlier than 92088c9 if predator effects matter.
- Warm-start from prior `phase12_results.json` works for the 24 mortality+fishing params; the 3 ssb_half params have no prior so they default. Always pass `--skip-warm-start-keys mortality.additional.rate.sp0` — the old optimum (log10≈0.57) sat against the cod-floor ceiling and biases the new (-3.0, 0.7) search.
