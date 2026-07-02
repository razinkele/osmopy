---
name: Beverton-Holt stock-recruitment SHIPPED
description: 2026-04-28 — Beverton-Holt + Ricker SR shipped on master as v0.11.0 (commits c6b1b2f → a24090d). Multiplicative correction at egg stage; ssb_half is the single shape parameter. Default type=none preserves Java parity.
type: project
originSessionId: 64a9ab9f-e4dd-45e2-b8ec-af84be8cf68c
---
**Status:** SHIPPED on master 2026-04-28. Tag v0.11.0. Plan executed via worktree `beverton-holt-sr` (now merged); 27/27 SR tests green; ruff clean. The 2026-04-29 08:57 cron job is now redundant for execution — it should re-run Phase 12 calibration with B-H active instead.

**What landed:**
- Schema: `stock.recruitment.type.sp{idx}` ∈ {none, beverton_holt, ricker}, `stock.recruitment.ssbhalf.sp{idx}` (tonnes). `osmose/schema/species.py:246-269`.
- Helper: `apply_stock_recruitment()` in `osmose/engine/processes/reproduction.py:15-59`. Multiplicative correction over linear `n_eggs`. B-H: `linear / (1 + SSB/ssb_half)`. Ricker: `linear * exp(-SSB/ssb_half)`.
- Wiring: `reproduction()` calls helper after computing `n_eggs_linear`. `recruitment_type[:n_sp]` slice keeps focal-only; background species get type="none" via `_merge_focal_background`.
- Validation: type!=none with ssb_half<=0 raises ValueError at `EngineConfig.from_dict` time.
- Docs: `docs/parity-roadmap.md` Post-parity divergences section; `CHANGELOG.md` entry under [Unreleased] (v0.11.0).

**Why:** Phase 12 cod-floor calibration (2026-04-27) showed DE compensates for forced adult-mortality bounds by dropping larval mortality 24× to preserve cod recruitment. Density-dependent SR is the only structural fix.

**How to apply:** When re-running Phase 12 calibration, expose `stock.recruitment.ssbhalf.sp{0,3,4,5}` (cod/flounder/perch/pikeperch) as DE parameters with biologically-informed priors. Set `type=beverton_holt` for those four. Leave clupeids/stickleback/smelt at type=none (less recruitment-limited).

**ICES priors (verified 2026-04-28 via ICES MCP):**
- **Cod (cod.27.24-32):** Blim=109 kt, Bpa=122 kt. Modern SSB 52-153 kt; historical max 457 kt (1980). Plan's prior 50-250 kt brackets Blim/Bpa appropriately.
- **Sprat (spr.27.22-32):** Blim=459 kt, Bpa=541 kt, MSY-Btrigger=541 kt. Forage-fish scale.
- **Herring (her.27.25-2932):** reference points are relative indices (Blim=0.5, Bpa=1.0), not absolute tonnes — would need full assessment series to set absolute ssb_half.
- **Flounder:** No single Baltic-wide stock; ICES splits into fle.27.{2223, 2425, 2628, 2729-32}. OSMOSE's lumped Baltic flounder needs ssb_half summed across all four sub-stocks. Plan's "5-50 kt" likely too narrow at the upper bound — verify against the four sub-stock SSB sum before re-calibrating.
- **Perch / pikeperch (Curonian Lagoon):** not assessed by ICES (HELCOM/national management). Wide priors (1-20 kt) appropriate; tune against simulated trajectory rather than literature point estimates.

**Per-step semantic (deliberate):** B-H/Ricker is applied per spawning event, not per year. For multi-season spawners (herring spring + autumn), each event sees current SSB and gets its own density-dependent reduction. Total annual recruitment = sum across events. Means literature α/β values calibrated against annual ICES SR pairs need reinterpretation as per-event for multi-season species.

**Mathematical sanity (verified):**
- B-H: at SSB=ssb_half, eggs = linear/2 (and = asymptote/2). Asymptote at high SSB = α·ssb_half. ✓ test_beverton_holt_at_half_saturation, test_beverton_holt_asymptote.
- Ricker: peak at SSB=ssb_half, with eggs(ssb_half) = α·ssb_half·exp(-1) ≈ 0.368·linear(ssb_half). ✓ test_ricker_at_peak.
- Both forms reduce to linear at SSB << ssb_half (Java parity preserved at low density). ✓ test_beverton_holt_low_ssb_approaches_linear.

**Plan version-target bug (corrected at execute time):** Plan Task 7.3 said "bump 0.9.3 → 0.10.0", but origin already had v0.10.0 at d4eebe1 (calibration speedup, 2026-04-21). Executor caught the conflict and corrected to 0.11.0 (commit a24090d, message: "version: bump 0.10.0 -> 0.11.0 (origin already has v0.10.0 tag at d4eebe1)"). No action needed; lesson: plans written in offline worktrees should re-verify version state against `git tag --sort=-v:refname` before specifying targets.
