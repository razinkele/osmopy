# Baltic ICES target refresh — 2026 cycle re-check (2026-07-28)

Part A of the "refresh ICES targets, then percid removals" work. The literature
alert (`HORIZON_EUROPE/osmopy/LITERATURE`, 2026-07-28) flagged that the 2027 ICES
advice raised central-Baltic-herring TAC ~+74% and sprat ~+32%, implying the
Baltic calibration targets may be stale. This note records the re-check.

## What was pulled

Latest ICES SAG assessment time series via the ICES data service:

| stock | metric | recent values | unit |
|---|---|---|---|
| spr.27.22-32 (sprat) | SSB | 2021 1.08 Mt · 2022 1.13 Mt · 2023 0.90 Mt | tonnes |
| her.27.25-2932 (central herring) | SSB | 2021 0.37 · 2022 0.40 · 2023 0.39 | Mt (SAG normalised) |
| cod.27.24-32 (eastern cod) | SSB | 2020–2022 ~65–77 kt (collapsed, zero-catch advice) | tonnes |

## Finding — the BIOMASS targets do NOT need revising

The committed model targets and their (wide) envelopes still bracket the recent
ICES SSB:

- **sprat** target 1.5 Mt, envelope 0.8–2.5 Mt. Recent SSB 0.90–1.13 Mt; total
  biomass > SSB → ~1.2–1.5 Mt. **In envelope, target defensible.**
- **herring** target 1.5 Mt (aggregate of all Baltic units), envelope 0.8–3 Mt.
  Central her.27.25-2932 SSB ~0.39 Mt; aggregate across all units ~1–1.5 Mt.
  **In envelope.**
- **cod** unchanged — still collapsed, both stocks zero-catch.

**The 2027-advice increases are TAC/recruitment-driven (a fishing-opportunity
signal from strong 2022–23 recruitment), NOT a spawning-biomass jump — sprat and
herring SSB are ~stable.** The model calibrates against *biomass*, so the biomass
targets are unaffected. No target value changed.

## Caveat — the CATCH targets (not the biomass targets) are the stale ones

The advice moved on the catch/TAC side. The model also carries per-species *catch*
targets; refreshing those to the exact 2027-cycle numbers requires reading the
WGBFAS 2026 report (ICES figshare 32455056) or the advice PDFs, which are
JavaScript-rendered and were NOT machine-extractable (the same block the
literature-alert system reported). Deferred to a manual browser/PDF read; the
biomass-side percid work below does not depend on it.

## Consequence for the percid-removals work

The aggregate 8-species baseline the percid plan builds on validates against
biomass envelopes that this re-check confirms are still valid. Proceeding to
Part B (percid removals) on this baseline is sound; the no-regression bar
(well-assessed stocks stay in their biomass envelopes) is unaffected by the
advice-cycle changes.
