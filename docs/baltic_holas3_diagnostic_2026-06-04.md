# HELCOM HOLAS-3 fish status vs the OSMOSE Baltic model — diagnostic

**Date:** 2026-06-04
**Status:** Closed via diagnostic — reference snapshot frozen, no validator built.
**Snapshot:** `data/baltic/reference/helcom_snapshots/` (pulled from the live HELCOM
ArcGIS service; refresh via `scripts/_pull_helcom_snapshots.py`).

## Question

Can we validate OSMOSE Baltic model outputs against HELCOM HOLAS-3 the way we already
validate them against ICES SAG (`osmose/validation/ices.py`) — i.e. a per-species,
quantitative, in-range / out-of-range check?

## Answer: no — and the reason is intrinsic to the HOLAS-3 fish data, not a tooling gap

HOLAS-3 fish exposes only a **0-1 quality ratio**, not an absolute target, and only at a
**two-bucket guild granularity** (DEM = demersal, PEL = pelagic), never per species. On top
of that, across the entire model domain the status is **uniformly sub-GES**, which removes
the last bit of discriminating signal a cross-check could have used. Both facts were
established by querying the live service (HELCOM ArcGIS `MADS/Indicators_and_assessments/
MapServer`, layers 434 commercial + 417 coastal), not inferred.

### Finding 1 — HOLAS-3 fish has no per-species quantitative target

- **Commercial fish (layer 434):** one record per ICES subdivision, fields `DEM`, `PEL`,
  `BQR` (= mean of DEM & PEL), each a 0-1 ratio. Cod, sprat, herring, flounder are collapsed
  into the two DEM/PEL buckets — there is no cod number, no sprat number, no tonnes.
- **Coastal fish (layer 417):** one aggregated indicator, "Integrated abundance coastal fish
  species" — an `EQR` (0-1) + `Status` (`Achieve`/`Fail`) per sub-basin. Perch and pikeperch
  are **not** broken out; the indicator is a single aggregate.
- The absolute baselines that define EQR/BQR = 1.0 (and the GES boundary) live only in
  narrative HOLAS-3 PDF reports — **not** as fetchable structured fields.

Contrast ICES SAG: SSB in tonnes per stock, directly comparable to model biomass. HOLAS-3
cannot support that comparison for **any** species. (Consistent with the existing project
finding that even on the ICES side, only sprat has solid Baltic tonnes coverage.)

### Finding 2 — the whole model domain is uniformly sub-GES (so a cross-check is signal-free)

Per-guild domain means over SD 22-32 (12 records; SD 28 split 28.1/28.2; GES boundary 0.6):

| Guild | mean BQR | SDs at GES (BQR ≥ 0.6) |
|---|---|---|
| Demersal (DEM) | **0.345** | **0 / 12** |
| Pelagic (PEL) | **0.306** | **1 / 12** (only SD 31, Bothnian Bay) |

Both guilds sit far below the 0.6 boundary across essentially the entire domain. A
"directional consistency" cross-check (does the model's guild biomass trend in the direction
HOLAS-3's status implies?) was considered and **rejected**: because HOLAS-3 "concern" is
constant-true for both guilds, such a check degenerates to the **sign of the model's own
within-run trend** — delete the HOLAS-3 snapshot and the verdict is unchanged. That is
numerology, not validation. (This mirrors the project's earlier re-pricing discipline: the
fisheries Kobe/B-Bmsy rescope and the percid close-without-building.)

Three further reasons the rejected cross-check was unsound, for the record:
1. **Different systems, different clocks.** HOLAS-3 BQR is the *real* Baltic's recent decadal
   status; a model run's within-window trend is the *simulation's* transient toward *its* own
   equilibrium under *its* forcing. The two are not commensurable.
2. **Garbage-in on the guild sum.** With cod overshooting ×17-48 (per prior calibration
   diagnostics), the demersal-guild biomass and the pelagic:demersal ratio are governed by a
   calibration artifact, not demersal ecology.
3. **DEM = 0 ≠ failing.** SD 30 & 31 (Gulf of Bothnia) report `DEM = 0` because there is no
   demersal commercial fishery to assess there — averaging those as "zero status" would
   misstate demersal status for a non-ecological reason.

### Finding 3 — there *is* real spatial structure, but the model can't resolve it

The data shows a clear gradient: demersal status is relatively higher in the southern/central
basins (SD 24-29, DEM up to 0.525) and collapses to 0 in the Bothnian north, while pelagic
status is best in the north (SD 31 = 0.9, SD 30/32 = 0.525) and poor in the south (0.15).
Coastal fish echo it — mostly `Fail` in the south/central, `Achieve` in the Gulfs of Bothnia
and Riga. The single genuinely informative model-vs-HOLAS-3 test would be **structural**:
does the model reproduce the south-demersal / north-pelagic dominance pattern? But the OSMOSE
Baltic model is a **single whole-domain grid** with no per-SD biomass output, so this test is
not achievable without a spatial-resolution change that is out of proportion to the payoff.

## Decision

- **No HOLAS-3 validator library / CLI / auto-flag is built.** The data cannot support a
  quantitative per-species check, and a guild-level directional check is signal-free.
- **The reference snapshot IS frozen** (`data/baltic/reference/helcom_snapshots/`) plus a
  one-shot puller (`scripts/_pull_helcom_snapshots.py`), so the HOLAS-3 fish status for the
  domain is on record and refreshable — useful policy context for interpreting the model's
  known overshoots, and a head-start if a future spatially-resolved model makes Finding 3's
  structural test viable.

## What HOLAS-3 *does* tell us (the citable takeaway)

HELCOM HOLAS-3 rates the **entire Baltic commercial-fish complex sub-GES** (demersal mean BQR
0.34, pelagic 0.31; 0-1 of 12 subdivisions at Good Environmental Status), and coastal fish as
mostly sub-GES outside the Gulfs of Bothnia and Riga. The model's per-species quantitative
validation stays with ICES SAG; HOLAS-3 contributes qualitative, domain-level policy context,
not a quantitative check.

## Don't re-investigate

Do **not** revisit a quantitative or directional HOLAS-3 fish validator for this model — the
limitation is in the HOLAS-3 data granularity + the model's whole-domain spatial resolution,
both established here against the live service. Revisit only if the model gains per-SD
biomass output (enabling Finding 3's structural pattern test) or if a future HOLAS vintage
publishes per-species absolute reference points as structured data.
