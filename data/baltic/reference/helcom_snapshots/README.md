# HELCOM HOLAS-3 fish snapshots (Baltic)

Frozen HELCOM HOLAS-3 fish-theme reference data for the OSMOSE Baltic domain
(ICES SD 22-32). Pulled from the live HELCOM ArcGIS service on the `created`
date in `index.json`.

## What this is — and is NOT

**This is a DIAGNOSTIC reference, not a quantitative validator.** Unlike the
ICES SAG snapshots (`../ices_snapshots/`), which give per-species SSB in
**tonnes** that model biomass can be compared against directly, HOLAS-3 fish
exposes only a **0-1 quality ratio** (BQR for commercial fish, EQR for coastal
fish) per ICES subdivision / sub-basin, split into two coarse buckets —
**DEM** (demersal) and **PEL** (pelagic). There is:

- **no per-species value** (cod/sprat/etc. are collapsed into DEM/PEL),
- **no absolute biomass/abundance target** (the baselines that define the
  ratio live only in narrative HOLAS-3 PDF reports, not in the service), and
- consequently **no way to build an ICES-style quantitative HOLAS-3 validator**.

See `docs/baltic_holas3_diagnostic_2026-06-04.md` for the full reasoning,
including why a model-vs-HOLAS-3 "directional consistency" check was rejected
as numerology (the whole domain is uniformly sub-GES, so HOLAS-3 carries no
discriminating signal — the check would collapse to the sign of the model's
own trend).

## Files

- `index.json` — manifest: source service/layers, domain, the external GES
  boundary (0.6), the model-species-to-guild mapping (intent only), and a
  `domain_summary` of the headline finding.
- `commercial_fish.json` — layer 434, SD 22-32: `{sub_division, area_full,
  dem, pel, bqr}`. SD 28 appears as two records (28.1 / 28.2). SD 30 & 31
  (Gulf of Bothnia) have `dem = 0` = **no demersal commercial assessment**,
  not a failing demersal score.
- `coastal_fish.json` — layer 417, assessed Baltic sub-basins:
  `{sub_basin, country, eqr, status}`. Aggregated "Integrated abundance
  coastal fish species" — **not per-species**. Blank/unassessed water bodies
  and Kattegat (SD 21) are dropped.

## How to refresh (next HOLAS assessment)

Re-run the one-shot puller from the repo root:

```bash
PYTHONPATH=. .venv/bin/python scripts/_pull_helcom_snapshots.py
```

It overwrites `commercial_fish.json` + `coastal_fish.json` from the live
service. Then update `index.json` by hand: bump `created`, and recompute
`domain_summary` (the per-guild mean BQR and GES fractions) if the pulled
values changed.

## Provenance notes (from review against the live service)

- The commercial layer keys SD by `Area_Full`, and the ICES Division letter
  **varies**: SD 22 = `27.3.c.22`, SD 23 = `27.3.b.23`, SD 24-32 = `27.3.d.*`.
  A `LIKE '27.3.d.%'` filter would silently drop SD 22 & 23 — the puller
  filters on `27.3.%` then keeps SD 22-32 in code.
- Coastal `EQR` + `Status` live on **layer 417** (layer 433 has `BQR`, not EQR).
- `Confidence` was `0` for every commercial record in this vintage — dropped.
- The 0.6 GES boundary is an **external HELCOM convention**, not a service field.
