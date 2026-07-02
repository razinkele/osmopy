---
name: project_holas3_diagnostic
description: HELCOM HOLAS-3 fish validator was RESCOPED to a diagnostic (no validator built) after in-loop review proved HOLAS-3 can't support a quantitative check; snapshot frozen. SHIPPED to origin/master 2026-06-04.
metadata:
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

Attempted "validate OSMOSE Baltic outputs vs HELCOM HOLAS-3, analogous to the ICES validator." **CLOSED via diagnostic — NO validator library/CLI/auto-flag built** (the percid pattern). Merged fast-forward to master + **pushed to origin/master 2026-06-04** (`77c1324..c890e12`, branch `feature/helcom-holas3-validator` deleted, origin synced).

## Why no validator (the in-loop review killed it — executing against LIVE HOLAS-3 data)
Brainstorm→spec→**3 in-loop reviewers** (data-grounding, codebase-accuracy, methodology-adversarial). The adversarial reviewer pulled live layer 434 and found two load-bearing facts that dissolved the feature:
1. **HOLAS-3 fish has NO per-species quantitative target.** Commercial layer 434 = a 0-1 ratio (BQR, GES boundary 0.6) at a **two-bucket DEM/PEL guild** granularity per ICES SD — no per-species, no tonnes. Coastal layer 417 = one aggregated "Integrated abundance coastal fish species" EQR+Achieve/Fail per sub-basin (perch/pikeperch NOT broken out). Baselines live only in narrative PDFs. So an ICES-style per-species validator is **impossible** (intrinsic to the data, not tooling).
2. **The whole SD 22-32 domain is uniformly sub-GES** (demersal mean BQR **0.345** 0/12 at GES; pelagic **0.306** 1/12 — only SD 31 Bothnian Bay). So a "directional-consistency" cross-check (model guild trend vs HOLAS-3 status) **collapses to the sign of the model's own within-run trend** — delete the snapshot, identical verdict = numerology. Plus: incommensurable time axes (real decadal status vs sim transient), guild sum dominated by cod's ×17-48 overshoot, truth-table hole at (sub-GES, stable) exactly where a calibrated run lands, DEM=0 in Gulf of Bothnia = "not assessed" not "failing".

Same re-pricing discipline as [[project_fisheries_fm_diagnostics]] (rescope) and [[project_percid_overshoot_diagnostic]] (close-without-building) — but here, unlike fisheries (kept sprat F/M=1.79), there was **no salvageable quantitative core**.

## What shipped (diagnostic + snapshot, all additive, no tests/engine change)
- `data/baltic/reference/helcom_snapshots/` — `commercial_fish.json` (12 SD 22-32 recs), `coastal_fish.json` (21 assessed recs: 6 Achieve/15 Fail), `index.json` (manifest + domain_summary), `README.md`.
- `scripts/_pull_helcom_snapshots.py` — one-shot httpx puller (ArcGIS `maps.helcom.fi/.../MADS/Indicators_and_assessments/MapServer`, layers 434+417). RAN clean against live service; output == MCP query == recomputed means.
- `docs/baltic_holas3_diagnostic_2026-06-04.md` — finding + reasoning + "don't re-investigate".
- `docs/superpowers/specs/2026-06-04-helcom-holas3-validator-design.md` — rescope banner + rejected validator design kept as provenance.

## Hard-won data facts (carry forward; corrections the recon got WRONG)
- Commercial layer 434 keys SD by `Area_Full` with a VARYING ICES Division letter: **SD 22=`27.3.c.22`, SD 23=`27.3.b.23`, SD 24-32=`27.3.d.*`**. A `LIKE '27.3.d.%'` filter silently DROPS SD 22+23 (western cod/flounder). Use `27.3.%` then filter SD 22-32 in code; SD 28 splits into 28.1/28.2.
- Coastal `EQR`+`Status` are on **layer 417** (layer 433 has `BQR`, not EQR).
- `Confidence` uniformly 0 in this vintage (dead field, dropped). GES boundary 0.6 is external (not a service field).
- HELCOM ArcGIS base `https://maps.helcom.fi/arcgis/rest/services`, service `MADS/Indicators_and_assessments/MapServer`, `?f=json`. The HELCOM MCP server source lives OUTSIDE this repo at `/home/razinka/helcom-mcp-server/helcom/arcgis.py`.
- `httpx` is installed only TRANSITIVELY (not in pyproject.toml) — same latent gap as `_pull_ices_snapshots.py`.
- Baltic species exact names (output-column keys): cod, herring, sprat, flounder, perch, pikeperch, smelt, **stickleback** (NOT "three-spined stickleback"); background = **GreySeal, Cormorant** only (whitefish is a PHANTOM — sp6 is smelt now).

## Don't re-investigate
Do NOT revisit a quantitative/directional HOLAS-3 fish validator — limited by HOLAS-3 granularity + the model's whole-domain (single-grid) spatial resolution. The one genuinely useful test (does the model reproduce the south-demersal/north-pelagic gradient?) needs per-SD model output the model doesn't have. Revisit only if the model gains per-SD biomass OR a future HOLAS publishes per-species absolute reference points as structured data. **Next: pick a fresh backlog item.**
