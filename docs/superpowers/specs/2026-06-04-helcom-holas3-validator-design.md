# HELCOM HOLAS-3 guild-status cross-check — Design

**Date:** 2026-06-04
**Status:** ⚠️ **RESCOPED to a diagnostic — the validator below was NOT built.**

> ## Outcome (post in-loop review)
>
> An in-loop review executing against the **live** HOLAS-3 data killed the central
> "directional-consistency" indicator before implementation. Pulling layer 434 for SD 22-32
> showed the whole domain is **uniformly sub-GES** (demersal mean BQR 0.345, pelagic 0.306;
> 0-1 of 12 subdivisions at GES), so "HOLAS-3 concern" is constant-true for both guilds and the
> indicator collapses to the **sign of the model's own within-run trend** — numerology, not
> validation. (Plus: incommensurable time axes, guild sums dominated by cod's ×17-48 overshoot,
> and a truth-table hole exactly where a calibrated run lands.)
>
> **Decision (user-approved):** rescope to the percid pattern — freeze the HOLAS-3 reference
> snapshot + a one-shot puller, write a diagnostic, build **no** validator library / CLI /
> auto-flag. Shipped artifacts:
> - `data/baltic/reference/helcom_snapshots/` (commercial + coastal JSON, manifest, README)
> - `scripts/_pull_helcom_snapshots.py` (one-shot puller, refresh mechanism)
> - `docs/baltic_holas3_diagnostic_2026-06-04.md` (the finding + the reasoning)
>
> The original validator design is retained below as provenance — what was considered and why
> it was rejected.
>
> ---

**Original status:** Approved direction (brainstormed; HELCOM-MCP-recon-grounded). New feature.

## Motivation

The repo already validates model outputs **quantitatively per species** against ICES SSB
envelopes (`osmose/validation/ices.py`). HELCOM HOLAS-3 is the official *holistic* status
assessment for the Baltic and a recognized policy benchmark for the exact model domain
(ICES SD 22–32). This adds a second, complementary lens: the model's **guild-level**
(demersal vs pelagic) biomass placed against HOLAS-3's Good-Environmental-Status (GES)
verdict.

## The decisive recon finding (shapes the whole design)

HOLAS-3 (via the HELCOM ArcGIS REST services the `mcp__helcom__` tools wrap) does **NOT**
expose per-species quantitative biomass/SSB targets. The fish theme gives only:

- **Commercial Fish Integrated Assessment** (service `MADS/Indicators_and_assessments/MapServer`,
  layer 434): per ICES subdivision, a **BQR** (Biological Quality Ratio, 0–1, GES boundary
  **0.6**) split into two buckets — **DEM** (demersal) and **PEL** (pelagic). No per-species
  number, no tonnes. SD 22–32 are all present (records keyed by `Area_Full` `27.3.d.<SD>` /
  `SubDivisio`).
- **Coastal fish** (layers 417 / 433): an **EQR** (0–1, string-typed) + categorical
  **Status** (`Achieve` / `Fail`), aggregated ("Integrated abundance coastal fish species"),
  keyed by `HELCOM_ID` / sub-basin name. Sparse coverage.
- **Pelagic-habitats / food-web** themes: purely categorical (a WFD word). **Out of scope.**

The absolute baselines that define those ratios live only in narrative PDF reports — not
fetchable as structured data. **Consequence:** an ICES-style *quantitative per-species*
validator is impossible against HOLAS-3. This design therefore builds a **directional /
contextual cross-check**, explicitly NOT a recomputed BQR and NOT a pass/fail score. This
is a deliberate re-pricing (same discipline as the fisheries Kobe/B-Bmsy rescope) — surfacing
that the categorical data cannot support a quantitative scorer, rather than building a
misleading one.

## Species → guild mapping (approved)

| Model species | HOLAS-3 bucket | Reference source |
|---|---|---|
| cod, flounder | **Commercial DEM** (demersal) | layer 434 `DEM` per SD |
| sprat, herring | **Commercial PEL** (pelagic) | layer 434 `PEL` per SD |
| perch, pikeperch | **Coastal fish** | layer 417/433 `EQR` / `Status` |
| smelt, three-spined stickleback | **Unassessed** (forage; in neither indicator) | none — model output reported, no HOLAS-3 ref |
| grey seal, cormorant, whitefish (background) | **Excluded** (not in the fish theme) | none |

The mapping is data-driven from `index.json` (not hardcoded in the library), so a different
config's species set can ship its own mapping.

## Architecture

Mirrors the ICES validator's three-part structure: one-shot **snapshot puller** → committed
**frozen JSON** → **validator library** + **CLI**.

### 1. Snapshot puller — `scripts/_pull_helcom_snapshots.py`

One-shot helper (run once; snapshots committed; idle until refresh). Hits the HELCOM ArcGIS
REST query endpoints **directly via `httpx`** (the same services `mcp__helcom__` wraps —
mirrors `_pull_ices_snapshots.py`, which hits the ICES SAG REST API directly rather than
through the MCP). Pulls:

- Layer 434 (commercial), filtered to `Area_Full LIKE '27.3.d.%'` (SD 22–32) — captures
  `SubDivisio`, `Area_Full`, `DEM`, `PEL`, `BQR`, `Confidence`.
- Layers 417 / 433 (coastal) for the model's coastal sub-basins — captures area key,
  `Indicator`, `EQR`, `Status`.

Writes flat lowercase-key JSON (MCP-equivalent), then is no longer needed until the next
HOLAS refresh. The exact ArcGIS base URL + layer IDs are confirmed at implementation time
against the live service (the puller is one-shot and network-dependent by nature).

### 2. Snapshot layout — `data/baltic/reference/helcom_snapshots/`

- `index.json` — manifest: `model_species_to_guild` (species → `demersal`/`pelagic`/
  `coastal`/`unassessed`/`excluded`), `ges_boundary` (0.6), `assessment` ("HOLAS 3"),
  `subdivisions` (the SD list pulled), `source_layers`.
- `commercial_fish.json` — list of `{sub_division, area_full, dem, pel, bqr, confidence}`.
- `coastal_fish.json` — list of `{area, indicator, eqr, status}`.
- `README.md` — how to refresh (re-run the puller) + an explicit "what this is NOT" note.

### 3. Validator library — `osmose/validation/helcom.py`

API shape parallels `ices.py`:

- `@dataclass Holas3Snapshot` — loaded bundle (manifest + commercial records + coastal
  records + `snapshot_dir`).
- `load_helcom_snapshot(snapshot_dir: Path) -> Holas3Snapshot`.
- `@dataclass GuildComparison` — per guild: `guild` (name), `species` (list), model
  `biomass_window_mean`, model `trend` (`"declining"`/`"stable"`/`"rising"`/`"n/a"`),
  `holas3_mean_bqr`, `holas3_ges_fraction` (fraction of SDs with BQR ≥ 0.6),
  `directional_consistency` (`"consistent"`/`"inconsistent"`/`"n/a"`), plus a free-text
  `note`. A separate light structure for coastal (EQR/status context) and for unassessed
  (model output only).
- `compare_outputs_to_holas3(results, snapshot, window_years) -> Holas3Report` — returns
  the per-guild comparisons + the domain-level pelagic:demersal ratio + an overall note.
- `format_markdown_report(report) -> str`.
- `osmose/validation/__init__.py` extended to export the new public names.

### 4. CLI — `scripts/validate_outputs_vs_holas3.py`

Flags mirror `validate_outputs_vs_ices.py`: `--results-dir`, `--snapshots-dir`
(default `data/baltic/reference/helcom_snapshots`), `--window-years` (default 5),
`--prefix` (default `osm`), `--report <path.md>`, `--json <path.json>`.

## Data flow

1. **One-shot:** puller → frozen JSON (committed).
2. **Per validation:** CLI loads `OsmoseResults(results_dir, strict=False)` + the snapshot →
   for each guild, sum the member species' biomass over the window (the 1D global outputs are
   **WIDE** — `Time` + per-species columns + a constant `species="all"`; window by the `Time`
   column in years, NOT by row count; per-species values are columns — reusing the
   delta-tracking lesson) → compute within-run trend → aggregate HOLAS-3 to the domain (mean
   BQR + GES fraction per guild across the pulled SDs) → directional flag → markdown/JSON.

## Directional-consistency definition (soft; "direction only, not a GES verdict")

Per commercial guild (DEM, PEL):

- HOLAS-3 **concern** = domain mean BQR < 0.6 (sub-GES).
- Model **stress** = guild biomass declining over the window (mean of the last third of the
  window < mean of the first third; "rising" if the reverse; "stable" if within ±5 %).
- `consistent` = (sub-GES AND declining) OR (GES AND stable/rising).
- `inconsistent` = (sub-GES AND rising) OR (GES AND declining).
- `n/a` = guild absent from HOLAS-3 for the domain, or < 2 distinct window time-points (can't
  compute a trend).

Coastal fish (perch/pikeperch): reported separately with EQR + `Status` as context (no
directional flag — the coastal layer is sparse and aggregated). Unassessed (smelt,
stickleback): model biomass + trend reported, no HOLAS-3 reference, no flag. The report
header states the indicator is **directional only**.

## Error handling

- Missing snapshot dir / required file → clear `FileNotFoundError`-style error (mirror the
  ICES loader).
- A mapped species absent from the model outputs → skip it, note in the guild's `note`.
- `window_years` longer than the run → use the available span, note it.
- A guild with no member species present, or absent from HOLAS-3 → `directional_consistency
  = "n/a"`.
- Coastal EQR string un-parseable / blank → treat as no-data, note.

## Testing

- `load_helcom_snapshot` on a tiny fixture snapshot dir (manifest + 2 commercial + 1 coastal
  record).
- Guild mapping resolution from `index.json` (incl. unassessed/excluded handling).
- Guild biomass window-mean + within-run trend on a synthetic **wide** biomass CSV (same
  fixture technique as the delta-tracking tests — `OsmoseResults(...).biomass().to_csv`
  shape; window by Time, sub-annual-cadence-safe).
- Domain aggregation: mean BQR + GES fraction across SD records.
- Directional-consistency **truth table** — all four quadrants (sub-GES×declining,
  sub-GES×rising, GES×declining, GES×rising) + the two `n/a` paths.
- `format_markdown_report` smoke (renders without error; contains the "directional only"
  caveat + each guild row).
- The puller's parse/flatten helper unit-tested on a captured ArcGIS-response fixture; **no
  live-network test** (the puller is one-shot).

## Scope / YAGNI

- **In:** the puller + frozen snapshot, the guild-level library (DEM/PEL directional check +
  coastal/unassessed context), the CLI, the tests, the docs.
- **Out:** recomputing the BQR; per-species HELCOM scoring (impossible from the data);
  the categorical pelagic-habitats / food-web theme; per-SD model resolution (the model is a
  single whole-domain grid); area-weighted BQR aggregation (simple mean across SDs — see
  Honest limitations); any change to the ICES validator or calibration.

## Honest limitations

- The directional flag is a **trend-vs-status** heuristic, not a GES classification — a single
  run has no absolute GES baseline. The report says so prominently.
- Domain aggregation uses a **simple mean** of per-SD BQR (and an unweighted GES fraction).
  Area-weighting is deferred (the polygons' `Shape_Area` is available but the model doesn't
  resolve per-SD biomass, so a weighting would be cosmetic).
- Coastal-fish coverage is sparse/aggregated → perch/pikeperch get **context, not a check**.
- HOLAS-3 is a fixed assessment snapshot (HOLAS 3, ~2023); the validator compares a model run
  to that fixed verdict, not to a moving target. Refresh = re-run the puller at the next HOLAS.

## Delivery

Single PR: `scripts/_pull_helcom_snapshots.py`, `data/baltic/reference/helcom_snapshots/*`,
`osmose/validation/helcom.py`, `osmose/validation/__init__.py` (exports),
`scripts/validate_outputs_vs_holas3.py`, tests, docs. No engine changes, no calibration runs.
