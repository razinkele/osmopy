# Salinity gradient & stock fragmentation — exploration (2026-07-24)

The Baltic's salinity gradient (this config's bottom climatology spans 0–35 PSU,
mean 9.8; 31% of cells < 6 PSU, 28% > 10 PSU) is the first-order driver of Baltic
fish ecology, and the model currently averages over most of it. This note records
what exists, what one lever does, and where the real leverage is.

## What exists

- **Salinity-gated occupancy** (`osmose/engine/processes/salinity_gate.py`, wired in
  `movement.py`; spec `docs/superpowers/specs/2026-07-04-salinity-gated-cod-occupancy-design.md`).
  A per-cell ramp `w(S) = clip((S − s_low)/(s_high − s_low), 0, 1)` multiplies a gated
  species' movement map, so it avoids low-salinity cells. Built, tested, and — as of
  today — **enabled for cod** (sp0, s_low=3, s_high=6) in the committed config. The
  real bottom-salinity climatology field is present (`baltic_salinity_bottom_climatology.nc`).
- **Salinity-limited reproduction** — in progress on the `fix+baltic-salinity-spawning`
  worktree/branch (distinct from the occupancy gate; not touched here).
- **No temperature or oxygen forcing** — salinity is the only environmental gradient wired in.

## Occupancy-gate effect (committed 5/8 config, 1 seed, nyear=40)

Enabling the cod gate is a realism improvement with a **minor** fit effect:

| species | baseline | cod-gated |
|---------|----------|-----------|
| cod | ×1.4 | ×1.4 |
| herring / sprat / flounder / stickleback | in range | in range (unchanged) |
| **pikeperch** | ×214.8 | **×201.3** |

The intended mechanism works directionally — confining cod to saline basins lowers
cod–percid spatial overlap, dropping pikeperch ~6% — but far too weakly to matter for
the percid overshoot (see `docs/baltic_percid_spatial_limitation_2026-07-24.md`). All
well-assessed species are unchanged, so **the 5/8 fit is robust to the gate**; it is
turned on for realism, not to move the fit.

## Where the real leverage is — stock fragmentation

The model lumps salinity-structured stocks into single species:
- **cod**: eastern (SD24-32, oligohaline, reproduction-limited, collapsed) + western
  (SD22-24, more saline) — genetically and dynamically distinct.
- **herring**: 4+ management units (central Baltic, Gulf of Bothnia, Gulf of Riga, western).
- **flounder**: two species (European + Baltic flounder) with different salinity
  spawning strategies.

Splitting these into salinity-niched sub-populations (distinct distributions +
recruitment) is the change that would let the model represent the real spatial
structure — e.g. a collapsed eastern cod coexisting with a healthier western stock —
and remove the aggregation bias `biomass_targets.csv` flags. This is a major
restructuring (more model species, new maps, re-calibration) and is scoped as its own
project: `docs/superpowers/specs/2026-07-24-baltic-stock-disaggregation-design.md`.
