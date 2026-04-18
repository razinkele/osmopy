# Baltic Sea OSMOSE Example — Full Provenance

> Living document. Last validated against repository state: **2026-04-18**. Update when parameter files change.

## Purpose

An end-to-end OSMOSE configuration for the Baltic Sea used to validate the Python engine port against Java parity targets and as a calibration sandbox. It is also the reference model for the ICES SAG cross-validation tooling (`scripts/validate_baltic_vs_ices_sag.py`) and the HPC preflight sensitivity workflow.

This document is a **navigation + provenance layer** over the ~800 lines of parameter CSVs in `data/baltic/`. Values themselves live in the CSVs; this doc explains where each family of parameters came from and why.

## At a glance

| Dimension                | Value                                                                 |
|--------------------------|-----------------------------------------------------------------------|
| Domain                   | 10–30° E × 54–66° N, 50 × 40 cells (~0.4° × 0.3° cell)                |
| Ocean cells (mask)       | 612 (rebuilt 2026-04-17 from fish-distribution union)                 |
| Time step                | 24 dt/yr (fortnightly), 50-yr default run                             |
| Focal species            | 8: cod, herring, sprat, flounder, perch, pikeperch, smelt, stickleback |
| LTL resource groups      | 6: diatoms, dinoflagellates, micro-/meso-/macrozooplankton, benthos   |
| Fisheries                | 8 (one per focal species)                                             |
| Forcing                  | Monthly CMEMS Baltic BGC (2024), depth-integrated 0–50 m              |
| Calibration target       | 2018–2022 equilibrium biomass + ICES F rates                          |
| Engine                   | Python (primary) and Java (reference parity); see `osmose/engine/`    |

## File inventory

```
data/baltic/
├── baltic_all-parameters.csv         # Master include list; what OSMOSE loads first
├── baltic_param-simulation.csv       # nyear, ndt/yr, nspecies, nresource
├── baltic_param-grid.csv             # lat/lon extents, mask + NetCDF paths
├── baltic_param-species.csv          # 156 lines: growth, maturity, L-W allometry, egg size
├── baltic_param-predation.csv        # accessibility matrix + ingestion, size-ratio, efficiency
├── baltic_param-fishing.csv          # 8 fisheries × F rates × selectivity × seasonality
├── baltic_param-movement.csv         # 219 lines: per-species per-life-stage map references
├── baltic_param-reproduction.csv     # per-species seasonality-file references
├── baltic_param-ltl.csv              # 6 resource-group definitions + NetCDF var mapping
├── baltic_param-output.csv           # which outputs to save; cutoff ages
├── baltic_param-starvation.csv       # starvation mortality cap
├── baltic_param-additional-mortality.csv  # larval + adult implicit mortality (M)
├── baltic_param-out-mortality.csv    # out-of-domain (migration) mortality
├── baltic_param-init-pop.csv         # initial seeding biomass per species
├── baltic_grid.nc                    # grid mask NetCDF (50×40, bool)
├── baltic_ltl_biomass.nc             # monthly LTL biomass forcing (t/cell)
├── predation-accessibility.csv       # 14×14 prey × predator accessibility matrix
├── fishery-catchability.csv          # per-species × per-fishery catchability (identity here)
├── fishery-discards.csv              # per-species × per-fishery discards (zero here)
│
├── maps/                             # 25 species × life-stage distribution maps (CSV grids)
│   ├── cod_{juvenile,adult,spawning}.csv
│   ├── herring_{juvenile,adult,spawning,spawning_autumn}.csv
│   ├── sprat_{juvenile,adult,spawning}.csv        # spawning = adult for sprat
│   ├── flounder_{juvenile,adult,spawning}.csv
│   ├── perch_{juvenile,adult,spawning}.csv
│   ├── pikeperch_{juvenile,adult,spawning}.csv
│   ├── smelt_{juvenile,adult,spawning}.csv
│   └── stickleback maps are shared with perch/smelt (coastal pattern)
│
├── grid/
│   └── baltic_mask.csv               # 50×40 ocean (1) / land (-99) mask, CSV form
│
├── fishing/
│   └── fishing-distrib.csv           # Baltic-wide fishing distribution (50×40, used by all 8 fleets)
│
├── reproduction/
│   └── reproduction-seasonality-sp{0..7}.csv  # GSI curves on the 24-step year (fraction per dt)
│
├── reference/
│   ├── biomass_targets.csv           # species × [lower, upper] tonnes + source + weight
│   └── ices_snapshots/               # frozen 2024-advice SAG payloads (see Validation)
│
└── calibration_results/              # gitignored — outputs of scripts/calibrate_baltic.py
```

Sizes in lines of CSV (excluding comments): species 156, movement 219, fishing 106, predation 67, LTL 64, output 54, additional-mortality 27. Master file `baltic_all-parameters.csv` is the OSMOSE entry point — it `include`s every other file.

## How this example was built

The Baltic example is the product of several overlapping threads. In chronological order:

1. **Scaffolding (initial port).** Started as a hand-authored 8-species × 50×40 grid mirroring the Java EEC example structure (`docs/plans/2026-02-21-osmose-python-port-plan.md`). Species list chosen to cover the main Baltic fishery management units: 3 pelagics (herring, sprat, stickleback), 2 demersals (cod, flounder), 3 coastal (perch, pikeperch, smelt).

2. **LTL forcing from CMEMS** (2026-04-16). `mcp_servers/copernicus/server.py` wraps Copernicus Marine `copernicusmarine` into three MCP tools; `generate_osmose_ltl()` downloads the Baltic BGC reanalysis (`cmems_mod_bal_bgc_anfc_P1M-m`), depth-integrates 0–50 m, splits phytoplankton into diatom/dinoflagellate by a seasonal-succession pattern, maps ERGOM's `zooc` to micro/meso/macrozoo by size fraction, and writes `baltic_ltl_biomass.nc` on the 50×40 OSMOSE grid. The tool is idempotent; re-run yearly for vintage refresh. Credentials read from repo-root `.env`.

3. **Species parameters from literature + FishBase** (2026-04-16 validation pass). Von Bertalanffy growth (Linf, K, t0) and L–W allometry sourced from FishBase (eight Baltic species) with cross-references for cod (Svedäng et al. 2024 for post-2015 growth impairment — *documented but not applied*; current config uses historical growth). Baltic-specific values preferred where FishBase offers multiple populations.

4. **Predation structure** (2026-04-16 corrections + 2026-04-17 sp6 rebuild). Accessibility matrix hand-coded from published Baltic stomach-content studies, then audited twice: herring-on-benthos lifted from 0 → 0.15 (mysid/amphipod presence in coastal areas); sprat phytoplankton accessibility reduced from 0.6/0.5 to 0.2/0.2 (strict zooplanktivore); sp6 swapped from whitefish to smelt (better matches the Baltic forage-fish niche). Ingestion rate max from Palomares & Pauly (1998) Q/B regression with Travers-Trolet et al. (2019) OSMOSE-EEC application.

5. **Fishing structure** (calibrated "R18" configuration). Eight fisheries, one per species, with knife-edge age-at-50%-selectivity. Base F rates calibrated against ICES SAG (see next thread). Seasonality uniform (1/24 per step) — no within-year concentration.

6. **Calibration (phase 1 larval M, phase 2 adult M + F).** `scripts/calibrate_baltic.py` uses the Python engine to run many short simulations and fit larval-mortality + selected adult parameters to the 2018–2022 ICES biomass envelope. The current committed values are from calibration round 18 ("R18" — noted in file comments).

7. **Mask rebuild** (2026-04-17). `scripts/rebuild_baltic_mask.py` shrank the mask from 912 to 612 ocean cells by keeping only cells where at least one species map or fishing map had a positive value. The 300 removed cells were "ocean-with-zero" cells in northern Norway/Sweden that had been mirrored from the southern Baltic pattern — no species ever used them. CSV grids rewritten so those cells became `-99` (land).

8. **ICES SAG cross-validation** (2026-04-18 — Front 1 of current work). Eight snapshots of the 2024 advice cycle (plus `cod.27.22-24` from 2022 since its 2024 assessment is category-3) frozen under `data/baltic/reference/ices_snapshots/`. `scripts/validate_baltic_vs_ices_sag.py` compares F rates and biomass envelopes; output at `docs/baltic_ices_validation_2026-04-18.md`.

### Scripts (what builds what)

| Script                                   | Rebuilds / produces                                        |
|------------------------------------------|------------------------------------------------------------|
| `scripts/_pull_ices_snapshots.py`        | `data/baltic/reference/ices_snapshots/*.json` + `index.json` |
| `scripts/calibrate_baltic.py`            | `calibration_results/*` (not committed)                    |
| `scripts/rebuild_baltic_mask.py`         | `grid/baltic_mask.csv`, `baltic_grid.nc`, CSV grids        |
| `scripts/relabel_baltic_grid_nc.py`      | Metadata pass over `baltic_grid.nc`                        |
| `scripts/validate_baltic_vs_ices_sag.py` | `docs/baltic_ices_validation_2026-04-18.md`                |
| `mcp_servers/copernicus/server.py`       | `baltic_ltl_biomass.nc` via `generate_osmose_ltl` MCP tool |

All scripts assume CWD = repo root. None write to checked-in files unless re-run intentionally.

## Parameter provenance

Each subsection below documents the **source** of a parameter family (not the values — those are in the CSVs). Values are reported only where picking them out from the CSV would be tedious (e.g. the accessibility matrix).

### Simulation timing (`baltic_param-simulation.csv`)

- **Source:** OSMOSE project convention.
- **Chosen values:** `ndtPerYear=24` (fortnightly), `nyear=50`, `nspecies=8`, `nresource=6`, `ncpu=1`.
- **Rationale:** 24 dt/yr is the OSMOSE-EEC default — enough temporal resolution for spring/autumn reproductive phenology without blowing up runtimes. 50 years is the spin-up needed for the Baltic to reach quasi-equilibrium from seeding biomass (observed from calibration test runs).
- **`mortality.subdt=10`:** sub-timestep for the finite-difference mortality integration, set in `baltic_all-parameters.csv`. OSMOSE default.

### Grid (`baltic_param-grid.csv`, `baltic_grid.nc`, `grid/baltic_mask.csv`)

- **Source:** Hand-designed domain to match the Baltic management extent used in ICES SAG advice (subdivisions 22–32 plus Kattegat-Skagerrak for cross-boundary species).
- **Extent:** 50 cells (10°–30° E) × 40 cells (54°–66° N). Each cell ≈ 40 km × 33 km.
- **Mask (612 cells):** Rebuilt 2026-04-17 from the union of species distribution maps. Rationale in `scripts/rebuild_baltic_mask.py` top docstring: the previous 912-cell mask had 300 "ocean-with-zero" cells that were mirrors of the southern Baltic pattern into northern Norway/Sweden — never used by any species, visually wrong.
- **NetCDF handshake:** `grid.netcdf.file = baltic_grid.nc` + `grid.var.mask = mask`. The Java engine reads the NetCDF; the Python engine reads either the NetCDF or the CSV (at `grid.mask.file`) — both kept in sync.

### Species (8 focal species, `baltic_param-species.csv`)

Comments at the top of the CSV list the FishBase cross-references. Growth parameters represent **historical (pre-2015) life history**. Eastern Baltic cod has undergone severe growth impairment since ~2015 (Svedäng et al. 2024, `doi:10.1002/ece3.70382`); current effective Linf is ~60–80 cm vs the configured 110 cm. This is a documented modeling choice — see `docs/baltic_ices_validation_2026-04-18.md` Findings for why we run in the "historical ecosystem" scenario rather than the post-collapse state.

| sp | name        | Linf (cm) | K    | t0    | a (×10⁻³) | b    | Lifespan | Source                                   |
|----|-------------|-----------|------|-------|-----------|------|----------|------------------------------------------|
| 0  | cod         | 110       | 0.15 | -0.20 | 8.70      | 3.05 | 20       | *Gadus morhua* (FishBase Baltic entry)   |
| 1  | herring     | 27        | 0.35 | -0.80 | 5.60      | 3.13 | 12       | *Clupea harengus membras* (Baltic subspecies) |
| 2  | sprat       | 16        | 0.45 | -0.50 | 4.90      | 3.12 | 8        | *Sprattus sprattus* (Baltic pop.)        |
| 3  | flounder    | 42        | 0.20 | -0.40 | 8.90      | 3.08 | 15       | *Platichthys flesus*                     |
| 4  | perch       | 45        | 0.15 | -0.50 | 10.70     | 3.10 | 15       | *Perca fluviatilis*                      |
| 5  | pikeperch   | 90        | 0.18 | -0.30 | 6.20      | 3.09 | 15       | *Sander lucioperca*                      |
| 6  | smelt       | 25        | 0.35 | -0.30 | 5.00      | 3.05 | 7        | *Osmerus eperlanus* (swapped in from whitefish 2026-04-17) |
| 7  | stickleback | 8         | 0.80 | -0.10 | 8.00      | 3.05 | 4        | *Gasterosteus aculeatus*                 |

Egg sizes (0.10–0.20 cm), maturity lengths, and size-at-stage thresholds are further down in the CSV — sourced from the same FishBase entries and, where present, annotated with the specific reference in a comment block.

### LTL resources (6 groups, `baltic_param-ltl.csv`, `baltic_ltl_biomass.nc`)

- **Source:** CMEMS Baltic BGC analysis/forecast (`cmems_mod_bal_bgc_anfc_P1M-m`), 2024 monthly, depth-integrated 0–50 m. Processed by `mcp_servers/copernicus/server.py::generate_osmose_ltl()`.
- **Variables:**
  - Phytoplankton: `phyc` (mmol C / m³) split into diatoms and dinoflagellates using a seasonal-succession pattern (diatom-dominated spring bloom, dinoflagellate-dominated summer).
  - Zooplankton: `zooc` split into microzoo / mesozoo / macrozoo by size class.
  - Benthos: fixed-biomass proxy; no dedicated CMEMS variable.
- **Trophic levels:** 1.0 (phyto), 2.0–3.0 (zoo by size), 2.5 (benthos). Hand-set per standard Baltic food-web description; see LTL CSV comment block.
- **Size ranges:** 0.0002 cm (phyto cell) to 10 cm (large benthos invert). Used by the predator–prey size-ratio predation kernel.
- **Accessibility to fish:** uniform 0.8 across all 6 groups. Not differentiated — OSMOSE lumps water-column accessibility into the predator–prey kernel; this parameter is a coarse "fraction available per timestep" knob.

### Predation (`baltic_param-predation.csv`, `predation-accessibility.csv`)

The 14-prey × 14-predator accessibility matrix is the hand-authored heart of the trophic model. File header lists every correction ever applied with date + rationale.

Structural notes:

- **Critical predation efficiency:** `0.57` for every focal species. This is the OSMOSE default from Shin & Cury (2004); not species-specific, and not well-constrained empirically.
- **Maximum ingestion rate** (g food / g predator / year): 3.0–7.0 range, species-specific. Source: Palomares & Pauly (1998, `doi:10.1071/mf98015`) Q/B regression + Travers-Trolet et al. (2019, `doi:10.1016/j.ecolmodel.2019.108800`) OSMOSE-EEC application. Higher for planktivores (sprat 7.0, herring 6.0, stickleback 5.0) than demersals (flounder 3.0). Cod (3.5) is flagged as potentially overstated — Ryberg et al. (2020, `doi:10.1093/conphys/coaa093`) documents Contracaecum osculatum parasite effects on eastern Baltic cod foraging.
- **Predator–prey size ratios (min / max):** 2.5–10 / 30–1000 range. Upper bound 500–1000 for small-mouthed planktivores means they only target very small prey; lower bound 2.5–5 for top predators captures the "must be larger than prey" rule.
- **Stage structure:** size-based for predator–prey kernel, age-based for accessibility. Thresholds set per species in the CSV.
- **Accessibility matrix** (see `predation-accessibility.csv`): 14 rows (prey) × 14 columns (predator). Cod eats mesozoo (0.5), macrozoo (0.6), benthos (0.6), and is cannibalistic (cod-on-cod 0.05). Sprat and herring are strict zooplanktivores (no phyto > 0.2). Stickleback the most-accessible prey in the matrix (heavy access from almost all piscivores). Smelt is the heaviest cross-trophic forage fish (access 0.5–0.8 from cod/perch/pikeperch/flounder).

### Reproduction (`baltic_param-reproduction.csv`, `reproduction/reproduction-seasonality-sp{0..7}.csv`)

- **Source:** Per-species spawning-season literature, all cross-referenced in the header block of `baltic_param-movement.csv` (the movement file groups spawning-map+timing together even though the actual per-step fractions live in the reproduction CSVs).
- **Data shape:** Each seasonality CSV is a 24-row `(time_year, fraction)` table where fractions sum to 1.0 across the spawning window.

Species-by-species spawning windows (with DOIs):

- **Cod (sp0):** March–August. Bleil et al. 2009, `doi:10.1111/j.1439-0426.2008.01172.x` — Bornholm Basin main spawning May–Aug with post-1990s shift to July peak; Western Baltic spring Mar–Apr. Combined window covers both stocks.
- **Herring (sp1):** March–June (spring spawner). Ory et al. 2024, `doi:10.1111/jfb.15811` for Western Baltic spring-spawning (WBSS); central Baltic spring also peaks Apr–Jun. Autumn-spawning herring NOT represented in the current config.
- **Sprat (sp2):** April–July. Haslob et al. 2013, `doi:10.1016/j.fishres.2012.08.002` — Baltic sprat spawns Jan–Jun, peak Apr–Jun in Bornholm Basin. Config captures the peak.
- **Flounder (sp3):** February–May. Pelagic spawner (*P. flesus*) spawns Feb–May in the deep southern Baltic; demersal spawner (*P. solemdali*) spawns Apr–Jun coastally. Config captures the pelagic spawner.
- **Perch (sp4):** May–June. Standard Baltic coastal perch spawning.
- **Pikeperch (sp5):** similar to perch; both are coastal spring spawners.
- **Smelt (sp6):** February–May. Coastal/estuarine spring spawner.
- **Stickleback (sp7):** spring–summer (nest-building; window overlaps the zooplankton bloom).

### Movement (`baltic_param-movement.csv`, `maps/*.csv`)

- **Method:** `movement.distribution.method.spN = maps` for every species. Life-stage maps (juvenile, adult, spawning) drive spatial distribution; no analytical movement kernel.
- **Random-walk range:** 2–3 cells/dt for most species (moderate dispersal); wider for pelagic species on long migrations. Hand-set.
- **Maps:** 25 CSV grids under `maps/`. Each is 50×40 with `1` = habitable, `0` = accessible-but-avoided, `-99` = land. File naming is `{species}_{stage}.csv` with stages `juvenile`, `adult`, `spawning` (plus `spawning_autumn` for herring — created to support the future autumn-spawning herring split but not currently wired).
- **Stage-to-map assignment:** controlled by `movement.distribution.ndt.spN` and the per-stage age thresholds in the species CSV. A single-stage species uses `adult` throughout; species with clear ontogenetic shifts (cod, flounder) split into juvenile/adult.
- **Sources:** map patterns hand-drawn from ICES distribution atlases, HELCOM species fact-sheets, and BalticNest WFD reports. Spawning maps tightened to known spawning grounds (e.g. cod spawning restricted to Bornholm/Gdansk/Gotland basins).
- **Rebuild:** if distribution data shifts, re-run `scripts/rebuild_baltic_mask.py` to sync the mask union; re-draw individual map CSVs by hand.

### Fishing (`baltic_param-fishing.csv`, `fishing/fishing-distrib.csv`, `fishery-catchability.csv`, `fishery-discards.csv`)

- **Structure:** 8 fisheries, one-per-species, knife-edge age selectivity, single fishing period per year, uniform seasonality (1/24 per dt). This is the simplest structurally defensible fishing model for a calibration sandbox.
- **Selectivity `a50` (age at 50% selectivity):** 1 yr for pelagics (herring, sprat, stickleback), 2 yr for cod/flounder/perch/pikeperch, 3 yr for smelt. Roughly mirrors minimum-landing-size regulations in the respective fisheries.
- **Base F rates** (year⁻¹), calibrated round 18:

  | fsh | species      | F base | ICES 2024 advice (weighted, 2018–2022) | Note                          |
  |-----|--------------|--------|-----------------------------------------|-------------------------------|
  | 0   | cod          | 0.08   | 0.91                                    | Scenario limitation — see docs/baltic_ices_validation_2026-04-18.md |
  | 1   | herring      | 0.15   | 0.21                                    | In [0.5×, 1.5×] tolerance     |
  | 2   | sprat        | 0.32   | 0.37                                    | In tolerance                  |
  | 3   | flounder     | 0.04   | 0.22                                    | Grid-resolution trade-off     |
  | 4   | perch        | 0.03   | (no ICES SAG)                           | Coastal — literature estimate |
  | 5   | pikeperch    | 0.03   | (no ICES SAG)                           | Coastal — literature estimate |
  | 6   | smelt        | 0.02   | (no ICES SAG)                           | Coastal — literature estimate |
  | 7   | stickleback  | 0.01   | (no ICES SAG)                           | Proxy — no commercial fishery |

- **Fishing distribution:** single Baltic-wide CSV (`fishing/fishing-distrib.csv`) shared across all 8 fleets. A simplification — real fleets have different footprints (pelagic trawl in open basins vs coastal gill-nets).
- **Catchability:** identity matrix (each fishery catches only its target species). No bycatch modeled.
- **Discards:** all zero. OSMOSE can model discards but this config doesn't need them.

### Mortality (outside fishing and predation)

- **Larval additional mortality** (`baltic_param-additional-mortality.csv`, rates 3.5–15.0 yr⁻¹). *These are calibration parameters, not measurable biology.* Larval M is the primary recruitment-control lever in OSMOSE; values set by `scripts/calibrate_baltic.py` (currently R18 values).
- **Adult additional mortality** (same file). Non-zero for:
  - **Cod 0.20:** includes implicit grey-seal predation. ICES estimates grey seal population >40 000 in Baltic, consuming ~30–50 kt cod/yr (comparable to fishing mortality). With stock ~120 kt, seal-induced F ≈ 0.25–0.4 yr⁻¹. Config uses conservative 0.15 yr⁻¹ on top of 0.05 baseline.
  - **Perch 0.05, pikeperch 0.06:** small increments for cormorant predation in coastal areas.
  - **Others:** 0–0.05 (herring, sprat, flounder, smelt, stickleback). These are OSMOSE's generic "M not otherwise accounted for."
- **Starvation mortality** (`baltic_param-starvation.csv`, cap 0.3 yr⁻¹): OSMOSE default. Starvation kicks in when foraging fails to meet maintenance rate. Cap prevents cascades.
- **Out-of-domain mortality** (`baltic_param-out-mortality.csv`): herring 0.05, sprat 0.08, others 0. Represents fish that exit the model domain (e.g. into Kattegat) and are lost to the modeled ecosystem. Calibrated.

### Output configuration (`baltic_param-output.csv`)

- **Record frequency:** 1/yr (`ndt=24` per record).
- **Outputs enabled:** abundance, biomass, diet composition, diet pressure, mortality decomposition, size distribution, catch size distribution, trophic-level distribution, mean TL by size, biomass by TL, yield, yield-by-age.
- **Cutoff ages:** 0.5 yr for all species (to match standard OSMOSE "recruit and older" reporting). Cutoff enabled.
- **Fishery-level outputs:** enabled (per-fleet catch time series).

### Initial population (`baltic_param-init-pop.csv`)

Per-species seeding biomass in tonnes. Values are order-of-magnitude anchors, not precise ICES SSB:

```
cod         150 000      (Baltic both stocks, historical median)
herring     800 000      (Baltic herring complex)
sprat       600 000      (Baltic sprat)
flounder     80 000
perch        30 000
pikeperch    15 000
smelt        20 000
stickleback 100 000      (Olsson et al. 2019, doi:10.1093/icesjms/fsz078)
```

OSMOSE re-distributes seeded biomass across age classes via the species life-history parameters at simulation start; after ~20 yr of spin-up the initial values lose meaning.

## Validation and known limitations

- **ICES SAG cross-validation** (`docs/baltic_ices_validation_2026-04-18.md`): F rates and biomass envelopes compared to the 2024 advice cycle (2022 for `cod.27.22-24` which is category-3 in 2024). Cod and flounder F flagged as deliberate scenario limitations; herring and sprat within tolerance. Cod biomass target upper bound (250 kt) flagged by science review as likely historical rather than 2018–2022-era — a future calibration-tuning pass will revisit.
- **Calibration targets** (`data/baltic/reference/biomass_targets.csv`): species × [lower, upper] tonnes + weight + source column. Weights 1.0 (well-assessed pelagics: herring, sprat) down to 0.2 (coarse-grid coastal: perch, pikeperch).
- **Stickleback biomass** (~200 kt target): Olsson et al. 2019, `doi:10.1093/icesjms/fsz078` — first large-scale biomass assessment. Wide range (50–500 kt) appropriate given boom-bust dynamics.

Documented limitations that this example **does not currently represent**:

- Post-2015 Eastern Baltic cod collapse state (growth impairment + low SSB). Model runs in "historical" scenario.
- Autumn-spawning herring stock (maps exist, not wired).
- Eastern Baltic flounder (`fle.27.24-32`) — no ICES SAG assessment exists for this stock; only `fle.27.2223` is validated against.
- Per-fleet spatial fishing footprints (all 8 fleets share one distribution map).
- Oxygen-limited cod reproduction (cod only spawns in basins with sufficient O₂, depth-dependent — not modeled).
- Parasite-induced mortality (Contracaecum osculatum on cod, documented but not applied).
- Climate-change trajectories (the 2024 CMEMS forcing is a single year; no scenarios).

## References

### Primary data

- **ICES SAG** Stock Assessment Graphs: https://sag.ices.dk — accessed via MCP server at `/home/razinka/ices-mcp-server/` (wraps the public REST API).
- **Copernicus Marine** Baltic BGC Analysis/Forecast (`cmems_mod_bal_bgc_anfc_P1M-m`): https://doi.org/10.48670/moi-00011 — accessed via `mcp_servers/copernicus/server.py`.
- **FishBase**: Froese & Pauly, https://www.fishbase.org — per-species Baltic population entries for growth and L–W allometry.

### Cited literature

- Bleil, Oeberst & Urrutia, 2009. *Seasonal maturity development of Baltic cod*. Journal of Applied Ichthyology. `doi:10.1111/j.1439-0426.2008.01172.x`.
- Haslob, Clemmesen, Schaber et al., 2013. *Hydrographic influence on Baltic sprat spawning*. Fisheries Research. `doi:10.1016/j.fishres.2012.08.002`.
- Olsson, Bergström, Sjöqvist et al., 2019. *A Baltic Sea-wide stickleback biomass assessment*. ICES Journal of Marine Science. `doi:10.1093/icesjms/fsz078`.
- Ory, Schade, Polte et al., 2024. *Western Baltic spring-spawning herring phenology*. Journal of Fish Biology. `doi:10.1111/jfb.15811`.
- Palomares & Pauly, 1998. *Predicting food consumption of fish populations as functions of mortality, food type, morphometrics, temperature and salinity*. Marine and Freshwater Research. `doi:10.1071/mf98015`.
- Ryberg, Buchmann, Skov et al., 2020. *Parasite-induced changes in Baltic cod foraging*. Conservation Physiology. `doi:10.1093/conphys/coaa093`.
- Shin & Cury, 2004. *Using an individual-based model of fish assemblages to study the response of size spectra to changes in fishing*. Canadian Journal of Fisheries and Aquatic Sciences.
- Svedäng, Hornborg & Casini, 2024. *Collapse of eastern Baltic cod — growth impairment*. Ecology and Evolution. `doi:10.1002/ece3.70382`.
- Travers-Trolet, Bourdaud, Genu et al., 2019. *OSMOSE-EEC ecosystem simulator*. Ecological Modelling. `doi:10.1016/j.ecolmodel.2019.108800`.

### Companion docs

- [Baltic ICES SAG Validation Report (2024 advice)](baltic_ices_validation_2026-04-18.md) — drift findings + unit caveats.
- [ICES snapshots README](../data/baltic/reference/ices_snapshots/README.md) — how to refresh the advice snapshots.
- [ICES MCP validation plan](superpowers/plans/2026-04-18-ices-mcp-baltic-validation-plan.md) — how the validator was built.
- [OSMOSE Python port plan](plans/2026-02-21-osmose-python-port-plan.md) — overall project context.
