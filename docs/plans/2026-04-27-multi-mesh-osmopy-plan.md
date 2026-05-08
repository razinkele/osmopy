# Multi-mesh support for OSMOSE-Python — implementation plan

> **Date:** 2026-04-27
> **Status:** plan, no code committed
> **Companion to:** [`docs/superpowers/specs/2026-04-24-non-rectangular-grids-exploration.md`](../superpowers/specs/2026-04-24-non-rectangular-grids-exploration.md) (feasibility study)
> **Author:** plan distilled from grid-topology survey + 2026-04-24 hex/FEM exploration

The 2026-04-24 feasibility study answered "is it possible?" (yes, in three flavours). This plan answers "if we commit, what are the *option packages* we can pick from, what does each ship, and in what order?". Five concrete option packages are described, sized, and contrasted; a sixth — "do nothing" — is the implicit baseline.

---

## TL;DR — picking from the menu

| # | Package | Engine impact | Cost | Ships when | Best for |
|---|---|---|---|---|---|
| **A** | Curvilinear-rectangular only | Minimal | **1–2 wk** | now | Coupling with NEMO/ROMS-style ocean models |
| **B** | Topology abstraction (no new mesh) | Refactor | **2–3 wk** | now | Cleanup that unblocks all later options without forking parity |
| **C** | A + B + Hex (H3) grid | New topology | **6–8 wk** | next quarter | Cleaner dispersal kernels, hierarchical refinement |
| **D** | A + B + Unstructured UGRID mesh | New topology + I/O | **3–4 mo** | next half | Coastal lagoon/estuary refinement |
| **E** | Pre-processor only (mesh→rect aggregator) | None | **1–2 wk** | now | Use external mesh data without touching engine |
| **F** | Nested-rectangular subgrids (AMR-lite) | Engine-internal | **4–6 wk** | next quarter | Refine specific zones (e.g., Vistula Lagoon) without leaving the rectangular world |

**Recommendation:** ship **B** unconditionally (it's a refactor with no behavior change and it pays for itself in code-quality + testability), then defer the choice between **A**, **C**, **D**, **F** to a domain-driven decision based on the use case. **E** is parallel and complementary to all others.

The Java-parity question is the single most consequential strategic decision. Options B and E preserve parity. A, C, D, F all break it.

---

## 0 · Where the engine is today (minimal recap)

`osmose/engine/grid.py:16-122` defines a single `Grid` class with hardcoded `(ny, nx)` 2D layout, row-major indexing (`cell_id = y*nx + x`), and 8-connectivity via integer `(dy, dx) ∈ {-1,0,1}²` arithmetic. Cell topology assumptions leak into:

- `osmose/engine/processes/movement.py:33-193` — random walk + map CSV
- `osmose/engine/processes/mortality.py:1788-1805` — cell grouping via `cell_y * grid.nx + cell_x`
- `osmose/engine/output.py:492-640` — NetCDF writer with `(time, species, lat, lon)` dims
- `ui/pages/grid_helpers.py:289-707` — polygon construction from meshgrid
- `osmose/schema/grid.py:5-98` — `grid.nlon`, `grid.nlat`, bounding-box config keys

The engine **already** separates cell-local (predation, forcing, fishing) from cell-topology (movement, neighbors, output) logic. That separation is what makes any of the five options below tractable.

A subtle gotcha caught in the 2026-04-24 study and re-verified for this plan: `Grid.from_netcdf()` at `grid.py:108-112` reads 2D `latitude`/`longitude` arrays from input NetCDFs but immediately *collapses them to 1D* via `lat = lat[:, 0]`. So OSMOSE-Python today **claims** curvilinear support but silently drops the across-row lat/lon variation. Option A starts by fixing this lie.

---

## 1 · Option A — Curvilinear-rectangular only

**What it ships:** OSMOSE-Python correctly consuming a 2D `(ny, nx)` `latitude`/`longitude` NetCDF — i.e., grids where cell centers don't sit on a regular lat-lon lattice. Used by NEMO, ROMS, POM, MOM, FVCOM, and Copernicus-Marine product grids.

**What it does NOT change:** Cell adjacency (still 8-connected `(dy, dx)`), cell ID scheme (still row-major), random walk semantics (still index-space), map CSV format (still 2D matrices). Internally the engine continues to think it has a regular `(ny, nx)` grid.

**Concrete tasks (sized in days):**

| # | Task | Files | Days |
|---|---|---|---|
| A.1 | Stop collapsing 2D lat/lon to 1D in `Grid.from_netcdf` | `grid.py:108-112` | 0.5 |
| A.2 | Store `lat2d`, `lon2d` optional fields on `Grid` | `grid.py:16-44` | 0.5 |
| A.3 | Output writer: emit `latitude(ny,nx)` + `longitude(ny,nx)` per CF when curvilinear | `output.py:492-640` | 1.5 |
| A.4 | UI overlay: pass curvilinear lat/lon through to `polygon_layer` (already partly supported) | `ui/pages/grid_helpers.py:469-518` | 1 |
| A.5 | Schema: add `grid.curvilinear: bool` flag, drop `grid.upleft.*` requirement when true | `osmose/schema/grid.py:5-98` | 0.5 |
| A.6 | Test fixture: ROMS-style 30×40 distorted grid; end-to-end Baltic-shaped run | `tests/test_curvilinear_grid.py` (new) | 2 |
| A.7 | Movement docstring + warning: "displacement is in cell-index space, not km" | `movement.py:163-193` | 0.5 |
| A.8 | Update `docs/parity-roadmap.md`: curvilinear is a Java-divergence | `docs/parity-roadmap.md` | 0.5 |

**Effort:** 7 person-days = **~1.5 weeks**.

**Pros:**
- Smallest change that unlocks coupling to widely-used ocean-model output (NEMO, ROMS).
- No engine logic touched.
- Zero risk to existing rectangular runs.

**Cons:**
- Doesn't help the dispersal-kernel-correctness question — random walk still in index space.
- Forces UI maintainers to handle two polygon-construction code paths.
- Breaks Java parity *the moment a curvilinear grid is loaded* — Java-OSMOSE has no curvilinear support.

**When to pick:** there is a concrete external dataset (e.g., a Copernicus Marine product, a NEMO-Baltic run) that we want to drive Baltic-OSMOSE forcing from, and which provides `(ny, nx)` 2D lat/lon. Otherwise this is busywork until Option B or E lands.

---

## 2 · Option B — Topology abstraction (no new mesh)

**What it ships:** A `GridTopology` protocol that the engine talks to instead of `Grid`. The current rectangular grid becomes the first implementation. *No behavioral change.* Every test passes byte-for-byte. But every place in the engine that previously did `y * nx + x` arithmetic now goes through the protocol.

**Why this is the foundation:** without B, options C/D/F all require 2× more invasive changes. With B, each new topology is a ~200-LOC subclass (plus its movement kernel + UI rendering) and slots into the existing engine.

**Concrete tasks:**

| # | Task | Files | Days |
|---|---|---|---|
| B.1 | Define `GridTopology` Protocol (n_cells, ocean_mask, lat, lon, neighbors, random_neighbor, walk_k_ring, cell_to_xy_or_None) | `osmose/engine/grid_topology.py` (new) | 1 |
| B.2 | Port `Grid` → `RectangularGrid` implementing the protocol | `osmose/engine/grid.py` | 1.5 |
| B.3 | Mortality: replace `cell_y * grid.nx + cell_x` with `topology.cell_id(cy, cx)` | `mortality.py:1788-1805` | 0.5 |
| B.4 | Movement random walk: factor neighbor selection into topology method | `movement.py:163-193` | 1.5 |
| B.5 | Movement maps: keep 2D CSV reader; add per-cell CSV reader; topology decides which | `movement.py:33-103` | 2 |
| B.6 | Output writer: branch on topology kind for spatial NetCDF dimensions | `output.py:492-640` | 1.5 |
| B.7 | UI: introduce `build_grid_layers(topology)` dispatch; rectangular path unchanged | `ui/pages/grid_helpers.py:289-707` | 2 |
| B.8 | Schema: `grid.type ∈ {rectangular}` (placeholder for future values); existing keys unchanged | `osmose/schema/grid.py` | 0.5 |
| B.9 | Property-test parity: pre/post refactor must produce identical biomass + diet outputs on EEC + Baltic 5-y runs | `tests/test_grid_topology_parity.py` (new) | 2 |
| B.10 | Documentation: how to add a new topology | `docs/architecture/grid-topologies.md` (new) | 0.5 |

**Effort:** 13 person-days = **~2.5 weeks**.

**Pros:**
- **No behavior change.** All existing tests stay green; user-visible behavior is unchanged.
- Cleans up the implicit `(y, x)` tuple everywhere — much easier to reason about.
- Pre-pays the refactor cost for *all* later options (C, D, F).
- Can ship independently and ship today if approved.

**Cons:**
- Ships nothing user-visible. Hard to "sell" to a non-engineer stakeholder.
- 2.5 weeks for "no change" feels expensive; need to frame as enabling investment.

**When to pick:** unconditionally. The only argument against B is "we'll never want non-rectangular" — and even then the abstraction tightens an under-modelled boundary.

---

## 3 · Option C — Hexagonal H3 grid (depends on B)

**What it ships:** A `HexGridTopology` implementation backed by Uber's H3 indexing library. Cells are regular hexagons at a chosen resolution (H3 levels 5–7 ≈ 8.5 km², 1.2 km², 0.17 km² respectively). 6-connectivity, uniform nearest-neighbor distance, hierarchical refinement, and battle-tested geographic indexing.

**Why hex over rect:** Cleaner dispersal kernels (no `√2` axial-vs-diagonal asymmetry), 6-connectivity matches biological diffusion better, and H3's hierarchical nesting allows running coarse-then-refined experiments without changing data formats.

**Concrete tasks (after B is shipped):**

| # | Task | Files | Days |
|---|---|---|---|
| C.1 | Add `h3` dependency (Apache 2.0) | `pyproject.toml` | 0.25 |
| C.2 | `HexGridTopology` from a ocean-cell H3-id list | `osmose/engine/hex_grid.py` (new) | 3 |
| C.3 | Hex-aware random walk kernel | `movement.py` | 1 |
| C.4 | Hex-aware map-based movement: per-cell CSV format (cell_h3_id, prob) | `movement.py` + map readers | 2 |
| C.5 | Output writer: UGRID NetCDF for hex (or per-cell flat with H3 ids as a coord) | `output.py` | 2 |
| C.6 | UI: hex polygon layer (6-vertex polygons via h3-to-geo-boundary) | `ui/pages/grid_helpers.py` | 2 |
| C.7 | Forcing ingestion: regrid rectangular NetCDF → hex via area-weighted aggregation | `osmose/engine/resources.py`, `background.py` | 3 |
| C.8 | Movement-statistics parity: validate that walk-on-hex produces dispersion within ε of walk-on-rect for the same biological intent | `tests/test_hex_movement_parity.py` (new) | 4 |
| C.9 | Tooling: `scripts/build_baltic_hex_grid.py` — produce a hex equivalent of the Baltic mask | `scripts/` | 2 |
| C.10 | Re-derive Baltic distribution maps on hex: 9 maps × ~0.5 day each | `data/baltic/` | 5 |
| C.11 | End-to-end Baltic 50-y hex calibration parity check vs rectangular | `tests/integration/` + run | 3 |
| C.12 | Documentation + tutorial | `docs/` | 1 |

**Effort:** 28.25 person-days = **~6 weeks**.

**Pros:**
- Most ecologically-defensible non-rect option (uniform neighbor distance).
- H3 dependency is industrial-strength.
- Hierarchical resolutions enable scale studies "for free" once infrastructure exists.

**Cons:**
- Distribution maps need rebuild from HELCOM/literature footprints (low-leverage but unavoidable).
- Random walk semantics shift: `walk_range=2` means different things on rect vs hex; per-species walk_range likely needs recalibration.
- Forcing regridding (rect→hex) introduces small numerical noise vs the rectangular reference.

**When to pick:** if the modelling motivation is *dispersal-kernel correctness* or *running scale studies* (e.g., what does Baltic look like at H3 res 5 vs 6?). Less compelling if the motivation is coastal refinement (Option D handles that better) or external coupling (Option A is sufficient).

---

## 4 · Option D — Unstructured UGRID mesh (depends on B + C tooling)

**What it ships:** An `UnstructuredGridTopology` that reads a UGRID-CF NetCDF describing arbitrary polygon cells (triangles, quads, Voronoi cells). Mesh generation is *out of scope* — we consume meshes produced by `gmsh`, `jigsaw`, or `OceanMesh2D`.

**Why FEM:** regional refinement. Lagoons (Vistula, Curonian, Pärnu, Gulf of Riga) currently sit in 1–4 cells of the 0.4°×0.3° rectangular grid; with a refined mesh they can have 50+ cells while the open-Baltic basin keeps the same coarse cell count. This is the only path to multi-resolution within OSMOSE.

**Concrete tasks (after B is shipped, ideally after C for tooling reuse):**

| # | Task | Files | Days |
|---|---|---|---|
| D.1 | Add `uxarray` (LGPL-3.0) or hand-rolled UGRID reader | `pyproject.toml`, `osmose/engine/ugrid_io.py` (new) | 4 |
| D.2 | `UnstructuredGridTopology` with ragged neighbor table (CSR-style) | `osmose/engine/unstructured_grid.py` (new) | 5 |
| D.3 | Random walk: graph-step vs km-distance modes; precomputed neighbor distances | `movement.py` | 4 |
| D.4 | Map-based movement: per-cell-ID CSV; back-compat detection vs 2D rect | `movement.py` | 2 |
| D.5 | Output writer: full UGRID-CF emission with `face_node_connectivity` etc. | `output.py` | 4 |
| D.6 | UI: arbitrary-polygon layer; per-cell vertex count varies | `ui/pages/grid_helpers.py` | 3 |
| D.7 | Mesh-quality validation: closed mesh, no degenerate cells, ocean-extent matches | `osmose/engine/ugrid_validation.py` (new) | 2 |
| D.8 | Forcing regridder: rect/curvilinear NetCDF → unstructured (area-weighted) | `resources.py`, `background.py` | 5 |
| D.9 | Pipeline doc: how to generate a Baltic mesh (gmsh + bathymetry inputs) | `docs/recipes/baltic-mesh.md` (new) | 2 |
| D.10 | Distribution map regen pipeline: HELCOM polygon → unstructured per-cell prob | `scripts/regen_maps_unstructured.py` | 5 |
| D.11 | End-to-end refined Baltic mesh: e.g., 0.4° offshore, 0.05° in lagoons | `data/baltic_unstructured/` | 8 |
| D.12 | Performance audit: 10k unstructured cells vs 2k rect cells (benchmarks + UI render) | `tests/perf/` | 3 |
| D.13 | Documentation, tutorial, examples | `docs/` | 3 |

**Effort:** 50 person-days = **~10 weeks** = **~2.5 months**, plus realistic-mesh build of ~2–4 weeks (collaboration with HELCOM bathymetry holders), so ~3–4 months calendar.

**Pros:**
- Only path to coastal/lagoon refinement.
- Aligns OSMOSE with modern coastal-ocean-model best practice.
- UGRID is an open CF standard, so meshes are portable across tools.

**Cons:**
- Largest engineering surface; mesh generation is upstream, mesh QA is on us.
- All distribution maps require rebuild on the new mesh — that's where most calendar time goes.
- Performance: 10× more cells in lagoons may dominate runtime even though they hold 5% of biomass.
- All Java parity gone; this is a research fork.

**When to pick:** *only* if Baltic-OSMOSE (or another regional config) needs lagoon/estuary resolution that rectangular grids cannot provide. Don't pick this for "nice to have" — the calendar cost is real.

---

## 5 · Option E — Pre-processor only (no engine change)

**What it ships:** A standalone pipeline `osmose/preprocess/mesh_to_rect.py` that ingests a high-resolution mesh dataset (HELCOM bathymetry, ERA5 forcing, biological survey points, etc.) and *aggregates* it onto OSMOSE's existing rectangular grid via area-weighted or point-in-polygon mapping. The engine is untouched.

**Concrete tasks:**

| # | Task | Files | Days |
|---|---|---|---|
| E.1 | Mesh reader (UGRID, gmsh `.msh`, ESRI shapefiles) | `osmose/preprocess/mesh_io.py` (new) | 2 |
| E.2 | Area-weighted aggregator: mesh field → rect cell | `osmose/preprocess/aggregate.py` (new) | 2 |
| E.3 | Point-cloud aggregator (e.g., HELCOM survey points → cell density) | `osmose/preprocess/aggregate.py` | 1 |
| E.4 | CLI wrapper: `python -m osmose.preprocess.mesh_to_rect input.nc output.csv` | `scripts/mesh_to_rect.py` | 1 |
| E.5 | Tests on Baltic ICES BITS hauls + HELCOM bathymetry | `tests/preprocess/` | 2 |
| E.6 | Documentation cookbook: "use HELCOM data with OSMOSE" | `docs/recipes/helcom-to-osmose.md` | 1 |

**Effort:** 9 person-days = **~2 weeks**.

**Pros:**
- Zero engine risk. No regressions possible.
- Java-parity preserved.
- Solves the most common practical motivation (using HELCOM/ICES high-resolution data without forking the engine).
- Complementary to all other options — the same pre-processor works on rect, hex, or unstructured engines.

**Cons:**
- Doesn't address dispersal-kernel correctness or coastal refinement *in the engine*.
- Aggregation loses information; unwary users may misinterpret the results.

**When to pick:** unconditionally as a "Tier 0" deliverable. It costs ~2 weeks, has no risk, and immediately makes external data sources usable. Shipping E does not preclude any other option.

---

## 6 · Option F — Nested-rectangular subgrids (AMR-lite)

**What it ships:** Within an OSMOSE rectangular grid, designate one or more rectangular zones to be sub-divided at higher resolution. The engine internally treats the world as a union of rectangular sub-grids glued together at boundaries. Cells in the lagoon zone are 5× smaller; cells in the open Baltic stay coarse. No new topology library; arithmetic stays integer.

**Why this might be attractive:** delivers a key ecological motivation (coastal refinement) without leaving the rectangular world. Avoids hex/UGRID dependency entirely. Sits between B and D in cost.

**Concrete tasks (after B):**

| # | Task | Files | Days |
|---|---|---|---|
| F.1 | `NestedRectGridTopology` — a coarse base + N refinement zones | `osmose/engine/nested_grid.py` (new) | 3 |
| F.2 | Cell-ID scheme: contiguous flat IDs, with boundary-face mappings between levels | `nested_grid.py` | 3 |
| F.3 | Refinement-zone schema: `grid.refine.zone.N.bbox`, `grid.refine.zone.N.factor` | `osmose/schema/grid.py` | 1 |
| F.4 | Movement: cross-level cell transitions handled by neighbor table | `movement.py` | 3 |
| F.5 | Map-based movement: refine zones get inflated CSV submatrices; engine stitches | `movement.py` | 3 |
| F.6 | Forcing regrid: parent-cell value broadcast to children unless per-child file present | `resources.py`, `background.py` | 3 |
| F.7 | Output writer: nested NetCDF (group per refinement zone, or flat with cell_id coord) | `output.py` | 3 |
| F.8 | UI: render parent + child polygons with overlay; deck.gl handles the levels | `ui/pages/grid_helpers.py` | 2 |
| F.9 | Test fixture: 1 refinement zone in Baltic Vistula Lagoon | `tests/test_nested_grid.py` (new) | 3 |
| F.10 | Documentation + worked example | `docs/recipes/nested-grids.md` (new) | 1 |

**Effort:** 25 person-days = **~5 weeks**.

**Pros:**
- Solves coastal-refinement motivation without UGRID complexity.
- Intermediate cost between B and D.
- All math stays integer-arithmetic — fast and easy to optimize with Numba.
- Distribution maps for refined zones can be regenerated only for those zones.

**Cons:**
- Cell-boundary mass conservation is fiddly (movement crossing levels needs careful accounting).
- Less standard than UGRID — not a recognized format, hard to share meshes with collaborators.
- May not generalize: if you later need *non-rectangular* refinement (curved coastline), F doesn't help and you're back to D.

**When to pick:** the Baltic case really wants Vistula / Curonian / Gulf of Riga lagoon resolution but a) the team can't take 3–4 months for D, and b) the refinement zones are clean rectangles or we're willing to over-cover them.

---

## 7 · Cross-option engineering concerns

### 7.1 Distribution maps need regeneration on any non-rect topology

The 2026-04-21 work to differentiate Baltic distribution maps (sp4, sp5, herring spring/autumn, smelt spawning, life-stage containment) was done on rectangular cells. On hex/unstructured/nested grids each map needs rebuild from the source HELCOM polygons + literature. The pipeline is the same (point-in-polygon against a cell-center list) — only the cell-center list changes. Estimated **~5 person-days per topology** if the pipeline is preserved.

### 7.2 Random walk semantics need a per-topology decision

`walk_range=2` means a 5×5 box on rect, an H3 2-ring (~19 cells) on hex, and "graph distance ≤ 2" on unstructured. Biological intent (typical home-range size in km) doesn't translate automatically. Recommendation: introduce a `walk_range_km` config key alongside the existing index-space `walk_range`, and let each topology convert. This is a one-time cost amortized across C, D, F.

### 7.3 Java parity is broken by A, C, D, F (preserved by B + E)

The Java OSMOSE engine is rectangular-only. Any engine-side topology beyond rectangular forks Python from Java numerically. This is **strategic**, not technical: are we still a parity port, or are we a divergent evolution? A clean answer here makes the option-selection matrix obvious. We have no current pressure to maintain parity (no recent reports of Python/Java divergence shipping production results), but historically parity has been valuable for cross-validation.

### 7.4 Multi-topology coexistence

Should the engine support *both* rectangular and hex configs in one binary, or is each topology a build-time choice? Recommendation: runtime dispatch via `GridTopology` protocol (Option B's natural shape). The protocol cost is paid once; users pick at config-load time which topology to load.

### 7.5 Forcing & calibration

All currently-calibrated parameters (mortality, fishing, ingestion, fecundity) are per-species scalars and *do not depend on grid topology.* Switching topology does not invalidate the 2026-04-22 phase 1 / 2026-04-24 phase 2 / 2026-04-27 phase 12 calibration as a starting point. What needs regeneration: distribution maps, walk_range (if moving from index-space to km-space), and possibly LTL accessibility coefficients (since the spatial averaging implicit in rectangular grids breaks).

### 7.6 Performance

A 2,016-cell rectangular Baltic grid currently takes ~6 minutes for a 50-y sim under predator-active calibration contention. Estimated post-refactor / per-topology slowdowns:

- B alone: 0% (refactor only; numba-compiled hot paths unchanged).
- C (H3, ~5,000 cells at res 5): +20–30% (more cells, but flatter neighbor lookup).
- D (unstructured, ~10,000 cells with 5× refined lagoons): **+100–150%**, dominated by lagoon cells holding most schools post-spawning.
- F (nested rect, 1 zone at 5× refinement): +30–50%.
- E: zero engine impact (pre-processor).

These are guesses; performance spikes are needed before committing to D.

---

## 8 · Recommended sequencing

If we're going to do this at all, the low-regret sequence is:

```
                         ┌─ A (curvilinear)        ←  if NEMO/ROMS coupling motivated
   ┌─ E (pre-processor)  │
   │                     │
   │                     ├─ C (hex H3)             ←  if dispersal-kernel motivated
B ─┤                     │
   │                     ├─ D (unstructured)       ←  if coastal-refinement motivated
   │                     │
   └─ (status quo)       └─ F (nested rect)        ←  if coastal-refinement & rect-only world preferred
```

Concretely:

1. **Now (week 1–2):** Ship **E**. ~2 weeks; immediate value; no engine risk; complementary to everything else.
2. **Now (week 1–4):** Ship **B**. ~3 weeks; the prerequisite refactor; pays itself back in code clarity.
3. **Decision point** (after B + E land): pick one of A, C, D, F based on the dominant motivation:
   - External-data coupling → **A** (1.5 wk).
   - Dispersal correctness / scale studies → **C** (6 wk).
   - Lagoon refinement, willing to take the calendar hit → **D** (3–4 mo).
   - Lagoon refinement, want to stay in rect-world → **F** (5 wk).
4. The chosen option ships in its quoted timeline.
5. Re-evaluate: did the new topology produce ecologically meaningfully different results, and are users actually picking it? Greenlight or shelve.

---

## 9 · Decision criteria (to fill in before committing)

Answer these *before* picking an option:

| Question | A | B | C | D | E | F |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Do we accept Java-parity divergence? | required | NO | required | required | NO | required |
| Is there a concrete external dataset we want to consume? | strong-yes | irrelevant | helpful | helpful | strong-yes | helpful |
| Is dispersal-kernel correctness the motivation? | no | no | yes | yes | no | no |
| Is coastal-lagoon resolution the motivation? | no | no | partial | yes | no | yes |
| Calendar budget available | 2 wk | 3 wk | 2 mo | 4 mo | 2 wk | 5 wk |
| Distribution-map regen acceptable | no | no | yes (5 days) | yes (5 days+) | no | yes (1 zone) |

---

## 10 · Risks & open questions

1. **Strategic risk (high impact):** Java parity. A unilateral fork without an explicit agreement could surprise users. Resolve before any engine-side option (A, C, D, F).
2. **Schedule risk (medium impact):** D's 3–4 month estimate is optimistic; mesh QA + distribution-map regen frequently slip. Budget +50% on D.
3. **Performance risk (medium impact):** D unstructured-mesh perf has not been benchmarked. Spike a 50-y Baltic-on-mesh run in week 2 of D before committing the rest of the engineering.
4. **Calibration risk (medium impact):** Walk-range semantics shift. Plan a per-species walk_range recalibration as part of any topology switch (1 week of compute, embedded inside C/D/F's calendar).
5. **Open question:** Is the Baltic team's actual unmet need *coastal refinement* (favors D/F) or *external-data coupling* (favors A/E)? This drives the option choice and should be answered in conversation, not in a planning doc.
6. **Open question:** Does HELCOM publish UGRID-format Baltic bathymetry meshes? If yes, D becomes much cheaper. If no, mesh-generation upstream cost is real.

---

## 11 · What this plan does NOT decide

- *Whether* to do any of this. The status quo (rectangular only, parity-port to Java) is a coherent option.
- The exact H3 resolution (Option C). Suggest spike at res 5, 6, 7 and pick based on cell count + UI render time.
- Mesh-generation tooling for Option D — `gmsh` vs `OceanMesh2D` is itself a multi-day evaluation.
- Whether to expose topology selection in the Shiny UI (probably yes, but UI work is bundled into each option's tasks).

---

## 12 · References

- 2026-04-24 feasibility study: `docs/superpowers/specs/2026-04-24-non-rectangular-grids-exploration.md` — survey of grid assumptions in the codebase + cost-per-subsystem table.
- UGRID CF convention: https://ugrid-conventions.github.io/ugrid-conventions/
- H3 geospatial indexing: https://h3geo.org/
- `uxarray` library: https://uxarray.readthedocs.io/
- `gmsh` mesh generator: https://gmsh.info/
- `OceanMesh2D` (MATLAB / Python mesh generator for coastal models): https://github.com/CHLNDDEV/OceanMesh2D
- Existing OSMOSE-Python grid code: `osmose/engine/grid.py`, `osmose/engine/processes/movement.py`, `ui/pages/grid_helpers.py`, `osmose/schema/grid.py`.

---

*Author note: this document is a planning artifact, not a design spec. The expected next step is a 30-minute conversation to answer the §9 decision-criteria table, after which 1 or 2 options become obviously correct.*
