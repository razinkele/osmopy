# Non-rectangular grids for OSMOSE-Python — feasibility study

> **Status:** exploration (2026-04-24). No implementation committed.
> Purpose: map what it would take to support hexagonal or unstructured (FEM-style) grids, identify engine changes, estimate cost per subsystem, and recommend a migration path.
> Scope: the Python engine, config schema, outputs, and UI. The Java engine is not in scope.

---

## TL;DR

OSMOSE-Python today is hard-wired to a rectangular `(nlat × nlon)` grid with row-major indexing and 8-connectivity via integer `(dr, dc)` arithmetic. The engine already separates *cell-local* logic (predation, forcing, fishing) from *cell-topology* logic (movement, neighborhood, output writing). That separation makes three generalization paths technically possible, in increasing order of engineering cost:

1. **Curvilinear rectangular grid** (still `(ny, nx)`, but distorted cells — e.g., a polar or tripolar ocean grid). **Cost: LOW–MEDIUM.** Already partially supported via `NcGrid` which reads 2D `latitude`/`longitude` arrays. Only output-writing and UI polygon construction need rework.
2. **Regular hexagonal grid.** **Cost: MEDIUM.** Re-indexing (axial or offset coordinates), 6-connectivity instead of 8-, new random-walk kernel, new UI polygons. Predation/forcing/fishing unaffected once cell IDs are abstracted.
3. **Unstructured mesh (triangles or arbitrary polygons, FEM-style).** **Cost: HIGH.** Grid is a graph; every `(dr, dc)` arithmetic disappears; movement maps become per-cell arrays indexed by cell ID; a substantial config+I/O+UI reworking is required.

The highest barriers across all three options are (a) random-walk movement, (b) map-based movement CSV format, and (c) UI rendering. The easiest are predation, forcing, and per-cell fishing — they are already cell-local.

**Recommendation:** introduce a `GridTopology` abstraction now (even if still rectangular under the hood) to isolate the cells-with-arithmetic assumption. Then iterate: curvilinear first (nearly free), hex second, FEM as a longer-term research track.

---

## 1 · Why consider non-rectangular cells?

Four ecological / numerical motivations that come up periodically for OSMOSE-type models:

1. **Coastline fidelity.** A 0.4° × 0.3° rectangular grid on the Baltic produces 1,384 land cells and 616 ocean cells; the coastline is blocky and a ~25 km coastal strip often straddles a single cell. Finer rectangular grids would quadruple the cell count for a linear improvement. Hex or adaptive meshes can resolve coastline at higher effective resolution for a given cell budget.
2. **Uniform neighbor distance.** Rectangular 8-connectivity has `√2` distance asymmetry between axial and diagonal neighbors. Hexagonal 6-connectivity has uniform nearest-neighbor distance, which makes dispersal kernels (random walk) interpretable without correction.
3. **Curvilinear coordinates.** High-latitude ocean models (ROMS, NEMO) use curvilinear grids to avoid the pole singularity. Forcing from such models currently has to be regridded onto OSMOSE's rectangular grid, losing mass-conservation properties.
4. **Regional refinement.** FEM/unstructured meshes let you put high resolution where it matters (coast, river mouths, spawning grounds) and coarse resolution offshore. For Baltic-OSMOSE, this would let us resolve Vistula Lagoon, Curonian Lagoon, Pärnu Bay coastal nurseries at fine scale without blowing up the ocean-basin cell count.

These are not enough to justify the change on their own. But they matter for Baltic-like setups where coastal resolution is the limiting factor.

---

## 2 · What the code assumes today (survey results)

Based on a codebase survey conducted 2026-04-24:

| Subsystem | File | Grid assumption | Generalization cost |
|---|---|---|---|
| Grid representation | `osmose/engine/grid.py:16-122` | 2D `(ny, nx)` arrays; row-major 1D via `cell_id = y*nx + x`; 8-connectivity by `for dy, dx in {-1,0,1}²` | MEDIUM |
| Random walk movement | `osmose/engine/processes/movement.py:163-193` | Cartesian `L∞` displacement; `walk_range` is an integer bounding box | **HIGH** |
| Map-based movement | `osmose/engine/processes/movement.py:33-103` | CSV is `(ny, nx)` matrix; rejection-samples from 2D probability grid; BFS patch uses `grid.neighbors()` | **HIGH** |
| Predation | `osmose/engine/processes/predation.py` | Per-cell matching; no neighbor interaction | LOW |
| LTL/background forcing | `osmose/engine/resources.py`, `osmose/engine/background.py` | Flat per-cell biomass; time interpolation only; NetCDF loaded by `(time, lat, lon)` | LOW |
| Fishing spatial maps | `osmose/engine/config.py:1417-1436` | CSV as `(ny, nx)`; per-cell rate lookup | LOW |
| Non-spatial output | `osmose/engine/output.py:23-91` | None | — |
| Spatial output | `osmose/engine/output.py:492-640` | NetCDF dims `(time, species, lat, lon)`; rectangular coordinates | MEDIUM |
| UI overlay rendering | `ui/pages/grid_helpers.py:289-518, 627-707` | Rectangular polygons from meshgrid; 4-corner construction | **HIGH** |
| Mask | `ui/pages/grid_helpers.py:117-138`, `osmose/engine/grid.py:74-115` | Per-cell boolean, CSV `(ny, nx)` or NetCDF 2D | MEDIUM |
| Config schema | `osmose/schema/grid.py:5-98` | `grid.nlon`, `grid.nlat`, bounding-box corners | MEDIUM |

The full-detail survey with file:line references lives in the notes that generated this document.

Key structural observation: **the engine already separates cell-local logic from cell-topology logic.** Predation, fishing, and forcing never ask "what are my neighbors?" — they read and write per-cell values. Movement is the one place that cares about topology (random walk) and 2D layout (map CSVs). The UI also cares, because it draws polygons.

---

## 3 · Option A — Curvilinear rectangular grid

**What this is:** Keep the `(ny, nx)` 2D indexing. Drop the requirement that cell-center longitudes are equal-spaced; drop the requirement that cells are axis-aligned rectangles. Cell `(i, j)` has arbitrary `lat[i, j]`, `lon[i, j]` — the standard curvilinear convention used by ROMS, NEMO, POM, and Copernicus Marine products.

**Why it's cheap.** `Grid.from_netcdf()` at `osmose/engine/grid.py:74-115` already reads 2D `latitude` / `longitude` arrays. The engine indexes cells by `(i, j)` pair, not by coordinate. Internal semantics don't change.

**What changes:**

- **Mask loader (`grid.py`):** already handles 2D lat/lon — no change.
- **Movement random walk:** still valid if interpreted in `(i, j)` index space (not physical km). Current code does this. But displacement is now anisotropic in km — cell `(i, j)` to `(i+1, j)` may be 5 km in one place and 50 km elsewhere. Users need to know this. *Change: add a docstring + optional per-cell km-weighted walk mode.*
- **Movement map CSV:** still valid — each row/column stays an (i, j) index.
- **Forcing / fishing:** unchanged.
- **Spatial output:** today's NetCDF writer assumes 1D `lat(ny,)` and 1D `lon(nx,)`. For curvilinear grids, needs to write `latitude(ny, nx)` and `longitude(ny, nx)` per CF-conventions. *Change: ~30 lines in `osmose/engine/output.py:492-640`.*
- **UI rendering:** `build_netcdf_grid_layers` in `ui/pages/grid_helpers.py:469-518` already handles 2D lat/lon via finite-difference cell edges. Will work for smooth curvilinear grids, may look jagged for highly distorted ones. *Change: none needed; later, could add a proper Voronoi-of-centers fallback.*
- **Config:** add a `grid.curvilinear = true` flag; remove the `grid.upleft.*` requirement when set.

**Rough cost:** **1–2 engineer-weeks.** Most effort is in the output writer and a test suite verifying a ROMS-style input runs through unchanged. This is the right first step regardless of where we go next — it forces us to stop assuming `dx = constant`.

---

## 4 · Option B — Regular hexagonal grid

**What this is:** All cells are regular hexagons of a fixed size, tiled across the domain. Each cell has exactly 6 neighbors. Indexing uses axial coordinates `(q, r)` (efficient for storage) or offset coordinates (easier visual mapping). Well-studied libraries: `h3` (Uber, millions of users), `astar` for pathing, `pyproj` for cell-to-lat/lon projection.

**Why hex over rectangular:**
- Uniform nearest-neighbor distance → dispersal kernels are cleaner.
- Natural 6-connectivity matches biological diffusion models better than rectangular 8-connectivity.
- H3 has hierarchical nesting (resolution levels 0–15), making mesh refinement straightforward.

**What changes:**

- **Grid class (`osmose/engine/grid.py`):** replace 2D array storage with an explicit cell table:
  ```python
  class HexGrid:
      cells: NDArray[np.int64]          # shape (n_cells,), the cell IDs
      lat: NDArray[np.float64]          # shape (n_cells,)
      lon: NDArray[np.float64]          # shape (n_cells,)
      ocean_mask: NDArray[np.bool_]     # shape (n_cells,)
      neighbors: NDArray[np.int64]      # shape (n_cells, 6), -1 for missing
  ```
  Neighbors are computed once at load time (via `h3.k_ring(cell, 1)` or direct hex arithmetic). *Effort: ~200 LOC new file, or new branch in `grid.py`.*
- **Random walk:** replace `(dr, dc) ∈ {-1,0,1}²` with random neighbor selection: `next_cell = neighbors[current_cell, rng.integers(0, 6)]`. The `walk_range` parameter becomes a *k-ring* radius (cells within k steps). *Change: ~40 LOC in `movement.py:163-193`.*
- **Map-based movement:** CSV format changes. Today each row/column is an `(i, j)` grid cell; for hex, a single column with `n_cells` rows indexed by cell ID is the minimum viable format. *Change: ~80 LOC to support a new CSV schema + back-compat detection.*
- **Predation / forcing / fishing:** no change to logic. They operate on flat cell IDs; today those happen to be `y*nx+x`, tomorrow they're H3 indices. Everything that ingests a NetCDF with `(time, n_cells)` or `(time, lat, lon)` needs to be told which representation the grid uses. *Change: minor — thread the grid representation through a few ingestion functions.*
- **Spatial output:** write UGRID-convention NetCDF with a `mesh` variable + `face_node_connectivity` + per-face data vars. `xarray` + `uxarray` supports this natively. *Change: ~60 LOC in `output.py`, pick up a dependency (`uxarray` or write UGRID manually with `xarray.Dataset`).*
- **UI rendering:** per-cell polygon data precomputed at grid load — 6 vertices per hex from hex-to-geographic projection. `polygon_layer` in `shiny-deckgl` accepts arbitrary polygons, so the primitive is already there. *Change: ~80 LOC in `ui/pages/grid_helpers.py`, replacing `build_grid_layers` with a `build_hex_grid_layers` branch.*
- **Config:** new key `grid.type ∈ {rectangular, hex}` plus `grid.hex.resolution` (0–15 for H3) or `grid.hex.size_km`. Mask + bathymetry load from per-cell CSV.

**Rough cost:** **4–6 engineer-weeks.** The biggest risk is validating that movement statistics (dispersion rate, home-range size) match the rectangular reference within calibration tolerance. This requires a parity test against a known rectangular case using the same underlying dispersal process.

**External dependency:** `h3` (Apache 2.0) is a one-line add to `pyproject.toml`. Alternative: hand-rolled axial coordinates (~100 LOC, no dependency).

---

## 5 · Option C — Unstructured mesh (FEM-style)

**What this is:** Cells are arbitrary polygons (triangles, quads, Voronoi cells). Topology is a graph — each cell knows its neighbors by ID, there is no coordinate arithmetic. Widely used in coastal ocean models (ADCIRC, FVCOM, SCHISM) and in finite-element fluid dynamics. The standard file format is UGRID CF convention on top of NetCDF.

**What changes (on top of Option B):**

- **Grid class:** same general structure as hex (`cells`, `lat`, `lon`, `ocean_mask`, `neighbors`), but neighbors is a ragged array (cells can have 3, 4, 5, 6, … neighbors). Store as `(cell_ids, neighbor_offsets)` CSR-style. *Effort: ~300 LOC including a UGRID reader.*
- **Random walk:** same as hex — pick random neighbor. But now walk distance in km depends on cell, so "walk_range=2" is less interpretable. Need to decide: is walk_range in *graph steps* or *km*? If km, need a precomputed neighbor-distance matrix (sparse). *Change: ~100 LOC.*
- **Map-based movement:** CSV must become per-cell (`n_cells`-row column). Same as hex.
- **Predation / forcing / fishing:** same as hex — unchanged logic, flat cell IDs.
- **Spatial output:** UGRID NetCDF with `face_node_connectivity`, `node_coordinates`, etc. `xarray` does not natively write UGRID; `uxarray` does. *Change: add `uxarray` dependency or roll our own.*
- **Mask generation:** offline mesh-generation pipeline (`gmsh`, `jigsaw`, `OceanMesh2D`). OSMOSE doesn't generate the mesh — it consumes one.
- **UI rendering:** render polygons from UGRID `face_node_connectivity`. Per-cell polygon count varies. `polygon_layer` handles this. *Change: ~150 LOC.*
- **Config:** new key `grid.type = unstructured` + `grid.mesh.file` pointing to a UGRID NetCDF. Remove all `nlon`/`nlat` expectations.
- **Validation:** mesh sanity (closed, no degenerate cells, matches ocean extent) — probably integrate `meshio` or `pyvista` for validation.

**Rough cost:** **3–4 engineer-months.** Not because any single piece is hard, but because the surface area is large: mesh reader, mesh QA, UGRID output, non-uniform dispersal kernels, UI polygon rendering, documentation, examples, and retrofitting every existing test fixture. Plus: mesh generation is an out-of-tree problem we don't want to own.

**External dependencies:** `uxarray` (PyPI, LGPL-3.0) for UGRID I/O; optionally `meshio` for format conversions. Neither is heavy.

---

## 6 · Cross-cutting engineering concerns

### 6.1 Abstraction layer first

Whatever we do, the cleanest first step is to introduce a `GridTopology` interface that the engine talks to *instead of* the current `Grid` with its `(ny, nx)` assumptions. Something like:

```python
class GridTopology(Protocol):
    n_cells: int
    ocean_mask: NDArray[np.bool_]        # shape (n_cells,)
    lat: NDArray[np.float64]             # shape (n_cells,)
    lon: NDArray[np.float64]             # shape (n_cells,)
    def neighbors(self, cell_id: int) -> NDArray[np.int64]: ...
    def random_neighbor(self, cell_id: int, rng) -> int: ...
    def walk_k_ring(self, cell_id: int, k: int) -> NDArray[np.int64]: ...
```

`RectangularGrid` becomes the first implementation. Engine callers stop using `(y, x)` tuples and use flat cell IDs. This refactor is purely mechanical — no test should change in outcome — and it exposes every place that currently assumes grid shape. Estimated **2 engineer-weeks** for the refactor alone.

After this, new topologies (hex, unstructured) slot in as additional implementations of the `GridTopology` protocol.

### 6.2 Movement subsystem is the main surgery

The survey pegged movement as HIGH cost on every non-rectangular option. Two specific sub-problems:

1. **Random walk semantics.** "walk_range = 2" today means a 5×5 integer box. On a hex grid it could mean "2-ring" (~19 cells). On an unstructured mesh it could mean "graph distance ≤ 2" or "≤ 2 × cell_size km". These are biologically different. Each species' calibrated walk_range value may need rescaling — probably a per-species calibration pass.
2. **Map-based movement CSV format.** Today maps are 2D matrices that align with the rectangular grid. For hex / unstructured, maps become per-cell probability vectors. The engine should keep accepting the 2D format for rectangular grids (back-compat) and add a new "per-cell" format keyed by cell ID. The existing `smelt_spawning.csv` and friends would then need regeneration for the new topology — reusing the HELCOM-literature-derived cell lists from the 2026-04-21 map-validation session.

### 6.3 Mass conservation and bathymetry

OSMOSE treats cells as abstract "schools live here" containers, not physical volumes. So strictly speaking, cell *area* doesn't enter the biomass arithmetic — biomass per cell is an abundance not a density. Two places where area would matter:

- **Forcing with density units.** If LTL biomass were in `mg/m³` instead of `tonnes/cell`, cell area matters. Today it's the latter, so this isn't an issue.
- **Fishing rate spatial aggregation.** Fishing is per-cell with a mortality rate; if effort is expressed as `boats/km²`, you need cell area.

Neither is currently a problem but will surface if OSMOSE tries to couple with physical/biogeochemical models more tightly.

### 6.4 Bathymetry / depth

OSMOSE-Python doesn't currently use depth at all. That's orthogonal to the grid-topology question but would become relevant if we add depth-stratified predation (cod in deep basins, flounder on the shelf). For unstructured meshes with depth, you might want voxel-level meshes (rare, but 3D versions of SCHISM exist). Deferred.

### 6.5 Calibration compatibility

All existing Baltic calibration parameters (mortality rates, fishing rates, etc.) are **per-species scalars** — they don't depend on grid topology. So a topology switch doesn't invalidate the 2026-04-22 phase 1 or 2026-04-24 phase 2 calibration as a starting point. Species-distribution maps (`smelt_spawning.csv` etc.) would need regeneration, not recalibration.

### 6.6 Parity with the Java reference

The Java OSMOSE engine is the reference implementation. It's also rectangular-only (as of the versions we track). Switching Python to a non-rectangular topology breaks numerical parity with Java. This is an explicit non-goal for this feature — it would fork the Python engine from being a parity port to being a divergent evolution. That's a strategic decision, not a technical one.

### 6.7 UI performance

`shiny-deckgl` renders polygons with GPU acceleration. 10,000 hex cells renders fine in our testing. 100,000+ unstructured cells may stutter; mitigate by viewport culling or switching to `bitmap_layer` for very fine meshes. Not a blocker at OSMOSE-scale domains.

---

## 7 · Migration path — recommendation

If this is worth pursuing, the low-regret sequence is:

1. **Phase 0 — Abstraction (2 weeks).** Introduce `GridTopology` protocol; port `RectangularGrid` to implement it; refactor every caller to use flat cell IDs; existing test suite still passes byte-for-byte. No functional change, but the change-point becomes visible.

2. **Phase 1 — Curvilinear NcGrid validation (1 week).** End-to-end test of running OSMOSE-Python against a ROMS-style 2D `(ny, nx)` grid with distorted cells. Validate spatial output writes CF-compliant curvilinear NetCDF. Deliver: first non-trivial non-uniform-dx grid running through the engine.

3. **Phase 2 — Hex grid (4–6 weeks).** Implement `HexGrid` on top of `h3`. Validate: run a Baltic-shaped hex grid through the existing calibration pipeline. Compare biomass trajectories against the rectangular-grid equivalent. If within calibration tolerance, hex is viable. Re-derive Baltic distribution maps on hex and re-run calibration.

4. **Phase 3 — Unstructured mesh (3–4 months).** Only if Phase 2 proves hex is worth the modeling effort. Implement `UnstructuredGrid` reading UGRID NetCDF. Use Baltic as a test bed with a mesh refined in coastal lagoons. Major downstream consequence: every distribution map needs a mesh-aware regeneration. Publish as a research track, not a stable feature.

Total horizon: **~6 months** if we commit fully; **~3 weeks** for Phase 0+1 alone, which already delivers real value (proper curvilinear grid support) and opens the door to later phases without prejudging them.

---

## 8 · What we would *not* get from this

Worth stating explicitly so the scope stays honest:

- **Not more accurate ecology without recalibration.** All species distribution maps + possibly walk_range would need regeneration for the new topology. The gain is *spatial resolution where it matters*, not automatic model improvement.
- **Not Java parity.** Java OSMOSE is rectangular. Any non-rectangular Python grid forks from Java.
- **Not mesh generation.** OSMOSE consumes meshes; we rely on `gmsh` / `jigsaw` / `OceanMesh2D` upstream. Our grid files are inputs.
- **Not solving the Baltic biomass-calibration gap.** That's a parameter-fitting problem (see 2026-04-24 Tier 1 plan), not a grid-resolution problem. A finer grid would expose coastal heterogeneity that the model doesn't currently represent, but the out-of-range biomasses at ×100+ target aren't caused by grid coarseness.

---

## 9 · Open questions

1. Is Java-parity breakage acceptable? (Strategic, needs project-level decision.)
2. Do we need simultaneous support for rectangular + non-rectangular, or a clean migration where only one topology exists at a time? (Config schema implication.)
3. Are there existing UGRID-format Baltic meshes we can reuse (e.g., from HELCOM bathymetry work)?
4. Is hex H3 at resolution 5 (~8.5 km²) or 6 (~1.2 km²) the right scale for Baltic? (h3 resolution choice).
5. Would SIMD-friendly layouts (StructureOfArrays of cell attributes) give enough speedup to offset the indirection cost of a neighbor graph? (Performance spike needed.)

---

## References

- Survey notes behind this document: generated by an Explore subagent on 2026-04-24, documenting file:line references for every grid-assumption in the codebase.
- UGRID CF convention: https://ugrid-conventions.github.io/ugrid-conventions/
- H3 geospatial indexing: https://h3geo.org/
- `uxarray`: UGRID-aware xarray extension, NSF-funded: https://uxarray.readthedocs.io/
- CMEMS Baltic bathymetry (for potential mesh-refinement sources): via Copernicus Marine Service.
- Existing OSMOSE-Python grid code: `osmose/engine/grid.py`, `osmose/engine/processes/movement.py`, `ui/pages/grid_helpers.py`, `osmose/schema/grid.py`.

---

*Author's note: this document is a scoping study, not a design spec. The next step — if pursued — is Phase 0 (the `GridTopology` abstraction), which has value on its own (cleaner code, easier testing) even if the non-rectangular work never happens.*
