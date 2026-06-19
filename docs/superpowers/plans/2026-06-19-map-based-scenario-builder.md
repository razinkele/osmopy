# Map-Based Scenario Builder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "Map Builder" Shiny page to author OSMOSE spatial grid maps (species distribution / land mask / generic zone) by drawing polygons + brushing cells on the loaded config's georeferenced grid, saving them as the engine's `;`-separated grid CSVs with type-aware config wiring.

**Architecture:** A pure, browser-free core (`osmose/maps/builder.py`: grid geometry, numpy ray-cast rasterization, paint/erase/mask ops, CSV (de)serialization with the south-row-0 flip, config-key wiring, save orchestration) + a thin Shiny page (`ui/pages/map_builder.py`: `shiny_deckgl` interactive map with stage-then-Apply polygon draw + cell brush, reactive grid state, save). All testable logic lives in the core; the deck.gl draw/pick seam is the only e2e-only part.

**Tech Stack:** Python 3.12, numpy (declared dep — no matplotlib), Shiny for Python, `shiny_deckgl`, pytest + Hypothesis. Run with `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-19-map-based-scenario-builder-design.md` (approved, 3-round-reviewed).

## File structure

- Create `osmose/maps/__init__.py`, `osmose/maps/builder.py` — the pure core.
- Modify `ui/pages/grid_helpers.py` — DRY: `build_grid_layers` calls `GridSpec.cell_polygon` (the ONLY shipped-code change; output must stay byte-identical, `tests/test_ui_grid.py` is the characterization guard).
- Create `ui/pages/map_builder.py` — the Shiny page (`map_builder_ui`, `map_builder_server`).
- Modify `app.py` — 4 nav touch-points (import, `nav_panel`, server call, JS nav-order array).
- Tests: `tests/test_maps_builder.py` (pure core), `tests/test_ui_map_builder.py` (save_map orchestration + page import + nav registration).

## Key verified facts

- Map CSV: `nlat` rows × `nlon` cols, `;`-separated, `-99` land. **File is SOUTH-row-0** (`movement_maps._load_csv_grid(path, ny, nx)` does `grid_row = ny-1-csv_row_idx`); **in-memory is NORTH-row-0** (matches `build_grid_layers`). `to_csv_text`/`from_csv_text` `np.flipud`.
- Cell geometry: `dx=(lr_lon-ul_lon)/nlon`, `dy=(ul_lat-lr_lat)/nlat`; cell `(r,c)` corners `[[lo0,la0],[lo1,la0],[lo1,la1],[lo0,la1]]` with `lo0=ul_lon+c*dx`, `la0=ul_lat-r*dy`, `lo1=lo0+dx`, `la1=la0-dy`; center `(ul_lat-(r+0.5)*dy, ul_lon+(c+0.5)*dx)`.
- Movement keys (single species per map; lastyear inclusive): `movement.species.map{N}`, `movement.file.map{N}`, `movement.steps.map{N}`, `movement.initialage.map{N}`, `movement.lastage.map{N}`, `movement.initialyear.map{N}`, `movement.lastyear.map{N}`, `movement.years.map{N}`. Discovery iterates `movement.species.map{N}`. (NOT `movement.map{N}.*`.)
- `shiny_deckgl` (async): `MapWidget.enable_draw(session, modes=["draw_polygon"], default_mode="draw_polygon")`, `disable_draw(session)`, `delete_drawn_features(session)`; inputs `input.{id}_drawn_features()` (cumulative FeatureCollection), `input.{id}_map_click()` (lon/lat), `input.deckgl_ready`. `partial_update(session, layers=[...])`.
- Species list: `[cfg.get(f"species.name.sp{i}") for i in range(int(float(cfg.get("simulation.nspecies","0") or "0")))]` (pattern at `grid.py:854-857`).

---

## Task 1: `GridSpec` + cell geometry (pure core)

**Files:** Create `osmose/maps/__init__.py` (empty), `osmose/maps/builder.py`; Test: `tests/test_maps_builder.py`

- [ ] **Step 1: Write the failing test**
```python
import numpy as np
from osmose.maps.builder import GridSpec

def test_gridspec_cell_polygon_and_center():
    g = GridSpec(nlon=50, nlat=40, upleft_lat=66.0, upleft_lon=10.0, lowright_lat=54.0, lowright_lon=30.0)
    # dx=(30-10)/50=0.4 ; dy=(66-54)/40=0.3
    poly = g.cell_polygon(0, 0)  # north-west cell
    assert poly == [[10.0, 66.0], [10.4, 66.0], [10.4, 65.7], [10.0, 65.7]]  # [UL,UR,LR,LL]
    lat, lon = g.cell_center(0, 0)
    assert abs(lat - 65.85) < 1e-9 and abs(lon - 10.2) < 1e-9

def test_gridspec_from_config():
    from osmose.maps.builder import GridSpec
    cfg = {"grid.nlon": "50", "grid.nlat": "40", "grid.upleft.lat": "66",
           "grid.upleft.lon": "10", "grid.lowright.lat": "54", "grid.lowright.lon": "30"}
    g = GridSpec.from_config(cfg)
    assert (g.nlon, g.nlat) == (50, 40) and g.upleft_lat == 66.0
```
- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_maps_builder.py -k gridspec -q`
- [ ] **Step 3: Implement** in `osmose/maps/builder.py`:
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GridSpec:
    nlon: int
    nlat: int
    upleft_lat: float
    upleft_lon: float
    lowright_lat: float
    lowright_lon: float

    @classmethod
    def from_config(cls, cfg: dict[str, str]) -> "GridSpec":
        return cls(
            nlon=int(float(cfg["grid.nlon"])),
            nlat=int(float(cfg["grid.nlat"])),
            upleft_lat=float(cfg["grid.upleft.lat"]),
            upleft_lon=float(cfg["grid.upleft.lon"]),
            lowright_lat=float(cfg["grid.lowright.lat"]),
            lowright_lon=float(cfg["grid.lowright.lon"]),
        )

    @property
    def dx(self) -> float:
        return (self.lowright_lon - self.upleft_lon) / self.nlon

    @property
    def dy(self) -> float:
        return (self.upleft_lat - self.lowright_lat) / self.nlat

    def cell_polygon(self, row: int, col: int) -> list[list[float]]:
        lo0 = self.upleft_lon + col * self.dx
        la0 = self.upleft_lat - row * self.dy
        lo1, la1 = lo0 + self.dx, la0 - self.dy
        return [[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1]]  # UL,UR,LR,LL (north-row-0)

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        return (self.upleft_lat - (row + 0.5) * self.dy, self.upleft_lon + (col + 0.5) * self.dx)
```
- [ ] **Step 4: Run, verify PASS.** Same `-k gridspec`. ruff + format clean on the new file.
- [ ] **Step 5: Commit**
```bash
git add osmose/maps/__init__.py osmose/maps/builder.py tests/test_maps_builder.py
git commit -m "feat(maps): GridSpec cell geometry for the map builder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: DRY — `build_grid_layers` calls `GridSpec.cell_polygon`

**Files:** Modify `ui/pages/grid_helpers.py`; Test: existing `tests/test_ui_grid.py` (characterization guard — do NOT modify it)

- [ ] **Step 1: Run the existing characterization tests GREEN first** (baseline): `.venv/bin/python -m pytest tests/test_ui_grid.py tests/test_grid_helpers.py -q` → all pass. These assert `build_grid_layers` per-cell `data` shape (`polygon`/`row`/`col`/`type`), the UL-corner value (`poly[0]==(-6.0,48.0)`), and the ocean/land layer split — they are the regression guard.
- [ ] **Step 2: Refactor** `build_grid_layers` (grid_helpers.py ~339-348): replace the inline corner math with `GridSpec(...).cell_polygon(row, col)` (build a `GridSpec` from the same bounds it already reads). Keep EVERYTHING else identical — the same per-cell dict keys, the same `[UL,UR,LR,LL]` corner order, the same north-row-0 row/col indexing, the same ocean/land split. Import `GridSpec` at module level (or in-function if a cycle appears — `osmose.maps.builder` has no UI deps, so module-level is fine).
- [ ] **Step 3: Run, verify the characterization tests STILL PASS unchanged.** `.venv/bin/python -m pytest tests/test_ui_grid.py tests/test_grid_helpers.py -q` → identical green. If any corner/orientation/shape assertion changes, the refactor diverged — fix until byte-identical. `.venv/bin/python -c "import ui.pages.grid, ui.pages.map_viewer"` clean.
- [ ] **Step 4: Commit**
```bash
git add ui/pages/grid_helpers.py
git commit -m "refactor(grid): build_grid_layers uses GridSpec.cell_polygon (single source of truth)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `rasterize_polygon` + `lonlat_to_cell`

**Files:** Modify `osmose/maps/builder.py`; Test: `tests/test_maps_builder.py`

- [ ] **Step 1: Write the failing tests** (append)
```python
def test_rasterize_polygon_centers_inside():
    from osmose.maps.builder import GridSpec, rasterize_polygon
    g = GridSpec(nlon=4, nlat=4, upleft_lat=4.0, upleft_lon=0.0, lowright_lat=0.0, lowright_lon=4.0)
    # square covering the NW 2x2 block of cell centers (centers at 0.5,1.5,2.5,3.5)
    ring = [[0.0, 4.0], [2.0, 4.0], [2.0, 2.0], [0.0, 2.0]]  # lon0..2, lat2..4
    cells = set(rasterize_polygon(g, ring, mask=None))
    assert cells == {(0, 0), (0, 1), (1, 0), (1, 1)}

def test_rasterize_excludes_masked_cells():
    from osmose.maps.builder import GridSpec, rasterize_polygon
    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    mask = np.zeros((4, 4)); mask[0, 0] = -99
    ring = [[0.0, 4.0], [2.0, 4.0], [2.0, 2.0], [0.0, 2.0]]
    cells = set(rasterize_polygon(g, ring, mask=mask))
    assert (0, 0) not in cells and (0, 1) in cells

def test_rasterize_polygon_outside_grid_empty():
    from osmose.maps.builder import GridSpec, rasterize_polygon
    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    assert rasterize_polygon(g, [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0]], mask=None) == []

def test_lonlat_to_cell():
    from osmose.maps.builder import GridSpec, lonlat_to_cell
    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    assert lonlat_to_cell(g, 0.5, 3.5) == (0, 0)   # NW
    assert lonlat_to_cell(g, 3.5, 0.5) == (3, 3)   # SE
    assert lonlat_to_cell(g, -1.0, 3.5) is None    # outside
```
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** (append). Use a scalar even-odd ray-cast — the parity guard short-circuits horizontal edges so there is NO divide-by-zero (no `np.errstate` needed):
```python
import numpy as np
from collections.abc import Iterable

def _point_in_ring(px: float, py: float, ring: list[list[float]]) -> bool:
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _open_ring(ring: list[list[float]]) -> list[list[float]]:
    return ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring  # drop GeoJSON closing dup

def rasterize_polygon(grid: GridSpec, polygon_lonlat, mask=None, *, mask_edit: bool = False):
    ring = _open_ring([list(p) for p in polygon_lonlat])
    out: list[tuple[int, int]] = []
    for r in range(grid.nlat):
        for c in range(grid.nlon):
            if mask is not None and not mask_edit and mask[r, c] == -99:
                continue
            lat, lon = grid.cell_center(r, c)
            if _point_in_ring(lon, lat, ring):
                out.append((r, c))
    return out

def lonlat_to_cell(grid: GridSpec, lon: float, lat: float):
    c = int((lon - grid.upleft_lon) / grid.dx)
    r = int((grid.upleft_lat - lat) / grid.dy)
    if 0 <= r < grid.nlat and 0 <= c < grid.nlon:
        return (r, c)
    return None
```
- [ ] **Step 4: Add the Hypothesis property test** (in-memory ONLY — no `tmp_path`/fixtures under `@given`):
```python
from hypothesis import given, strategies as st

@given(
    lons=st.lists(st.floats(0.1, 3.9), min_size=3, max_size=6),
    lats=st.lists(st.floats(0.1, 3.9), min_size=3, max_size=6),
)
def test_rasterize_matches_center_membership(lons, lats):
    from osmose.maps.builder import GridSpec, rasterize_polygon, _point_in_ring, _open_ring
    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    n = min(len(lons), len(lats))
    ring = [[lons[i], lats[i]] for i in range(n)]
    got = set(rasterize_polygon(g, ring, mask=None))
    expected = {
        (r, c) for r in range(4) for c in range(4)
        if _point_in_ring(*(lambda la, lo: (lo, la))(*g.cell_center(r, c)), _open_ring(ring))
    }
    assert got == expected
```
- [ ] **Step 5: Verify + commit.** `.venv/bin/python -m pytest tests/test_maps_builder.py -k "rasterize or lonlat or center_membership" -q` → pass. ruff/format clean.
```bash
git add osmose/maps/builder.py tests/test_maps_builder.py
git commit -m "feat(maps): numpy ray-cast rasterize_polygon + lonlat_to_cell"
```

---

## Task 4: `MapGrid` paint/erase/mask ops

**Files:** Modify `osmose/maps/builder.py`; Test: `tests/test_maps_builder.py`

- [ ] **Step 1: Write the failing tests** (append)
```python
def test_mapgrid_apply_erase_mask():
    from osmose.maps.builder import GridSpec, MapGrid
    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    mg = MapGrid.blank(g)              # all 0, no mask
    mg.apply_cells([(0, 0), (1, 1)], 1.0)
    assert mg.array[0, 0] == 1.0 and mg.array[1, 1] == 1.0
    mg.erase([(0, 0)]); assert mg.array[0, 0] == 0.0
    mg.set_mask([(2, 2)], True); assert mg.array[2, 2] == -99
    mg.set_mask([(2, 2)], False); assert mg.array[2, 2] == 0.0

def test_mapgrid_blank_seeds_base_mask():
    from osmose.maps.builder import GridSpec, MapGrid
    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    base = np.zeros((4, 4)); base[0, 0] = -99
    mg = MapGrid.blank(g, base_mask=base)
    assert mg.array[0, 0] == -99 and mg.array[1, 1] == 0.0

def test_mapgrid_apply_polygon():
    from osmose.maps.builder import GridSpec, MapGrid
    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    mg = MapGrid.blank(g)
    mg.apply_polygon(g, [[0.0, 4.0], [2.0, 4.0], [2.0, 2.0], [0.0, 2.0]], 1.0)
    assert mg.array[0, 0] == 1.0 and mg.array[3, 3] == 0.0
```
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** (append):
```python
class MapGrid:
    def __init__(self, array: np.ndarray):
        self._a = array

    @classmethod
    def blank(cls, grid: GridSpec, base_mask: np.ndarray | None = None) -> "MapGrid":
        a = np.zeros((grid.nlat, grid.nlon), dtype=float)
        if base_mask is not None:
            a[base_mask == -99] = -99
        return cls(a)

    @property
    def array(self) -> np.ndarray:
        return self._a

    def apply_cells(self, cells: Iterable[tuple[int, int]], value: float) -> None:
        for r, c in cells:
            self._a[r, c] = value

    def apply_polygon(self, grid: GridSpec, polygon_lonlat, value: float, *, mask_edit: bool = False) -> None:
        self.apply_cells(rasterize_polygon(grid, polygon_lonlat, self._a, mask_edit=mask_edit), value)

    def erase(self, cells: Iterable[tuple[int, int]]) -> None:
        self.apply_cells(cells, 0.0)

    def set_mask(self, cells: Iterable[tuple[int, int]], masked: bool) -> None:
        self.apply_cells(cells, -99.0 if masked else 0.0)
```
- [ ] **Step 4: Verify + commit.** `-k mapgrid` pass; ruff/format clean.
```bash
git add osmose/maps/builder.py tests/test_maps_builder.py
git commit -m "feat(maps): MapGrid apply/erase/mask ops"
```

---

## Task 5: `to_csv_text`/`from_csv_text` with the orientation flip (engine round-trip)

**Files:** Modify `osmose/maps/builder.py`; Test: `tests/test_maps_builder.py`

- [ ] **Step 1: Write the failing tests** (append) — the round-trip goes through the ENGINE reader:
```python
def test_csv_roundtrip_through_engine_loader(tmp_path):
    from osmose.maps.builder import GridSpec, MapGrid, to_csv_text
    from osmose.engine.movement_maps import _load_csv_grid
    g = GridSpec(3, 2, 2.0, 0.0, 0.0, 3.0)  # nlat=2, nlon=3
    mg = MapGrid.blank(g)
    mg.apply_cells([(0, 0)], 1.0)   # north-row-0 cell
    f = tmp_path / "m.csv"
    f.write_text(to_csv_text(mg))
    loaded = _load_csv_grid(f, 2, 3)   # engine reads file (south-row-0), flips to north-row-0
    assert np.array_equal(loaded, mg.array)   # painted NW cell survives un-flipped

def test_from_csv_text_roundtrip_and_dim_validation():
    from osmose.maps.builder import GridSpec, MapGrid, to_csv_text, from_csv_text
    import pytest
    g = GridSpec(3, 2, 2.0, 0.0, 0.0, 3.0)
    mg = MapGrid.blank(g); mg.apply_cells([(1, 2)], 5.0)
    back = from_csv_text(to_csv_text(mg), g)
    assert np.array_equal(back.array, mg.array)
    with pytest.raises(ValueError):
        from_csv_text("1;2;3;4\n1;2;3;4\n", g)   # too many columns (nlon=3)
```
- [ ] **Step 2: Run, verify FAIL.** Confirm `_load_csv_grid(path, ny, nx)` import works.
- [ ] **Step 3: Implement** (append) — `np.flipud` bridges north-row-0 (memory) ↔ south-row-0 (file):
```python
def to_csv_text(mg: "MapGrid") -> str:
    south_first = np.flipud(mg.array)   # north-row-0 -> south-row-0 (engine file order)
    lines = []
    for row in south_first:
        lines.append(";".join(_fmt(v) for v in row))
    return "\n".join(lines) + "\n"

def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.10g}"

def from_csv_text(text: str, grid: GridSpec) -> "MapGrid":
    rows = [ln for ln in text.splitlines() if ln.strip()]
    data = [[float(x) for x in ln.split(";")] for ln in rows]
    if len(data) != grid.nlat or any(len(r) != grid.nlon for r in data):
        raise ValueError(f"CSV dims {len(data)}x{len(data[0]) if data else 0} != grid {grid.nlat}x{grid.nlon}")
    return MapGrid(np.flipud(np.array(data, dtype=float)))   # south-row-0 file -> north-row-0 memory
```
- [ ] **Step 4: Verify + commit.** `-k csv` pass; ruff/format clean.
```bash
git add osmose/maps/builder.py tests/test_maps_builder.py
git commit -m "feat(maps): CSV (de)serialization with south-row-0 flip (engine round-trip verified)"
```

---

## Task 6: `validate`

**Files:** Modify `osmose/maps/builder.py`; Test: `tests/test_maps_builder.py`

- [ ] **Step 1: Write the failing test** (append)
```python
def test_validate_dim_and_mask_and_land_warning():
    from osmose.maps.builder import GridSpec, MapGrid, validate
    g = GridSpec(4, 4, 4.0, 0.0, 0.0, 4.0)
    base = np.zeros((4, 4)); base[0, 0] = -99
    mg = MapGrid.blank(g, base_mask=base); mg.apply_cells([(0, 1)], 1.0)
    assert validate(mg, g, map_type="distribution", base_mask=base) == []   # clean
    mg.array[0, 0] = 1.0   # painted on land
    probs = validate(mg, g, map_type="distribution", base_mask=base)
    assert any("land" in p.lower() for p in probs)   # warns, not blocks
    bad = MapGrid(np.zeros((3, 3)))
    assert any("dim" in p.lower() for p in validate(bad, g, map_type="mask", base_mask=base))
```
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** (append):
```python
def validate(mg: "MapGrid", grid: GridSpec, *, map_type: str, base_mask: np.ndarray | None) -> list[str]:
    problems: list[str] = []
    if mg.array.shape != (grid.nlat, grid.nlon):
        problems.append(f"dimension mismatch: {mg.array.shape} != ({grid.nlat}, {grid.nlon})")
        return problems
    if base_mask is not None and map_type in ("distribution", "zone"):
        on_land = (mg.array != -99) & (mg.array != 0) & (base_mask == -99)
        if on_land.any():
            problems.append(f"{int(on_land.sum())} cell(s) painted on base-mask land (engine treats as absent)")
    return problems
```
- [ ] **Step 4: Verify + commit.** `-k validate` pass; ruff/format clean.
```bash
git add osmose/maps/builder.py tests/test_maps_builder.py
git commit -m "feat(maps): validate (dims + distribution-on-land warning)"
```

---

## Task 7: `wire_map_into_config`

**Files:** Modify `osmose/maps/builder.py`; Test: `tests/test_maps_builder.py`

- [ ] **Step 1: Write the failing tests** (append)
```python
def test_wire_distribution_real_keys_and_next_index():
    from osmose.maps.builder import wire_map_into_config
    cfg = {"movement.species.map0": "cod", "movement.file.map0": "maps/cod.csv"}
    appl = {"species": "herring", "initialage": 0.0, "lastage": 5.0, "steps": [0, 1, 2], "initialyear": 0, "lastyear": 9}
    out, summary = wire_map_into_config(cfg, "distribution", "maps/herring.csv", applicability=appl)
    assert out["movement.species.map1"] == "herring"
    assert out["movement.file.map1"] == "maps/herring.csv"
    assert out["movement.steps.map1"] == "0;1;2"           # always emitted
    assert out["movement.initialage.map1"] == "0" and out["movement.lastage.map1"] == "5"
    assert out["movement.initialyear.map1"] == "0" and out["movement.lastyear.map1"] == "9"
    assert "movement.map1.species" not in out               # NOT the inverted form

def test_wire_mask_and_zone():
    from osmose.maps.builder import wire_map_into_config
    out, _ = wire_map_into_config({}, "mask", "grid/mask.csv")
    assert out["grid.mask.file"] == "grid/mask.csv"
    out2, _ = wire_map_into_config({"a": "b"}, "zone", "maps/z.csv")
    assert out2 == {"a": "b"}   # zone wires no keys

def test_wire_steps_defaults_to_all_when_unspecified():
    from osmose.maps.builder import wire_map_into_config
    out, _ = wire_map_into_config({"simulation.time.ndtperyear": "12"}, "distribution",
                                  "maps/x.csv", applicability={"species": "sole"})
    assert out["movement.steps.map0"] == ";".join(str(i) for i in range(12))   # all steps
```
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** (append):
```python
import re

def _next_map_index(config: dict[str, str]) -> int:
    used = set()
    pat = re.compile(r"^movement\.[a-z]+\.map(\d+)$")
    for k in config:
        m = pat.match(k)
        if m:
            used.add(int(m.group(1)))
    i = 0
    while i in used:
        i += 1
    return i

def wire_map_into_config(config, map_type, rel_path, *, applicability=None):
    out = dict(config)
    if map_type == "mask":
        out["grid.mask.file"] = rel_path
        return out, f"Set grid.mask.file = {rel_path}"
    if map_type == "zone":
        return out, f"Wrote zone map {rel_path} (not wired into the config)"
    # distribution
    appl = applicability or {}
    n = _next_map_index(out)
    out[f"movement.species.map{n}"] = str(appl["species"])
    out[f"movement.file.map{n}"] = rel_path
    ndt = int(float(out.get("simulation.time.ndtperyear", "0") or "0"))
    steps = appl.get("steps")
    if not steps:
        steps = list(range(ndt)) if ndt else []
    out[f"movement.steps.map{n}"] = ";".join(str(int(s)) for s in steps)
    if "initialage" in appl:
        out[f"movement.initialage.map{n}"] = _fmt(float(appl["initialage"]))
    if "lastage" in appl:
        out[f"movement.lastage.map{n}"] = _fmt(float(appl["lastage"]))
    if "initialyear" in appl:
        out[f"movement.initialyear.map{n}"] = str(int(appl["initialyear"]))
    if "lastyear" in appl:
        out[f"movement.lastyear.map{n}"] = str(int(appl["lastyear"]))
    return out, f"Registered movement.*.map{n} for species '{appl['species']}' → {rel_path}"
```
- [ ] **Step 4: Verify + commit.** `-k wire` pass; ruff/format clean.
```bash
git add osmose/maps/builder.py tests/test_maps_builder.py
git commit -m "feat(maps): wire_map_into_config (real movement keys, next-free index, steps always)"
```

---

## Task 8: `save_map` orchestration (filename sanitize + write + wire)

**Files:** Modify `osmose/maps/builder.py`; Test: `tests/test_maps_builder.py`

- [ ] **Step 1: Write the failing tests** (append)
```python
def test_save_map_writes_csv_and_wires(tmp_path):
    from osmose.maps.builder import GridSpec, MapGrid, save_map
    g = GridSpec(3, 2, 2.0, 0.0, 0.0, 3.0)
    mg = MapGrid.blank(g); mg.apply_cells([(0, 0)], 1.0)
    cfg = {"grid.nlon": "3", "grid.nlat": "2", "grid.upleft.lat": "2", "grid.upleft.lon": "0",
           "grid.lowright.lat": "0", "grid.lowright.lon": "3", "simulation.time.ndtperyear": "2"}
    new_cfg, summary, path = save_map(mg, g, "distribution", "herring", cfg, tmp_path,
                                      applicability={"species": "herring"})
    assert path == tmp_path / "maps" / "herring.csv" and path.exists()
    assert new_cfg["movement.file.map0"] == "maps/herring.csv"

def test_save_map_rejects_bad_filename(tmp_path):
    from osmose.maps.builder import GridSpec, MapGrid, save_map
    import pytest
    g = GridSpec(3, 2, 2.0, 0.0, 0.0, 3.0); mg = MapGrid.blank(g)
    for bad in ["", "../evil", "a/b", "no_ext"]:
        with pytest.raises(ValueError):
            save_map(mg, g, "zone", bad, {}, tmp_path)
```
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** (append):
```python
from pathlib import Path

def _sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"invalid map filename: {name!r}")
    if not name.endswith(".csv"):
        name = name + ".csv"
    if name == ".csv":
        raise ValueError("empty map filename")
    return name

def save_map(mg, grid, map_type, filename, config, config_dir, *, applicability=None):
    fname = _sanitize_filename(filename)
    subdir = "grid" if map_type == "mask" else "maps"
    rel_path = f"{subdir}/{fname}"
    dest = Path(config_dir) / subdir / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(to_csv_text(mg))
    new_cfg, summary = wire_map_into_config(config, map_type, rel_path, applicability=applicability)
    return new_cfg, summary, dest
```
- [ ] **Step 4: Verify + commit.** `-k save_map` pass; full core file: `.venv/bin/python -m pytest tests/test_maps_builder.py -q` green. ruff/format/pyright clean on `osmose/maps/builder.py`.
```bash
git add osmose/maps/builder.py tests/test_maps_builder.py
git commit -m "feat(maps): save_map orchestration (sanitize + write + wire)"
```

---

## Task 9: Shiny page skeleton + nav registration

**Files:** Create `ui/pages/map_builder.py`; Modify `app.py`; Test: `tests/test_ui_map_builder.py`

- [ ] **Step 1: Write the failing test**
```python
def test_map_builder_imports_and_registered():
    import ui.pages.map_builder as mb
    assert hasattr(mb, "map_builder_ui") and hasattr(mb, "map_builder_server")
    from pathlib import Path
    app_src = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    assert "map_builder_ui" in app_src and 'value="map_builder"' in app_src
    assert "map_builder_server" in app_src and "'map_builder'" in app_src  # nav-order array
```
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement.** Create `ui/pages/map_builder.py` with `map_builder_ui()` and `map_builder_server(input, output, session, state)`. The UI: a `MapWidget` output, a tool-mode `input_radio_buttons` (Polygon-draw/Brush/Eraser/Mask-edit), a paint-value `input_numeric` (default 1), a map-type selector (Distribution/Land mask/Generic zone), the distribution applicability inputs (species select, initial/last age, season steps, initial/last year), Start controls (New blank / Load existing select), an "Apply polygon(s)" button, a filename input + Save button. `map_builder_server` body can start minimal (read `GridSpec.from_config(state.config.get())` guarded by a try/except → hint UI when no grid). Follow the `navset_pill_list` page pattern of `ui/pages/map_viewer.py`.
  Then wire the 4 `app.py` touch-points: (1) `from ui.pages.map_builder import map_builder_ui, map_builder_server`; (2) a `ui.nav_panel("Map Builder", map_builder_ui(), value="map_builder")` in the `navset_pill_list`; (3) `map_builder_server(input, output, session, state)` in `server()`; (4) add `"map_builder"` to the JS nav-order array at `app.py:235`.
- [ ] **Step 4: Verify.** `.venv/bin/python -m pytest tests/test_ui_map_builder.py -k registered -q` pass. `.venv/bin/python -c "import app"` clean.
- [ ] **Step 5: Commit**
```bash
git add ui/pages/map_builder.py app.py tests/test_ui_map_builder.py
git commit -m "feat(ui): Map Builder page skeleton + nav registration"
```

---

## Task 10: Grid render + reactive state + deckgl_ready gating

**Files:** Modify `ui/pages/map_builder.py`; Test: manual / e2e (render is the deck.gl seam)

- [ ] **Step 1: Implement** the render in `map_builder_server`:
  - `grid_array = reactive.Value(...)` initialized from "New blank map" (`MapGrid.blank(grid, base_mask=load_mask(cfg, config_dir))`) — `load_mask` returns `None` ⇒ all-sea.
  - A `@reactive.calc` builds the deck layers: `build_grid_layers(...)` for the base grid + a value-colored cells layer (reuse the `load_csv_overlay` value→RGBA ramp from `grid_helpers.py:736-745`), view-state from grid bounds (mirror `map_viewer._DEFAULT_VIEW_STATE` but centered on the config grid).
  - Gate the initial render + `enable_draw` on `@reactive.event(input.deckgl_ready)` (mirror `grid.py:500-505`). Read config under `reactive.isolate()`.
- [ ] **Step 2: Verify** `.venv/bin/python -c "import ui.pages.map_builder"` clean; ruff/format/pyright clean. (Visual correctness is the e2e seam — Task 15.)
- [ ] **Step 3: Commit**
```bash
git add ui/pages/map_builder.py
git commit -m "feat(ui): map builder grid render + deckgl_ready-gated init"
```

---

## Task 11: Brush / eraser / mask via `_map_click` + tool-mode draw toggle

**Files:** Modify `ui/pages/map_builder.py`

- [ ] **Step 1: Implement:**
  - `@reactive.effect @reactive.event(input.tool_mode)`: `await map.enable_draw(session, modes=["draw_polygon"], default_mode="draw_polygon")` when mode == polygon, else `await map.disable_draw(session)`.
  - `@reactive.effect @reactive.event(input.{mapid}_map_click)`: early-return unless mode in {brush, eraser, mask}; `cell = lonlat_to_cell(grid, lon, lat)`; if `cell`: `apply_cells([cell], paint_value)` / `erase([cell])` / `set_mask([cell], True)` per mode (block painting on `-99` unless mask mode); bump the dirty-flag (Task 13).
- [ ] **Step 2: Verify** import + ruff/format/pyright clean.
- [ ] **Step 3: Commit**
```bash
git add ui/pages/map_builder.py
git commit -m "feat(ui): map builder brush/eraser/mask + tool-mode draw toggle"
```

---

## Task 12: Polygon draw — stage-then-Apply

**Files:** Modify `ui/pages/map_builder.py`

- [ ] **Step 1: Implement:**
  - `staged = reactive.Value(None)`, `applied_ids = reactive.Value(set())`.
  - `@reactive.effect @reactive.event(input.{mapid}_drawn_features)`: store the FeatureCollection in `staged` (empty ⇒ no-op); update an "N staged" indicator. Do NOT paint here.
  - `@reactive.effect @reactive.event(input.apply_polygons)` (Apply button): for each feature in `staged` whose `feature.id` ∉ `applied_ids`, `rasterize_polygon`→`apply_polygon` at the CURRENT paint value (mask_edit if mode==mask), add the id to `applied_ids`; then `await map.delete_drawn_features(session)` and reset `applied_ids`/`staged`; bump dirty-flag.
- [ ] **Step 2: Verify** import + ruff/format/pyright clean.
- [ ] **Step 3: Commit**
```bash
git add ui/pages/map_builder.py
git commit -m "feat(ui): map builder polygon draw (stage-then-Apply, id-diff, clear)"
```

---

## Task 13: Coalesced render (dirty-flag + partial_update)

**Files:** Modify `ui/pages/map_builder.py`

- [ ] **Step 1: Implement:** a `dirty = reactive.Value(0)` bumped by every paint op; a debounced/coalesced effect (mirror the `live_movement` `reactive.poll`/timer change-detection pattern) that, when `dirty` changed since last render, rebuilds ONLY the value-cells layer and `await map.partial_update(session, layers=[cells_layer])` (not a full `update()` of the whole stack).
- [ ] **Step 2: Verify** import + ruff/format/pyright clean.
- [ ] **Step 3: Commit**
```bash
git add ui/pages/map_builder.py
git commit -m "feat(ui): map builder coalesced partial_update render"
```

---

## Task 14: Map-type/applicability + Save (with species source, warns)

**Files:** Modify `ui/pages/map_builder.py`; Test: `tests/test_ui_map_builder.py`

- [ ] **Step 1: Write the failing test** (the save path is already pure-tested via `save_map` in Task 8; here test the species-list helper + the no-grid/config_dir guards as extracted helpers):
```python
def test_species_choices_from_config():
    from ui.pages.map_builder import _species_choices
    cfg = {"simulation.nspecies": "2", "species.name.sp0": "cod", "species.name.sp1": "herring"}
    assert _species_choices(cfg) == ["cod", "herring"]
```
- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement:**
  - `_species_choices(cfg)` helper (the verified pattern) → populate the species select.
  - Show the applicability form only when map-type == Distribution; pre-fill defaults (initialage 0, lastage = species lifespan if available else blank, steps = all, years = all).
  - Save button handler: resolve `config_dir = state.config_dir.get()`; if `None`, create a session `tempfile.mkdtemp()` dir + `state.config_dir.set(it)` (or block with a toast if not writable). Block a distribution save with no species. Call `validate` (toast warnings, don't block on land-warning); confirm on overwrite + warn on duplicate species/age/step overlap. Call `save_map(...)`; apply `new_cfg` via `state.load_config(new_cfg, ...)`/`update_config`; toast the summary.
- [ ] **Step 4: Verify.** `.venv/bin/python -m pytest tests/test_ui_map_builder.py -q` pass; import clean; ruff/format/pyright clean.
- [ ] **Step 5: Commit**
```bash
git add ui/pages/map_builder.py tests/test_ui_map_builder.py
git commit -m "feat(ui): map builder type/applicability + Save (species source, validate, warns)"
```

---

## Task 15: Final gates + e2e for the draw seam

**Files:** Test: `tests/test_e2e_map_builder.py` (optional, viztest-gated)

- [ ] **Step 1: Full suite.** `.venv/bin/python -m pytest -q -m "not e2e and not visual"` → report counts (known `test_runner`/`test_study_fullmodel` xdist flakes re-run isolated). New: `tests/test_maps_builder.py` + `tests/test_ui_map_builder.py` green; `tests/test_ui_grid.py` unchanged + green (DRY characterization).
- [ ] **Step 2: Lint/format/pyright.** `.venv/bin/ruff check osmose/ ui/ tests/` + `.venv/bin/ruff format --check osmose/ ui/ tests/` clean; `.venv/bin/pyright --pythonpath .venv/bin/python osmose/maps/builder.py ui/pages/map_builder.py ui/pages/grid_helpers.py` 0 NEW errors. `.venv/bin/python -c "import app"` clean.
- [ ] **Step 3: e2e (optional, the draw/pick seam — `test_e2e_*` naming so conftest collect-ignore + `[viztest]` gating apply).** A Playwright test: load Baltic → nav to Map Builder → confirm the grid renders, brush a cell (simulate `_map_click`), draw + Apply a polygon, Save → assert the map CSV appears under `state.config_dir/maps/` and the config gained `movement.file.map{N}`. Dismiss the "What's new" changelog modal first (per the e2e gotcha). If Playwright/browser unavailable, document the manual check.
- [ ] **Step 4: Commit any gate fixes** (explicit paths; not `git add -A`).
```bash
git add <changed files>
git commit -m "test(maps): map builder final gates + e2e draw seam"
```

---

## Notes

- **The pure core (Tasks 1–8) is the bulk and is fully CI-tested**; the Shiny page (Tasks 9–14) keeps all logic in the core, so its tests are import + the `save_map`/`_species_choices` extractions + nav registration; the deck.gl draw/pick seam is the only e2e/manual part (Task 15).
- **Orientation invariant** (north-row-0 in memory, south-row-0 on disk, flip in `to/from_csv_text`) is locked by the engine round-trip test (Task 5) — never compare self-round-trip only.
- **DRY refactor (Task 2)** is the ONLY change to shipped code; `tests/test_ui_grid.py` must stay unmodified and green as the characterization guard.
- **Real movement keys only** (`movement.species/file/steps/initialage/lastage/initialyear/lastyear/years.map{N}`); never the inverted `movement.map{N}.*`.
- **No new runtime deps** (numpy ray-cast; no matplotlib/shapely).
- Out of scope (per spec): multi-species-per-map (N blocks), spatial-fishing effort mechanics, undo/redo, grid-bounds authoring, raster/shapefile import.
