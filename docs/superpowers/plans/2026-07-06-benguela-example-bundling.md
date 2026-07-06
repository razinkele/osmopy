# Southern Benguela Example Bundling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle the Southern Benguela (`osmose-ben_v4.3_Florance`) config as a runnable, stable,
unfished `benguela` demo in osmopy's registry, on the Python engine, with tests.

**Architecture:** One-time build scripts under `scripts/` read a vendored source snapshot
(`data/benguela_src/`) and emit the committed bundle `data/benguela/` (merged resource forcing,
CSV movement maps, analytic seeding, fishing stripped). `osmose/demo.py` gains a `_generate_benguela`
generator; `osmose/runner.py` gains a real Python-only guard. Correctness is proven by a committed
smoke/stability test that runs the demo at an empirically pinned horizon.

**Tech Stack:** Python 3.12/3.13, numpy, xarray/netCDF4, pandas, pytest, the osmopy engine
(`osmose.config.reader.OsmoseConfigReader`, `osmose.engine.PythonEngine`,
`osmose.engine.config.EngineConfig`, `osmose.engine.resources.ResourceState`,
`osmose.engine.movement_maps`).

## Global Constraints

- **Python-engine example only.** No native-4.4.x conversion, no cross-engine Java parity. osmopy
  auto-migrates the 4.3.3 keys to 4.4.0 on read (verified — 7 deprecated keys migrate).
- **Unfished v1.** Fishing disabled AND its unconditional-read file keys stripped.
- **Source of truth for seeding** = `data/benguela_src/osmose-ben_seeding.R` values:
  `sp0=3129213 sp1=3888750 sp2=3029155 sp3=1286364 sp4=1138339 sp5=1439984 sp6=198865 sp7=81054
  sp8=575361 sp9=591907` (tonnes).
- **Resource forcing must be ONE multi-variable NetCDF** with variables named exactly
  `sphy/lphy/szoo/lzoo` (= `species.name.sp300-303`), dims `time=24, ny=62, nx=56`.
- **Movement CSV format:** semicolon-delimited `(ny=62) × (nx=56)` grid; `-99` = land/absent,
  `0..1` = ocean presence. Every emitted map index carries `movement.species.mapN` (binds index→
  species), `movement.file.mapN`, `movement.steps.mapN`, `movement.initialage.mapN`,
  `movement.lastage.mapN`.
- **Resource-biomass checks use `np.nansum` / ocean-mask**, never `.sum()` (forcing is NaN over land,
  ~54% of cells, like EEC's shipped forcing).
- **Stability bound (smoke gate):** for each species and all timesteps,
  `biomass[t] ≤ 1000 × seeding_biomass[species]` AND `biomass[t] ≤ 1e9` tonnes; no NaN; final
  biomass > 0.
- **Determinism:** `PythonEngine().run_in_memory(raw, seed=42)`.
- `scripts/` is OUTSIDE the ruff/pyright scope; `osmose/ ui/ tests/` must stay ruff+pyright clean.
- Branch: `feat/benguela-example`. Commit after each task.
- Source snapshot lives at `data/benguela_src/` (sibling of the demo bundle, NOT copied by the demo
  generator). Everything the demo needs at runtime lives under `data/benguela/`.

---

## File Structure

- `data/benguela_src/` — **vendored source snapshot** (the `osmose-ben_v4.3_Florance` clone). Build
  input; committed for reproducibility. Not a demo.
- `data/benguela/` — **committed runtime bundle** (what `_generate_benguela` copies).
  - `benguela_all-parameters.csv` — synthesized flat master.
  - `input/` — `grid-mask.nc`, `roms_climatological_merged.nc`, `predation-accessibility-25mars2015.csv`,
    species/param CSVs, `reproduction/reproduction-seasonality-sp{0..9}.csv`.
  - `maps/` — converted CSV presence grids.
- `scripts/merge_benguela_forcing.py` — Task 1.
- `scripts/convert_benguela_maps.py` — Task 2.
- `scripts/derive_benguela_seeding.py` — Task 3.
- `scripts/build_benguela_config.py` — Task 4.
- `scripts/validate_benguela_stability.py` — Task 5 (one-time; determines pinned nyear).
- `osmose/demo.py` — Task 6 (registry wiring).
- `osmose/runner.py` — Task 7 (Python-only guard).
- `tests/test_benguela_bundle.py` — artifact tests (Tasks 1–4).
- `tests/test_benguela_demo.py` — demo + smoke/stability/determinism tests (Tasks 6–8).

Build/run order: **1,2,3 → 4 → 5 → 6 → 7 → 8**.

---

### Task 1: Vendor source + merge resource forcing

**Files:**
- Create: `data/benguela_src/` (vendored clone), `scripts/merge_benguela_forcing.py`,
  `data/benguela/input/roms_climatological_merged.nc` (output)
- Test: `tests/test_benguela_bundle.py`

**Interfaces:**
- Produces: `merge_forcing(src_dir: Path, out_path: Path) -> None` — writes a 4-variable NetCDF.

- [ ] **Step 1: Vendor the source snapshot**

```bash
SRC=/tmp/claude-1000/-home-razinka-osmose/f7b91731-5bf2-427b-aaab-4e339882ae8b/scratchpad/osmose-ben/osmose-ben_v4.3_Florance
mkdir -p /home/razinka/osmopy/data/benguela_src
cp -r "$SRC"/. /home/razinka/osmopy/data/benguela_src/
ls /home/razinka/osmopy/data/benguela_src/input/roms_climatological-*.nc | wc -l   # expect 4
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_benguela_bundle.py`:
```python
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data" / "benguela" / "input" / "roms_climatological_merged.nc"
RES_VARS = ["sphy", "lphy", "szoo", "lzoo"]


def test_merged_forcing_has_all_four_resources():
    assert MERGED.exists(), "run scripts/merge_benguela_forcing.py"
    ds = xr.open_dataset(MERGED)
    try:
        for v in RES_VARS:
            assert v in ds.data_vars, f"{v} missing from merged forcing"
            assert ds[v].shape == (24, 62, 56), f"{v} wrong dims {ds[v].shape}"
            assert float(np.nansum(ds[v].values)) > 0, f"{v} is all-zero/NaN"
    finally:
        ds.close()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py::test_merged_forcing_has_all_four_resources -q`
Expected: FAIL (`run scripts/merge_benguela_forcing.py`).

- [ ] **Step 4: Write the merge script**

Create `scripts/merge_benguela_forcing.py`:
```python
"""Merge Benguela's 4 single-variable ROMS forcing NetCDFs into one multi-variable file.

osmopy's ResourceState loads a SINGLE resource NetCDF and looks up each resource BY NAME in it
(resources.py:216). Benguela ships one file per resource, so only the first would load. Merge them
so all 4 (sphy/lphy/szoo/lzoo) resolve.
"""
from __future__ import annotations
import sys
from pathlib import Path
import xarray as xr

RES = {
    "sphy": "roms_climatological-sphy_benguela_15days_2000_2009.nc",
    "lphy": "roms_climatological-lphy_benguela_15days_2000_2009.nc",
    "szoo": "roms_climatological-szoo_benguela_15days_2000_2009.nc",
    "lzoo": "roms_climatological-lzoo_benguela_15days_2000_2009.nc",
}


def merge_forcing(src_dir: Path, out_path: Path) -> None:
    data_vars = {}
    coords = None
    for name, fname in RES.items():
        ds = xr.open_dataset(src_dir / "input" / fname)
        # each file holds exactly one data variable; take it and rename to the resource name
        var = list(ds.data_vars)[0]
        da = ds[var]
        data_vars[name] = da.rename(name) if da.name != name else da
        coords = ds.coords if coords is None else coords
    merged = xr.Dataset(data_vars)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(out_path)
    merged.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "benguela_src"
    out = root / "data" / "benguela" / "input" / "roms_climatological_merged.nc"
    merge_forcing(src, out)
    print(f"wrote {out}")
```

- [ ] **Step 5: Run the merge script**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/merge_benguela_forcing.py`
Expected: `wrote .../roms_climatological_merged.nc`

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py::test_merged_forcing_has_all_four_resources -q`
Expected: PASS. (If a var's shape isn't `(24,62,56)`, transpose to `(time, ny, nx)` in the script.)

- [ ] **Step 7: Commit**

```bash
cd /home/razinka/osmopy
git add data/benguela_src data/benguela/input/roms_climatological_merged.nc scripts/merge_benguela_forcing.py tests/test_benguela_bundle.py
git commit -m "feat(benguela): vendor source + merge ROMS forcing into one multi-var NetCDF"
```

---

### Task 2: Convert NetCDF movement maps to CSV

**Files:**
- Create: `scripts/convert_benguela_maps.py`, `data/benguela/maps/*.csv` (output),
  `data/benguela/_movement_keys.txt` (emitted key block, consumed by Task 4)
- Test: `tests/test_benguela_bundle.py`

**Interfaces:**
- Consumes: `data/benguela_src/` (source maps + `movement.*.mapN` declarations).
- Produces: `convert_maps(src_dir, maps_out_dir, keys_out_path) -> list[dict]`; the CSV files; and a
  `_movement_keys.txt` file of `key ; value` lines (the rewritten movement block) for Task 4.

**Context:** Each source `movement.*.mapN` declares `movement.species.mapN`, `movement.variable.mapN`
(=`stage0/1/2`, a variable in `input/maps/<species>.nc`, shape `(24,62,56)`),
`movement.initialAge.mapN`, `movement.lastAge.mapN`, `movement.file.mapN` (the species `.nc`). osmopy
reads static CSV grids and expresses time-variation via multiple indices with different
`movement.steps.mapN`. So each source index expands into one osmopy index per DISTINCT time-slice of
its stage variable. Land cells (grid-mask==0) → `-99`; ocean → the slice value.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benguela_bundle.py`:
```python
from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig

MAPS_DIR = ROOT / "data" / "benguela" / "maps"
MOVE_KEYS = ROOT / "data" / "benguela" / "_movement_keys.txt"
BEN_SPECIES = ["euphausiids", "anchovy", "sardine", "redeye", "horsemackerel",
               "mesopelagic", "silverkob", "snoek", "shallowwaterhake", "deepwaterhake"]


def _load_movement_keys():
    d = {}
    for ln in MOVE_KEYS.read_text().splitlines():
        if ";" in ln:
            k, v = ln.split(";", 1)
            d[k.strip().lower()] = v.strip()
    return d


def test_movement_maps_converted_and_wired():
    assert MOVE_KEYS.exists(), "run scripts/convert_benguela_maps.py"
    keys = _load_movement_keys()
    # every map index carries a species binding + a file that exists + steps + age range
    idxs = sorted({int(k.split(".map")[1]) for k in keys if k.startswith("movement.species.map")})
    assert idxs, "no movement.species.mapN emitted (species would be orphaned -> is_out)"
    for n in idxs:
        sp = keys[f"movement.species.map{n}"]
        assert sp in BEN_SPECIES, f"map{n} species '{sp}' not a Benguela species"
        f = keys[f"movement.file.map{n}"]
        csv = ROOT / "data" / "benguela" / f
        assert csv.exists(), f"map{n} file {f} missing"
        assert f"movement.steps.map{n}" in keys
        assert f"movement.initialage.map{n}" in keys
        assert f"movement.lastage.map{n}" in keys


def test_movement_csv_format_and_ocean_within_grid():
    ds = xr.open_dataset(ROOT / "data" / "benguela_src" / "input" / "grid-mask.nc")
    ocean = ds["mask"].values.astype(float)  # 1=ocean, 0/NaN=land; align orientation below
    ds.close()
    sample = sorted(MAPS_DIR.glob("*.csv"))[0]
    grid = np.genfromtxt(sample, delimiter=";", filling_values=np.nan)
    grid = grid[:, :56] if grid.shape[1] > 56 else grid   # tolerate trailing ';'
    assert grid.shape == (62, 56), f"CSV shape {grid.shape} != (62,56)"
    present = (grid > 0) & (grid != -99)
    ocean_mask = np.nan_to_num(ocean) > 0
    # every 'present' cell must be an ocean cell (no fish on land)
    assert present[~ocean_mask].sum() == 0, "map places presence on land — orientation/masking wrong"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py -k movement -q`
Expected: FAIL (`run scripts/convert_benguela_maps.py`).

- [ ] **Step 3: Write the converter script**

Create `scripts/convert_benguela_maps.py`:
```python
"""Convert Benguela's per-species NetCDF movement maps into osmopy CSV maps.

Each source movement.*.mapN references a stage variable (24,62,56) in input/maps/<species>.nc. osmopy
reads static CSV grids; time-variation is expressed by multiple indices with different
movement.steps.mapN. So each source index -> one osmopy index per DISTINCT 62x56 time-slice.
CSV format: semicolon-delimited, -99=land, ocean value = slice value; orientation matched to grid.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from osmose.config.reader import OsmoseConfigReader


def _grid_ocean(src_dir: Path) -> np.ndarray:
    ds = xr.open_dataset(src_dir / "input" / "grid-mask.nc")
    ocean = np.nan_to_num(ds["mask"].values.astype(float)) > 0
    ds.close()
    return ocean  # (62,56), True=ocean


def _write_csv(path: Path, grid: np.ndarray) -> None:
    lines = [";".join(f"{v:g}" for v in row) + ";" for row in grid]
    path.write_text("\n".join(lines) + "\n")


def convert_maps(src_dir: Path, maps_out: Path, keys_out: Path) -> list[dict]:
    maps_out.mkdir(parents=True, exist_ok=True)
    raw = dict(OsmoseConfigReader().read(str(src_dir / "osmose-ben.R")))
    ocean = _grid_ocean(src_dir)
    # discover source map indices
    src_idxs = sorted({int(k.split(".map")[1]) for k in raw if k.startswith("movement.species.map")})
    out_rows: list[dict] = []
    seen: dict[bytes, str] = {}   # slice bytes -> csv relpath (content dedup)
    out_n = 0
    for si in src_idxs:
        sp = raw[f"movement.species.map{si}"]
        stage = raw[f"movement.variable.map{si}"]            # stage0/1/2
        a0 = raw[f"movement.initialage.map{si}"]
        a1 = raw[f"movement.lastage.map{si}"]
        nc = src_dir / raw[f"movement.file.map{si}"]         # input/maps/<species>.nc
        da = xr.open_dataset(nc)[stage].values               # (24,62,56)
        # align orientation to the grid: verify presence lands on ocean, else flipud
        def _mask_land(slice2d: np.ndarray) -> np.ndarray:
            g = np.where(ocean, slice2d, -99.0)
            return g
        # group timesteps by identical slice
        groups: dict[bytes, list[int]] = {}
        oriented = []
        for t in range(da.shape[0]):
            s = da[t]
            # choose orientation once (t==0): the one whose presence sits on ocean
            if t == 0:
                pres_as_is = ((s > 0) & ~np.isnan(s)) & ~ocean
                pres_flip = ((np.flipud(s) > 0) & ~np.isnan(np.flipud(s))) & ~ocean
                flip = pres_flip.sum() < pres_as_is.sum()
            s = np.flipud(s) if flip else s
            g = _mask_land(np.nan_to_num(s))
            oriented.append(g)
            groups.setdefault(g.tobytes(), []).append(t)
        for gb, steps in groups.items():
            g = oriented[steps[0]]
            key = gb
            if key in seen:
                rel = seen[key]
            else:
                fn = f"{sp}_{stage}_g{out_n}.csv"
                _write_csv(maps_out / fn, g)
                rel = f"maps/{fn}"
                seen[key] = rel
                out_n += 1
            out_rows.append({"species": sp, "file": rel, "steps": steps, "a0": a0, "a1": a1})
    # emit osmopy movement key block
    lines = []
    for n, r in enumerate(out_rows):
        lines += [
            f"movement.species.map{n} ; {r['species']}",
            f"movement.file.map{n} ; {r['file']}",
            f"movement.steps.map{n} ; {';'.join(str(s) for s in r['steps'])}",
            f"movement.initialage.map{n} ; {r['a0']}",
            f"movement.lastage.map{n} ; {r['a1']}",
        ]
    keys_out.write_text("\n".join(lines) + "\n")
    return out_rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "benguela_src"
    convert_maps(src, root / "data" / "benguela" / "maps",
                 root / "data" / "benguela" / "_movement_keys.txt")
    print("maps converted")
```

- [ ] **Step 4: Run the converter, then the tests**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/convert_benguela_maps.py`
Then: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py -k movement -q`
Expected: PASS. (If `test_..._ocean_within_grid` fails, the per-index orientation heuristic picked
wrong; make the flip decision global by majority vote across all indices/timesteps.)

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add data/benguela/maps data/benguela/_movement_keys.txt scripts/convert_benguela_maps.py tests/test_benguela_bundle.py
git commit -m "feat(benguela): convert NetCDF movement maps to osmopy CSV + steps wiring"
```

---

### Task 3: Derive analytic seeding block

**Files:**
- Create: `scripts/derive_benguela_seeding.py`, `data/benguela/_seeding_keys.txt` (output)
- Test: `tests/test_benguela_bundle.py`

**Interfaces:**
- Produces: `derive_seeding(src_dir, out_path) -> dict[int, float]` + a `_seeding_keys.txt` of
  `population.seeding.biomass.spN ; <tonnes>` lines for Task 4.

**Context:** Use the authors' `osmose-ben_seeding.R` values as primary; cross-check against restart
aggregation (`abundance × weight` per species in `ben-initial_conditions.nc`) and warn if any species
differs by >5×.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benguela_bundle.py`:
```python
SEED_KEYS = ROOT / "data" / "benguela" / "_seeding_keys.txt"
EXPECTED_SEED = {0: 3129213, 1: 3888750, 2: 3029155, 3: 1286364, 4: 1138339,
                 5: 1439984, 6: 198865, 7: 81054, 8: 575361, 9: 591907}


def test_seeding_block_matches_authors_values():
    assert SEED_KEYS.exists(), "run scripts/derive_benguela_seeding.py"
    got = {}
    for ln in SEED_KEYS.read_text().splitlines():
        if "seeding.biomass.sp" in ln:
            k, v = ln.split(";", 1)
            got[int(k.strip().split(".sp")[1])] = float(v)
    assert set(got) == set(range(10))
    for sp, exp in EXPECTED_SEED.items():
        assert abs(got[sp] - exp) < 1.0, f"sp{sp} seed {got[sp]} != {exp}"
        assert got[sp] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py::test_seeding_block_matches_authors_values -q`
Expected: FAIL.

- [ ] **Step 3: Write the seeding script**

Create `scripts/derive_benguela_seeding.py`:
```python
"""Derive Benguela's analytic seeding block from osmose-ben_seeding.R (authors' values),
cross-checked against the restart file's per-species standing stock."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from osmose.config.reader import OsmoseConfigReader


def derive_seeding(src_dir: Path, out_path: Path) -> dict[int, float]:
    raw = dict(OsmoseConfigReader().read(str(src_dir / "osmose-ben_seeding.R")))
    seed = {sp: float(raw[f"population.seeding.biomass.sp{sp}"]) for sp in range(10)}
    # cross-check against restart aggregation (abundance*weight -> tonnes)
    ds = xr.open_dataset(src_dir / "input" / "ben-initial_conditions.nc")
    spid = ds["species"].values; ab = ds["abundance"].values; w = ds["weight"].values
    for sp in range(10):
        m = spid == sp
        restart_t = float(np.nansum(ab[m] * w[m])) / 1e6
        if restart_t > 0 and not (0.2 < seed[sp] / restart_t < 5):
            print(f"WARN sp{sp}: authors={seed[sp]:.0f} vs restart={restart_t:.0f} (>5x apart)")
    ds.close()
    out_path.write_text("\n".join(
        f"population.seeding.biomass.sp{sp} ; {seed[sp]:.0f}" for sp in range(10)) + "\n")
    return seed


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "benguela_src"
    derive_seeding(src, root / "data" / "benguela" / "_seeding_keys.txt")
    print("seeding derived")
```

- [ ] **Step 4: Run script + test**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/derive_benguela_seeding.py`
Then: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py::test_seeding_block_matches_authors_values -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add scripts/derive_benguela_seeding.py data/benguela/_seeding_keys.txt tests/test_benguela_bundle.py
git commit -m "feat(benguela): derive analytic seeding block from authors' seeding config"
```

---

### Task 4: Synthesize the master config (fishing stripped, everything wired)

**Files:**
- Create: `scripts/build_benguela_config.py`, `data/benguela/benguela_all-parameters.csv` (output),
  copied statics under `data/benguela/input/`
- Test: `tests/test_benguela_bundle.py`

**Interfaces:**
- Consumes: `_movement_keys.txt` (Task 2), `_seeding_keys.txt` (Task 3), merged forcing (Task 1).
- Produces: `build_config(src_dir, bundle_dir) -> Path` writing `benguela_all-parameters.csv`.

**Context — exact edit set applied to the flattened source config:**
- Drop keys (families): `population.initialization.file`, `osmose.configuration.initialization`,
  every `movement.*.map*` key (replaced by `_movement_keys.txt`), `fisheries.catchability.file`,
  `fisheries.discards.file`, `fisheries.seasonality.file.fsh1..9`.
- Set: `population.seeding.biomass.sp0..9` (from `_seeding_keys.txt`), `population.seeding.year.max`
  (default 30; Task 5 may trim), `species.file.sp300 .. sp303 = input/roms_climatological_merged.nc`,
  `fisheries.enabled = FALSE`, `simulation.fishing.mortality.enabled = FALSE`,
  `simulation.nfisheries = 0`, `output.file.prefix = benguela`,
  `simulation.time.nyear = 50` (placeholder; Task 5 pins it).
- Copy statics into `data/benguela/input/`: `grid-mask.nc`, `predation-accessibility-25mars2015.csv`,
  every `input/*.csv` param file, and the whole `input/reproduction/` subdir. Do NOT copy
  `input/fisheries/`, `input/maps/` (source NetCDFs), the 4 separate ROMS files, or the restart file.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benguela_bundle.py`:
```python
import shutil
MASTER = ROOT / "data" / "benguela" / "benguela_all-parameters.csv"


def test_master_loads_and_is_wired():
    assert MASTER.exists(), "run scripts/build_benguela_config.py"
    raw = dict(OsmoseConfigReader().read(str(MASTER)))
    # forcing repointed to merged file
    for sp in range(300, 304):
        assert raw[f"species.file.sp{sp}"].endswith("roms_climatological_merged.nc")
    # fishing off + stripped
    assert "fisheries.catchability.file" not in raw
    assert "fisheries.discards.file" not in raw
    assert raw["simulation.nfisheries"] == "0"
    # restart-init dropped, seeding present
    assert "population.initialization.file" not in raw
    assert float(raw["population.seeding.biomass.sp1"]) == 3888750
    assert raw["output.file.prefix"] == "benguela"


def test_master_loads_without_fisheries_dir():
    # from_dict must not touch any fisheries file (input/fisheries/ is NOT bundled)
    raw = dict(OsmoseConfigReader().read(str(MASTER)))
    cfg = EngineConfig.from_dict(raw)   # raises if any _require_file is unmet
    assert cfg.n_species == 10
    assert not (ROOT / "data" / "benguela" / "input" / "fisheries").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py -k master -q`
Expected: FAIL.

- [ ] **Step 3: Write the synthesis script**

Create `scripts/build_benguela_config.py`:
```python
"""Synthesize data/benguela/benguela_all-parameters.csv from the vendored source, applying the
Benguela-bundling edit set (seeding, merged forcing, converted maps, fishing stripped)."""
from __future__ import annotations
import re
import sys
from pathlib import Path


def _flatten(master: Path) -> dict[str, str]:
    """Recursively read an OSMOSE master + its osmose.configuration.* includes into a flat dict.
    Keys lowercased; supports 'key = val' and 'key ; val'."""
    out: dict[str, str] = {}

    def read(p: Path):
        for ln in p.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            m = re.split(r"\s*[;=]\s*", s, maxsplit=1)
            if len(m) != 2:
                continue
            k, v = m[0].strip().lower(), m[1].strip()
            out[k] = v
            if k.startswith("osmose.configuration."):
                inc = (p.parent / v)
                if inc.exists():
                    read(inc)
    read(master)
    return out


def _lines(path: Path) -> dict[str, str]:
    d = {}
    for ln in path.read_text().splitlines():
        if ";" in ln:
            k, v = ln.split(";", 1)
            d[k.strip().lower()] = v.strip()
    return d


def build_config(src_dir: Path, bundle_dir: Path) -> Path:
    raw = _flatten(src_dir / "osmose-ben_seeding.R")   # seeding variant = base
    # --- drop families ---
    drop_exact = {"population.initialization.file", "osmose.configuration.initialization",
                  "fisheries.catchability.file", "fisheries.discards.file"}
    for k in list(raw):
        if k in drop_exact or k.startswith("movement.") and ".map" in k:
            del raw[k]
        elif re.match(r"fisheries\.seasonality\.file\.fsh\d+$", k):
            del raw[k]
        elif k.startswith("osmose.configuration."):
            del raw[k]   # we emit a single flat master, no includes
    # --- set scalars ---
    raw.update({
        "species.file.sp300": "input/roms_climatological_merged.nc",
        "species.file.sp301": "input/roms_climatological_merged.nc",
        "species.file.sp302": "input/roms_climatological_merged.nc",
        "species.file.sp303": "input/roms_climatological_merged.nc",
        "fisheries.enabled": "FALSE",
        "simulation.fishing.mortality.enabled": "FALSE",
        "simulation.nfisheries": "0",
        "output.file.prefix": "benguela",
        "population.seeding.year.max": "30",
        "simulation.time.nyear": "50",
    })
    # --- merge in seeding + movement blocks ---
    raw.update(_lines(bundle_dir / "_seeding_keys.txt"))
    raw.update(_lines(bundle_dir / "_movement_keys.txt"))
    # --- copy statics into the bundle ---
    import shutil
    idir = bundle_dir / "input"; idir.mkdir(parents=True, exist_ok=True)
    for f in (src_dir / "input").glob("*.csv"):
        shutil.copy(f, idir / f.name)
    shutil.copy(src_dir / "input" / "grid-mask.nc", idir / "grid-mask.nc")
    rep = src_dir / "input" / "reproduction"
    if rep.exists():
        shutil.copytree(rep, idir / "reproduction", dirs_exist_ok=True)
    # --- write flat master ---
    master = bundle_dir / "benguela_all-parameters.csv"
    master.write_text("\n".join(f"{k} ; {v}" for k, v in sorted(raw.items())) + "\n")
    return master


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "benguela_src"
    p = build_config(src, root / "data" / "benguela")
    print(f"wrote {p}")
```

- [ ] **Step 4: Run script + tests**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/build_benguela_config.py`
Then: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py -k master -q`
Expected: PASS. (If `from_dict` raises on an unexpected `_require_file`, add that key to the drop set
or copy the referenced file; re-run.)

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add scripts/build_benguela_config.py data/benguela/benguela_all-parameters.csv data/benguela/input tests/test_benguela_bundle.py
git commit -m "feat(benguela): synthesize master config (fishing stripped, forcing+maps+seeding wired)"
```

---

### Task 5: Stability validation + pin the demo horizon

**Files:**
- Create: `scripts/validate_benguela_stability.py`
- Modify: `data/benguela/benguela_all-parameters.csv` (set final `simulation.time.nyear`)
- Test: covered by Task 8's committed smoke test (this task is a one-time determination + a decision gate)

**Interfaces:**
- Consumes: `data/benguela/benguela_all-parameters.csv` (Task 4).
- Produces: the pinned `simulation.time.nyear` value (written into the master) + a printed report.

**⚠ DECISION GATE:** This is where the project risk concentrates. If NO bounded horizon exists (all
species explode or collapse even with blockers 1–4 fixed), STOP and escalate to the human — do not
ship an exploding demo. Options to raise: revisit the unfished decision, trim seeding magnitude /
`year.max`, or accept a short pinned horizon.

- [ ] **Step 1: Write the validation script**

Create `scripts/validate_benguela_stability.py`:
```python
"""Run the wired Benguela config over a long horizon and report per-species stability, to pin a safe
simulation.time.nyear. Reports seeding-fire diagnostics so instability is attributable."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

SEED = {0: 3129213, 1: 3888750, 2: 3029155, 3: 1286364, 4: 1138339,
        5: 1439984, 6: 198865, 7: 81054, 8: 575361, 9: 591907}


def run(master: Path, nyear: int) -> dict:
    raw = dict(OsmoseConfigReader().read(str(master)))
    raw["simulation.time.nyear"] = str(nyear)
    res = PythonEngine().run_in_memory(raw, seed=42)
    b = res.biomass()
    sp_cols = [c for c in b.columns if c not in ("Time", "time", "species")]
    return {c: b[c].to_numpy(dtype=float) for c in sp_cols}


def bounded(cols: dict, seeds: dict) -> dict[str, bool]:
    names = list(cols)
    verdict = {}
    for i, c in enumerate(names):
        v = cols[c]
        cap = 1000.0 * seeds.get(i, max(seeds.values()))
        verdict[c] = bool(np.all(np.isfinite(v)) and np.all(v <= cap) and np.all(v <= 1e9)
                          and v[-1] > 0)
    return verdict


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    master = root / "data" / "benguela" / "benguela_all-parameters.csv"
    for ny in (int(sys.argv[1]),) if len(sys.argv) > 1 else (100, 50, 30, 15):
        cols = run(master, ny)
        v = bounded(cols, SEED)
        print(f"nyear={ny}: bounded={sum(v.values())}/{len(v)}  fails={[k for k,ok in v.items() if not ok]}")
```

- [ ] **Step 2: Run the validation sweep**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/validate_benguela_stability.py`
Inspect the output. Pick the LARGEST `nyear` at which `bounded == 10/10`.
- If some horizon is fully bounded → that's the pinned value.
- If NO horizon is bounded → **STOP, escalate** (decision gate above).

- [ ] **Step 3: Pin nyear in the master**

Edit `data/benguela/benguela_all-parameters.csv`: set `simulation.time.nyear ; <pinned>`. If the
pinned horizon ≤ `population.seeding.year.max` (30), also trim `population.seeding.year.max` to well
below it (e.g. a 5-year warm-up) so seeding is not live for the whole demo run, and re-run Step 2 to
confirm still bounded.

- [ ] **Step 4: Commit**

```bash
cd /home/razinka/osmopy
git add scripts/validate_benguela_stability.py data/benguela/benguela_all-parameters.csv
git commit -m "feat(benguela): validate stability + pin demo horizon (nyear=<pinned>)"
```

---

### Task 6: Wire the demo into the registry

**Files:**
- Modify: `osmose/demo.py` (`list_demos`, `DEMO_INFO`, `_generate_benguela`, `generators`)
- Test: `tests/test_benguela_demo.py`

**Interfaces:**
- Consumes: `data/benguela/` bundle.
- Produces: `_generate_benguela(output_dir: Path) -> dict` with `{config_file, output_dir}`;
  `benguela` added to `list_demos()` and `DEMO_INFO`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benguela_demo.py`:
```python
from pathlib import Path
from osmose.demo import list_demos, demo_info, osmose_demo
from osmose.config.reader import OsmoseConfigReader


def test_benguela_registered():
    assert "benguela" in list_demos()
    info = demo_info("benguela")
    assert info is not None
    for field in ("title", "region", "species", "resources", "engine", "summary"):
        assert info.get(field), f"DEMO_INFO['benguela'] missing {field}"


def test_benguela_generates_and_copies(tmp_path: Path):
    out = osmose_demo("benguela", tmp_path)
    cfg = Path(out["config_file"])
    assert cfg.name == "benguela_all-parameters.csv" and cfg.exists()
    cfgdir = cfg.parent
    assert (cfgdir / "input" / "roms_climatological_merged.nc").exists()
    assert (cfgdir / "input" / "reproduction").is_dir()
    assert list((cfgdir / "maps").glob("*.csv"))
    assert not (cfgdir / "input" / "fisheries").exists()
    raw = dict(OsmoseConfigReader().read(str(cfg)))
    assert raw["simulation.nspecies"] == "10"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_demo.py -q`
Expected: FAIL (`benguela` not registered).

- [ ] **Step 3: Add `benguela` to the registry**

In `osmose/demo.py`:
1. Add `"benguela"` to the `list_demos()` return list.
2. Add a `DEMO_INFO["benguela"]` entry:
```python
    "benguela": {
        "title": "Southern Benguela",
        "region": "SE Atlantic upwelling (Benguela)",
        "species": "10 focal species",
        "resources": "4 ROMS plankton groups",
        "engine": "Python",
        "summary": "Southern Benguela upwelling ecosystem (anchovy, sardine, redeye, hakes, "
        "snoek, …) forced by ROMS plankton; unfished. Python engine only.",
    },
```
3. Add the generator (mirrors `_generate_baltic`):
```python
def _generate_benguela(output_dir: Path) -> dict:
    """Generate the Southern Benguela demo (Python-engine, unfished)."""
    data_dir = _bundled_data_dir("benguela")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)
    if data_dir is not None:
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "benguela_all-parameters.csv").write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.nspecies ; 10\n"
            "simulation.nresource ; 4\n"
            "simulation.ncpu ; 1\n"
        )
    return {"config_file": config_dir / "benguela_all-parameters.csv", "output_dir": sim_output}
```
4. Register in the `generators` dict inside `osmose_demo`: `"benguela": _generate_benguela,`.

- [ ] **Step 4: Run tests + the auto-parametrized suites**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_demo.py tests/test_demo.py tests/test_ui_load_scenarios.py -q`
Expected: PASS (including `test_demo_info_covers_all_demos_with_full_fields`,
`test_all_demos_produce_unique_configs`).

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add osmose/demo.py tests/test_benguela_demo.py
git commit -m "feat(benguela): register benguela demo (list_demos, DEMO_INFO, generator)"
```

---

### Task 7: Enforce Python-only in the Java-engine guard

**Files:**
- Modify: `osmose/runner.py` (`java_engine_block_reason`)
- Test: `tests/test_benguela_demo.py`

**Interfaces:**
- Consumes: a config dict / config path for a run.
- Produces: `java_engine_block_reason` returns a non-None reason for Benguela.

**Context:** `java_engine_block_reason` (`osmose/runner.py:17-55`) currently blocks only on
`simulation.nbackground > 0`. Benguela has 0 background but is Python-only by scope (unmerged Java
forcing story, converted maps). Add a marker-based block. Use `output.file.prefix == "benguela"` (set
in Task 4) as the marker so it keys off the config, not a hardcoded demo name.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benguela_demo.py`:
```python
from osmose.runner import java_engine_block_reason


def test_benguela_blocks_java_engine():
    raw = {"output.file.prefix": "benguela", "simulation.nbackground": "0"}
    reason = java_engine_block_reason(raw)
    assert reason is not None and "benguela" in reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_demo.py::test_benguela_blocks_java_engine -q`
Expected: FAIL (returns None).

- [ ] **Step 3: Add the marker block**

In `osmose/runner.py::java_engine_block_reason`, before the final `return None`, add (match the
function's actual dict/config accessor — read lines 17-55 first):
```python
    if str(cfg.get("output.file.prefix", "")).strip().lower() == "benguela":
        return ("The Southern Benguela demo is a Python-engine example (merged resource forcing and "
                "converted movement maps have no Java-side equivalent). Run it on the Python engine.")
```

- [ ] **Step 4: Run the test + the existing runner tests**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_demo.py::test_benguela_blocks_java_engine tests/ -k "runner or java_engine" -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add osmose/runner.py tests/test_benguela_demo.py
git commit -m "feat(benguela): block Java engine for the Python-only benguela demo"
```

---

### Task 8: Committed smoke / stability / determinism test

**Files:**
- Test: `tests/test_benguela_demo.py`

**Interfaces:**
- Consumes: the full pipeline (Tasks 1–7). Runs the demo via `osmose_demo` + `PythonEngine`.

**Context:** This is the load-bearing permanent gate. Uses the pinned `nyear` from the config. The
`np.nansum` resource check and the `1000×seeding` + `1e9` bounds are mandatory (Global Constraints).

- [ ] **Step 1: Write the smoke/stability/determinism test**

Add to `tests/test_benguela_demo.py`:
```python
import numpy as np
import xarray as xr
from osmose.engine import PythonEngine

EXPECTED_SEED = {0: 3129213, 1: 3888750, 2: 3029155, 3: 1286364, 4: 1138339,
                 5: 1439984, 6: 198865, 7: 81054, 8: 575361, 9: 591907}


def _run(tmp_path):
    out = osmose_demo("benguela", tmp_path)
    raw = dict(OsmoseConfigReader().read(str(out["config_file"])))
    return PythonEngine().run_in_memory(raw, seed=42)


def test_benguela_smoke_bounded_and_positive(tmp_path):
    res = _run(tmp_path)
    b = res.biomass()
    sp_cols = [c for c in b.columns if c not in ("Time", "time", "species")]
    assert len(sp_cols) == 10
    for i, c in enumerate(sp_cols):
        v = b[c].to_numpy(dtype=float)
        assert np.all(np.isfinite(v)), f"{c} has NaN/inf"
        assert v[-1] > 0, f"{c} collapsed to 0"
        assert np.all(v <= 1000.0 * EXPECTED_SEED[i]), f"{c} exceeds 1000x seed (explosion)"
        assert np.all(v <= 1e9), f"{c} exceeds 1e9 t (explosion)"


def test_benguela_resources_load_nonzero(tmp_path):
    # all 4 merged resources must carry biomass (nansum, not sum — NaN over land)
    ds = xr.open_dataset(Path(osmose_demo("benguela", tmp_path)["config_file"]).parent
                         / "input" / "roms_climatological_merged.nc")
    try:
        for v in ("sphy", "lphy", "szoo", "lzoo"):
            assert float(np.nansum(ds[v].values)) > 0, f"resource {v} is empty"
    finally:
        ds.close()


def test_benguela_deterministic(tmp_path):
    a = _run(tmp_path / "a").biomass()
    cols = [c for c in a.columns if c not in ("Time", "time", "species")]
    va = a[cols].to_numpy(dtype=float)
    vb = _run(tmp_path / "b").biomass()[cols].to_numpy(dtype=float)
    assert np.array_equal(va, vb), "fixed-seed runs diverge"
```

- [ ] **Step 2: Run the smoke tests**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_demo.py -q`
Expected: PASS. (If a bound fails, the stability pin in Task 5 was too loose — revisit Task 5's
decision gate. If determinism fails, ensure the run uses a single thread / fixed seed like other
demos.)

- [ ] **Step 3: Full-suite + lint + type check**

Run:
```bash
cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_demo.py tests/test_benguela_bundle.py tests/test_demo.py tests/test_ui_load_scenarios.py -q
.venv/bin/ruff format osmose/ ui/ tests/ && .venv/bin/ruff check osmose/ ui/ tests/
.venv/bin/pyright osmose/ ui/ tests/
```
Expected: all green (mark known-fragile emergent tests as skipped only if they match the documented
CI-skip set — Benguela's smoke test is deterministic and must NOT be skipped).

- [ ] **Step 4: Commit**

```bash
cd /home/razinka/osmopy
git add tests/test_benguela_demo.py
git commit -m "test(benguela): committed smoke/stability/determinism gate for the demo"
```

---

## Self-Review

**Spec coverage:** Component 1 (seeding)→Task 3; Component 2 (forcing merge)→Task 1; Component 3
(maps)→Task 2; Component 4 (master synthesis + fishing strip)→Task 4; Component 5 (stability/horizon)
→Task 5; Component 6 (demo wiring + Java guard)→Tasks 6+7; Component 7 (gates/tests)→Tasks 1–4 artifact
tests + Task 8 smoke/determinism. All spec success criteria map to a task. The Java-guard enforcement
(spec Component 6) is Task 7.

**Placeholder scan:** `simulation.time.nyear` is intentionally a placeholder (50) in Task 4 and
resolved empirically in Task 5 — this is a spec-sanctioned deferral (the horizon must be measured, not
guessed), not a plan gap. No other TBDs.

**Type/name consistency:** `merge_forcing`, `convert_maps`, `derive_seeding`, `build_config`,
`_movement_keys.txt`, `_seeding_keys.txt`, the seeding value dict, and the CSV/`np.nansum` conventions
are used identically across tasks. Master filename `benguela_all-parameters.csv` and demo key
`benguela` are consistent throughout.

**Note for executor:** Task 5 contains a hard DECISION GATE — if the config cannot be made bounded,
STOP and escalate rather than shipping an exploding demo.
