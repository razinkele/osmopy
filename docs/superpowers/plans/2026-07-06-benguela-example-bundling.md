# Southern Benguela Example Bundling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle the Southern Benguela (`osmose-ben_v4.3_Florance`) config as a runnable, stable,
unfished `benguela` demo in osmopy's registry, on the Python engine, with tests.

**Architecture:** One-time build scripts under `scripts/` read the external source clone (passed via
`--source`) and emit the committed bundle `data/benguela/` (merged resource forcing, CSV movement
maps, analytic seeding, fishing stripped). `osmose/demo.py` gains a `_generate_benguela` generator;
`osmose/runner.py` gains a real Python-only guard. Correctness is proven by a committed
smoke/stability test that runs the demo at an empirically pinned horizon.

**Tech Stack:** Python 3.12/3.13, numpy, xarray/netCDF4, pandas, pytest, the osmopy engine
(`osmose.config.reader.OsmoseConfigReader`, `osmose.engine.PythonEngine`,
`osmose.engine.config.EngineConfig`, `osmose.engine.resources.ResourceState`,
`osmose.engine.movement_maps`, `osmose.engine.grid.Grid`).

## Plan-review status (feasibility CONFIRMED)

A 10-agent adversarial workflow review + a direct feasibility spike were run before implementation.
Key result: **the fully-wired config (merged forcing + correctly-oriented CSV maps + seeding, fishing
stripped) runs STABLE and BOUNDED** — `nyear=1` and `nyear=5` both give no-NaN, 10/10 species finite &
positive & ≥3 orders of magnitude under the `1000×seed` cap. The review caught one root-cause defect
(movement-CSV vertical-flip, below) that had produced both the NaN cascade and the 10²² explosion; the
fix is folded into Task 2 and spike-verified. Task 5's decision gate remains as a safety net but is
unlikely to fire.

## Global Constraints

- **Python-engine example only.** No native-4.4.x conversion, no cross-engine Java parity. osmopy
  auto-migrates the 4.3.3 keys to 4.4.0 on read (verified — 7 deprecated keys migrate).
- **Unfished v1.** Fishing disabled AND its unconditional-read file keys stripped.
- **Source clone is external, read at build time, NOT committed.** Pass it via `--source`. During this
  build it is:
  `SRC = /tmp/claude-1000/-home-razinka-osmose/f7b91731-5bf2-427b-aaab-4e339882ae8b/scratchpad/osmose-ben/osmose-ben_v4.3_Florance`
  Only `data/benguela/` is committed (Success Criterion 5: self-contained ~1.6 MB).
- **Seeding values** (from `<SRC>/osmose-ben_seeding.R`):
  `sp0=3129213 sp1=3888750 sp2=3029155 sp3=1286364 sp4=1138339 sp5=1439984 sp6=198865 sp7=81054
  sp8=575361 sp9=591907` (tonnes).
- **Resource forcing = ONE multi-variable NetCDF** with variables named exactly
  `sphy/lphy/szoo/lzoo` (= `species.name.sp300-303`), dims `time=24, ny=62, nx=56`.
- **Movement CSV format:** semicolon-delimited `(ny=62) × (nx=56)` grid; `-99` = land/absent,
  `0..1` = ocean presence. **The grid MUST be written `np.flipud`'d** — the runtime loader
  `movement_maps.py::_load_csv_grid` reverses rows on load (`grid_row = ny-1-csv_row_idx`); every other
  CSV-grid writer in the repo (`osmose/maps/builder.py::to_csv_text`) flips first. Every emitted map
  index carries `movement.species.mapN` (binds index→species — omitting it orphans the species →
  `is_out`), `movement.file.mapN`, `movement.steps.mapN`, `movement.initialage.mapN`,
  `movement.lastage.mapN`.
- **Resource-biomass checks use `np.nansum` / ocean-mask**, never `.sum()` (forcing is NaN over land,
  ~54% of cells, like EEC's shipped forcing).
- **Stability bound (smoke gate):** for each species and all timesteps,
  `biomass[t] ≤ 1000 × seeding_biomass[species]` AND `biomass[t] ≤ 1e9` tonnes; no NaN; final
  biomass > 0.
- **Determinism:** `PythonEngine().run_in_memory(raw, seed=42)`.
- `scripts/` is OUTSIDE the ruff/pyright scope; `osmose/ ui/ tests/` must stay ruff+pyright clean.
- Branch: `feat/benguela-example`. Commit after each task.

## File Structure

- `data/benguela/` — **the only committed artifact** (what `_generate_benguela` copies):
  - `benguela_all-parameters.csv` — synthesized flat master.
  - `input/` — `grid-mask.nc`, `roms_climatological_merged.nc`, `predation-accessibility-25mars2015.csv`,
    species/param CSVs, `reproduction/reproduction-seasonality-sp{0..9}.csv`.
  - `maps/` — converted CSV presence grids.
  - `_movement_keys.txt`, `_seeding_keys.txt` — intermediate key blocks (committed; harmless, ignored
    by the engine which only reads `*all-parameters*.csv`).
- `scripts/merge_benguela_forcing.py` — Task 1.
- `scripts/convert_benguela_maps.py` — Task 2.
- `scripts/derive_benguela_seeding.py` — Task 3.
- `scripts/build_benguela_config.py` — Task 4.
- `scripts/validate_benguela_stability.py` — Task 5 (one-time; determines pinned nyear).
- `osmose/demo.py` — Task 6 (registry wiring).
- `osmose/runner.py` — Task 7 (Python-only guard).
- `tests/test_benguela_bundle.py` — artifact tests (Tasks 1–4).
- `tests/test_benguela_demo.py` — demo + smoke/stability/determinism tests (Tasks 6–8).

Build/run order: **1,2,3 → 4 → 5 → 6 → 7 → 8**. Each build script takes the source path as `sys.argv[1]`
(the `SRC` above).

---

### Task 1: Merge resource forcing into one NetCDF

**Files:**
- Create: `scripts/merge_benguela_forcing.py`, `data/benguela/input/roms_climatological_merged.nc`
- Test: `tests/test_benguela_bundle.py`

**Interfaces:**
- Produces: `merge_forcing(src_dir: Path, out_path: Path) -> None` — writes a 4-variable NetCDF.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benguela_bundle.py`:
```python
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "benguela"
MERGED = BUNDLE / "input" / "roms_climatological_merged.nc"
RES_VARS = ["sphy", "lphy", "szoo", "lzoo"]


def test_merged_forcing_has_all_four_resources():
    assert MERGED.exists(), "run scripts/merge_benguela_forcing.py <SRC>"
    ds = xr.open_dataset(MERGED)
    try:
        for v in RES_VARS:
            assert v in ds.data_vars, f"{v} missing from merged forcing"
            assert ds[v].shape == (24, 62, 56), f"{v} wrong dims {ds[v].shape}"
            assert float(np.nansum(ds[v].values)) > 0, f"{v} is all-zero/NaN"
    finally:
        ds.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py::test_merged_forcing_has_all_four_resources -q`
Expected: FAIL (`run scripts/merge_benguela_forcing.py`).

- [ ] **Step 3: Write the merge script**

Create `scripts/merge_benguela_forcing.py`:
```python
"""Merge Benguela's 4 single-variable ROMS forcing NetCDFs into one multi-variable file.

osmopy's ResourceState loads a SINGLE resource NetCDF and looks up each resource BY NAME in it
(resources.py:216). Benguela ships one file per resource, so only the first would load. Merge them
so all 4 (sphy/lphy/szoo/lzoo) resolve. Pass the external source clone dir as argv[1].
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
    for name, fname in RES.items():
        ds = xr.open_dataset(src_dir / "input" / fname)
        var = list(ds.data_vars)[0]      # each file holds exactly one data variable
        data_vars[name] = ds[var].rename(name)
    merged = xr.Dataset(data_vars)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(out_path)
    merged.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1])
    merge_forcing(src, root / "data" / "benguela" / "input" / "roms_climatological_merged.nc")
    print("wrote merged forcing")
```

- [ ] **Step 4: Run the merge script (pass SRC)**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/merge_benguela_forcing.py "$SRC"`
(where `$SRC` is the Global-Constraints source path). Expected: `wrote merged forcing`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py::test_merged_forcing_has_all_four_resources -q`
Expected: PASS. (If a var's shape isn't `(24,62,56)`, transpose to `(time, ny, nx)` in the script.)

- [ ] **Step 6: Commit**

```bash
cd /home/razinka/osmopy
git add data/benguela/input/roms_climatological_merged.nc scripts/merge_benguela_forcing.py tests/test_benguela_bundle.py
git commit -m "feat(benguela): merge ROMS forcing into one multi-var NetCDF"
```

---

### Task 2: Convert NetCDF movement maps to CSV (with the load-time flip)

**Files:**
- Create: `scripts/convert_benguela_maps.py`, `data/benguela/maps/*.csv`,
  `data/benguela/_movement_keys.txt`
- Test: `tests/test_benguela_bundle.py`

**Interfaces:**
- Produces: `convert_maps(src_dir, maps_out_dir, keys_out_path) -> list[dict]`; the CSV files; and a
  `_movement_keys.txt` of `key ; value` lines (the rewritten movement block) for Task 4.

**Context:** Each source `movement.*.mapN` declares `movement.species.mapN`, `movement.variable.mapN`
(=`stage0/1/2`, a variable in `input/maps/<species>.nc`, shape `(24,62,56)`),
`movement.initialAge.mapN`, `movement.lastAge.mapN`, `movement.file.mapN` (the species `.nc`). osmopy
reads static CSV grids and expresses time-variation via multiple indices with different
`movement.steps.mapN`. So each source index expands into one osmopy index per DISTINCT time-slice.
Land cells (grid-mask==0) → `-99`; ocean → the slice value. **The written grid MUST be `np.flipud`'d**
because `_load_csv_grid` reverses rows on load — this is the single most important detail (a spike
confirmed that without the flip the fish are placed on land → NaN cascade / explosion; with it the run
is stable and bounded).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_benguela_bundle.py`:
```python
from osmose.engine.movement_maps import _load_csv_grid
from osmose.engine.grid import Grid

MAPS_DIR = BUNDLE / "maps"
MOVE_KEYS = BUNDLE / "_movement_keys.txt"
GRID_NC = BUNDLE / "input" / "grid-mask.nc"
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
    assert MOVE_KEYS.exists(), "run scripts/convert_benguela_maps.py <SRC>"
    keys = _load_movement_keys()
    idxs = sorted({int(k.split(".map")[1]) for k in keys if k.startswith("movement.species.map")})
    assert idxs, "no movement.species.mapN emitted (species would be orphaned -> is_out)"
    for n in idxs:
        assert keys[f"movement.species.map{n}"] in BEN_SPECIES
        assert (BUNDLE / keys[f"movement.file.map{n}"]).exists()
        assert f"movement.steps.map{n}" in keys
        assert f"movement.initialage.map{n}" in keys
        assert f"movement.lastage.map{n}" in keys


def test_movement_csv_loads_ocean_within_grid_via_real_loader():
    # Load via the PRODUCTION loader (which flips rows) + the real grid ocean mask.
    assert GRID_NC.exists(), "run scripts/build_benguela_config.py to copy grid-mask.nc"
    ocean = Grid.from_netcdf(str(GRID_NC)).ocean_mask   # (62,56) bool, engine convention
    for csv in sorted(MAPS_DIR.glob("*.csv")):
        grid = _load_csv_grid(str(csv), 62, 56)
        present = (grid > 0) & (grid != -99)
        assert present[~ocean].sum() == 0, f"{csv.name} places presence on land (flip/orientation)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py -k movement -q`
Expected: FAIL. (`test_movement_csv_loads...` also needs `grid-mask.nc` in the bundle — Task 4 copies
it; for a Task-2-only run, temporarily copy it, or accept this test goes green after Task 4.)

- [ ] **Step 3: Write the converter script**

Create `scripts/convert_benguela_maps.py`:
```python
"""Convert Benguela's per-species NetCDF movement maps into osmopy CSV maps.

Each source movement.*.mapN references a stage variable (24,62,56) in input/maps/<species>.nc. osmopy
reads static CSV grids; time-variation is expressed by multiple indices with different
movement.steps.mapN. Each source index -> one osmopy index per DISTINCT 62x56 time-slice.
CSV format: semicolon-delimited, -99=land, ocean value = slice value. The grid is written np.flipud'd
because _load_csv_grid reverses rows on load (mirrors osmose/maps/builder.py::to_csv_text). Pass the
external source clone dir as argv[1].
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
    return ocean  # (62,56), True=ocean, engine (unflipped) convention


def _write_csv(path: Path, grid: np.ndarray) -> None:
    # flip so CSV row 0 = grid row ny-1, matching _load_csv_grid's row-reversal on read
    lines = [";".join(f"{v:g}" for v in row) + ";" for row in np.flipud(grid)]
    path.write_text("\n".join(lines) + "\n")


def convert_maps(src_dir: Path, maps_out: Path, keys_out: Path) -> list[dict]:
    maps_out.mkdir(parents=True, exist_ok=True)
    raw = dict(OsmoseConfigReader().read(str(src_dir / "osmose-ben.R")))
    ocean = _grid_ocean(src_dir)
    src_idxs = sorted({int(k.split(".map")[1]) for k in raw if k.startswith("movement.species.map")})
    out_rows: list[dict] = []
    seen: dict[bytes, str] = {}
    out_n = 0
    for si in src_idxs:
        sp = raw[f"movement.species.map{si}"]
        stage = raw[f"movement.variable.map{si}"]
        a0 = raw[f"movement.initialage.map{si}"]
        a1 = raw[f"movement.lastage.map{si}"]
        da = xr.open_dataset(src_dir / raw[f"movement.file.map{si}"])[stage].values  # (24,62,56)
        groups: dict[bytes, list[int]] = {}
        oriented = []
        flip = False
        for t in range(da.shape[0]):
            s = da[t]
            if t == 0:
                pa = ((s > 0) & ~np.isnan(s)) & ~ocean
                pf = ((np.flipud(s) > 0) & ~np.isnan(np.flipud(s))) & ~ocean
                flip = pf.sum() < pa.sum()
            s = np.flipud(s) if flip else s
            g = np.where(ocean, np.nan_to_num(s), -99.0)
            oriented.append(g)
            groups.setdefault(g.tobytes(), []).append(t)
        for gb, steps in groups.items():
            g = oriented[steps[0]]
            if gb in seen:
                rel = seen[gb]
            else:
                fn = f"{sp}_{stage}_g{out_n}.csv"
                _write_csv(maps_out / fn, g)
                rel = f"maps/{fn}"
                seen[gb] = rel
                out_n += 1
            out_rows.append({"species": sp, "file": rel, "steps": steps, "a0": a0, "a1": a1})
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
    src = Path(sys.argv[1])
    convert_maps(src, root / "data" / "benguela" / "maps",
                 root / "data" / "benguela" / "_movement_keys.txt")
    print("maps converted")
```

- [ ] **Step 4: Run the converter, then the tests**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/convert_benguela_maps.py "$SRC"`
Then (after ensuring `data/benguela/input/grid-mask.nc` exists — copy from `$SRC/input/` if Task 4
hasn't run yet): `PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py -k movement -q`
Expected: PASS with 0 land violations (spike-confirmed with the flip).

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add data/benguela/maps data/benguela/_movement_keys.txt scripts/convert_benguela_maps.py tests/test_benguela_bundle.py
git commit -m "feat(benguela): convert NetCDF movement maps to CSV (flip for loader row-reversal)"
```

---

### Task 3: Derive analytic seeding block

**Files:**
- Create: `scripts/derive_benguela_seeding.py`, `data/benguela/_seeding_keys.txt`
- Test: `tests/test_benguela_bundle.py`

**Interfaces:**
- Produces: `derive_seeding(src_dir, out_path) -> dict[int, float]` + a `_seeding_keys.txt` of
  `population.seeding.biomass.spN ; <tonnes>` lines for Task 4.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benguela_bundle.py`:
```python
SEED_KEYS = BUNDLE / "_seeding_keys.txt"
EXPECTED_SEED = {0: 3129213, 1: 3888750, 2: 3029155, 3: 1286364, 4: 1138339,
                 5: 1439984, 6: 198865, 7: 81054, 8: 575361, 9: 591907}


def test_seeding_block_matches_authors_values():
    assert SEED_KEYS.exists(), "run scripts/derive_benguela_seeding.py <SRC>"
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
cross-checked against the restart file's per-species standing stock. Pass source clone as argv[1]."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from osmose.config.reader import OsmoseConfigReader


def derive_seeding(src_dir: Path, out_path: Path) -> dict[int, float]:
    raw = dict(OsmoseConfigReader().read(str(src_dir / "osmose-ben_seeding.R")))
    seed = {sp: float(raw[f"population.seeding.biomass.sp{sp}"]) for sp in range(10)}
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
    src = Path(sys.argv[1])
    derive_seeding(src, root / "data" / "benguela" / "_seeding_keys.txt")
    print("seeding derived")
```

- [ ] **Step 4: Run script + test**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/derive_benguela_seeding.py "$SRC"`
Then: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py::test_seeding_block_matches_authors_values -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add scripts/derive_benguela_seeding.py data/benguela/_seeding_keys.txt tests/test_benguela_bundle.py
git commit -m "feat(benguela): derive analytic seeding block from authors' seeding config"
```

---

### Task 4: Synthesize the master config + integration smoke

**Files:**
- Create: `scripts/build_benguela_config.py`, `data/benguela/benguela_all-parameters.csv`, copied
  statics under `data/benguela/input/`
- Test: `tests/test_benguela_bundle.py`

**Interfaces:**
- Consumes: `_movement_keys.txt` (Task 2), `_seeding_keys.txt` (Task 3), merged forcing (Task 1).
- Produces: `build_config(src_dir, bundle_dir) -> Path` writing `benguela_all-parameters.csv`.

**Context — exact edit set applied to the flattened source config:**
- Drop key families: `population.initialization.file`, `osmose.configuration.*` (single flat master,
  no includes), every `movement.*.map*` key (replaced by `_movement_keys.txt`),
  `fisheries.catchability.file`, `fisheries.discards.file`, `fisheries.seasonality.file.fsh1..9`, AND
  the whole `fisheries.movement.*` family (dangles to unbundled `mapFleets.nc`; harmless at
  `nfisheries=0` but stripped for cleanliness).
- Set: `population.seeding.biomass.sp0..9` (from `_seeding_keys.txt`), `population.seeding.year.max=30`
  (Task 5 may trim), `species.file.sp300..303 = input/roms_climatological_merged.nc`,
  `fisheries.enabled=FALSE`, `simulation.fishing.mortality.enabled=FALSE`, `simulation.nfisheries=0`
  (load-bearing — short-circuits the fishing loaders), `output.file.prefix=benguela`,
  `simulation.time.nyear=50` (placeholder; Task 5 pins it).
- Copy statics into `data/benguela/input/`: `grid-mask.nc`, `predation-accessibility-25mars2015.csv`,
  every `input/*.csv`, and the whole `input/reproduction/` subdir. Do NOT copy `input/fisheries/`,
  `input/maps/` NetCDFs, the 4 separate ROMS files, or the restart file.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_benguela_bundle.py`:
```python
from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine import PythonEngine

MASTER = BUNDLE / "benguela_all-parameters.csv"


def test_master_loads_and_is_wired():
    assert MASTER.exists(), "run scripts/build_benguela_config.py <SRC>"
    raw = dict(OsmoseConfigReader().read(str(MASTER)))
    for sp in range(300, 304):
        assert raw[f"species.file.sp{sp}"].endswith("roms_climatological_merged.nc")
    assert "fisheries.catchability.file" not in raw
    assert "fisheries.discards.file" not in raw
    assert not any(k.startswith("fisheries.movement.") for k in raw)
    assert raw["simulation.nfisheries"] == "0"
    assert "population.initialization.file" not in raw
    assert float(raw["population.seeding.biomass.sp1"]) == 3888750
    assert raw["output.file.prefix"] == "benguela"


def test_master_loads_without_fisheries_dir():
    raw = dict(OsmoseConfigReader().read(str(MASTER)))
    cfg = EngineConfig.from_dict(raw)   # raises if any _require_file is unmet
    assert cfg.n_species == 10
    assert not (BUNDLE / "input" / "fisheries").exists()


def test_master_runs_without_nan_integration_smoke():
    # nyear=1 engine run: catches the class of integration failure (mis-oriented maps -> NaN) at
    # Task 4's own gate, not only at Task 5. Spike-confirmed to be NaN-free with correct maps.
    raw = dict(OsmoseConfigReader().read(str(MASTER)))
    raw["simulation.time.nyear"] = "1"
    res = PythonEngine().run_in_memory(raw, seed=42)
    b = res.biomass()
    cols = [c for c in b.columns if c not in ("Time", "time", "species")]
    import numpy as np
    v = b[cols].to_numpy(dtype=float)
    assert not np.isnan(v).any(), "integration produced NaN biomass (check map orientation flip)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py -k master -q`
Expected: FAIL.

- [ ] **Step 3: Write the synthesis script**

Create `scripts/build_benguela_config.py`:
```python
"""Synthesize data/benguela/benguela_all-parameters.csv from the external source clone, applying the
Benguela-bundling edit set (seeding, merged forcing, converted maps, fishing stripped). argv[1]=src."""
from __future__ import annotations
import re
import shutil
import sys
from pathlib import Path


def _flatten(master: Path) -> dict[str, str]:
    out: dict[str, str] = {}

    def read(p: Path):
        for ln in p.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            parts = re.split(r"\s*[;=]\s*", s, maxsplit=1)
            if len(parts) != 2:
                continue
            k, v = parts[0].strip().lower(), parts[1].strip()
            out[k] = v
            if k.startswith("osmose.configuration.") and (p.parent / v).exists():
                read(p.parent / v)
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
    raw = _flatten(src_dir / "osmose-ben_seeding.R")
    drop_exact = {"population.initialization.file", "osmose.configuration.initialization",
                  "fisheries.catchability.file", "fisheries.discards.file"}
    for k in list(raw):
        if (k in drop_exact
                or (k.startswith("movement.") and ".map" in k)
                or k.startswith("fisheries.movement.")
                or k.startswith("osmose.configuration.")
                or re.match(r"fisheries\.seasonality\.file\.fsh\d+$", k)):
            del raw[k]
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
    raw.update(_lines(bundle_dir / "_seeding_keys.txt"))
    raw.update(_lines(bundle_dir / "_movement_keys.txt"))
    idir = bundle_dir / "input"; idir.mkdir(parents=True, exist_ok=True)
    for f in (src_dir / "input").glob("*.csv"):
        shutil.copy(f, idir / f.name)
    shutil.copy(src_dir / "input" / "grid-mask.nc", idir / "grid-mask.nc")
    rep = src_dir / "input" / "reproduction"
    if rep.exists():
        shutil.copytree(rep, idir / "reproduction", dirs_exist_ok=True)
    master = bundle_dir / "benguela_all-parameters.csv"
    master.write_text("\n".join(f"{k} ; {v}" for k, v in sorted(raw.items())) + "\n")
    return master


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1])
    p = build_config(src, root / "data" / "benguela")
    print(f"wrote {p}")
```

- [ ] **Step 4: Run script + tests**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/build_benguela_config.py "$SRC"`
Then: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_bundle.py -q`
Expected: PASS (all artifact tests incl. the movement-loader test — grid-mask.nc is now bundled — and
the nyear=1 NaN-free smoke). If `from_dict` raises on an unexpected `_require_file`, add the key to the
drop set or copy the file. If the NaN smoke fails, re-check Task 2's flip.

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy
git add scripts/build_benguela_config.py data/benguela/benguela_all-parameters.csv data/benguela/input tests/test_benguela_bundle.py
git commit -m "feat(benguela): synthesize master (fishing stripped, forcing+maps+seeding wired)"
```

---

### Task 5: Stability validation + pin the demo horizon

**Files:**
- Create: `scripts/validate_benguela_stability.py`
- Modify: `data/benguela/benguela_all-parameters.csv` (set final `simulation.time.nyear`)
- Test: covered by Task 8's committed smoke test (this is a one-time determination + decision gate)

**Interfaces:**
- Consumes: `data/benguela/benguela_all-parameters.csv` (Task 4).
- Produces: the pinned `simulation.time.nyear` (written into the master) + a printed report.

**⚠ DECISION GATE (now a safety net):** The feasibility spike already showed the wired config is
bounded at nyear=1 and nyear=5. This task finds the largest safe horizon. If — contrary to the spike —
NO horizon is bounded, STOP and escalate (revisit unfished decision / seeding magnitude / `year.max`).

- [ ] **Step 1: Write the validation script (with attribution diagnostics)**

Create `scripts/validate_benguela_stability.py`:
```python
"""Run the wired Benguela config over long horizons and report per-species stability + seeding
diagnostics, to pin a safe simulation.time.nyear. Diagnostics attribute instability (seeding
re-injection vs food-web) per the spec."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

SEED = {0: 3129213, 1: 3888750, 2: 3029155, 3: 1286364, 4: 1138339,
        5: 1439984, 6: 198865, 7: 81054, 8: 575361, 9: 591907}


def run(master: Path, nyear: int):
    raw = dict(OsmoseConfigReader().read(str(master)))
    raw["simulation.time.nyear"] = str(nyear)
    raw["output.ssb.enabled"] = "true"
    res = PythonEngine().run_in_memory(raw, seed=42)
    b = res.biomass()
    cols = [c for c in b.columns if c not in ("Time", "time", "species")]
    bio = {c: b[c].to_numpy(dtype=float) for c in cols}
    try:
        s = res.ssb()
        ssb = {c: s[c].to_numpy(dtype=float) for c in cols if c in s.columns}
    except Exception:
        ssb = {}
    return cols, bio, ssb


def bounded(cols, bio) -> dict[str, bool]:
    v = {}
    for i, c in enumerate(cols):
        x = bio[c]
        cap = 1000.0 * SEED.get(i, max(SEED.values()))
        v[c] = bool(np.all(np.isfinite(x)) and np.all(x <= cap) and np.all(x <= 1e9) and x[-1] > 0)
    return v


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    master = root / "data" / "benguela" / "benguela_all-parameters.csv"
    sweep = (int(sys.argv[1]),) if len(sys.argv) > 1 else (100, 50, 30, 15)
    for ny in sweep:
        cols, bio, ssb = run(master, ny)
        v = bounded(cols, bio)
        print(f"nyear={ny}: bounded={sum(v.values())}/{len(v)}  fails={[k for k, ok in v.items() if not ok]}")
        # attribution: for each species, first step natural SSB exceeds its seed (seeding no longer needed)
        for i, c in enumerate(cols):
            if c in ssb:
                over = np.where(ssb[c] > SEED[i])[0]
                first = int(over[0]) if len(over) else -1
                print(f"    {c:18s} bio[-1]={bio[c][-1]:.3g} ssb>seed@step={first}")
```

- [ ] **Step 2: Run the validation sweep**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/validate_benguela_stability.py`
Pick the LARGEST `nyear` with `bounded == 10/10`. If none → STOP, escalate (gate above).

- [ ] **Step 3: Pin nyear in the master**

Edit `data/benguela/benguela_all-parameters.csv`: `simulation.time.nyear ; <pinned>`. If pinned ≤
`population.seeding.year.max` (30), also trim `population.seeding.year.max` to a short warm-up (e.g. 5)
so seeding isn't live for the whole demo, and re-run Step 2 to confirm still bounded.

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
Expected: FAIL.

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
Expected: PASS (incl. `test_demo_info_covers_all_demos_with_full_fields` and
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
- Produces: `java_engine_block_reason` returns a non-None reason for a Benguela config.

**Context:** `java_engine_block_reason(config, jar_version=None)` (`osmose/runner.py:17-55`) returns
`None` at its FIRST statement `if n_bg <= 0: return None` (line 32-33). Benguela has 0 background, so a
check placed later never runs. The marker check MUST be the FIRST statement in the function (before the
`n_bg` computation), and MUST use the real parameter name `config` (not `cfg`).

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
Expected: FAIL (returns None — the `n_bg <= 0` early return fires first for a 0-background config).

- [ ] **Step 3: Add the marker block at the TOP of the function**

In `osmose/runner.py::java_engine_block_reason`, insert as the FIRST statement after the docstring,
before the `try:`/`n_bg` computation and the `if n_bg <= 0: return None` early exit:
```python
    if str(config.get("output.file.prefix", "")).strip().lower() == "benguela":
        return ("The Southern Benguela demo is a Python-engine example (merged resource forcing and "
                "converted movement maps have no Java-side equivalent). Run it on the Python engine.")
```
(Use `config`, the actual parameter name — NOT `cfg`.)

- [ ] **Step 4: Run the test + existing runner tests**

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

**Context:** Load-bearing permanent gate at the pinned `nyear`. The `1000×seed` + `1e9` bounds and the
`np.nansum` resource check are mandatory (Global Constraints). Biomass column order == species-index
order sp0..sp9 (verified — `b.columns` are the species names in `species.name.spN` order), so
`EXPECTED_SEED[i]` aligns with `sp_cols[i]`.

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
    b = _run(tmp_path).biomass()
    sp_cols = [c for c in b.columns if c not in ("Time", "time", "species")]
    assert len(sp_cols) == 10
    for i, c in enumerate(sp_cols):
        v = b[c].to_numpy(dtype=float)
        assert np.all(np.isfinite(v)), f"{c} has NaN/inf"
        assert v[-1] > 0, f"{c} collapsed to 0"
        assert np.all(v <= 1000.0 * EXPECTED_SEED[i]), f"{c} exceeds 1000x seed (explosion)"
        assert np.all(v <= 1e9), f"{c} exceeds 1e9 t (explosion)"


def test_benguela_resources_load_nonzero(tmp_path):
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
Expected: PASS. (If a bound fails, Task 5's pin was too loose — revisit its decision gate.)

- [ ] **Step 3: Full-suite + lint + type check**

Run:
```bash
cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python -m pytest tests/test_benguela_demo.py tests/test_benguela_bundle.py tests/test_demo.py tests/test_ui_load_scenarios.py -q
.venv/bin/ruff format osmose/ ui/ tests/ && .venv/bin/ruff check osmose/ ui/ tests/
.venv/bin/pyright osmose/ ui/ tests/
```
Expected: all green. Benguela's smoke test is deterministic and must NOT be CI-skipped.

- [ ] **Step 4: Commit**

```bash
cd /home/razinka/osmopy
git add tests/test_benguela_demo.py
git commit -m "test(benguela): committed smoke/stability/determinism gate for the demo"
```

---

## Self-Review

**Spec coverage:** seeding→Task 3; forcing merge→Task 1; maps→Task 2; master synthesis + fishing
strip→Task 4; stability/horizon→Task 5; demo wiring→Task 6; Java guard→Task 7; gates→Tasks 1–4 artifact
tests + Task 4 integration smoke + Task 8 smoke/determinism. All spec success criteria map to a task.

**Placeholder scan:** `simulation.time.nyear=50` in Task 4 is a spec-sanctioned placeholder resolved
empirically in Task 5. No other TBDs.

**Type/name consistency:** `merge_forcing`, `convert_maps`, `derive_seeding`, `build_config`,
`_movement_keys.txt`, `_seeding_keys.txt`, `benguela_all-parameters.csv`, demo key `benguela`, the
seeding dict, `np.flipud`/`np.nansum`/`_load_csv_grid` conventions used identically across tasks.

**Plan-review corrections folded in:** Task 2 CSV `np.flipud` + real-loader test (root-cause fix,
spike-verified); Task 4 `fisheries.movement.*` strip + nyear=1 NaN smoke; Task 5 SSB attribution
diagnostics; Task 7 top-of-function placement + `config` accessor; vendoring dropped (source read via
`--source`, not committed); unused imports removed. Feasibility spike confirmed the wired config is
stable & bounded (no NaN, ≥3 OoM under cap at nyear 1 and 5).
