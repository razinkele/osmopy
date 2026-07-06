# BoB 4.4.1 Migration + Phase 3 ICES Consistency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `data/examples` (Bay of Biscay) from Java 4.3.3 to a fully-native 4.4.1 config by bin-averaging its 365-day resource forcing to 24 steps, gate the migration, then run a Phase-3 ICES cross-engine consistency check across EEC + BoB.

**Architecture:** BoB's resource forcing NetCDF has 365 daily steps but the sim runs 24 steps/year; the 4.4.1 engine requires the forcing's steps/year to divide 24. We resample the NetCDF to 24 bin-averaged steps, convert the config to fully-native 4.4.1 (drop legacy `ltl.*`, add `species.file`/rename `species.tl`), then verify with a Python baseline, a 4.4.1 jar smoke, a load-path-equivalence bit-exact gate, and a cross-engine statistical parity gate. Phase 3 reuses those runs for an ICES `magnitude_factor` consistency check.

**Tech Stack:** Python 3.12, xarray/netCDF4 (NetCDF), NumPy/SciPy (stats), pytest, the OSMOSE Java jar (subprocess), the ICES MCP tools + `osmose/validation/ices.py`. Run with `.venv/bin/python`, `PYTHONPATH=.` for scripts that import `ui.pages.run`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-06-bob-440-migration-phase3-ices-design.md` — read it; every task traces to it.
- **Resampling method:** bin-average — input day `d` → output step `floor(d * 24 / 365)`, mean each bin. Conserves each window's mean. Do NOT subsample/interpolate.
- **Fully-native, matching EEC:** the migrated BoB carries per-species `species.type/name/file/size.min/size.max/trophic.level/accessibility2fish` and NO `ltl.*` keys. `species.biomass.*` keys are NOT baked on disk (emitted at Java-stage time by `_emit_resource_biomass_forcing`).
- **BoB is not dynamics-neutral:** the resample changes what the Python engine reads (it subsamples the 365 file today). So the migration↔old comparison is characterized, not gated bit-exact; only the *key conversion* is gated bit-exact (Task 6).
- **Determinism:** `PythonEngine().run_in_memory(raw, seed=<int>)` — the `seed=` arg drives determinism. Do NOT rely on `simulation.rng.fixed` (dead key the engine never reads).
- **Both jars stay bundled** (`osmose-java/osmose_4.3.3-*.jar` = rollback; `osmose-java/osmose-4.4.1-*.jar` = default). No bare write-default flip; no prod redeploy.
- **Config snapshot** `data/examples_433_orig/` is the config rollback and the source for the Task 6 intermediate + Task 9 OLD arm.
- **CI:** real-engine ensemble/jar gates (Tasks 5, 9, 12) are numerically non-reproducible across runner cores → mark CI-skip (`@pytest.mark.skipif`/manual), run locally. Deterministic pieces (Tasks 2, 4, 6) run on CI.
- **Rescope trigger:** if the Task 5 smoke fails for a reason UNRELATED to resource forcing (e.g. BoB's legacy per-species fishing on 4.4.1), STOP and rescope — a fishing migration is separate, unbudgeted work.
- **`OsmoseResults` reader prefix MUST equal the config's `output.file.prefix`, NOT the config-dir name.** BoB writes `biscay_*.csv` (`data/examples/osm_param-output.csv:3` `output.file.prefix ; biscay`); EEC writes `eec_*.csv`. So every `OsmoseResults(odir, prefix=P)` must set `P = raw["output.file.prefix"]` (read from the config dict). Deriving the prefix from the directory name (`eec_full`, `bob`) makes `OsmoseResults` glob `{prefix}_{type}*.csv`, match nothing, and raise `FileNotFoundError` under the default `strict=True` (`osmose/results.py:317,731,757-758`). Java-arm reads should also pass `strict=False` so a mismatch degrades to a reported-empty ensemble instead of crashing mid-run. (Confirmed by the plan review — this was the top defect.)
- **The parity harness runs each arm ONLY if that arm can load the given config.** BoB has a two-config split: the NATIVE 24-step config runs on {Python, 4.4.1-Java} (the 4.3.3 jar can't load native resource forcing); the ORIGINAL 365-step ltl snapshot runs on {Python, 4.3.3-Java} (the 4.4.1 jar can't consume 365 steps — the original blocker). So the harness needs an `--engines` selector; NEVER run 4.4.1 on the 365-step config or 4.3.3 on the native config.

---

## File Structure

**Create:**
- `scripts/resample_bob_forcing.py` — one-shot: 365-step NetCDF → 24-step bin-averaged NetCDF.
- `data/examples/ltl/roms_n2p2z2d2_biscay_24step.nc` — output artifact (committed).
- `data/examples_433_orig/` — snapshot of the pre-migration config tree (committed).
- `scripts/bob_forcing_characterization.py` — report Python-365 vs Python-24 divergence (A4.3).
- `data/examples/reference/ices_snapshots/` + `data/eec_full/reference/ices_snapshots/` — ICES snapshots (B1).
- `scripts/phase3_ices_consistency.py` — cross-engine `magnitude_factor` consistency (B2).
- Tests: `tests/test_resample_bob_forcing.py`, `tests/test_migrate_bob_native.py`, `tests/test_bob_loadpath_equiv.py`, `tests/test_bob_440_smoke.py`.

**Modify:**
- `scripts/migrate_bundled_to_440.py` — add `examples` to IN_SCOPE + a name-gated BoB conversion.
- `scripts/native_440_parity.py` — add a `bob-loadpath` load-path-equivalence mode.
- `scripts/cross_engine_parity_440.py` — parametrize config-path + fix tmp-path + wire absolute gate + persist one `OsmoseResults`.
- `tests/test_engine_java_comparison.py` — repoint at the 4.3.3 snapshot (or bump to 4.4.1).
- `tests/baselines/parity_baseline_bob_1yr_seed42.npz`, `tests/baselines/statistical_baseline_bob_1yr_10seeds.npz` — regenerate post-migration.
- `CHANGELOG.md`, `docs/parity-roadmap.md` — document.

---

## Task 1: Verify BoB runs on the Python engine (A0 — HARD gate)

**Files:**
- Test: `tests/test_bob_440_smoke.py` (create; Python-baseline portion)

**Interfaces:**
- Consumes: `osmose.engine.PythonEngine.run_in_memory(raw: dict, seed: int) -> OsmoseResults`; `osmose.config.reader.OsmoseConfigReader().read(path) -> dict`.
- Produces: confidence that the rest of the plan is viable. If this fails, STOP — the plan is moot.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bob_440_smoke.py
from pathlib import Path
import numpy as np
import pytest
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parents[1]
BOB = ROOT / "data" / "examples" / "osm_all-parameters.csv"

@pytest.mark.skipif(not BOB.exists(), reason="no BoB config")
def test_bob_runs_on_python_engine():
    raw = dict(OsmoseConfigReader().read(str(BOB)))
    raw["simulation.time.nyear"] = "3"  # pin: do NOT inherit nyear;50
    res = PythonEngine().run_in_memory(raw, seed=42)
    bio = res.biomass()
    assert bio is not None and len(bio) > 0
    vals = bio[[c for c in bio.columns if c not in ("Time", "species")]].to_numpy(dtype=float)
    assert np.isfinite(vals).any() and np.nansum(vals) > 0
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bob_440_smoke.py::test_bob_runs_on_python_engine -v`
Expected: PASS if BoB already runs on Python. If it FAILS (import/load/run error), that is the real Task 0 — diagnose and fix before continuing; the whole plan depends on it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bob_440_smoke.py
git commit -m "test(bob): verify Bay-of-Biscay runs on the Python engine (A0 gate)"
```

---

## Task 2: Forcing resampler — 365 → 24 bin-average (A1)

**Files:**
- Create: `scripts/resample_bob_forcing.py`
- Create: `data/examples/ltl/roms_n2p2z2d2_biscay_24step.nc` (generated in Step 4)
- Test: `tests/test_resample_bob_forcing.py`

**Interfaces:**
- Produces: `resample_to_24_steps(ds: xr.Dataset) -> xr.Dataset` (24-step, bin-averaged, same vars/coords); `main()` writing `roms_n2p2z2d2_biscay_24step.nc`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resample_bob_forcing.py
import numpy as np
import xarray as xr
from scripts.resample_bob_forcing import resample_to_24_steps

def _synthetic_365():
    # 6 vars, 365 daily steps, 4x5 grid; each var = its own linear ramp so bin means are exact.
    data = {}
    for i, name in enumerate(["SmallPhyto", "LargePhyto", "SmallZoo", "LargeZoo",
                              "SmallDetritus", "LargeDetritus"]):
        arr = (np.arange(365)[:, None, None] + i).astype(float) * np.ones((365, 4, 5))
        data[name] = (("time", "lat", "lon"), arr)
    return xr.Dataset(data, coords={"time": np.arange(365), "lat": np.arange(4), "lon": np.arange(5)})

def test_output_is_24_steps_same_grid_and_vars():
    out = resample_to_24_steps(_synthetic_365())
    assert out.sizes["time"] == 24
    assert out.sizes["lat"] == 4 and out.sizes["lon"] == 5
    assert set(out.data_vars) == {"SmallPhyto", "LargePhyto", "SmallZoo", "LargeZoo",
                                  "SmallDetritus", "LargeDetritus"}

def test_bins_conserve_window_mean():
    ds = _synthetic_365()
    out = resample_to_24_steps(ds)
    # step s = mean over days d where floor(d*24/365)==s
    step_of_day = (np.arange(365) * 24) // 365
    for s in range(24):
        days = np.where(step_of_day == s)[0]
        expected = ds["SmallZoo"].isel(time=days).mean("time").values
        np.testing.assert_allclose(out["SmallZoo"].isel(time=s).values, expected)

def test_idempotent_all_bins_nonempty():
    out = resample_to_24_steps(_synthetic_365())
    assert np.isfinite(out["SmallPhyto"].values).all()  # no empty bins -> no NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_resample_bob_forcing.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.resample_bob_forcing`

- [ ] **Step 3: Write the implementation**

```python
# scripts/resample_bob_forcing.py
"""Bin-average BoB's 365-day resource forcing to a 24-step/year axis.

The 4.4.1 engine requires the forcing's steps/year to divide ndt=24; 365 does not.
Input day d -> output step floor(d*24/365); mean each bin (window mean conserved).
Writes roms_n2p2z2d2_biscay_24step.nc next to the original (original kept).

  PYTHONPATH=. .venv/bin/python scripts/resample_bob_forcing.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay.nc"
DST = ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay_24step.nc"
NSTEPS = 24

def resample_to_24_steps(ds: xr.Dataset) -> xr.Dataset:
    n_in = ds.sizes["time"]
    step_of_day = (np.arange(n_in) * NSTEPS) // n_in  # day -> 0..23
    groups = xr.DataArray(step_of_day, dims="time", name="step")
    out = ds.groupby(groups).mean("time").rename({"step": "time"})
    out = out.assign_coords(time=np.arange(NSTEPS))
    return out[list(ds.data_vars)]  # preserve var order

def main() -> None:
    ds = xr.open_dataset(SRC, decode_times=False)
    out = resample_to_24_steps(ds)
    for v in out.data_vars:              # carry attrs
        out[v].attrs = ds[v].attrs
    out.attrs = ds.attrs
    out.to_netcdf(DST)
    print(f"wrote {DST} (time={out.sizes['time']}, vars={list(out.data_vars)})")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + generate the real artifact**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_resample_bob_forcing.py -v` → Expected: PASS
Run: `PYTHONPATH=. .venv/bin/python scripts/resample_bob_forcing.py`
Expected: `wrote .../roms_n2p2z2d2_biscay_24step.nc (time=24, vars=[...6 vars...])`
Verify: `PYTHONPATH=. .venv/bin/python -c "import xarray;print(xarray.open_dataset('data/examples/ltl/roms_n2p2z2d2_biscay_24step.nc').sizes)"` → `time: 24, lat: 20, lon: 30`

- [ ] **Step 5: Commit**

```bash
git add scripts/resample_bob_forcing.py tests/test_resample_bob_forcing.py data/examples/ltl/roms_n2p2z2d2_biscay_24step.nc
git commit -m "feat(bob): bin-average 365-day resource forcing to 24-step axis"
```

---

## Task 3: Snapshot the pre-migration config (A0b)

**Files:**
- Create: `data/examples_433_orig/` (copy of `data/examples`)

**Interfaces:**
- Produces: `data/examples_433_orig/osm_all-parameters.csv` — the original 4.3.3 ltl config, source for Task 6's intermediate and Task 9's OLD arm, and the config rollback.

- [ ] **Step 1: Copy the tree (before any in-place mutation)**

Run: `cp -r data/examples data/examples_433_orig`
Verify: `grep -c '^osmose.version;4.3.3' data/examples_433_orig/osm_all-parameters.csv` → `1`; `grep -c 'ltl.name.rsc' data/examples_433_orig/osm_param-ltl.csv` → `6`

- [ ] **Step 2: Commit**

```bash
git add data/examples_433_orig
git commit -m "chore(bob): snapshot pre-migration 4.3.3 config (parity baseline + rollback)"
```

---

## Task 4: BoB fully-native conversion (A2)

**Files:**
- Modify: `scripts/migrate_bundled_to_440.py` (IN_SCOPE line 30; add `_convert_bob_native`; call it in `convert_config`)
- Test: `tests/test_migrate_bob_native.py`

**Interfaces:**
- Consumes: `_collect_param_files(master) -> list[Path]` (existing), `_SEP_RE` (existing).
- Produces: after running `convert_config(data/examples)`, the config has `osmose.version;4.4.1`, `species.trophic.level.sp8-13` (no `species.tl.sp8-13`), `species.file.sp8-13 = ltl/roms_n2p2z2d2_biscay_24step.nc`, and ZERO `ltl.*` keys.

- [ ] **Step 1: Write the failing test** (operate on a temp copy so the test is repeatable)

```python
# tests/test_migrate_bob_native.py
import shutil
from pathlib import Path
import pytest
from scripts.migrate_bundled_to_440 import convert_config, _collect_param_files

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "examples_433_orig"  # migrate a copy of the ORIGINAL

def _all_keys(master: Path) -> dict[str, str]:
    keys = {}
    for f in _collect_param_files(master):
        for ln in f.read_text().splitlines():
            s = ln.strip()
            if s and not s.startswith("#") and (";" in s):
                k, v = s.split(";", 1)
                keys[k.strip()] = v.strip()
    return keys

@pytest.mark.skipif(not SRC.exists(), reason="need Task 3 snapshot")
def test_bob_converts_to_fully_native(tmp_path):
    cfg = tmp_path / "examples"
    shutil.copytree(SRC, cfg)
    convert_config(cfg)
    keys = _all_keys(cfg / "osm_all-parameters.csv")
    assert keys["osmose.version"] == "4.4.1"
    # species.tl renamed to species.trophic.level (engine species.type path reads the latter)
    assert "species.tl.sp10" not in keys
    assert keys["species.trophic.level.sp10"] == "2.0"
    assert keys["species.trophic.level.sp11"] == "2.5"
    # per-species forcing path added, pointing at the 24-step file
    assert keys["species.file.sp8"] == "ltl/roms_n2p2z2d2_biscay_24step.nc"
    assert keys["species.file.sp13"] == "ltl/roms_n2p2z2d2_biscay_24step.nc"
    # ALL ltl.* keys dropped (a single leftover ltl.name.rscN re-routes the Python engine)
    assert not any(k.startswith("ltl.") for k in keys)
    # species.biomass.* NOT baked on disk (emitted at Java-stage time)
    assert not any(k.startswith("species.biomass.") for k in keys)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_migrate_bob_native.py -v`
Expected: FAIL — `convert_config` raises `SystemExit("examples not in scope ...")`.

- [ ] **Step 3: Implement** — add `examples` to IN_SCOPE and the name-gated BoB pass.

In `scripts/migrate_bundled_to_440.py`, change line 30:

```python
IN_SCOPE = {"eec_full", "minimal", "baltic", "baltic_ev", "examples"}
```

Add this function (after `_convert_line`):

```python
_BOB_RESOURCE_SP = range(8, 14)  # sp8..sp13 resources
_FORCING_24 = "ltl/roms_n2p2z2d2_biscay_24step.nc"

def _convert_bob_native(config_dir: Path) -> None:
    """BoB-specific fully-native fixups (run AFTER the generic per-line conversion).

    (a) rename species.tl.spN -> species.trophic.level.spN (Python species.type path reads
        species.trophic.level; BoB carries species.tl); (b) add per-species species.file.spN ->
        the 24-step forcing (drives both the Python species.type forcing read AND the Java-stage
        species.biomass.file emit); (c) drop every ltl.* key across all param files (a single
        leftover ltl.name.rscN re-routes the Python engine back onto _load_config_ltl).
    """
    master = next(iter(config_dir.glob("*all-parameters*.csv")))
    for f in _collect_param_files(master):
        out = []
        for ln in f.read_text().splitlines(keepends=True):
            s = ln.strip()
            if s and not s.startswith("#"):
                m = _SEP_RE.search(ln)
                if m:
                    key = ln[: m.start()].strip().lower()
                    if key.startswith("ltl."):
                        continue  # drop the whole ltl.* family
                    if key.startswith("species.tl.sp"):
                        idx = key.rsplit("sp", 1)[1]
                        out.append(f"species.trophic.level.sp{idx}{m.group(0)}{ln[m.end():]}")
                        continue
            out.append(ln)
        f.write_text("".join(out))
    # append the per-species forcing paths to the master (idempotent: skip if present)
    text = master.read_text()
    existing = {ln.split(_SEP_RE.search(ln).group(0))[0].strip().lower()
                for ln in text.splitlines() if _SEP_RE.search(ln) and not ln.strip().startswith("#")}
    add = [f"species.file.sp{i} ; {_FORCING_24}\n"
           for i in _BOB_RESOURCE_SP if f"species.file.sp{i}" not in existing]
    if add:
        if not text.endswith("\n"):
            text += "\n"
        master.write_text(text + "# Osmose 4.4.1 - per-species resource forcing (24-step)\n" + "".join(add))
```

Then in `convert_config`, after the per-file conversion loop (after line 131 `f.write_text(...)`), before the `print`:

```python
    if name == "examples":
        _convert_bob_native(config_dir)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_migrate_bob_native.py -v` → Expected: PASS
Run guard tests for the other configs still pass: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_migrate_bundled_440.py -v` (if present) → Expected: PASS (name-gate must not affect eec/minimal/baltic).

- [ ] **Step 5: Convert the real `data/examples` in place + verify it still runs on Python**

Run: `PYTHONPATH=. .venv/bin/python scripts/migrate_bundled_to_440.py data/examples`
Expected: `examples: converted N param file(s) -> native 4.4.0`
Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bob_440_smoke.py::test_bob_runs_on_python_engine -v` (now reads the 24-step file via species.type path) → Expected: PASS
Verify no ltl keys remain: `grep -rc 'ltl\.' data/examples/*.csv | grep -v ':0'` → only comment lines, zero active `ltl.` keys.

- [ ] **Step 6: Commit**

```bash
git add scripts/migrate_bundled_to_440.py tests/test_migrate_bob_native.py data/examples
git commit -m "feat(bob): fully-native 4.4.1 conversion (drop ltl.*, add species.file, rename tl->trophic.level)"
```

---

## Task 5: 4.4.1 jar smoke gate (A4.1)

**Files:**
- Test: `tests/test_bob_440_smoke.py` (add the jar-smoke test)

**Interfaces:**
- Consumes: the migrated `data/examples` (Task 4); the 4.4.1 jar; `ui.pages.run.write_temp_config`.
- Produces: proof the migrated BoB loads AND completes ≥1 full year on the 4.4.1 jar (exercises the per-step `update()` path + BoB's legacy fishing on 4.4.1 for the first time).

- [ ] **Step 1: Write the test (CI-skipped — needs Java)**

```python
# tests/test_bob_440_smoke.py  (append)
import shutil, subprocess
JAR_441 = ROOT / "osmose-java" / "osmose-4.4.1-jar-with-dependencies.jar"
_java = shutil.which("java") is not None and JAR_441.exists()

@pytest.mark.skipif(not (_java and BOB.exists()), reason="Java/jar/config unavailable")
def test_bob_runs_on_441_jar(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import write_temp_config
    raw = dict(OsmoseConfigReader().read(str(BOB)))
    stage = tmp_path / "stage"
    write_temp_config(raw, stage, source_dir=BOB.parent, target_version="4.4.1")
    master = stage / "osm_all-parameters.csv"
    odir = tmp_path / "out"; odir.mkdir()
    r = subprocess.run(
        ["java", "-Xmx2g", "-jar", str(JAR_441), str(master),
         f"-Poutput.dir.path={odir}", "-Psimulation.time.nyear=3", "-Poutput.start.year=0"],
        capture_output=True, text=True, timeout=900)
    # A load/parameter error unrelated to resource forcing (e.g. legacy fishing on 4.4.1) => RESCOPE.
    assert r.returncode == 0, f"4.4.1 jar failed on BoB:\n{r.stderr[-2000:]}"
    assert list(odir.glob("*.csv")), "no outputs produced"
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bob_440_smoke.py::test_bob_runs_on_441_jar -v`
Expected: PASS. If it FAILS on a fishing/other key (not resource forcing), STOP per the rescope trigger and report — do not paper over it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bob_440_smoke.py
git commit -m "test(bob): 4.4.1 jar smoke gate for migrated Bay-of-Biscay"
```

---

## Task 6: Load-path-equivalence gate (A4.2)

**Files:**
- Modify: `scripts/native_440_parity.py` (add a `bob-loadpath` mode)
- Test: `tests/test_bob_loadpath_equiv.py`

**Interfaces:**
- Consumes: `run_outputs(config_dir, years, seed)` and `max_rel_diff(a, b)` (existing in native_440_parity).
- Produces: `bob_loadpath_equiv(years=3, seed=42) -> float` (max abs diff across biomass/abundance/yield between the ltl-24 intermediate and the native-24 config); 0.0 == bit-exact.

Rationale: isolates the *key conversion* from the *forcing resample*. Intermediate = the ORIGINAL ltl config (Task 3 snapshot) with `ltl.netcdf.file` repointed to the 24-step file (Python reads via `_load_config_ltl`); compare bit-exact to the native config (Python reads via `_load_config_species_type`). Both use identical forcing, so any diff is a key-conversion defect. Preconditions (verified): BoB larval rates all 0.0; no `species.lmax`/`species.beta`; no `species.multiplier`/`offset`/`accessibility2fish.file` — so bit-exact is valid.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bob_loadpath_equiv.py
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "examples_433_orig").exists()
    or not (ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay_24step.nc").exists(),
    reason="need Task 3 snapshot + Task 2 24-step file")

def test_ltl_and_native_load_paths_are_bit_exact():
    from scripts.native_440_parity import bob_loadpath_equiv
    assert bob_loadpath_equiv(years=3, seed=42) == 0.0  # np.array_equal via max abs diff
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bob_loadpath_equiv.py -v`
Expected: FAIL — `ImportError: cannot import name 'bob_loadpath_equiv'`

- [ ] **Step 3: Implement** — add to `scripts/native_440_parity.py`:

```python
import shutil, tempfile

def _build_ltl_24_intermediate(dst: Path) -> Path:
    """Copy the ORIGINAL ltl config (examples_433_orig) and repoint ltl.netcdf.file at the
    24-step file, so the Python _load_config_ltl path reads the same forcing the native config does."""
    src = ROOT / "data" / "examples_433_orig"
    shutil.copytree(src, dst)
    ltl = dst / "osm_param-ltl.csv"
    lines = []
    for ln in ltl.read_text().splitlines(keepends=True):
        if ln.strip().lower().startswith("ltl.netcdf.file"):
            lines.append("ltl.netcdf.file ; ltl/roms_n2p2z2d2_biscay_24step.nc\n")
        else:
            lines.append(ln)
    ltl.write_text("".join(lines))
    # bring the 24-step NetCDF into the intermediate's ltl/ dir
    shutil.copy(ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay_24step.nc",
                dst / "ltl" / "roms_n2p2z2d2_biscay_24step.nc")
    return dst

def bob_loadpath_equiv(years: int = 3, seed: int = 42) -> float:
    """Max abs diff (biomass/abundance/yield) between the ltl-24 intermediate and native-24 BoB.
    Both read identical 24-step forcing; a nonzero result is a key-conversion defect."""
    native = ROOT / "data" / "examples"          # already native after Task 4
    with tempfile.TemporaryDirectory() as td:
        inter = _build_ltl_24_intermediate(Path(td) / "examples")
        a = run_outputs(inter, years=years, seed=seed)
        b = run_outputs(native, years=years, seed=seed)
    worst = 0.0
    for k in set(a) & set(b):
        import numpy as np
        if a[k].shape != b[k].shape:
            return float("inf")
        worst = max(worst, float(np.nanmax(np.abs(a[k] - b[k])) if a[k].size else 0.0))
    return worst
```

Wire the subcommand in `__main__` (replace the bottom dispatch so `bob-loadpath` is accepted):

```python
if __name__ == "__main__":
    cmd = sys.argv[1]  # read cmd FIRST — bob-loadpath takes no target (avoid IndexError)
    if cmd == "bob-loadpath":
        w = bob_loadpath_equiv()
        print(f"bob load-path equiv: max abs diff = {w:.2e} {'PASS' if w == 0.0 else 'FAIL'}")
        assert w == 0.0, f"BoB key conversion NOT lossless: {w:.2e}"
    else:
        target = sys.argv[2]
        if target not in IN_SCOPE:
            raise SystemExit(f"{target} not in scope {IN_SCOPE}")
        capture_baseline(target) if cmd == "capture" else gate(target)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bob_loadpath_equiv.py -v` → Expected: PASS (== 0.0).
If nonzero: a real key-conversion defect (or an unmet precondition). Diagnose — do NOT loosen to a tolerance without confirming the precondition (larva=0 / no lmax / no multiplier) still holds.

- [ ] **Step 5: Commit**

```bash
git add scripts/native_440_parity.py tests/test_bob_loadpath_equiv.py
git commit -m "test(bob): load-path-equivalence gate proves key conversion is lossless"
```

---

## Task 7: Forcing-change characterization (A4.3 — report, not a gate)

**Files:**
- Create: `scripts/bob_forcing_characterization.py`
- Create: `docs/superpowers/notes/2026-07-06-bob-forcing-resample-characterization.md` (written from the script output)

**Interfaces:**
- Consumes: `native_440_parity.run_outputs`; the Task 3 snapshot (365-step) + native (24-step).
- Produces: a per-species report of Python-365-subsample vs Python-24-bin-average divergence, confirming it is small/seasonally faithful — documenting the intended change.

- [ ] **Step 1: Write the script**

```python
# scripts/bob_forcing_characterization.py
"""Report Python-engine divergence: 365-step (subsample) vs 24-step (bin-average) BoB.
This is the DELIBERATE forcing change (A4.3) — characterized, not gated. Run after Task 4."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scripts.native_440_parity import run_outputs

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    old = run_outputs(ROOT / "data" / "examples_433_orig", years=5, seed=42)  # 365-step
    new = run_outputs(ROOT / "data" / "examples", years=5, seed=42)            # 24-step
    print(f"{'metric':<12}{'max|rel|':>12}{'median|rel|':>14}")
    for k in sorted(set(old) & set(new)):
        a, b = old[k].ravel(), new[k].ravel()
        n = min(a.size, b.size)
        rel = np.abs(a[:n] - b[:n]) / np.maximum(np.abs(a[:n]), 1e-30)
        print(f"{k:<12}{np.nanmax(rel):>12.3f}{np.nanmedian(rel):>14.3f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it + record**

Run: `PYTHONPATH=. .venv/bin/python scripts/bob_forcing_characterization.py`
Write the table + one paragraph ("bin-average conserves window means; divergence is the intended loss of sub-window variability; both engines now read identical 24-step forcing") into `docs/superpowers/notes/2026-07-06-bob-forcing-resample-characterization.md`.

- [ ] **Step 3: Commit**

```bash
git add scripts/bob_forcing_characterization.py docs/superpowers/notes/2026-07-06-bob-forcing-resample-characterization.md
git commit -m "docs(bob): characterize 365-subsample vs 24-bin-average forcing change"
```

---

## Task 8: Parametrize the cross-engine parity harness (A4.4 prerequisite)

**Files:**
- Modify: `scripts/cross_engine_parity_440.py`

**Interfaces:**
- Produces: `--config <path>`, `--engines <csv>` (subset of `python,4.4.1,4.3.3`), `--persist-results <dir>` args; `python_rep`/`java_rep`/`ensemble` take a `config` path; the **`OsmoseResults` reader prefix is read from the config's `output.file.prefix`** (NOT the dir name); `overall_fail` gates the absolute equivalence `eq1`; the hardcoded tmp dir is replaced with `tempfile`. Runs only the selected engines (so BoB's two-config split works: native→{python,4.4.1}, snapshot→{python,4.3.3}).

- [ ] **Step 1: Parametrize config + reader-prefix + engines; update EVERY call site**

Delete `EEC = ROOT / "data" / "eec_full" / "eec_all-parameters.csv"` (line 42). Thread `config` and a **reader prefix taken from the config** through, and add an `--engines` selector. The reader prefix MUST equal `output.file.prefix` (BoB=`biscay`, EEC=`eec`) — deriving it from the dir name breaks every Java read (global constraint). Concretely:

Add a helper + change the signatures:

```python
def _read_prefix(config: Path) -> str:
    """The Java output-file prefix (OsmoseResults globs '{prefix}_{type}*.csv')."""
    raw = dict(OsmoseConfigReader().read(str(config)))
    return raw.get("output.file.prefix", config.parent.name)

def python_rep(config: Path, years: int, seed: int, spinup: int):
    raw = dict(OsmoseConfigReader().read(str(config)))
    raw["simulation.time.nyear"] = str(years)
    res = PythonEngine().run_in_memory(raw, seed=seed)
    return {m: _final_mean(_reader(res, m), years, spinup) for m in METRICS}

def java_rep(ver: str, master: Path, years: int, odir: Path, spinup: int, prefix: str):
    odir.mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-Xmx2g", "-jar", str(JARS[ver]), str(master),
           f"-Poutput.dir.path={odir}", f"-Psimulation.time.nyear={years}", "-Poutput.start.year=0"]
    if subprocess.run(cmd, capture_output=True, text=True, timeout=900).returncode != 0:
        return None
    res = OsmoseResults(odir, prefix=prefix, strict=False)  # strict=False: report-empty, don't crash
    return {m: _final_mean(_reader(res, m), years, spinup) for m in METRICS}

def ensemble(engine: str, config: Path, prefix: str, years: int, n: int, spinup: int, tmp: Path):
    reps = []
    if engine == "python":
        for s in range(n):
            reps.append(python_rep(config, years, 1000 + s, spinup))   # <-- was python_rep(years, ...)
    else:
        raw = dict(OsmoseConfigReader().read(str(config)))
        master = tmp / f"stage_{engine}"
        write_temp_config(raw, master, source_dir=config.parent, target_version=engine)
        master = master / "osm_all-parameters.csv"
        for s in range(n):
            r = java_rep(engine, master, years, tmp / f"out_{engine}_{s}", spinup, prefix)  # <-- +prefix
            if r is not None:
                reps.append(r)
    out = {}
    for m in METRICS:
        species = sorted({k for r in reps for k in r[m]})
        out[m] = {sp: np.array([r[m].get(sp, np.nan) for r in reps], dtype=float) for sp in species}
    return out
```

In `main`, replace args + the tmp + the determinism check + the ensemble calls:

```python
    ap.add_argument("--config", type=Path, default=ROOT / "data" / "eec_full" / "eec_all-parameters.csv")
    ap.add_argument("--engines", default="python,4.4.1,4.3.3", help="comma list subset")
    ap.add_argument("--persist-results", type=Path, default=None)
    args = ap.parse_args()
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="xengine_"))
    prefix = _read_prefix(args.config)
    engines = [e.strip() for e in args.engines.split(",")]

    if "python" in engines:  # determinism check (was python_rep(3,7,1) at line 162 — now needs config)
        a, b = python_rep(args.config, 3, 7, 1), python_rep(args.config, 3, 7, 1)
        det = all(np.isclose(a["biomass"][k], b["biomass"][k]) for k in a["biomass"])
        print(f"[determinism] Python same-seed reproducible: {det}")

    ens = {}   # only the selected engines (each config supports a specific set)
    for e in engines:
        ens[e] = ensemble(e, args.config, prefix, args.years, args.n, args.spinup_years, tmp)
    py = ens.get("python"); j441 = ens.get("4.4.1"); j433 = ens.get("4.3.3")
```

(This replaces lines 162-169. `py`/`j441`/`j433` may be `None` when an engine wasn't selected — Step 2's gate handles that.)

- [ ] **Step 2: Wire the absolute-equivalence gate AND make it arm-aware**

Replace the whole analysis/gate section (the `for eng in (py, j441, j433):` mean_weight block through the final GATE print, ~lines 177-219) with an arm-aware version that (a) tolerates a missing arm, (b) gates the **absolute** `eq1` for the 4.4.1 arm (not the relative test), (c) treats 4.3.3 as a reported reference. The MEF/Spearman community block (which required both java arms) is dropped — it is not the gate.

```python
    for eng in filter(None, (py, j441, j433)):
        eng["mean_weight"] = {sp: eng["biomass"][sp] / np.clip(eng["abundance"][sp], 1e-9, None)
                              for sp in eng["biomass"] if sp in eng["abundance"]}
    analysis_metrics = METRICS + ("mean_weight",)
    floors = {"mean_weight": 1e-9}

    if py is None:
        print("no python arm — nothing to gate"); return
    present = [v for v in ("4.4.1", "4.3.3") if ens.get(v) is not None]
    overall_fail = []
    for m in analysis_metrics:
        floor = floors.get(m, COLLAPSE)
        sp_all = [s for s in py[m] if all(s in ens[v][m] for v in present)]
        print(f"==================== METRIC: {m} ====================")
        for sp in sp_all:
            row = f"{sp:<22}"
            if j441 is not None:
                d1, ci1, p1, eq1, ks1, vr1 = tost(py[m][sp], j441[m][sp], args.delta, floor)
                if not eq1 or abs(d1) >= 1.0:          # PRIMARY gate: absolute equivalence + 1-OoM tripwire
                    overall_fail.append(f"{m}:{sp}")
                row += f"  441 d={d1:>6.2f} eq={'Y' if eq1 else 'n'} KS={ks1:.2f}"
            if j433 is not None:                        # reference / reported only, never gated
                d3, _, _, eq3, _, _ = tost(py[m][sp], j433[m][sp], args.delta, floor)
                row += f"  | 433 d={d3:>6.2f} eq={'Y' if eq3 else 'n'}"
            print(row)
    tag = "absolute Python<->4.4.1 equivalence + within 1 OoM" if j441 is not None else "reference run (no 4.4.1 arm — not gated)"
    print(f"GATE ({tag}): {'PASS' if not overall_fail else 'REVIEW: ' + ', '.join(overall_fail)}")
```

- [ ] **Step 3: Persist the selected Java arm(s) for Phase 3 (config-supported versions only)**

Add, after the gate loop in `main` (before it returns), a persist block that stages ONLY the selected Java versions — each config supports specific versions (native→4.4.1, snapshot→4.3.3); staging an unsupported version yields a config that jar cannot load. The persisted dir is named `{prefix}_{ver}` with `prefix = output.file.prefix` (so `biscay_4.4.1` / `biscay_4.3.3`), and the Java files inside are `{prefix}_*.csv` — Task 12 reads them with the SAME prefix.

```python
    if args.persist_results:
        args.persist_results.mkdir(parents=True, exist_ok=True)
        for ver in [e for e in engines if e != "python"]:   # only selected java arms
            raw = dict(OsmoseConfigReader().read(str(args.config)))
            st = tmp / f"persist_stage_{ver}"
            write_temp_config(raw, st, source_dir=args.config.parent, target_version=ver)
            java_rep(ver, st / "osm_all-parameters.csv", args.years,
                     args.persist_results / f"{prefix}_{ver}", args.spinup_years, prefix)
        # Python arm is in-memory; Task 12 recomputes it directly (no persist needed).
```

So the NEW run (`--engines python,4.4.1` on the native config) persists `{prefix}_4.4.1`, and the OLD run (`--engines python,4.3.3` on the snapshot) persists `{prefix}_4.3.3` — each from a config that version can actually load.

- [ ] **Step 4: Verify EEC still runs unchanged (regression)**

Run: `PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py --n 2 --years 3`
Expected: runs to a GATE line (small N just checks plumbing; the real N=16 run is a manual gate). No crash from the parametrization; `--config` defaults to EEC.

- [ ] **Step 5: Commit**

```bash
git add scripts/cross_engine_parity_440.py
git commit -m "feat(parity): parametrize harness (config/prefix/persist) + gate absolute equivalence"
```

---

## Task 9: Run BoB cross-engine parity (A4.4 — manual gate)

**Files:**
- (No new code; a runnable gate using Task 8's harness. Record the result in the Task 13 docs.)

**Interfaces:**
- Consumes: parametrized `cross_engine_parity_440.py`; the native BoB (`data/examples`) + the snapshot (`data/examples_433_orig`).

Reader prefix auto-resolves to `biscay` (both the native config and the snapshot carry `output.file.prefix ; biscay`), so no `--prefix` flag is passed. `PERSIST` = `/tmp/claude-1000/-home-razinka-osmose/f7b91731-5bf2-427b-aaab-4e339882ae8b/scratchpad/phase3_results`.

- [ ] **Step 1: NEW pair — native 24-step BoB on {Python, 4.4.1-Java}** (persists `biscay_4.4.1`)

Run: `PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py --config data/examples/osm_all-parameters.csv --engines python,4.4.1 --n 16 --years 10 --persist-results $PERSIST`
Expected: a per-metric table; **GATE PASS** = absolute equivalence (eq1) holds for all species (+ within 1 OoM). Both arms read the 24-step forcing. `--engines python,4.4.1` — do NOT run the 4.3.3 arm here (it cannot load the native config).

- [ ] **Step 2: OLD reference — original 365-step BoB on {Python, 4.3.3-Java}** (persists `biscay_4.3.3`)

Run: `PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py --config data/examples_433_orig/osm_all-parameters.csv --engines python,4.3.3 --n 16 --years 10 --persist-results $PERSIST`
Expected: the pre-migration reference (no 4.4.1 arm → prints "reference run … not gated"). `--engines python,4.3.3` — do NOT run the 4.4.1 arm here (it cannot consume the 365-step forcing — the original blocker). The load-bearing claim is Step 1's absolute equivalence, not the relative comparison (the arms use different forcing).

- [ ] **Step 3: Record** the per-species tables + verdict for Task 13. If Step 1's absolute gate FAILS for a species, diagnose (a real 4.4.1 dynamics change the Python engine doesn't mirror) before proceeding to Phase 3.

- [ ] **Step 4: Commit** (results captured in Task 13 docs; nothing to commit here unless you save the tables to `docs/superpowers/notes/`).

---

## Task 10: Existing-test remediation (task 5)

**Files:**
- Modify: `tests/baselines/parity_baseline_bob_1yr_seed42.npz`, `tests/baselines/statistical_baseline_bob_1yr_10seeds.npz` (regenerate)
- Modify: `tests/test_engine_java_comparison.py:38` (JAR_PATH / config)

**Interfaces:**
- Consumes: the migrated `data/examples`; `scripts/save_parity_baseline.py --config bob`.

- [ ] **Step 1: Audit for other readers**

Run: `grep -rln "examples\|EXAMPLES_CONFIG\|roms_n2p2z2d2\|_bob_" tests/`
Handle any file beyond the two below that reads `data/examples` / the bob baselines. Report anything unexpected.

- [ ] **Step 2: Regenerate BOTH BoB baselines (they fail by design post-migration)**

The BoB parity tests are `TestBaselineParity` in `tests/test_engine_parity.py` (they use `EXAMPLES_CONFIG` = `data/examples`); the exact-match ones are gated `@_exact_match_local_only` (skip off CPython-3.12/CI). **Do NOT use `-k bob`** — no test name contains "bob", so it selects zero tests (verified: "17 deselected, 0 selected"). Run the class node id:

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_parity.py::TestBaselineParity -v`
Expected (local CPython 3.12): FAIL on the `atol=0` biomass/abundance/mortality match — the load path + forcing both changed. Confirm the failure is the intended dynamics change (cross-check magnitudes against Task 7's characterization), THEN regenerate BOTH baselines (the parity test loads the single-seed `.npz`; the statistical baseline is also listed as modified and uses a separate `--statistical` mode with no `--config`):

Run: `PYTHONPATH=. .venv/bin/python scripts/save_parity_baseline.py --config bob --years 1 --seed 42`
Run: `PYTHONPATH=. .venv/bin/python scripts/save_parity_baseline.py --statistical --years 1 --seeds 10`
Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_parity.py::TestBaselineParity -v` → Expected: PASS against the new baseline.

- [ ] **Step 3: Fix `tests/test_engine_java_comparison.py`** — it hardcodes the 4.3.3 jar against the (now native) `data/examples`. Repoint it at the 4.3.3 snapshot for a 4.3.3-only check:

Change `EXAMPLES_CONFIG = EXAMPLES_DIR / "osm_all-parameters.csv"` region so the 4.3.3 test uses `data/examples_433_orig` (which the 4.3.3 jar can read), OR bump `JAR_PATH` to the 4.4.1 jar and stage via `write_temp_config(target_version="4.4.1")`. Prefer the snapshot repoint (keeps a real 4.3.3 regression):

```python
EXAMPLES_DIR = PROJECT_DIR / "data" / "examples_433_orig"   # 4.3.3-loadable original
EXAMPLES_CONFIG = EXAMPLES_DIR / "osm_all-parameters.csv"
```

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_java_comparison.py -v` → Expected: PASS (or skip if Java unavailable).

- [ ] **Step 4: Commit**

```bash
git add tests/baselines/parity_baseline_bob_1yr_seed42.npz tests/baselines/statistical_baseline_bob_1yr_10seeds.npz tests/test_engine_java_comparison.py
git commit -m "test(bob): regenerate post-migration baselines + repoint java-comparison at 4.3.3 snapshot"
```

---

## Task 11: Build the EEC + BoB ICES snapshot (B1)

**Files:**
- Create: `data/eec_full/reference/ices_snapshots/index.json` (+ per-stock files)
- Create: `data/examples/reference/ices_snapshots/index.json` (+ per-stock files)

**Interfaces:**
- Consumes: ICES MCP (`list_stocks`, `get_stock_assessment`, `get_reference_points`); the `index.json` schema from `osmose/validation/ices.py` (`model_species_to_ices_stocks`, `units_by_stock`, `advice_year_by_stock`) — mirror `data/baltic/reference/ices_snapshots/index.json`.

- [ ] **Step 1: Read the schema + the Baltic exemplar**

Read `osmose/validation/ices.py` (`IcesSnapshot`, `load_snapshot`) and `data/baltic/reference/ices_snapshots/index.json` to copy the exact layout.

- [ ] **Step 2: Fetch + write the cleanly-mapping tonnes-unit stocks only**

Via the ICES MCP, fetch SSB time series + reference points for: EEC sole→`sol.27.7d`, plaice→`ple.27.7d`; BoB (sp0/1/6/5) → `ane.27.8`, `pil.27.8abd`, `sol.27.8ab`, `hke.27.8c9a`. Write `index.json` for each config with `model_species_to_ices_stocks` mapping the model species name to the stock, `units_by_stock` = tonnes, `advice_year_by_stock`. Mark all other species uncovered (comment) — do NOT invent index-unit mappings.

**The `model_species_to_ices_stocks` keys MUST match the model species name EXACTLY, including case** (they are matched against the config's `species.name.spN`). BoB names are **capitalized** — use `"Anchovy"` (sp0), `"Sardine"` (sp1), `"Hake"` (sp5), `"Sole"` (sp6) (verified `data/examples/osm_param-species.csv`). EEC names are **lowercase** — `"sole"` (sp7), `"plaice"` (sp8). A case mismatch makes `compare_outputs_to_ices` silently return zero comparisons for that species.

- [ ] **Step 3: Verify the snapshot loads**

Run: `PYTHONPATH=. .venv/bin/python -c "from osmose.validation.ices import load_snapshot; print(load_snapshot('data/examples/reference/ices_snapshots'))"` → Expected: loads without error, lists the 4 BoB stocks.

- [ ] **Step 4: Commit**

```bash
git add data/eec_full/reference/ices_snapshots data/examples/reference/ices_snapshots
git commit -m "feat(ices): EEC + BoB ICES snapshots (cleanly-mapping tonnes-unit stocks only)"
```

---

## Task 12: Phase 3 cross-engine ICES consistency gate (B2)

**Files:**
- Create: `scripts/phase3_ices_consistency.py`

**Interfaces:**
- Consumes: `osmose.validation.ices.load_snapshot` + `compare_outputs_to_ices(...)`; the persisted Java `OsmoseResults` dirs from Task 9 (`phase3_results/biscay_4.4.1`, `biscay_4.3.3` — prefix is `biscay`, BoB's `output.file.prefix`); `OsmoseResults`, `PythonEngine`.
- Produces: a per-species `magnitude_factor` table across {Python, 4.3.3-Java, 4.4.1-Java}; gate = the three agree within Δ = log10(3), with a **loud non-empty assertion** so an empty comparison FAILS instead of PASSing vacuously.

`compare_outputs_to_ices` returns `list[SpeciesBiomassComparison]` (`.species`, `.magnitude_factor = model_mean / sqrt(ices_min*ices_max)`). **The review found `compare_outputs_to_ices` cannot read a real eec/BoB `OsmoseResults`:** `model_biomass_window_mean` calls `results.biomass(species=X)`, but for these configs `biomass()` is a WIDE cross-species frame (species are columns; the `species` column is the literal `"all"`), so `biomass(species=X)` returns 0 rows → `ValueError` → `compare_outputs_to_ices` silently skips EVERY species → empty factors → vacuous `GATE PASS`. Step 1 fixes the reader; Step 2 adds the non-empty assertion as a backstop.

- [ ] **Step 1: Fix `model_biomass_window_mean` to read the wide all-species frame (TDD)**

Add a failing test to `tests/test_validation_ices.py`:

```python
def test_window_mean_reads_wide_all_species_frame():
    import pandas as pd
    from osmose.validation.ices import model_biomass_window_mean
    wide = pd.DataFrame({"Time": [0,1,2,3,4,5], "Anchovy": [10,10,10,20,20,20], "species": ["all"]*6})
    class R:
        output_dir = "x"
        def biomass(self, species=None):
            return wide[wide["species"] == species] if species is not None else wide  # species!='all' -> empty
    assert model_biomass_window_mean(R(), "Anchovy", window_years=3) == 20.0
```

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validation_ices.py::test_window_mean_reads_wide_all_species_frame -v` → Expected: FAIL (`ValueError: no biomass time series for 'Anchovy'`).

Replace `model_biomass_window_mean` in `osmose/validation/ices.py` with a wide-frame fallback (keep the existing long-form path for configs that emit it, e.g. Baltic):

```python
def model_biomass_window_mean(results, species, *, window_years=5):
    df = results.biomass(species=species)
    if df is None or len(df) == 0 or "value" not in getattr(df, "columns", []):
        # Fallback: WIDE cross-species frame (species are columns; 'species' col == 'all').
        wide = results.biomass()
        if wide is not None and species in getattr(wide, "columns", []):
            s = wide.sort_values("Time")[species] if "Time" in wide.columns else wide[species]
            n_window = min(window_years, len(s))
            if n_window <= 0:
                raise ValueError(f"empty biomass window for {species!r}")
            return float(s.iloc[-n_window:].mean())
        raise ValueError(f"no biomass time series for {species!r} in {results.output_dir}")
    if "time" in df.columns:
        df = df.sort_values("time")
    n_window = min(window_years, len(df))
    if n_window <= 0:
        raise ValueError(f"empty biomass window for {species!r}")
    return float(df.iloc[-n_window:]["value"].mean())
```

Run the test → Expected: PASS. Then run the whole ICES suite for regressions: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validation_ices.py -v` → Expected: PASS.

- [ ] **Step 2: Write the Phase 3 script (prefix=`biscay`, non-empty assertion)**

```python
# scripts/phase3_ices_consistency.py
"""Phase 3: cross-engine ICES consistency (magnitude_factor, NOT in_range — uncalibrated configs).
Gate: the three engines agree within Delta = log10(3) on each mapped species' magnitude_factor.
Reuses the persisted Java OsmoseResults from Task 9; recomputes the Python arm directly."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults
from osmose.validation.ices import load_snapshot, compare_outputs_to_ices

ROOT = Path(__file__).resolve().parents[1]
DELTA = np.log10(3)

def _factors(results, snapshot) -> dict[str, float]:
    cmp = compare_outputs_to_ices(results, snapshot, window_years=5)  # -> list[SpeciesBiomassComparison]
    return {r.species: r.magnitude_factor for r in cmp if r.magnitude_factor is not None}

def run(config: Path, snap_dir: Path, persisted: Path, prefix: str, years: int = 10) -> None:
    snapshot = load_snapshot(str(snap_dir))
    raw = dict(OsmoseConfigReader().read(str(config)))
    raw["simulation.time.nyear"] = str(years)   # >= spinup + window_years (=5)
    py = _factors(PythonEngine().run_in_memory(raw, seed=1000), snapshot)
    j441 = _factors(OsmoseResults(persisted / f"{prefix}_4.4.1", prefix=prefix, strict=False), snapshot)
    j433 = _factors(OsmoseResults(persisted / f"{prefix}_4.3.3", prefix=prefix, strict=False), snapshot)
    common = sorted(set(py) & set(j441) & set(j433))
    # Fail LOUDLY if nothing mapped, rather than printing a vacuous PASS (review finding).
    assert common, (f"Phase 3 compared ZERO species — check snapshot species-name case and that "
                    f"model_biomass_window_mean read the persisted results (py={len(py)}, "
                    f"441={len(j441)}, 433={len(j433)}).")
    fails = []
    print(f"{'species':<16}{'Python':>10}{'4.3.3':>10}{'4.4.1':>10}{'agree<=D':>10}")
    for sp in common:
        vals = np.log10([py[sp], j433[sp], j441[sp]])
        agree = (vals.max() - vals.min()) <= DELTA
        if not agree:
            fails.append(sp)
        print(f"{sp:<16}{py[sp]:>10.2f}{j433[sp]:>10.2f}{j441[sp]:>10.2f}{'Y' if agree else 'N':>10}")
    print(f"\nGATE ({len(common)} species, agree within {10**DELTA:.1f}x): {'PASS' if not fails else 'REVIEW: '+', '.join(fails)}")

if __name__ == "__main__":
    persisted = Path("/tmp/claude-1000/-home-razinka-osmose/f7b91731-5bf2-427b-aaab-4e339882ae8b/scratchpad/phase3_results")
    run(ROOT / "data" / "examples" / "osm_all-parameters.csv",
        ROOT / "data" / "examples" / "reference" / "ices_snapshots", persisted, prefix="biscay")
```

- [ ] **Step 3: Run it (after Task 9 persisted the Java results)**

Run: `PYTHONPATH=. .venv/bin/python scripts/phase3_ices_consistency.py`
Expected: a per-species table + GATE line over ≥1 species (the assertion fires if zero mapped). Consistency (not realism): `in_range` is uniformly False; the gate is `magnitude_factor` agreement within Δ.

- [ ] **Step 4: Commit**

```bash
git add osmose/validation/ices.py tests/test_validation_ices.py scripts/phase3_ices_consistency.py
git commit -m "feat(ices): Phase 3 magnitude_factor consistency gate + wide-frame biomass reader fix"
```

---

## Task 13: Docs + CHANGELOG (B2 framing + wrap-up)

**Files:**
- Modify: `CHANGELOG.md`, `docs/parity-roadmap.md`

**Interfaces:**
- Consumes: the results from Tasks 5, 9, 12.

- [ ] **Step 1: CHANGELOG entry** — add under the current version: "BoB (`data/examples`) migrated to fully-native 4.4.1 (5th native bundled config): 365-day resource forcing bin-averaged to 24 steps; cross-engine parity PASS (absolute Python-24 ↔ 4.4.1-Java-24 equivalence); Phase 3 ICES consistency PASS (cross-engine `magnitude_factor` within log10(3)); 4.3.3 jar retained for rollback."

- [ ] **Step 2: `docs/parity-roadmap.md`** — record the BoB result + the honest framing: BoB migration is not dynamics-neutral (bin-average vs old subsample, characterized in the Task 7 note); Phase 3 is cross-engine consistency on uncalibrated demo configs, NOT empirical realism (the real ICES/HOLAS-3 anchor is Baltic-on-Python).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md docs/parity-roadmap.md
git commit -m "docs(bob): record 4.4.1 migration + Phase 3 ICES consistency results"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** A0→Task 1; A0b→Task 3; A1→Task 2; A2→Task 4; A4.1→Task 5; A4.2→Task 6; A4.3→Task 7; A4.4→Tasks 8-9; test-remediation→Task 10; B1→Task 11; B2→Task 12; docs→Task 13. All spec sections mapped.
- **Ordering:** Task 1 → 2 → 3 (snapshot BEFORE 4's in-place rewrite) → 4 → 5 → 6 → 7 → 8 → 9 → (10 ‖ 11) → 12 → 13. Task 8 (harness parametrization) precedes Tasks 9 and 12.
- **Rescope trigger:** Task 5 Step 2 — a non-forcing smoke failure (legacy fishing on 4.4.1) STOPS the plan for rescope.
- **Adapt-before-code flag:** Task 11 requires reading `ices.py` (`IcesSnapshot`/`load_snapshot`) + the Baltic exemplar `data/baltic/reference/ices_snapshots/index.json` for the exact snapshot layout before writing it — the ICES time series come from the MCP, so the values can't be pre-baked. (Task 12's `compare_outputs_to_ices` return type is now pinned: `list[SpeciesBiomassComparison]` with `.species`/`.magnitude_factor`.)
- **Type consistency:** the `OsmoseResults` reader prefix is BoB's `output.file.prefix` = `biscay` everywhere (auto-resolved from the config in Task 8; persisted dirs `biscay_4.4.1`/`biscay_4.3.3` in Tasks 8/9; Task 12 reads them with `prefix="biscay"`). The harness args (`--config`/`--engines`/`--persist-results`) introduced in Task 8 are used verbatim in Tasks 9 and 12.
- **Plan-review findings applied (2026-07-06, 25 confirmed via multi-agent workflow `wf_056d3985-8d7`):** (1) reader prefix must equal `output.file.prefix` not the dir name (was the top critical — Task 8/9/12); (2) `compare_outputs_to_ices` couldn't read the wide biomass → vacuous Phase-3 PASS → fixed `model_biomass_window_mean` + non-empty assertion (Task 12); (3) `--engines` selector so BoB's two-config split never runs 4.4.1 on the 365-step config or 4.3.3 on the native config (Task 8/9); (4) all `python_rep`/`java_rep` call sites updated incl. the determinism check (Task 8); (5) `bob-loadpath` `__main__` IndexError guard (Task 6); (6) `-k bob` → `TestBaselineParity` + regenerate BOTH baselines (Task 10); (7) capitalized BoB species names in `index.json` (Task 11). One finding refuted (the 4.3.3 jar CAN read the native species.file forcing — but we persist 4.3.3 from the snapshot regardless, which is more correct).
