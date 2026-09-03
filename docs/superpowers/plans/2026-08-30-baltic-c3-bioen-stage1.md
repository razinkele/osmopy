# Baltic C3 Stage 1 — Bioenergetics Parity, Temperature Forcing, Offline Fit and A/B — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Python bioenergetics path Java-parity (budget, ingestion cap, starvation, reproduction, dispatch), wire a two-layer temperature forcing, fit a 9-species bioen parameter set offline, and run one pre-registered A/B whose gates and decision rule are defined in the spec.

**Architecture:** Engine changes are all behind `config.bioen_enabled`; bioen-off arithmetic is untouched and proven bit-identical (Gate A: committed master fixture + fixed-seed EEC/BoB baselines). Bioen-on is proven against Java 4.3.3 (Gate B) before any Baltic measurement. Forcing, fit, and harness are new files following the C4/B2 harness pattern (gates first, engine runs second, committed JSON last).

**Tech Stack:** Python 3.12, NumPy, SciPy (`least_squares`, `brentq`, `cKDTree`), xarray/NetCDF, pytest; Java 4.3.3/4.4.1 jars in `osmose-java/`; `.venv/bin/python` for everything.

**Spec:** `docs/superpowers/specs/2026-08-30-baltic-c3-bioen-stage1-design.md` (read it first; §0 table = the parity contract, §4 = the gates).

## Global Constraints

- Run tests with `.venv/bin/python -m pytest`; lint with `.venv/bin/ruff check osmose/ ui/ tests/ scripts/`; format with `.venv/bin/ruff format`. Line length 100.
- **Never run two engine jobs concurrently** on this machine (ESTAS_II shares it). Long runs go through `setsid nohup … &` with a log file, never through a tool call with a timeout.
- **Bioen-off must stay bit-identical**: `tests/test_engine_parity.py` (baselines in `tests/baselines/parity_baseline_{bob,eec}_1yr_seed42.npz`, generated on master 2026-08-30) must pass after every engine task. If it fails, the task is wrong — do not regenerate baselines.
- Java reference sources: `/home/razinka/osmose-reference/osmose-master/java/src/main/java/fr/ird/osmose/` (4.3.3). When a step says "Java does X", the line reference is in the spec §0 table.
- Config keys reaching the engine are **lowercase** (`osmose/config/reader.py:168` lowercases file keys). `EngineConfig.from_dict` does NOT lowercase — tests that build dicts directly must use lowercase keys.
- Units: school `weight` is tonnes **per fish**; `abundance` is fish; `biomass = abundance*weight`; `preyed_biomass`/`e_gross`/`e_maint`/`e_net` are tonnes **per school** (after this plan); `gonad_weight` tonnes per fish; `egg_weight_override` tonnes.
- Commit after every task with the trailer:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01KSP4ExqHQmyMWf8KsfmZU1`. Never `git add -A`; the working tree carries the user's unrelated uncommitted files (`osmose/runner.py`, `osmose/cli.py`, `osmose/engine/movement_maps.py`, `.mcp.json`, `mcp_servers/copernicus/server.py`, three tests). Add files by path.
- Work on branch `c3-bioen-stage1` off master (`git checkout -b c3-bioen-stage1`).

## File structure

| File | Responsibility |
|---|---|
| `osmose/engine/processes/energy_budget.py` (modify) | per-school budget, `enet_faced`, ρ (Task 3) |
| `osmose/engine/processes/bioen_starvation.py` (modify) | add `bioen_starvation_substep` (Task 4) |
| `osmose/engine/processes/bioen_predation.py` (modify) | `per_fish_ingestion_cap` (Task 4) |
| `osmose/engine/processes/mortality.py` (modify) | dispatch gate, `_kill` survivor scaling, cap in loop, interleaved bioen starvation (Task 4) |
| `osmose/engine/simulate.py` (modify) | `_bioen_step` (Tasks 3, 4, 7), `_bioen_reproduction` (Task 5), `_load_temperature_data` (Task 7), `meanEnetFaced` collection (Task 7) |
| `osmose/engine/processes/reproduction.py` (modify) | `regulate_recruitment`, `create_egg_schools` extracted (Task 5) |
| `osmose/engine/processes/bioen_reproduction.py` (modify) | `bioen_egg_release` (Task 5) |
| `osmose/engine/config.py` (modify) | key case, larval threshold, merged `bioen_i_max` (Task 2) |
| `osmose/engine/physical_data.py` (modify) | 4-D layers (Task 7) |
| `osmose/engine/config_validation.py` (modify) | `temperature.*` honoured (Task 7) |
| `osmose/engine/output.py` (modify) | `meanEnetFaced` CSV (Task 7) |
| `osmose/calibration/bioen_offline.py` (create) | offline Java-form growth model, T_p solve, fit (Task 8) |
| `scripts/fit_baltic_bioen_params.py` (create) | CLI: emit `baltic_param-bioen.csv` + overlay JSON, or a Gate-B config (Tasks 8, 11) |
| `scripts/c3_gate_a_reference.py` (create) | produce/check the committed master fixture (Task 1) |
| `scripts/cross_engine_parity_440.py` (modify) | `.bioen` staging, rep-count assert, non-degeneracy, mean_size metric (Task 9) |
| `scripts/build_baltic_temperature_forcing.py` (create) | two-layer climatology + `bottom_depth` (Task 10) |
| `scripts/baltic_c3_bioen_ab.py` (create) | harness: gates C–F, 3 arms × 5 seeds, JSON (Task 12) |
| `data/baltic/forcing/baltic_temperature_2layer_climatology.nc` (create) | Task 10 |
| `data/baltic/scenarios/c3_bioen/{baltic_param-bioen.csv,c3_bioen_arm.json}` (create) | Task 11 |
| `docs/diagnostics/c3_gate_a_master_baseline.json` (create) | Task 1 |
| tests: `tests/test_c3_gate_a_fixture.py`, `tests/test_engine_bioen_config_keys.py`, `tests/test_engine_bioen_budget_parity.py`, `tests/test_engine_bioen_mortality_parity.py`, `tests/test_engine_bioen_reproduction_parity.py`, `tests/test_engine_temperature_loader.py`, `tests/test_bioen_offline_fit.py`, `tests/test_build_baltic_temperature_forcing.py`, `tests/test_baltic_c3_harness.py`, `tests/test_baltic_c3_bioen_smoke.py` | one per task |

---

### Task 1: Gate-A master fixture (committed) + check script

**Files:**
- Create: `scripts/c3_gate_a_reference.py`
- Create: `docs/diagnostics/c3_gate_a_master_baseline.json`
- Test: `tests/test_c3_gate_a_fixture.py`

**Interfaces:**
- Consumes: `tests/baselines/baltic_master_75e92da_50yr_5seeds.npz` (already generated on master; keys `columns`, `seed42`, `seed123`, `seed7`, `seed999`, `seed2024`, each `(50, 11)` annual `biomass()` arrays).
- Produces: `load_gate_a_fixture(path) -> dict` returning `{"engine_commit": str, "seeds": [int], "columns": [str], "series": {"42": [[...]], ...}}`; `check_against_fixture(fixture, seed, bio_df) -> list[str]` returning violating column names (empty = identical). Later tasks (12) call both.

- [ ] **Step 1: Write the failing fixture-format test**

```python
# tests/test_c3_gate_a_fixture.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.c3_gate_a_reference import check_against_fixture, load_gate_a_fixture

FIXTURE = Path(__file__).resolve().parents[1] / "docs" / "diagnostics" / "c3_gate_a_master_baseline.json"


def test_fixture_exists_and_has_five_seeds_and_eleven_columns():
    fx = load_gate_a_fixture(FIXTURE)
    assert fx["seeds"] == [42, 123, 7, 999, 2024]
    assert len(fx["columns"]) == 11 and "cod_west" in fx["columns"] and "GreySeal" in fx["columns"]
    for s in fx["seeds"]:
        arr = np.asarray(fx["series"][str(s)], dtype=float)
        assert arr.shape == (50, 11) and np.all(np.isfinite(arr))
    assert len(fx["engine_commit"]) >= 7


def test_check_against_fixture_reports_only_differing_columns():
    fx = load_gate_a_fixture(FIXTURE)
    arr = np.asarray(fx["series"]["42"], dtype=float)
    df = pd.DataFrame(arr, columns=fx["columns"])
    df.insert(0, "Time", np.arange(50, dtype=float))
    assert check_against_fixture(fx, 42, df) == []
    df2 = df.copy()
    df2["herring"] = df2["herring"] * (1 + 1e-12)
    assert check_against_fixture(fx, 42, df2) == ["herring"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_c3_gate_a_fixture.py -q`
Expected: FAIL (`ModuleNotFoundError: scripts.c3_gate_a_reference`). If `scripts/` has no `__init__.py`, import via importlib exactly as `scripts/baltic_c4_salinity_ab.py:83-100` does (`importlib.util.spec_from_file_location`) — copy that pattern into the test instead of a package import.

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python3
"""Gate A reference for C3: production Baltic (bioen OFF) biomass(), 5 seeds x 50 yr.

`--produce` runs the engine at the CURRENT commit and writes the fixture JSON (only ever run
on the untouched master engine); `--check` re-runs and asserts bit-identity against the
committed fixture; `--from-npz` converts the local npz written on 2026-08-30 (commit 75e92da).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs" / "diagnostics" / "c3_gate_a_master_baseline.json"
SEEDS = (42, 123, 7, 999, 2024)
N_YEAR = 50


def _engine_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()


def _production_config() -> dict[str, str]:
    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo

    tmp = Path(tempfile.mkdtemp(prefix="c3_gate_a_"))
    cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    cfg["simulation.time.nyear"] = str(N_YEAR)
    return cfg


def run_seed(seed: int) -> pd.DataFrame:
    from osmose.engine import PythonEngine

    warnings.simplefilter("ignore")
    return PythonEngine().run_in_memory(_production_config(), seed=seed).biomass()


def load_gate_a_fixture(path: Path = FIXTURE) -> dict:
    return json.loads(Path(path).read_text())


def check_against_fixture(fixture: dict, seed: int, bio_df: pd.DataFrame) -> list[str]:
    """Columns whose series differ (array_equal) from the fixture for this seed."""
    ref = np.asarray(fixture["series"][str(seed)], dtype=np.float64)
    bad = []
    for j, col in enumerate(fixture["columns"]):
        got = bio_df[col].to_numpy(dtype=np.float64)
        if got.shape != ref[:, j].shape or not np.array_equal(got, ref[:, j]):
            bad.append(col)
    return bad


def write_fixture(series: dict[int, np.ndarray], columns: list[str], commit: str) -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine_commit": commit,
        "n_year": N_YEAR,
        "seeds": list(SEEDS),
        "columns": list(columns),
        "series": {str(s): series[s].tolist() for s in SEEDS},
    }
    FIXTURE.write_text(json.dumps(payload))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--produce", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--from-npz", type=Path)
    a = ap.parse_args(argv)
    if a.from_npz:
        z = np.load(a.from_npz)
        cols = [str(c) for c in z["columns"]]
        commit = a.from_npz.name.split("_")[2]  # baltic_master_<commit>_50yr_5seeds.npz
        write_fixture({s: z[f"seed{s}"] for s in SEEDS}, cols, commit)
        print(f"wrote {FIXTURE} from {a.from_npz} (commit {commit})")
        return 0
    if a.produce:
        series, cols = {}, None
        for s in SEEDS:
            df = run_seed(s)
            cols = cols or [c for c in df.columns if c not in ("Time", "species")]
            series[s] = df[cols].to_numpy(dtype=np.float64)
        write_fixture(series, cols, _engine_commit())
        print(f"wrote {FIXTURE} at {_engine_commit()}")
        return 0
    fx = load_gate_a_fixture()
    bad = {s: check_against_fixture(fx, s, run_seed(s)) for s in SEEDS}
    ok = all(not v for v in bad.values())
    print(f"Gate A vs fixture {fx['engine_commit']}: {'IDENTICAL' if ok else 'DIFFERS ' + str(bad)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Produce the fixture from the npz and run the test**

Run: `PYTHONPATH=. .venv/bin/python scripts/c3_gate_a_reference.py --from-npz tests/baselines/baltic_master_75e92da_50yr_5seeds.npz`
Then: `.venv/bin/python -m pytest tests/test_c3_gate_a_fixture.py -q` → PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git checkout -b c3-bioen-stage1
git add scripts/c3_gate_a_reference.py docs/diagnostics/c3_gate_a_master_baseline.json tests/test_c3_gate_a_fixture.py
git commit -m "test(c3): commit Gate-A master fixture (production Baltic, 5 seeds x 50 yr, engine 75e92da) + check script"
```

---

### Task 2: Config — key case, larval threshold, merged `bioen_i_max`

**Files:**
- Modify: `osmose/engine/config.py:2488-2545` (bioen parse block), the `EngineConfig` dataclass fields near `:1883-1904`, and the constructor call site where `bioen_k_for=bioen_k_for` is passed (find with `grep -n "bioen_k_for=" osmose/engine/config.py`).
- Test: `tests/test_engine_bioen_config_keys.py`

**Interfaces:**
- Produces on `EngineConfig` (all `None` unless bioen on): `bioen_tp`, `bioen_e_d` now read from lowercase keys `species.bioen.mobilized.tp.sp{i}` / `species.bioen.mobilized.e.d.sp{i}`; new `bioen_larvae_thres_dt: NDArray[np.int32]` (per focal species; Java default 1) from `species.larvae.growth.threshold.age.sp{i}` (years × ndt, rounded); new `bioen_i_max_all: NDArray[np.float64]` of length `n_species + n_background` = focal `bioen_i_max` followed by each background species' `ingestion_rate` (same canonical key). Task 4 reads `bioen_i_max_all` and `bioen_larvae_thres_dt`; Task 3 reads `bioen_larvae_thres_dt`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_bioen_config_keys.py
"""C3 Task 2: bioen keys must survive the reader's lowercasing; larval threshold; merged Imax."""
from pathlib import Path

import numpy as np

from osmose.config import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from tests.test_bioen_orchestration import _make_bioen_config_dict  # 2-species synthetic dict


def _lower(d: dict[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in d.items()}


def test_tp_and_ed_are_read_from_lowercase_keys():
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    cfg["species.bioen.mobilized.tp.sp0"] = "9.5"
    cfg["species.bioen.mobilized.e.d.sp1"] = "1.25"
    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_tp[0] == 9.5 and ec.bioen_e_d[1] == 1.25


def test_reader_roundtrip_delivers_tp(tmp_path: Path):
    cfg = _make_bioen_config_dict(n_species=2)
    cfg["species.bioen.mobilized.Tp.sp0"] = "11.0"  # mixed case, as a user would write it
    p = tmp_path / "osm_all-parameters.csv"
    p.write_text("".join(f"{k} ; {v}\n" for k, v in cfg.items()))
    raw = dict(OsmoseConfigReader().read(str(p)))
    ec = EngineConfig.from_dict(raw)
    assert ec.bioen_tp[0] == 11.0, "Tp lost through the reader -> engine path"


def test_larvae_threshold_default_is_one_dt_and_key_is_years():
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    ec = EngineConfig.from_dict(cfg)
    assert list(ec.bioen_larvae_thres_dt) == [1, 1]
    cfg["species.larvae.growth.threshold.age.sp1"] = "0.5"  # years
    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_larvae_thres_dt[1] == round(0.5 * ec.n_dt_per_year)


def test_bioen_i_max_all_has_focal_then_background_entries():
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_i_max_all.shape[0] == ec.n_species + ec.n_background
    np.testing.assert_array_equal(ec.bioen_i_max_all[: ec.n_species], ec.bioen_i_max)


def test_bioen_fields_none_when_disabled():
    cfg = _lower(_make_bioen_config_dict(n_species=2))
    cfg["module.bioenergetics.enabled"] = "false"
    ec = EngineConfig.from_dict(cfg)
    assert ec.bioen_larvae_thres_dt is None and ec.bioen_i_max_all is None
```

Note: `_make_bioen_config_dict` (`tests/test_bioen_orchestration.py:~60-80`) uses legacy `species.bioen.maturity.*` and `predation.ingestion.rate.max.bioen.*` keys, which `from_dict` does not canonicalize; that is why the test lowercases and sets the canonical keys it asserts on directly. Task 6 rewrites that helper to canonical keys.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_config_keys.py -q` → 4 FAIL (`bioen_tp` stays 20.0; `AttributeError: bioen_larvae_thres_dt`).

- [ ] **Step 3: Implement**

In `osmose/engine/config.py` bioen parse block:

```python
            bioen_e_d = _species_float_optional(cfg, "species.bioen.mobilized.e.d.sp{i}", n_sp, 1.5)
            bioen_tp = _species_float_optional(cfg, "species.bioen.mobilized.tp.sp{i}", n_sp, 20.0)
            # Java Species.java:205-216: larvae->adult threshold in YEARS, default 1 dt under bioen.
            _larv_yrs = _species_float_optional(
                cfg, "species.larvae.growth.threshold.age.sp{i}", n_sp, -1.0
            )
            bioen_larvae_thres_dt = np.where(
                _larv_yrs < 0, 1, np.rint(_larv_yrs * n_dt).astype(np.int32)
            ).astype(np.int32)
            # Java BioenPredationMortality.init reads Imax for every predator index (focal AND
            # background); the canonical key is shared with the standard rate (lossy alias).
            bioen_i_max_all = np.concatenate(
                [bioen_i_max, np.array([b.ingestion_rate for b in background_list], dtype=np.float64)]
            )
```

Initialise `bioen_larvae_thres_dt = bioen_i_max_all = None` beside the other `= None` lines before the `if _bioen_enabled:` block; add the two dataclass fields next to `bioen_k_for` (`:1899`):

```python
    bioen_larvae_thres_dt: NDArray[np.int32] | None = None  # Java larvaeThresDt (default 1)
    bioen_i_max_all: NDArray[np.float64] | None = None  # Imax for focal + background predators
```

and pass `bioen_larvae_thres_dt=bioen_larvae_thres_dt, bioen_i_max_all=bioen_i_max_all` at the constructor call beside `bioen_k_for=bioen_k_for`. `background_list` is in scope in `from_dict` (it is passed to `_parse_predation_params`, `config.py:~2223`); if the parse block runs before `background_list` exists, move the `bioen_i_max_all` line to just after `background_list` is built. Also update the schema entries' `key_pattern` in `osmose/schema/bioenergetics.py` for `species.bioen.mobilized.Tp.sp{idx}` and `.e.D.` to lowercase, and add `species.larvae.growth.threshold.age.sp{idx}` to `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py` only if `tests/test_engine_config_validation.py` complains (the AST walker should capture it).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_config_keys.py tests/test_engine_bioen_integration.py tests/test_engine_config_validation.py tests/test_schema_all.py -q` → all PASS. If `test_engine_bioen_integration.py::test_bioen_thermal_params_parsed` asserts on the old capitalised keys, change that test's dict keys to lowercase (it builds a dict directly).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py osmose/schema/bioenergetics.py tests/test_engine_bioen_config_keys.py tests/test_engine_bioen_integration.py
git commit -m "fix(engine): bioen Tp/e.D read from lowercase keys; larval threshold key; merged Imax for background predators"
```

---

### Task 3: Energy budget in Java's per-school framework

**Files:**
- Modify: `osmose/engine/processes/energy_budget.py` (replace `compute_energy_budget`, replace `update_e_net_avg` with `update_enet_faced`)
- Modify: `osmose/engine/simulate.py:_bioen_step` step 3 and step 5 (the call and the `e_net_avg` update)
- Test: `tests/test_engine_bioen_budget_parity.py`

**Interfaces:**
- Produces:
  ```python
  def update_enet_faced(enet_faced_prev, e_net, abundance, weight, age_dt, first_feeding_age_dt,
                        larvae_thres_dt: int, larval_coef: float, beta: float, n_dt_per_year: int) -> NDArray
  def compute_energy_budget(ingestion, weight, abundance, gonad_weight, age_dt, length, temp_c,
                            assimilation, c_m, beta, eta, r, m0, m1, e_maint_energy, phi_t, f_o2,
                            n_dt_per_year, enet_faced) -> (dw_t_per_fish, dg_t_per_fish, e_net, e_gross, e_maint, rho)
  ```
  `enet_faced` passed in is the value ALREADY updated with this step's `e_net` (Java order: `computeEnetFaced` then `getRho`); the caller computes `e_net` first via `energy_terms(...)`:
  ```python
  def energy_terms(ingestion, weight, abundance, temp_c, assimilation, c_m, beta, e_maint_energy,
                   phi_t, f_o2, n_dt_per_year) -> (e_gross, e_maint, e_net)   # all tonnes/school
  ```
- `state.e_net_avg` keeps its name but now holds Java's `enet_faced` (docstring in `state.py` updated).

- [ ] **Step 1: Write the failing Gate-G budget tests (hand-computed from Java)**

```python
# tests/test_engine_bioen_budget_parity.py
"""Gate G (spec §4): energy budget transcribed from EnergyBudget.java, hand-computed."""
import numpy as np
import pytest

from osmose.engine.processes.energy_budget import (
    compute_energy_budget,
    energy_terms,
    update_enet_faced,
)
from osmose.engine.processes.temp_function import arrhenius

K = dict(assimilation=0.7, c_m=1.0e12, beta=0.8, eta=1.0, r=0.5, m0=30.0, m1=0.0,
         e_maint_energy=0.65, n_dt_per_year=24)


def _three_schools():
    # school 0: N=1e3, school 1: N=1e6 (same per-fish weight), school 2: immature small fish
    weight = np.array([1e-3, 1e-3, 1e-5])            # t/fish (1 kg, 1 kg, 10 g)
    abundance = np.array([1e3, 1e6, 1e6])
    ingestion = np.array([0.05, 50.0, 0.1])           # t/school this step (1e3x apart for 0 vs 1)
    length = np.array([50.0, 50.0, 10.0])
    age_dt = np.array([120, 120, 12], dtype=np.int32)
    gonad = np.zeros(3)
    return weight, abundance, ingestion, length, age_dt, gonad


def test_maintenance_is_per_school_tonnes():
    weight, abundance, ingestion, *_ = _three_schools()
    T = 10.0
    e_gross, e_maint, e_net = energy_terms(ingestion, weight, abundance, T, K["assimilation"],
                                            K["c_m"], K["beta"], K["e_maint_energy"], 1.0, 1.0, 24)
    w_g = weight * 1e6
    expected = K["c_m"] * w_g ** K["beta"] * arrhenius(np.array(T), 0.65) / 24 * abundance * 1e-6
    np.testing.assert_allclose(e_maint, expected, rtol=1e-12)
    # 1e3x abundance at equal per-fish weight -> 1e3x maintenance (per-school framework)
    assert e_maint[1] / e_maint[0] == pytest.approx(1e3, rel=1e-12)
    np.testing.assert_allclose(e_net, ingestion * 0.7 - e_maint, rtol=1e-12)


def test_dw_is_per_fish_and_independent_of_abundance_at_equal_intake_per_fish():
    weight, abundance, ingestion, length, age_dt, gonad = _three_schools()
    T = 10.0
    e_gross, e_maint, e_net = energy_terms(ingestion, weight, abundance, T, 0.7, K["c_m"], 0.8, 0.65, 1.0, 1.0, 24)
    faced = update_enet_faced(np.zeros(3), e_net, abundance, weight, age_dt,
                              np.ones(3, dtype=np.int32), larvae_thres_dt=1, larval_coef=1.0, beta=0.8, n_dt_per_year=24)
    dw, dg, e_net2, e_gross2, e_maint2, rho = compute_energy_budget(
        ingestion, weight, abundance, gonad, age_dt, length, T, 0.7, K["c_m"], 0.8, 1.0, 0.5, 30.0, 0.0,
        0.65, 1.0, 1.0, 24, faced)
    # schools 0 and 1 have identical per-fish intake (0.05/1e3 == 50/1e6) -> identical dw, dg
    assert dw[0] == pytest.approx(dw[1], rel=1e-12) and dg[0] == pytest.approx(dg[1], rel=1e-12)
    np.testing.assert_allclose(dw, (1 - rho) * np.maximum(e_net, 0) / abundance, rtol=1e-12)
    np.testing.assert_allclose(dg, rho * np.maximum(e_net, 0) / abundance, rtol=1e-12)
    assert rho[2] == 0.0  # immature (10 cm < m0=30)


def test_enet_faced_matches_java_computeEnetFaced():
    weight = np.array([1e-3, 1e-3, 1e-3]); abundance = np.array([1e3, 1e3, 1e3])
    e_net = np.array([0.05, 0.05, 0.05])
    age_dt = np.array([1, 3, 50], dtype=np.int32); ff = np.ones(3, dtype=np.int32)
    prev = np.array([9.0, 9.0, 9.0])
    faced = update_enet_faced(prev, e_net, abundance, weight, age_dt, ff, larvae_thres_dt=5,
                              larval_coef=2.0, beta=0.8, n_dt_per_year=24)
    per_fish = 0.05 * 24 / 1e3 * 1e6 / (1e-3 * 1e6) ** 0.8
    assert faced[0] == pytest.approx(per_fish / 2.0)                       # ageDt == firstFeeding: divided by coef, no averaging
    assert faced[1] == pytest.approx((per_fish / 2.0 + 9.0 * 3) / 4)       # larval: /coef, weighted by ageDt
    assert faced[2] == pytest.approx((per_fish + 9.0 * 50) / 51)           # adult: no coef
    # pre-feeding stays at previous value (Java: output = 0 only when never fed; keep prev)
    faced0 = update_enet_faced(prev, e_net, abundance, weight, np.array([0], dtype=np.int32)[:1].repeat(3), ff,
                               5, 2.0, 0.8, 24)
    np.testing.assert_array_equal(faced0, np.zeros(3))


def test_rho_guard_matches_java_clamp_semantics():
    weight, abundance, ingestion, length, age_dt, gonad = _three_schools()
    T = 10.0
    e_net = np.array([1.0, 1.0, 1.0])
    faced = np.array([0.0, -1.0, 1e-9])  # zero -> +inf -> 1 ; negative -> 0 ; tiny positive -> clamp 1
    with np.errstate(divide="ignore", invalid="ignore"):
        *_, rho = compute_energy_budget(ingestion, weight, abundance, gonad, age_dt, length, T, 0.7,
                                        K["c_m"], 0.8, 1.0, 0.5, 30.0, 0.0, 0.65, 1.0, 1.0, 24, faced)
    assert rho[0] == 1.0 and rho[1] == 0.0
    assert rho[2] == 0.0  # school 2 immature regardless


def test_zero_abundance_school_gets_no_increment():
    weight = np.array([1e-3]); abundance = np.array([0.0]); ingestion = np.array([0.0])
    dw, dg, *_ = compute_energy_budget(ingestion, weight, abundance, np.zeros(1), np.array([120], dtype=np.int32),
                                       np.array([50.0]), 10.0, 0.7, 0.0, 0.8, 1.0, 0.5, 30.0, 0.0, 0.65, 1.0, 1.0, 24,
                                       np.array([1.0]))
    assert dw[0] == 0.0 and dg[0] == 0.0 and np.isfinite(dw[0])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_budget_parity.py -q` → ImportError (`energy_terms`, `update_enet_faced`).

- [ ] **Step 3: Implement `energy_budget.py`**

Replace the two functions with:

```python
def energy_terms(ingestion, weight, abundance, temp_c, assimilation, c_m, beta, e_maint_energy,
                 phi_t, f_o2, n_dt_per_year):
    """Java EnergyBudget.getEgross/getMaintenance: all three in TONNES PER SCHOOL.

    ingestion: tonnes/school this step (survivor-scaled by mortality); weight: t/fish;
    abundance: fish (post-mortality instantaneous abundance).
    """
    e_gross = ingestion * assimilation * phi_t * f_o2
    w_grams = weight * 1e6
    e_maint = (
        c_m * np.power(w_grams, beta) * arrhenius(np.asarray(temp_c), e_maint_energy) / n_dt_per_year
    ) * abundance * 1e-6  # per fish (g) -> per school (t): Java `output *= N * 1e-6`
    return e_gross, e_maint, e_gross - e_maint


def update_enet_faced(enet_faced_prev, e_net, abundance, weight, age_dt, first_feeding_age_dt,
                      larvae_thres_dt, larval_coef, beta, n_dt_per_year):
    """Java EnergyBudget.computeEnetFaced: per-fish, per-g^beta, annualized running mean."""
    w_b = np.power(weight * 1e6, beta)
    safe_n = np.where(abundance > 0, abundance, 1.0)
    per_fish = e_net * n_dt_per_year / safe_n * 1e6 / w_b
    age = age_dt.astype(np.float64)
    first = age_dt == first_feeding_age_dt
    larval = (age_dt > first_feeding_age_dt) & (age_dt < larvae_thres_dt)
    adult = age_dt >= np.maximum(larvae_thres_dt, first_feeding_age_dt + 1)
    out = np.where(age_dt < first_feeding_age_dt, 0.0, enet_faced_prev)
    out = np.where(first, per_fish / larval_coef, out)
    out = np.where(larval, (per_fish / larval_coef + enet_faced_prev * age) / (age + 1.0), out)
    out = np.where(adult & ~first, (per_fish + enet_faced_prev * age) / (age + 1.0), out)
    return out


def compute_energy_budget(ingestion, weight, abundance, gonad_weight, age_dt, length, temp_c,
                          assimilation, c_m, beta, eta, r, m0, m1, e_maint_energy, phi_t, f_o2,
                          n_dt_per_year, enet_faced):
    """Java EnergyBudget.run order: E_gross, E_maint, E_net, (enet_faced already updated), rho, dw, dg.

    Returns dw/dg in tonnes PER FISH (Java divides by instantaneous abundance); e_* per school.
    """
    e_gross, e_maint, e_net = energy_terms(ingestion, weight, abundance, temp_c, assimilation, c_m,
                                           beta, e_maint_energy, phi_t, f_o2, n_dt_per_year)
    w_grams = weight * 1e6
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    is_mature = length >= (m0 + m1 * age_years)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_raw = r / (eta * enet_faced) * np.power(w_grams, 1.0 - beta)  # Java: unguarded division
    rho_raw = np.where(np.isnan(rho_raw), 1.0, rho_raw)  # 0/0 only when r==0: Java 0*inf=NaN -> treat as 1 then clip below
    rho = np.where(is_mature, np.clip(rho_raw, 0.0, 1.0), 0.0)
    e_pos = np.maximum(e_net, 0.0)
    alive = abundance > 0
    safe_n = np.where(alive, abundance, 1.0)
    dw = np.where(alive, (1.0 - rho) * e_pos / safe_n, 0.0)
    dg = np.where(alive, rho * e_pos / safe_n, 0.0)
    return dw, dg, e_net, e_gross, e_maint, rho
```

Note on `r == 0`: Java computes `0/(eta*faced)` = 0 for finite faced and `0*inf` = NaN when faced == 0 — then `rho<0 ? 0 : rho` and `rho>1 ? 1 : rho` both compare false for NaN, so Java's rho would be NaN. That is a Java bug we do not replicate; with `r=0` the immature path is what matters and NaN→1 keeps dw finite (recorded in the docstring).

- [ ] **Step 4: Wire `_bioen_step`**

In `simulate.py:_bioen_step` step 3, replace the per-species `compute_energy_budget` call with (keep the `_resolve` overrides):

```python
    for sp, mask in sp_masks:
        e_gross_sp, e_maint_sp, e_net_sp = energy_terms(
            capped_ingestion[mask], state.weight[mask], state.abundance[mask], temp_c_arr[mask],
            float(config.bioen_assimilation[sp]), float(config.bioen_c_m[sp]), float(config.bioen_beta[sp]),
            float(config.bioen_e_maint[sp]), phi_t_arr[mask], f_o2_arr[mask], config.n_dt_per_year)
        faced_sp = update_enet_faced(
            state.e_net_avg[mask], e_net_sp, state.abundance[mask], state.weight[mask], state.age_dt[mask],
            state.first_feeding_age_dt[mask], int(config.bioen_larvae_thres_dt[sp]),
            float(config.bioen_theta[sp]), float(config.bioen_beta[sp]), config.n_dt_per_year)
        dw_sp, dg_sp, en_sp, eg_sp, em_sp, rho_sp = compute_energy_budget(
            ingestion=capped_ingestion[mask], weight=state.weight[mask], abundance=state.abundance[mask],
            gonad_weight=state.gonad_weight[mask], age_dt=state.age_dt[mask], length=state.length[mask],
            temp_c=temp_c_arr[mask], assimilation=float(config.bioen_assimilation[sp]),
            c_m=float(config.bioen_c_m[sp]), beta=float(config.bioen_beta[sp]), eta=float(config.bioen_eta[sp]),
            r=cast(float, _resolve("bioen_r", sp, mask)), m0=cast(float, _resolve("bioen_m0", sp, mask)),
            m1=cast(float, _resolve("bioen_m1", sp, mask)), e_maint_energy=float(config.bioen_e_maint[sp]),
            phi_t=phi_t_arr[mask], f_o2=f_o2_arr[mask], n_dt_per_year=config.n_dt_per_year, enet_faced=faced_sp)
        new_e_net_avg[mask] = faced_sp
        dw_tonnes[mask] = dw_sp; dg_tonnes[mask] = dg_sp; e_net_arr[mask] = en_sp
        e_gross_arr[mask] = eg_sp; e_maint_arr[mask] = em_sp; rho_arr[mask] = rho_sp
```

Declare `new_e_net_avg = state.e_net_avg.copy()` before the loop and delete the old step-5 `update_e_net_avg` loop (keep `e_net_avg=new_e_net_avg` in the final `state.replace`). Keep `capped_ingestion = state.preyed_biomass.copy()` for now (Task 4 removes the cap). Replace the imports (`update_e_net_avg` → `update_enet_faced`, add `energy_terms`). Add `"bioen_larvae_thres_dt"` to `_BIOEN_REQUIRED`.

- [ ] **Step 5: Run the new tests and the parity gate**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_budget_parity.py tests/test_engine_parity.py -q` → PASS. (Other bioen suites may now fail — Task 6 reconciles them; do not fix them here except `tests/test_engine_energy_budget.py`, whose expectations you rewrite to the per-school formulas in this task.)

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/energy_budget.py osmose/engine/simulate.py osmose/engine/state.py tests/test_engine_bioen_budget_parity.py tests/test_engine_energy_budget.py
git commit -m "fix(engine): bioen energy budget in Java's per-school tonnes framework; enet_faced; rho clamp semantics"
```

---

### Task 4: Mortality under bioen — dispatch, per-fish cap, survivor scaling, interleaved starvation

**Files:**
- Modify: `osmose/engine/processes/bioen_predation.py` (add `per_fish_ingestion_cap`)
- Modify: `osmose/engine/processes/bioen_starvation.py` (add `bioen_starvation_substep`)
- Modify: `osmose/engine/processes/mortality.py`: `_get_mortality_causes:58-75`, `_apply_starvation_for_school:81-127`, `_apply_additional_for_school:129-178`, `_apply_fishing_for_school:180-280`, `_apply_foraging_for_school:282-345`, `_apply_predation_for_school:347-570` (the `max_eatable` line ~390 and the prey-death line ~528), `_mortality_in_cell` signature + Python fallback `:1599-1797`, `mortality()` dispatch `:1970` and return `:2078-2085`
- Modify: `osmose/engine/simulate.py:_bioen_step` (remove Step 1 cap and Step 4 starvation; mask `is_out`)
- Test: `tests/test_engine_bioen_mortality_parity.py`

**Interfaces:**
- Produces:
  ```python
  # bioen_predation.py
  def per_fish_ingestion_cap(weight, species_id, age_dt, i_max_all, beta, larvae_thres_dt, theta, c_rate,
                             n_species, n_dt_per_year, n_subdt) -> NDArray   # tonnes per fish per sub-step
  # bioen_starvation.py
  def bioen_starvation_substep(e_net, gonad_weight, weight, eta, n_subdt) -> tuple[float, float, float]
      # (n_dead, new_gonad, new_e_net) for ONE school, ONE sub-step; Java computeStarvation
  # mortality.py
  def _kill(state, idx, cause, n_dead, inst_abd, bioen: bool) -> None   # records death + survivor scaling
  ```
- `mortality()` under bioen never enters the batched Numba kernels; `_mortality_in_cell` gains `cap_fish: NDArray | None` and the Python fallback passes it to `_apply_predation_for_school(..., cap_fish=cap_fish)`.
- `_bioen_step` no longer caps ingestion nor applies starvation; `state.preyed_biomass` arriving from `mortality()` is already Java's survivor-scaled `ingestion`.

- [ ] **Step 1: Write the failing Gate-G mortality tests**

```python
# tests/test_engine_bioen_mortality_parity.py
"""Gate G (spec §4): ingestion cap form, survivor scaling, interleaved starvation, dispatch."""
import numpy as np
import pytest

from osmose.engine.processes import mortality as M
from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap
from osmose.engine.processes.bioen_starvation import bioen_starvation_substep


def test_per_fish_cap_matches_java_bioen_predation_mortality():
    # Java: Imax_eff = (Imax + (coef-1)*c_rate)/ndt ; cap_school = Imax_eff * (w*1e6)^beta / subdt * N * 1e-6
    weight = np.array([1e-3, 1e-3]); species = np.array([0, 0], dtype=np.int32)
    age = np.array([10, 0], dtype=np.int32)  # second is larval (age < thres=1? no: 0 < 1 -> larval)
    cap = per_fish_ingestion_cap(weight, species, age, i_max_all=np.array([3.5]), beta=np.array([0.8]),
                                 larvae_thres_dt=np.array([1], dtype=np.int32), theta=np.array([2.0]),
                                 c_rate=np.array([1.0]), n_species=1, n_dt_per_year=24, n_subdt=10)
    adult = (3.5 / 24) * (1e-3 * 1e6) ** 0.8 / 10 * 1e-6
    larval = ((3.5 + (2.0 - 1.0) * 1.0) / 24) * (1e-3 * 1e6) ** 0.8 / 10 * 1e-6
    assert cap[0] == pytest.approx(adult, rel=1e-12) and cap[1] == pytest.approx(larval, rel=1e-12)


def test_background_predator_uses_its_own_entry_with_default_beta():
    weight = np.array([1e-2]); species = np.array([3], dtype=np.int32); age = np.array([10], dtype=np.int32)
    cap = per_fish_ingestion_cap(weight, species, age, i_max_all=np.array([1.0, 1.0, 1.0, 9.0]),
                                 beta=np.array([0.8, 0.8, 0.8]), larvae_thres_dt=np.ones(3, dtype=np.int32),
                                 theta=np.ones(3), c_rate=np.zeros(3), n_species=3, n_dt_per_year=24, n_subdt=1)
    assert cap[0] == pytest.approx((9.0 / 24) * (1e4) ** 0.8 * 1e-6, rel=1e-12)


def test_kill_scales_ingestion_and_enet_by_survivor_fraction_only_under_bioen():
    from osmose.engine.state import SchoolState
    st = SchoolState.create(1)
    st = st.replace(preyed_biomass=np.array([4.0]), e_net=np.array([-2.0]))
    inst = np.array([100.0])
    M._kill(st, 0, M._PREDATION, 50.0, inst, bioen=True)
    assert inst[0] == 50.0 and st.n_dead[0, M._PREDATION] == 50.0
    assert st.preyed_biomass[0] == 2.0 and st.e_net[0] == -1.0
    st2 = SchoolState.create(1).replace(preyed_biomass=np.array([4.0]), e_net=np.array([-2.0]))
    inst2 = np.array([100.0])
    M._kill(st2, 0, M._PREDATION, 50.0, inst2, bioen=False)
    assert st2.preyed_biomass[0] == 4.0 and st2.e_net[0] == -2.0


def test_starvation_substep_matches_java_branches():
    # sufficient gonad: pays eta*deficit, repays e_net, no deaths
    n_dead, gonad, e_net = bioen_starvation_substep(e_net=-10.0, gonad_weight=5.0, weight=1e-3, eta=1.0, n_subdt=2)
    assert n_dead == 0.0 and gonad == 0.0 and e_net == -5.0   # deficit per subdt = 5; gonad 5 >= 5
    # insufficient: gonad flushed, zero repayment (Java flush-then-credit ordering), deaths = deficit/weight
    n_dead, gonad, e_net = bioen_starvation_substep(e_net=-10.0, gonad_weight=1.0, weight=1e-3, eta=1.0, n_subdt=2)
    assert gonad == 0.0 and e_net == -10.0 and n_dead == pytest.approx(5.0 / 1e-3)
    # positive e_net: nothing happens
    assert bioen_starvation_substep(1.0, 3.0, 1e-3, 1.0, 2) == (0.0, 3.0, 1.0)


def test_causes_include_starvation_and_foraging_under_bioen():
    from types import SimpleNamespace
    assert M._get_mortality_causes(SimpleNamespace(bioen_enabled=True)) == [
        M._PREDATION, M._STARVATION, M._ADDITIONAL, M._FISHING, M._FORAGING]
    assert M._get_mortality_causes(SimpleNamespace(bioen_enabled=False)) == [
        M._PREDATION, M._STARVATION, M._ADDITIONAL, M._FISHING]


def test_mortality_never_enters_batched_numba_under_bioen(monkeypatch):
    """The batched kernels must not be reached when bioen is on (spec §0 'Numba dispatch' row)."""
    import tests.test_bioen_orchestration as orch
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.resources import ResourceState

    cfg = {k.lower(): v for k, v in orch._make_bioen_config_dict(n_species=2).items()}
    cfg["temperature.value"] = "10.0"
    config = EngineConfig.from_dict(cfg)
    grid = Grid.from_dimensions(ny=5, nx=5)
    state = orch._make_school_state(n_schools=20, n_species=2)
    resources = ResourceState(config=cfg, grid=grid, oxygen=None)
    calls = {"batched": 0}

    def _boom(*a, **k):
        calls["batched"] += 1
        raise AssertionError("batched Numba kernel reached under bioen")

    monkeypatch.setattr(M, "_HAS_NUMBA", True, raising=False)
    monkeypatch.setattr(M, "_mortality_all_cells_numba", _boom, raising=False)
    monkeypatch.setattr(M, "_mortality_all_cells_parallel", _boom, raising=False)
    M.mortality(state, resources, config, np.random.default_rng(0), grid, step=3)
    assert calls["batched"] == 0
```

If `ResourceState(config=..., grid=..., oxygen=None)` does not match the constructor used in `simulate.py:1561`, copy that call exactly; if the synthetic config has no resources, pass `resources=None` (the fallback path accepts `None`, see `_apply_predation_for_school`'s `if resources is not None`).

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_mortality_parity.py -q` → ImportError / AttributeError.

- [ ] **Step 3: Implement the two pure helpers**

`bioen_predation.py` (keep the existing `bioen_ingestion_cap` for now; Task 6 deletes it if unused):

```python
def per_fish_ingestion_cap(weight, species_id, age_dt, i_max_all, beta, larvae_thres_dt, theta,
                           c_rate, n_species, n_dt_per_year, n_subdt):
    """Java BioenPredationMortality.getMaxPredationRate x (w*1e6)^beta / subdt, per FISH, in tonnes.

    max_eatable for a school = cap[p] * instantaneous_abundance[p] (Java multiplies by
    getInstantaneousAbundance() at every predator visit). Background predators (species_id >=
    n_species) take i_max_all[species_id] with beta 0.8, coef 1, c_rate 0 (Java reads their own keys;
    the overlay supplies Imax in bioen units under the shared canonical key).
    """
    sp = np.asarray(species_id)
    is_focal = sp < n_species
    sp_f = np.where(is_focal, sp, 0)
    b = np.where(is_focal, beta[sp_f], 0.8)
    imax = i_max_all[sp]
    larval = is_focal & (np.asarray(age_dt) < larvae_thres_dt[sp_f])
    i_eff = np.where(larval, imax + (theta[sp_f] - 1.0) * c_rate[sp_f], imax) / n_dt_per_year
    return i_eff * np.power(weight * 1e6, b) / n_subdt * 1e-6
```

`bioen_starvation.py`:

```python
def bioen_starvation_substep(e_net: float, gonad_weight: float, weight: float, eta: float, n_subdt: int):
    """Java BioenStarvationMortality.computeStarvation for one school and one sub-step.

    Replicates Java's ordering quirks on purpose (spec §0): the deficit is the SCHOOL's E_net
    (tonnes/school) divided by subdt while the gonad is per FISH; in the insufficient branch the gonad
    is flushed BEFORE the credit is computed, so the credit is zero. Returns (n_dead, gonad, e_net).
    """
    if e_net >= 0.0:
        return 0.0, gonad_weight, e_net
    deficit = abs(e_net) / n_subdt
    if gonad_weight >= eta * deficit:
        return 0.0, gonad_weight - eta * deficit, e_net + deficit
    if weight <= 0.0:
        return 0.0, 0.0, e_net
    return deficit / weight, 0.0, e_net
```

- [ ] **Step 4: Implement the mortality changes**

(a) `_get_mortality_causes`: bioen → `[_PREDATION, _STARVATION, _ADDITIONAL, _FISHING, _FORAGING]`; rewrite the docstring: "Bioen starvation runs INSIDE the interleaved loop with the previous step's e_net (Java step order mortality → EnergyBudget → reproduction)".

(b) Add after `_get_mortality_causes`:

```python
def _kill(state, idx: int, cause: int, n_dead: float, inst_abd, bioen: bool) -> None:
    """Record `n_dead` deaths of one school for `cause` and, under bioen, scale the school's
    accumulated ingestion and stored E_net by the survivor fraction (Java School.setNdead)."""
    if n_dead <= 0.0:
        return
    before = inst_abd[idx]
    state.n_dead[idx, cause] += n_dead
    inst_abd[idx] = before - n_dead
    if bioen and before > 0.0:
        f = max(inst_abd[idx], 0.0) / before
        state.preyed_biomass[idx] *= f
        state.e_net[idx] *= f
```

Replace every `state.n_dead[idx, X] += n_dead; inst_abd[idx] -= n_dead` pair in `_apply_additional_for_school`, `_apply_fishing_for_school` (record the discard split first: `state.n_dead[idx, _DISCARDS] += n_dead * discard_r` then `_kill(state, idx, _FISHING, n_dead * (1 - discard_r), inst_abd, bioen)` and, to keep the abundance decrement exact for the bioen-off path, do NOT route the discard share through `_kill` — instead keep the original two lines verbatim when `not config.bioen_enabled` and use `_kill` only when `config.bioen_enabled`), `_apply_foraging_for_school`, and the prey-death site in `_apply_predation_for_school` (`_kill(state, q_idx, _PREDATION, n_dead_prey, inst_abd, config.bioen_enabled)`). **Bit-identity rule:** for `bioen=False` the numbers written must be exactly what the old two lines wrote (same operations, same order) — `_kill` above does that; the guard `n_dead <= 0` returns early where the old code would add 0.0 (harmless: `x += 0.0` is exact).

(c) `_apply_starvation_for_school` bioen branch:

```python
    if config.bioen_enabled:
        if state.age_dt[idx] <= state.first_feeding_age_dt[idx]:
            return  # Java isStarvationEnabledBioen: ageDt > firstFeedingAgeDt (strict)
        from osmose.engine.processes.bioen_starvation import bioen_starvation_substep
        sp_i = state.species_id[idx]
        eta = float(config.bioen_eta[sp_i]) if config.bioen_eta is not None else 1.0
        n_dead, new_gonad, new_enet = bioen_starvation_substep(
            float(state.e_net[idx]), float(state.gonad_weight[idx]), float(state.weight[idx]), eta, n_subdt)
        state.gonad_weight[idx] = new_gonad
        state.e_net[idx] = new_enet
        _kill(state, idx, _STARVATION, n_dead, inst_abd, bioen=True)
```

(the existing early return `if state.age_dt[idx] < state.first_feeding_age_dt[idx]: return` stays for the standard branch).

(d) Cap in the loop: `_apply_predation_for_school` gains `cap_fish: NDArray | None = None`; replace the `max_eatable` line with

```python
    if cap_fish is None:
        max_eatable = biomass_p * config.ingestion_rate[sp_pred] / (config.n_dt_per_year * n_subdt)
    else:
        max_eatable = cap_fish[p_idx] * inst_abd_p
```

`_mortality_in_cell` gains `cap_fish=None` and passes it through in the Python fallback call. In `mortality()`, after `work_state` is built and before the sub-step loop:

```python
    cap_fish = None
    if config.bioen_enabled:
        from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap
        cap_fish = per_fish_ingestion_cap(
            work_state.weight, work_state.species_id, work_state.age_dt, config.bioen_i_max_all,
            config.bioen_beta, config.bioen_larvae_thres_dt, config.bioen_theta, config.bioen_c_rate,
            config.n_species, config.n_dt_per_year, n_subdt)
```

and pass `cap_fish=cap_fish` to `_mortality_in_cell` in the Python fallback branch.

(e) Dispatch: change `if _HAS_NUMBA and len(valid_indices) > 0:` (`:1970`) to `if _HAS_NUMBA and len(valid_indices) > 0 and not config.bioen_enabled:`. Add `gonad_weight=work_state.gonad_weight, e_net=work_state.e_net` to the final `state.replace(...)` in `mortality()` so the in-loop starvation edits are explicit (they are the same array objects, but say so).

(f) `_bioen_step`: delete Step 1 (`capped_ingestion` becomes `state.preyed_biomass` directly), delete Step 4 and the `starvation_dead` / `new_n_dead` / `new_abundance` code (abundance and `n_dead` pass through unchanged; `new_gonad = np.maximum(state.gonad_weight + dg_tonnes, 0.0)`); build `sp_masks` with `(state.species_id == sp) & ~state.is_out` so out-of-domain schools get no budget this step (spec decision 18).

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_mortality_parity.py tests/test_engine_bioen_budget_parity.py tests/test_engine_parity.py tests/test_engine_mortality_causes.py -q` → PASS. `tests/test_engine_bioen_starvation_rate_suppressed.py` and `tests/test_engine_bioen_activation.py` will fail on the old cause set — update their expectations in this task (bioen causes now include STARVATION; the Numba `eff_starv[:] = 0` line stays and its test still passes).

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/bioen_predation.py osmose/engine/processes/bioen_starvation.py osmose/engine/processes/mortality.py osmose/engine/simulate.py tests/test_engine_bioen_mortality_parity.py tests/test_engine_bioen_starvation_rate_suppressed.py tests/test_engine_bioen_activation.py
git commit -m "fix(engine): bioen mortality parity -- per-fish Imax cap in the loop, survivor scaling, interleaved starvation, no batched Numba under bioen"
```

---

### Task 4b: Bioen-aware Numba mortality kernel — SUPERSEDED

Superseded 2026-08-31 by `docs/superpowers/plans/2026-08-31-bioen-numba-kernel.md` — see that
document for the current plan (rev. 2, post adversarial review); do not execute this section.

---

### Task 5: Reproduction — regulation helper extraction + Java-parity bioen reproduction

**Files:**
- Modify: `osmose/engine/processes/reproduction.py` (extract `regulate_recruitment` and `create_egg_schools` from `reproduction():118-247`; `reproduction()` calls them)
- Modify: `osmose/engine/processes/bioen_reproduction.py` (replace `bioen_egg_production` with `bioen_egg_release`)
- Modify: `osmose/engine/simulate.py:_bioen_reproduction:574-760` (rewrite body)
- Test: `tests/test_engine_bioen_reproduction_parity.py`

**Interfaces:**
- Produces in `reproduction.py`:
  ```python
  def regulate_recruitment(n_eggs_linear: NDArray, ssb: NDArray, seeded_this_step: NDArray[np.bool_],
                           config: EngineConfig, step: int) -> NDArray      # SR curve + RV/ceiling/thermal/depensation gates
  def create_egg_schools(n_eggs: NDArray, seeded_this_step: NDArray[np.bool_], config: EngineConfig,
                         egg_length: NDArray | None = None) -> list[SchoolState]   # n_schools[sp] unlocated eggs
  ```
  `egg_length=None` → `config.egg_size` (standard). Both are pure extractions: `reproduction()` must remain bit-identical (Gate A).
- Produces in `bioen_reproduction.py`:
  ```python
  def bioen_egg_release(gonad_weight, abundance, is_mature, season: float, sex_ratio: float,
                        egg_weight_t: float) -> tuple[NDArray, NDArray]   # (n_eggs_per_school, w_egg_per_fish)
  ```
  Java: `wEgg = gonad*season` (mature only), `nEgg = wEgg*sexRatio/eggWeight*N` (weights both in tonnes here, so no 1e6).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_bioen_reproduction_parity.py
"""Gate G (spec §4): Java BioenReproductionProcess egg count, gonad decrement, nSchool schools; the
regulation helper leaves the standard path bit-identical."""
import numpy as np
import pytest

from osmose.engine.processes.bioen_reproduction import bioen_egg_release
from osmose.engine.processes.reproduction import create_egg_schools, regulate_recruitment


def test_egg_release_matches_java_formula():
    gonad = np.array([2e-6, 2e-6, 0.0]); N = np.array([1e6, 1e3, 1e6]); mature = np.array([True, True, True])
    n_eggs, w_egg = bioen_egg_release(gonad, N, mature, season=0.25, sex_ratio=0.5, egg_weight_t=1e-9)
    # wEgg = 2e-6 * 0.25 = 5e-7 t/fish ; nEgg = 5e-7 * 0.5 / 1e-9 * N
    np.testing.assert_allclose(w_egg, [5e-7, 5e-7, 0.0])
    np.testing.assert_allclose(n_eggs, [5e-7 * 0.5 / 1e-9 * 1e6, 5e-7 * 0.5 / 1e-9 * 1e3, 0.0])
    assert n_eggs[0] / n_eggs[1] == pytest.approx(1e3)  # scales with abundance (v1 bug: it did not)
    n2, w2 = bioen_egg_release(gonad, N, np.array([False, True, True]), 0.25, 0.5, 1e-9)
    assert n2[0] == 0.0 and w2[0] == 0.0  # immature: nothing released


def test_regulate_recruitment_is_identity_when_no_regulation_configured():
    from tests.test_bioen_orchestration import _make_bioen_config_dict
    from osmose.engine.config import EngineConfig
    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    config = EngineConfig.from_dict(cfg)
    lin = np.array([100.0, 5.0]); ssb = np.array([10.0, 0.0]); seeded = np.array([False, True])
    out = regulate_recruitment(lin, ssb, seeded, config, step=3)
    # recruitment_type defaults to 'none' in the synthetic config -> identity (if the default is a
    # curve, set cfg['species.recruitment.type.sp0'] = 'none' etc. and re-run)
    np.testing.assert_array_equal(out, lin)


def test_create_egg_schools_makes_n_schools_unlocated_with_given_length():
    from tests.test_bioen_orchestration import _make_bioen_config_dict
    from osmose.engine.config import EngineConfig
    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    config = EngineConfig.from_dict(cfg)
    n_eggs = np.array([1e6, 0.0]); seeded = np.array([False, False])
    schools = create_egg_schools(n_eggs, seeded, config, egg_length=np.array([0.49, 0.49]))
    assert len(schools) == 1
    s = schools[0]
    assert len(s) == int(config.n_schools[0]) and np.all(s.cell_x == -1) and np.all(s.is_egg)
    assert s.abundance.sum() == pytest.approx(1e6) and np.all(s.length == 0.49)


def test_standard_reproduction_bit_identical_after_extraction():
    """Gate A at unit level: run the Baltic-free EEC demo 1 yr before/after is covered by
    tests/test_engine_parity.py; here: reproduction() output equals the pre-extraction reference
    saved in tests/baselines/reproduction_reference_seed3.npz (create it ONCE on master with
    scripts/save_parity_baseline.py-style code BEFORE editing reproduction.py; see Step 2)."""
    import numpy as np
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "tests" / "baselines"
    ref = base / "reproduction_reference_seed3.npz"
    meta = base / "reproduction_reference_seed3.json"   # config dict + state field names (no pickle)
    if not ref.exists() or not meta.exists():
        pytest.skip("reference not generated on master")
    import json
    z = np.load(ref)  # plain arrays only; never allow_pickle (security policy)
    m = json.loads(meta.read_text())
    from osmose.engine.config import EngineConfig
    from osmose.engine.processes.reproduction import reproduction
    from osmose.engine.state import SchoolState
    config = EngineConfig.from_dict(m["config"])
    st = SchoolState(**{k: z[f"state_{k}"] for k in m["state_fields"]})
    out = reproduction(st, config, int(m["step"]), np.random.default_rng(3), grid_ny=10, grid_nx=10)
    for k in ("abundance", "weight", "length", "age_dt", "species_id"):
        np.testing.assert_array_equal(getattr(out, k), z[f"out_{k}"])
```

- [ ] **Step 2: Generate the reproduction reference on the CURRENT tree before touching reproduction.py**

Write and run once (throwaway, in the scratchpad) — it builds the 2-species synthetic config from `tests/test_engine_bioen_integration.py`'s `minimal_config` fixture dict (bioen OFF, with `species.recruitment.type.sp{i}=shepherd`, `species.recruitment.ssbhalf.sp{i}=50` set so the SR branch is exercised), a 30-school state from `tests/test_bioen_orchestration._make_school_state` with `length`/`age_dt` above the maturity thresholds for half the schools, calls `reproduction(state, config, step=5, rng(3))`, and saves every state field (`state_<name>`) plus the five `out_*` arrays with plain `np.savez` (no pickle; `None`-valued optional fields such as `imax_trait` are omitted) to `tests/baselines/reproduction_reference_seed3.npz`, and `{"config": cfg, "state_fields": [...], "step": 5}` as JSON to `tests/baselines/reproduction_reference_seed3.json` (both gitignored like the other npz — add `tests/baselines/*.json` to `.gitignore` if not covered). Run the new test file: the reference test PASSES (identity), the other three FAIL (ImportError).

- [ ] **Step 3: Extract the helpers**

In `reproduction.py`, cut lines `118-190` (from `n_eggs = apply_stock_recruitment(...)` through the depensation block) into:

```python
def regulate_recruitment(n_eggs_linear, ssb, seeded_this_step, config, step):
    """Stock-recruitment curve + RV / ceiling / thermal / depensation gates (verbatim extraction)."""
    n_sp = config.n_species
    n_eggs = apply_stock_recruitment(
        n_eggs_linear, ssb, config.recruitment_ssb_half[:n_sp], config.recruitment_type[:n_sp],
        config.shepherd_beta[:n_sp])
    if config.seeding_mode == "linear" and seeded_this_step.any():
        n_eggs = np.where(seeded_this_step, n_eggs_linear, n_eggs)
    # ... the four gate blocks, moved verbatim ...
    return n_eggs
```

and lines `192-247` (the `for sp in range(n_sp): ... new_schools_list.append(new)` loop) into `create_egg_schools(n_eggs, seeded_this_step, config, egg_length=None)` returning the list, with `egg_len = config.egg_size[sp] if egg_length is None else float(egg_length[sp])` and everything else verbatim (including the `egg_weight` derivation from `egg_len` — **note:** under `egg_length is not None` the egg WEIGHT must still come from `config.egg_size` / the override, not from the bioen length; compute `egg_weight` from `config.egg_size[sp]` explicitly). `reproduction()` then reads `n_eggs = regulate_recruitment(...)`, `new_schools_list = create_egg_schools(n_eggs, seeded_this_step, config)`. Run `tests/test_engine_parity.py` + the reference test: both PASS (bit-identity).

- [ ] **Step 4: Implement `bioen_egg_release` and rewrite `_bioen_reproduction`**

```python
# bioen_reproduction.py
def bioen_egg_release(gonad_weight, abundance, is_mature, season, sex_ratio, egg_weight_t):
    """Java BioenReproductionProcess.run: wEgg = gonad*season per fish; nEgg = wEgg*sexRatio/eggWeight*N."""
    w_egg = np.where(is_mature, gonad_weight * season, 0.0)
    safe_ew = max(float(egg_weight_t), 1e-20)
    n_eggs = w_egg * sex_ratio / safe_ew * abundance
    return n_eggs, w_egg
```

`_bioen_reproduction` body (replace everything between the docstring and the age-increment block):

```python
    from osmose.engine.processes.bioen_reproduction import bioen_egg_release
    from osmose.engine.processes.reproduction import create_egg_schools, regulate_recruitment

    n_sp = config.n_species
    age_years = state.age_dt.astype(np.float64) / config.n_dt_per_year
    gonad = state.gonad_weight.copy()
    ssb = np.zeros(n_sp); n_lin = np.zeros(n_sp); seeded = np.zeros(n_sp, dtype=np.bool_)
    egg_len = np.zeros(n_sp)
    if config.spawning_season is not None:
        season_all = config.spawning_season[:, step % config.spawning_season.shape[1]]
    else:
        season_all = np.full(n_sp, 1.0 / config.n_dt_per_year)
    for sp in range(n_sp):
        mask = (state.species_id == sp) & (state.abundance > 0) & ~state.is_egg
        m0 = _resolve_trait("bioen_m0", sp, mask, config, trait_overrides)  # same helper semantics as _bioen_step's _resolve
        m1 = _resolve_trait("bioen_m1", sp, mask, config, trait_overrides)
        mature = np.zeros(len(state), dtype=np.bool_)
        mature[mask] = state.length[mask] >= (m0 + m1 * age_years[mask])
        ssb[sp] = float((state.abundance[mature] * state.weight[mature]).sum())
        ew = np.nan if config.egg_weight_override is None else config.egg_weight_override[sp]
        if np.isnan(ew):
            ew = config.condition_factor[sp] * config.egg_size[sp] ** config.allometric_power[sp] * 1e-6
        # Java Species.java:327: under bioen eggs are created at computeLength(eggWeight)
        egg_len[sp] = (ew * 1e6 / max(config.condition_factor[sp], 1e-20)) ** (1.0 / config.allometric_power[sp])
        season = float(season_all[sp])
        if ssb[sp] == 0.0 and step < config.seeding_max_step[sp] and config.seeding_biomass[sp] > 0:
            seeded[sp] = True
            ssb[sp] = config.seeding_biomass[sp]
            n_lin[sp] = config.sex_ratio[sp] * config.relative_fecundity[sp] * ssb[sp] * season * 1_000_000.0
        else:
            n_eggs_sch, w_egg = bioen_egg_release(gonad, state.abundance, mature, season,
                                                  float(config.sex_ratio[sp]), float(ew))
            n_lin[sp] = float(n_eggs_sch[mature].sum())
            gonad = gonad - w_egg  # Java incrementGonadWeight(-wEgg); only the released share
    n_eggs = regulate_recruitment(n_lin, ssb, seeded, config, step)
    new_egg_schools = create_egg_schools(n_eggs, seeded, config, egg_length=egg_len)
    state = state.replace(gonad_weight=gonad)
    for egg_school in new_egg_schools:
        state = state.append(egg_school)
```

`_resolve_trait` is a 4-line module-level helper: `return trait_overrides[name][mask] if trait_overrides and name in trait_overrides else float(getattr(config, name)[sp])` — when it returns an array, the maturity comparison broadcasts elementwise (same as today). Keep the age-increment / `is_egg` block that follows unchanged. Delete the old seeding block, `bioen_egg_production` call, single-school creation, and the `rng` cell draw (eggs are unlocated; `rng` stays in the signature for API compatibility). `n_new = 1` for the seeding case keeps `create_egg_schools`'s "fewer eggs than schools" rule.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_bioen_reproduction_parity.py tests/test_engine_parity.py tests/test_engine_bioen_reproduction.py tests/test_engine_bioen_reproduction_wiring.py -q`. The last two encode the OLD behaviour (single located school; `bioen_egg_production`): rewrite their assertions to the new contract in this task (n_schools unlocated schools; eggs ∝ abundance; gonad decremented by `gonad*season`, not zeroed; seeding only when mature SSB == 0).

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/reproduction.py osmose/engine/processes/bioen_reproduction.py osmose/engine/simulate.py tests/test_engine_bioen_reproduction_parity.py tests/test_engine_bioen_reproduction.py tests/test_engine_bioen_reproduction_wiring.py
git commit -m "fix(engine): bioen reproduction parity -- Java egg release, nSchool unlocated eggs, shared recruitment regulation, egg length from egg weight"
```

---

### Task 6: Reconcile legacy bioen suites; full Gate A check

**Files:**
- Modify: `tests/test_bioen_orchestration.py` (`_make_bioen_config_dict` → canonical lowercase keys: `species.maturity.{eta,r,m0,m1}.sp{i}`, `predation.ingestion.rate.max.sp{i}`, `predation.larval.ingestion.rate.increase.ratio.sp{i}`, `species.bioen.mobilized.tp.sp{i}`, `species.bioen.mobilized.e.d.sp{i}`; fixtures set `temperature.value`), `tests/test_engine_bioen_integration.py`, `tests/test_engine_bioen_outputs_complete.py` (`compute_energy_budget` now returns 6 values with the new signature), `tests/test_engine_bioen_predation.py` (drop tests of the deleted post-hoc cap; test `per_fish_ingestion_cap` instead), `tests/test_engine_bioen_starvation.py` (keep the old `bioen_starvation` tests only if the function is still imported anywhere — `grep -rn bioen_starvation( osmose/` — otherwise delete the function and the file's tests for it), `tests/test_genetics_bioen_integration.py`, `tests/test_baltic_ev_fixture_bioen.py` (the `preflight` integration tests stay marked/skipped).
- Delete: `bioen_ingestion_cap` from `bioen_predation.py` and `update_e_net_avg` from `energy_budget.py` once no caller remains.

- [ ] **Step 1: Run the whole bioen-related suite and list failures**

Run: `.venv/bin/python -m pytest tests/ -q -k "bioen or energy_budget or genetics or foraging" -x --no-header -p no:cacheprovider 2>&1 | tail -30`. For each failure: if it asserts the OLD (non-parity) behaviour, rewrite the assertion to the spec §0 contract; if it reveals a real defect in Tasks 3–5, fix the engine and add a regression case to the matching `*_parity.py` file. No test may be deleted without a replacement that covers the same code path under the new contract.

- [ ] **Step 2: Full test suite + lint**

Run: `.venv/bin/python -m pytest -q -x` (expect ≥ 4448 passed, 0 failed; skips allowed) and `.venv/bin/ruff check osmose/ tests/ scripts/ && .venv/bin/ruff format --check osmose/ tests/ scripts/`.

- [ ] **Step 3: Gate A at the Baltic level**

Run (≈ 17 min, sequential, foreground is fine but use a 20-min budget or `setsid nohup`): `PYTHONPATH=. .venv/bin/python scripts/c3_gate_a_reference.py --check` → prints `Gate A vs fixture 75e92da: IDENTICAL`. If it prints DIFFERS, an engine change leaked into the bioen-off path — bisect Tasks 3–5 (`git stash`/`git checkout` per file) before continuing. Record the line in the commit message.

- [ ] **Step 4: Commit**

```bash
git add tests/ osmose/engine/processes/bioen_predation.py osmose/engine/processes/energy_budget.py
git commit -m "test(bioen): reconcile legacy bioen suites with the Java-parity contract; Gate A IDENTICAL (75e92da fixture)"
```

---

### Task 7: Temperature loader, 4-D layers, `zlayer`, gridded `f_o2`, allowlist, `meanEnetFaced`

**Files:**
- Modify: `osmose/engine/physical_data.py` (4-D support), `osmose/engine/simulate.py` (`_load_temperature_data` module-level beside `_load_oxygen_data:33-80`; the loader call at `:1600-1605`; `_bioen_step` steps 2 (φT), the O₂ block `:405-421`, the temperature block `:424-441`; `_collect_bioen:1096-1142`; `StepOutput:165-169`; `_average_step_outputs:1350-1420`; the final `StepOutput(...)` near `:1498-1508`), `osmose/engine/output.py:774` (bioen outputs list), `osmose/engine/config_validation.py:94-100` and `:204-208`, `tests/test_issue_123_known_but_unread_keys.py:176-185`, `osmose/schema/bioenergetics.py:6` (comment)
- Test: `tests/test_engine_temperature_loader.py`

**Interfaces:**
- Produces:
  ```python
  # physical_data.py
  PhysicalData.from_netcdf(path, varname, nsteps_year, factor, offset)   # accepts (t,y,x) or (t,z,y,x)
  PhysicalData.n_layers -> int                                            # 1 for 3-D / constant
  PhysicalData.get_grid(step, layer=0) -> (ny, nx); PhysicalData.get_value(step, y, x, layer=0)
  # simulate.py
  def _load_temperature_data(raw_config: dict, config_dir: Path | None) -> PhysicalData | None
  # StepOutput.bioen_enet_faced_by_species: NDArray | None ; CSV "meanEnetFaced" (always written under bioen)
  ```
- `_bioen_step(..., debug_capture: dict | None = None)`: when a dict is passed, it stores `{"temp_c": temp_c_arr, "species_id": state.species_id, "is_out": state.is_out}` (Gate E hook, Task 12).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_temperature_loader.py
import numpy as np
import pytest
import xarray as xr

from osmose.engine.physical_data import PhysicalData
from osmose.engine.simulate import _load_temperature_data


def _write(tmp_path, arr, name="temperature", dims=None):
    dims = dims or (["time", "layer", "latitude", "longitude"] if arr.ndim == 4 else ["time", "latitude", "longitude"])
    p = tmp_path / "t.nc"
    xr.Dataset({name: (dims, arr.astype(np.float32))}).to_netcdf(p)
    return p


def test_from_netcdf_4d_layers_and_3d_compat(tmp_path):
    a4 = np.arange(24 * 2 * 3 * 4, dtype=float).reshape(24, 2, 3, 4)
    pd4 = PhysicalData.from_netcdf(_write(tmp_path, a4), varname="temperature", nsteps_year=24)
    assert pd4.n_layers == 2
    np.testing.assert_array_equal(pd4.get_grid(25, layer=1), a4[1, 1])   # step % 24, layer 1
    assert pd4.get_value(0, 2, 3, layer=0) == a4[0, 0, 2, 3]
    with pytest.raises(IndexError):
        pd4.get_grid(0, layer=2)
    a3 = np.ones((24, 3, 4))
    pd3 = PhysicalData.from_netcdf(_write(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, a3),
                                   varname="temperature", nsteps_year=24)
    assert pd3.n_layers == 1 and pd3.get_grid(5).shape == (3, 4) and pd3.get_grid(5, layer=0).shape == (3, 4)


def test_loader_java_precedence_value_then_file_then_none(tmp_path):
    p = _write(tmp_path, np.full((24, 2, 3, 4), 7.0))
    raw = {"temperature.filename": str(p), "temperature.varname": "temperature",
           "temperature.nsteps.year": "24", "simulation.time.ndtperyear": "24"}
    assert _load_temperature_data(raw, None).n_layers == 2
    raw["temperature.value"] = "5.5"
    assert _load_temperature_data(raw, None).is_constant     # .value wins (Java PhysicalData.init)
    assert _load_temperature_data({"simulation.time.ndtperyear": "24"}, None) is None


def test_loader_frame_mismatch_and_factor_offset(tmp_path):
    p = _write(tmp_path, np.full((12, 2, 3, 4), 7.0))
    raw = {"temperature.filename": str(p), "temperature.varname": "temperature",
           "temperature.nsteps.year": "12", "simulation.time.ndtperyear": "24"}
    with pytest.raises(ValueError, match="24"):
        _load_temperature_data(raw, None)
    p2 = _write(tmp_path / "c" if (tmp_path / "c").mkdir() is None else tmp_path, np.full((24, 3, 4), 7.0))
    raw2 = {"temperature.filename": str(p2), "temperature.varname": "temperature",
            "temperature.nsteps.year": "24", "simulation.time.ndtperyear": "24", "temperature.offset": "2.0"}
    assert _load_temperature_data(raw2, None).get_value(0, 0, 0) == 9.0


def test_bioen_without_temperature_source_raises():
    from tests.test_bioen_orchestration import _make_bioen_config_dict
    from osmose.engine import PythonEngine
    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    cfg.pop("temperature.value", None)
    cfg["simulation.time.nyear"] = "1"
    with pytest.raises(ValueError, match="temperature"):
        PythonEngine().run_in_memory(cfg, seed=0)


def test_bioen_step_reads_assigned_layer_and_skips_out_schools():
    from tests.test_bioen_orchestration import _make_bioen_config_dict, _make_school_state
    from osmose.engine.config import EngineConfig
    from osmose.engine.simulate import _bioen_step
    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    cfg["species.zlayer.sp1"] = "1"
    config = EngineConfig.from_dict(cfg)
    st = _make_school_state(n_schools=6, n_species=2)
    st = st.replace(cell_x=np.array([1, 1, 2, 2, 0, 0], dtype=np.int32),
                    cell_y=np.array([1, 1, 2, 2, 0, 0], dtype=np.int32),
                    is_out=np.array([False, False, False, False, True, True]))
    grid = np.zeros((24, 2, 5, 5)); grid[:, 0] = 4.0; grid[:, 1] = 12.0
    td = PhysicalData(data=grid, constant=None, nsteps_year=24)
    cap: dict = {}
    _bioen_step(st, config, td, step=0, debug_capture=cap)
    t = cap["temp_c"]
    assert t[0] == 4.0 and t[1] == 12.0 and t[2] == 4.0 and t[3] == 12.0   # sp0 -> layer 0, sp1 -> layer 1
    assert np.isnan(t[4]) and np.isnan(t[5])                              # out schools: no temperature, no budget


def test_gridded_o2_reaches_f_o2():
    from tests.test_bioen_orchestration import _make_bioen_config_dict, _make_school_state
    from osmose.engine.config import EngineConfig
    from osmose.engine.processes.oxygen_function import f_o2
    from osmose.engine.simulate import _bioen_step
    cfg = {k.lower(): v for k, v in _make_bioen_config_dict(n_species=2).items()}
    cfg["simulation.bioen.fo2.enabled"] = "true"
    config = EngineConfig.from_dict(cfg)
    st = _make_school_state(n_schools=4, n_species=2).replace(
        preyed_biomass=np.full(4, 1e-3), cell_x=np.zeros(4, dtype=np.int32), cell_y=np.zeros(4, dtype=np.int32))
    o2 = PhysicalData(data=np.full((24, 5, 5), 3.0), constant=None, nsteps_year=24)
    td = PhysicalData.from_constant(10.0)
    cap: dict = {}
    _bioen_step(st, config, td, step=0, o2_data=o2, debug_capture=cap)
    expected = f_o2(np.array(3.0), config.bioen_o2_c1[0], config.bioen_o2_c2[0])
    assert cap["f_o2"][0] == pytest.approx(float(expected))
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_engine_temperature_loader.py -q` → ImportError / TypeError.

- [ ] **Step 3: Implement `PhysicalData` layers**

```python
    @classmethod
    def from_netcdf(cls, path, varname="temp", nsteps_year=12, factor=1.0, offset=0.0):
        from osmose.engine._netcdf import open_dataset_safe
        ds = open_dataset_safe(path)
        raw = ds[varname].values
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        if raw.ndim not in (3, 4):
            raise ValueError(f"{path}:{varname} must be (time,y,x) or (time,z,y,x); got {raw.shape}")
        data = factor * (raw.astype(np.float64) + offset)
        return cls(data=data, constant=None, nsteps_year=nsteps_year)

    @property
    def n_layers(self) -> int:
        return 1 if self._data is None or self._data.ndim == 3 else int(self._data.shape[1])

    def _frame(self, step: int, layer: int):
        assert self._data is not None
        t_idx = step % self._data.shape[0]
        if self._data.ndim == 3:
            if layer != 0:
                raise IndexError(f"layer {layer} requested from a single-layer field")
            return self._data[t_idx]
        if not 0 <= layer < self._data.shape[1]:
            raise IndexError(f"layer {layer} out of range (n_layers={self._data.shape[1]})")
        return self._data[t_idx, layer]

    def get_value(self, step, cell_y, cell_x, layer=0):
        if self._constant is not None:
            return self._constant
        return float(self._frame(step, layer)[cell_y, cell_x])

    def get_grid(self, step, layer=0):
        if self._constant is not None:
            raise ValueError("Constant PhysicalData has no spatial grid")
        return self._frame(step, layer)
```

- [ ] **Step 4: Implement the loader and wire `_bioen_step`**

`_load_temperature_data` (beside `_load_oxygen_data`), Java precedence:

```python
def _load_temperature_data(raw_config: dict, config_dir: Path | None) -> PhysicalData | None:
    """Temperature forcing for bioenergetics. Java PhysicalData.init precedence: `temperature.value`
    first, else the file triple, else None. Frame count must equal simulation.time.ndtperyear."""
    val = raw_config.get("temperature.value", "")
    if val:
        return PhysicalData.from_constant(float(val), factor=float(raw_config.get("temperature.factor", "1.0")),
                                          offset=float(raw_config.get("temperature.offset", "0.0")))
    filename = raw_config.get("temperature.filename", "")
    if not filename:
        return None
    path = resolve_data_path(filename, config_dir=str(config_dir) if config_dir is not None else "")
    if path is None:
        raise FileNotFoundError(f"temperature.filename={filename!r} not found (config_dir={config_dir})")
    data = PhysicalData.from_netcdf(path, varname=raw_config.get("temperature.varname", "temperature"),
                                    nsteps_year=int(raw_config.get("temperature.nsteps.year", "12")),
                                    factor=float(raw_config.get("temperature.factor", "1.0")),
                                    offset=float(raw_config.get("temperature.offset", "0.0")))
    n_frames = data._data.shape[0]
    n_dt = int(raw_config.get("simulation.time.ndtperyear", "24"))
    if n_frames != n_dt:
        raise ValueError(f"Temperature forcing {path} has {n_frames} frame(s), but simulation.time.ndtperyear={n_dt}. "
                         "PhysicalData indexes step % frame_count; regenerate the file with the right frame count.")
    return data
```

At `simulate.py:1600-1605`: `temp_data = _load_temperature_data(config.raw_config, config_dir) if config.bioen_enabled else None`, followed by `if config.bioen_enabled and temp_data is None: raise ValueError("Bioenergetics is enabled but no temperature source is configured (temperature.value or temperature.filename) — Java requires one for the Arrhenius term even with phit disabled")`. (`config_dir` is the variable `_load_oxygen_data` receives at `:1556`; reuse it.) Note the silent-fallback path in `_bioen_step` (`:429-441`) is deleted: with the guard above `temp_data` is never `None` there.

In `_bioen_step`: build `temp_c_arr = np.full(len(state), np.nan)`; for each `sp, mask` (masks already exclude `is_out`): `layer = int(config.bioen_zlayer[sp])`; constant → `temp_c_arr[mask] = temp_data.get_value(step, 0, 0)`; gridded → `g = temp_data.get_grid(step, layer); temp_c_arr[mask] = g[state.cell_y[mask], state.cell_x[mask]]`. φT per species from `temp_c_arr[mask]` (the existing loop, now reading the per-species layer). O₂ block: add the `else:` branch — `grid = o2_data.get_grid(step); f_o2_arr[mask] = f_o2(grid[state.cell_y[mask], state.cell_x[mask]], c1, c2)` per species. Add `debug_capture: dict | None = None` to the signature and, before the budget loop, `if debug_capture is not None: debug_capture.update(temp_c=temp_c_arr.copy(), f_o2=f_o2_arr.copy(), species_id=state.species_id.copy(), is_out=state.is_out.copy())`. `_BIOEN_REQUIRED` gains `"bioen_zlayer"`.

- [ ] **Step 5: `meanEnetFaced` output + allowlists**

`_collect_bioen`: add a sixth return, the abundance-weighted mean of `state.e_net_avg` over focal, feeding (`age_dt >= first_feeding_age_dt`), in-domain schools per species (`np.add.at` of `e_net_avg*abundance` and `abundance`; 0 where no fish). Thread it as `bioen_enet_faced_by_species` through `StepOutput`, `_average_step_outputs` (`_avg_bioen("bioen_enet_faced_by_species")`), the two `StepOutput(...)` constructor sites, and `output.py:774` (`("bioen_enet_faced_by_species", "meanEnetFaced", True)`). `config_validation.py`: move the five `temperature.*` entries from `_ALLOWLIST_JAVA_ONLY` to `_ALLOWLIST_PY_HONORED` (alphabetical, beside `oxygen.*`); in `tests/test_issue_123_known_but_unread_keys.py:176-185` remove them from the java-only expectation. Fix the `schema/bioenergetics.py:6` comment.

- [ ] **Step 6: Run tests**

`.venv/bin/python -m pytest tests/test_engine_temperature_loader.py tests/test_bioen_orchestration.py tests/test_engine_bioen_outputs_complete.py tests/test_issue_123_known_but_unread_keys.py tests/test_engine_config_validation.py tests/test_engine_o2_wiring.py tests/test_engine_parity.py -q` → PASS. (`test_engine_bioen_outputs_complete` expects 5 CSVs — update to 6 with `meanEnetFaced`.)

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/physical_data.py osmose/engine/simulate.py osmose/engine/output.py osmose/engine/config_validation.py osmose/schema/bioenergetics.py tests/test_engine_temperature_loader.py tests/test_issue_123_known_but_unread_keys.py tests/test_engine_bioen_outputs_complete.py tests/test_bioen_orchestration.py
git commit -m "feat(engine): temperature forcing loader (Java precedence, frame guard), per-species depth layers, gridded f_o2, meanEnetFaced output"
```

---

### Task 8: Offline Java-form growth model, T_p solve, (Imax, r) fit, parameter-file writer

**Files:**
- Create: `osmose/calibration/bioen_offline.py`
- Create: `scripts/fit_baltic_bioen_params.py` (CLI; Baltic mode completed in Task 11, `--gate-b` mode here)
- Create: `data/examples_bioen/` (copy of `data/examples/` + `osm_param-bioen.csv` + master edits) — the Gate-B config
- Test: `tests/test_bioen_offline_fit.py`

**Interfaces:**
- Produces (all pure numpy/scipy):
  ```python
  @dataclass(frozen=True) class BioenFixed: a=0.7; beta=0.8; eta=1.0; e_m=0.65; e_d=1.5; e_maint=0.65;
                                              m_share=0.3; t_ref=16.0; larval_coef=1.0; larvae_thres_dt=1; first_feeding_dt=1
  @dataclass class SpeciesTargets: name; linf; k; t0; cf; b; egg_weight_g; m0; m1; lifespan_years; t_opt; t24: NDArray(24)
  @dataclass class FitResult: name; imax; r; c_m; t_p; t_opt; rms_len_pct; w_inf_fit_g; w_inf_vb_g; larval_ratio_half_year; n_points
  def solve_tp(t_opt: float, fx: BioenFixed) -> float
  def c_m_from_share(imax: float, t_p: float, fx: BioenFixed) -> float
  def g_net(T, imax, c_m, t_p, fx) -> NDArray                      # g·g^-beta·yr^-1
  def simulate_growth(imax, r, t_p, c_m, t24, w_egg_g, n_steps, ndt, cf, b, m0, m1, fx) -> NDArray  # grams, len n_steps+1
  def vbgf_weight(age_years, linf, k, t0, cf, b) -> NDArray
  def fit_species(tg: SpeciesTargets, fx: BioenFixed, ndt: int = 24) -> FitResult
  def bioen_param_lines(results: list[FitResult], fx: BioenFixed, zlayer: dict[str,int], sp_index: dict[str,int],
                        background_imax: dict[int, float], notes: dict[str, str]) -> list[str]   # CSV lines, full Java inventory
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bioen_offline_fit.py
import numpy as np
import pytest

from osmose.calibration.bioen_offline import (
    BioenFixed, FitResult, SpeciesTargets, bioen_param_lines, c_m_from_share, fit_species, g_net,
    simulate_growth, solve_tp, vbgf_weight)
from osmose.engine.processes.temp_function import arrhenius, phi_t

FX = BioenFixed()


def test_solve_tp_puts_net_growth_optimum_at_t_opt():
    for t_opt in (10.0, 15.0, 21.7, 27.0):
        tp = solve_tp(t_opt, FX)
        assert tp > t_opt                       # maintenance pulls the optimum below the phiT peak
        T = np.linspace(t_opt - 8, t_opt + 8, 1601)
        cm = c_m_from_share(1.0, tp, FX)
        g = g_net(T, 1.0, cm, tp, FX)
        assert abs(T[np.argmax(g)] - t_opt) <= 0.05
        assert phi_t(np.array(tp), FX.e_m, FX.e_d, tp) == 1.0


def test_c_m_share_is_m_at_t_ref():
    tp = solve_tp(12.0, FX); imax = 4.0
    cm = c_m_from_share(imax, tp, FX)
    share = cm * arrhenius(np.array(FX.t_ref), FX.e_maint) / (FX.a * imax * phi_t(np.array(FX.t_ref), FX.e_m, FX.e_d, tp))
    assert share == pytest.approx(FX.m_share, rel=1e-12)


def test_simulate_growth_saturates_where_rho_reaches_one():
    tp = solve_tp(10.0, FX); imax, r = 6.0, 0.4
    cm = c_m_from_share(imax, tp, FX)
    w = simulate_growth(imax, r, tp, cm, np.full(24, 10.0), w_egg_g=1e-3, n_steps=24 * 40, ndt=24,
                        cf=0.0087, b=3.05, m0=38.0, m1=0.0, fx=FX)
    assert np.all(np.diff(w) >= 0) and np.isfinite(w).all()
    gbar = g_net(np.array(10.0), imax, cm, tp, FX)
    w_inf = (FX.eta * gbar / r) ** (1.0 / (1.0 - FX.beta))
    assert w[-1] == pytest.approx(w_inf, rel=0.05)


def test_fit_recovers_known_parameters_from_its_own_model():
    tp = solve_tp(10.0, FX); imax_true, r_true = 5.0, 0.35
    cm = c_m_from_share(imax_true, tp, FX)
    t24 = 8.0 + 4.0 * np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    w = simulate_growth(imax_true, r_true, tp, cm, t24, 1e-3, 24 * 20, 24, 0.0087, 3.05, 38.0, 0.0, FX)
    ages = np.arange(24, 24 * 20 + 1) / 24.0
    L = (w[24:] / 0.0087) ** (1 / 3.05)
    # fit a vBGF to the model's own curve, then ask the fitter to recover (imax, r) from that vBGF
    from scipy.optimize import curve_fit
    (linf, k, t0), _ = curve_fit(lambda a, L_, k_, t0_: L_ * (1 - np.exp(-k_ * (a - t0_))), ages, L, p0=(100, 0.15, -0.2))
    tg = SpeciesTargets("synthetic", linf, k, t0, 0.0087, 3.05, 1e-3, 38.0, 0.0, 20.0, 10.0, t24)
    res = fit_species(tg, FX)
    assert res.imax == pytest.approx(imax_true, rel=0.05) and res.r == pytest.approx(r_true, rel=0.05)
    assert res.rms_len_pct < 3.0 and res.t_p == pytest.approx(tp)


def test_param_lines_cover_the_java_inventory_and_background():
    res = [FitResult("cod", 4.0, 0.3, 1e12, 13.0, 10.0, 2.0, 5e3, 5.1e3, 0.6, 400)]
    lines = bioen_param_lines(res, FX, zlayer={"cod": 1}, sp_index={"cod": 0}, background_imax={15: 2.5},
                              notes={"cod": "T_opt 10 C (Bjornsson & Steinarsson 2002)"}, m0={"cod": 2.0})
    text = "\n".join(lines)
    for key in ("species.bioen.mobilized.tp.sp0;13.0", "species.bioen.maint.energy.c_m.sp0;1e+12",
                "species.maturity.m0.sp0;2.0", "species.maturity.r.sp0;0.3", "predation.ingestion.rate.max.sp0;4.0",
                "species.zlayer.sp0;1", "species.bioen.forage.k_for.sp0;0.0", "predation.c.bioen.sp0;0.0",
                "predation.larval.ingestion.rate.increase.ratio.sp0;1.0", "species.oxygen.c2.sp0;",
                "predation.ingestion.rate.max.sp15;2.5"):
        assert key in text, key
    assert "T_opt 10 C" in text
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement `osmose/calibration/bioen_offline.py`**

```python
"""Offline Java-form bioenergetics growth model for parameter fitting (C3 spec §3.4).

Per fish, grams, one 24-step year cycle. Same equations as EnergyBudget.java / the parity-fixed
engine with N = 1 and ingestion at the cap (food-unlimited). Used to (a) solve T_p so that the
net-growth optimum equals a cited growth optimum, (b) fit (Imax, r) to a config's own vBGF curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from osmose.engine.processes.temp_function import arrhenius, phi_t


@dataclass(frozen=True)
class BioenFixed:
    a: float = 0.7
    beta: float = 0.8
    eta: float = 1.0
    e_m: float = 0.65
    e_d: float = 1.5
    e_maint: float = 0.65
    m_share: float = 0.3      # maintenance / maximal ingestion at t_ref (Bernreuther et al. 2012, 16 C)
    t_ref: float = 16.0
    larval_coef: float = 1.0
    larvae_thres_dt: int = 1
    first_feeding_dt: int = 1


@dataclass
class SpeciesTargets:
    name: str
    linf: float
    k: float
    t0: float
    cf: float
    b: float
    egg_weight_g: float
    m0: float
    m1: float
    lifespan_years: float
    t_opt: float
    t24: NDArray[np.float64]


@dataclass
class FitResult:
    name: str
    imax: float
    r: float
    c_m: float
    t_p: float
    t_opt: float
    rms_len_pct: float
    w_inf_fit_g: float
    w_inf_vb_g: float
    larval_ratio_half_year: float
    n_points: int


def c_m_from_share(imax: float, t_p: float, fx: BioenFixed) -> float:
    tr = np.array(fx.t_ref)
    return fx.m_share * fx.a * imax * float(phi_t(tr, fx.e_m, fx.e_d, t_p)) / float(arrhenius(tr, fx.e_maint))


def g_net(T, imax: float, c_m: float, t_p: float, fx: BioenFixed):
    T = np.asarray(T, dtype=np.float64)
    return fx.a * phi_t(T, fx.e_m, fx.e_d, t_p) * imax - c_m * arrhenius(T, fx.e_maint)


def _dg_dT_at(t_opt: float, t_p: float, fx: BioenFixed, h: float = 1e-3) -> float:
    cm = c_m_from_share(1.0, t_p, fx)
    return float(g_net(t_opt + h, 1.0, cm, t_p, fx) - g_net(t_opt - h, 1.0, cm, t_p, fx)) / (2 * h)


def solve_tp(t_opt: float, fx: BioenFixed) -> float:
    """T_p such that argmax_T g_net(T) == t_opt (Imax cancels; depends on m, e_*, t_ref only)."""
    lo, hi = t_opt, t_opt + 25.0
    assert _dg_dT_at(t_opt, lo, fx) < 0.0, "at t_p == t_opt the slope must be negative (maintenance)"
    return float(brentq(lambda tp: _dg_dT_at(t_opt, tp, fx), lo, hi, xtol=1e-6))


def vbgf_weight(age_years, linf, k, t0, cf, b):
    L = linf * (1.0 - np.exp(-k * (np.asarray(age_years, dtype=np.float64) - t0)))
    return cf * np.power(np.maximum(L, 1e-9), b)


def simulate_growth(imax, r, t_p, c_m, t24, w_egg_g, n_steps, ndt, cf, b, m0, m1, fx: BioenFixed):
    """Per-fish weight (g) at step index 0..n_steps; step index == age_dt. Java budget with N = 1."""
    t24 = np.asarray(t24, dtype=np.float64)
    w = np.empty(n_steps + 1); w[0] = w_egg_g
    faced = 0.0
    for age_dt in range(1, n_steps + 1):
        wg = w[age_dt - 1]
        T = t24[(age_dt - 1) % ndt]
        wb = wg ** fx.beta
        if age_dt < fx.first_feeding_dt:
            w[age_dt] = wg; continue
        e_gross = fx.a * float(phi_t(np.array(T), fx.e_m, fx.e_d, t_p)) * imax * wb / ndt
        e_maint = c_m * float(arrhenius(np.array(T), fx.e_maint)) * wb / ndt
        e_net = e_gross - e_maint
        per_fish = e_net * ndt / wb
        if age_dt == fx.first_feeding_dt:
            faced = per_fish / fx.larval_coef
        elif age_dt < fx.larvae_thres_dt:
            faced = (per_fish / fx.larval_coef + faced * age_dt) / (age_dt + 1)
        else:
            faced = (per_fish + faced * age_dt) / (age_dt + 1)
        L = (wg / cf) ** (1.0 / b)
        mature = L >= m0 + m1 * (age_dt / ndt)
        if mature:
            with np.errstate(divide="ignore", invalid="ignore"):
                rho = r * wg ** (1.0 - fx.beta) / (fx.eta * faced)
            rho = 1.0 if np.isnan(rho) else float(np.clip(rho, 0.0, 1.0))
        else:
            rho = 0.0
        w[age_dt] = wg + (1.0 - rho) * max(e_net, 0.0)
    return w


def fit_species(tg: SpeciesTargets, fx: BioenFixed, ndt: int = 24) -> FitResult:
    t_p = solve_tp(tg.t_opt, fx)
    n_steps = int(round(tg.lifespan_years * ndt))
    idx = np.arange(ndt, n_steps + 1)                   # ages >= 1 yr (spec 3.4: larval phase not fitted)
    ages = idx / ndt
    target_w = vbgf_weight(ages, tg.linf, tg.k, tg.t0, tg.cf, tg.b)

    def resid(x):
        imax, r = np.exp(x)
        w = simulate_growth(imax, r, t_p, c_m_from_share(imax, t_p, fx), tg.t24, tg.egg_weight_g,
                            n_steps, ndt, tg.cf, tg.b, tg.m0, tg.m1, fx)
        return np.log(np.maximum(w[idx], 1e-12)) - np.log(target_w)

    sol = least_squares(resid, x0=np.log([5.0, 0.3]), bounds=(np.log([1e-3, 1e-4]), np.log([1e3, 1e2])))
    imax, r = np.exp(sol.x)
    c_m = c_m_from_share(imax, t_p, fx)
    w = simulate_growth(imax, r, t_p, c_m, tg.t24, tg.egg_weight_g, n_steps, ndt, tg.cf, tg.b, tg.m0, tg.m1, fx)
    len_model = (w[idx] / tg.cf) ** (1 / tg.b)
    len_target = (target_w / tg.cf) ** (1 / tg.b)
    rms = float(np.sqrt(np.mean(((len_model - len_target) / len_target) ** 2)) * 100)
    gbar = float(np.mean(g_net(tg.t24, imax, c_m, t_p, fx)))
    w_inf_fit = (fx.eta * gbar / r) ** (1.0 / (1.0 - fx.beta))
    half = ndt // 2
    w_half_target = tg.cf * (tg.linf * (1 - np.exp(-tg.k * (0.5 - tg.t0)))) ** tg.b
    return FitResult(tg.name, float(imax), float(r), float(c_m), t_p, tg.t_opt, rms, float(w_inf_fit),
                     float(tg.cf * tg.linf ** tg.b), float(w[half] / w_half_target), int(idx.size))


def bioen_param_lines(results, fx: BioenFixed, zlayer, sp_index, background_imax, notes, m0):
    """Full Java 4.3.3 bioen key inventory (canonical 4.4.0 spellings), one block per species.

    m0: {species name: maturity length (cm)} -- the config's species.maturity.size values."""
    out = ["# Generated by scripts/fit_baltic_bioen_params.py -- DO NOT EDIT BY HAND (C3 spec 3.4)",
           f"# fixed: a={fx.a} beta={fx.beta} eta={fx.eta} e_M={fx.e_m} e_D={fx.e_d} e_maint={fx.e_maint} eV "
           f"(engine defaults); m={fx.m_share} at {fx.t_ref} C (Bernreuther et al. 2012)"]
    for res in results:
        i = sp_index[res.name]
        out += [f"# --- {res.name} (sp{i}): {notes.get(res.name, '')}; growth optimum {res.t_opt} C -> engine T_p {res.t_p:.4f} C; "
                f"fit RMS length {res.rms_len_pct:.1f}% (ages >= 1 yr); W_inf fit {res.w_inf_fit_g:.0f} g vs vBGF {res.w_inf_vb_g:.0f} g; "
                f"larval ratio at 0.5 yr {res.larval_ratio_half_year:.2f}",
                f"species.beta.sp{i};{fx.beta!r}", f"species.zlayer.sp{i};{zlayer[res.name]}",
                f"species.bioen.assimilation.sp{i};{fx.a!r}", f"species.bioen.maint.energy.c_m.sp{i};{res.c_m!r}",
                f"species.bioen.maint.e.maint.sp{i};{fx.e_maint!r}", f"species.bioen.mobilized.e.mobi.sp{i};{fx.e_m!r}",
                f"species.bioen.mobilized.e.d.sp{i};{fx.e_d!r}", f"species.bioen.mobilized.tp.sp{i};{res.t_p!r}",
                f"species.maturity.eta.sp{i};{fx.eta!r}", f"species.maturity.r.sp{i};{res.r!r}",
                f"species.maturity.m0.sp{i};{m0[res.name]!r}",
                f"species.maturity.m1.sp{i};0.0", f"predation.ingestion.rate.max.sp{i};{res.imax!r}",
                f"predation.larval.ingestion.rate.increase.ratio.sp{i};{fx.larval_coef!r}", f"predation.c.bioen.sp{i};0.0",
                f"species.oxygen.c1.sp{i};1.0", f"species.oxygen.c2.sp{i};60.0",
                f"species.bioen.forage.k_for.sp{i};0.0", f"species.bioen.forage.k1_for.sp{i};0.0",
                f"species.bioen.forage.k2_for.sp{i};0.0"]
    for b_idx, imax in sorted(background_imax.items()):
        out += [f"# background predator sp{b_idx}: Imax in bioen units = standard rate * mean-weight factor",
                f"predation.ingestion.rate.max.sp{b_idx};{imax!r}",
                f"predation.larval.ingestion.rate.increase.ratio.sp{b_idx};1.0", f"predation.c.bioen.sp{b_idx};0.0"]
    return out
```

`species.oxygen.c2` = 60.0 mmol m⁻³ (inert with fo2 off; matches the benthos coupling's c50). `repr` floats are written so the round-trip through the reader is exact (Gate F pin).

- [ ] **Step 4: `scripts/fit_baltic_bioen_params.py --gate-b`**

CLI with `--gate-b OUT_DIR` (this task) and `--baltic` (Task 11). Gate-B mode: copy `data/examples/` to `OUT_DIR` (default `data/examples_bioen/`), read the copied master with `OsmoseConfigReader`, build `SpeciesTargets` per species from `species.linf/k/t0/lifespan/length2weight.*/egg.size/egg.weight/maturity.size` (egg weight in grams = `egg.weight` if present else `cf*egg.size^b`), `t_opt = 15.0`, `t24 = np.full(24, 15.0)`, fit, and write `osm_param-bioen.csv` (plus for background species, if any, `Imax = rate*(w_mean*1e6)^(1-beta)` with `w_mean` from the species' mean class weight — the examples config has none, so `background_imax={}`), then append to the master: `osmose.configuration.bioen;osm_param-bioen.csv`, `module.bioenergetics.enabled;true`, `simulation.bioen.phit.enabled;true`, `simulation.bioen.fo2.enabled;false`, `temperature.value;15.0`, `oxygen.value;300.0`. Print the FitResult table. Run it and commit `data/examples_bioen/` (≈ the size of `data/examples/`; check with `du -sh`; if > 5 MB because of NetCDF/maps, write only the master + bioen CSV there and make the master's relative paths point at `../examples/…` instead of copying).

- [ ] **Step 5: Run tests** — `.venv/bin/python -m pytest tests/test_bioen_offline_fit.py -q` → PASS; `PYTHONPATH=. .venv/bin/python scripts/fit_baltic_bioen_params.py --gate-b` prints eight FitResults with RMS ≤ 15 % (if a species fails the pin, print it and continue — Gate B does not need a perfect fit, only a runnable, non-degenerate one; record the values in the commit message).

- [ ] **Step 6: Commit**

```bash
git add osmose/calibration/bioen_offline.py scripts/fit_baltic_bioen_params.py data/examples_bioen tests/test_bioen_offline_fit.py
git commit -m "feat(calibration): offline Java-form bioen growth model, T_p solve from growth optimum, (Imax, r) fit; Gate-B BoB bioen config"
```

---

### Task 9: Gate B — cross-engine parity of bioen-on (Python vs Java 4.3.3 gated, 4.4.1 reported)

**Files:**
- Modify: `scripts/cross_engine_parity_440.py` (`ensemble():98-118`, `main():143-249`, `METRICS`)
- Create: `docs/diagnostics/c3_gate_b_cross_engine.md` (the run's output, verbatim) 
- Test: `tests/test_cross_engine_parity_bioen_staging.py`

**Interfaces:**
- Produces in `cross_engine_parity_440.py`:
  ```python
  def inject_java_bioen_keys(master: Path, raw: dict[str, str]) -> int   # appends 4.3.3-only bioen keys; returns count
  def nondegenerate(ens: dict, metric: str, n: int, floor: float, frac: float = 0.9) -> dict[str, bool]  # per species
  ```
  New CLI flags: `--gate-engine {4.4.1,4.3.3}` (default 4.4.1 — keep existing behaviour; C3 passes `4.3.3`), `--delta-mean-weight` (default `log10(1.5)`), `--require-nondegenerate` (FAIL when any species collapses in ≥ 10 % of reps in either engine). `mean_size` added to the reported metrics when both `results.mean_size()` frames are non-empty.

- [ ] **Step 1: Write the failing staging test**

```python
# tests/test_cross_engine_parity_bioen_staging.py
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xeng", ROOT / "scripts" / "cross_engine_parity_440.py")
xeng = importlib.util.module_from_spec(spec); spec.loader.exec_module(xeng)


def test_inject_java_bioen_keys_appends_bioen_imax_for_every_predator(tmp_path):
    master = tmp_path / "osm_all-parameters.csv"
    master.write_text("predation.ingestion.rate.max.sp0 ; 3.5\nspecies.type.sp15 ; background\n")
    raw = {"module.bioenergetics.enabled": "true", "predation.ingestion.rate.max.sp0": "3.5",
           "predation.ingestion.rate.max.sp15": "2.5", "species.type.sp15": "background",
           "predation.larval.ingestion.rate.increase.ratio.sp0": "1.0", "predation.c.bioen.sp0": "0.0",
           "simulation.nspecies": "1"}
    n = xeng.inject_java_bioen_keys(master, raw)
    text = master.read_text()
    assert "predation.ingestion.rate.max.bioen.sp0 ; 3.5" in text
    assert "predation.ingestion.rate.max.bioen.sp15 ; 2.5" in text
    assert "predation.coef.ingestion.rate.max.larvae.bioen.sp15 ; 1.0" in text and "predation.c.bioen.sp15 ; 0.0" in text
    assert "predation.ingestion.rate.max.sp0 ; 3.5" in text     # legacy standard key kept (Java reads both)
    assert n >= 4


def test_inject_is_noop_without_bioen(tmp_path):
    master = tmp_path / "m.csv"; master.write_text("a ; 1\n")
    assert xeng.inject_java_bioen_keys(master, {"module.bioenergetics.enabled": "false"}) == 0
    assert master.read_text() == "a ; 1\n"


def test_nondegenerate_flags_species_collapsed_in_too_many_reps():
    import numpy as np
    ens = {"biomass": {"A": np.array([500.0] * 16), "B": np.array([500.0] * 13 + [0.5] * 3)}}
    nd = xeng.nondegenerate(ens, "biomass", n=16, floor=1.0, frac=0.9)
    assert nd == {"A": True, "B": False}
```

- [ ] **Step 2: Run → fails (AttributeError).**

- [ ] **Step 3: Implement**

```python
def inject_java_bioen_keys(master: Path, raw: dict[str, str]) -> int:
    """Java 4.3.3 reads predation.ingestion.rate.max.bioen / .coef...larvae.bioen / predation.c.bioen for
    EVERY predator index (focal + background) and exits on a missing key; the writer's reverse aliases
    never emit .bioen (lossy merge). Append them from the canonical values. No-op unless bioen is on."""
    if str(raw.get("module.bioenergetics.enabled", "false")).lower() != "true":
        return 0
    n_sp = int(raw.get("simulation.nspecies", "0"))
    idx = list(range(n_sp)) + sorted(int(k.split(".sp")[-1]) for k, v in raw.items()
                                     if k.startswith("species.type.sp") and v.strip().lower() == "background")
    lines = []
    for i in idx:
        imax = raw.get(f"predation.ingestion.rate.max.sp{i}")
        if imax is None:
            raise KeyError(f"predation.ingestion.rate.max.sp{i} missing; cannot stage bioen for Java 4.3.3")
        lines.append(f"predation.ingestion.rate.max.bioen.sp{i} ; {imax}\n")
        lines.append(f"predation.coef.ingestion.rate.max.larvae.bioen.sp{i} ; "
                     f"{raw.get(f'predation.larval.ingestion.rate.increase.ratio.sp{i}', '1.0')}\n")
        lines.append(f"predation.c.bioen.sp{i} ; {raw.get(f'predation.c.bioen.sp{i}', '0.0')}\n")
    with master.open("a") as fh:
        fh.writelines(lines)
    return len(lines)


def nondegenerate(ens, metric, n, floor, frac=0.9):
    out = {}
    for sp, arr in ens[metric].items():
        ok = np.isfinite(arr) & (arr > 100.0 * floor)
        out[sp] = bool(arr.size == n and ok.mean() >= frac)
    return out
```

In `ensemble()` (Java branch): after `write_temp_config(...)` and before the rep loop, `if engine == "4.3.3": inject_java_bioen_keys(master, raw)`; after the loop, `if len(reps) != n: raise RuntimeError(f"engine {engine}: {len(reps)}/{n} reps succeeded — see stderr of the first failure")` and make `java_rep` return the captured stderr tail on failure so the message carries it (`cp = subprocess.run(...)`; on non-zero, `print(cp.stderr[-2000:])`). In `main()`: `--gate-engine` chooses which Java arm feeds `overall_fail` (default `4.4.1` keeps today's behaviour); per-metric Δ = `args.delta_mean_weight` for `mean_weight` and `mean_size`; add the `mean_size` reader (`results.mean_size()` if the frame is non-empty, else skip the metric with a printed note); with `--require-nondegenerate`, compute `nondegenerate(...)` for biomass and abundance on the Python arm and the gate arm and print per species `collapse_frac`; any `False` → `verdict = "FAIL (degenerate: …)"`. Print the 90 % CI half-width already computed by `tost()` per species (it is in `ci`).

- [ ] **Step 4: Run Gate B (detached; ~30–60 min: 16 reps × 3 engines × 10 yr)**

```bash
printf '%s\n' '#!/usr/bin/env bash' 'cd /home/razinka/osmopy' 'export PYTHONPATH=.' \
 '.venv/bin/python scripts/cross_engine_parity_440.py --config data/examples_bioen/osm_all-parameters.csv --n 16 --years 10 --engines python,4.3.3,4.4.1 --gate-engine 4.3.3 --require-nondegenerate > /tmp/claude-1000/-home-razinka-osmopy/f19fe0be-9cc5-4217-979c-2d0a13c87eda/scratchpad/gate_b.log 2>&1' \
 > /tmp/claude-1000/-home-razinka-osmopy/f19fe0be-9cc5-4217-979c-2d0a13c87eda/scratchpad/run_gate_b.sh
setsid nohup bash /tmp/claude-1000/-home-razinka-osmopy/f19fe0be-9cc5-4217-979c-2d0a13c87eda/scratchpad/run_gate_b.sh > /dev/null 2>&1 &
```

Then also run the **control**: the same command with `--config data/examples/osm_all-parameters.csv` (bioen off) to a second log. Expected: control `GATE … PASS` (nothing but bioen differs between the two configs); bioen `GATE (… 4.3.3 …): PASS` with every species non-degenerate. If the bioen run says `REVIEW:` for `mean_weight` only, inspect which species and the `d` sign: a consistent Python-high `mean_weight` means a remaining per-fish/per-school factor — go back to Task 3/4, do not widen Δ. If Java 4.3.3 exits on a missing key, read the stderr line, add the key to `inject_java_bioen_keys` (with the same value source) and re-run.

- [ ] **Step 5: Record**

Copy both logs verbatim into `docs/diagnostics/c3_gate_b_cross_engine.md` under headings "bioen (gated 4.3.3, reported 4.4.1)" and "control (bioen off)", with the git commit, date, and the CLI lines.

- [ ] **Step 6: Commit**

```bash
git add scripts/cross_engine_parity_440.py tests/test_cross_engine_parity_bioen_staging.py docs/diagnostics/c3_gate_b_cross_engine.md
git commit -m "test(c3): Gate B cross-engine parity of bioen-on -- Java 4.3.3 staging of .bioen keys, non-degeneracy precondition, mean_size metric; results recorded"
```

---

### Task 10: Two-layer Baltic temperature climatology with `bottom_depth`

**Files:**
- Create: `scripts/build_baltic_temperature_forcing.py`
- Create: `data/baltic/forcing/baltic_temperature_2layer_climatology.nc`
- Test: `tests/test_build_baltic_temperature_forcing.py`

**Interfaces:**
- Produces (importable via importlib like the C4 harness):
  ```python
  def layer0_from_thetao(thetao_tzyx: NDArray) -> NDArray          # nan-aware depth mean -> (t, y, x)
  def monthly_climatology(frames_by_year: list[NDArray]) -> NDArray # (12, y, x) nan-aware mean over years
  def masked_regrid(raw_tyx, src_lat, src_lon, tlat, tlon, wet) -> NDArray   # = make_baltic_oxygen_forcing._masked_regrid (import it)
  def duplicate_months(clim12: NDArray) -> NDArray                  # (24, ...) frames 2m, 2m+1 = month m
  def bottom_depth_from_so(so_tzyx: NDArray, depth: NDArray) -> NDArray   # deepest finite level's depth (y, x), NaN on land
  def build(thetao_files, bottomt_files, so_file, grid_nc) -> xr.Dataset  # variables temperature(24,2,40,50) float32, bottom_depth(40,50)
  def validate(ds: xr.Dataset, wet: NDArray) -> None                # raises on any spec 3.3 pin failure
  ```

- [ ] **Step 1: Write the failing tests (synthetic native grid)**

```python
# tests/test_build_baltic_temperature_forcing.py
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("btf", ROOT / "scripts" / "build_baltic_temperature_forcing.py")
btf = importlib.util.module_from_spec(spec); spec.loader.exec_module(btf)


def test_layer0_is_nan_aware_over_depth():
    a = np.full((2, 3, 2, 2), np.nan); a[:, 0] = 10.0; a[:, 1, 0, 0] = 12.0   # pixel (0,0) has 2 levels, others 1
    out = btf.layer0_from_thetao(a)
    assert out.shape == (2, 2, 2) and out[0, 0, 0] == 11.0 and out[0, 1, 1] == 10.0


def test_duplicate_months_convention():
    clim = np.arange(12, dtype=float)[:, None, None] * np.ones((1, 2, 2))
    d = btf.duplicate_months(clim)
    assert d.shape == (24, 2, 2) and d[0, 0, 0] == 0 and d[1, 0, 0] == 0 and d[16, 0, 0] == 8 and d[17, 0, 0] == 8


def test_bottom_depth_from_so_takes_deepest_finite_level():
    so = np.full((1, 3, 2, 2), np.nan); so[0, 0] = 7; so[0, 1, 0, 0] = 8; so[0, 2, 0, 0] = 9; so[0, 1, 1, 1] = 8
    depth = np.array([1.0, 20.0, 60.0])
    bd = btf.bottom_depth_from_so(so, depth)
    assert bd[0, 0] == 60.0 and bd[1, 1] == 20.0 and bd[0, 1] == 1.0


def test_validate_fires_on_swapped_layers_and_passes_on_correct():
    wet = np.ones((2, 2), dtype=bool)
    temp = np.zeros((24, 2, 2, 2), dtype=np.float32); temp[:, 0] = 15.0; temp[:, 1] = 5.0   # surface warm, bottom cold
    bd = np.full((2, 2), 80.0)
    ds = xr.Dataset({"temperature": (["time", "layer", "latitude", "longitude"], temp),
                     "bottom_depth": (["latitude", "longitude"], bd)})
    btf.validate(ds, wet)
    swapped = ds.copy(); swapped["temperature"].values[:, [0, 1]] = swapped["temperature"].values[:, [1, 0]]
    with pytest.raises(AssertionError, match="layer-order"):
        btf.validate(swapped, wet)
    bad = ds.copy(); bad["temperature"].values[3, 0, 0, 0] = 45.0
    with pytest.raises(AssertionError, match="range"):
        btf.validate(bad, wet)
```

- [ ] **Step 2: Run → fails (file missing).**

- [ ] **Step 3: Implement the builder**

Module docstring: provenance (product `cmems_mod_bal_phy_my_P1M-m`, files, years 1993–2021, layer definitions, month-duplication convention, land NaN, `bottom_depth` from one full-depth `so` year-file). Core functions:

```python
def layer0_from_thetao(a):            return np.nanmean(a, axis=1)          # (t, z, y, x) -> (t, y, x); warnings suppressed for all-NaN pixels
def monthly_climatology(frames):      return np.nanmean(np.stack(frames, axis=0), axis=0)   # list of (12, y, x) -> (12, y, x)
def duplicate_months(clim12):         return np.repeat(clim12, 2, axis=0)
def bottom_depth_from_so(so, depth):
    finite = np.isfinite(so[0]); ndepth = so.shape[1]
    rev_first = np.argmax(finite[::-1], axis=0); idx = (ndepth - 1) - rev_first
    return np.where(finite.any(axis=0), depth[idx], np.nan)
```

`build()` reads each year-file with xarray (`thetao` → `layer0_from_thetao`; `bottomT` as-is), accumulates 12-month frames per year, `monthly_climatology`, `masked_regrid` (import `_masked_regrid` from `scripts/make_baltic_oxygen_forcing.py` via importlib; it needs the target coords from `osmose.forcing.grid.target_coords(GridSpec)` and the wet mask from `osmose.engine.grid.Grid.from_netcdf(grid_nc).ocean_mask` — copy the exact calls from `make_baltic_oxygen_forcing.build():179-234`), sets land to NaN (`out[:, ~wet] = np.nan`), duplicates to 24, stacks `[surface, bottom]` on axis 1 → `(24, 2, 40, 50)` float32; `bottom_depth` = `masked_regrid` of the static field (1 frame) with land NaN. Dataset attrs: `frame_convention="month duplicated x2 (frames 2m,2m+1 = month m), matches baltic_oxygen_bottom.nc; NOT resample_to_24 interpolation"`, `land="NaN"`, `layers="0: surface nan-mean thetao 0.50-4.68 m; 1: CMEMS bottomT"`, `source_years="1993-2021"`, `generator=<script>@<commit>`. `validate(ds, wet)`:

```python
def validate(ds, wet):
    t = ds["temperature"].values
    assert t.shape[0] == 24 and t.shape[1] == 2, f"shape {t.shape}"
    wet_vals = t[:, :, wet]
    assert np.isfinite(wet_vals).all(), "finite: NaN on a wet cell"
    assert (wet_vals >= -2.0).all() and (wet_vals <= 30.0).all(), "range: wet-cell temperature outside [-2, 30] C"
    deep = wet & (ds["bottom_depth"].values > 40.0)
    aug = t[16:18]
    assert np.all(aug[:, 1][:, deep] <= aug[:, 0][:, deep] + 1e-6), "layer-order: August bottom > surface on deep cells"
```

CLI: `--out data/baltic/forcing/baltic_temperature_2layer_climatology.nc`, `--cache data/cmems_cache/cmems_downloads`, `--so-file <first so year-file>`, `--grid data/baltic/baltic_grid.nc`; runs `build`, `validate`, writes with `encoding={"temperature": {"dtype": "float32"}}`, prints per-layer wet-cell min/mean/max per month and the deep-cell count.

- [ ] **Step 4: Build the file and check it**

Run: `PYTHONPATH=. .venv/bin/python scripts/build_baltic_temperature_forcing.py` (reads 29+29+1 files; a few minutes). Then a one-liner check: open the file, print `dict(ds.sizes)`, `ds.temperature.dtype`, `float(np.nanmean(ds.temperature[:, 0]))`, `float(np.nanmean(ds.temperature[:, 1]))` (expect surface annual mean ≈ 7–9 °C, bottom ≈ 4–7 °C), and `int((ds.bottom_depth > 40).sum())`. Write these numbers into the commit message.

- [ ] **Step 5: Tests + commit**

`.venv/bin/python -m pytest tests/test_build_baltic_temperature_forcing.py -q` → PASS.

```bash
git add scripts/build_baltic_temperature_forcing.py data/baltic/forcing/baltic_temperature_2layer_climatology.nc tests/test_build_baltic_temperature_forcing.py
git commit -m "feat(baltic): two-layer temperature climatology (surface thetao nan-mean, CMEMS bottomT) + bottom_depth, month-duplicated 24 frames, land NaN"
```

---

### Task 11: Baltic parameter set and flat overlay (`--baltic` mode)

**Files:**
- Modify: `scripts/fit_baltic_bioen_params.py` (add `--baltic`)
- Create: `data/baltic/scenarios/c3_bioen/baltic_param-bioen.csv`, `data/baltic/scenarios/c3_bioen/c3_bioen_arm.json`, `data/baltic/scenarios/c3_bioen/README.md`
- Test: extend `tests/test_bioen_offline_fit.py` with the habitat-series and overlay tests below

**Interfaces:**
- Produces in the script (importable):
  ```python
  SPECIES_T_OPT = {"cod_west": 10.0, "cod_east": 10.0, "herring": 15.0, "sprat": 18.0, "flounder": 19.0,
                   "perch": 25.0, "pikeperch": 27.0, "smelt": 15.0, "stickleback": 21.7}      # spec §1
  SPECIES_ZLAYER = {"cod_west": 1, "cod_east": 1, "flounder": 1, "herring": 0, "sprat": 0,
                    "stickleback": 0, "smelt": 0, "perch": 0, "pikeperch": 0}                  # spec decision 4
  SPECIES_NOTE = {...}   # the §1 label per species, verbatim (provisional / secondary / consumption proxy / size compromise)
  def habitat_t24(temp_nc: Path, layer: int, map_files: list[Path], ny: int, nx: int) -> NDArray(24)
  def background_imax(config: EngineConfig, b: BackgroundSpeciesInfo, beta: float = 0.8) -> float
  def build_overlay(csv_path: Path, temp_nc: Path) -> dict[str, str]     # FLAT dict: bioen CSV keys + switches + temperature keys
  ```
- `c3_bioen_arm.json` = `build_overlay(...)` serialised (all values strings), with `"_meta": {"spec": ..., "generated_by": ..., "commit": ...}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_habitat_t24_uses_engine_map_loader_and_layer(tmp_path):
    import xarray as xr
    from osmose.engine.movement_maps import _load_csv_grid
    fit = _load_fit_module()   # importlib load of scripts/fit_baltic_bioen_params.py (same pattern as the harness tests)
    temp = np.zeros((24, 2, 3, 3)); temp[:, 0] = 5.0; temp[:, 1] = np.arange(24)[:, None, None]
    nc = tmp_path / "t.nc"
    xr.Dataset({"temperature": (["time", "layer", "latitude", "longitude"], temp.astype(np.float32))}).to_netcdf(nc)
    m = tmp_path / "map.csv"
    m.write_text("0;0;0\n0;1;0\n0;0;0\n")     # one habitat cell; orientation handled by _load_csv_grid
    t24 = fit.habitat_t24(nc, layer=1, map_files=[m], ny=3, nx=3)
    np.testing.assert_allclose(t24, np.arange(24))
    assert fit.habitat_t24(nc, layer=0, map_files=[m], ny=3, nx=3).mean() == 5.0


def test_build_overlay_is_flat_and_carries_every_bioen_key(tmp_path):
    fit = _load_fit_module()
    csv = tmp_path / "baltic_param-bioen.csv"
    csv.write_text("# c\nspecies.bioen.mobilized.tp.sp0;12.5\nspecies.maturity.r.sp0;0.3\n")
    ov = fit.build_overlay(csv, tmp_path / "temp.nc")
    assert ov["module.bioenergetics.enabled"] == "true" and ov["simulation.bioen.phit.enabled"] == "true"
    assert ov["simulation.bioen.fo2.enabled"] == "false"
    assert ov["species.bioen.mobilized.tp.sp0"] == "12.5" and ov["species.maturity.r.sp0"] == "0.3"
    assert ov["temperature.filename"] == str((tmp_path / "temp.nc").resolve())
    assert ov["temperature.varname"] == "temperature" and ov["temperature.nsteps.year"] == "24"
    assert not any(k.startswith("osmose.configuration.") for k in ov) and "temperature.value" not in ov
```

- [ ] **Step 2: Implement `--baltic`**

`habitat_t24`: load the field with `PhysicalData.from_netcdf(nc, varname="temperature", nsteps_year=24)`, the maps with `_load_csv_grid(path, ny, nx)` (values > 0 = habitat; union over the species' map files listed in `data/baltic/baltic_param-movement.csv` for that species — parse `movement.species.map{n}` / `movement.file.map{n}`), and return `np.array([np.nanmean(pd.get_grid(s, layer)[habitat]) for s in range(24)])`; raise if any step is NaN (habitat must have finite temperature). `background_imax(config, b)`: `w_mean` = the abundance-unweighted mean class weight `cf*L^b*1e-6` over `b.lengths` (t/fish); return `b.ingestion_rate * (w_mean * 1e6) ** (1 - beta)` (the bioen cap then equals the standard cap at that weight). Baltic mode: read the production config through `osmose_demo("baltic", tmp)` + `OsmoseConfigReader`, build `EngineConfig`, one `SpeciesTargets` per focal species (`t24 = habitat_t24(temp_nc, SPECIES_ZLAYER[name], maps)`, egg weight grams from `config.egg_weight_override` (t→g) or `cf*egg_size^b`, `m0 = config.maturity_size`, lifespan from `species.lifespan`), fit all nine, print the table, write `baltic_param-bioen.csv` via `bioen_param_lines(..., background_imax={15: …, 16: …}, notes=SPECIES_NOTE, m0=…)`, write the overlay JSON and a README (what the file is, that it is an ARM not production, the two temperatures per species, the labels). `build_overlay(csv, temp_nc)`: `OsmoseConfigReader().read_file(csv)`-style parse (or a 6-line parser: split on `;`, skip `#`, lowercase keys) + `{"module.bioenergetics.enabled": "true", "simulation.bioen.phit.enabled": "true", "simulation.bioen.fo2.enabled": "false", "temperature.filename": str(temp_nc.resolve()), "temperature.varname": "temperature", "temperature.nsteps.year": "24"}`.

- [ ] **Step 3: Run it and inspect**

`PYTHONPATH=. .venv/bin/python scripts/fit_baltic_bioen_params.py --baltic` → table with, per species: growth optimum, engine T_p, Imax, r, c_m, RMS %, W∞ fit vs vBGF, larval ratio, φT(T̄), inflation factor `1/(φT(T̄)(1−m))`. Sanity pins in the script (raise): `phi_t(t_p)==1.0`, argmax within 0.1 °C, `Imax>0`, `r>0`, RMS ≤ 15 %. If a species fails RMS ≤ 15 % (likely candidates: stickleback, sprat — short-lived with fast K), print its curve comparison and widen the fitted age window to ≥ 0.75 yr for that species only, recording it in the README; do not loosen the pin silently.

- [ ] **Step 4: Tests + commit**

```bash
git add scripts/fit_baltic_bioen_params.py data/baltic/scenarios/c3_bioen tests/test_bioen_offline_fit.py
git commit -m "feat(baltic): C3 bioen parameter set fitted offline (T_p solved from growth optima, m at 16 C) + flat scenario overlay"
```

---

### Task 12: Harness — gates C–F, three arms, committed JSON

**Files:**
- Create: `scripts/baltic_c3_bioen_ab.py`
- Test: `tests/test_baltic_c3_harness.py`

**Interfaces:**
- Produces (importable): `ARMS = ("baseline", "bioen", "bioen_plus2C")`, `SEEDS = (42, 123, 7, 999, 2024)`, `N_YEAR = 50`;
  `arm_config(base_cfg, arm, overlay) -> dict`; `gate_c_load_through(arm_cfg, base_nc, builder_recompute, wet) -> None`; `gate_d_structure(arm_cfg, engine_cfg, fit_csv_values) -> None`; `gate_e_zlayer(arm_cfg, seed) -> None`; `gate_f_thermal(engine_cfg, temp_field, habitat_masks) -> dict`; `length_at_age(results, config) -> dict[species, NDArray]`; `run_c3(seeds) -> dict`; results JSON at `docs/diagnostics/baltic_c3_bioen_report.json`.

- [ ] **Step 1: Write the failing gate tests (each gate must FIRE on a synthetic violation)**

```python
# tests/test_baltic_c3_harness.py
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c3ab", ROOT / "scripts" / "baltic_c3_bioen_ab.py")
c3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)


def test_arm_config_overlay_only_on_bioen_arms():
    base = {"simulation.time.nyear": "50", "predation.ingestion.rate.max.sp0": "3.5"}
    ov = {"module.bioenergetics.enabled": "true", "predation.ingestion.rate.max.sp0": "4.7", "temperature.filename": "/x.nc"}
    assert c3.arm_config(base, "baseline", ov) == base
    b = c3.arm_config(base, "bioen", ov)
    assert b["predation.ingestion.rate.max.sp0"] == "4.7" and "temperature.offset" not in b
    p = c3.arm_config(base, "bioen_plus2C", ov)
    assert p["temperature.offset"] == "2.0"
    with pytest.raises(AssertionError, match="bioen-off"):
        c3.arm_config({**base, "module.bioenergetics.enabled": "false"}, "baseline", ov | {"module.bioenergetics.enabled": "false"})


def test_gate_d_fires_on_engine_parsed_mismatch():
    class EC:  # minimal stand-in with the fields Gate D reads
        n_species = 1; bioen_tp = np.array([12.5]); bioen_r = np.array([0.3]); bioen_zlayer = np.array([1])
    fit_vals = {"species.bioen.mobilized.tp.sp0": "12.5", "species.maturity.r.sp0": "0.3", "species.zlayer.sp0": "1"}
    c3.gate_d_structure({"temperature.filename": "/x.nc"}, EC(), fit_vals, expected_zlayer={0: 1})
    bad = dict(fit_vals); bad["species.bioen.mobilized.tp.sp0"] = "20.0"
    with pytest.raises(AssertionError, match="tp"):
        c3.gate_d_structure({"temperature.filename": "/x.nc"}, EC(), bad, expected_zlayer={0: 1})
    with pytest.raises(AssertionError, match="temperature.value"):
        c3.gate_d_structure({"temperature.filename": "/x.nc", "temperature.value": "5"}, EC(), fit_vals, expected_zlayer={0: 1})


def test_gate_c_plus2_is_exact_in_float64():
    raw32 = np.array([[3.7, 8.25], [np.nan, 12.125]], dtype=np.float32)
    base = raw32.astype(np.float64); arm = 1.0 * (raw32.astype(np.float64) + 2.0)
    wet = np.array([[True, True], [False, True]])
    c3.assert_plus2_exact(arm, base, wet)          # engine_arm - engine_base == 2.0 exactly on wet cells
    with pytest.raises(AssertionError):
        c3.assert_plus2_exact(arm + 1e-9, base, wet)


def test_gate_f_direction_is_sign_of_topt_minus_tbar():
    # g_net shifts up under +2 C when T̄ < t_opt, down when T̄ > t_opt
    out = c3.gate_f_direction(t_bar={"a": 5.0, "b": 26.0}, t_opt={"a": 10.0, "b": 15.0},
                              g_base={"a": 1.0, "b": 1.0}, g_plus2={"a": 1.2, "b": 0.9})
    assert out == {"a": True, "b": True}
    with pytest.raises(AssertionError, match="direction"):
        c3.gate_f_direction({"a": 5.0}, {"a": 10.0}, {"a": 1.0}, {"a": 0.8})


def test_length_at_age_bins_use_same_convention_for_both_arms():
    import pandas as pd
    ab = pd.DataFrame({"Time": [49.0, 49.0], "age": [0, 1], "cod": [1e6, 1e5]})
    bb = pd.DataFrame({"Time": [49.0, 49.0], "age": [0, 1], "cod": [1e6 * 1e-6 * 0.5, 1e5 * 1e-6 * 400.0]})
    L = c3.length_from_age_bins(ab, bb, cf=0.0087, b=3.05, species="cod")
    assert L[1] == pytest.approx((400.0 / 0.0087) ** (1 / 3.05))
```

If the in-memory `abundance_by_age()` frame layout differs from the assumed long form (`Time, age, <species>`), read one real frame in the task (`PythonEngine().run_in_memory(cfg, seed=0).abundance_by_age("cod_west").head()`) and adapt `length_from_age_bins` + this test to the actual columns — do not guess.

- [ ] **Step 2: Implement the harness**

Follow `scripts/baltic_c4_salinity_ab.py` structure (`run_c4:266-430`): build `base_cfg` from `osmose_demo("baltic")`, `N_YEAR`, overlay = `json.load(c3_bioen_arm.json)` minus `_meta`; `arm_config(base, arm, overlay)`: baseline → `dict(base)`; bioen → `{**base, **overlay}`; plus2C → the same + `temperature.offset="2.0"`; assert an overlay never lands on a config whose `module.bioenergetics.enabled` is not `"true"` ("bioen-off"). Gates in order, all before any engine run: **Gate A** — load `docs/diagnostics/c3_gate_a_master_baseline.json` and after the baseline runs `check_against_fixture` per seed (violations → AssertionError); **Gate C** — `_load_temperature_data(arm_cfg, None)._data` vs `xr.open_dataset(file)["temperature"].values.astype(float64)` vs the builder's `build(...)` recomputation from the cache (skip the recomputation with a printed note if `--no-recompute`, the cache read takes minutes); for plus2C `assert_plus2_exact(engine_plus2, engine_base, wet)` with `wet = Grid.from_netcdf(grid).ocean_mask` and both arrays compared per layer in float64; wet-cell finite/range check; **Gate D** — frames 24, layers 2, `config.bioen_zlayer` == `SPECIES_ZLAYER`, no `temperature.value`, every `EngineConfig.bioen_*`/`maturity` field per species equals the CSV value (compare `float(csv_value) == float(engine_value)` exactly; the CSV holds `repr` floats); **Gate E** — one `_bioen_step` call at step 12 with `debug_capture` on a state produced by running the bioen arm for 2 steps (use `simulate(...)`'s internals: simplest is to run `PythonEngine().run_in_memory` for 1 year with `simulation.time.nyear=1` and then call `_bioen_step` on a fresh `SchoolState` built from the movement maps — if that is awkward, place 10 synthetic schools per species on random wet cells at step 12 and call `_bioen_step` directly; assert `temp_c[i] == field[12, zlayer[sp], y, x]` for every school and NaN for `is_out`); **Gate F** — per species `phi_t(T_p)==1.0`, argmax `g_net` == `t_opt ± 0.1` (import from `osmose.calibration.bioen_offline`), field φT ∈ (0,1] on wet cells, and `gate_f_direction` using habitat-mean `g_net` at T̄ and T̄+2. Then engine runs per seed × arm (`PythonEngine().run_in_memory`), the Gate A check on the baseline arm, and the REPORTED section: final-decade means/spreads, ratio to `docs/baltic_certification_2026-08-14.md` means (hard-code the nine certified means in a dict with the doc as source) and to the `ENVELOPE` dict imported from `scripts/baltic_stability_certify.py`, persistence, `meanEnetFaced` final-decade mean per species ÷ fitted `g_net` mean (ē/ĝ), realized ration `f = 1 - (1 - ē/ĝ)·(1 − m)` (inverse of decision 7's relation), realized annual ingestion per species (`ingestion` bioen CSV, `bioen_ingestion_by_species` = E_gross; divide by `a·φT` is not available — report E_gross/a as the assimilated-ingestion proxy and label it), `length_from_age_bins` for baseline vs bioen (paired RMS % over ages ≥ 1), seeding diagnostics (count schools with `from_seeding` at the last step is not available from results — instead report the per-species first year with SSB > 0 from `results.ssb()` if present, else skip with a note), plus2C deltas. **Decision rule** (spec §4) evaluated and printed as `STAGE 2: WARRANTED` / `CLOSE BY CHARACTERIZATION (failed: …)`. JSON written to `docs/diagnostics/baltic_c3_bioen_report.json` with seeds, arms, gates, all reported tables, and the verdict. CLI: `--seeds`, `--years`, `--no-recompute`, `--out`.

- [ ] **Step 3: Tests + commit**

`.venv/bin/python -m pytest tests/test_baltic_c3_harness.py -q` → PASS.

```bash
git add scripts/baltic_c3_bioen_ab.py tests/test_baltic_c3_harness.py
git commit -m "feat(baltic): C3 bioen A/B harness -- gates A/C/D/E/F, three arms x five seeds, ē/ĝ and length-at-age instruments, pre-registered decision rule"
```

---

### Task 13: Realistic-config bioen smoke test and run-time measurement

**Files:**
- Create: `tests/test_baltic_c3_bioen_smoke.py` (marked `integration`)
- Modify: `tests/test_baltic_ev_fixture_bioen.py` (docstring pointer: the C3 smoke test is the realistic-config bioen regression; the `baltic_ev` preflight stays self-skipping)

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_baltic_c3_bioen_smoke.py
"""C3 spec §7: the realistic-config bioen regression. Runs the production Baltic config with the C3 overlay,
`population.seeding.year.max = 1` (so every species leaves its seeding window in year 1), long enough for
gonad-derived spawning to carry the populations, and asserts no monotone decay at the end."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"


@pytest.mark.integration
def test_bioen_arm_sustains_populations_past_the_seeding_window():
    cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", Path(tempfile.mkdtemp()))["config_file"])))
    ov = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if not k.startswith("_")}
    cfg.update(ov)
    cfg["simulation.time.nyear"] = "8"
    cfg["population.seeding.year.max"] = "1"
    res = PythonEngine().run_in_memory(cfg, seed=42)
    bio = res.biomass()
    for sp in ("cod_west", "cod_east", "herring", "sprat", "flounder"):
        s = bio[sp].to_numpy(dtype=float)
        assert np.isfinite(s).all() and s[-1] > 0.0, sp
        tail = s[-3:]
        assert not (tail[2] < tail[1] < tail[0] and tail[2] < 0.5 * tail[0]), f"{sp}: monotone collapse in years 6-8"
    # gonad-derived spawning happened after the window closed: SSB > 0 in the last year for every assessed stock
    ssb = res.ssb() if hasattr(res, "ssb") else None
    if ssb is not None:
        for sp in ("cod_west", "cod_east", "herring", "sprat", "flounder"):
            assert ssb[sp].to_numpy(dtype=float)[-1] > 0.0, sp
```

If `population.seeding.year.max` is not the key `config.py:533-551` reads, use the key it reads (`grep -n "seeding" osmose/engine/config.py`).

- [ ] **Step 2: Run it and time a 10-year single-seed bioen run**

`.venv/bin/python -m pytest tests/test_baltic_c3_bioen_smoke.py -q -m integration` → PASS (if it fails on monotone collapse, that is a real finding: check the harness's ē/ĝ and consumption instruments on a 10-yr run before deciding whether Task 11's parameters or a remaining engine defect is the cause — the Gate B PASS from Task 9 bounds the engine side). Then time: a scratchpad script that runs the bioen arm for 10 years at seed 42 and prints seconds per simulated year; record the number (expect 5–10× the production 4 s/yr).

- [ ] **Step 3: Commit**

```bash
git add tests/test_baltic_c3_bioen_smoke.py tests/test_baltic_ev_fixture_bioen.py
git commit -m "test(c3): realistic-config bioen smoke regression past the seeding window; bioen run-time recorded"
```

---

### Task 14: Run the A/B, write the results doc, CLAUDE.md, memory

**Files:**
- Create: `docs/baltic_c3_bioen_stage1_2026-<date>.md`, `docs/diagnostics/baltic_c3_bioen_report.json` (from the run)
- Modify: `CLAUDE.md` (Gotchas), `docs/baltic_temperature_forcing_diagnostic_2026-06-04.md` (a two-line correction pointer at the top: the "NO engine bug" verdict covered the thermal functions only; see the C3 spec §0), `docs/tutorials/fie-on-baltic-cod.md` (one-line pointer under the boom-bust caveat)
- Memory: `/home/razinka/.claude/projects/-home-razinka-osmopy/memory/baltic-c3-bioen-stage1.md` + `MEMORY.md` line

- [ ] **Step 1: Run the harness detached**

```bash
printf '%s\n' '#!/usr/bin/env bash' 'cd /home/razinka/osmopy' 'export PYTHONPATH=.' \
 '.venv/bin/python scripts/baltic_c3_bioen_ab.py > /tmp/claude-1000/-home-razinka-osmopy/f19fe0be-9cc5-4217-979c-2d0a13c87eda/scratchpad/c3_ab.log 2>&1' \
 > /tmp/claude-1000/-home-razinka-osmopy/f19fe0be-9cc5-4217-979c-2d0a13c87eda/scratchpad/run_c3_ab.sh
setsid nohup bash /tmp/claude-1000/-home-razinka-osmopy/f19fe0be-9cc5-4217-979c-2d0a13c87eda/scratchpad/run_c3_ab.sh > /dev/null 2>&1 &
```

Budget: 5 seeds × (1 production run at 3.4 min + 2 bioen runs at the Task 13 rate). Poll the log with a background `until grep -q "STAGE 2\|CLOSE BY\|BLOCKED\|Traceback" …; do sleep 60; done` (re-arm every 10 min). A `BLOCKED` gate stops the stage: fix, re-run.

- [ ] **Step 2: Write the results doc**

Headline = the decision-rule verdict and the three criteria's numbers. Sections: 1 the parity finding (§0 table of the spec, with the Gate B result and the control), 2 Gate A–G evidence (one line each, with the log/JSON pointer), 3 the parameter table (growth optimum → engine T_p, Imax, r, c_m, φT(T̄), inflation factor, labels), 4 the A/B table (final-decade means, spread, ratio to certified means and envelope), 5 instruments (ē/ĝ and f per species; length-at-age paired RMS; consumption ratios; seeding diagnostics), 6 the +2 °C arm, 7 labels (spec §4 list, restated), 8 what Stage 2 would do (or why C3 closes), 9 follow-ups (Numba bioen kernel; fo2 spec; maturity latch; B2 bottom-T swap). Every number from the JSON, none typed by hand.

- [ ] **Step 3: CLAUDE.md gotchas (append under "Gotchas")**

- Bioen runs in Java's **tonnes-per-school** framework (`E_gross/E_maint/E_net` per school; `dw = E_net(1−ρ)/N`); `state.e_net_avg` holds Java's `enet_faced`; starvation runs inside the mortality loop with the previous step's `E_net`; the batched Numba kernels are bypassed under bioen (5–10× slower). Spec §0 table is the contract.
- `species.zlayer.sp{i}` selects the layer of a `(time, z, y, x)` temperature file; `temperature.value` takes precedence over the file (Java); bioen without a temperature source raises.
- Frame conventions differ across the Baltic physical files: O₂ and temperature = month duplicated ×2; salinity = linear `resample_to_24`. Land: O₂ 0.0, salinity NaN, temperature NaN.
- The reader lowercases keys; engine key patterns must be lowercase (`species.bioen.mobilized.tp`, not `.Tp`).
- `predation.ingestion.rate.max.sp{i}` is bioen `Imax` (g·g^−β·yr⁻¹) when bioen is on and the standard rate otherwise (lossy alias); Java 4.3.3 staging must inject `.bioen` keys (`scripts/cross_engine_parity_440.py`).
- A dict overlay never resolves `osmose.configuration.*` includes — flatten (`data/baltic/scenarios/c3_bioen/c3_bioen_arm.json` is flat by construction).

- [ ] **Step 4: Memory + commit + merge**

Memory file (type project): the verdict, the parity finding, Gate B numbers, the traps above, follow-ups; `MEMORY.md` line. Then:

```bash
git add docs/ CLAUDE.md
git commit -m "docs(baltic): C3 bioen Stage 1 results -- <verdict>; parity finding, gates, parameter table, instruments"
git checkout master && git merge --no-ff c3-bioen-stage1 -m "merge: C3 bioen Stage 1 (parity fix, temperature forcing, offline fit, A/B)"
.venv/bin/python -m pytest -q   # full suite on master
git push
```

---

## Self-review (done while writing; re-check at execution)

- Spec coverage: §0 rows → Tasks 3 (budget, enet_faced, ρ), 4 (cap, survivor scaling, starvation, dispatch), 5 (reproduction, egg length), 2 (key case, larval threshold, background Imax), 7 (loader, layers, f_o2, fail-fast, allowlist, meanEnetFaced); §3.3 → Task 10; §3.4 → Tasks 8, 11; §3.5 + §4 gates → Task 12 (A, C, D, E, F), Task 9 (B), Tasks 3–5 (G); §7 integration → Task 13; §3.6/§5 → Task 14. Decision 12 (banner) already done on master. Maturity latch, fo2 activation, Numba bioen kernel: non-goals, recorded in Task 14's follow-ups.
- Placeholders: none — every step has code or an exact command; the two "adapt to the real frame layout" notes (Tasks 12, 13) instruct reading the real structure first rather than guessing.
- Type consistency: `compute_energy_budget(..., enet_faced)` (Task 3) is what `_bioen_step` passes (Task 3 Step 4); `per_fish_ingestion_cap` signature (Task 4 helper) matches the `mortality()` call (Task 4 Step 4d); `bioen_egg_release`/`regulate_recruitment`/`create_egg_schools` (Task 5) match `_bioen_reproduction`'s calls; `_load_temperature_data(raw, config_dir)` (Task 7) is what the harness Gate C calls (Task 12); `bioen_param_lines(..., notes, m0)` (Task 8) matches the Task 8 test and the Task 11 call; `check_against_fixture(fixture, seed, df)` (Task 1) is what Gate A uses (Task 12).
