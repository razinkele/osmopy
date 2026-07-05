# Baltic Percid Thermal Recruitment Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable, temperature-driven per-year recruitment factor for the two Baltic percids (perch `sp4`, pikeperch `sp5`), plus the CMEMS `thetao` data pipeline that feeds it, shipped inert-by-default; test whether percid overshoot is recruitment-driven.

**Architecture:** Mirror the merged RV-gate feature exactly: a sidecar CSV → a fail-fast loader (`_load_thermal_gate`) that applies a logistic temperature response and a mode normalization → precomputed `EngineConfig` fields → a pure per-step helper (`thermal_gate_factor`) → a single guarded block in `reproduction()` (`n_eggs[sp] *= factor[sp]`). The mechanism is built and tested first with a small example sidecar; the CMEMS `thetao` builder comes last so a credential blocker cannot stall the core.

**Tech Stack:** Python 3.12, NumPy, pandas, xarray (CMEMS NetCDF), pytest, ruff. Run everything with `.venv/bin/python`.

## Global Constraints

- Repo root: `/home/razinka/osmopy`. Branch: `baltic-percid-thermal-recruitment-gate` (already created; commit here).
- Run tests: `.venv/bin/python -m pytest`. Lint: `.venv/bin/ruff check osmose/ tests/` AND `.venv/bin/ruff format --check osmose/ tests/`.
- OSMOSE config keys are lowercase dot-separated; species-indexed keys use `sp{idx}`.
- **Inert-by-default and bit-identical when off** is non-negotiable: master switch `reproduction.thermal.gate.enabled` defaults `false`; when off, `_load_thermal_gate` returns `(None, None, 0)` and `reproduction()` skips the block, producing byte-identical output to baseline.
- Percid-only feature. Do NOT touch cod, the RV gate, or the recruitment ceiling. They are cod-only, so there is zero interaction.
- Fail-fast: every invalid configuration raises a clear `ValueError`/`FileNotFoundError` naming the offending key/file. No silent fallback.
- Percid species indices in `data/baltic/baltic_param-species.csv`: `species.name.sp4;perch`, `species.name.sp5;pikeperch`.
- Precedent to imitate line-for-line: `_load_rv_gate` (`osmose/engine/config.py:1082`), the pure helper `osmose/engine/processes/recruitment_gate.py`, the reproduction insertion `osmose/engine/processes/reproduction.py:158-178`, sidecar format `data/baltic/forcing/baltic_rv_gate_series.csv` (`year,spawning_rv`).
- Config-validation: keys built from `f"...sp{sp}"` are auto-captured by the `config_validation` AST walker (same as the RV gate); `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs` must stay warning-free.

---

### Task 1: Pure temperature-response + normalization functions

**Files:**
- Create: `osmose/engine/processes/thermal_gate.py`
- Test: `tests/test_thermal_gate.py`

**Interfaces:**
- Produces: `logistic_response(temp: NDArray[float64], t50: float, slope: float) -> NDArray[float64]` and `normalize_factor(r: NDArray[float64], mode: str, r_ref: float, window_idx: list[int], floor: float) -> NDArray[float64]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_thermal_gate.py
import numpy as np
import pytest
from osmose.engine.processes.thermal_gate import logistic_response, normalize_factor


def test_logistic_is_half_at_t50():
    r = logistic_response(np.array([18.5]), t50=18.5, slope=1.5)
    assert r[0] == pytest.approx(0.5)


def test_logistic_monotone_increasing():
    r = logistic_response(np.array([10.0, 15.0, 18.5, 22.0]), t50=18.5, slope=1.5)
    assert np.all(np.diff(r) > 0)


def test_thermal_cap_clips_to_unit_and_floor():
    # r/r_ref: cold years << r_ref -> near 0 (floored), warm years >= r_ref -> clipped to 1
    r = np.array([0.01, 0.5, 0.99])
    out = normalize_factor(r, mode="thermal_cap", r_ref=0.5, window_idx=[0, 1, 2], floor=0.05)
    assert out[0] == pytest.approx(0.05)          # floored
    assert out[1] == pytest.approx(1.0)           # 0.5/0.5 = 1.0 (>= cap)
    assert out[2] == pytest.approx(1.0)           # clipped to 1
    assert np.all(out <= 1.0) and np.all(out >= 0.05)


def test_mean_preserving_has_unit_mean_over_window():
    r = np.array([0.2, 0.4, 0.6, 0.8])
    out = normalize_factor(r, mode="mean_preserving", r_ref=0.0, window_idx=[0, 1, 2, 3], floor=0.0)
    assert np.mean(out) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_thermal_gate.py -q`
Expected: FAIL with `ModuleNotFoundError: osmose.engine.processes.thermal_gate`.

- [ ] **Step 3: Write the module with the pure functions**

```python
# osmose/engine/processes/thermal_gate.py
"""Percid thermal recruitment gate — pure helpers.

Engine-state-free. The response curve + mode normalization are applied in the
config loader (osmose/engine/config.py:_load_thermal_gate); the per-step
multiplier is read back by thermal_gate_factor (added in a later task). Percid
year-class strength is temperature-gated (Pekcan-Hekim et al. 2011).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from osmose.engine.config import EngineConfig


def logistic_response(temp: NDArray[np.float64], t50: float, slope: float) -> NDArray[np.float64]:
    """Saturating year-class response to summer temperature, in (0, 1).

    0.5 at temp == t50; rises over ~slope degrees. Logistic (not linear)
    encodes that strong percid year-classes are the exception: cool years
    mostly fail, warm years above threshold succeed.
    """
    return 1.0 / (1.0 + np.exp(-(temp - t50) / slope))


def normalize_factor(
    r: NDArray[np.float64],
    mode: str,
    r_ref: float,
    window_idx: list[int],
    floor: float,
) -> NDArray[np.float64]:
    """Turn a per-year response into a per-year egg multiplier.

    thermal_cap (mean-reducing): clip(r / r_ref, 0, 1) — most years < 1.
    mean_preserving (realism only): r / mean(r over the sampled model years).
    Both then floored at ``floor``.
    """
    if mode == "thermal_cap":
        factor = np.clip(r / r_ref, 0.0, 1.0)
    elif mode == "mean_preserving":
        denom = float(np.mean(r[window_idx]))
        if denom == 0.0:
            raise ValueError("mean_preserving denominator is 0 over the run window.")
        factor = r / denom
    else:
        raise ValueError(f"unknown thermal gate mode: {mode!r}")
    return np.maximum(factor, floor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_thermal_gate.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/thermal_gate.py tests/test_thermal_gate.py
git commit -m "feat: percid thermal-gate pure response + normalization helpers"
```

---

### Task 2: Pure per-step helper `thermal_gate_factor`

**Files:**
- Modify: `osmose/engine/processes/thermal_gate.py` (append function)
- Test: `tests/test_thermal_gate.py` (append)

**Interfaces:**
- Consumes: reads `config.n_species`, `config.n_dt_per_year`, `config.thermal_gate_factor_by_index` (`(n_years, n_species)` | None), `config.thermal_gate_enabled` (`(n_species,)` bool | None), `config.thermal_gate_offset` (int). These `EngineConfig` fields are formally added in Task 3; tests here use a lightweight stub with those attributes.
- Produces: `thermal_gate_factor(config: EngineConfig, step: int) -> NDArray[float64]` — per-species egg multiplier, 1.0 for disabled species / when off.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_thermal_gate.py
from types import SimpleNamespace
from osmose.engine.processes.thermal_gate import thermal_gate_factor


def _stub(factor, enabled, offset=0, n_species=6, n_dt=24):
    return SimpleNamespace(
        n_species=n_species,
        n_dt_per_year=n_dt,
        thermal_gate_factor_by_index=factor,
        thermal_gate_enabled=enabled,
        thermal_gate_offset=offset,
    )


def test_factor_all_ones_when_off():
    out = thermal_gate_factor(_stub(None, None), step=0)
    assert np.array_equal(out, np.ones(6))


def test_factor_applies_only_to_enabled_species_for_current_year():
    factor = np.array([[1.0, 1.0, 1.0, 1.0, 0.3, 0.7],   # year 0
                       [1.0, 1.0, 1.0, 1.0, 0.9, 0.8]])  # year 1
    enabled = np.array([False, False, False, False, True, True])
    out = thermal_gate_factor(_stub(factor, enabled), step=0)          # year 0
    assert out[4] == pytest.approx(0.3) and out[5] == pytest.approx(0.7)
    assert out[0] == 1.0 and out[3] == 1.0
    out1 = thermal_gate_factor(_stub(factor, enabled), step=24)        # year 1 (24 dt/yr)
    assert out1[4] == pytest.approx(0.9)


def test_factor_year_index_wraps_around_series():
    factor = np.array([[1, 1, 1, 1, 0.3, 0.3], [1, 1, 1, 1, 0.9, 0.9]], dtype=float)
    enabled = np.array([False, False, False, False, True, True])
    # run year 2 with a 2-row series -> wraps to index 0
    out = thermal_gate_factor(_stub(factor, enabled), step=48)
    assert out[4] == pytest.approx(0.3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_thermal_gate.py -q -k factor`
Expected: FAIL with `ImportError: cannot import name 'thermal_gate_factor'`.

- [ ] **Step 3: Append the helper**

```python
# append to osmose/engine/processes/thermal_gate.py
def thermal_gate_factor(config: "EngineConfig", step: int) -> NDArray[np.float64]:
    """Per-species egg-production multiplier for this timestep.

    1.0 for every species when the gate is off or the species is disabled;
    otherwise the current model year's per-species factor (constant within a
    model year), with the series index wrapping if the run outlasts the series.
    """
    out = np.ones(config.n_species, dtype=np.float64)
    factor = config.thermal_gate_factor_by_index
    if factor is None:
        return out
    year = step // config.n_dt_per_year
    idx = (config.thermal_gate_offset + year) % factor.shape[0]
    mask = config.thermal_gate_enabled
    out[mask] = factor[idx, mask]
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_thermal_gate.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/thermal_gate.py tests/test_thermal_gate.py
git commit -m "feat: thermal_gate_factor per-step helper"
```

---

### Task 3: Config loader `_load_thermal_gate` + EngineConfig fields + from_dict wiring

**Files:**
- Modify: `osmose/engine/config.py` (add loader after `_load_rv_gate` ~line 1145; add 3 `EngineConfig` fields near the rv_gate fields ~line 1510; call + pass in `from_dict` ~lines 2246 and 2322)
- Test: `tests/test_thermal_gate_loader.py`
- Create (fixture): `tests/data/percid_thermal_ok.csv`

**Interfaces:**
- Consumes: `_require_file`, `_cfg_dir` (existing helpers in `config.py`); `logistic_response`, `normalize_factor` (Task 1).
- Produces: `_load_thermal_gate(cfg: dict[str,str], n_species: int, n_dt_per_year: int, n_year: int) -> tuple[NDArray[float64] | None, NDArray[bool_] | None, int]` returning `(factor_by_index (n_years, n_species), enabled_mask (n_species,), offset)` or `(None, None, 0)` when off. New `EngineConfig` fields: `thermal_gate_factor_by_index`, `thermal_gate_enabled`, `thermal_gate_offset`.

- [ ] **Step 1: Create the fixture sidecar**

```csv
# tests/data/percid_thermal_ok.csv
year,temp_sp4,temp_sp5
2000,15.0,16.0
2001,17.0,18.0
2002,19.0,20.0
2003,21.0,22.0
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_thermal_gate_loader.py
import numpy as np
import pytest
from pathlib import Path
from osmose.engine.config import _load_thermal_gate

FIX = Path(__file__).parent / "data" / "percid_thermal_ok.csv"


def _cfg(**over):
    base = {
        "reproduction.thermal.gate.enabled": "true",
        "reproduction.thermal.gate.series.file": str(FIX),
        "reproduction.thermal.gate.species.enabled.sp4": "true",
        "reproduction.thermal.gate.species.enabled.sp5": "true",
        "reproduction.thermal.gate.mode": "thermal_cap",
    }
    base.update(over)
    return base


def test_off_returns_none():
    f, e, o = _load_thermal_gate({"reproduction.thermal.gate.enabled": "false"}, 6, 24, 4)
    assert f is None and e is None and o == 0


def test_thermal_cap_shapes_and_enabled_mask():
    f, e, o = _load_thermal_gate(_cfg(), n_species=6, n_dt_per_year=24, n_year=4)
    assert f.shape == (4, 6)
    assert list(np.where(e)[0]) == [4, 5]
    # disabled species column stays 1.0
    assert np.allclose(f[:, 0], 1.0)
    # warm final year (21/22 C) -> factor near 1 for both percids; cold first year -> < 1
    assert f[3, 4] == pytest.approx(1.0, abs=0.05)
    assert f[0, 4] < f[3, 4]


def test_mean_preserving_unit_mean():
    f, e, o = _load_thermal_gate(_cfg(**{"reproduction.thermal.gate.mode": "mean_preserving"}), 6, 24, 4)
    assert np.mean(f[:, 4]) == pytest.approx(1.0)


def test_missing_file_raises():
    with pytest.raises(ValueError, match="series.file is empty"):
        _load_thermal_gate(_cfg(**{"reproduction.thermal.gate.series.file": ""}), 6, 24, 4)


def test_no_species_enabled_raises():
    cfg = _cfg(**{"reproduction.thermal.gate.species.enabled.sp4": "false",
                  "reproduction.thermal.gate.species.enabled.sp5": "false"})
    with pytest.raises(ValueError, match="no species enabled"):
        _load_thermal_gate(cfg, 6, 24, 4)


def test_missing_species_column_raises():
    # enable sp3, which has no temp_sp3 column in the fixture
    cfg = _cfg(**{"reproduction.thermal.gate.species.enabled.sp3": "true"})
    with pytest.raises(ValueError, match="temp_sp3"):
        _load_thermal_gate(cfg, 6, 24, 4)


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        _load_thermal_gate(_cfg(**{"reproduction.thermal.gate.mode": "bogus"}), 6, 24, 4)


def test_bad_floor_raises():
    with pytest.raises(ValueError, match="floor"):
        _load_thermal_gate(_cfg(**{"reproduction.thermal.gate.floor": "1.5"}), 6, 24, 4)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_thermal_gate_loader.py -q`
Expected: FAIL with `ImportError: cannot import name '_load_thermal_gate'`.

- [ ] **Step 4: Add the loader** (insert immediately after `_load_rv_gate`, before `_load_salinity_gate`, in `osmose/engine/config.py`)

```python
def _load_thermal_gate(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int, n_year: int
) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None, int]:
    """Load the percid thermal recruitment gate (spec 2026-07-05).

    Returns (factor_by_index, enabled_mask, offset). factor_by_index has shape
    (n_years, n_species) with the logistic response + mode already applied;
    columns for disabled species are 1.0. All three are (None, None, 0) when the
    master switch is off. Raises a clear error on any invalid configuration.
    """
    from osmose.engine.processes.thermal_gate import logistic_response, normalize_factor

    if cfg.get("reproduction.thermal.gate.enabled", "false").lower() != "true":
        return None, None, 0

    file_key = cfg.get("reproduction.thermal.gate.series.file", "")
    if not file_key:
        raise ValueError("Thermal gate enabled but reproduction.thermal.gate.series.file is empty.")
    path = _require_file(file_key, _cfg_dir(cfg), "reproduction.thermal.gate.series.file")
    df = pd.read_csv(path)
    if df.shape[0] == 0 or "year" not in df.columns:
        raise ValueError(f"Thermal gate series {path} has no data rows or missing 'year' column.")
    years = df["year"].to_numpy()
    first_year = int(years[0])
    if not np.array_equal(years, np.arange(first_year, first_year + len(years))):
        raise ValueError(f"Thermal gate series {path} years must be contiguous and ascending.")
    n_years = len(years)

    enabled = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        if cfg.get(f"reproduction.thermal.gate.species.enabled.sp{sp}", "false").lower() == "true":
            enabled[sp] = True
    if not enabled.any():
        raise ValueError(
            "Thermal gate enabled but no species enabled "
            "(reproduction.thermal.gate.species.enabled.sp{idx})."
        )

    mode = cfg.get("reproduction.thermal.gate.mode", "thermal_cap")
    if mode not in ("thermal_cap", "mean_preserving"):
        raise ValueError(f"unknown reproduction.thermal.gate.mode: {mode!r}")
    floor = float(cfg.get("reproduction.thermal.gate.floor", "0.0"))
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"reproduction.thermal.gate.floor must be in [0,1], got {floor}.")
    start_year = int(cfg.get("reproduction.thermal.gate.start.year", str(first_year)))
    offset = start_year - first_year
    window_idx = [(offset + y) % n_years for y in range(n_year)]

    factor = np.ones((n_years, n_species), dtype=np.float64)
    for sp in range(n_species):
        if not enabled[sp]:
            continue
        col = f"temp_sp{sp}"
        if col not in df.columns:
            raise ValueError(f"Thermal gate series {path} missing column {col!r} for enabled sp{sp}.")
        temp = df[col].to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(temp)) or np.any(temp < -2.0) or np.any(temp > 30.0):
            raise ValueError(
                f"Thermal gate series {path} column {col} has NaN or out-of-range (-2..30 C) values."
            )
        t50 = float(cfg.get(f"reproduction.thermal.gate.t50.sp{sp}", "18.5"))
        slope = float(cfg.get(f"reproduction.thermal.gate.slope.sp{sp}", "1.5"))
        if slope <= 0.0:
            raise ValueError(f"reproduction.thermal.gate.slope.sp{sp} must be > 0, got {slope}.")
        r = logistic_response(temp, t50, slope)
        if mode == "thermal_cap":
            tref = float(cfg.get(f"reproduction.thermal.gate.tref.sp{sp}", "20.0"))
            r_ref = float(logistic_response(np.array([tref]), t50, slope)[0])
            if r_ref <= 0.0:
                raise ValueError(f"thermal_cap r_ref for sp{sp} is 0 (check tref/t50/slope).")
        else:
            r_ref = 0.0
        factor[:, sp] = normalize_factor(r, mode, r_ref, window_idx, floor)

    return factor.astype(np.float64), enabled, offset
```

- [ ] **Step 5: Add the three `EngineConfig` fields** (immediately after the `rv_gate_offset` field, ~line 1512)

```python
    # Percid thermal recruitment gate (all None/0 when disabled)
    thermal_gate_factor_by_index: NDArray[np.float64] | None  # (n_years, n_species)
    thermal_gate_enabled: NDArray[np.bool_] | None  # (n_species,) per-species enable mask
    thermal_gate_offset: int  # start_year - first_year
```

- [ ] **Step 6: Wire into `from_dict`** — add the call next to the `_load_rv_gate` call (~line 2246). NOTE: the locals in `from_dict` are named `n_sp`, `n_dt`, `n_yr` (see the `_load_rv_gate(cfg, n_sp, n_dt, n_yr)` precedent at `config.py:2246-2248`) — use those exact names, NOT `n_species`/`n_dt_per_year`/`n_year` (which are undefined there and would raise `NameError` on every `EngineConfig.from_dict`, including the inert default path):

```python
        thermal_gate_factor_by_index, thermal_gate_enabled, thermal_gate_offset = _load_thermal_gate(
            cfg, n_sp, n_dt, n_yr
        )
```

and pass them into the `EngineConfig(...)` construction next to the `rv_gate_*` args (~line 2322):

```python
            thermal_gate_factor_by_index=thermal_gate_factor_by_index,
            thermal_gate_enabled=thermal_gate_enabled,
            thermal_gate_offset=thermal_gate_offset,
```

- [ ] **Step 7: Register the 9 config keys in the schema** (`osmose/schema/species.py`) — both siblings did this (RV gate ~lines 442-511, ceiling ~512-540) so the config UI/tooling can see the keys; without it the gate is invisible in the UI. Insert immediately before the closing `]` of the field list (right after the ceiling `species.enabled.sp{idx}` field):

```python
    # ── Recruitment: percid thermal gate (Pekcan-Hekim et al. 2011) ─────────
    OsmoseField(
        key_pattern="reproduction.thermal.gate.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description=(
            "Master switch for the percid thermal recruitment gate. When false "
            "the gate is inert and output is bit-identical."
        ),
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.thermal.gate.mode",
        param_type=ParamType.ENUM,
        default="thermal_cap",
        choices=["thermal_cap", "mean_preserving"],
        description=(
            "Thermal gate mode. 'thermal_cap' clips clip(r(T)/r(tref),0,1) "
            "(mean-reducing; the overshoot-damping mode). 'mean_preserving' "
            "normalises to mean 1 over the run window (realism only)."
        ),
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.thermal.gate.series.file",
        param_type=ParamType.FILE_PATH,
        default="",
        description="CSV of per-year per-species summer SST (year,temp_sp{idx},...).",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.thermal.gate.floor",
        param_type=ParamType.FLOAT,
        default=0.0,
        min_val=0.0,
        max_val=1.0,
        description="Optional lower bound on the gate factor (sensitivity knob).",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.thermal.gate.start.year",
        param_type=ParamType.INT,
        default=1993,
        min_val=0,
        max_val=3000,
        description="Real calendar year that model year 0 maps to for the thermal series.",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.thermal.gate.species.enabled.sp{idx}",
        param_type=ParamType.BOOL,
        default=False,
        description="Per-species enable for the thermal gate (percids only for Baltic).",
        category="reproduction",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.thermal.gate.t50.sp{idx}",
        param_type=ParamType.FLOAT,
        default=18.5,
        min_val=0.0,
        max_val=40.0,
        description="Logistic midpoint temperature (C) of the year-class response, per species.",
        category="reproduction",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.thermal.gate.slope.sp{idx}",
        param_type=ParamType.FLOAT,
        default=1.5,
        min_val=1e-9,
        max_val=40.0,
        description="Logistic slope (C) of the year-class response, per species.",
        category="reproduction",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.thermal.gate.tref.sp{idx}",
        param_type=ParamType.FLOAT,
        default=20.0,
        min_val=0.0,
        max_val=40.0,
        description="Reference temperature (C) at which thermal_cap saturates to 1.0, per species.",
        category="reproduction",
        indexed=True,
        required=False,
    ),
```

- [ ] **Step 8: Add a validation test that actually exercises the thermal keys** (the bundled-config test does NOT — inert configs contain zero `thermal.gate.*` keys, so it cannot evidence auto-capture). Append to `tests/test_thermal_gate_loader.py`:

```python
def test_thermal_keys_are_recognized_by_config_validation():
    """The thermal keys must be recognized once registered in the schema (Step 7)
    — i.e. config_validation reports zero unknown keys for them. Real API:
    config_validation.validate(cfg, mode) -> list[UnknownKey]
    (see tests/test_engine_config_validation.py:355 for the same idiom)."""
    from osmose.engine import config_validation as cv
    keys = {
        "reproduction.thermal.gate.enabled": "true",
        "reproduction.thermal.gate.series.file": str(FIX),
        "reproduction.thermal.gate.mode": "thermal_cap",
        "reproduction.thermal.gate.floor": "0.0",
        "reproduction.thermal.gate.start.year": "2000",
        "reproduction.thermal.gate.species.enabled.sp4": "true",
        "reproduction.thermal.gate.species.enabled.sp5": "true",
        "reproduction.thermal.gate.t50.sp4": "18.5",
        "reproduction.thermal.gate.slope.sp4": "1.5",
        "reproduction.thermal.gate.tref.sp4": "20.0",
    }
    assert cv.validate(keys, mode="error") == []  # zero unknown keys
```

> NOTE to implementer: this test depends on Step 7's schema registration — the registered `OsmoseField`s feed `build_known_keys` (`config_validation.py:432-434`), so `validate(..., mode="error")` returns `[]`. If it doesn't, a key is mis-spelled between Step 7 and here.

- [ ] **Step 9: Run loader + validation tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_thermal_gate_loader.py tests/test_engine_config_validation.py -q`
Expected: PASS (loader tests + the new auto-capture test pass; `test_from_dict_warn_mode_clean_on_example_configs` stays warning-free).

- [ ] **Step 10: Commit**

```bash
git add osmose/engine/config.py osmose/schema/species.py tests/test_thermal_gate_loader.py tests/data/percid_thermal_ok.csv
git commit -m "feat: _load_thermal_gate loader + EngineConfig fields + from_dict + schema keys"
```

---

### Task 4: Wire the gate into `reproduction()` (inert-by-default + integration)

**Files:**
- Modify: `osmose/engine/processes/reproduction.py` (add a block after the recruitment-ceiling block, ~line 178)
- Test: `tests/test_reproduction_thermal_gate.py`

**Interfaces:**
- Consumes: `thermal_gate_factor` (Task 2); `config.thermal_gate_factor_by_index`, `config.thermal_gate_enabled` (Task 3); the existing `n_eggs`, `seeded_this_step`, `n_sp`, `step` locals in `reproduction()`.
- Produces: percid `n_eggs[sp]` multiplied by the per-step gate factor when enabled and not a seeding step.

- [ ] **Step 1: Write the failing integration tests**

```python
# tests/test_reproduction_thermal_gate.py
"""Gate is inert by default (bit-identical) and, when on, deterministically
reduces percid recruitment. Uses the bundled Baltic config and the real engine
API (same as scripts/baltic_recruitment_ceiling_diagnostic.py)."""
from pathlib import Path

import numpy as np

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

# Baltic entry config = the all-parameters file (same glob the salinity builder uses).
BALTIC = sorted(Path("data/baltic").glob("*all-parameters*.csv"))[0]
THERMAL = Path("tests/data/percid_thermal_ok.csv").resolve()
DET = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}


def _series(overrides):
    base = dict(OsmoseConfigReader().read(str(BALTIC)))
    base.update(DET)
    base["simulation.time.nyear"] = "6"  # shrink the in-suite Baltic run (RV-gate test does the same)
    base.update(overrides)
    # biomass(): wide frame keyed by species NAME; enable keys use species INDEX.
    return PythonEngine().run_in_memory(base, seed=0).biomass()


def test_gate_off_is_bit_identical_to_baseline():
    base = _series({})
    off = _series({"reproduction.thermal.gate.enabled": "false"})
    np.testing.assert_array_equal(base.to_numpy(), off.to_numpy())  # convention: readable diff, NaN-equal


def _rel_change(off, on, sp):
    a, b = off[sp].to_numpy(), on[sp].to_numpy()
    d = float(np.abs(a).sum())
    return float(np.abs(b - a).sum()) / d if d else 0.0


def test_gate_on_changes_percids_which_dominate_cod():
    off = _series({})
    on = _series({
        "reproduction.thermal.gate.enabled": "true",
        "reproduction.thermal.gate.series.file": str(THERMAL),
        "reproduction.thermal.gate.mode": "thermal_cap",
        "reproduction.thermal.gate.species.enabled.sp4": "true",
        "reproduction.thermal.gate.species.enabled.sp5": "true",
    })
    # Percids are the PRIMARY effect (the block fired). Cod is NOT asserted
    # bit-identical: cod eats perch (0.15) & pikeperch (0.10), pikeperch eats cod
    # (0.05), they share prey, and changed percid survival desyncs the shared
    # mortality RNG — so cod legitimately shifts. This mirrors the RV-gate test
    # test_gate_on_changes_cod_and_cod_dominates (tests/test_rv_recruitment_gate.py).
    assert _rel_change(off, on, "perch") > 0.02
    assert _rel_change(off, on, "pikeperch") > 0.02
    assert _rel_change(off, on, "perch") > _rel_change(off, on, "cod")
    assert _rel_change(off, on, "pikeperch") > _rel_change(off, on, "cod")
```

> NOTE to implementer: species names (`perch`, `pikeperch`, `cod`) come from `data/baltic/baltic_param-species.csv`. Do NOT assert cod bit-identity — cod is trophically coupled to both percids (see `data/baltic/predation-accessibility.csv`) so it legitimately changes; the contract is "percids change and dominate cod's secondary change." The `nyear=6` override keeps the run cheap; the 4-row fixture wraps and cool years still make `thermal_cap` bite, so the signal survives.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_reproduction_thermal_gate.py -q`
Expected: FAIL — `test_gate_on_changes_percids_which_dominate_cod` fails because the gate block does not exist yet (off == on, so `_rel_change` is 0 for the percids). `test_gate_off_is_bit_identical_to_baseline` passes.

- [ ] **Step 3: Add the gate block in `reproduction()`** (immediately after the recruitment-ceiling block that ends ~line 178)

```python
    # Percid thermal recruitment gate (McGregor-style per-year factor; spec 2026-07-05).
    # Inert unless enabled. Percid-only; independent of the cod-only RV gate/ceiling.
    if config.thermal_gate_factor_by_index is not None:
        from osmose.engine.processes.thermal_gate import thermal_gate_factor

        assert config.thermal_gate_enabled is not None  # set together in _load_thermal_gate
        tgate = thermal_gate_factor(config, step)
        for sp in range(n_sp):
            if config.thermal_gate_enabled[sp] and not seeded_this_step[sp]:
                n_eggs[sp] *= tgate[sp]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_reproduction_thermal_gate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the broader engine + parity suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/ -q -k "reproduction or parity or config"`
Expected: PASS (no regressions; inert-by-default holds).

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/reproduction.py tests/test_reproduction_thermal_gate.py
git commit -m "feat: apply percid thermal gate in reproduction() (inert by default)"
```

---

### Task 5: CMEMS `thetao` → summer-SST-per-year builder

**Files:**
- Create: `osmose/forcing/percid_thermal.py` (pure builder function)
- Create: `scripts/build_percid_thermal_series.py` (download + driver → sidecar CSV)
- Test: `tests/test_percid_thermal_builder.py`

**Interfaces:**
- Produces: `summer_sst_by_year(temp_tyx, times_year, times_month, mask_yx, months) -> (years_sorted, mean_temp_per_year)` averaging masked surface cells over the selected months; and `load_thetao_surface(dl, grid) -> (temp_tyx, times_year, times_month)` which discovers CMEMS thetao files in `dl`, raises `FileNotFoundError` loudly when none exist (spec §7), and regrids the surface layer. Both live in the importable module so the loud-failure path is unit-testable without CMEMS.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_percid_thermal_builder.py
import numpy as np
import pytest
from osmose.forcing.percid_thermal import summer_sst_by_year


def test_averages_masked_cells_over_selected_months():
    # 2 years x months 6,7,8 ; 2x2 grid ; mask selects one cell
    temp = np.zeros((6, 2, 2))
    times_year = np.array([2000, 2000, 2000, 2001, 2001, 2001])
    times_month = np.array([6, 7, 8, 6, 7, 8])
    temp[:, 0, 0] = [10, 20, 30, 40, 50, 60]  # only this cell is unmasked
    mask = np.array([[True, False], [False, False]])
    years, means = summer_sst_by_year(temp, times_year, times_month, mask, months=(6, 7))
    assert list(years) == [2000, 2001]
    assert means[0] == pytest.approx(15.0)   # mean(10,20)
    assert means[1] == pytest.approx(45.0)   # mean(40,50)


def test_ignores_nan_ocean_fill():
    temp = np.full((2, 1, 2), np.nan)
    temp[:, 0, 0] = [12.0, 14.0]
    times_year = np.array([2000, 2000])
    times_month = np.array([6, 7])
    mask = np.array([[True, True]])
    years, means = summer_sst_by_year(temp, times_year, times_month, mask, months=(6, 7))
    assert means[0] == pytest.approx(13.0)   # nanmean over the one valid cell across 2 months


def test_load_thetao_surface_raises_loudly_when_no_files(tmp_path):
    # spec §7/§8: absent thetao -> loud FileNotFoundError, never a synthetic field.
    # (grid unused before the file check, so None is fine here.)
    from osmose.forcing.percid_thermal import load_thetao_surface
    with pytest.raises(FileNotFoundError, match="thetao"):
        load_thetao_surface(tmp_path, grid=None)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_percid_thermal_builder.py -q`
Expected: FAIL with `ModuleNotFoundError: osmose.forcing.percid_thermal`.

- [ ] **Step 3: Write the pure builder function**

```python
# osmose/forcing/percid_thermal.py
"""Build a per-year percid summer surface-temperature index from CMEMS thetao.

The pure core (summer_sst_by_year) is grid/data-source agnostic and unit-tested
with synthetic arrays. load_thetao_surface discovers the CMEMS baltic_phy
monthly-reanalysis thetao files and regrids the surface layer, raising loudly
when none exist. The script scripts/build_percid_thermal_series.py wires these
to the percid habitat masks and writes the sidecar CSV.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def summer_sst_by_year(
    temp_tyx: NDArray[np.float64],
    times_year: NDArray[np.int_],
    times_month: NDArray[np.int_],
    mask_yx: NDArray[np.bool_],
    months: tuple[int, ...],
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Mean surface temperature over habitat cells for the given summer months.

    Returns (years_sorted, mean_per_year). NaN ocean-fill is ignored via nanmean.
    """
    sel_month = np.isin(times_month, months)
    years_sorted = np.array(sorted(set(times_year[sel_month].tolist())), dtype=int)
    means = np.empty(years_sorted.shape[0], dtype=np.float64)
    for i, yr in enumerate(years_sorted):
        sel = sel_month & (times_year == yr)
        block = temp_tyx[sel][:, mask_yx]  # (n_selected_months, n_masked_cells)
        means[i] = float(np.nanmean(block))
    return years_sorted, means


def load_thetao_surface(dl, grid):
    """Load surface thetao from the CMEMS files in `dl`, regridded to `grid`.

    Returns (temp_tyx, times_year, times_month). Fails loudly with
    FileNotFoundError when no thetao files are present — NEVER substitutes a
    synthetic field (spec §7). The file check precedes the xarray/grid imports so
    the loud-failure path is testable without CMEMS access or those deps.
    """
    files = sorted(dl.glob("baltic_phy_monthly_reanalysis_thetao_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"no thetao files under {dl}. Download with "
            "scripts/download_baltic_rv_forcing.py --vars thetao --depth-min 0 --depth-max 5 "
            "(needs CMEMS credentials — Risk R1)."
        )
    import xarray as xr

    from osmose.forcing.grid import get_coords, regrid

    slices, yrs, mons = [], [], []
    for f in files:
        ds = xr.open_dataset(f)
        theta = ds["thetao"].values  # (12, nlev, nlat, nlon) or (12, nlat, nlon)
        surf = theta[:, 0] if theta.ndim == 4 else theta  # surface level
        src_lat, src_lon = get_coords(ds)
        for m in range(surf.shape[0]):
            slices.append(regrid(surf[m][None], src_lat, src_lon, grid)[0])  # (ny, nx)
            yrs.append(int(str(f.stem).split("_")[-1][:4]))
            mons.append(m + 1)
        ds.close()
    return np.stack(slices), np.array(yrs), np.array(mons)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_percid_thermal_builder.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5a: Add `thetao` to the CMEMS download script** (required — `scripts/download_baltic_rv_forcing.py`'s `--vars` uses `choices=list(FIELDS)`, so `--vars thetao` is rejected until `FIELDS` gains the entry). Add to the `FIELDS` dict (`download_baltic_rv_forcing.py:42-44`):

```python
    "thetao": {"dataset_id": "cmems_mod_bal_phy_my_P1M-m", "tag": "phy_monthly_reanalysis"},
```

`--depth-min 0` already works (`--depth-min` is `type=float`). Real download (credential-gated, Risk R1): `PYTHONPATH=. .venv/bin/python scripts/download_baltic_rv_forcing.py --vars thetao --depth-min 0 --depth-max 5 --start 1993 --end 2010`.

- [ ] **Step 5b: Write the driver script** (habitat mask + summer-SST + write sidecar). Reuses the salinity builder's CMEMS access + grid/mask loading.

> CAUTION (orientation): `_load_spatial_csv` returns a `np.flipud`'d array whereas `regrid` output follows `target_coords` orientation. Before the real run, verify the habitat mask and the regridded `temp_tyx` share the same row orientation (add a one-cell sanity check: a known warm coastal cell should read a plausibly warm value). Confirm against how `scripts/build_baltic_salinity_forcing.py` reconciles mask vs regridded field.

```python
# scripts/build_percid_thermal_series.py
"""Build the percid thermal sidecar from CMEMS thetao (surface).

Reuses the salinity pipeline exactly (scripts/build_baltic_salinity_forcing.py):
thetao lives in the SAME product as so (cmems_mod_bal_phy_my_P1M-m), so download
it with scripts/download_baltic_rv_forcing.py extended to accept `thetao`
(--vars thetao --depth-min 0 --depth-max 5), producing
data/cmems_cache/cmems_downloads/baltic_phy_monthly_reanalysis_thetao_*.nc.

Per-species summer window: perch (sp4) = (6, 7); pikeperch (sp5) = (7, 8).
Habitat mask per species = cells with nonzero occupancy in its movement map.
Fails loudly if thetao files are absent (Risk R1) — never substitutes a field.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from osmose.config import OsmoseConfigReader
from osmose.engine.config import _load_spatial_csv
from osmose.forcing.percid_thermal import load_thetao_surface, summer_sst_by_year
from osmose.maps.builder import GridSpec

SPECIES = {4: (6, 7), 5: (7, 8)}  # species index -> summer months
ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "data" / "cmems_cache" / "cmems_downloads"
OUT = ROOT / "data" / "baltic" / "forcing" / "baltic_percid_thermal_series.csv"


def _habitat_mask(cfg: dict[str, str], sp: int, cfg_dir: Path) -> np.ndarray:
    """Union of cells with nonzero occupancy across species sp's movement maps.

    Real Baltic movement-map keys (data/baltic/baltic_param-movement.csv) are
    `movement.species.map{n}` and `movement.file.map{n}`, and the species value
    is the NAME (e.g. 'perch'), not the index. So resolve sp -> name first.
    """
    sp_name = cfg[f"species.name.sp{sp}"].strip()
    mask = None
    n = 0
    while True:
        name = cfg.get(f"movement.species.map{n}", None)
        map_key = cfg.get(f"movement.file.map{n}", "")
        if map_key == "":
            break
        if name is not None and name.strip() == sp_name:
            arr = _load_spatial_csv(cfg_dir / map_key)  # NB: _load_spatial_csv np.flipud's the CSV
            m = arr > 0
            mask = m if mask is None else (mask | m)
        n += 1
    if mask is None:
        raise ValueError(f"no movement map found for {sp_name} (sp{sp}); check movement.species.map* keys")
    return mask


def main() -> int:
    cfg_path = sorted((ROOT / "data" / "baltic").glob("*all-parameters*.csv"))[0]
    cfg = OsmoseConfigReader().read(str(cfg_path))
    cfg_dir = cfg_path.parent
    grid = GridSpec.from_config(cfg)  # confirm the GridSpec constructor used by the salinity builder
    temp_tyx, ty, tm = load_thetao_surface(DL, grid)  # loud FileNotFoundError if thetao absent

    per_sp = {}
    for sp, months in SPECIES.items():
        mask = _habitat_mask(cfg, sp, cfg_dir)
        years, means = summer_sst_by_year(temp_tyx, ty, tm, mask, months)
        per_sp[sp] = pd.Series(means, index=years)

    common = sorted(set.intersection(*[set(s.index) for s in per_sp.values()]))
    if list(common) != list(range(common[0], common[0] + len(common))):
        raise ValueError(f"thermal series years not contiguous: {common}")
    df = pd.DataFrame({"year": common})
    for sp in SPECIES:
        df[f"temp_sp{sp}"] = [float(per_sp[sp][y]) for y in common]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} years, cols {list(df.columns)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> NOTE to implementer: three call sites need a 1-line confirmation against the live code before running — the movement-map keys (`movement.map{n}.species` / `.file`: grep `data/baltic/*.csv` for the exact spelling), and `GridSpec.from_config` (match however `scripts/build_baltic_salinity_forcing.py` obtains its `GridSpec` at line ~90). Also extend `scripts/download_baltic_rv_forcing.py`'s `FIELDS` dict with `"thetao": {"dataset_id": "cmems_mod_bal_phy_my_P1M-m", "tag": "phy_monthly_reanalysis"}` and allow `--depth-min 0`. The real download is gated on the CMEMS credential rotation (Risk R1); the pure core (`summer_sst_by_year`) and everything downstream are provable now via the committed example sidecar (Task 6).

- [ ] **Step 6: Commit**

```bash
git add osmose/forcing/percid_thermal.py scripts/build_percid_thermal_series.py scripts/download_baltic_rv_forcing.py tests/test_percid_thermal_builder.py
git commit -m "feat: percid summer-SST builder core + CMEMS thetao download + driver"
```

---

### Task 6: A/B diagnostic + committed example sidecar

**Files:**
- Create: `data/baltic/forcing/baltic_percid_thermal_series_example.csv` (plausible temps spanning the Baltic run years, for the A/B until the real CMEMS series is built)
- Create: `scripts/baltic_percid_thermal_gate_diagnostic.py`
- Test: `tests/test_percid_thermal_gate_diagnostic.py` (smoke)

**Interfaces:**
- Consumes: the run/result API used by `scripts/baltic_recruitment_ceiling_diagnostic.py` (imitate it exactly); the loader/helper/wiring from Tasks 3–4.
- Produces: a script printing perch (sp4) and pikeperch (sp5) mean biomass and overshoot ratio for gate off vs on, plus a smoke test asserting it runs and reports both species.

- [ ] **Step 1: Create the example sidecar** (years matching the RV series span 1993–2010; plausible NE-Baltic coastal summer SST, cool→warm variation so `thermal_cap` bites in most years)

```csv
# data/baltic/forcing/baltic_percid_thermal_series_example.csv
year,temp_sp4,temp_sp5
1993,15.8,16.4
1994,17.1,17.6
1995,16.2,16.9
1996,14.9,15.5
1997,17.8,18.3
1998,15.4,16.0
1999,18.2,18.9
2000,16.7,17.2
2001,15.1,15.8
2002,19.0,19.6
2003,18.4,19.1
2004,16.0,16.6
2005,17.3,17.9
2006,19.3,20.0
2007,16.9,17.5
2008,15.6,16.2
2009,17.7,18.4
2010,18.9,19.5
```

- [ ] **Step 2: Write the smoke test**

```python
# tests/test_percid_thermal_gate_diagnostic.py
import subprocess, sys


def test_diagnostic_runs_and_reports_both_percids():
    # --nyear 4 keeps the smoke test cheap (two short Baltic runs); the real A/B
    # is run by hand with the full horizon. The default suite does not exclude
    # `slow`, so a short horizon (not a marker) is the real cost lever.
    out = subprocess.run(
        [sys.executable, "scripts/baltic_percid_thermal_gate_diagnostic.py", "--nyear", "4"],
        capture_output=True, text=True, timeout=900,
    )
    assert out.returncode == 0, out.stderr
    assert "perch" in out.stdout.lower() and "pikeperch" in out.stdout.lower()
    assert "overshoot" in out.stdout.lower()
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_percid_thermal_gate_diagnostic.py -q`
Expected: FAIL (script does not exist → non-zero return).

- [ ] **Step 4: Write the diagnostic** (imitate `scripts/baltic_recruitment_ceiling_diagnostic.py`: two deterministic runs, off vs `thermal_cap` on, print per-species mean biomass + overshoot ratio)

```python
# scripts/baltic_percid_thermal_gate_diagnostic.py
"""A/B: does the percid thermal gate (thermal_cap) damp percid overshoot?

Two deterministic Baltic runs (fixed movement + mortality seeds), gate off vs on
for perch (sp4) + pikeperch (sp5), reporting mean biomass and the overshoot
ratio. Honest-negative permitted: if the ratio does not fall, say so plainly.
Uses the committed example sidecar until the real CMEMS series is built.
Reuses overshoot_ratio + run API from baltic_recruitment_ceiling_diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

SERIES = (Path(__file__).resolve().parent.parent
          / "data" / "baltic" / "forcing" / "baltic_percid_thermal_series_example.csv")
DET = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}
PERCIDS = (("perch", 4), ("pikeperch", 5))


def overshoot_ratio(series: np.ndarray, late_frac: float = 1.0 / 3.0) -> float:
    """peak biomass / late-window mean (same definition as the ceiling diagnostic)."""
    b = np.asarray(series, dtype=np.float64)
    late = b[int(len(b) * (1.0 - late_frac)):]
    lm = float(np.mean(late))
    return float("inf") if lm <= 0 else float(np.max(b)) / lm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyear", type=int, default=None,
                    help="override simulation.time.nyear (small = cheap smoke run; omit for full A/B)")
    args = ap.parse_args()

    cfg_path = sorted((Path("data") / "baltic").glob("*all-parameters*.csv"))[0]
    base = dict(OsmoseConfigReader().read(str(cfg_path)))
    if args.nyear is not None:
        base["simulation.time.nyear"] = str(args.nyear)
    off_cfg = {**base, **DET}
    on_cfg = {**off_cfg,
              "reproduction.thermal.gate.enabled": "true",
              "reproduction.thermal.gate.series.file": str(SERIES),
              "reproduction.thermal.gate.mode": "thermal_cap",
              "reproduction.thermal.gate.species.enabled.sp4": "true",
              "reproduction.thermal.gate.species.enabled.sp5": "true"}

    off = PythonEngine().run_in_memory(off_cfg, seed=0).biomass()
    on = PythonEngine().run_in_memory(on_cfg, seed=0).biomass()

    print("species     mean_off   mean_on    overshoot_off  overshoot_on  verdict")
    for name, _sp in PERCIDS:
        so, sn = off[name].to_numpy(), on[name].to_numpy()
        oo, on_r = overshoot_ratio(so), overshoot_ratio(sn)
        verdict = "damped" if on_r < oo * 0.98 else ("worse" if on_r > oo * 1.02 else "no change")
        print(f"{name:<11} {np.mean(so):.4g}  {np.mean(sn):.4g}   "
              f"{oo:.3f}          {on_r:.3f}        {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_percid_thermal_gate_diagnostic.py -q`
Expected: PASS (script runs, prints both percids + overshoot). Record the off-vs-on overshoot numbers in the commit message.

- [ ] **Step 6: Full lint + test sweep**

Run: `.venv/bin/ruff check osmose/ tests/ scripts/ && .venv/bin/ruff format --check osmose/ tests/ scripts/ && .venv/bin/python -m pytest tests/ -q -k thermal`
Expected: PASS (lint clean; all thermal tests green).

- [ ] **Step 7: Commit**

```bash
git add data/baltic/forcing/baltic_percid_thermal_series_example.csv scripts/baltic_percid_thermal_gate_diagnostic.py tests/test_percid_thermal_gate_diagnostic.py
git commit -m "feat: percid thermal-gate A/B diagnostic + example sidecar

A/B result (example series): <record perch/pikeperch overshoot off->on here>."
```

---

## Post-plan verification (run before declaring done)

- [ ] `.venv/bin/python -m pytest tests/ -q` — full suite green.
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/ scripts/` and `.venv/bin/ruff format --check osmose/ ui/ tests/ scripts/` — clean.
- [ ] Confirm inert-by-default: the bundled Baltic config has no `reproduction.thermal.gate.*` keys, so default runs are unchanged (Task 4 parity test covers this).
- [ ] Record the A/B outcome in the branch and in memory: does `thermal_cap` damp percid overshoot, or is it an honest negative (percid overshoot is not recruitment-driven → next lever is cannibalism)?

## In-loop review round 1 (4 rotating reviewers, verified at file:line — fixed)

- **BLOCKER (2 reviewers):** Task 4 asserted cod bit-identity, but cod↔percid predation (`data/baltic/predation-accessibility.csv`) + shared-RNG desync make cod legitimately shift → replaced with the RV-gate `rel_change` dominance pattern; no cod bit-identity.
- **BLOCKER:** Task 3 `from_dict` call used undefined locals → fixed to `_load_thermal_gate(cfg, n_sp, n_dt, n_yr)`.
- **BLOCKER:** Task 5 habitat-mask used wrong keys → fixed to `movement.species.map{n}`/`movement.file.map{n}`, resolved by species NAME.
- **CRITICAL:** missing schema registration → added Task 3 Step 7 (9 `OsmoseField`s in `osmose/schema/species.py`).
- **CRITICAL:** `thetao` download hand-waved → promoted to Task 5 Step 5a (`FIELDS` edit).
- **MINOR:** `nyear=6`/`--nyear` overrides for cheap in-suite runs; `np.testing.assert_array_equal`; builder flip-orientation caution; added a real validation test that exercises the thermal keys (Task 3 Step 8).
- **Confirmed correct by reviewers (no change):** all Task 1/3 numeric assertions (logistic/normalization arithmetic shown), every `pytest.raises` regex, determinism keys, the `.biomass()` name-keyed wide frame, offset/window semantics vs `_load_rv_gate`, and example-sidecar year alignment.

## In-loop review round 2 (2 reviewers: fix-verification + fresh-eyes — converged)

- Both reviewers independently found the SAME single MINOR: the Step 8 validation test imported a non-existent `unknown_keys` → fixed to `config_validation.validate(keys, mode="error") == []` (the exact idiom at `tests/test_engine_config_validation.py:355`).
- Cosmetic: docstring filename `baltic_config-movement.csv` → `baltic_param-movement.csv`.
- Everything else re-verified CLEAN against live code: all 3 blocker fixes + 2 critical fixes confirmed correct at file:line; the thetao filename year-parse traced and confirmed correct (`.stem` strips `.nc`); step renumbering, commit file lists, every symbol reference, fixture/year alignment, and all fail/pass expectations check out. No new issues found by the fresh-eyes pass.
- Convergence: round 2's only defect was one pre-verified mechanical API-name swap; the fresh-eyes pass otherwise found nothing. Plan is execution-ready.

## In-loop review round 3 (execution-readiness + adversarial fresh-eyes)

- **Execution-readiness reviewer: CLEAN (0 defects).** Verified the round-2 `validate()` fix, the schema block, every import, and the Task 4 integration path would run green from repo root. Side-finding (out of scope): the existing `tests/test_rv_recruitment_gate.py` uses stale absolute paths (`…/osmose-python/…`) and would error post-repo-move; this plan uses repo-relative paths and does not inherit that.
- **Adversarial reviewer: 1 MINOR (fixed).** The builder's loud-failure path (`thetao` absent → `FileNotFoundError`) is specced (§7/§8) and NOT credential-gated, yet was the one fail-fast branch left untested. Fix: moved file-discovery into the importable `load_thetao_surface(dl, grid)` (file check before xarray/grid imports) and added `test_load_thetao_surface_raises_loudly_when_no_files`. The adversarial pass otherwise empirically confirmed the `temp_tyx[sel][:, mask_yx]` indexing, mean_preserving faithfulness vs `_load_rv_gate`, absent-vs-`false` inertness, the reproduction guard, and no schema-key collisions.

## Self-review notes (checked against the spec)

- **Spec §4.1 builder** → Task 5 (pure core tested; real CMEMS wiring is the one credential-gated step, isolated so it cannot block Tasks 1–4, 6).
- **Spec §4.2 loader / §5 keys / §7 error handling** → Task 3 (every fail-fast branch has a test).
- **Spec §4.3 helper** → Task 2. **§4.4 reproduction wiring / §6 determinism** → Task 4 (inert parity + on≠off).
- **Spec §4.5 A/B / §9 R2 honest-negative** → Task 6.
- **Modes:** `thermal_cap` default (mean-reducing) and `mean_preserving` (realism) both implemented in `normalize_factor` (Task 1) and exercised in the loader tests (Task 3).
- **Type consistency:** `thermal_gate_factor_by_index` `(n_years, n_species)`, `thermal_gate_enabled` `(n_species,)`, `thermal_gate_offset` int — used identically in Tasks 2, 3, 4.
- **Open questions (spec §10):** `tref` default fixed at 20.0 (Task 3); shared mode/floor + per-species t50/slope/tref — reflected in the key table and loader. If the user chose to derive `tref`, adjust Task 5's driver to compute it and Task 3's default accordingly.
