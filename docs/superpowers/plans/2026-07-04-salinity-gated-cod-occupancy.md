# Salinity-Gated Cod Occupancy — Implementation Plan (spike)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototype a config-gated, inert-by-default mechanism that weights cod's movement-map occupancy by a per-cell salinity factor, so cod avoids oligohaline cells (reduced cod–percid overlap is emergent).

**Architecture:** Two pure helpers compute a salinity→occupancy weight and apply it to a map (with an all-zero guard). The reference movement placement function `_map_move_school` gains an optional per-step weight grid: Step 3a (rejection sampling) uses `nanmax(wmap)` as its normalizer; Step 3b (random walk) switches from uniform to weighted selection ∝ `wmap`; both stay bit-identical when the grid is `None`. `movement()`'s Python path computes the weight grid per step from a config-loaded salinity field and passes it only for gated predator species. A loader + four `EngineConfig` fields wire the config.

**Tech Stack:** Python 3.12, NumPy, pytest. OSMOSE schema-driven config, pure-Python engine.

## Global Constraints

- Branch: `salinity-gated-cod-occupancy` (already created).
- Run everything with `.venv/bin/python` (system `python` may not exist).
- Line length 100; lint `.venv/bin/ruff check osmose/ tests/` and `format --check` on touched files.
- **Inert by default:** with `movement.salinity.gate.enabled=false` (default) every existing config (Baltic, EEC, Bay of Biscay) is bit-identical. The gate code executes only when `salinity_weight_grid is not None` (movement) / `salinity_field is not None` (caller).
- **Determinism keys** for parity: `movement.randomseed.fixed` + `stochastic.mortality.randomseed.fixed` + `simulation.rng.fixed=true`.
- `test_from_dict_warn_mode_clean_on_example_configs` must stay warning-free.
- **Spike scope:** hook only the Python reference movement path. The Numba batch path (`_map_move_batch_numba`), real CMEMS `so` forcing, and a full Baltic run are explicit **follow-ups**, not in this plan.
- Thresholds default `s_low=3.0`, `s_high=6.0` (weight `clip((S-3)/3,0,1)`: 0 at ≤3 psu, 0.5 at 4.5, 1.0 at ≥6).

---

## File Structure

- `osmose/engine/processes/salinity_gate.py` — new: two pure helpers.
- `osmose/schema/movement.py` — add 6 `OsmoseField`s (append before the closing `]`).
- `osmose/engine/config.py` — `_load_salinity_gate()` + 4 `EngineConfig` fields + `from_dict` wiring.
- `osmose/engine/processes/movement.py` — optional `salinity_weight_grid` on `_map_move_school` + Python-path caller wiring in `movement()`.
- `osmose/engine/config_validation.py` — allowlist entries only if the AST walker misses the keys (fallback).
- `tests/test_salinity_gate.py` — all tests.

---

## Task 1: Pure helpers (`salinity_gate.py`)

**Files:**
- Create: `osmose/engine/processes/salinity_gate.py`
- Test: `tests/test_salinity_gate.py`

**Interfaces:**
- Produces:
  - `salinity_weight(salinity, s_low, s_high) -> NDArray | float` — `clip((salinity - s_low)/(s_high - s_low), 0, 1)`; raises `ValueError` if `s_high <= s_low`. Accepts scalar or ndarray.
  - `salinity_weighted_map(map2d, weight_grid) -> NDArray` — returns `map2d * weight_grid`; **if the product has no positive finite cell (`>0`, not NaN/inf), returns the original `map2d` object unchanged** (all-zero guard). Takes the *precomputed weight grid* (not raw salinity) — this refines the spec's §4.1 signature so the weight is computed once per step by the caller and the guard can be detected by object identity.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_salinity_gate.py`:

```python
import numpy as np
import pytest

from osmose.engine.processes.salinity_gate import salinity_weight, salinity_weighted_map


def test_salinity_weight_ramp_scalar():
    assert salinity_weight(2.0, 3.0, 6.0) == 0.0          # below low
    assert salinity_weight(3.0, 3.0, 6.0) == 0.0          # at low
    assert salinity_weight(4.5, 3.0, 6.0) == pytest.approx(0.5)  # mid
    assert salinity_weight(6.0, 3.0, 6.0) == 1.0          # at high
    assert salinity_weight(8.0, 3.0, 6.0) == 1.0          # above high


def test_salinity_weight_array():
    S = np.array([2.0, 4.5, 8.0])
    np.testing.assert_allclose(salinity_weight(S, 3.0, 6.0), [0.0, 0.5, 1.0])


def test_salinity_weight_bad_thresholds_raise():
    with pytest.raises(ValueError):
        salinity_weight(5.0, 6.0, 6.0)   # s_high <= s_low


def test_weighted_map_zeros_low_keeps_high():
    m = np.ones((2, 3))
    w = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    out = salinity_weighted_map(m, w)
    np.testing.assert_allclose(out, w)      # 1 * w == w
    assert out is not m                      # gated -> new array


def test_weighted_map_all_zero_guard_returns_original():
    m = np.ones((2, 2))
    w = np.zeros((2, 2))
    out = salinity_weighted_map(m, w)
    assert out is m                          # identity: guard fell back to original
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine.processes.salinity_gate'`.

- [ ] **Step 3: Write the module**

Create `osmose/engine/processes/salinity_gate.py`:

```python
"""Salinity-dependent occupancy weighting for movement (prototype spike).

Pure helpers: a salinity -> [0,1] occupancy weight and its application to a
2D movement map with an all-zero guard. See
docs/superpowers/specs/2026-07-04-salinity-gated-cod-occupancy-design.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def salinity_weight(salinity, s_low: float, s_high: float):
    """Per-cell occupancy weight in [0,1]: clip((S - s_low)/(s_high - s_low), 0, 1).

    Weight 0 at/below s_low (predator excluded), 1 at/above s_high (full), linear
    between. Accepts a scalar or an ndarray. Raises ValueError if s_high <= s_low.
    """
    if s_high <= s_low:
        raise ValueError(f"salinity_weight: s_high ({s_high}) must be > s_low ({s_low})")
    return np.clip((np.asarray(salinity, dtype=np.float64) - s_low) / (s_high - s_low), 0.0, 1.0)


def salinity_weighted_map(
    map2d: NDArray[np.float64], weight_grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply a precomputed per-cell weight to a movement map.

    Returns ``map2d * weight_grid``. If the product has no positive finite cell,
    returns the ORIGINAL ``map2d`` object unchanged (all-zero guard) so a predator
    is never left with zero valid cells; callers detect the fallback by identity
    (``result is map2d``).
    """
    wmap = map2d * weight_grid
    finite_pos = np.isfinite(wmap) & (wmap > 0.0)
    if not finite_pos.any():
        return map2d
    return wmap
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/salinity_gate.py tests/test_salinity_gate.py
git commit -m "feat: salinity occupancy-weight pure helpers (salinity_gate.py)"
```

---

## Task 2: Schema fields (`movement.py`)

**Files:**
- Modify: `osmose/schema/movement.py` (append inside the field list, before the final `]`)
- Test: `tests/test_salinity_gate.py`

**Interfaces:**
- Produces registered keys: `movement.salinity.gate.enabled` (bool), `movement.salinity.gate.species.enabled.sp{idx}` (bool, indexed), `movement.salinity.gate.s.low` (float), `movement.salinity.gate.s.high` (float), `movement.salinity.field.constant` (float), `movement.salinity.field.file` (file path), `movement.salinity.field.varname` (string).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_salinity_gate.py`:

```python
from osmose.schema import build_registry


def test_salinity_gate_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "movement.salinity.gate.enabled" in keys
    assert "movement.salinity.gate.species.enabled.sp{idx}" in keys
    assert "movement.salinity.gate.s.low" in keys
    assert "movement.salinity.gate.s.high" in keys
    assert "movement.salinity.field.constant" in keys
    assert "movement.salinity.field.file" in keys
    assert "movement.salinity.field.varname" in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py::test_salinity_gate_keys_registered -v`
Expected: FAIL — assertion error.

- [ ] **Step 3: Add the schema fields**

In `osmose/schema/movement.py`, insert immediately after the `movement.randomseed.fixed` `OsmoseField` (the last entry), before the closing `]`:

```python
    # ── Salinity-gated occupancy (prototype spike) ────────────────────────
    OsmoseField(
        key_pattern="movement.salinity.gate.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description=(
            "Master switch for salinity-gated predator occupancy. When false "
            "the gate is inert and movement output is bit-identical."
        ),
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.gate.species.enabled.sp{idx}",
        param_type=ParamType.BOOL,
        default=False,
        description="Per-species enable for salinity occupancy gating (cod only for Baltic).",
        category="movement",
        indexed=True,
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.gate.s.low",
        param_type=ParamType.FLOAT,
        default=3.0,
        description="Salinity (psu) at/below which occupancy weight is 0.",
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.gate.s.high",
        param_type=ParamType.FLOAT,
        default=6.0,
        description="Salinity (psu) at/above which occupancy weight is 1.",
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.field.constant",
        param_type=ParamType.FLOAT,
        description="Constant salinity (psu) forcing (alternative to a NetCDF field).",
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.field.file",
        param_type=ParamType.FILE_PATH,
        description="NetCDF salinity field (alternative to a constant).",
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.field.varname",
        param_type=ParamType.STRING,
        default="so",
        description="Variable name in the salinity NetCDF (CMEMS 'so').",
        category="movement",
        advanced=True,
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py::test_salinity_gate_keys_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/schema/movement.py tests/test_salinity_gate.py
git commit -m "feat: schema fields for salinity-gated occupancy"
```

---

## Task 3: Loader + `EngineConfig` fields + `from_dict` wiring

**Files:**
- Modify: `osmose/engine/config.py`
- Test: `tests/test_salinity_gate.py`

**Interfaces:**
- Consumes: `cfg: dict[str, str]`, `n_species: int`.
- Produces:
  - `_load_salinity_gate(cfg, n_species) -> tuple[bool, NDArray[np.bool_] | None, float, float, PhysicalData | None]` returning `(enabled, species_mask, s_low, s_high, salinity_field)`. Off → `(False, None, 3.0, 6.0, None)`.
  - `EngineConfig.salinity_gate_enabled: bool`, `.salinity_gate_species: NDArray[np.bool_] | None`, `.salinity_gate_s_low: float`, `.salinity_gate_s_high: float`, `.salinity_field: PhysicalData | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_salinity_gate.py`:

```python
from osmose.engine.config import _load_salinity_gate


def test_salinity_gate_off_returns_defaults():
    enabled, mask, lo, hi, field = _load_salinity_gate({}, 3)
    assert enabled is False and mask is None and field is None
    assert (lo, hi) == (3.0, 6.0)


def _on_cfg(**extra):
    cfg = {
        "movement.salinity.gate.enabled": "true",
        "movement.salinity.field.constant": "8.0",
        "movement.salinity.gate.species.enabled.sp0": "true",
    }
    cfg.update(extra)
    return cfg


def test_salinity_gate_on_constant_field():
    enabled, mask, lo, hi, field = _load_salinity_gate(_on_cfg(), 3)
    assert enabled is True
    assert list(mask) == [True, False, False]
    assert (lo, hi) == (3.0, 6.0)
    assert field is not None and field.is_constant and field.get_scalar() == 8.0


def test_salinity_gate_custom_thresholds():
    _, _, lo, hi, _ = _load_salinity_gate(
        _on_cfg(**{"movement.salinity.gate.s.low": "4.0", "movement.salinity.gate.s.high": "7.0"}), 3
    )
    assert (lo, hi) == (4.0, 7.0)


def test_salinity_gate_bad_thresholds_raise():
    with pytest.raises(ValueError, match="s.high|s_high"):
        _load_salinity_gate(_on_cfg(**{"movement.salinity.gate.s.high": "3.0"}), 3)


def test_salinity_gate_no_species_raises():
    cfg = {"movement.salinity.gate.enabled": "true", "movement.salinity.field.constant": "8.0"}
    with pytest.raises(ValueError, match="no species"):
        _load_salinity_gate(cfg, 3)


def test_salinity_gate_no_field_raises():
    cfg = {
        "movement.salinity.gate.enabled": "true",
        "movement.salinity.gate.species.enabled.sp0": "true",
    }
    with pytest.raises(ValueError, match="salinity field|field"):
        _load_salinity_gate(cfg, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -k "salinity_gate_off or salinity_gate_on or thresholds or no_species or no_field" -v`
Expected: FAIL — `ImportError: cannot import name '_load_salinity_gate'`.

- [ ] **Step 3: Add the loader**

In `osmose/engine/config.py`, add after `_load_rv_gate` (locate it: `grep -n "def _load_rv_gate" osmose/engine/config.py`). Ensure `from osmose.engine.physical_data import PhysicalData` is importable (it is used engine-wide; add a top-level import if config.py doesn't already have it — `grep -n "import PhysicalData" osmose/engine/config.py`; if absent, add `from osmose.engine.physical_data import PhysicalData` with the other engine imports).

```python
def _load_salinity_gate(
    cfg: dict[str, str], n_species: int
) -> tuple[bool, NDArray[np.bool_] | None, float, float, "PhysicalData | None"]:
    """Load the salinity-gated occupancy config (spec 2026-07-04).

    Returns (enabled, species_mask, s_low, s_high, salinity_field). Off →
    (False, None, 3.0, 6.0, None). Fail-fast (ValueError / FileNotFoundError)
    on bad config: s_high <= s_low, no gated species, or no resolvable field.
    """
    s_low = float(cfg.get("movement.salinity.gate.s.low", "3.0"))
    s_high = float(cfg.get("movement.salinity.gate.s.high", "6.0"))
    if cfg.get("movement.salinity.gate.enabled", "false").lower() != "true":
        return False, None, s_low, s_high, None

    if s_high <= s_low:
        raise ValueError(
            f"movement.salinity.gate.s.high ({s_high}) must be > s.low ({s_low})"
        )

    mask = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        if cfg.get(f"movement.salinity.gate.species.enabled.sp{sp}", "false").lower() == "true":
            mask[sp] = True
    if not mask.any():
        raise ValueError(
            "salinity gate enabled but no species enabled "
            "(movement.salinity.gate.species.enabled.sp{idx})."
        )

    from osmose.engine.physical_data import PhysicalData

    const_str = cfg.get("movement.salinity.field.constant", "")
    file_str = cfg.get("movement.salinity.field.file", "")
    if const_str:
        field = PhysicalData.from_constant(float(const_str))
    elif file_str:
        path = _require_file(file_str, _cfg_dir(cfg), "movement.salinity.field.file")
        varname = cfg.get("movement.salinity.field.varname", "so")
        field = PhysicalData.from_netcdf(path, varname=varname)
    else:
        raise ValueError(
            "salinity gate enabled but no salinity field "
            "(set movement.salinity.field.constant or .file)."
        )
    return True, mask, s_low, s_high, field
```

- [ ] **Step 4: Add the EngineConfig fields**

In `osmose/engine/config.py`, locate the RV-gate fields (`grep -n "rv_gate_offset" osmose/engine/config.py`) and add immediately after the `rv_gate_offset: int` field:

```python

    # Salinity-gated occupancy (prototype spike; feature inert when disabled)
    salinity_gate_enabled: bool = False
    salinity_gate_species: NDArray[np.bool_] | None = None
    salinity_gate_s_low: float = 3.0
    salinity_gate_s_high: float = 6.0
    salinity_field: "PhysicalData | None" = None
```

(If the `EngineConfig` dataclass has no other defaulted fields and field-ordering forbids defaults here, instead add them as non-defaulted fields and pass all five explicitly in `from_dict` — check `grep -n "= None$\|= False$" osmose/engine/config.py | head` near the field block; the RV-gate fields are non-defaulted, so if the class is all-non-default, declare these non-defaulted too and always pass them in `from_dict`.)

- [ ] **Step 5: Wire into from_dict**

In `osmose/engine/config.py`, at the `_load_rv_gate(...)` call site (`grep -n "_load_rv_gate(" osmose/engine/config.py`), add after it:

```python
        (
            salinity_gate_enabled,
            salinity_gate_species,
            salinity_gate_s_low,
            salinity_gate_s_high,
            salinity_field,
        ) = _load_salinity_gate(cfg, n_sp)
```

Then in the `EngineConfig(...)` kwargs block (next to the `rv_gate_*` kwargs), add:

```python
            salinity_gate_enabled=salinity_gate_enabled,
            salinity_gate_species=salinity_gate_species,
            salinity_gate_s_low=salinity_gate_s_low,
            salinity_gate_s_high=salinity_gate_s_high,
            salinity_field=salinity_field,
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -v`
Expected: PASS (all loader + earlier tests).

- [ ] **Step 7: Verify config-validation stays clean**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -v`
Expected: PASS. If it flags any new `movement.salinity.*` key (the per-species f-string key is read literally, so the AST walker should capture it — like `reproduction.rv.gate.species.enabled.sp{sp}`), add the flagged pattern(s) to `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py`.

Also run the direct-construction guard (adding EngineConfig fields can break `EngineConfig(**cfg)` test factories):
Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -q`
Expected: PASS. If a `_minimal_config`-style factory fails with "missing argument", the new fields must be defaulted (Step 4's default form) OR the factory updated — prefer the defaulted form so no test factory needs changing.

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/config.py tests/test_salinity_gate.py
git commit -m "feat: _load_salinity_gate loader + EngineConfig fields + wiring"
```

---

## Task 4: `_map_move_school` gate (Step 3a + Step 3b)

**Files:**
- Modify: `osmose/engine/processes/movement.py` (`_map_move_school`, lines ~33-103)
- Test: `tests/test_salinity_gate.py`

**Interfaces:**
- Consumes: `salinity_weighted_map` (Task 1).
- Produces: `_map_move_school(..., salinity_weight_grid: NDArray[np.float64] | None = None)` — new trailing keyword. `None` → bit-identical to today. When set: occupancy ∝ `current_map · weight` in both placement and random-walk.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_salinity_gate.py`:

```python
from osmose.engine.grid import Grid
from osmose.engine.movement_maps import MovementMapSet
from osmose.engine.processes.movement import _map_move_school
from osmose.engine.processes.salinity_gate import salinity_weight


def _uniform_map_set(ny, nx):
    """A MovementMapSet whose single presence map is 1.0 over all cells."""
    ms = MovementMapSet.__new__(MovementMapSet)
    ms.maps = [np.ones((ny, nx), dtype=np.float64)]
    # shape (lifespan_dt, n_total_steps); index 0 for ALL age/step so get_map is
    # valid for age_dt 0 AND 1 (the random-walk test uses age_dt=1, step=1).
    ms.index_maps = np.zeros((10, 1000), dtype=np.int32)
    ms.max_proba = np.array([0.0])                         # presence/absence -> uniform accept
    ms.n_maps = 1
    return ms


def _draw_columns(gate_grid, n=4000):
    ny, nx = 5, 6
    ms = _uniform_map_set(ny, nx)
    ocean = np.ones((ny, nx), dtype=np.bool_)
    rng = np.random.default_rng(0)
    cols = np.zeros(nx, dtype=np.int64)
    for _ in range(n):
        x, y, out = _map_move_school(
            0, -1, -1, ny, nx, ocean, ms, 1, 0, rng, salinity_weight_grid=gate_grid
        )
        assert not out
        cols[x] += 1
    return cols


def test_placement_excludes_low_and_grades_mid_vs_high():
    ny, nx = 5, 6
    # three salinity bands by column: cols 0-1 = 2 psu, 2-3 = 4.5 psu, 4-5 = 8 psu
    S = np.zeros((ny, nx))
    S[:, 0:2] = 2.0
    S[:, 2:4] = 4.5
    S[:, 4:6] = 8.0
    w = salinity_weight(S, 3.0, 6.0)          # weights 0 / 0.5 / 1.0
    cols = _draw_columns(w)
    assert cols[0] == 0 and cols[1] == 0       # excluded (weight 0)
    high = cols[4] + cols[5]
    mid = cols[2] + cols[3]
    assert mid > 0 and high > 0
    assert high / mid == pytest.approx(2.0, rel=0.15)  # graded ~2x (Step 3a path)


def test_placement_ungated_is_uniform():
    cols = _draw_columns(None)
    # No gate: all 6 columns populated roughly equally
    assert (cols > 0).all()


def test_random_walk_weighted(monkeypatch):
    # Force Step 3b (same-map, located school): weighted selection ~2x high vs mid.
    ny, nx = 5, 6
    ms = _uniform_map_set(ny, nx)
    ocean = np.ones((ny, nx), dtype=np.bool_)
    S = np.zeros((ny, nx))
    S[:, 2:4] = 4.5
    S[:, 4:6] = 8.0
    w = salinity_weight(S, 3.0, 6.0)
    rng = np.random.default_rng(1)
    cols = np.zeros(nx, dtype=np.int64)
    # start located at (cx=3, cy=2), walk_range large enough to reach cols 2-5
    for _ in range(4000):
        x, y, out = _map_move_school(
            1, 3, 2, ny, nx, ocean, ms, 5, 1, rng, salinity_weight_grid=w
        )
        cols[x] += 1
    high = cols[4] + cols[5]
    mid = cols[2] + cols[3]
    assert cols[0] == 0 and cols[1] == 0
    assert high / mid == pytest.approx(2.0, rel=0.2)
```

Note: `MovementMapSet.get_map`/`get_index`/`max_proba` are read by `_map_move_school`. `_uniform_map_set` builds a minimal instance directly (bypassing CSV loading). Verify the attribute names against `osmose/engine/movement_maps.py` (`maps`, `index_maps`, `max_proba`, `n_maps`) and `get_map`/`get_index` methods before running; adjust the fixture if the accessors differ.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -k "placement or random_walk" -v`
Expected: FAIL — `_map_move_school() got an unexpected keyword argument 'salinity_weight_grid'`.

- [ ] **Step 3: Modify `_map_move_school`**

In `osmose/engine/processes/movement.py`, change the signature (add the trailing keyword) and body. Add `salinity_weight_grid: NDArray[np.float64] | None = None,` as the last parameter (after `rng`). Add the import at the top of the file: `from osmose.engine.processes.salinity_gate import salinity_weighted_map`.

After the `current_map is None` check (the `return -1, -1, True` block), insert:

```python
    # Salinity-gated occupancy (inert when salinity_weight_grid is None).
    if salinity_weight_grid is not None:
        wmap = salinity_weighted_map(current_map, salinity_weight_grid)
        gated = wmap is not current_map      # guard returns the original object on all-zero
    else:
        wmap = current_map
        gated = False
```

In **Step 3a** (placement), replace the `max_p`/`proba` lines so they use `wmap`:
- `max_p = map_set.max_proba[index_map]` → `max_p = float(np.nanmax(wmap)) if gated else map_set.max_proba[index_map]`
- `proba = current_map[j, i]` → `proba = wmap[j, i]`

In **Step 3b** (random walk), replace the accessible-list build + uniform pick:

```python
    # Step 3b — Random walk (same map, school is located)
    accessible: list[tuple[int, int]] = []
    weights: list[float] = []
    for yi in range(max(0, cy - walk_range), min(grid_ny, cy + walk_range + 1)):
        for xi in range(max(0, cx - walk_range), min(grid_nx, cx + walk_range + 1)):
            v = wmap[yi, xi]
            if ocean_mask[yi, xi] and v > 0 and not np.isnan(v):
                accessible.append((xi, yi))
                weights.append(float(v))
    if len(accessible) == 0:
        return cx, cy, False  # stranded — stay in place
    if gated:
        w = np.asarray(weights, dtype=np.float64)
        idx = int(rng.choice(len(accessible), p=w / w.sum()))
    else:
        idx = rng.integers(0, len(accessible))
    return accessible[idx][0], accessible[idx][1], False
```

(`np` is already imported in movement.py.) The ungated branch keeps the exact original `rng.integers` draw, so master runs stay bit-identical.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -k "placement or random_walk" -v`
Expected: PASS.

- [ ] **Step 5: Run the movement regression suite**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py -q`
Expected: PASS (no regression — the default `salinity_weight_grid=None` preserves behavior).

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/movement.py tests/test_salinity_gate.py
git commit -m "feat: salinity_weight_grid gate in _map_move_school (3a nanmax, 3b weighted)"
```

---

## Task 5: `movement()` caller wiring + inert-by-default parity

**Files:**
- Modify: `osmose/engine/processes/movement.py` (`movement()` Python-path branch)
- Test: `tests/test_salinity_gate.py`

**Interfaces:**
- Consumes: `_map_move_school(..., salinity_weight_grid=...)` (Task 4); `EngineConfig.salinity_*` fields (Task 3); `salinity_weight` (Task 1).
- Produces: `_movement_salinity_weight(config, grid, step) -> NDArray[np.float64] | None` in `movement.py` — the per-step weight grid (or `None` when the gate is off). Extracting this as a named seam makes the caller-wiring computation directly testable without constructing full `movement()` inputs.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_salinity_gate.py`. First the extracted-seam tests (concrete, no full-engine construction — the helper reads only four config fields + `grid.ny`/`grid.nx`, so a `SimpleNamespace` stands in):

```python
from types import SimpleNamespace

from osmose.engine.physical_data import PhysicalData
from osmose.engine.processes.movement import _movement_salinity_weight


def _cfg_grid(enabled, field):
    cfg = SimpleNamespace(
        salinity_gate_enabled=enabled,
        salinity_field=field,
        salinity_gate_s_low=3.0,
        salinity_gate_s_high=6.0,
    )
    grid = SimpleNamespace(ny=5, nx=6)
    return cfg, grid


def test_movement_weight_off_returns_none():
    cfg, grid = _cfg_grid(False, None)
    assert _movement_salinity_weight(cfg, grid, 0) is None


def test_movement_weight_enabled_but_no_field_returns_none():
    cfg, grid = _cfg_grid(True, None)
    assert _movement_salinity_weight(cfg, grid, 0) is None


def test_movement_weight_constant_high_all_ones():
    cfg, grid = _cfg_grid(True, PhysicalData.from_constant(8.0))
    w = _movement_salinity_weight(cfg, grid, 0)
    assert w.shape == (5, 6)
    np.testing.assert_array_equal(w, np.ones((5, 6)))


def test_movement_weight_constant_low_all_zeros():
    cfg, grid = _cfg_grid(True, PhysicalData.from_constant(2.0))
    np.testing.assert_array_equal(_movement_salinity_weight(cfg, grid, 0), np.zeros((5, 6)))
```

Then the engine-level inert-by-default parity test:

```python
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine


def test_gate_off_is_bit_identical():
    cfg = OsmoseConfigReader().read("data/eec_full/eec_all-parameters.csv")
    cfg["simulation.time.nyear"] = "2"
    cfg["simulation.rng.fixed"] = "true"
    cfg["movement.randomseed.fixed"] = "true"
    cfg["stochastic.mortality.randomseed.fixed"] = "true"
    base = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
    cfg["movement.salinity.gate.enabled"] = "false"
    off = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
    np.testing.assert_array_equal(base.to_numpy(), off.to_numpy())
```

(Copy the run + `.biomass()` accessor exactly from the proven harness in `tests/test_recruitment_ceiling.py::test_ceiling_off_is_bit_identical`. The gate-OFF assertion holds regardless of the Numba/Python movement branch because `salinity_field is None` makes `_movement_salinity_weight` return `None`, so `_map_move_school` is called exactly as today.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -k "movement_weight or gate_off" -v`
Expected: the `movement_weight` tests FAIL with `ImportError: cannot import name '_movement_salinity_weight'`. `gate_off` PASSES once Task 3 wiring exists (or ERRORS on an unknown `salinity_*` kwarg if Task 3 is incomplete).

- [ ] **Step 3: Add the seam + wire the caller**

In `osmose/engine/processes/movement.py`, add the helper near the top (module level, after imports):

```python
def _movement_salinity_weight(config, grid, step):
    """Per-step salinity occupancy-weight grid for the gate, or None when off.

    Reads config.salinity_gate_enabled / .salinity_field / .salinity_gate_s_low /
    .salinity_gate_s_high and grid.ny/nx. Constant fields broadcast to the grid.
    """
    if not (config.salinity_gate_enabled and config.salinity_field is not None):
        return None
    from osmose.engine.processes.salinity_gate import salinity_weight

    if config.salinity_field.is_constant:
        S = np.full((grid.ny, grid.nx), config.salinity_field.get_scalar())
    else:
        S = config.salinity_field.get_grid(step)
    return salinity_weight(S, config.salinity_gate_s_low, config.salinity_gate_s_high)
```

Then in `movement()`'s **Python-path branch** (the `else` of `if _HAS_NUMBA and flat_map_data is not None:`), compute the grid once before the per-school loop:

```python
            sal_w = _movement_salinity_weight(config, grid, step)
```

and in the `_map_move_school(...)` call inside the loop, add the trailing argument:

```python
                    salinity_weight_grid=(
                        sal_w
                        if (sal_w is not None and config.salinity_gate_species[sp_id])
                        else None
                    ),
```

The per-species selection (`salinity_gate_species[sp_id]`) is trivial glue; it is covered indirectly (a non-gated species passes `None`) and the graded mechanism itself is proven at the `_map_move_school` level in Task 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/movement.py tests/test_salinity_gate.py
git commit -m "feat: movement() salinity-gate caller wiring + inert-by-default parity"
```

---

## Task 6: Full-suite gate + lint + demonstration note

**Files:** none new (verification), optional demo script.

- [ ] **Step 1: Lint and format**

Run: `.venv/bin/ruff check osmose/ tests/` and `.venv/bin/ruff format --check osmose/ tests/`
Expected: clean on touched files. Fix findings on the feature files only (do not reformat unrelated pre-existing files; note any pre-existing unformatted files).

- [ ] **Step 2: Run the feature + related suites**

Run: `.venv/bin/python -m pytest tests/test_salinity_gate.py tests/test_engine_map_movement.py tests/test_engine_config_validation.py -q`
Expected: all PASS.

- [ ] **Step 3: Confirm inert-by-default across bundled configs**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -q`
Expected: PASS (no `movement.salinity.*` warnings on EEC / Bay of Biscay / Baltic).

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint + final verification for salinity-gated occupancy"
```

---

## Self-Review Notes (author, against the spec)

- **Spec §4.1 helpers:** Task 1. Deliberate refinement: `salinity_weighted_map` takes the *precomputed weight grid* (not raw salinity + thresholds) — matches §4.3 which passes the weight grid to `_map_move_school`, avoids recomputing the weight per school, and enables the identity-based all-zero-guard detection. The `salinity_weight` helper carries the thresholds.
- **Spec §4.2 `_map_move_school` (3a nanmax, 3b weighted, guard, inert):** Task 4. The Critical R4 finding (3b was uniform) is implemented as weighted `rng.choice`; ungated 3b keeps the exact `rng.integers` draw for bit-identical parity.
- **Spec §4.3 caller wiring (constant + gridded, per-species mask):** Task 5.
- **Spec §4.4 loader + fields + 5-tuple contract:** Task 3.
- **Spec §5 config keys + §8 schema in movement.py:** Task 2.
- **Spec §6 tests (units, placement 3-band, gate-ON wiring, parity, loader fail-fast):** Tasks 1/3/4/5.
- **Spec §2 non-goals (Numba path, real CMEMS, full run):** honored — no Numba path touched; constant/synthetic field only.
- **Known integration point flagged, not a placeholder:** the exact `PythonEngine` run/`.biomass()` accessors (Task 5) are copied from the proven harness in `tests/test_recruitment_ceiling.py`; the `MovementMapSet` minimal-fixture attribute names (Task 4) must be verified against `movement_maps.py` before running. Both are named at their use sites.
