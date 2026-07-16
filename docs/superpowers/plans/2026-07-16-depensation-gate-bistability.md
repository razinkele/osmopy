# Depensation Gate + Bistability Placement (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a config-plumbed recruitment depensation/Allee gate for the Baltic OSMOSE model, plus the grid-sweep placement harness that searches for a bistable operating point with a realistic (O(100kt) SSB, stable) healthy cod basin.

**Architecture:** A per-species multiplicative Allee factor `A(SSB)=SSB^θ/(S50^θ+SSB^θ)` applied to egg production in `reproduction()` right after the existing recruitment gates — default-off, byte-identical when off, plain Python (no Numba). The bistable operating point is found by a deterministic grid sweep (larval-M scale × S50 × θ) run through the existing warm-start reciprocal-invasion classifier, validated by a same-scale no-Allee control + a fishing-hysteresis F-ramp, and shipped as a `data/baltic_depensation` overlay.

**Tech Stack:** Python, NumPy, pytest; the OSMOSE Python engine (`osmose/engine/`), the warm-start harness `scripts/baltic_bistability_chunk0.py`, the byyear-F tooling.

**Spec:** `docs/superpowers/specs/2026-07-16-depensation-gate-bistability-design.md` (converged via 9 in-loop review rounds).

## Global Constraints

- The gate mirrors the RV/thermal gates: default-off, **byte-identical when off**, per-species-configurable, skipped on seeded steps. Plain Python (no Numba).
- Config keys: `reproduction.depensation.gate.enabled`, `reproduction.depensation.gate.species.enabled.sp{i}`, `reproduction.depensation.gate.s50.sp{i}` (tonnes, >0), `reproduction.depensation.gate.theta.sp{i}` (≥1).
- `_load_depensation_gate` mirrors `_load_thermal_gate`'s STRUCTURE: loaded directly at the call site near `config.py:2429-2431` (AFTER `_merge_focal_background`, focal-only `n_sp`), fields near `config.py:1684-1687`, constructor kwargs near `config.py:2506-2508`. Do NOT touch the `_merge_focal_background` blocks. **Returns a 3-tuple `(enabled, s50, theta)`, each `None` when off** (never bare `None`). Fail-fast (`raise ValueError`) on: global-on/no-species, θ<1, s50≤0.
- Wiring in `reproduction.py` MUST include `assert config.depensation_s50 is not None` / `assert config.depensation_theta is not None` (pyright CI leg fails without them).
- EngineConfig field types: `depensation_gate_enabled: NDArray[np.bool_] | None`, `depensation_s50: NDArray[np.float64] | None`, `depensation_theta: NDArray[np.float64] | None`.
- Overlay `data/baltic_depensation`: `mortality.additional.larva.rate.sp{i}` stored as `engine_value × 24` (ndt-migration gotcha); needs an EXPLICIT new Java guard in `runner.py::java_engine_block_reason` (the existing nbackground guard does NOT block it).
- Base F for the F-ramp = `fisheries.rate.base.fsh0` (=0.08), NOT legacy `mortality.fishing.rate.sp{i}`; resolve mode-agnostically via `EngineConfig.fishing_rate[0]` / `osmose/validation/fmsy_sweep.py::fishing_override`.
- Emergent analysis tests (harness smoke, gate-on integration) carry `@pytest.mark.skipif` for CI (real-engine Baltic is core-count-sensitive).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `osmose/engine/processes/depensation_gate.py` (create) | Pure `depensation_factor(ssb,s50,theta,enabled)` | 1 |
| `tests/test_depensation_gate.py` (create) | Gate math + config-parse + wiring tests | 1,2,3 |
| `osmose/engine/config.py` (modify) | `_load_depensation_gate` + fields + plumbing | 2 |
| `tests/test_engine_config_validation.py` (modify) | `_minimal_config` fixture: 3 new fields | 2 |
| `osmose/engine/processes/reproduction.py` (modify) | Wiring block after thermal gate | 3 |
| `osmose/runner.py` (modify) | Java guard for depensation gate | 4 |
| `osmose/schema/species.py` (modify) | `OsmoseField` entries for the 4 keys | 4 |
| `data/baltic_depensation/` (create) | Overlay: gate + S50/θ + larval-M (operating point) | 5 |
| `tests/test_baltic_depensation_demo.py` (create) | Overlay loads / Java-blocked / strict-valid | 5 |
| `scripts/calibrate_depensation_bistability.py` (create) | Unit 2 grid-sweep placement harness | 6 |
| `scripts/validate_depensation_hysteresis.py` (create) | Unit 3 no-Allee control + F-ramp | 7 |
| `docs/diagnostics/2026-07-16-depensation-placement.md` (create) | Sweep result (GO/AMBIGUOUS/NO-GO) | 8 (analysis) |

**Merge-green boundary:** Tasks 1–7 are CI-testable code and constitute the "merge once green" deliverable. **Task 8 is the long analysis run** (288–480 multi-decade sims + a 9,000–15,000 sim-year F-ramp) with an uncertain scientific outcome; it runs AFTER merge and fills the overlay's real operating point (or documents AMBIGUOUS/NO-GO).

---

## Task 1: Pure depensation gate function

**Files:**
- Create: `osmose/engine/processes/depensation_gate.py`
- Test: `tests/test_depensation_gate.py`

**Interfaces:**
- Produces: `depensation_factor(ssb: NDArray[np.float64], s50: NDArray[np.float64], theta: NDArray[np.float64], enabled: NDArray[np.bool_]) -> NDArray[np.float64]` — per-species Allee multiplier in (0,1]; 1.0 where disabled.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_depensation_gate.py
import numpy as np
import pytest
from osmose.engine.processes.depensation_gate import depensation_factor


def test_half_at_s50():
    # A(S50) = 0.5 exactly, any theta
    f = depensation_factor(
        np.array([50_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert f[0] == pytest.approx(0.5)


def test_approaches_zero_at_low_ssb():
    f = depensation_factor(
        np.array([1_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert 0.0 < f[0] < 1e-4


def test_zero_at_ssb_zero():
    f = depensation_factor(
        np.array([0.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert f[0] == 0.0


def test_approaches_one_at_high_ssb():
    f = depensation_factor(
        np.array([5_000_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([True])
    )
    assert f[0] == pytest.approx(1.0, abs=1e-3)


def test_disabled_is_one():
    f = depensation_factor(
        np.array([1_000.0]), np.array([50_000.0]), np.array([4.0]), np.array([False])
    )
    assert f[0] == 1.0


def test_theta_one_boundary():
    # theta=1: A = SSB/(S50+SSB); at S50 still 0.5, still <1 at low SSB
    f = depensation_factor(
        np.array([50_000.0]), np.array([50_000.0]), np.array([1.0]), np.array([True])
    )
    assert f[0] == pytest.approx(0.5)


def test_multi_species_isolation():
    # only enabled species differ from 1.0
    ssb = np.array([1_000.0, 1_000.0])
    s50 = np.array([50_000.0, 50_000.0])
    theta = np.array([4.0, 4.0])
    enabled = np.array([True, False])
    f = depensation_factor(ssb, s50, theta, enabled)
    assert f[0] < 1e-4
    assert f[1] == 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_depensation_gate.py -x -q`
Expected: FAIL with `ModuleNotFoundError: osmose.engine.processes.depensation_gate`.

- [ ] **Step 3: Implement the module**

```python
# osmose/engine/processes/depensation_gate.py
"""Recruitment depensation / Allee gate — pure helper.

Engine-state-free. The multiplier is a function of the CURRENT per-species SSB
(state-dependent, unlike the step-driven RV/thermal gates). Applied to egg
production in reproduction() after apply_stock_recruitment. A depensatory Allee
term creates a low-SSB recruitment trap (Liermann-Hilborn form).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def depensation_factor(
    ssb: NDArray[np.float64],
    s50: NDArray[np.float64],
    theta: NDArray[np.float64],
    enabled: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Per-species Allee multiplier A(SSB)=SSB^theta/(S50^theta+SSB^theta), in (0, 1].

    1.0 where disabled. A->0 as SSB->0, A=0.5 at SSB==S50, A->1 as SSB->inf.
    All arguments are length n_sp.
    """
    out = np.ones(ssb.shape[0], dtype=np.float64)
    for sp in range(ssb.shape[0]):
        if not enabled[sp]:
            continue
        s = ssb[sp]
        if s <= 0.0:
            out[sp] = 0.0  # full suppression at SSB=0; harmless (n_eggs already 0) + skipped-when-seeded
            continue
        out[sp] = s ** theta[sp] / (s50[sp] ** theta[sp] + s ** theta[sp])
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_depensation_gate.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/depensation_gate.py tests/test_depensation_gate.py
git commit -m "feat(engine): pure depensation/Allee gate factor"
```

---

## Task 2: Config loader + EngineConfig plumbing

**Files:**
- Modify: `osmose/engine/config.py` (add `_load_depensation_gate`; 3 dataclass fields near :1684-1687; load call near :2429-2431; constructor kwargs near :2506-2508)
- Modify: `tests/test_engine_config_validation.py` (`_minimal_config`: add the 3 new fields)
- Test: `tests/test_depensation_gate.py` (append config-parse cases)

**Interfaces:**
- Consumes: `depensation_factor` (Task 1).
- Produces: `EngineConfig.depensation_gate_enabled: NDArray[np.bool_]|None`, `.depensation_s50: NDArray[np.float64]|None`, `.depensation_theta: NDArray[np.float64]|None` (all None when off, set together). `_load_depensation_gate(cfg: dict, n_sp: int) -> tuple[NDArray[np.bool_]|None, NDArray[np.float64]|None, NDArray[np.float64]|None]`.

- [ ] **Step 1: Read the thermal-gate precedent**

Read `osmose/engine/config.py::_load_thermal_gate` (~lines 1225-1301), its dataclass fields (~1684-1687), its load call site (~2429-2431), and its constructor kwargs (~2506-2508). Mirror the STRUCTURE (direct load after merge, focal-only `n_sp`), not the CSV/time-series content.

- [ ] **Step 2: Write the failing config-parse tests**

```python
# append to tests/test_depensation_gate.py
import numpy as np
import pytest
from osmose.engine.config import _load_depensation_gate


def _cfg(**over):
    base = {
        "reproduction.depensation.gate.enabled": "true",
        "reproduction.depensation.gate.species.enabled.sp0": "true",
        "reproduction.depensation.gate.s50.sp0": "60000",
        "reproduction.depensation.gate.theta.sp0": "4.0",
    }
    base.update(over)
    return base


def test_loader_off_returns_triple_of_none():
    assert _load_depensation_gate({}, 2) == (None, None, None)


def test_loader_parses_enabled_species():
    enabled, s50, theta = _load_depensation_gate(_cfg(), 2)
    assert list(enabled) == [True, False]
    assert s50[0] == 60000.0
    assert theta[0] == 4.0


def test_loader_failfast_theta_below_one():
    with pytest.raises(ValueError):
        _load_depensation_gate(_cfg(**{"reproduction.depensation.gate.theta.sp0": "0.5"}), 2)


def test_loader_failfast_s50_nonpositive():
    with pytest.raises(ValueError):
        _load_depensation_gate(_cfg(**{"reproduction.depensation.gate.s50.sp0": "0"}), 2)


def test_loader_failfast_global_on_no_species():
    with pytest.raises(ValueError):
        _load_depensation_gate({"reproduction.depensation.gate.enabled": "true"}, 2)
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_depensation_gate.py -k loader -q`
Expected: FAIL with `ImportError: cannot import name '_load_depensation_gate'`.

- [ ] **Step 4: Implement `_load_depensation_gate`** (add near `_load_thermal_gate` in `config.py`)

```python
def _load_depensation_gate(cfg, n_sp):
    """Recruitment depensation/Allee gate params. Returns (enabled, s50, theta),
    each None when the gate is off. Fail-fast on invalid enabled config."""
    from osmose.engine.config import _enabled  # existing bool-parse helper

    if not _enabled(cfg, "reproduction.depensation.gate.enabled"):
        return None, None, None
    enabled = np.zeros(n_sp, dtype=np.bool_)
    s50 = np.zeros(n_sp, dtype=np.float64)
    theta = np.ones(n_sp, dtype=np.float64)
    for i in range(n_sp):
        if _enabled(cfg, f"reproduction.depensation.gate.species.enabled.sp{i}"):
            enabled[i] = True
            s50[i] = float(cfg.get(f"reproduction.depensation.gate.s50.sp{i}", "0"))
            theta[i] = float(cfg.get(f"reproduction.depensation.gate.theta.sp{i}", "1"))
            if theta[i] < 1.0:
                raise ValueError(
                    f"depensation gate theta.sp{i}={theta[i]} < 1.0 is not an Allee trap"
                )
            if s50[i] <= 0.0:
                raise ValueError(f"depensation gate s50.sp{i}={s50[i]} must be > 0")
    if not enabled.any():
        raise ValueError(
            "reproduction.depensation.gate.enabled=true but no species enabled "
            "(reproduction.depensation.gate.species.enabled.sp{i})"
        )
    return enabled, s50, theta
```

(If `_enabled` has a different name/signature in this file, use the same helper `_load_thermal_gate` uses to parse booleans.)

- [ ] **Step 5: Add the dataclass fields** (near `config.py:1684-1687`, beside `thermal_gate_*`)

```python
    depensation_gate_enabled: NDArray[np.bool_] | None
    depensation_s50: NDArray[np.float64] | None
    depensation_theta: NDArray[np.float64] | None
```

- [ ] **Step 6: Add the load call** (near `config.py:2429-2431`, beside the `_load_thermal_gate(...)` call, using focal-only `n_sp`)

```python
    depensation_gate_enabled, depensation_s50, depensation_theta = _load_depensation_gate(cfg, n_sp)
```

- [ ] **Step 7: Add the constructor kwargs** (near `config.py:2506-2508`, beside `thermal_gate_*=`)

```python
        depensation_gate_enabled=depensation_gate_enabled,
        depensation_s50=depensation_s50,
        depensation_theta=depensation_theta,
```

- [ ] **Step 8: Update `_minimal_config` fixture** in `tests/test_engine_config_validation.py`

Find the `EngineConfig(...)` construction in `_minimal_config` and add the three fields:
```python
        depensation_gate_enabled=None,
        depensation_s50=None,
        depensation_theta=None,
```

- [ ] **Step 9: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_depensation_gate.py tests/test_engine_config_validation.py -q`
Expected: all passed (loader tests + config-validation fixture green).

- [ ] **Step 10: Commit**

```bash
git add osmose/engine/config.py tests/test_depensation_gate.py tests/test_engine_config_validation.py
git commit -m "feat(engine): _load_depensation_gate + EngineConfig plumbing"
```

---

## Task 3: Wire the gate into reproduction()

**Files:**
- Modify: `osmose/engine/processes/reproduction.py` (guarded block after the thermal-gate block, ends ~line 190, before "Create new schools from eggs" ~line 192)
- Test: `tests/test_depensation_gate.py` (append byte-identical + seeded-skip + CI-skip integration)

**Interfaces:**
- Consumes: `depensation_factor` (Task 1), `EngineConfig.depensation_*` (Task 2).

- [ ] **Step 1: Write the failing byte-identical + seeded-skip tests**

```python
# append to tests/test_depensation_gate.py
import os
import numpy as np
import pytest


def _run_cod_ssb(overrides, seed=0, n_year=8):
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    base = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    raw = {**base, "simulation.time.nyear": str(n_year), "output.ssb.enabled": "true", **overrides}
    return PythonEngine().run_in_memory(raw, seed=seed).ssb()["cod"].to_numpy(dtype=float)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="real-engine Baltic, core-count-sensitive")
def test_gate_off_is_bit_identical_to_baseline():
    base = _run_cod_ssb({})
    off = _run_cod_ssb({"reproduction.depensation.gate.enabled": "false"})
    np.testing.assert_array_equal(base, off)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="real-engine Baltic, core-count-sensitive")
def test_gate_on_changes_cod_recruitment():
    base = _run_cod_ssb({})
    on = _run_cod_ssb(
        {
            "reproduction.depensation.gate.enabled": "true",
            "reproduction.depensation.gate.species.enabled.sp0": "true",
            "reproduction.depensation.gate.s50.sp0": "200000",
            "reproduction.depensation.gate.theta.sp0": "4.0",
        }
    )
    assert not np.array_equal(base, on)  # a strong Allee at high S50 must move cod
```

Add a pure-unit seeded-skip test modeled on `tests/test_recruitment_ceiling.py::test_reproduction_ceiling_skips_seeded_step` (empty `SchoolState` → seeded SSB → assert the gate multiplier is NOT applied on the seeded step). Read that test first and mirror it.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_depensation_gate.py -k "gate_off or gate_on or seeded" -q`
Expected: `test_gate_on_changes_cod_recruitment` fails (gate not wired → output identical); byte-identical passes trivially (also not wired). After wiring, both meaningful.

- [ ] **Step 3: Add the wiring block** in `reproduction.py` after the thermal-gate block (~line 190)

```python
    # Recruitment depensation / Allee gate (SSB-dependent, not step-dependent). Inert unless
    # enabled; skipped on seeded steps so the SSB=0 bootstrap can't be trapped, like the other gates.
    if config.depensation_gate_enabled is not None:
        from osmose.engine.processes.depensation_gate import depensation_factor

        assert config.depensation_s50 is not None  # invariant: set together in _load_depensation_gate
        assert config.depensation_theta is not None
        dfac = depensation_factor(
            ssb, config.depensation_s50, config.depensation_theta, config.depensation_gate_enabled
        )
        for sp in range(n_sp):
            if config.depensation_gate_enabled[sp] and not seeded_this_step[sp]:
                n_eggs[sp] *= dfac[sp]
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_depensation_gate.py -q`
Expected: all passed (CI-skipped integration tests run locally, not on CI).

- [ ] **Step 5: Verify pyright clean**

Run: `.venv/bin/pyright --pythonversion 3.12 osmose/engine/processes/reproduction.py`
Expected: 0 errors (the asserts narrow `depensation_s50`/`depensation_theta`).

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/reproduction.py tests/test_depensation_gate.py
git commit -m "feat(engine): wire depensation gate into reproduction (skipped when seeded)"
```

---

## Task 4: Java guard + schema registration

**Files:**
- Modify: `osmose/runner.py::java_engine_block_reason` (new check for the depensation gate)
- Modify: `osmose/schema/species.py` (`OsmoseField` entries for the 4 keys)
- Test: `tests/test_depensation_gate.py` (Java-guard test); schema strict-validation is exercised via Task 5's overlay test.

**Interfaces:**
- Produces: `java_engine_block_reason` returns a non-None reason when `reproduction.depensation.gate.enabled=true`.

- [ ] **Step 1: Read the precedents**

Read `osmose/runner.py::java_engine_block_reason` (the `ltl.depletable.enabled` string-check, ~lines 28-32) and `tests/test_baltic_a2_demo.py::test_a2_blocks_java_engine`. Read the thermal-gate `OsmoseField` entries in `osmose/schema/species.py` (~594/655/666).

- [ ] **Step 2: Write the failing Java-guard test**

```python
# append to tests/test_depensation_gate.py
from osmose.runner import java_engine_block_reason


def test_java_engine_blocked_for_depensation_gate():
    reason = java_engine_block_reason(
        {"reproduction.depensation.gate.enabled": "true"}, jar_version="4.4.1"
    )
    assert reason is not None and "depensation" in reason.lower()


def test_java_engine_not_blocked_when_gate_off():
    reason = java_engine_block_reason({"reproduction.depensation.gate.enabled": "false"}, jar_version="4.4.1")
    # off => the gate itself does not block (other guards may still apply for other configs)
    assert reason is None or "depensation" not in (reason or "").lower()
```

(Match the real `java_engine_block_reason` signature — check whether it takes `jar_version` as shown in `test_a2_blocks_java_engine` and adapt.)

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_depensation_gate.py -k java -q`
Expected: FAIL (no depensation check yet).

- [ ] **Step 4: Add the guard** in `java_engine_block_reason` (mirror the `ltl.depletable.enabled` check)

```python
    if _truthy(config.get("reproduction.depensation.gate.enabled")):
        return (
            "The recruitment depensation/Allee gate is a Python-engine feature the "
            "Java engine ignores; run this config on the Python engine."
        )
```

(Use the same truthiness helper the function already uses for `ltl.depletable.enabled`.)

- [ ] **Step 5: Add schema entries** in `osmose/schema/species.py` (mirror the thermal-gate `OsmoseField`s)

```python
    OsmoseField(key_pattern="reproduction.depensation.gate.enabled", type=bool, default=False),
    OsmoseField(key_pattern="reproduction.depensation.gate.species.enabled.sp{idx}", type=bool, default=False),
    OsmoseField(key_pattern="reproduction.depensation.gate.s50.sp{idx}", type=float, min=0.0),
    OsmoseField(key_pattern="reproduction.depensation.gate.theta.sp{idx}", type=float, min=1.0),
```

(Match the exact `OsmoseField` constructor signature used by the thermal-gate entries — field names may differ, e.g. `minimum`/`min`.)

- [ ] **Step 6: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_depensation_gate.py -k java -q && .venv/bin/python -m pytest tests/test_schema_engine_key_parity.py -q`
Expected: java tests pass; schema-parity green (new keys are engine-readable).

- [ ] **Step 7: Commit**

```bash
git add osmose/runner.py osmose/schema/species.py tests/test_depensation_gate.py
git commit -m "feat: Java guard + schema registration for depensation gate"
```

---

## Task 5: Config overlay scaffold + demo tests

**Files:**
- Create: `data/baltic_depensation/` (DRY overlay on `data/baltic` — only the changed keys)
- Create: `tests/test_baltic_depensation_demo.py`
- Modify: whatever registers demos/presets (mirror how `baltic_a2` is registered — grep `baltic_a2` to find it)

**Interfaces:**
- Consumes: the gate feature (Tasks 1-4).
- Produces: a loadable `baltic_depensation` demo. **Operating-point values (S50/θ/larval-M scale) are PLACEHOLDERS** marked `# TBD from Task 8 sweep`; the overlay must LOAD, be Java-blocked, and pass strict validation with the placeholders.

- [ ] **Step 1: Find the overlay/preset mechanism**

Run: `grep -rn "baltic_a2" osmose/ data/ tests/ | grep -iv test_ | head`. Read how `data/baltic_a2` overlays `data/baltic` and how `osmose_demo`/preset registry lists it. Mirror exactly.

- [ ] **Step 2: Write the failing demo tests**

```python
# tests/test_baltic_depensation_demo.py
import tempfile
from pathlib import Path
import pytest
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.runner import java_engine_block_reason


def _load():
    tmp = Path(tempfile.mkdtemp())
    return dict(OsmoseConfigReader().read(str(osmose_demo("baltic_depensation", tmp)["config_file"])))


def test_overlay_loads_and_enables_gate():
    cfg = _load()
    assert cfg["reproduction.depensation.gate.enabled"] == "true"
    assert cfg["reproduction.depensation.gate.species.enabled.sp0"] == "true"


def test_overlay_blocks_java_engine():
    assert java_engine_block_reason(_load(), jar_version="4.4.1") is not None


def test_overlay_passes_strict_validation():
    # mirror tests/test_baltic_a2_demo.py::test_a2_passes_strict_validation
    from osmose.engine.config import validate_config  # adjust to the real strict-validate entry
    validate_config(_load(), "error")
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_depensation_demo.py -q`
Expected: FAIL (demo `baltic_depensation` not registered).

- [ ] **Step 4: Create the overlay**

Create `data/baltic_depensation/` mirroring `data/baltic_a2`'s overlay structure. Set only:
```
reproduction.depensation.gate.enabled;true
reproduction.depensation.gate.species.enabled.sp0;true
reproduction.depensation.gate.s50.sp0;90000        # TBD from Task 8 sweep
reproduction.depensation.gate.theta.sp0;4.0        # TBD from Task 8 sweep
# larval-M override (STORED = engine_value x 24 per the ndt-migration gotcha) # TBD from Task 8 sweep
mortality.additional.larva.rate.sp0;<scale*15.0*24>   # placeholder scale=1.0 => 360.0
```
Register the demo the same way `baltic_a2` is registered (Step 1 finding).

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_depensation_demo.py -q`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add data/baltic_depensation tests/test_baltic_depensation_demo.py <demo-registry-file>
git commit -m "feat: baltic_depensation overlay scaffold (operating point TBD from sweep)"
```

---

## Task 6: Placement harness (Unit 2 grid sweep)

**Files:**
- Create: `scripts/calibrate_depensation_bistability.py`
- Test: `tests/test_depensation_placement_smoke.py` (CI-skipped 1-point smoke)

**Interfaces:**
- Consumes: the gate (Tasks 1-4); `scripts/baltic_bistability_chunk0.py` helpers (`read_base_config`, `read_base_larva_rates`, `larva_scale_override`, `cod_rich_seeding`, `cod_poor_seeding`, `warmstart_override`, `basins_differ`, `is_stationary`); `PythonEngine().run_in_memory().ssb()`.
- Produces: a grid-sweep runner that classifies each `(scale, S50, θ)` point as GO/AMBIGUOUS/NO-GO with SSB-based, full-horizon stability.

- [ ] **Step 1: Read the harness helpers**

Read `scripts/baltic_bistability_chunk0.py` (the named helpers above) and `scripts/spikes/depensation_bistability_spike.py` (the monkeypatch spike — the placement harness replaces the monkeypatch with the real config-plumbed gate via overrides).

- [ ] **Step 2: Implement the harness** (key structure — full code)

```python
#!/usr/bin/env python
"""Unit 2 placement harness: grid-sweep (larval-scale x S50 x theta) with the REAL
config-plumbed depensation gate, using the warm-start reciprocal-invasion classifier +
a two-tier SSB stability discriminator (50yr coarse screen + 150-200yr arbiter).
Emits GO / AMBIGUOUS / NO-GO per the spec's success criteria."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from baltic_bistability_chunk0 import (  # noqa: E402
    cod_poor_seeding,
    cod_rich_seeding,
    larva_scale_override,
    read_base_config,
    read_base_larva_rates,
    warmstart_override,
)

COD = 0
SCALES = [0.6, 0.75, 0.85, 0.90, 0.95, 1.0]
S50_GRID = [30_000.0, 60_000.0, 90_000.0, 120_000.0]
THETA_GRID = [2.0, 4.0]
SEEDS = (0, 1, 2)
SCREEN_YEARS = 50
ARBITER_YEARS = 175
GO_BAND = (40_000.0, 300_000.0)
COLLAPSE_T = 6_000.0  # classify_state collapse_frac(0.05) x Bpa(120k)


def gate_overrides(s50, theta):
    return {
        "reproduction.depensation.gate.enabled": "true",
        "reproduction.depensation.gate.species.enabled.sp0": "true",
        "reproduction.depensation.gate.s50.sp0": str(s50),
        "reproduction.depensation.gate.theta.sp0": str(theta),
        "output.ssb.enabled": "true",
    }


def cod_ssb_series(base, base_rates, scale, rich, s50, theta, seed, n_year, gate=True):
    from osmose.engine import PythonEngine

    raw = {**base, "simulation.time.nyear": str(n_year)}
    raw.update(warmstart_override(True))
    raw.update(cod_rich_seeding() if rich else cod_poor_seeding())
    raw.update(larva_scale_override(scale, base_rates))
    if gate:
        raw.update(gate_overrides(s50, theta))
    else:
        raw["output.ssb.enabled"] = "true"
    return PythonEngine().run_in_memory(raw, seed=seed).ssb()["cod"].to_numpy(dtype=float)


def decade_means(series):
    n = len(series)
    d = max(1, n // 10)
    return [float(np.mean(series[i : i + 10])) for i in range(0, n - 9, 10)] or [float(np.mean(series))]


def passes_coarse_screen(series):
    dm = decade_means(series)
    if len(dm) < 2:
        return GO_BAND[0] <= dm[-1] <= GO_BAND[1]
    decline = (dm[-2] - dm[-1]) / max(dm[-2], 1.0)
    return decline <= 0.10 and GO_BAND[0] <= dm[-1] <= GO_BAND[1]


def arbiter_stable(series):
    from baltic_bistability_chunk0 import is_stationary

    tail = series[-20:]
    mean = float(np.mean(tail))
    if not (GO_BAND[0] <= mean <= GO_BAND[1]):
        return False, mean
    cv = float(np.std(tail) / (mean + 1.0))
    trend = abs(float(np.polyfit(range(len(tail)), tail, 1)[0])) / (mean + 1.0)
    return is_stationary(cv, trend) and float(np.min(series[len(series) // 3 :])) > COLLAPSE_T, mean


def classify_point(base, base_rates, scale, s50, theta):
    # rich vs poor at 50yr coarse screen (median over seeds)
    rich = np.median([cod_ssb_series(base, base_rates, scale, True, s50, theta, s, SCREEN_YEARS) for s in SEEDS], axis=0)
    poor = np.median([cod_ssb_series(base, base_rates, scale, False, s50, theta, s, SCREEN_YEARS) for s in SEEDS], axis=0)
    from baltic_bistability_chunk0 import basins_differ

    split = basins_differ(np.mean(rich[-10:]), np.mean(poor[-10:]), 0.5) if hasattr_basins() else (
        abs(np.mean(rich[-10:]) - np.mean(poor[-10:])) / max(np.mean(rich[-10:]), np.mean(poor[-10:]), 1.0) > 0.5
    )
    if not split or not passes_coarse_screen(rich):
        return {"scale": scale, "s50": s50, "theta": theta, "verdict": "no-split", "rich": float(np.mean(rich[-10:]))}
    # arbiter: 175yr re-run at the rich IC
    arb = np.median([cod_ssb_series(base, base_rates, scale, True, s50, theta, s, ARBITER_YEARS) for s in SEEDS], axis=0)
    stable, mean = arbiter_stable(arb)
    return {
        "scale": scale, "s50": s50, "theta": theta,
        "verdict": "GO" if stable else "arbiter-fail",
        "healthy_ssb_mean": mean, "poor": float(np.mean(poor[-10:])),
    }


def hasattr_basins():
    try:
        from baltic_bistability_chunk0 import basins_differ  # noqa: F401
        return True
    except Exception:
        return False


def main():
    base = read_base_config()
    base_rates = read_base_larva_rates(base)
    results = []
    for scale in SCALES:
        for s50 in S50_GRID:
            for theta in THETA_GRID:
                r = classify_point(base, base_rates, scale, s50, theta)
                results.append(r)
                print(r, flush=True)
    gos = [r for r in results if r["verdict"] == "GO"]
    print("\n=== GO points ===")
    for r in sorted(gos, key=lambda r: (abs(r["healthy_ssb_mean"] - 120_000), 1 - r["scale"])):
        print(r)
    return results


if __name__ == "__main__":
    main()
```

(Note: consult the real `basins_differ`/`is_stationary`/`classify_state` signatures and simplify `classify_point` to call them directly rather than the `hasattr` guard — that guard is a defensive placeholder; replace with the real import once the signatures are confirmed in Step 1.)

- [ ] **Step 3: Write a CI-skipped 1-point smoke test**

```python
# tests/test_depensation_placement_smoke.py
import os
import sys
from pathlib import Path
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="multi-decade real-engine run")
def test_classify_point_runs_and_returns_verdict():
    from calibrate_depensation_bistability import classify_point, read_base_config, read_base_larva_rates

    base = read_base_config()
    r = classify_point(base, read_base_larva_rates(base), 0.85, 60000.0, 4.0)
    assert "verdict" in r
```

- [ ] **Step 4: Run the smoke test locally**

Run: `.venv/bin/python -m pytest tests/test_depensation_placement_smoke.py -q` (local; several minutes)
Expected: 1 passed (a verdict is returned; value not asserted).

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check scripts/calibrate_depensation_bistability.py tests/test_depensation_placement_smoke.py
.venv/bin/ruff format scripts/calibrate_depensation_bistability.py tests/test_depensation_placement_smoke.py
git add scripts/calibrate_depensation_bistability.py tests/test_depensation_placement_smoke.py
git commit -m "feat: depensation bistability placement harness (Unit 2 grid sweep)"
```

---

## Task 7: Hysteresis validation harness (Unit 3)

**Files:**
- Create: `scripts/validate_depensation_hysteresis.py`
- Test: `tests/test_depensation_hysteresis_smoke.py` (CI-skipped)

**Interfaces:**
- Consumes: the gate; `mortality.fishing.rate.byyear.file.sp0` (byyear-F, per `scripts/spikes/ssb_f_hindcast_spike.py`); `osmose/validation/fmsy_sweep.py::fishing_override` (base-F resolution).
- Produces: (a) same-scale no-Allee warm-start control; (b) a quasi-static F-ramp (≥10 levels to ≥30× base, per-level equilibration) with a gate-off control ramp; reports F_collapse/F_recover in real-F terms.

- [ ] **Step 1: Read the byyear-F + base-F precedents**

Read `scripts/spikes/ssb_f_hindcast_spike.py` (byyear-F CSV via `np.savetxt` → `mortality.fishing.rate.byyear.file.sp0`) and `osmose/validation/fmsy_sweep.py::fishing_override` (resolve base F = `fisheries.rate.base.fsh0`=0.08 mode-agnostically).

- [ ] **Step 2: Implement the validation harness**

```python
#!/usr/bin/env python
"""Unit 3 validation: (1) same-scale no-Allee warm-start control (must be monostable);
(2) quasi-static fishing-hysteresis F-ramp (>=10 levels to >=30x base, per-level
equilibration) with a gate-off control ramp; report F_collapse/F_recover in real-F terms
for the reachability check (F_collapse <= historical peak ~2.3, F_recover > present ~0.16)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

BASE_F = 0.08  # fisheries.rate.base.fsh0 for cod; confirm via fishing_override(config, 0)
LEVELS = [0.5, 1, 2, 4, 8, 12, 16, 20, 25, 30]  # x base; >=30x brackets historical peak ~2.3 (28.6x)
DWELL_CAP_Y = 75
CONV_TOL = 0.05  # |decade-over-decade rel change| < 5%


def equilibrate_level(base, base_rates, scale, s50, theta, f_mult, seed, gate=True, start_ssb=None):
    # Run one F level to convergence (cap DWELL_CAP_Y); return (mean_ssb, converged, series).
    # F level applied via byyear-F: constant column = f_mult * BASE_F for DWELL_CAP_Y years.
    from osmose.engine import PythonEngine

    tmp = Path(tempfile.mkdtemp())
    f_csv = tmp / "cod_f.csv"
    np.savetxt(f_csv, np.full(DWELL_CAP_Y, f_mult * BASE_F))
    from calibrate_depensation_bistability import gate_overrides
    from baltic_bistability_chunk0 import cod_rich_seeding, larva_scale_override, warmstart_override

    raw = {**base, "simulation.time.nyear": str(DWELL_CAP_Y), "output.ssb.enabled": "true",
           "mortality.fishing.rate.byyear.file.sp0": str(f_csv)}
    raw.update(warmstart_override(True)); raw.update(cod_rich_seeding()); raw.update(larva_scale_override(scale, base_rates))
    if gate:
        raw.update(gate_overrides(s50, theta))
    s = PythonEngine().run_in_memory(raw, seed=seed).ssb()["cod"].to_numpy(dtype=float)
    dm = [float(np.mean(s[i:i+10])) for i in range(0, len(s) - 9, 10)]
    converged = len(dm) >= 2 and abs(dm[-1] - dm[-2]) / max(dm[-2], 1.0) < CONV_TOL
    return float(np.mean(s[-10:])), converged, s


def ramp(base, base_rates, scale, s50, theta, seed, gate=True):
    up = [(m, *equilibrate_level(base, base_rates, scale, s50, theta, m, seed, gate)[:2]) for m in LEVELS]
    down = [(m, *equilibrate_level(base, base_rates, scale, s50, theta, m, seed, gate)[:2]) for m in reversed(LEVELS)]
    return up, down


def main(scale, s50, theta):
    from baltic_bistability_chunk0 import read_base_config, read_base_larva_rates
    base = read_base_config(); base_rates = read_base_larva_rates(base)
    print("=== no-Allee control (must be monostable at this scale) ===")
    # rich vs poor with gate OFF at the chosen scale -> expect same basin (see calibrate harness)
    print("=== depensation F-ramp ===")
    up, down = ramp(base, base_rates, scale, s50, theta, seed=0, gate=True)
    print("up:", up); print("down:", down)
    print("=== control F-ramp (gate off) ===")
    cup, cdown = ramp(base, base_rates, scale, s50, theta, seed=0, gate=False)
    print("control up:", cup); print("control down:", cdown)
    # F_collapse: lowest up-leg f_mult where SSB drops below COLLAPSE_T; F_recover: highest down-leg
    # f_mult where SSB climbs back above GO band. Report both x base and absolute (x0.08).


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--scale", type=float, default=0.85)
    p.add_argument("--s50", type=float, default=90000.0); p.add_argument("--theta", type=float, default=4.0)
    a = p.parse_args(); main(a.scale, a.s50, a.theta)
```

(Finalize F_collapse/F_recover extraction + the AMBIGUOUS-when-fold-level-uncapped rule per spec Unit 3 §2; flag any fold-adjacent level that hits `DWELL_CAP_Y` without converging.)

- [ ] **Step 3: CI-skipped smoke test** (one level equilibrates)

```python
# tests/test_depensation_hysteresis_smoke.py
import os, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="multi-decade real-engine run")
def test_equilibrate_level_returns_tuple():
    from validate_depensation_hysteresis import equilibrate_level
    from baltic_bistability_chunk0 import read_base_config, read_base_larva_rates
    base = read_base_config()
    mean, conv, series = equilibrate_level(base, read_base_larva_rates(base), 0.85, 90000.0, 4.0, 1.0, 0)
    assert mean >= 0.0 and isinstance(conv, bool) and len(series) > 0
```

- [ ] **Step 4: Run smoke locally + lint + commit**

```bash
.venv/bin/python -m pytest tests/test_depensation_hysteresis_smoke.py -q
.venv/bin/ruff check scripts/validate_depensation_hysteresis.py tests/test_depensation_hysteresis_smoke.py
.venv/bin/ruff format scripts/validate_depensation_hysteresis.py tests/test_depensation_hysteresis_smoke.py
git add scripts/validate_depensation_hysteresis.py tests/test_depensation_hysteresis_smoke.py
git commit -m "feat: depensation hysteresis validation harness (Unit 3)"
```

---

## Task 8 (ANALYSIS — runs AFTER the green merge, not a TDD task)

**This is the long compute run, not CI code.** It runs the harnesses from Tasks 6–7 and produces the scientific outcome. It is explicitly OUTSIDE the "merge once green" boundary.

- [ ] **Step 1:** Run `scripts/calibrate_depensation_bistability.py` (288–480 multi-decade sims; hours). Capture the GO/AMBIGUOUS/NO-GO map.
- [ ] **Step 2:** If GO points exist, select the operating point (closest-to-Bpa healthy basin, then scale closest to 1.0). Run `scripts/validate_depensation_hysteresis.py --scale --s50 --theta` at that point (no-Allee control + F-ramp, 9,000–15,000 sim-years).
- [ ] **Step 3:** Apply the F-reachability gate: F_collapse ≤ ~2.3, F_recover > ~0.16 (in absolute F). Classify final outcome GO / AMBIGUOUS / NO-GO.
- [ ] **Step 4:** Write `docs/diagnostics/2026-07-16-depensation-placement.md` (mapped region, chosen point, hysteresis loop vs control, F-reachability, larval-M departure from baseline, SP2-inherited assumptions).
- [ ] **Step 5:** If GO, update `data/baltic_depensation` with the real S50/θ/larval-M (larval-M stored ×24) and re-run `tests/test_baltic_depensation_demo.py`. If AMBIGUOUS/NO-GO, document honestly (the gate feature still ships).
- [ ] **Step 6:** Commit the diagnostics doc + overlay update; report the outcome.

---

## Self-Review Notes

- **Spec coverage:** Unit 1 → Tasks 1-4; Unit 4 overlay → Task 5; Unit 2 harness → Task 6; Unit 3 validation → Task 7; the placement RUN + diagnostics + overlay finalization → Task 8. Success Criteria #1 (gate shipped) = Tasks 1-5; #2/#3 (bistable + F-reachable) = Tasks 6-8.
- **Merge-green boundary:** Tasks 1-7 (all CI-testable). Task 8 is the post-merge analysis.
- **Determinism:** Task 3's byte-identical test is the guard; Task 2's fixture update keeps existing tests green.
