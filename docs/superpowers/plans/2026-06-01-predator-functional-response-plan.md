# Predator Functional Response (aggregate, opt-in) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selectable, opt-in, per-predator-species Holling functional-response forms (type-I / type-II / type-III) to OSMOSE's live predation kernel, default-off and bit-exact-preserving when off, plus a Baltic calibration phase (phase-14) validated at the process level.

**Architecture:** Two new per-species config keys (`predation.functional.response.shape.sp{i}` + `…halfsat.sp{i}`) parse into two `EngineConfig` arrays (`fr_shape: int32[n_total]`, `fr_halfsat: float64[n_total]`) built exactly like `recruitment_shepherd_beta`. The arrays are threaded into the single live predation injection point in **both** kernels — the Python fallback `_apply_predation_for_school` (`config` in scope) and the numba `_apply_predation_numba` (threaded as positional args through three `@njit` callers, mirroring `ingestion_rate`). At the injection point, `eaten_total = min(total_available, max_eatable)` is replaced by a branch: `fr_shape==1` keeps the verbatim existing statement (bit-exact); `==2/3` apply the ration-capped Holling form `eaten_total = max_eatable · min(g_form(r), min(r,1))`. Delivered as two PRs (PR-A engine capability, PR-B Baltic calibration), mirroring PR #50.

**Tech Stack:** Python 3.12, NumPy, Numba (`@njit`), pytest, scipy `differential_evolution` (calibration), ruff. Run tests with `.venv/bin/python -m pytest`.

**Reference spec:** `docs/superpowers/specs/2026-05-31-predator-functional-response-design.md` (round-5 verified).

---

## File Structure

**PR-A (engine capability):**
- `osmose/schema/predation.py` — MODIFY: add two `OsmoseField`s (`functional.response.shape`, `…halfsat`).
- `osmose/engine/config.py` — MODIFY: parse both keys (focal `build_engine_config`), enum→int map, focal-dict entries, focal+bkg concat, no-bkg path, `EngineConfig` field declarations, `per_species_arrays` registration, `from_dict` threading, the parse-time validation (enum + halfsat-required-iff + bound).
- `osmose/engine/processes/background.py` — MODIFY: parse `…shape.sp14/15` + `…halfsat.sp14/15` into the background portion (enum→int, identical codes).
- `osmose/engine/processes/mortality.py` — MODIFY: branch at both injection points; add `fr_shape`/`fr_halfsat` to the `_apply_predation_numba` signature + the three njit callers' signatures + the three numba call sites + the one Python call site.
- `tests/test_engine_functional_response.py` — CREATE: curve math, conservation, config/parse, kernel behavior, numba-vs-Python parity, diagnostic unit tests.
- `docs/` config reference — MODIFY: document both keys + the type-II destabilization / type-III recommendation note.

**PR-B (Baltic calibration + science):**
- `phase13_results.json` (calibration output dir) — CREATE: commit the PR #50 phase-13 result as the phase-14 base.
- `scripts/calibrate_baltic.py` — MODIFY: `get_phase14_params()` + `phase == "14"` branch with phase-13 inheritance.
- `scripts/evaluate_calibration_vs_ices.py` — MODIFY: add `shepherd-fr` to `--mode` choices; inject `shape.sp{0,5,14,15}=type3` + calibrated halfsat; report objective + ICES delta + process diagnostic.
- `scripts/fr_process_diagnostic.py` (or a function in the eval script) — CREATE: FR-on vs FR-off realized-mortality diagnostic at diet-tracking width 16.

---

# PART A — Engine Capability (PR-A)

## Task A1: Config schema fields

**Files:**
- Modify: `osmose/schema/predation.py`
- Test: `tests/test_engine_functional_response.py` (create)

- [ ] **Step 1: Inspect the existing predation schema and the recruitment `shape` field for the house style**

Run: `grep -n "OsmoseField\|key_pattern\|recruitment" osmose/schema/predation.py | head -40`
Then `grep -rn "stock.recruitment.shape" osmose/schema/` to find the recruitment shape field to mirror its `OsmoseField` construction (category, type, default, bounds, description).

- [ ] **Step 2: Add the two fields**

Add two `OsmoseField`s to `osmose/schema/predation.py` (match the exact constructor signature used by neighboring fields in that file — copy a nearby per-species field and edit). The two fields:

```python
# Predator functional response (post-parity; opt-in, default type1 ≡ existing behavior)
OsmoseField(
    key_pattern="predation.functional.response.shape.sp{i}",
    label="Functional response shape",
    field_type="enum",
    choices=["type1", "type2", "type3"],
    default="type1",
    category="predation",
    description=(
        "Holling functional-response form for this predator. type1 (default) = "
        "existing linear-with-ration-ceiling behavior (bit-exact). type2 = saturating "
        "(Holling disc; classically destabilizing — paradox of enrichment). type3 = "
        "sigmoid with low-density prey refuge (recommended/validated form)."
    ),
),
OsmoseField(
    key_pattern="predation.functional.response.halfsat.sp{i}",
    label="Functional response half-saturation (K)",
    field_type="float",
    default=None,
    min_value=0.1,
    max_value=5.0,
    category="predation",
    description=(
        "Dimensionless ration-relative half-saturation K for type2/type3. "
        "Required when shape != type1. Range [0.1, 5.0]. Well-scaled for DE, "
        "not a transferable biological constant."
    ),
),
```

(If `OsmoseField` does not have an `enum`/`choices` kwarg, follow whatever the recruitment-type field uses — it has the same `{none, beverton_holt, …}` enum, so copy that field's mechanism exactly.)

- [ ] **Step 3: Write a schema-registration test**

In a new `tests/test_engine_functional_response.py`:

```python
from osmose.schema.registry import get_field  # adjust import to the actual registry accessor

def test_fr_schema_fields_registered():
    shape = get_field("predation.functional.response.shape.sp{i}")
    assert shape.default == "type1"
    assert set(shape.choices) == {"type1", "type2", "type3"}
    halfsat = get_field("predation.functional.response.halfsat.sp{i}")
    assert halfsat.min_value == 0.1
    assert halfsat.max_value == 5.0
```

Run: `grep -n "def get_field\|def get\b\|registry" osmose/schema/registry.py | head` to confirm the accessor name; adjust the import/call if different.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py::test_fr_schema_fields_registered -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/schema/predation.py tests/test_engine_functional_response.py
git commit -m "feat(fr): add predator functional-response schema fields"
```

---

## Task A2: Parse keys + enum→int map + validation (focal path)

**Files:**
- Modify: `osmose/engine/config.py` (parse near `:535-549`; focal dict near `:562`)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Write failing validation tests**

Add to the test file. These assert the parse-time validation contract from spec §2:

```python
import numpy as np
import pytest
from osmose.engine.config import build_engine_config  # adjust to the actual builder used by from_dict

def _minimal_focal_cfg(**overrides):
    # Build the smallest cfg dict that build_engine_config accepts for a 2-species, no-background setup.
    # Look at an existing config.py parse test (tests/test_engine_config*.py) and copy its fixture,
    # then apply overrides. Keep the helper in this test module.
    base = _load_existing_minimal_cfg_fixture()  # see Step 2 note
    base.update(overrides)
    return base

def test_fr_halfsat_required_when_shape_not_type1():
    cfg = _minimal_focal_cfg(**{"predation.functional.response.shape.sp0": "type3"})
    with pytest.raises(ValueError, match="is required when"):
        build_engine_config(cfg)

def test_fr_halfsat_out_of_range_raises():
    cfg = _minimal_focal_cfg(**{
        "predation.functional.response.shape.sp0": "type3",
        "predation.functional.response.halfsat.sp0": 0.0,
    })
    with pytest.raises(ValueError):
        build_engine_config(cfg)

def test_fr_shape_invalid_enum_raises():
    cfg = _minimal_focal_cfg(**{"predation.functional.response.shape.sp0": "type9"})
    with pytest.raises(ValueError):
        build_engine_config(cfg)
```

Note for Step 1: find an existing parse-level test that builds a minimal config dict (`grep -rln "build_engine_config\|from_dict" tests/ | head`); reuse its fixture helper rather than inventing one. If the only entry point is `EngineConfig.from_dict`, call that instead of `build_engine_config` and adjust all three tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "fr_halfsat or fr_shape_invalid" -v`
Expected: FAIL (keys ignored, no validation yet)

- [ ] **Step 3: Add the parse + enum-map + validation**

In `osmose/engine/config.py`, immediately after the recruitment block (after `:549`), add (mirror `_species_*_optional` helpers used there — confirm exact helper for enum/string parse with `grep -n "_species_str_optional\|_species_enum\|_species_float_optional" osmose/engine/config.py`):

```python
    # Predator functional response (post-parity; opt-in, default type1 ≡ existing)
    fr_shape_str = _species_str_optional(
        cfg,
        "predation.functional.response.shape.sp{i}",
        n_sp,
        default="type1",
        allowed={"type1", "type2", "type3"},
    )
    fr_halfsat_focal = _species_float_optional(
        cfg, "predation.functional.response.halfsat.sp{i}", n_sp, default=_FR_HALFSAT_SENTINEL
    )
    _FR_SHAPE_CODE = {"type1": 1, "type2": 2, "type3": 3}
    fr_shape_focal = np.array([_FR_SHAPE_CODE[s] for s in fr_shape_str], dtype=np.int32)
    for i in range(n_sp):
        if fr_shape_str[i] != "type1":
            hv = cfg.get(f"predation.functional.response.halfsat.sp{i}")
            if hv is None:
                raise ValueError(
                    f"predation.functional.response.halfsat.sp{i} is required when "
                    f"predation.functional.response.shape.sp{i} = {fr_shape_str[i]}"
                )
            if not (0.1 <= float(hv) <= 5.0):
                raise ValueError(
                    f"predation.functional.response.halfsat.sp{i} = {hv} out of range [0.1, 5.0]"
                )
```

Add the sentinel constant near the top of the module (a value type1 never reads, kept finite to avoid NaN propagation if mis-indexed):

```python
_FR_HALFSAT_SENTINEL = 1.0  # inert: type1 never reads fr_halfsat
```

Use the actual optional-enum helper name. If no `_species_str_optional` with an `allowed=` kwarg exists, copy the mechanism `recruitment_type` uses at `:529-534` (which validates against an `allowed` set) and reuse it.

- [ ] **Step 4: Add focal-dict entries**

In the focal-dict assembled around `:562` (where `"focal_recruitment_shepherd_beta": recruitment_shepherd_beta` lives), add:

```python
        "focal_fr_shape": fr_shape_focal,
        "focal_fr_halfsat": fr_halfsat_focal,
```

- [ ] **Step 5: Run validation tests**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "fr_halfsat or fr_shape_invalid" -v`
Expected: the `is required when` test PASS, out-of-range PASS, invalid-enum PASS. (Full from_dict wiring lands in A4; if the builder errors earlier for unrelated reasons, defer the green to A4 and note it.)

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_functional_response.py
git commit -m "feat(fr): parse functional-response keys with strict validation (focal path)"
```

---

## Task A3: Background-predator parse (`background.py`)

**Files:**
- Modify: `osmose/engine/processes/background.py` (near `:378` numbering; find where background per-species config is read)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Locate how background.py reads per-species config keys**

Run: `grep -n "ingestion\|sp{\|species_id\|n_focal\|recruitment\|config.get\|\.sp" osmose/engine/processes/background.py | head -40`
Identify where background predators read their config (the `sp14`/`sp15` keys) and where `species_id = self._n_focal + bkg_idx` is set (`:378`). Background predators must produce `fr_shape`/`fr_halfsat` codes **identical** to the focal mapping.

- [ ] **Step 2: Write a failing test for background enum→int on the runtime slot**

```python
def test_fr_background_enum_maps_to_runtime_slot():
    # Baltic-like: 8 focal + 2 background; sp14 -> runtime slot 8, sp15 -> slot 9.
    cfg = _baltic_like_cfg_with_background(**{
        "predation.functional.response.shape.sp14": "type3",
        "predation.functional.response.halfsat.sp14": 1.0,
    })
    ecfg = _build(cfg)  # from_dict / build path that includes background
    assert ecfg.fr_shape[8] == 3
    assert ecfg.fr_halfsat[8] == 1.0
    assert ecfg.fr_shape[9] == 1  # untouched background default
```

Reuse an existing background-enabled config fixture (`grep -rln "n_background\|background" tests/ | head`); the Baltic example config under `data/baltic/` is the canonical 8+2 case.

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py::test_fr_background_enum_maps_to_runtime_slot -v`
Expected: FAIL (background arrays don't exist yet / wrong length)

- [ ] **Step 4: Parse FR keys in background.py**

Where background.py builds its per-predator arrays, read the two keys for each background species using the **same** `_FR_SHAPE_CODE` mapping and the **same** validation as A2 (enum membership; halfsat required-iff; bound). Emit a `fr_shape` (int32) and `fr_halfsat` (float64) array of length `n_background`, ordered by `bkg_idx` so they concatenate at runtime slots `n_focal + bkg_idx`. Import or duplicate `_FR_SHAPE_CODE`/`_FR_HALFSAT_SENTINEL` from `config.py` (prefer import to keep one source of truth). Background default shape = `type1` (code 1); default halfsat = sentinel.

- [ ] **Step 5: Run the test**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py::test_fr_background_enum_maps_to_runtime_slot -v`
Expected: PASS (after A4 wires the concat — if the array isn't surfaced on `EngineConfig` yet, this goes green in A4; note it).

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/background.py tests/test_engine_functional_response.py
git commit -m "feat(fr): parse functional-response keys on background-predator path"
```

---

## Task A4: EngineConfig fields, concat, registration, from_dict threading

**Files:**
- Modify: `osmose/engine/config.py` (field decl near `:1221`; concat near `:776`/`:826`; `per_species_arrays` near `:1449`; from_dict near `:1641`/`:1937`)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Write a failing sizing/registration test**

```python
def test_fr_arrays_sized_n_total_and_registered():
    cfg = _baltic_like_cfg_with_background()  # 8 focal + 2 background
    ecfg = _build(cfg)
    assert ecfg.fr_shape.dtype == np.int32
    assert ecfg.fr_halfsat.dtype == np.float64
    assert len(ecfg.fr_shape) == ecfg.n_species + ecfg.n_background == 10
    assert len(ecfg.fr_halfsat) == 10

def test_fr_mis_sized_array_raises_in_post_init():
    cfg = _baltic_like_cfg_with_background()
    ecfg = _build(cfg)
    bad = dataclasses.replace(ecfg, fr_shape=np.ones(ecfg.n_species, dtype=np.int32))  # wrong length
    # __post_init__ runs via replace; expect the per_species_arrays length check to raise
    with pytest.raises(ValueError, match="fr_shape"):
        bad.__post_init__()  # or trigger however replace/validation is invoked in this codebase
```

(Confirm how `__post_init__` validation is invoked under `dataclasses.replace` in this codebase — `grep -n "__post_init__\|def validate\|object.__setattr__" osmose/engine/config.py`. If `replace` doesn't re-run validation, construct an `EngineConfig(...)` directly with a mis-sized array instead.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "fr_arrays_sized or fr_mis_sized" -v`
Expected: FAIL (`EngineConfig` has no `fr_shape`)

- [ ] **Step 3: Declare the two EngineConfig fields**

In the dataclass near `:1221` (after `shepherd_beta`), add:

```python
    # Predator functional response (post-parity; opt-in, default code 1 ≡ existing)
    fr_shape: NDArray[np.int32]  # per-species Holling form code: 1=type-I, 2=type-II, 3=type-III
    fr_halfsat: NDArray[np.float64]  # per-species ration-relative half-saturation K (type2/3 only)
```

- [ ] **Step 4: Concat (with-background path) — near `:776`**

Add to the with-background return dict, mirroring `recruitment_shepherd_beta`:

```python
            "fr_shape": np.concatenate(
                [focal["focal_fr_shape"], _bkg_fr_shape]
            ),
            "fr_halfsat": np.concatenate(
                [focal["focal_fr_halfsat"], _bkg_fr_halfsat]
            ),
```

where `_bkg_fr_shape` / `_bkg_fr_halfsat` come from background.py (A3). If background.py returns its arrays through the same merge dict that `bkg_ingestion` flows through, use those; otherwise default to `np.ones(n_bkg, dtype=np.int32)` / `np.full(n_bkg, _FR_HALFSAT_SENTINEL)`. **The background arrays from A3 must win** when FR is configured on sp14/sp15 — confirm the background merge supplies them.

- [ ] **Step 5: Concat (no-background path) — near `:826`**

```python
            "fr_shape": focal["focal_fr_shape"],
            "fr_halfsat": focal["focal_fr_halfsat"],
```

- [ ] **Step 6: Register in `per_species_arrays` — near `:1449`**

```python
            "shepherd_beta": self.shepherd_beta,
            "fr_shape": self.fr_shape,
            "fr_halfsat": self.fr_halfsat,
```

- [ ] **Step 7: Thread through from_dict construction — near `:1641` and `:1937`**

At `:1641` (where `recruitment_shepherd_beta = _merged["recruitment_shepherd_beta"]`), add:

```python
        fr_shape = _merged["fr_shape"]
        fr_halfsat = _merged["fr_halfsat"]
```

At the `EngineConfig(...)` constructor near `:1937` (where `shepherd_beta=recruitment_shepherd_beta,`), add:

```python
            fr_shape=fr_shape,
            fr_halfsat=fr_halfsat,
```

- [ ] **Step 8: Run sizing + registration + A2/A3 tests (now fully wired)**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -v`
Expected: all config/parse/sizing/background tests PASS.

- [ ] **Step 9: Run the config-validation warning test (allowlist sanity)**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -v`
Expected: PASS, warning-free (the AST walker auto-captures the literal-prefix keys; if not, add both prefixes to `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py` and re-run).

- [ ] **Step 10: Commit**

```bash
git add osmose/engine/config.py osmose/engine/processes/background.py tests/test_engine_functional_response.py
git commit -m "feat(fr): EngineConfig fr_shape/fr_halfsat fields, concat, registration, from_dict"
```

---

## Task A5: Kernel branch — Python fallback (`_apply_predation_for_school`)

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (signature `:335`; injection `:484`; call site `:1659`)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Write failing curve-math + conservation tests (pure, kernel-independent)**

Add a tiny reference implementation in the test to anchor exact values, then test it AND wire it to the kernel later. First the math contract:

```python
def _g_ref(r, shape, k):
    if shape == 1:
        return min(r, 1.0)
    if shape == 2:
        g = r / (r + k)
    else:
        g = (r * r) / (r * r + k * k)
    cap = min(r, 1.0)
    return min(g, cap)

@pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 5.0])
@pytest.mark.parametrize("r", [0.0, 0.01, 0.1, 0.5, 0.9, 1.0, 2.0, 10.0])
def test_fr_conservation_clamp(r, k):
    # eaten_total <= total_available  <=>  g(r) <= min(r, 1)
    for shape in (2, 3):
        assert _g_ref(r, shape, k) <= min(r, 1.0) + 1e-12

def test_fr_type2_anchor():
    assert _g_ref(1.0, 2, 1.0) == pytest.approx(0.5)  # before clamp; clamp = min(0.5, 1) = 0.5

def test_fr_type3_anchor():
    assert _g_ref(1.0, 3, 1.0) == pytest.approx(0.5)

def test_fr_type3_refuge_vs_type2_small_r():
    # type-III takes a smaller fraction at low r (zero initial slope) than type-II (slope ~1/k)
    r, k = 0.05, 1.0
    assert _g_ref(r, 3, k) < _g_ref(r, 2, k)

def test_fr_type1_is_min():
    assert _g_ref(0.3, 1, 999) == 0.3
    assert _g_ref(5.0, 1, 999) == 1.0
```

- [ ] **Step 2: Run — these pass immediately (they test the reference)**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "fr_conservation or fr_type" -v`
Expected: PASS. (This locks the math the kernel must match.)

- [ ] **Step 3: Add `fr_shape`/`fr_halfsat` params to `_apply_predation_for_school` signature**

At `:335`, add two parameters after `ingestion_rate` (keep ordering consistent with the numba version in A6):

```python
def _apply_predation_for_school(
    ...
    ingestion_rate,
    fr_shape,
    fr_halfsat,
    ...
):
```

- [ ] **Step 4: Replace the injection point at `:484` with the branch**

Replace exactly:

```python
    eaten_total = min(total_available, max_eatable)
```

with:

```python
    sp_pred = state.species_id[p_idx]
    if fr_shape[sp_pred] == 1:
        eaten_total = min(total_available, max_eatable)  # verbatim type-I (bit-exact)
    else:
        r = total_available / max_eatable
        k = fr_halfsat[sp_pred]
        if fr_shape[sp_pred] == 2:
            g_form = r / (r + k)
        else:  # type-III
            g_form = (r * r) / (r * r + k * k)
        cap = r if r < 1.0 else 1.0  # min(r, 1)
        g = g_form if g_form < cap else cap  # conservation clamp
        eaten_total = max_eatable * g
```

`sp_pred` may already be computed earlier in this function — if so, reuse it; do not recompute. The Phase-3 `success = min(eaten_total / max_eatable, 1.0)` at `:541` stays **unchanged** (it is correct for all forms). **Do not** route type-I through `max_eatable * g`: the multiply round-trip can differ by 1 ULP and break parity.

- [ ] **Step 5: Pass the arrays at the call site `:1659`**

At the `_apply_predation_for_school(...)` call (`config` in scope here), add after `config.ingestion_rate` (or wherever `ingestion_rate` is passed):

```python
                    config.fr_shape,
                    config.fr_halfsat,
```

- [ ] **Step 6: Write a kernel-level type-III behavior test (Python backend)**

```python
def test_python_kernel_type3_reduces_eaten_at_low_r():
    # Construct a single-cell scenario where total_available < max_eatable (food-limited, r<1),
    # run one predation step with shape=type1 vs shape=type3, assert type3 removes less prey biomass.
    # Reuse an existing mortality unit-test harness; grep tests for _apply_predation_for_school usage.
    ...
```

Find the existing harness: `grep -rn "_apply_predation_for_school\|run_mortality\|mortality(" tests/ | head`. Build the scenario from the closest existing predation test.

- [ ] **Step 7: Run kernel + full FR test file**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/processes/mortality.py tests/test_engine_functional_response.py
git commit -m "feat(fr): functional-response branch in Python predation kernel"
```

---

## Task A6: Kernel branch — numba (`_apply_predation_numba`) + thread through 3 njit callers

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (signature `:786`; injection `:952`; njit callers `:1040`,`:1178`,`:1341`; call sites `:1100`,`:1263`,`:1438`; Python callers near `:1626`,`:1918`)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Write a failing numba-vs-Python parity test (FR on)**

```python
@pytest.mark.parametrize("shape,k", [(2, 1.0), (3, 1.0), (3, 0.5)])
def test_numba_python_parity_fr_on(shape, k):
    # Same config + seed, run with use_numba=True and use_numba=False, FR enabled on a predator.
    # Assert final biomass per species equal within tight tolerance (the two backends must agree).
    out_numba = _run_short_sim(numba=True, fr={"sp0": (shape, k)})
    out_py = _run_short_sim(numba=False, fr={"sp0": (shape, k)})
    np.testing.assert_allclose(out_numba, out_py, rtol=1e-9, atol=0)
```

Build `_run_short_sim` from the existing engine smoke/integration tests (`grep -rn "use_numba\|PythonEngine\|run.*sim" tests/ | head`). Use the smallest existing config (e.g. a 2–3 species fixture) for speed.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py::test_numba_python_parity_fr_on -v`
Expected: FAIL (numba kernel ignores FR / TypeError on missing args)

- [ ] **Step 3: Add params to the `_apply_predation_numba` signature**

At `:786`, add after `ingestion_rate` (`:802`), matching the Python order from A5:

```python
        ingestion_rate,
        fr_shape,
        fr_halfsat,
```

- [ ] **Step 4: Replace the numba injection point at `:952`**

`sp_pred` is already computed at `:843` (`sp_pred = species_id[p_idx]`). Replace exactly:

```python
        eaten_total = min(total_available, max_eatable)
```

with:

```python
        if fr_shape[sp_pred] == 1:
            eaten_total = min(total_available, max_eatable)  # verbatim type-I (bit-exact)
        else:
            r = total_available / max_eatable
            k = fr_halfsat[sp_pred]
            if fr_shape[sp_pred] == 2:
                g_form = r / (r + k)
            else:  # type-III
                g_form = (r * r) / (r * r + k * k)
            cap = r if r < 1.0 else 1.0  # min(r, 1)
            g = g_form if g_form < cap else cap  # conservation clamp
            eaten_total = max_eatable * g
```

Phase-3 `success` at `:993` stays unchanged.

- [ ] **Step 5: Thread `fr_shape`/`fr_halfsat` into the 3 njit callers and their call sites**

For each of `_mortality_in_cell_numba` (`:1040`, call `:1100`), `_mortality_all_cells_numba` (`:1178`, call `:1263`), `_mortality_all_cells_parallel` (`:1341`, call `:1438`):
  1. Add `fr_shape, fr_halfsat` to the njit function's parameter list (placed adjacent to where `ingestion_rate` appears in its signature — search each function for `ingestion_rate`).
  2. Pass `fr_shape, fr_halfsat` at the inner `_apply_predation_numba(...)` call (adjacent to `ingestion_rate`).

- [ ] **Step 6: Thread from the two Python callers near `:1626` and `:1918`**

Find where these Python-level functions invoke the njit callers passing `config.ingestion_rate` (`grep -n "ingestion_rate" osmose/engine/processes/mortality.py` will show `:1626` and `:1918` as the Python-side passes). Add `config.fr_shape, config.fr_halfsat` adjacent to each `config.ingestion_rate` argument.

- [ ] **Step 7: Run numba-vs-Python parity + full FR file**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -v`
Expected: PASS (numba recompiles once on the additive signature change)

- [ ] **Step 8: Add NaN/guard + determinism tests**

```python
def test_fr_type3_empty_cell_no_nan():
    # type-3 predator in a cell with total_available == 0 -> guard returns before injection; no NaN.
    out = _run_short_sim(numba=True, fr={"sp0": (3, 1.0)}, force_empty_prey=True)
    assert np.all(np.isfinite(out))

def test_fr_determinism():
    a = _run_short_sim(numba=True, fr={"sp0": (3, 1.0)}, seed=42)
    b = _run_short_sim(numba=True, fr={"sp0": (3, 1.0)}, seed=42)
    np.testing.assert_array_equal(a, b)
```

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "no_nan or determinism" -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add osmose/engine/processes/mortality.py tests/test_engine_functional_response.py
git commit -m "feat(fr): functional-response branch in numba kernel + njit arg threading"
```

---

## Task A7: Bit-exact parity gate (FR off) + background-on behavior test

**Files:**
- Test: `tests/test_engine_functional_response.py` + existing Java-parity suite

- [ ] **Step 1: Identify the 12/12 Java-parity suite**

Run: `grep -rln "parity\|java" tests/ | head` and `.venv/bin/python -m pytest tests/ -k parity --collect-only -q | tail -20` to name the exact parity test ids.

- [ ] **Step 2: Run the parity suite unmodified (FR defaults off everywhere)**

Run: `.venv/bin/python -m pytest <parity-suite-path> -v`
Expected: 12/12 PASS, no baseline regeneration. (Every existing config omits the FR keys ⇒ `type1` ⇒ verbatim branch ⇒ bit-exact.)

- [ ] **Step 3: Add an explicit "type1 == absent-key" engine test**

```python
def test_fr_explicit_type1_equals_absent_key():
    out_absent = _run_short_sim(numba=True, fr=None, seed=7)
    out_type1 = _run_short_sim(numba=True, fr={"sp0": (1, None)}, seed=7)  # explicit type1, no halfsat
    np.testing.assert_array_equal(out_absent, out_type1)
```

- [ ] **Step 4: Add background-path behavior test (catches n_species sizing bug)**

```python
def test_fr_on_background_predator_changes_outcome():
    base = _run_baltic_short(seed=11, fr=None)
    fr_on = _run_baltic_short(seed=11, fr={"sp14": (3, 1.0)})  # GreySeal, runtime slot 8
    assert not np.array_equal(base, fr_on)  # background FR must actually bite
```

- [ ] **Step 5: Run**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "type1_equals or background_predator" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_engine_functional_response.py
git commit -m "test(fr): bit-exact parity-off gate + explicit-type1 + background-on behavior"
```

---

## Task A8: Process-diagnostic unit tests (the gate's falsifiable basis)

**Files:**
- Test: `tests/test_engine_functional_response.py`
- (Possibly) Modify: a diet-aggregation helper if it must be extended to include background slots

- [ ] **Step 1: Inspect diet-tracking + aggregation**

Run: `grep -n "def enable_diet_tracking\|def aggregate_diet_by_species\|focal_mask\|diet_matrix" osmose/engine/processes/predation.py osmose/engine/output.py | head`
Confirm: `enable_diet_tracking` production width defaults to `n_species + n_background` (`simulate.py:1436`), and `aggregate_diet_by_species` uses `focal_mask = species_id < n_pred_species` (excludes background slots 8/9).

- [ ] **Step 2: Write the diet-width survival test**

```python
def test_diagnostic_diet_width_keeps_background_and_resource_columns():
    # Enable diet tracking at width n_species + n_background + n_resources (=16 for Baltic),
    # run a short sim with a background predator + cod (benthos col 13) active,
    # assert background-predator rows and resource columns (>=10) are non-zero (not silently dropped).
    width = n_species + n_background + n_resources  # 8 + 2 + 6 = 16
    dm = _run_baltic_short_with_diet(width=width, fr={"sp0": (3, 1.0), "sp14": (3, 1.0)})
    assert dm.shape[1] == 16
    assert dm[8:10, :].sum() > 0     # background predators ate something
    assert dm[:, 10:16].sum() > 0    # resource (benthos etc.) columns survived
```

- [ ] **Step 3: Write the per-species aggregation-including-background test**

```python
def test_diagnostic_aggregation_includes_background_slots():
    # The diagnostic aggregation must include species_id 8/9 (background), unlike
    # aggregate_diet_by_species' focal_mask. Either call a diagnostic-specific aggregator
    # or assert the production one excludes them (documenting why the diagnostic can't reuse it).
    ...
```

If no diagnostic-specific aggregator exists, this task **creates** a small helper (e.g. `aggregate_diet_all_predators(diet_matrix, species_id)` with no focal mask) in `osmose/engine/output.py` or the eval script, and tests it directly. Keep it minimal.

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k diagnostic -v`
Expected: PASS

- [ ] **Step 5: Run the full FR suite + lint**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -v`
Run: `.venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/ tests/`
Expected: all PASS, lint+format clean. (CI lint = `ruff check` AND `ruff format --check` on `osmose/ ui/ tests/` — match it exactly.)

- [ ] **Step 6: Commit**

```bash
git add tests/test_engine_functional_response.py osmose/engine/output.py
git commit -m "test(fr): process-diagnostic unit tests (diet width 16, background aggregation)"
```

---

## Task A9: Documentation + PR-A finalization

**Files:**
- Modify: config reference doc (find: `grep -rln "stock.recruitment.shape\|recruitment.type" docs/`)

- [ ] **Step 1: Document both keys**

Add `predation.functional.response.shape.sp{i}` and `…halfsat.sp{i}` to the config reference alongside the recruitment `shape` key. Include: default `type1` ≡ existing behavior (bit-exact); `K` range `[0.1, 5.0]` required iff shape ≠ type1; **type-II is classically destabilizing (paradox of enrichment); type-III is the recommended/validated form**; combined-pool caveat (cod's refuge diluted by benthos).

- [ ] **Step 2: Run the entire test suite once**

Run: `.venv/bin/python -m pytest -q`
Expected: full suite green (note any pre-existing failures unrelated to FR; the 9 latent failures from PR #49 follow-ups A/B/C may still be open — confirm none are newly caused by FR).

- [ ] **Step 3: Commit + open PR-A**

```bash
git add docs/
git commit -m "docs(fr): document functional-response config keys"
```

Use superpowers:requesting-code-review before opening, then superpowers:finishing-a-development-branch to open PR-A. Gate for PR-A: 12/12 parity off + all opt-in tests green. PR-A is self-contained and shippable regardless of the Baltic calibration outcome.

---

# PART B — Baltic Calibration + Science (PR-B)

> PR-B depends on PR-A merged (or rebased onto it). Calibration runs are multi-hour and empirical; these tasks set up the run and the disposition, not bite-sized TDD.

## Task B1: Commit phase-13 result as `phase13_results.json`

**Files:**
- Create: `phase13_results.json` (in the calibration results dir calibrate_baltic.py reads)

- [ ] **Step 1: Locate the PR #50 phase-13 result and the expected path**

Run: `grep -rn "phase13_results\|phase12_results\|results.json\|get_phase1" scripts/calibrate_baltic.py | head` to find where prior-phase JSONs are loaded. Find the PR #50 phase-13 calibration output (the Shepherd result: 39 params = 16 mortality + 8 fishing + 7 ssb_half + 8 Shepherd β).

- [ ] **Step 2: Place the file**

Copy the PR #50 phase-13 result JSON to the path `calibrate_baltic.py` expects for `phase14` inheritance. Verify its `parameters` block has all 39 keys.

- [ ] **Step 3: Commit**

```bash
git add phase13_results.json
git commit -m "chore(fr): commit phase-13 Shepherd result as phase-14 calibration base"
```

---

## Task B2: phase-14 scaffolding in `calibrate_baltic.py`

**Files:**
- Modify: `scripts/calibrate_baltic.py`

- [ ] **Step 1: Inspect the phase ladder + phase-2 inheritance pattern**

Run: `grep -n "def get_phase\|phase ==\|base_config\|get_phase2_params\|get_phase13_params" scripts/calibrate_baltic.py | head -40`
Study how phase-2 freezes phase-1 params into `base_config` while returning only its own free params (disjoint sets).

- [ ] **Step 2: Add `get_phase14_params()`**

Returns **exactly the 4 new K keys** (free), type-III fixed, DE bounds `[0.5, 5.0]`:

```python
def get_phase14_params():
    # FR on cod(sp0), pikeperch(sp5), GreySeal(sp14->slot8), Cormorant(sp15->slot9); type-III fixed.
    return {
        "predation.functional.response.halfsat.sp0": (0.5, 5.0),
        "predation.functional.response.halfsat.sp5": (0.5, 5.0),
        "predation.functional.response.halfsat.sp14": (0.5, 5.0),
        "predation.functional.response.halfsat.sp15": (0.5, 5.0),
    }
```

(Match the exact return shape `get_phase13_params` uses — dict of key→(low,high), or whatever the local convention is.)

- [ ] **Step 3: Add the `phase == "14"` branch**

Load all 39 phase-13 params from `phase13_results.json` as fixed `base_config` overrides; additionally inject `predation.functional.response.shape.sp{0,5,14,15} = type3` into `base_config` (the shape is fixed, only K is free). Assert the free set (4 K keys) is disjoint from `base_config`. Reuse the bounded-runtime guards: `--patience 20 --wall-clock-cap-h 12 --checkpoint-every 5`, multi-seed re-ranking, `eff_popsize = max(15, 10*4) = 40`.

- [ ] **Step 4: Smoke-test the scaffolding (no full run)**

Run a 1-generation, 1-seed dry invocation to confirm the branch assembles a valid 4-D problem and the engine accepts the injected shape keys:

Run: `.venv/bin/python scripts/calibrate_baltic.py --phase 14 --generations 1 --popsize 8 --seeds 1 --dry-run` (adjust flags to the script's actual CLI; if no `--dry-run`, use the smallest gens/popsize).
Expected: assembles, runs a few evals, writes a checkpoint, no validation error on the FR keys.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_baltic.py
git commit -m "feat(fr): phase-14 calibration scaffolding (4 K params, type-III fixed, phase-13 frozen)"
```

---

## Task B3: Run phase-14 calibration

- [ ] **Step 1: Launch the bounded run**

Run (background, with guards): `.venv/bin/python scripts/calibrate_baltic.py --phase 14 --patience 20 --wall-clock-cap-h 12 --checkpoint-every 5 --seeds <multi>` (match the launch-wrapper convention; `OSMOSE_DE_WORKERS=16`).
Use `run_in_background: true`. Monitor via checkpoint files. Schedule a long fallback wakeup (~1800s) and rely on harness notification.

- [ ] **Step 2: Record outcome**

Capture: best objective vs phase-13 (6.008 was B-H, 2.133 was Shepherd phase-13), in-range count delta, multi-seed std (the noise band, analogous to PR #50's ±0.012), and whether it converged via patience or hit the wall-clock cap (report capped-best as capped if so).

---

## Task B4: Process diagnostic (FR-on vs FR-off realized mortality)

**Files:**
- Create/Modify: `scripts/fr_process_diagnostic.py` or a function in the eval script

- [ ] **Step 1: Implement the diagnostic**

Run the **same calibrated config twice** with diet tracking on at **width 16** — FR-off (type-I) vs FR-on (type-III with calibrated K). For each FR predator (cod, pikeperch, GreySeal, Cormorant) compute realized predation mortality on each prey = (Σ eaten of prey q by predator p over last 10 yr) / (mean biomass of q over the window), per year, using the aggregation that **includes background slots 8/9** (from A8). Report the FR-on − FR-off delta per predator-prey pair.

- [ ] **Step 2: Run + record**

Run the diagnostic on the B3 calibrated config. Record which predator (if any) shows a mortality reduction **exceeding the multi-seed noise band**, and whether the affected prey moved toward its ICES range.

---

## Task B5: Eval-script `--mode shepherd-fr`

**Files:**
- Modify: `scripts/evaluate_calibration_vs_ices.py`

- [ ] **Step 1: Inspect the existing `--mode` dispatch**

Run: `grep -n "mode\|choices\|shepherd\|bh\|make_objective\|inject" scripts/evaluate_calibration_vs_ices.py | head -40`
Find how the `shepherd` branch injects `stock.recruitment.type` so the new branch mirrors it.

- [ ] **Step 2: Add `shepherd-fr` to `--mode` choices + the branch**

Add `"shepherd-fr"` to the `choices` (currently `{"bh", "shepherd"}`). In the new branch: do everything the `shepherd` branch does **plus** inject `predation.functional.response.shape.sp{0,5,14,15}=type3` and the calibrated `halfsat` from the phase-14 JSON `parameters`. Import the objective wrapper (`make_objective` — pin the exact import path: `grep -rn "def make_objective" osmose/ scripts/`) and report objective (FR-on vs phase-13), ICES in-range delta, and the B4 diagnostic.

- [ ] **Step 3: Run the eval**

Run: `.venv/bin/python scripts/evaluate_calibration_vs_ices.py --mode shepherd-fr <args>`
Expected: prints objective vs phase-13, in-range delta, and the diagnostic table.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate_calibration_vs_ices.py scripts/fr_process_diagnostic.py
git commit -m "feat(fr): shepherd-fr eval mode + FR-on/FR-off process diagnostic"
```

---

## Task B6: Disposition + PR-B

- [ ] **Step 1: Apply the go/no-go disposition (spec §5)**

- **Binding gate:** 12/12 Java parity bit-exact (FR off) + all §4 opt-in tests green. (Already satisfied by PR-A; re-confirm on the PR-B branch.)
- **Reported, not gated:** objective vs phase-13.
- Ships as a **calibrated Baltic improvement** iff: objective does not regress **AND** the process diagnostic shows a mortality reduction **exceeding the multi-seed noise band** for ≥1 predator (a bare "some negative delta on lowest-biomass prey" is NOT sufficient — type-III guarantees that), ideally corroborated by that prey moving toward its ICES range.
- Otherwise ships as **engine capability only**, explicitly not a Baltic improvement. State the verdict honestly in the PR description (mirroring PR #50's honest caveat).

- [ ] **Step 2: Open PR-B**

Use superpowers:requesting-code-review then superpowers:finishing-a-development-branch. PR-B body states the empirical verdict.

---

## Self-Review (completed by plan author)

**Spec coverage:** §1 engine math → A5/A6 (branch + clamp); §2 config schema → A1/A2/A3/A4 (keys, enum-map, validation, sizing, registration, allowlist); §3 kernel changes → A5/A6 (both kernels, njit threading corrected per round-5); §4 testing → A5–A8 (curve/conservation/config/parity/numba-vs-python/NaN/determinism/background/diagnostic); §5 calibration → B1–B6 (phase13 commit, phase14, runtime, predator selection, diagnostic, eval mode, disposition); §6 cross-feature → documented as caveats in A9 docs + B6 verdict; §7 delivery (two PRs) → Part A / Part B split. ✅ All sections mapped.

**Placeholder scan:** Test bodies marked `...` in A5 Step 6, A8 Step 3, and B4 are scenario-construction stubs that explicitly instruct the executor to build from the nearest existing harness (named via `grep`) — acceptable because the exact harness import differs by what A-tasks expose; the math/contract they must satisfy is fully specified. No "TODO/handle edge cases" placeholders remain.

**Type consistency:** `fr_shape` (int32) / `fr_halfsat` (float64), codes `type1→1, type2→2, type3→3`, sentinel `_FR_HALFSAT_SENTINEL=1.0`, and the branch structure (`g_form`, `cap`, `g`) are identical across A2/A5/A6 and the `_g_ref` test oracle. ✅
