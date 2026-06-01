# Predator Functional Response (aggregate, opt-in) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selectable, opt-in, per-predator-species Holling functional-response forms (type-I / type-II / type-III) to OSMOSE's live predation kernel, default-off and bit-exact-preserving when off, plus a Baltic calibration phase (phase-14) validated at the process level.

**Architecture:** Two new per-species config keys (`predation.functional.response.shape.sp{i}` + `…halfsat.sp{i}`) parse into two `EngineConfig` arrays (`fr_shape: int32[n_total]`, `fr_halfsat: float64[n_total]`) built exactly like `recruitment_shepherd_beta` — which flows through **four** layers (`_parse_reproduction_params` → `_repro` unpack → `_focal` rebuild → `_merge_focal_background` concat). The arrays are threaded into the single live predation injection point in **both** kernels — the Python fallback `_apply_predation_for_school` (`config` in scope) and the numba `_apply_predation_numba` (threaded as positional args through three `@njit` callers, mirroring `ingestion_rate`). At the injection point, `eaten_total = min(total_available, max_eatable)` is replaced by a branch: `fr_shape==1` keeps the verbatim existing statement (bit-exact); `==2/3` apply the ration-capped Holling form `eaten_total = max_eatable · min(g_form(r), min(r,1))`. Delivered as two PRs (PR-A engine capability, PR-B Baltic calibration), mirroring PR #50.

**Tech Stack:** Python 3.12, NumPy, Numba (`@njit`), pytest, scipy `differential_evolution` (calibration), ruff. Run tests with `.venv/bin/python -m pytest`.

**Reference spec:** `docs/superpowers/specs/2026-05-31-predator-functional-response-design.md` (round-5 verified).

> **Revision note (round-6 review):** This plan was rewritten after a 4-angle in-loop review found BLOCKERs in the config-plumbing and schema sections. Verified facts now baked in:
> - `OsmoseField` kwargs are `param_type=ParamType.ENUM/FLOAT`, `min_val`/`max_val`, `indexed=True`, placeholder `{idx}` (NOT `field_type`/`min_value`/`{i}`). `__post_init__` raises on `{i}`.
> - Registry accessor: `build_registry().get_field("…sp{idx}")` (method, exact `{idx}`), not a module-level `get_field`.
> - `shepherd_beta` flows through 4 layers: focal-dict at `config.py:562`, `_repro` unpack at `:1539`, `_focal` rebuild at `:1601`, concat at `:776`/`:826`.
> - Background file is `osmose/engine/background.py` (NOT `processes/`); `BackgroundSpeciesInfo` dataclass at `:52`, ingestion field `:88`, parsed at `:197`/`:218`; bkg arrays built inline in `_merge_focal_background` (`config.py:735`).
> - New **required** `EngineConfig` fields break `tests/test_engine_config_validation.py` (`_minimal_config` at `:90`, six `EngineConfig(**cfg)` calls) — fixture must be updated.
> - `config.py` imports from `background.py` at module level → importing `_FR_SHAPE_CODE` the other way is a CIRCULAR import. Duplicate + parity-test instead.
> - Parity suite is `tests/test_engine_parity.py` (12 collected; 3 are `_exact_match_local_only`, skipped on CI → "9 passed, 3 skipped" unless run locally on the baseline interpreter).
> - Production diet width is hardwired to `n_species + n_background` at `simulate.py:1436` → width-16 diagnostic needs a monkeypatch (diagnostic-only).
> - Calibration: `get_phase13_shepherd_params()` returns `(keys, bounds, x0)` tuple (NOT a dict); freeze pattern to copy is **phase-2** (`calibrate_baltic.py` p1-load block), not phase-13; results path is `data/baltic/calibration_results/phase{N}_results.json`; `make_objective` is at `calibrate_baltic.py:260` (sibling import).

---

## File Structure

**PR-A (engine capability):**
- `osmose/schema/predation.py` — MODIFY: add two `OsmoseField`s.
- `osmose/engine/config.py` — MODIFY: parse + enum→int + validation (after `:549`); focal-dict (`:562`); `_repro` unpack (`:1539`); `_focal` rebuild (`:1601`); `_merge_focal_background` concat (`:776`/`:826`); `EngineConfig` field decl (`:1221`); `per_species_arrays` (`:1449`); `from_dict` `_merged` read (`:1641`) + constructor (`:1937`).
- `osmose/engine/background.py` — MODIFY: add `fr_shape`/`fr_halfsat` to `BackgroundSpeciesInfo` (`:88`); parse + validate in `parse_background_species` (`:197` area); duplicate `_FR_SHAPE_CODE`/`_FR_HALFSAT_SENTINEL` (no import — circular).
- `osmose/engine/processes/mortality.py` — MODIFY: branch at both injection points (`:484`, `:952`); thread `fr_shape`/`fr_halfsat` through `_apply_predation_numba` signature (`:786`) + 3 njit callers (`:1040`,`:1178`,`:1341`) + their call sites + the 2 Python callers (`:1626`,`:1918`) + the 1 Python call site (`:1659`).
- `tests/test_engine_functional_response.py` — CREATE.
- `tests/test_engine_config_validation.py` — MODIFY: add `fr_shape`/`fr_halfsat` to `_minimal_config` (`:90` area).
- `docs/` config reference — MODIFY.

**PR-B (Baltic calibration + science):**
- `data/baltic/calibration_results/phase13_results.json` — CREATE.
- `scripts/calibrate_baltic.py` — MODIFY: `get_phase14_params()` + `phase == "14"` branch (phase-2-style freeze).
- `scripts/evaluate_calibration_vs_ices.py` — MODIFY: add `shepherd-fr` to `--mode`.
- `scripts/fr_process_diagnostic.py` — CREATE: FR-on vs FR-off realized-mortality diagnostic (multi-seed, width-16).

---

# PART A — Engine Capability (PR-A)

## Task A0: Test helpers (first-class — many later tests depend on these)

**Files:**
- Create/extend: `tests/test_engine_functional_response.py` (helper section + conftest-style fixtures)

These helpers are load-bearing across A5–A8. Define and smoke-test them BEFORE the kernel tasks so later tasks don't invent divergent ad-hoc harnesses.

- [ ] **Step 1: Find the smallest existing engine run harness**

Run: `grep -rn "PythonEngine\|use_numba\|def run\|from_dict\|EngineConfig" tests/test_engine_*.py | grep -i "run\|engine(" | head -30`
Identify the smallest existing integration/smoke test that builds a config dict and runs the Python engine for a few steps, and how it toggles numba vs Python (e.g. a `use_numba` kwarg or env). Reuse its config fixture.

- [ ] **Step 2: Implement the helpers**

Add to the top of `tests/test_engine_functional_response.py`:

```python
import numpy as np

_FR_KEY_SHAPE = "predation.functional.response.shape.sp{i}"
_FR_KEY_HALFSAT = "predation.functional.response.halfsat.sp{i}"

def _apply_fr(cfg: dict, fr: dict | None) -> dict:
    """Inject FR config keys. fr maps species token ('sp0','sp14') -> (shape_int, k|None).
    Rule: shape_int==1 (type1) emits the shape key but NO halfsat key (type1 must not
    require halfsat); shape 2/3 emit both. Background tokens (sp14/sp15) are the CONFIG
    keys; the engine maps them to runtime slots 8/9 internally."""
    if not fr:
        return cfg
    code_to_name = {1: "type1", 2: "type2", 3: "type3"}
    for tok, (shape_int, k) in fr.items():
        i = tok[2:]  # 'sp0' -> '0'
        cfg[_FR_KEY_SHAPE.format(i=i)] = code_to_name[shape_int]
        if shape_int != 1:
            cfg[_FR_KEY_HALFSAT.format(i=i)] = k
    return cfg

def _base_cfg(background: bool):
    """Return the smallest valid config dict (2-3 focal species, optionally with the
    Baltic 8+2 background setup). Built from the existing fixture found in Step 1."""
    ...  # construct from the Step-1 fixture; keep deterministic seed

def _run_short_sim(numba=True, fr=None, seed=7, background=False, force_empty_prey=False):
    """Run a few steps; return per-species end-of-run biomass as np.ndarray.
    force_empty_prey: zero out all resource biomass AND remove non-predator schools so a
    type-3 predator hits a cell with total_available==0 (exercises the guard/NaN path)."""
    cfg = _apply_fr(_base_cfg(background), fr)
    ...  # build EngineConfig.from_dict(cfg), run with the numba toggle, return biomass

def _run_baltic_short(seed=11, fr=None):
    return _run_short_sim(numba=True, fr=fr, seed=seed, background=True)
```

The `...` bodies are filled from the Step-1 harness; the SIGNATURES and the `(type1, None)→no-halfsat` rule are fixed contracts the later tasks rely on.

- [ ] **Step 3: Smoke-test the helpers (no FR yet)**

```python
def test_helpers_run_baseline():
    a = _run_short_sim(numba=True, fr=None, seed=7)
    b = _run_short_sim(numba=True, fr=None, seed=7)
    np.testing.assert_array_equal(a, b)  # deterministic
    assert np.all(np.isfinite(a))
```

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py::test_helpers_run_baseline -v`
Expected: PASS (proves the harness runs before any FR code exists; `_apply_fr(cfg, None)` is a no-op).

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_functional_response.py
git commit -m "test(fr): shared run harness + FR config-injection helpers"
```

---

## Task A1: Config schema fields

**Files:**
- Modify: `osmose/schema/predation.py`
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Confirm the predation schema list + import**

Run: `grep -n "OsmoseField\|ParamType\|^from\|^import\|FIELDS\|= \[" osmose/schema/predation.py | head -30`
Confirm `ParamType` is imported (from `osmose.schema.base`) and find the list the predation fields are appended to (so the new fields register).

- [ ] **Step 2: Add the two fields (REAL kwargs — verified against base.py:51-63)**

Append to the predation field list in `osmose/schema/predation.py`:

```python
OsmoseField(
    key_pattern="predation.functional.response.shape.sp{idx}",
    param_type=ParamType.ENUM,
    default="type1",
    choices=["type1", "type2", "type3"],
    category="predation",
    indexed=True,
    required=False,
    description=(
        "Holling functional-response form for this predator. type1 (default) = "
        "existing linear-with-ration-ceiling behavior (bit-exact). type2 = saturating "
        "(Holling disc; classically destabilizing — paradox of enrichment). type3 = "
        "sigmoid with low-density prey refuge (recommended/validated form)."
    ),
),
OsmoseField(
    key_pattern="predation.functional.response.halfsat.sp{idx}",
    param_type=ParamType.FLOAT,
    default=None,
    min_val=0.1,
    max_val=5.0,
    category="predation",
    indexed=True,
    required=False,
    description=(
        "Dimensionless ration-relative half-saturation K for type2/type3. "
        "Required when shape != type1. Range [0.1, 5.0]. Well-scaled for DE, "
        "not a transferable biological constant."
    ),
),
```

- [ ] **Step 3: Schema-registration test (REAL accessor — verified against test_schema.py:100)**

```python
from osmose.schema import build_registry

def test_fr_schema_fields_registered():
    reg = build_registry()
    shape = reg.get_field("predation.functional.response.shape.sp{idx}")
    assert shape.param_type.value == "enum"
    assert shape.default == "type1"
    assert set(shape.choices) == {"type1", "type2", "type3"}
    halfsat = reg.get_field("predation.functional.response.halfsat.sp{idx}")
    assert halfsat.min_val == 0.1
    assert halfsat.max_val == 5.0
```

(Confirm `build_registry` is exported from `osmose.schema` with `grep -n "build_registry" osmose/schema/__init__.py`; adjust if the export name differs.)

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py::test_fr_schema_fields_registered -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/schema/predation.py tests/test_engine_functional_response.py
git commit -m "feat(fr): add predator functional-response schema fields"
```

---

## Task A2: Parse keys + enum→int + validation (focal path)

**Files:**
- Modify: `osmose/engine/config.py` (parse after `:549`; focal-dict `:562`)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Decide the entry point and write failing validation tests (all with `match=`)**

The public builder is `build_engine_config` (in config.py); `EngineConfig.from_dict` runs the allowlist. Use whichever the existing `tests/test_engine_config_validation.py` parse tests use — confirm with `grep -n "build_engine_config\|from_dict\|_minimal\|_parse" tests/test_engine_config_validation.py | head`. Call that same entry point here so the red is a real validation red, not an allowlist/KeyError red.

```python
import pytest

def test_fr_halfsat_required_when_shape_not_type1():
    cfg = _apply_fr(_base_cfg(background=False), {"sp0": (3, None)})  # type3, no halfsat
    with pytest.raises(ValueError, match="is required when"):
        _build_via_entry_point(cfg)  # = build_engine_config or from_dict per Step 1

def test_fr_halfsat_out_of_range_raises():
    cfg = _base_cfg(background=False)
    cfg["predation.functional.response.shape.sp0"] = "type3"
    cfg["predation.functional.response.halfsat.sp0"] = 0.0
    with pytest.raises(ValueError, match="out of range"):
        _build_via_entry_point(cfg)

def test_fr_shape_invalid_enum_raises():
    cfg = _base_cfg(background=False)
    cfg["predation.functional.response.shape.sp0"] = "type9"
    with pytest.raises(ValueError, match="(?i)type9|not.*one of|invalid"):
        _build_via_entry_point(cfg)
```

- [ ] **Step 2: Run — confirm they fail FOR THE RIGHT REASON**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "fr_halfsat or fr_shape_invalid" -v`
Expected: FAIL. **Read the failure text**: at this point the keys are unknown, so the most likely failure is "no ValueError raised" (keys silently ignored) — that is the correct red. If instead you see an allowlist-rejection ValueError whose message is NOT the validation message, note it; the `match=` substrings guard against a false green when A4 later allowlists the keys.

- [ ] **Step 3: Add parse + enum-map + validation (after `:549`)**

Add the sentinel constant near the module top:

```python
_FR_HALFSAT_SENTINEL = 1.0  # inert: type1 never reads fr_halfsat
_FR_SHAPE_CODE = {"type1": 1, "type2": 2, "type3": 3}
```

After the recruitment block (`:549`), using `_species_str_optional` (confirmed at config.py:77-95, accepts `allowed=` and lowercases input):

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

- [ ] **Step 4: Add focal-dict entries (at `:562`, beside `focal_recruitment_shepherd_beta`)**

```python
        "focal_fr_shape": fr_shape_focal,
        "focal_fr_halfsat": fr_halfsat_focal,
```

- [ ] **Step 5: Run validation tests (full green awaits A4 wiring)**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "fr_halfsat or fr_shape_invalid" -v`
Expected: the three validation tests PASS (validation runs at parse, before the array wiring). If the chosen entry point errors later for missing arrays, the `match=` confirms the raise is the validation one; defer any non-validation failure to A4 and note it.

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_functional_response.py
git commit -m "feat(fr): parse functional-response keys with strict validation (focal path)"
```

---

## Task A3: Background-predator parse (`osmose/engine/background.py`)

**Files:**
- Modify: `osmose/engine/background.py` (`BackgroundSpeciesInfo` `:88`; `parse_background_species` `:197` area)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Write a failing background enum→slot test**

```python
def test_fr_background_enum_maps_to_runtime_slot():
    cfg = _apply_fr(_base_cfg(background=True), {"sp14": (3, 1.0)})  # GreySeal config key
    ecfg = _build_via_entry_point(cfg)
    assert ecfg.fr_shape[8] == 3       # runtime slot 8 = n_focal(8) + bkg_idx(0)
    assert ecfg.fr_halfsat[8] == 1.0
    assert ecfg.fr_shape[9] == 1       # untouched background default
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py::test_fr_background_enum_maps_to_runtime_slot -v`
Expected: FAIL (no `fr_shape` on EngineConfig yet)

- [ ] **Step 3: Duplicate the code map (NO import — circular)**

`config.py` imports from `background.py` at module level (`config.py:19`), so importing `_FR_SHAPE_CODE` back into `background.py` is a circular import. Duplicate the two constants near the top of `background.py`:

```python
_FR_HALFSAT_SENTINEL = 1.0  # inert: type1 never reads fr_halfsat
_FR_SHAPE_CODE = {"type1": 1, "type2": 2, "type3": 3}
```

- [ ] **Step 4: Add fields to `BackgroundSpeciesInfo` (`:88`, beside `ingestion_rate`)**

```python
    fr_shape: int = 1            # functional-response code: 1=type-I, 2=type-II, 3=type-III
    fr_halfsat: float = 1.0      # ration-relative half-saturation K (type2/3 only)
```

(If the dataclass has no defaults, place these after the last defaulted field or supply them at every construction site.)

- [ ] **Step 5: Parse + validate in `parse_background_species` (beside ingestion at `:197`/`:218`)**

```python
        fr_shape_str = cfg.get(f"predation.functional.response.shape.sp{i}", "type1").strip().lower()
        if fr_shape_str not in _FR_SHAPE_CODE:
            raise ValueError(
                f"predation.functional.response.shape.sp{i} = {fr_shape_str!r} not one of "
                f"{sorted(_FR_SHAPE_CODE)}"
            )
        fr_shape = _FR_SHAPE_CODE[fr_shape_str]
        if fr_shape_str != "type1":
            hv = cfg.get(f"predation.functional.response.halfsat.sp{i}")
            if hv is None:
                raise ValueError(
                    f"predation.functional.response.halfsat.sp{i} is required when "
                    f"predation.functional.response.shape.sp{i} = {fr_shape_str}"
                )
            if not (0.1 <= float(hv) <= 5.0):
                raise ValueError(
                    f"predation.functional.response.halfsat.sp{i} = {hv} out of range [0.1, 5.0]"
                )
            fr_halfsat = float(hv)
        else:
            fr_halfsat = _FR_HALFSAT_SENTINEL
```

Pass `fr_shape=fr_shape, fr_halfsat=fr_halfsat` to the `BackgroundSpeciesInfo(...)` constructor at `:218`. **NOTE the `i` here is the CONFIG index (sp14/sp15)**, not the runtime slot — confirm the loop variable maps to the sp14/sp15 config keys (the background config keys), per `background.py` parse convention.

- [ ] **Step 6: Add a constants-parity regression test (catches drift without coupling imports)**

```python
def test_fr_shape_code_parity_across_modules():
    from osmose.engine.config import _FR_SHAPE_CODE as A
    from osmose.engine.background import _FR_SHAPE_CODE as B
    assert A == B
```

- [ ] **Step 7: Run (full green awaits A4 concat)**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "fr_background or shape_code_parity" -v`
Expected: parity test PASS; slot test goes green in A4 (note it).

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/background.py tests/test_engine_functional_response.py
git commit -m "feat(fr): parse functional-response keys on background-predator path"
```

---

## Task A4: EngineConfig field, the 4-layer concat, registration, from_dict, fixture fix

**Files:**
- Modify: `osmose/engine/config.py` (`:1221`, `:776`/`:826`, `:1449`, `:1539`, `:1601`, `:1641`, `:1937`, `:735` area)
- Modify: `tests/test_engine_config_validation.py` (`:90`)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Write failing sizing + registration tests**

```python
import dataclasses

def test_fr_arrays_sized_n_total_and_registered():
    ecfg = _build_via_entry_point(_base_cfg(background=True))  # 8 focal + 2 bkg
    assert ecfg.fr_shape.dtype == np.int32
    assert ecfg.fr_halfsat.dtype == np.float64
    assert len(ecfg.fr_shape) == ecfg.n_species + ecfg.n_background == 10
    assert len(ecfg.fr_halfsat) == 10

def test_fr_mis_sized_array_raises():
    ecfg = _build_via_entry_point(_base_cfg(background=True))
    cfg_kwargs = dataclasses.asdict(ecfg)
    cfg_kwargs["fr_shape"] = np.ones(ecfg.n_species, dtype=np.int32)  # wrong length
    with pytest.raises(ValueError, match="fr_shape"):
        type(ecfg)(**cfg_kwargs)  # direct construction re-runs __post_init__ length check
```

(If `dataclasses.asdict` chokes on non-trivial fields, build the kwargs by copying `ecfg.__dict__` instead.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "fr_arrays_sized or fr_mis_sized" -v`
Expected: FAIL (`EngineConfig` has no `fr_shape`)

- [ ] **Step 3: Declare the two EngineConfig fields (after `shepherd_beta` at `:1221`)**

```python
    # Predator functional response (post-parity; opt-in, default code 1 ≡ existing)
    fr_shape: NDArray[np.int32]  # per-species Holling form code: 1=type-I, 2=type-II, 3=type-III
    fr_halfsat: NDArray[np.float64]  # per-species ration-relative half-saturation K (type2/3 only)
```

These are required (no-default) fields placed in the required block (before the defaulted block at `:1326`). Step 7 fixes the only direct-construction fixture that breaks.

- [ ] **Step 4: `_repro` unpack (`:1539`) + `_focal` rebuild (`:1601`)**

At `:1539` (after `focal_recruitment_shepherd_beta = _repro["focal_recruitment_shepherd_beta"]`):

```python
        focal_fr_shape = _repro["focal_fr_shape"]
        focal_fr_halfsat = _repro["focal_fr_halfsat"]
```

In the `_focal` dict at `:1601` (after `"focal_recruitment_shepherd_beta": focal_recruitment_shepherd_beta,`):

```python
            "focal_fr_shape": focal_fr_shape,
            "focal_fr_halfsat": focal_fr_halfsat,
```

- [ ] **Step 5: Build bkg arrays inline + concat (with-bkg `:776`, no-bkg `:826`)**

In `_merge_focal_background`, beside `bkg_ingestion = np.array([b.ingestion_rate for b in background_list])` (`:735`):

```python
        bkg_fr_shape = np.array([b.fr_shape for b in background_list], dtype=np.int32)
        bkg_fr_halfsat = np.array([b.fr_halfsat for b in background_list], dtype=np.float64)
```

With-background return dict (at `:776`):

```python
            "fr_shape": np.concatenate([focal["focal_fr_shape"], bkg_fr_shape]),
            "fr_halfsat": np.concatenate([focal["focal_fr_halfsat"], bkg_fr_halfsat]),
```

No-background return dict (at `:826`):

```python
            "fr_shape": focal["focal_fr_shape"],
            "fr_halfsat": focal["focal_fr_halfsat"],
```

- [ ] **Step 6: Register in `per_species_arrays` (`:1449`)**

```python
            "shepherd_beta": self.shepherd_beta,
            "fr_shape": self.fr_shape,
            "fr_halfsat": self.fr_halfsat,
```

- [ ] **Step 7: from_dict `_merged` read (`:1641`) + constructor (`:1937`) + FIXTURE FIX**

At `:1641`:

```python
        fr_shape = _merged["fr_shape"]
        fr_halfsat = _merged["fr_halfsat"]
```

At the `EngineConfig(...)` constructor `:1937` (after `shepherd_beta=recruitment_shepherd_beta,`):

```python
            fr_shape=fr_shape,
            fr_halfsat=fr_halfsat,
```

**Audit + fix all direct `EngineConfig(**...)` sites.** Run `grep -rn "EngineConfig(" osmose/ tests/ | grep -v "from_dict\|build_engine"`. The only direct-construction site is `tests/test_engine_config_validation.py` via `_minimal_config()` (`:9-94`, six calls). Add to its `defaults` dict near `:90`:

```python
        fr_shape=np.ones(n_total, dtype=np.int32),
        fr_halfsat=np.full(n_total, 1.0),
```

- [ ] **Step 8: Run sizing + registration + A2/A3 deferred tests (now wired)**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -v`
Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -v`
Expected: all PASS (config-validation fixture no longer `TypeError`s; FR sizing/background/validation green).

- [ ] **Step 9: Allowlist warning check**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -v`
Expected: PASS warning-free (AST walker auto-captures the literal-prefix keys read via `_species_*_optional`; if a warning appears, add both key prefixes to `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py` and re-run).

- [ ] **Step 10: Commit**

```bash
git add osmose/engine/config.py osmose/engine/background.py tests/test_engine_functional_response.py tests/test_engine_config_validation.py
git commit -m "feat(fr): EngineConfig fr_shape/fr_halfsat 4-layer wiring + fixture fix"
```

---

## Task A5: Kernel branch — Python fallback + the math oracle pinned to the kernel

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (signature `:335`; injection `:484`; call site `:1659`)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Define the math oracle `_g_ref` and its PROPERTY tests**

```python
def _g_ref(r, shape, k):
    if shape == 1:
        return min(r, 1.0)
    g = (r / (r + k)) if shape == 2 else ((r * r) / (r * r + k * k))
    return min(g, min(r, 1.0))  # conservation clamp

@pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 5.0])
@pytest.mark.parametrize("r", [0.01, 0.1, 0.5, 0.9, 1.0, 2.0, 10.0])
def test_oracle_conservation(r, k):
    for shape in (2, 3):
        assert _g_ref(r, shape, k) <= min(r, 1.0) + 1e-12

def test_oracle_clamp_is_load_bearing():
    # At K=0.1, r=0.5 the RAW type-III form violates conservation, so the min() must engage.
    r, k = 0.5, 0.1
    raw = (r * r) / (r * r + k * k)
    assert raw > min(r, 1.0)            # raw violates g<=min(r,1)
    assert _g_ref(r, 3, k) == min(r, 1.0)  # clamp pulled it back to the ration cap

def test_oracle_anchors_and_limits():
    assert _g_ref(1.0, 2, 1.0) == pytest.approx(0.5)
    assert _g_ref(1.0, 3, 1.0) == pytest.approx(0.5)
    assert _g_ref(0.3, 1, 999) == 0.3 and _g_ref(5.0, 1, 999) == 1.0
    assert _g_ref(1e6, 2, 1.0) == pytest.approx(1.0, abs=1e-3)   # g->1 as r->inf
    assert _g_ref(1e6, 3, 1.0) == pytest.approx(1.0, abs=1e-3)
    # monotonic in r
    xs = [0.05, 0.2, 0.5, 1.0, 2.0]
    for shape in (2, 3):
        gs = [_g_ref(x, shape, 1.0) for x in xs]
        assert all(b >= a - 1e-12 for a, b in zip(gs, gs[1:]))

def test_oracle_type3_refuge_ratio_increasing():
    # type-III refuge: g_form(r)/r increasing on a strict r<K grid (smaller fraction at low r)
    k = 1.0
    rs = [0.05, 0.1, 0.2, 0.4]  # all < k
    ratios = [((r * r) / (r * r + k * k)) / r for r in rs]
    assert all(b > a for a, b in zip(ratios, ratios[1:]))
    assert _g_ref(0.05, 3, k) < _g_ref(0.05, 2, k)  # type3 takes less than type2 at low r
```

- [ ] **Step 2: Run — these pass immediately (they lock the intended math)**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k oracle -v`
Expected: PASS. (These test the oracle; Step 6 pins the KERNEL to the oracle — that is the load-bearing edge.)

- [ ] **Step 3: Add `fr_shape`/`fr_halfsat` to `_apply_predation_for_school` signature (`:335`)**

Add two params after `ingestion_rate` (match the numba order in A6):

```python
    ingestion_rate,
    fr_shape,
    fr_halfsat,
```

- [ ] **Step 4: Replace the injection point (`:484`) — do NOT recompute sp_pred**

`sp_pred = state.species_id[p_idx]` already exists at `:371`; reuse it. Replace exactly:

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

`success = min(eaten_total / max_eatable, 1.0)` at `:541` stays **unchanged** (correct for all forms). Do NOT route type-I through `max_eatable * g` (1-ULP parity risk).

- [ ] **Step 5: Pass the arrays at the call site (`:1659`, `config` in scope)**

After `config.ingestion_rate` in the `_apply_predation_for_school(...)` call:

```python
                    config.fr_shape,
                    config.fr_halfsat,
```

- [ ] **Step 6: THE load-bearing test — kernel output equals `max_eatable · _g_ref(r,…)`**

Construct a single-predator, single-cell scenario with KNOWN `total_available` and `max_eatable` so `r` is a known scalar, run `_apply_predation_for_school`, and assert the resulting `eaten_total` (read it back via `state.preyed_biomass[p_idx]`, which `:543` sets to `eaten_total`) matches the oracle:

```python
@pytest.mark.parametrize("shape,k", [(1, 1.0), (2, 1.0), (3, 1.0), (3, 0.1), (3, 5.0)])
@pytest.mark.parametrize("r", [0.05, 0.5, 0.95, 2.0])
def test_python_kernel_matches_oracle(shape, k, r):
    # Build a minimal predation call where total_available = r * max_eatable.
    # Reuse the closest existing _apply_predation_for_school unit-test harness:
    # grep -rn "_apply_predation_for_school" tests/
    eaten = _run_single_predation_step_python(r=r, shape=shape, k=k)  # returns preyed_biomass
    max_eatable = _max_eatable_of_that_step()
    assert eaten == pytest.approx(max_eatable * _g_ref(r, shape, k), rel=1e-12)
```

Build `_run_single_predation_step_python` from the existing direct unit test of `_apply_predation_for_school` (find via grep); it must let you set `total_available`/`max_eatable` (e.g. via a single prey school's eligible biomass and the predator's ingestion-derived `max_eatable`). This is the test that makes `_g_ref` load-bearing.

- [ ] **Step 7: Run**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -v`
Expected: PASS (incl. `test_python_kernel_matches_oracle`)

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/processes/mortality.py tests/test_engine_functional_response.py
git commit -m "feat(fr): FR branch in Python kernel + kernel-vs-oracle value test"
```

---

## Task A6: Kernel branch — numba + thread through 3 njit callers + numba value/parity tests

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (signature `:786`; injection `:952`; njit callers `:1040`,`:1178`,`:1341`; their call sites ~`:1116`,`:1279`,`:1454`; Python callers `:1626`,`:1918`)
- Test: `tests/test_engine_functional_response.py`

- [ ] **Step 1: Write failing numba-vs-oracle + numba-vs-Python tests**

```python
@pytest.mark.parametrize("shape,k", [(2, 1.0), (3, 1.0), (3, 0.1)])
@pytest.mark.parametrize("r", [0.05, 0.5, 0.95, 2.0])
def test_numba_kernel_matches_oracle(shape, k, r):
    eaten = _run_single_predation_step_numba(r=r, shape=shape, k=k)
    max_eatable = _max_eatable_of_that_step()
    assert eaten == pytest.approx(max_eatable * _g_ref(r, shape, k), rel=1e-12)

@pytest.mark.parametrize("shape,k", [(2, 1.0), (3, 1.0), (3, 0.5)])
def test_numba_python_parity_fr_on(shape, k):
    out_numba = _run_short_sim(numba=True, fr={"sp0": (shape, k)}, seed=3)
    out_py = _run_short_sim(numba=False, fr={"sp0": (shape, k)}, seed=3)
    np.testing.assert_allclose(out_numba, out_py, rtol=1e-9, atol=0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "numba_kernel_matches or numba_python_parity" -v`
Expected: FAIL

- [ ] **Step 3: Add params to `_apply_predation_numba` signature (`:786`, after `ingestion_rate` at `:802`)**

```python
        ingestion_rate,
        fr_shape,
        fr_halfsat,
```

- [ ] **Step 4: Replace the numba injection point (`:952`) — `sp_pred` already at `:843`**

Replace exactly `eaten_total = min(total_available, max_eatable)` with:

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

`success` at `:993` stays unchanged.

- [ ] **Step 5: Thread through the 3 njit callers + their call sites**

For each of `_mortality_in_cell_numba` (`:1040`), `_mortality_all_cells_numba` (`:1178`), `_mortality_all_cells_parallel` (`:1341`): each already takes `ingestion_rate` in its OWN signature (positional, not a closure). (1) Add `fr_shape, fr_halfsat` adjacent to `ingestion_rate` in the njit signature. (2) Pass `fr_shape, fr_halfsat` adjacent to `ingestion_rate` at the inner `_apply_predation_numba(...)` call (find via `grep -n "ingestion_rate" osmose/engine/processes/mortality.py`; the inner calls are near `:1116`/`:1279`/`:1454`).

- [ ] **Step 6: Thread from the 2 Python callers (`:1626`,`:1918`)**

Where each passes `config.ingestion_rate` into an njit caller, add `config.fr_shape, config.fr_halfsat` adjacent.

- [ ] **Step 7: Run numba value + parity + full file**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -v`
Expected: PASS (numba recompiles once on the additive signature change)

- [ ] **Step 8: NaN/guard + determinism tests**

```python
def test_fr_type3_empty_cell_no_nan():
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
git commit -m "feat(fr): FR branch in numba kernel + njit threading + numba value/parity tests"
```

---

## Task A7: Bit-exact parity gate (FR off) + behavior tests

**Files:**
- Test: `tests/test_engine_functional_response.py` + `tests/test_engine_parity.py`

- [ ] **Step 1: Run the named parity suite — confirm the skip semantics**

The parity suite is `tests/test_engine_parity.py` (12 collected). **3 are `@_exact_match_local_only`** (`test_biomass_match`, `test_abundance_match`, `test_mortality_match`) and are SKIPPED on CI / non-baseline interpreters — these are exactly the bit-exact checks an FR type-I-path regression would trip.

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -v`
Expected: **9 passed, 3 skipped** on a normal run (no baseline regen). If your environment IS the baseline interpreter, expect 12 passed.

- [ ] **Step 2: Force-run the 3 bit-exact tests locally**

The 3 `_exact_match_local_only` tests gate bit-exactness; run them on the baseline interpreter explicitly (the marker reads an env/flag — find it: `grep -n "_exact_match_local_only\|OSMOSE.*BASELINE\|skipif" tests/test_engine_parity.py`). Set that flag and run:

Run: `<BASELINE_FLAG>=1 .venv/bin/python -m pytest tests/test_engine_parity.py -k "biomass_match or abundance_match or mortality_match" -v`
Expected: 3 passed (FR defaults off ⇒ verbatim type-I ⇒ bit-exact). If they can't run in this environment, document that bit-exactness was verified by the explicit `test_fr_explicit_type1_equals_absent_key` (Step 3) instead, and flag for CI/baseline confirmation.

- [ ] **Step 3: Explicit "type1 == absent-key" engine test**

```python
def test_fr_explicit_type1_equals_absent_key():
    out_absent = _run_short_sim(numba=True, fr=None, seed=7)
    out_type1 = _run_short_sim(numba=True, fr={"sp0": (1, None)}, seed=7)  # explicit type1, no halfsat
    np.testing.assert_array_equal(out_absent, out_type1)
```

- [ ] **Step 4: Background-on behavior + accepted-but-inert + focal enum→int tests**

```python
def test_fr_on_background_predator_changes_outcome():
    base = _run_baltic_short(seed=11, fr=None)
    fr_on = _run_baltic_short(seed=11, fr={"sp14": (3, 1.0)})  # GreySeal, runtime slot 8
    assert not np.array_equal(base, fr_on)  # background FR must actually bite

def test_fr_focal_enum_maps_to_slot0():
    ecfg = _build_via_entry_point(_apply_fr(_base_cfg(background=False), {"sp0": (2, 1.0)}))
    assert ecfg.fr_shape[0] == 2

def test_fr_non_type1_on_prey_only_species_inert():
    # A non-type1 shape on a non-predator species parses and has no runtime effect.
    out_base = _run_short_sim(numba=True, fr=None, seed=5)
    out_inert = _run_short_sim(numba=True, fr={"sp1": (3, 1.0)}, seed=5)  # sp1 = a prey-only species
    np.testing.assert_array_equal(out_base, out_inert)  # adjust sp1 to a confirmed prey-only id
```

(Confirm which focal id is prey-only via the diet/accessibility matrix; pick one that never acts as a predator.)

- [ ] **Step 5: Run**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -k "type1_equals or background_predator or focal_enum or prey_only" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_engine_functional_response.py
git commit -m "test(fr): parity-off gate, type1==absent, background-on, inert-prey-only"
```

---

## Task A8: Process-diagnostic unit tests (the gate's falsifiable basis) + width-16 hook

**Files:**
- Test: `tests/test_engine_functional_response.py`
- Create: `aggregate_diet_all_predators` in `osmose/engine/output.py`

- [ ] **Step 1: Confirm diet-tracking width + aggregation mask**

Production `enable_diet_tracking` is called at `simulate.py:1436` with width `config.n_species + config.n_background` (=10 for Baltic) — resource columns (≥10) are dropped by the `prey_sp < diet_matrix.shape[1]` guard (mortality.py:517/537/973/989). `aggregate_diet_by_species` (output.py:269) uses `focal_mask = species_id < n_pred_species`, excluding background slots 8/9. The diagnostic needs width 16 (8+2+6) AND a background-inclusive aggregator.

- [ ] **Step 2: Create the background-inclusive aggregator**

In `osmose/engine/output.py`, add (do NOT reuse `aggregate_diet_by_species` — its focal_mask drops slots 8/9):

```python
def aggregate_diet_all_predators(diet_matrix, species_id, n_total):
    """Sum the per-school diet matrix into per-PREDATOR-SPECIES rows, INCLUDING background
    predators (runtime slots n_focal..n_total-1). Unlike aggregate_diet_by_species, applies
    no focal_mask. Returns (n_total, n_prey_cols)."""
    out = np.zeros((n_total, diet_matrix.shape[1]), dtype=diet_matrix.dtype)
    for s in range(diet_matrix.shape[0]):
        out[species_id[s]] += diet_matrix[s]
    return out
```

- [ ] **Step 3: Width-16 diagnostic test (with a width-override hook)**

Production can't emit width 16 (hardwired at `simulate.py:1436`). For the diagnostic, monkeypatch `enable_diet_tracking` to allocate width 16 (diagnostic-only; does not change production default):

```python
def test_diagnostic_diet_width_keeps_background_and_resource_columns(monkeypatch):
    import osmose.engine.processes.predation as pred
    real = pred.enable_diet_tracking
    monkeypatch.setattr(pred, "enable_diet_tracking",
                        lambda n_schools, n_species, ctx=None: real(n_schools, 16, ctx=ctx))
    dm = _run_baltic_short_with_diet(fr={"sp0": (3, 1.0), "sp14": (3, 1.0)})  # returns the raw diet_matrix
    assert dm.shape[1] == 16
    sid = _species_id_of_that_run()
    agg = aggregate_diet_all_predators(dm, sid, n_total=10)
    assert agg[8:10, :].sum() > 0   # background predators ate something (not dropped)
    assert agg[:, 10:16].sum() > 0  # resource columns (benthos etc.) survived width 16
```

(Confirm `enable_diet_tracking`'s exact signature with `grep -n "def enable_diet_tracking" osmose/engine/processes/predation.py`; adjust the lambda. Also confirm where `simulate` reads the width so `_run_baltic_short_with_diet` actually routes through the patched call.)

- [ ] **Step 4: Aggregation-includes-background test**

```python
def test_aggregate_all_predators_includes_background_slots():
    sid = np.array([0, 0, 8, 9])            # 2 focal sp0 schools + 1 GreySeal + 1 Cormorant
    dm = np.array([[1, 0], [2, 0], [0, 3], [0, 4]], dtype=float)
    agg = aggregate_diet_all_predators(dm, sid, n_total=10)
    assert agg[0].tolist() == [3, 0]        # focal summed
    assert agg[8].tolist() == [0, 3]        # background slot 8 PRESENT (not masked out)
    assert agg[9].tolist() == [0, 4]
```

- [ ] **Step 5: Run + lint**

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py -v`
Run: `.venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/ tests/`
Expected: all PASS; lint+format clean (CI lint = `ruff check` AND `ruff format --check` on `osmose/ ui/ tests/`).

- [ ] **Step 6: Commit**

```bash
git add tests/test_engine_functional_response.py osmose/engine/output.py
git commit -m "test(fr): diagnostic width-16 + background-inclusive diet aggregator"
```

---

## Task A9: Documentation (incl. all §6 caveats) + PR-A finalization

**Files:**
- Modify: config reference doc (find: `grep -rln "stock.recruitment.shape\|recruitment.type" docs/`)

- [ ] **Step 1: Document both keys + the full caveat set (spec §2 + §6)**

Add `predation.functional.response.shape.sp{i}` and `…halfsat.sp{i}` to the config reference alongside the recruitment `shape` key. Include ALL of:
- default `type1` ≡ existing behavior (bit-exact); `K` range `[0.1, 5.0]`, required iff shape ≠ type1.
- **type-II is classically destabilizing (paradox of enrichment); type-III is the recommended/validated form.**
- **Combined-pool caveat:** FR acts on the fused fish+resource pool; for cod, abundant benthos keeps `r` high so the refuge rarely triggers (effect may be small).
- **Bioenergetics double-cap (§6):** under bioen mode, growth/starvation read `preyed_biomass` re-capped by `bioen_ingestion_cap`, bypassing `pred_success_rate`; FR composes as a double-cap. FR+bioen is **unvalidated** — Baltic calibration uses no bioen.
- **FIE/Ev-OSMOSE (§6):** FR is enabled on cod (sp0), the FIE species in `baltic_ev`. FR+FIE on the same species is **unvalidated**; do not enable both in calibration.
- **Trophic-level output (§6):** FR shifts the realized diet mix → minor perturbation of the TL diagnostic.
- **Interleaved 4-cause mortality (§6):** FR reducing predation's bite leaves more abundance for starvation/additional/fishing in the per-school-shuffled sub-step; realized fishing mortality on FR-predators' prey is indirectly coupled.

- [ ] **Step 2: Spec §4 downstream consequence — directional prey-survival test (the non-bioen consequence)**

The spec §4 requires a bioenergetic-consequence test OR an explicit downgrade. Baltic uses no bioen, so deliver the observable non-bioen consequence and DOCUMENT the downgrade. Add to the test file:

```python
def test_fr_type3_increases_prey_survival(monkeypatch=None):
    # type-III refuge on a predator -> less predation on its prey -> higher prey end biomass.
    base = _run_baltic_short(seed=21, fr=None)
    fr_on = _run_baltic_short(seed=21, fr={"sp14": (3, 1.0)})  # GreySeal type-III
    prey_id = _a_primary_prey_of_greyseal()  # from the diet matrix
    assert fr_on[prey_id] >= base[prey_id]   # refuge raises (or holds) prey survival
```

Add a one-line note in the docs / plan: *"Spec §4 bioenergetic-consequence requirement is satisfied via the observable prey-survival delta; the `pred_success_rate→growth/starvation` path is NOT separately tested because the Baltic config uses no bioenergetics mode (§6)."*

Run: `.venv/bin/python -m pytest tests/test_engine_functional_response.py::test_fr_type3_increases_prey_survival -v`
Expected: PASS

- [ ] **Step 3: Full suite + commit + open PR-A**

Run: `.venv/bin/python -m pytest -q`
Expected: full suite green (note any pre-existing PR #49 follow-up failures A/B/C; confirm none are newly FR-caused).

```bash
git add docs/ tests/test_engine_functional_response.py
git commit -m "docs(fr): config keys + all cross-feature caveats + prey-survival consequence test"
```

Use superpowers:requesting-code-review, then superpowers:finishing-a-development-branch to open PR-A. **PR-A gate:** `tests/test_engine_parity.py` (9 pass/3 skip, + 3 bit-exact local) + all FR opt-in tests green. Self-contained; shippable regardless of the Baltic outcome.

---

# PART B — Baltic Calibration + Science (PR-B)

> Depends on PR-A merged/rebased. Calibration runs are multi-hour and empirical.

## Task B1: Commit phase-13 result as `phase13_results.json`

**Files:**
- Create: `data/baltic/calibration_results/phase13_results.json`

- [ ] **Step 1: Confirm the path convention**

Run: `grep -n "RESULTS_DIR\|phase.*_results.json\|json.load\|json.dump" scripts/calibrate_baltic.py | head`
Confirm `RESULTS_DIR = data/baltic/calibration_results` and that phase-N inheritance reads `RESULTS_DIR / "phaseN_results.json"`.

- [ ] **Step 2: Place the PR #50 phase-13 (Shepherd) result there**

Copy the PR #50 phase-13 calibration output JSON to `data/baltic/calibration_results/phase13_results.json`. Verify its `parameters` block has all 39 keys (16 mortality + 8 fishing + 7 ssb_half + 8 Shepherd β).

- [ ] **Step 3: Commit**

```bash
git add data/baltic/calibration_results/phase13_results.json
git commit -m "chore(fr): commit phase-13 Shepherd result as phase-14 calibration base"
```

---

## Task B2: phase-14 scaffolding (phase-2-style freeze)

**Files:**
- Modify: `scripts/calibrate_baltic.py`

- [ ] **Step 1: Study the phase-2 freeze + the phase-13 accessor shape**

Run: `grep -n "def get_phase\|phase ==\|p1_file\|base_config\|get_phase13_shepherd_params\|param_keys, bounds, x0" scripts/calibrate_baltic.py | head -40`
Note: `get_phase13_shepherd_params()` returns a `(keys, bounds, x0)` **tuple** (NOT a dict). Phase-13 does NOT inherit; the freeze template is **phase-2** (its `p1_file` JSON-load → `base_config` override block).

- [ ] **Step 2: Add `get_phase14_params()` returning the (keys, bounds, x0) tuple**

```python
def get_phase14_params():
    # FR on cod(sp0), pikeperch(sp5), GreySeal(sp14->slot8), Cormorant(sp15->slot9); type-III fixed.
    keys = [
        "predation.functional.response.halfsat.sp0",
        "predation.functional.response.halfsat.sp5",
        "predation.functional.response.halfsat.sp14",
        "predation.functional.response.halfsat.sp15",
    ]
    bounds = [(0.5, 5.0)] * 4   # clamp-free type-III DE bound
    x0 = [1.0] * 4
    return keys, bounds, x0
```

(Match the EXACT tuple order/shape `get_phase13_shepherd_params` returns.)

- [ ] **Step 3: Add the `phase == "14"` branch (copy the phase-2 freeze block)**

Load all 39 params from `data/baltic/calibration_results/phase13_results.json` as fixed `base_config` overrides (mirror the phase-2 `p1_file` load + the "missing → warn" guard). Additionally inject `predation.functional.response.shape.sp{0,5,14,15} = "type3"` into `base_config` (shape fixed, only K free). Assert the 4 free K keys are disjoint from `base_config`. Set `eff_popsize = max(15, 10*4) = 40`; reuse `--patience 20 --wall-clock-cap-h 12 --checkpoint-every 5` + multi-seed re-ranking.

- [ ] **Step 4: Smoke-test (tiny run)**

Run: `.venv/bin/python scripts/calibrate_baltic.py --phase 14 --generations 1 --popsize 8 --seeds 1` (adjust to the actual CLI; smallest gens/popsize).
Expected: assembles a 4-D problem, runs a few evals, writes a checkpoint, no FR-key validation error.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_baltic.py
git commit -m "feat(fr): phase-14 calibration scaffolding (4 K params, type-III, phase-2-style freeze)"
```

---

## Task B3: Run phase-14 calibration (multi-seed)

- [ ] **Step 1: Launch the bounded run**

Run (background, guards on, `OSMOSE_DE_WORKERS=16`): `.venv/bin/python scripts/calibrate_baltic.py --phase 14 --patience 20 --wall-clock-cap-h 12 --checkpoint-every 5 --seeds <multi>`. Use `run_in_background: true`; monitor via checkpoints; schedule a long fallback wakeup (~1800s).

- [ ] **Step 2: Record outcome**

Capture best objective vs phase-13 (Shepherd phase-13 was 2.133; B-H was 6.008), in-range count delta, multi-seed objective std, and convergence (patience vs capped — report capped-best as capped).

---

## Task B4: Process diagnostic (FR-on vs FR-off, multi-seed mortality-delta noise band)

**Files:**
- Create: `scripts/fr_process_diagnostic.py`

- [ ] **Step 1: Implement the diagnostic with its OWN noise band**

Run the calibrated config **across the same multi-seed set as B3**, twice per seed — FR-off (type-I) vs FR-on (type-III with calibrated K) — with diet tracking at width 16 (the A8 monkeypatch/hook) and `aggregate_diet_all_predators`. For each FR predator (cod, pikeperch, GreySeal, Cormorant) and each prey: realized mortality = (Σ eaten of prey q by predator p over last 10 yr) / (mean biomass of q over the window), per year. Report, **per predator-prey pair**, the FR-on − FR-off delta as **mean ± std across seeds** — this std is the mortality-delta noise band (NOT the objective std; they are different units).

- [ ] **Step 2: Run + record**

Record which predator-prey pairs show `|mean delta| > 2·std` (exceed the mortality-delta noise band), and whether the affected prey moved toward its ICES range.

---

## Task B5: Eval-script `--mode shepherd-fr`

**Files:**
- Modify: `scripts/evaluate_calibration_vs_ices.py`

- [ ] **Step 1: Inspect the `--mode` dispatch + `_apply_mode`**

Run: `grep -n "mode\|choices\|shepherd\|_apply_mode\|inject\|make_objective" scripts/evaluate_calibration_vs_ices.py | head -40`
The `shepherd` injection logic lives in `_apply_mode`. `make_objective` is at `scripts/calibrate_baltic.py:260` — import it as `from calibrate_baltic import make_objective` (sibling script; confirm the eval script's sys.path includes `scripts/`, else add it).

- [ ] **Step 2: Add `shepherd-fr` to choices + the branch**

Add `"shepherd-fr"` to `choices` (currently `{"bh", "shepherd"}`). The branch does everything `shepherd` does PLUS inject `predation.functional.response.shape.sp{0,5,14,15}=type3` + calibrated `halfsat` from the phase-14 JSON `parameters`. Report objective (FR-on vs phase-13 via `make_objective`), ICES in-range delta, and the B4 diagnostic.

- [ ] **Step 3: Run**

Run: `.venv/bin/python scripts/evaluate_calibration_vs_ices.py --mode shepherd-fr <args>`
Expected: prints objective vs phase-13, in-range delta, diagnostic table.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate_calibration_vs_ices.py scripts/fr_process_diagnostic.py
git commit -m "feat(fr): shepherd-fr eval mode + FR-on/FR-off multi-seed process diagnostic"
```

---

## Task B6: Disposition + PR-B

- [ ] **Step 1: Apply the go/no-go disposition (spec §5)**

- **Binding gate:** `tests/test_engine_parity.py` bit-exact (FR off) + all FR opt-in tests green (re-confirm on the PR-B branch).
- **Reported, not gated:** objective vs phase-13.
- Ships as a **calibrated Baltic improvement** iff: objective does not regress **AND** the B4 diagnostic shows a per-predator-prey mortality reduction whose `|mean delta|` **exceeds the mortality-delta noise band** (`> 2·std` across seeds, B4) for ≥1 predator — a bare negative delta is NOT sufficient (type-III guarantees that; non-falsifiable). Ideally corroborated by that prey moving toward its ICES range.
- Otherwise ships as **engine capability only**, explicitly not a Baltic improvement.
- State the verdict honestly in the PR description (mirroring PR #50's honest caveat).

- [ ] **Step 2: Open PR-B**

Use superpowers:requesting-code-review then superpowers:finishing-a-development-branch. PR-B body states the empirical verdict.

---

## Self-Review (round-6, completed by plan author)

**Spec coverage:** §1 → A5/A6 (branch + clamp + oracle-pinned value tests); §2 → A1/A2/A3/A4 (REAL schema kwargs, 4-layer concat, both-path enum, sizing+registration, fixture fix, allowlist); §3 → A5/A6 (both kernels, njit threading); §4 → A0/A5–A8 (oracle-vs-kernel value test, clamp-engagement, monotonic/limit/refuge-ratio, config, parity 9+3, numba-vs-python, NaN, determinism, background, inert-prey-only, focal enum, width-16 diagnostic, background-inclusive aggregator, prey-survival consequence); §5 → B1–B6 (phase13 commit at correct path, phase-2-style freeze, tuple return shape, mortality-delta noise band, eval mode, disposition); §6 → A9 docs (all 5 caveats) + §4 downgrade note + B6 verdict; §7 → Part A / Part B. ✅

**Round-6 BLOCKERs fixed:** (1) A1 `OsmoseField` kwargs corrected to `param_type`/`min_val`/`max_val`/`indexed`/`{idx}` + `build_registry().get_field`. (2) A4 Step 7 adds the missing `_minimal_config` fixture fields + audits all `EngineConfig(` sites. (3) A3/A4 background wiring rewritten to the real inline `np.array([b.x for b in background_list])` mechanism at the correct file `osmose/engine/background.py`; "default to np.ones" hedge deleted. (4) A4 Step 4 adds the missing `_repro`→`_focal` hand-off layer. (5) A5 Step 6 + A6 Step 1 add the direct kernel-vs-`_g_ref` value tests (the missing edge that let a wrong curve ship green). (6) A3 Step 3 duplicates `_FR_SHAPE_CODE` (+ parity test) instead of the circular import. (7) A7 names `tests/test_engine_parity.py` with 9-pass/3-skip semantics + local bit-exact run. (8) B1/B2 use the real results path + `(keys,bounds,x0)` tuple + phase-2 freeze template. (9) B4/B6 define the mortality-delta noise band in matching units.

**Placeholder scan:** remaining `...` are in A0 helper bodies (signatures + contracts fixed; bodies filled from the Step-1 harness) and the A5/A6/A8 scenario builders (`_run_single_predation_step_*`, `_run_baltic_short_with_diet`) which name the exact existing harness to copy and the exact value to assert. No "TODO/handle edge cases".

**Type consistency:** `fr_shape` int32 / `fr_halfsat` float64; codes `type1→1/type2→2/type3→3`; sentinel `_FR_HALFSAT_SENTINEL=1.0`; branch vars `r,k,g_form,cap,g`; oracle `_g_ref` identical across A5/A6 and the kernel. Background `BackgroundSpeciesInfo.fr_shape:int`/`fr_halfsat:float` → `np.array(..., dtype=np.int32/float64)` at concat. ✅
