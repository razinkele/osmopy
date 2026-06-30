# C2 — Run nbackground>0 configs on Java 4.4.1 from the UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline — subagent dispatch is spend-limited) or subagent-driven-development if available. Steps use checkbox (`- [ ]`).

**Goal:** From the UI with the 4.4.1 jar selected, a Baltic / baltic_ev run on the Java engine is no longer blocked — it stages the background species (A's recipe) and runs to completion. With the 4.3.3 jar (or an unrecognised background species) it stays blocked with a clear reason.

**Architecture:** Extract A's staging from `scripts/baltic_440_smoke.py` into `osmose/java_background_staging.py` (generalized over background species), make `java_engine_block_reason` version-aware, and wire `stage_background_for_java` (+ its `-P` overrides) into `_run_java_engine`. Spec: `docs/superpowers/specs/2026-06-30-c2-ui-java-440-background-design.md`.

**Tech Stack:** Python 3.12, numpy/xarray (predator NetCDF), the Java 4.4.1 jar, Shiny, pytest.

## Global Constraints

- **Main venv ABSOLUTE path** `/home/razinka/osmose/osmose-python/.venv/bin/python` — NO `.venv` in the worktree. Lint: `ruff check` + `ruff format --check` on `osmose/ ui/ tests/`.
- **Baltic-specific scope**: `BG_ACCESS` + `BG_DIET_STAGE_THRESHOLD` are A's hand-authored Baltic tables. The block ALLOWS `nbackground>0` on a `≥4.4.0` jar ONLY when `background_staging_supported(config)` (every `type=background` species name is in `BG_ACCESS`).
- **No import cycle**: `osmose/java_background_staging.py` must NOT import `osmose.runner` or `ui.pages.run` (no `write_temp_config` — the orchestrator takes an already-staged dir). Deps: reader, numpy, xarray, stdlib.
- **No Python-engine change.** **Staged-copy only** — never touch `data/`.
- **Cutoff override via `-P`**: `stage_background_for_java` returns `{"output.cutoff.enabled": "false"}`; the UI passes it to `runner.run(overrides=...)` (A's validated CLI path).
- Version compare via `osmose.config.aliases._numeric_version`; jar→version via `target_version_for_jar`.

---

### Task 1: Extract + generalize A's staging into `osmose/java_background_staging.py`

**Files:** Create `osmose/java_background_staging.py`; modify `scripts/baltic_440_smoke.py` (import from the module); create `tests/test_java_background_staging.py` (move from `tests/test_baltic_440_staging.py`).

**Interfaces produced:** `BG_ACCESS`, `BG_DIET_STAGE_THRESHOLD`, `inline_biomass_series`, `augment_accessibility`, `background_staging_supported(config)->bool`, `stage_background_for_java(stage_dir, raw_config)->dict[str,str]`.

- [ ] **Step 1: Create the module** — move VERBATIM from `scripts/baltic_440_smoke.py`: `BG_ACCESS`, `inline_biomass_series`, `augment_accessibility`, `_write_background_movement_maps` (+ its helpers `_read_*` only if needed by staging; the validation helpers `assert_predators_feed`/`_read_biomass_means` STAY in the smoke script). Add `BG_DIET_STAGE_THRESHOLD = {"GreySeal": 90, "Cormorant": 65}`.

- [ ] **Step 2: Add `background_staging_supported`**:
```python
def _background_species(config):
    """Yield (idx:str, name:str) for each species.type.spN == 'background'."""
    for k, v in config.items():
        if k.startswith("species.type.sp") and str(v).strip().lower() == "background":
            idx = k[len("species.type.sp"):]
            if idx.isdigit():
                yield idx, config.get(f"species.name.sp{idx}", "")

def background_staging_supported(config) -> bool:
    names = [name for _, name in _background_species(config)]
    return bool(names) and all(n in BG_ACCESS for n in names)
```

- [ ] **Step 3: Generalize the orchestrator** `stage_background_for_java(stage_dir, raw_config) -> dict[str,str]` from `scripts/baltic_440_smoke.py::stage_and_run` (the staging part, NOT `write_temp_config`/the jar run). Replace the hardcoded `("14","GreySeal"),("15","Cormorant")` / `nclass=2` / `cod_juvenile.csv` with config-derived iteration:
```python
def stage_background_for_java(stage_dir, raw_config):
    master = stage_dir / "osm_all-parameters.csv"
    ndt = int(float(raw_config.get("simulation.time.ndtperyear", "24") or "24"))
    nc = next(stage_dir.glob("*predator*biomass*.nc"))  # the bg-forcing NetCDF (Baltic: baltic_predator_biomass.nc)
    ref_map = next((stage_dir / "maps").glob("*juvenile*.csv"), None) or next((stage_dir / "maps").glob("*.csv"))
    extra, predators = [], {}
    for idx, name in _background_species(raw_config):
        nclass = int(float(raw_config.get(f"species.nclass.sp{idx}", "1") or "1"))
        series = inline_biomass_series(nc, name)
        extra.append(f"species.biomass.sp{idx} ; " + ";".join(f"{v:.6g}" for v in series))
        extra.append(f"species.biomass.nsteps.year.sp{idx} ; {ndt}")
        extra.append(f"simulation.nschool.sp{idx} ; 10")
        extra.append(f"output.diet.stage.threshold.sp{idx} ; {BG_DIET_STAGE_THRESHOLD[name]}")
        extra.extend(_write_background_movement_maps(stage_dir, [(name, nclass)], list(range(ndt)), ref_map))
        predators[name] = BG_ACCESS[name]
    extra.append("output.cutoff.enabled ; false")  # belt-and-suspenders; the -P override is authoritative
    master.write_text(master.read_text() + "\n".join(extra) + "\n")
    augment_accessibility(stage_dir / "predation-accessibility.csv", predators)
    _augment_catchability_discards(stage_dir, predators)   # zero rows; move from the smoke script
    return {"output.cutoff.enabled": "false"}
```
(Extract the catchability/discards zero-row loop from `stage_and_run` into `_augment_catchability_discards(stage_dir, predators)` in the module.)

- [ ] **Step 4: Point the smoke script at the module** — `scripts/baltic_440_smoke.py` imports `inline_biomass_series, augment_accessibility, BG_ACCESS, stage_background_for_java` from `osmose.java_background_staging`; its `stage_and_run` now does `write_temp_config(...)` then `overrides = stage_background_for_java(stage, raw)` then runs the jar with `cmd += [f"-P{k}={v}" for k,v in overrides.items()]`. Delete the now-duplicated staging code from the script.

- [ ] **Step 5: Move + retarget tests** — `tests/test_java_background_staging.py`: move the cases from `tests/test_baltic_440_staging.py` (re-import from `osmose.java_background_staging`), KEEPING the source-untouched guard; add `test_background_staging_supported` (Baltic source config → True; a `{species.type.sp9: background, species.name.sp9: Unknown}` config → False) and a `stage_background_for_java` test asserting it returns `{"output.cutoff.enabled":"false"}` and emits `species.biomass.sp14`/`output.diet.stage.threshold.sp14` into a staged Baltic dir. Delete `tests/test_baltic_440_staging.py`.

- [ ] **Step 6: Run unit tests + lint**
`PYTHONPATH=. .venv/bin/python -m pytest tests/test_java_background_staging.py -q` → PASS. `ruff check/format` clean.

- [ ] **Step 7: Integration — prove extraction fidelity** (the key gate):
`PYTHONPATH=. .venv/bin/python scripts/baltic_440_smoke.py` → exit 0, no collapse, predators feed (identical to A). If it regresses, the generalization drifted from A's hardcoded behaviour.

- [ ] **Step 8: Commit** `git add osmose/java_background_staging.py scripts/baltic_440_smoke.py tests/test_java_background_staging.py && git rm tests/test_baltic_440_staging.py && git commit -m "feat(c2): extract+generalize Baltic Java-4.4.1 staging into osmose/java_background_staging.py"`

---

### Task 2: Version-aware `java_engine_block_reason` + `engine_capabilities` threading

**Files:** Modify `osmose/runner.py` (`java_engine_block_reason`), `osmose/engine_capabilities.py` (`describe_engine`/`_describe_java`). Test: `tests/test_engine_capabilities.py` (+ a runner test).

- [ ] **Step 1: Failing test** — `tests/test_engine_capabilities.py` (append) / a new `tests/test_java_block_version.py`:
```python
import pytest
from osmose.runner import java_engine_block_reason
from osmose.config.reader import OsmoseConfigReader

def _baltic():
    return dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))

def test_block_matrix():
    focal = {"simulation.nbackground": "0"}
    assert java_engine_block_reason(focal, "4.4.1") is None          # no bg -> allow
    bal = _baltic()
    assert java_engine_block_reason(bal, "4.3.3") is not None         # bg + 4.3.3 -> block
    assert java_engine_block_reason(bal, None) is not None            # bg + unknown -> block
    assert java_engine_block_reason(bal, "4.4.1") is None             # bg + 4.4.1 + supported -> allow
    unknown = {"simulation.nbackground": "1", "species.type.sp9": "background", "species.name.sp9": "Yeti"}
    assert java_engine_block_reason(unknown, "4.4.1") is not None     # bg + 4.4.1 + unsupported -> block
```

- [ ] **Step 2: Run — fail** (current signature takes only `config`).

- [ ] **Step 3: Implement** `java_engine_block_reason(config, jar_version: str | None = None)`:
```python
    n_bg = ...  # existing parse
    if n_bg <= 0:
        return None
    from osmose.config.aliases import _numeric_version
    if jar_version is not None and _numeric_version(jar_version) >= _numeric_version("4.4.0"):
        from osmose.java_background_staging import background_staging_supported
        if background_staging_supported(config):
            return None
        unsupported = [n for _, n in _bg_names(config) if ...]   # names not in BG_ACCESS
        return (f"Background species {unsupported} are not staging-supported on the Java engine; "
                "use the Python engine.")
    return (<existing nbackground>0 reason>)
```
(Keep the existing reason string for the `<4.4.0`/`None` path.) Then thread the jar version:
`engine_capabilities.describe_engine(engine, config, jar_version=None)` → `_describe_java(config, jar_version)` → `java_engine_block_reason(config, jar_version)`.

- [ ] **Step 4: Run — pass** `pytest tests/test_java_block_version.py tests/test_engine_capabilities.py -q` → PASS.

- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(c2): version-aware java_engine_block_reason (allow staging-supported nbackground>0 on >=4.4.0 jar) + thread jar_version through engine_capabilities"`

---

### Task 3: Wire into the UI run path + gate

**Files:** Modify `ui/pages/run.py` (`_run_java_engine` + the `:824` gate + the `describe_engine` call at `:23`/usage). Test: `tests/test_ui_run.py` (or the existing run-page test).

- [ ] **Step 1: Gate passes the jar version** — `run.py:824`: `block = java_engine_block_reason(config, target_version_for_jar(Path(state.jar_path.get())))`. Update the `describe_engine(...)` call site(s) to pass `jar_version=target_version_for_jar(Path(state.jar_path.get()))`.

- [ ] **Step 2: Stage + pass overrides in `_run_java_engine`** — after `write_temp_config` (line 369) builds `config_path`, before `runner.run`:
```python
    extra_overrides = {}
    n_bg = int(float(config.get("simulation.nbackground", "0") or "0"))
    if n_bg > 0 and _numeric_version(target_version_for_jar(jar_path)) >= _numeric_version("4.4.0"):
        from osmose.java_background_staging import stage_background_for_java
        extra_overrides = stage_background_for_java(config_path.parent, config)
```
and pass `overrides=extra_overrides` to `runner.run(...)` (line ~408). (Confirm the run config dict variable name carries the background `species.type/name` — it's the same `config` the gate checked.)

- [ ] **Step 3: UI gate test** — assert (via a small test calling the gate logic or `java_engine_block_reason` with the wired jar version) that Baltic is NOT blocked with the 4.4.1 jar and IS blocked with the 4.3.3 jar. Reuse the block-matrix coverage if a full UI-handler test is heavy.

- [ ] **Step 4: Full-suite sweep** — `PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider` → only the known pre-existing failures remain (docs `0.13.0`, `run_observer`); fix any NEW failure from the changed signatures (callers of `java_engine_block_reason`/`describe_engine`). Lint clean.

- [ ] **Step 5: Optional manual smoke** — launch the app, load Baltic, select the 4.4.1 jar + Java engine → run proceeds (not blocked). (Document; not a CI gate.)

- [ ] **Step 6: Commit** `git add -A && git commit -m "feat(c2): wire background staging + version-aware gate into the UI Java run path"`

---

## Notes for the executor
- **Extraction fidelity is the risk** — the generalized orchestrator must reproduce A's exact Baltic result. Task 1 Step 7 (the real 4.4.1 run) is the gate; if it drifts, diff against `stage_and_run`'s hardcoded values.
- The orchestrator takes an **already-staged dir** (write_temp_config done by the caller) — do NOT import `write_temp_config`/`ui.pages.run` into the staging module (cycle).
- `runner.run` already accepts `overrides: dict` → `-Pkey=value` (`runner.py:128`); pass the staging's return there.
- Inline execution (executing-plans); main-venv absolute path; never create a `.venv` in the worktree.
