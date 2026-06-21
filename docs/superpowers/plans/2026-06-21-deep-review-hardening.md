# Deep-Review Latent-Item Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four LATENT-only deep-review items: warn on unsupported fishing selectivity types 2/3, stop the wizard/maps/validation tempdir leaks, record real run `duration_sec`, and add a numba-disabled CI leg that exercises the pure-Python engine fallbacks.

**Architecture:** Four independent, low-risk changes. Items 1–3 are warn/cleanup/telemetry fixes with focused unit tests (TDD); item 4 is a CI-config addition verified by the PR run. No bundled-config behavior or Java-parity change.

**Tech Stack:** Python 3.12, pytest, numpy, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-06-21-deep-review-hardening-design.md`

**Conventions (CLAUDE.md):** use `.venv/bin/python` / `.venv/bin/<tool>`; ruff line length 100; lint scope is `osmose/ ui/ tests/`. Stage only the files each task names — the working tree has unrelated untracked files (screenshots, scripts); never `git add -A`.

---

## File structure

| File | Change |
|---|---|
| `osmose/engine/config.py` | extend `_warn_unsupported_mortality_features` with a selectivity 2/3 warning (item 1) |
| `tests/test_engine_selectivity_warning.py` | new test for the selectivity warning (item 1) |
| `osmose/cleanup.py` | add `osmose_wizard_`/`osmose_maps_`/`osmose_val_` to `_OSMOSE_PREFIXES` (item 2) |
| `tests/test_cleanup_prefixes.py` | new test: prefixes registered + swept (item 2) |
| `ui/pages/run.py` | thread `start_monotonic` into `_handle_result`; real `duration_sec` (item 3) |
| `tests/test_run_duration.py` | new test: `_handle_result` records real duration (item 3) |
| `.github/workflows/ci.yml` | new `test-no-numba` job (item 4) |

---

## Task 1: Warn on fishing selectivity types 2/3

**Files:**
- Create: `tests/test_engine_selectivity_warning.py`
- Modify: `osmose/engine/config.py` (append a block at the end of `_warn_unsupported_mortality_features`, which currently ends at ~line 1572 just before `@classmethod def from_dict`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_selectivity_warning.py`:

```python
"""Item 1: the interleaved mortality loop knife-edges fishing selectivity types
2 (Gaussian) / 3 (log-normal). EngineConfig must warn rather than silently
diverge. (Warn, not wire -- consistent with the PR-1 reject-not-wire decision.)
"""

import logging

from osmose.engine import config as config_module
from osmose.engine.config import EngineConfig

# A minimal valid 1-species config (mirrors tests/test_engine_config_validation.py).
_MINIMAL = {
    "simulation.nspecies": "1",
    "simulation.nschool.sp0": "20",
    "species.name.sp0": "Anchovy",
    "species.linf.sp0": "15.0",
    "species.k.sp0": "0.4",
    "species.t0.sp0": "-0.1",
    "species.egg.size.sp0": "0.1",
    "species.length2weight.condition.factor.sp0": "0.006",
    "species.length2weight.allometric.power.sp0": "3.0",
    "species.lifespan.sp0": "3",
    "species.vonbertalanffy.threshold.age.sp0": "1.0",
    "mortality.subdt": "10",
    "predation.ingestion.rate.max.sp0": "3.5",
    "predation.efficiency.critical.sp0": "0.57",
}


def _fresh_config() -> EngineConfig:
    config_module._WARNED_UNSUPPORTED_MORTALITY.clear()
    return EngineConfig.from_dict(_MINIMAL)


def test_warns_on_selectivity_type_2(caplog):
    cfg = _fresh_config()
    cfg.fishing_selectivity_type[0] = 2  # Gaussian
    with caplog.at_level(logging.WARNING, logger="osmose.engine.config"):
        cfg._warn_unsupported_mortality_features()
    assert any("selectivity type 2" in r.getMessage() for r in caplog.records)


def test_warns_on_selectivity_type_3(caplog):
    cfg = _fresh_config()
    cfg.fishing_selectivity_type[0] = 3  # log-normal
    with caplog.at_level(logging.WARNING, logger="osmose.engine.config"):
        cfg._warn_unsupported_mortality_features()
    assert any("selectivity type" in r.getMessage() and "3" in r.getMessage()
               for r in caplog.records)


def test_no_selectivity_warning_for_types_0_and_1(caplog):
    cfg = _fresh_config()
    cfg.fishing_selectivity_type[0] = 1  # logistic -- supported
    with caplog.at_level(logging.WARNING, logger="osmose.engine.config"):
        cfg._warn_unsupported_mortality_features()
    assert not any("selectivity type" in r.getMessage() for r in caplog.records)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_selectivity_warning.py -q`
Expected: `test_warns_on_selectivity_type_2` and `_3` FAIL (no warning emitted yet); `test_no_selectivity_warning...` passes.

- [ ] **Step 3: Add the selectivity warning block**

In `osmose/engine/config.py`, inside `_warn_unsupported_mortality_features`, append this block AFTER the existing `frc = self.fishing_rate_by_dt_by_class` block and BEFORE the method ends (i.e. as the last block of the method, before the blank line preceding `@classmethod`):

```python
        sel = self.fishing_selectivity_type
        affected = [_sp(i) for i in range(self.n_species) if sel[i] in (2, 3)]
        if affected:
            _warn_once(
                "Fishing selectivity type 2 (Gaussian) / 3 (log-normal) is configured for "
                f"{', '.join(affected)} but the Python engine's interleaved mortality loop "
                "applies only knife-edge (type 0) and logistic (type 1) selectivity — types "
                "2/3 are silently treated as length knife-edge. Use selectivity type 0 or 1, "
                "or run the Java engine."
            )
```

Notes: iterate `range(self.n_species)` (focal only) — `fishing_selectivity_type` is length n_total with trailing background `-1` entries, and `_sp` keys off the focal-only `species_names`. `_warn_once` and `_sp` are the local closures already defined at the top of this method.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_selectivity_warning.py -q`
Expected: all 3 PASS.

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff check osmose/engine/config.py tests/test_engine_selectivity_warning.py && .venv/bin/ruff format --check osmose/engine/config.py tests/test_engine_selectivity_warning.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_selectivity_warning.py
git commit -m "engine: warn on unsupported fishing selectivity types 2/3 (silently knife-edged)"
```

---

## Task 2: Sweep the wizard / maps / validation tempdir prefixes

**Files:**
- Create: `tests/test_cleanup_prefixes.py`
- Modify: `osmose/cleanup.py` (the `_OSMOSE_PREFIXES` tuple at lines 20-26)

**Context:** These three `mkdtemp` prefixes are session-lifetime or transient dirs that are NOT in `_OSMOSE_PREFIXES`, so `cleanup_old_temp_dirs()` never sweeps them. They MUST NOT be deleted promptly in their handlers (`osmose_wizard_` and `osmose_maps_` back `state.config_dir`, read by run/calibration/map after the handler). The fix is sweep-membership only.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cleanup_prefixes.py`:

```python
"""Item 2: wizard/maps/validation tempdirs must be swept by cleanup_old_temp_dirs.

These dirs are session-lifetime (osmose_wizard_/osmose_maps_ back state.config_dir)
or transient (osmose_val_), so they are cleaned by the age-gated sweep, never by a
prompt rmtree in their create handler.
"""

import tempfile
from pathlib import Path

from osmose.cleanup import _OSMOSE_PREFIXES, cleanup_old_temp_dirs


def test_new_prefixes_registered():
    for prefix in ("osmose_wizard_", "osmose_maps_", "osmose_val_"):
        assert prefix in _OSMOSE_PREFIXES


def test_sweep_removes_new_prefix_dirs():
    made = [Path(tempfile.mkdtemp(prefix=p))
            for p in ("osmose_wizard_", "osmose_maps_", "osmose_val_")]
    for d in made:
        assert d.exists()
    cleanup_old_temp_dirs(max_age_hours=0)  # 0 == remove all osmose temp dirs
    for d in made:
        assert not d.exists(), f"{d} was not swept"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cleanup_prefixes.py -q`
Expected: both tests FAIL (prefixes not yet registered; dirs not swept).

- [ ] **Step 3: Register the prefixes**

In `osmose/cleanup.py`, replace the `_OSMOSE_PREFIXES` tuple:

```python
_OSMOSE_PREFIXES = (
    "osmose_run_",
    "osmose_cal_",
    "osmose_sens_",
    "osmose_export_",
    "osmose_demo_",
    "osmose_wizard_",
    "osmose_maps_",
    "osmose_val_",
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cleanup_prefixes.py -q`
Expected: both PASS.

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff check osmose/cleanup.py tests/test_cleanup_prefixes.py && .venv/bin/ruff format --check osmose/cleanup.py tests/test_cleanup_prefixes.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add osmose/cleanup.py tests/test_cleanup_prefixes.py
git commit -m "cleanup: sweep osmose_wizard_/osmose_maps_/osmose_val_ temp dirs"
```

---

## Task 3: Record real run `duration_sec` (both engine paths)

**Files:**
- Create: `tests/test_run_duration.py`
- Modify: `ui/pages/run.py`

**Context:** `_handle_result` (run.py:380) builds `RunRecord(..., duration_sec=0, ...)`. The Java path calls it synchronously with `config` passed as a direct arg (run.py:377); the Python path is fire-and-forget and passes `config` via `_run_config_cell` read by the poll at run.py:503. `RunResult` (runner.py) has no timing field, so the launch time must be threaded as a parameter (mirroring `config`) — a single shared cell would misfire on the Java path. `import time` is already present (run.py:7).

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_duration.py`:

```python
"""Item 3: _handle_result must persist the real elapsed time, not 0."""

import types

import ui.pages.run as run_mod


def test_handle_result_records_real_duration(monkeypatch, tmp_path):
    captured = {}

    class _FakeHistory:
        def save(self, record):
            captured["record"] = record

    monkeypatch.setattr("osmose.history.default_run_history", lambda: _FakeHistory())
    # Freeze the "now" end time; start is passed in as 95.0 -> elapsed 5.0s.
    monkeypatch.setattr(run_mod.time, "monotonic", lambda: 100.0)

    result = types.SimpleNamespace(
        returncode=0, output_dir=str(tmp_path), status="ok", message=""
    )
    state = types.SimpleNamespace(
        run_result=types.SimpleNamespace(set=lambda v: None),
        output_dir=types.SimpleNamespace(set=lambda v: None),
    )
    status = types.SimpleNamespace(set=lambda v: None)

    run_mod._handle_result(
        result, {"k": "v"}, state, None, status, start_monotonic=95.0
    )

    assert "record" in captured
    assert captured["record"].duration_sec == 5.0


def test_both_engine_paths_thread_start_time():
    """Guard the wiring: the start time is threaded as a parameter through both
    engine paths (not read from a single Python-only cell)."""
    import pathlib

    src = pathlib.Path(run_mod.__file__).read_text()
    assert "start_monotonic" in src
    assert "_run_start_cell" in src  # Python fire-and-forget path
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run_duration.py -q`
Expected: FAIL — `_handle_result` has no `start_monotonic` parameter yet (TypeError) and the source assertions fail.

- [ ] **Step 3a: Add the `start_monotonic` parameter + real duration in `_handle_result`**

In `ui/pages/run.py`, change the `_handle_result` signature (line 380):

```python
def _handle_result(result, config, state, run_log, status, start_monotonic=None):
```

And replace the `RunRecord(...)` construction (the `duration_sec=0` line, ~line 401):

```python
            duration_sec = (
                max(0.0, time.monotonic() - start_monotonic)
                if start_monotonic is not None
                else 0.0
            )
            record = RunRecord(
                config_snapshot=config,
                duration_sec=duration_sec,
                output_dir=str(result.output_dir),
                summary={},
            )
```

- [ ] **Step 3b: Capture `t0` once in `handle_run` and thread it through both paths**

In `handle_run`, immediately before the `if engine_mode == "python":` line (run.py:803), add:

```python
        run_t0 = time.monotonic()
```

Python path: next to `_run_config_cell[0] = config` (run.py:820), add:

```python
            _run_start_cell[0] = run_t0
```

Declare the companion cell next to `_run_config_cell` (run.py:444):

```python
    _run_start_cell: list = [None]  # run start (time.monotonic) for duration_sec
```

Python poll site (run.py:503) — pass the start time:

```python
            _handle_result(
                result, _run_config_cell[0], state, run_log, status, _run_start_cell[0]
            )
```

Java path: pass `run_t0` into `_run_java_engine(...)` at its call site (run.py:830) as a new trailing argument `start_monotonic=run_t0`; add `start_monotonic=None` to the `_run_java_engine` signature (run.py:311); and forward it in that function's `_handle_result(...)` call (run.py:377):

```python
    _handle_result(result, config, state, run_log, status, start_monotonic)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_run_duration.py -q`
Expected: both PASS.

- [ ] **Step 5: Lint (note: `ui/pages/run.py` IS in ruff scope)**

Run: `.venv/bin/ruff check ui/pages/run.py tests/test_run_duration.py && .venv/bin/ruff format --check ui/pages/run.py tests/test_run_duration.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/run.py tests/test_run_duration.py
git commit -m "run: record real run duration_sec across both engine paths"
```

---

## Task 4: numba-disabled CI leg (exercise pure-Python fallbacks)

**Files:**
- Modify: `.github/workflows/ci.yml` (add a `test-no-numba` job after the existing `docker` job)

**Context:** With numba uninstalled, `_HAS_NUMBA = False` and the engine transparently uses the pure-Python fallbacks. The leg must run GENERAL engine tests that run a full step (mortality/predation/movement) — NOT the numba-specific files (`test_movement_numba.py`, `test_jit_determinism.py`, the `_HAS_NUMBA`-gated parts of `test_engine_predation_helpers.py`), which `skipif`/`importorskip` themselves out when numba is absent. The chosen general tests do not skip without numba and so exercise the fallback. Confirmed safe: every `import numba` is `ImportError`-guarded.

- [ ] **Step 1: Add the job**

Append to `.github/workflows/ci.yml` (sibling of `lint`/`type-check`/`test`/`docker`, same indentation):

```yaml
  test-no-numba:
    # Exercise the pure-Python engine fallbacks (osmose/engine/processes/
    # {mortality,predation,movement}.py) that the main test job never hits
    # because numba is always installed there. General engine tests below run a
    # full simulation step, so with numba absent they transparently use the
    # _HAS_NUMBA == False path. (numba-specific tests skipif themselves out
    # without numba, so they are deliberately NOT in this subset.)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Uninstall numba (force pure-Python fallback)
        run: pip uninstall -y numba

      - name: Assert numba is absent
        run: python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('numba') is None else 1)"

      - name: Run engine subset on the pure-Python path
        run: |
          pytest -p no:cacheprovider -q \
            tests/test_engine_mortality_loop.py \
            tests/test_engine_mortality.py \
            tests/test_engine_movement.py \
            tests/test_engine_map_movement.py \
            tests/test_engine_diet.py \
            tests/test_engine_natural.py
```

- [ ] **Step 2: Validate the workflow YAML parses**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML OK')"`
Expected: prints `YAML OK`.

- [ ] **Step 3: Verify the subset runs (does not all-skip) without numba — in a THROWAWAY venv**

Do NOT uninstall numba from `.venv` (other tasks/agents share it). Use a disposable venv:

```bash
python -m venv /tmp/nonumba && /tmp/nonumba/bin/pip install -q -e ".[dev]" && /tmp/nonumba/bin/pip uninstall -y numba
/tmp/nonumba/bin/python -c "import importlib.util; assert importlib.util.find_spec('numba') is None"
/tmp/nonumba/bin/pytest -p no:cacheprovider -q tests/test_engine_mortality_loop.py tests/test_engine_mortality.py tests/test_engine_movement.py tests/test_engine_map_movement.py tests/test_engine_diet.py tests/test_engine_natural.py
```

Expected: the run reports a healthy number of PASSED tests (NOT "all skipped") and 0 failures. If any listed file is entirely skipped or fails purely due to numba-absence, drop/replace it with another general engine test (e.g. `tests/test_engine_fishing.py`, `tests/test_engine_growth.py`) and update both the job (Step 1) and this command, then re-run. Remove the throwaway venv when done: `rm -rf /tmp/nonumba`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add numba-disabled leg exercising pure-Python engine fallbacks"
```

---

## Final verification (before finishing the branch)

- [ ] `.venv/bin/python -m pytest tests/test_engine_selectivity_warning.py tests/test_cleanup_prefixes.py tests/test_run_duration.py -q` → all pass.
- [ ] Full suite (numba path): `.venv/bin/python -m pytest -n auto -q` → green (no behavior regression).
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/` and `.venv/bin/ruff format --check osmose/ ui/ tests/` → clean.
- [ ] `.venv/bin/pyright --pythonpath .venv/bin/python` (or per repo convention) → no new errors in the changed files.
- [ ] `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` → parses.

Then use **superpowers:finishing-a-development-branch**.

---

## Spec coverage map

- Item 1 (selectivity 2/3 warn, focal-only) → Task 1.
- Item 2 (wizard/maps/val sweep-membership, no prompt delete) → Task 2.
- Item 3 (real duration_sec, parameter-threaded across both engine paths) → Task 3.
- Item 4 (numba-disabled CI leg, general-engine subset that doesn't all-skip) → Task 4.
- Out of scope (wiring 2/3, .env rotation, parity/bundled-config behavior) → not implemented, per spec.
