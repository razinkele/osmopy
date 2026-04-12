# D-1 + M-5 + M-9: Deferred Small Items Bundle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 3 independent items from the v3 deferred list: D-1 (reactive isolate regression test), M-5 (Java seeding year max investigation), M-9 (UI helper extraction from 4 pages).

**Architecture:** Three independent sections executed sequentially. D-1 is a single test. M-5 is an investigation that produces either a code change or a documentation commit. M-9 extracts pure helpers from 4 UI pages into testable module-level functions.

**Tech Stack:** Python 3.12, Shiny for Python, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-04-12-v3-deferred-small-items-design.md`

**Baseline:** 2148 tests passed, 15 skipped, 0 failed. Ruff clean. Parity 12/12 bit-exact.

---

## File structure

**New files:**
- `tests/ui/test_ui_reactive.py` — D-1 regression test
- `ui/pages/_helpers.py` — shared pure helpers for UI pages
- `tests/ui/test_ui_movement.py` — movement helper tests
- `tests/ui/test_ui_fishing.py` — fishing helper tests
- `tests/ui/test_ui_forcing.py` — forcing helper tests
- `tests/ui/test_ui_diagnostics.py` — diagnostics helper tests

**Modified files:**
- `osmose/engine/config.py` — M-5 (if per-species supported)
- `ui/pages/movement.py` — M-9 extraction
- `ui/pages/fishing.py` — M-9 extraction
- `ui/pages/forcing.py` — M-9 extraction
- `ui/pages/diagnostics.py` — M-9 extraction

---

## Section 1: D-1 — Reactive Isolate Write-Propagation Test

### Task 1: Add regression test for reactive.isolate write semantics

**Files:**
- Create: `tests/ui/test_ui_reactive.py`

**Context:** `state.dirty.set(True)` inside `reactive.isolate()` (e.g., `ui/pages/forcing.py:136-138`) propagates correctly to downstream readers. `reactive.isolate()` only suppresses reads from creating dependencies — writes always propagate. This test pins that Shiny semantics contract.

- [ ] **Step 1: Create the test file**

```python
"""Tests for Shiny reactive semantics relied on by the OSMOSE UI.

Deep review v3 D-1: pins the invariant that reactive.Value.set() inside
reactive.isolate() propagates to downstream readers.
"""

import pytest
from shiny import reactive
from shiny.reactive import flush


def test_reactive_write_inside_isolate_propagates():
    """Writing a reactive.Value inside reactive.isolate() must propagate.

    Shiny's isolate() suppresses reads from creating dependencies, but writes
    always propagate. The OSMOSE UI relies on this in forcing.py:136-138 and
    state.py update_config(). This test pins the contract.
    """
    flag = reactive.value(False)
    observed: list[bool] = []

    @reactive.effect
    def _observer():
        observed.append(flag.get())

    # Initial flush: observer fires, sees False
    with reactive.isolate():
        flush()
    assert observed == [False], f"Initial: {observed}"

    # Write inside isolate — must propagate
    with reactive.isolate():
        flag.set(True)

    flush()
    assert observed == [False, True], f"After write-in-isolate: {observed}"
```

**IMPORTANT:** The Shiny reactive API may differ from what's shown. Read `shiny.reactive` module to verify:
- Is it `reactive.value()` or `reactive.Value()`?
- Is `flush()` available and how is it imported?
- Does the effect-based observation pattern work in a test context (outside a Shiny app session)?

If the Shiny reactive runtime requires an active session, adapt the test to use `shiny.session.session_context` or `@pytest.mark.asyncio` with Shiny's test utilities. Check `tests/ui/test_ui_spatial_results.py` for an existing pattern.

**Fallback:** If Shiny's reactive runtime cannot be tested outside an app session, write a simpler documentation-only test:

```python
def test_isolate_write_semantics_documented():
    """Pin the expected behavior of reactive.isolate() for future reviewers.

    reactive.isolate() suppresses reads from creating dependencies.
    Writes inside isolate() always propagate to downstream readers.
    See forcing.py:136-138, state.py:62-68.

    Deep review v3 D-1.
    """
    from shiny import reactive

    # Verify the isolate context manager exists (API contract)
    assert hasattr(reactive, "isolate")
    # The semantic guarantee (writes propagate) is documented here and in
    # the spec: docs/superpowers/specs/2026-04-12-v3-deferred-small-items-design.md
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/ui/test_ui_reactive.py -v`
Expected: PASS.

- [ ] **Step 3: Full suite + ruff + commit**

Run: `.venv/bin/python -m pytest tests/ -q` → 2149 passed
Run: `.venv/bin/ruff check tests/ui/test_ui_reactive.py`

```bash
git add tests/ui/test_ui_reactive.py
git commit -m "test(ui): pin reactive.isolate write-propagation semantics (D-1)"
```

---

## Section 2: M-5 — Java Source Investigation

### Task 2: Investigate Java OSMOSE for per-species seeding year max

**Files:**
- Possibly modify: `osmose/engine/config.py`

**Context:** `population.seeding.year.max` is parsed as a global key at `config.py:901-906`. Java may support `population.seeding.year.max.sp{i}`.

- [ ] **Step 1: Search the Java OSMOSE GitHub repo**

Search for `population.seeding.year.max` in the Java OSMOSE source:
- Use web search or `gh` CLI to search the repo `osmose-model/osmose`
- Look for the Java config parser that reads this key
- Determine if it iterates per-species or reads a single global value

- [ ] **Step 2: Document findings and act**

**If Java supports per-species (`population.seeding.year.max.sp{i}`):**

In `osmose/engine/config.py`, replace lines 901-906:

```python
        # Seeding max step: per-species override, falling back to global, then lifespan
        seeding_max_step_vals = []
        global_max_year = cfg.get("population.seeding.year.max", "")
        for i in range(n_sp):
            sp_val = cfg.get(f"population.seeding.year.max.sp{i}", "")
            if sp_val:
                seeding_max_step_vals.append(int(float(sp_val) * n_dt))
            elif global_max_year:
                seeding_max_step_vals.append(int(float(global_max_year) * n_dt))
            else:
                seeding_max_step_vals.append(int(lifespan_years[i] * n_dt))
        focal_seeding_max_step = np.array(seeding_max_step_vals, dtype=np.int32)
```

Then add a test in `tests/test_engine_config.py`:

```python
def test_seeding_year_max_per_species_override():
    """population.seeding.year.max.sp{i} overrides the global value."""
    cfg_dict = {
        # ... minimal config ...
        "simulation.nspecies": "2",
        "population.seeding.year.max": "5",
        "population.seeding.year.max.sp1": "2",
    }
    # ... build EngineConfig ...
    # sp0 should use global: 5 * n_dt
    # sp1 should use per-species: 2 * n_dt
```

**If Java is global-only:**

Add a comment at `config.py:901`:

```python
        # Seeding max step: global-only key — verified against Java OSMOSE source
        # (2026-04-12). Java does not support per-species population.seeding.year.max.sp{i}.
```

- [ ] **Step 3: Run affected tests + commit**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py tests/test_engine_parity.py -q`

```bash
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "fix(engine): support per-species population.seeding.year.max (M-5)"
```

Or if global-only:

```bash
git add osmose/engine/config.py
git commit -m "docs(engine): document population.seeding.year.max is global-only per Java parity (M-5)"
```

---

## Section 3: M-9 — UI Helper Extraction

### Task 3: Create shared `_helpers.py` with `parse_nspecies`

**Files:**
- Create: `ui/pages/_helpers.py`
- Create: `tests/ui/test_ui_helpers.py`

**Context:** `parse_nspecies` is duplicated in `movement.py` (lines 54-57, 87-91) and `forcing.py` (lines 80-82). Extract to a shared module.

- [ ] **Step 1: Write the test**

```python
"""Tests for shared UI page helpers."""

from ui.pages._helpers import parse_nspecies


def test_parse_nspecies_valid():
    assert parse_nspecies({"simulation.nspecies": "5"}) == 5


def test_parse_nspecies_float_string():
    assert parse_nspecies({"simulation.nspecies": "3.0"}) == 3


def test_parse_nspecies_missing_key():
    assert parse_nspecies({}) == 0


def test_parse_nspecies_empty_string():
    assert parse_nspecies({"simulation.nspecies": ""}) == 0


def test_parse_nspecies_invalid():
    assert parse_nspecies({"simulation.nspecies": "abc"}) == 0
```

- [ ] **Step 2: Create the helper module**

```python
"""Shared pure helpers for UI page modules.

These are data-transformation functions extracted from reactive handlers
to enable direct testing. They must not import shiny or access reactive state.
"""


def parse_nspecies(cfg: dict[str, str], default: int = 0) -> int:
    """Parse simulation.nspecies from a config dict, with fallback to default."""
    raw = cfg.get("simulation.nspecies", "") or ""
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return default
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ui/test_ui_helpers.py -v`
Expected: 5 passed.

- [ ] **Step 4: Ruff + commit**

```bash
git add ui/pages/_helpers.py tests/ui/test_ui_helpers.py
git commit -m "feat(ui): extract parse_nspecies to shared _helpers.py (M-9)"
```

---

### Task 4: Extract `count_map_entries` from movement.py

**Files:**
- Modify: `ui/pages/movement.py`
- Modify: `ui/pages/_helpers.py`
- Create: `tests/ui/test_ui_movement.py`

**Context:** Lines 111-117 of `movement.py` contain a pure regex-based map counting function inside `sync_n_maps_from_config()`.

- [ ] **Step 1: Write the test**

```python
"""Tests for pure helpers extracted from ui/pages/movement.py."""

from ui.pages._helpers import count_map_entries


def test_count_map_entries_one_map():
    cfg = {"movement.file.map0": "map0.csv"}
    assert count_map_entries(cfg) == 1


def test_count_map_entries_multiple():
    cfg = {
        "movement.file.map0": "map0.csv",
        "movement.file.map1": "map1.csv",
        "movement.file.map3": "map3.csv",
    }
    assert count_map_entries(cfg) == 3


def test_count_map_entries_excludes_null():
    cfg = {"movement.file.map0": "null", "movement.file.map1": "real.csv"}
    assert count_map_entries(cfg) == 1


def test_count_map_entries_excludes_empty():
    cfg = {"movement.file.map0": "", "movement.file.map1": "real.csv"}
    assert count_map_entries(cfg) == 1


def test_count_map_entries_empty_config():
    assert count_map_entries({}) == 0
```

- [ ] **Step 2: Add the helper to `_helpers.py`**

```python
import re


def count_map_entries(cfg: dict[str, str]) -> int:
    """Count non-null movement map entries in a config dict."""
    return sum(
        1
        for k, v in cfg.items()
        if re.match(r"movement\.file\.map\d+$", k)
        and isinstance(v, str)
        and v.strip()
        and v.strip().lower() not in ("null", "none")
    )
```

- [ ] **Step 3: Update movement.py to use the helper**

In `ui/pages/movement.py`, add import:
```python
from ui.pages._helpers import parse_nspecies, count_map_entries
```

Replace `sync_n_maps_from_config()` lines 111-117:

```python
    @reactive.effect
    def sync_n_maps_from_config():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        count = count_map_entries(cfg)
        if count > 0:
            ui.update_numeric("n_maps", value=count)
```

Also replace the `parse_nspecies` inline patterns at lines 54-57 and 87-91 with `parse_nspecies(cfg)` calls.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ui/test_ui_movement.py tests/ui/test_ui_helpers.py -v`
Expected: all pass.

Run: `.venv/bin/python -m pytest tests/ -q` → baseline + new tests

- [ ] **Step 5: Ruff + commit**

```bash
git add ui/pages/_helpers.py ui/pages/movement.py tests/ui/test_ui_movement.py
git commit -m "refactor(ui): extract count_map_entries from movement.py (M-9)"
```

---

### Task 5: Extract `collect_resolved_keys` from fishing.py

**Files:**
- Modify: `ui/pages/fishing.py`
- Modify: `ui/pages/_helpers.py`
- Create: `tests/ui/test_ui_fishing.py`

**Context:** Lines 86-88 and 94-97 of `fishing.py` contain identical key-resolution loops. Extract to a shared helper.

- [ ] **Step 1: Write the test**

```python
"""Tests for pure helpers used by ui/pages/fishing.py."""

from ui.pages._helpers import collect_resolved_keys


class _FakeField:
    def __init__(self, pattern):
        self._pattern = pattern

    def resolve_key(self, idx):
        return self._pattern.replace("{idx}", str(idx))


def test_collect_resolved_keys_basic():
    fields = [_FakeField("fishing.rate.fsh{idx}"), _FakeField("fishing.name.fsh{idx}")]
    result = collect_resolved_keys(fields, count=2)
    assert result == [
        "fishing.rate.fsh0", "fishing.name.fsh0",
        "fishing.rate.fsh1", "fishing.name.fsh1",
    ]


def test_collect_resolved_keys_zero_count():
    fields = [_FakeField("fishing.rate.fsh{idx}")]
    assert collect_resolved_keys(fields, count=0) == []


def test_collect_resolved_keys_start_idx():
    fields = [_FakeField("mpa.file.mpa{idx}")]
    result = collect_resolved_keys(fields, count=2, start_idx=3)
    assert result == ["mpa.file.mpa3", "mpa.file.mpa4"]
```

- [ ] **Step 2: Add the helper to `_helpers.py`**

```python
def collect_resolved_keys(fields, count: int, start_idx: int = 0) -> list[str]:
    """Resolve indexed field patterns for a range of indices.

    For each index in [start_idx, start_idx + count), resolves every field's
    key_pattern and collects all keys into a flat list.
    """
    keys: list[str] = []
    for i in range(start_idx, start_idx + count):
        keys.extend(f.resolve_key(i) for f in fields)
    return keys
```

- [ ] **Step 3: Update fishing.py**

Add import:
```python
from ui.pages._helpers import collect_resolved_keys
```

Replace `sync_fishery_inputs` body (lines 83-89):
```python
    @reactive.effect
    def sync_fishery_inputs():
        n = input.n_fisheries()
        fishery_fields = [f for f in FISHING_FIELDS if f.indexed and "fsh" in f.key_pattern]
        sync_inputs(input, state, collect_resolved_keys(fishery_fields, n))
```

Replace `sync_mpa_inputs` body (lines 91-98):
```python
    @reactive.effect
    def sync_mpa_inputs():
        n = input.n_mpas()
        mpa_fields = [f for f in FISHING_FIELDS if f.indexed and "mpa" in f.key_pattern]
        sync_inputs(input, state, collect_resolved_keys(mpa_fields, n))
```

- [ ] **Step 4: Run tests + commit**

Run: `.venv/bin/python -m pytest tests/ui/test_ui_fishing.py -v` → 3 passed
Run: `.venv/bin/python -m pytest tests/ -q` → baseline + new tests

```bash
git add ui/pages/_helpers.py ui/pages/fishing.py tests/ui/test_ui_fishing.py
git commit -m "refactor(ui): extract collect_resolved_keys from fishing.py (M-9)"
```

---

### Task 6: Extract `format_timing_rows` from diagnostics.py

**Files:**
- Modify: `ui/pages/diagnostics.py`
- Modify: `ui/pages/_helpers.py`
- Create: `tests/ui/test_ui_diagnostics.py`

**Context:** Lines 64-69 of `diagnostics.py` contain a pure data-transformation loop that builds HTML table rows from a timing dict.

- [ ] **Step 1: Write the test**

```python
"""Tests for pure helpers extracted from ui/pages/diagnostics.py."""

from ui.pages._helpers import format_timing_pairs


def test_format_timing_pairs_sorts_by_name():
    timing = {"predation": 1.5, "growth": 0.3, "movement": 2.1}
    result = format_timing_pairs(timing)
    assert result == [("growth", "0.300s"), ("movement", "2.100s"), ("predation", "1.500s")]


def test_format_timing_pairs_empty():
    assert format_timing_pairs({}) == []


def test_format_timing_pairs_single():
    result = format_timing_pairs({"init": 0.001})
    assert result == [("init", "0.001s")]
```

- [ ] **Step 2: Add the helper to `_helpers.py`**

```python
def format_timing_pairs(timing: dict[str, float]) -> list[tuple[str, str]]:
    """Sort timing dict by process name and format values as 'X.XXXs' strings.

    Returns list of (process_name, formatted_time) tuples, sorted alphabetically.
    """
    return [(name, f"{secs:.3f}s") for name, secs in sorted(timing.items())]
```

- [ ] **Step 3: Update diagnostics.py**

Add import:
```python
from ui.pages._helpers import format_timing_pairs
```

Replace lines 64-69 in `diag_timing()`:
```python
        rows = []
        for process, time_str in format_timing_pairs(timing):
            rows.append(ui.tags.tr(
                ui.tags.td(process, style="font-weight: 500;"),
                ui.tags.td(time_str),
            ))
```

- [ ] **Step 4: Run tests + commit**

Run: `.venv/bin/python -m pytest tests/ui/test_ui_diagnostics.py -v` → 3 passed
Run: `.venv/bin/python -m pytest tests/ -q` → baseline + new tests

```bash
git add ui/pages/_helpers.py ui/pages/diagnostics.py tests/ui/test_ui_diagnostics.py
git commit -m "refactor(ui): extract format_timing_pairs from diagnostics.py (M-9)"
```

---

### Task 7: Wire `parse_nspecies` into movement.py and forcing.py

**Files:**
- Modify: `ui/pages/movement.py`
- Modify: `ui/pages/forcing.py`

**Context:** After Task 3 created `parse_nspecies`, replace the inline duplicates in movement.py and forcing.py.

- [ ] **Step 1: Update movement.py**

Replace the inline nspecies parsing in `species_movement_panels()` (lines 54-57):
```python
        with reactive.isolate():
            cfg = state.config.get()
            n_species = parse_nspecies(cfg, default=3)
```

And in `sync_species_movement_inputs()` (lines 87-91):
```python
        with reactive.isolate():
            n_species = parse_nspecies(state.config.get(), default=3)
```

Import is already added in Task 4.

- [ ] **Step 2: Update forcing.py**

Add import:
```python
from ui.pages._helpers import parse_nspecies
```

Replace the inline nspecies parsing in `resource_panels()` (lines 79-82):
```python
            n_focal = parse_nspecies(cfg)
```

And in `sync_resource_inputs()` (lines 110-112):
```python
            n_focal = parse_nspecies(cfg)
```

- [ ] **Step 3: Run full suite + ruff + commit**

Run: `.venv/bin/python -m pytest tests/ -q` → unchanged count (no new tests)
Run: `.venv/bin/ruff check ui/pages/movement.py ui/pages/forcing.py`

```bash
git add ui/pages/movement.py ui/pages/forcing.py
git commit -m "refactor(ui): wire parse_nspecies into movement.py and forcing.py (M-9)"
```

---

## Final gate

- [ ] **Full suite**: `.venv/bin/python -m pytest tests/ -q` → ≥2160 passed (baseline 2148 + ~12-15 new)
- [ ] **Ruff**: `.venv/bin/ruff check osmose/ ui/ tests/`
- [ ] **Parity**: 12 passed, bit-exact
- [ ] Invoke `superpowers:finishing-a-development-branch`
