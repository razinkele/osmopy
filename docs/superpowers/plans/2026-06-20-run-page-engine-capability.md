# Run-Page Engine-Capability Transparency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Run page honestly communicate what the selected engine will and won't produce for the loaded config, and replace the misleading Java/Python tabs with a clear active-engine indicator + capability panel.

**Architecture:** A new pure module `osmose/engine_capabilities.py` is the single source of truth: `describe_engine(engine, config)` returns an `EngineCapability` dataclass (can-run, block-reason, populated/empty pages, notable-outputs line). The Run page (`ui/pages/run.py`) drops the read-only `run_engine_tabs` navset and its mirror observer, and adds three `@render.ui` slots driven by `state.engine_mode` / `state.config`. No change to engine/runner execution or to `app.py` (the header engine toggle stays the single writer of `engine_mode`).

**Tech Stack:** Python 3.12, Shiny for Python, pytest. No new dependencies.

---

## File Structure

- **Create:** `osmose/engine_capabilities.py` — pure capability core (dataclass + `describe_engine` + truthiness helper). No Shiny, no engine imports except `osmose.runner.java_engine_block_reason`.
- **Create:** `tests/test_engine_capabilities.py` — unit tests for the pure core.
- **Modify:** `ui/pages/run.py` — remove `run_engine_tabs` navset (~187–234) + `_sync_engine_tab` (~577–581); add `engine_settings` + `engine_indicator` + `engine_capability` render slots in `run_ui()` and `run_server()`.
- **Modify (if present):** `tests/test_ui_run.py` — a smoke test that the page imports and the new slots exist; otherwise create `tests/test_ui_run_capability.py`.

---

## Task 1: Capability dataclass + truthiness helper

**Files:**
- Create: `osmose/engine_capabilities.py`
- Test: `tests/test_engine_capabilities.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_capabilities.py
from osmose.engine_capabilities import EngineCapability, _is_enabled


def test_is_enabled_truthiness():
    assert _is_enabled({"k": "true"}, "k") is True
    assert _is_enabled({"k": "True"}, "k") is True
    assert _is_enabled({"k": "1"}, "k") is True
    assert _is_enabled({"k": "false"}, "k") is False
    assert _is_enabled({"k": ""}, "k") is False
    assert _is_enabled({}, "k") is False


def test_capability_dataclass_fields():
    cap = EngineCapability(
        engine="python",
        can_run=True,
        block_reason=None,
        pages_populated=["Results"],
        pages_empty=["Genetics"],
        notable_outputs="x",
    )
    assert cap.engine == "python"
    assert cap.can_run is True
    assert cap.pages_populated == ["Results"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_capabilities.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.engine_capabilities'`

- [ ] **Step 3: Write minimal implementation**

```python
# osmose/engine_capabilities.py
"""Pure, browser-free description of what each engine produces for a config.

Single source of truth for the Run-page capability panel. No Shiny imports;
the only engine dependency is ``java_engine_block_reason`` (already pure).
"""

from __future__ import annotations

from dataclasses import dataclass

from osmose.runner import java_engine_block_reason


@dataclass
class EngineCapability:
    engine: str  # "python" | "java"
    can_run: bool  # for THIS config
    block_reason: str | None  # why not, if can_run is False
    pages_populated: list[str]  # result/diagnostic pages that WILL have data
    pages_empty: list[str]  # pages that will NOT, for this engine+config
    notable_outputs: str  # one concise line of family-level differences


def _is_enabled(config: dict[str, str], key: str) -> bool:
    """True when a config flag reads as enabled (mirrors the engine convention)."""
    return str(config.get(key, "")).strip().lower() in ("true", "1")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_capabilities.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine_capabilities.py tests/test_engine_capabilities.py
git commit -m "feat(capabilities): EngineCapability dataclass + truthiness helper"
```

---

## Task 2: `describe_engine` for the Python engine

**Files:**
- Modify: `osmose/engine_capabilities.py`
- Test: `tests/test_engine_capabilities.py`

The canonical 4.4.0 config keys (verified in `osmose/config/aliases.py:118-120` and `config_validation.py:120`): `module.genetics.enabled`, `module.bioeconomics.enabled`, `output.spatial.enabled`. `state.config` is canonicalized on load, so these are read directly.

Page names match the real nav values in `app.py`: `Results`, `Spatial Results`, `Diagnostics`, `Genetics`, `Economic`. Community/Sheldon metrics live inside the Diagnostics page (which gates on `engine_mode=="python"` at `diagnostics.py:57`), so they are covered by `Diagnostics`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_engine_capabilities.py
from osmose.engine_capabilities import describe_engine


def test_python_base_pages_always_populate():
    cap = describe_engine("python", {})
    assert cap.engine == "python"
    assert cap.can_run is True
    assert cap.block_reason is None
    assert "Results" in cap.pages_populated
    assert "Diagnostics" in cap.pages_populated
    # disabled-by-default modules are empty
    assert "Genetics" in cap.pages_empty
    assert "Economic" in cap.pages_empty
    assert "Spatial Results" in cap.pages_empty


def test_python_genetics_gated_on_module_flag():
    cap = describe_engine("python", {"module.genetics.enabled": "true"})
    assert "Genetics" in cap.pages_populated
    assert "Genetics" not in cap.pages_empty


def test_python_economics_and_spatial_gates():
    cap = describe_engine(
        "python",
        {"module.bioeconomics.enabled": "true", "output.spatial.enabled": "true"},
    )
    assert "Economic" in cap.pages_populated
    assert "Spatial Results" in cap.pages_populated


def test_python_notable_outputs_mentions_java_only_families():
    cap = describe_engine("python", {})
    assert "sizeSpectrum" in cap.notable_outputs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_capabilities.py -k python -v`
Expected: FAIL — `ImportError: cannot import name 'describe_engine'`

- [ ] **Step 3: Write minimal implementation**

Add to `osmose/engine_capabilities.py`:

```python
# module flag -> page name, for Python conditional pages
_PYTHON_GATED_PAGES = [
    ("module.genetics.enabled", "Genetics"),
    ("module.bioeconomics.enabled", "Economic"),
    ("output.spatial.enabled", "Spatial Results"),
]

_PYTHON_NOTABLE = (
    "Not produced on the Python engine: sizeSpectrum, meanSize, meanTLByAge, "
    "yieldN, fishery-yield (run these on the Java engine)."
)
_JAVA_NOTABLE = (
    "Java run: no genetics, economics, or community size-spectrum outputs; "
    "cross-engine results are statistically equivalent, not bit-identical."
)


def _describe_python(config: dict[str, str]) -> EngineCapability:
    populated = ["Results", "Diagnostics"]
    empty: list[str] = []
    for flag, page in _PYTHON_GATED_PAGES:
        (populated if _is_enabled(config, flag) else empty).append(page)
    return EngineCapability(
        engine="python",
        can_run=True,
        block_reason=None,
        pages_populated=populated,
        pages_empty=empty,
        notable_outputs=_PYTHON_NOTABLE,
    )


def describe_engine(engine: str, config: dict[str, str]) -> EngineCapability:
    """Describe what ``engine`` will produce for ``config``. Total — never raises."""
    config = config or {}
    if engine == "python":
        return _describe_python(config)
    raise NotImplementedError  # Java handled in Task 3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_capabilities.py -k python -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine_capabilities.py tests/test_engine_capabilities.py
git commit -m "feat(capabilities): describe_engine for the Python engine"
```

---

## Task 3: `describe_engine` for the Java engine

**Files:**
- Modify: `osmose/engine_capabilities.py`
- Test: `tests/test_engine_capabilities.py`

Java populates only `Results` (the Results dropdown carries the rich Java-only families). The dedicated pages `Diagnostics`, `Genetics`, `Economic`, `Spatial Results` are all Python-gated → empty. `can_run`/`block_reason` come from `java_engine_block_reason` (non-None for `simulation.nbackground>0`, e.g. Baltic).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_engine_capabilities.py
def test_java_plain_config_runs_results_only():
    cap = describe_engine("java", {})
    assert cap.engine == "java"
    assert cap.can_run is True
    assert cap.block_reason is None
    assert cap.pages_populated == ["Results"]
    for page in ("Diagnostics", "Genetics", "Economic", "Spatial Results"):
        assert page in cap.pages_empty


def test_java_background_species_blocked():
    cap = describe_engine("java", {"simulation.nbackground": "2"})
    assert cap.can_run is False
    assert cap.block_reason is not None
    assert "background" in cap.block_reason.lower()


def test_java_notable_outputs_mentions_equivalence():
    cap = describe_engine("java", {})
    assert "statistically equivalent" in cap.notable_outputs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_capabilities.py -k java -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

In `osmose/engine_capabilities.py`, add `_describe_java` and wire it into `describe_engine`:

```python
_JAVA_EMPTY_PAGES = ["Diagnostics", "Genetics", "Economic", "Spatial Results"]


def _describe_java(config: dict[str, str]) -> EngineCapability:
    block = java_engine_block_reason(config)
    return EngineCapability(
        engine="java",
        can_run=block is None,
        block_reason=block,
        pages_populated=["Results"],
        pages_empty=list(_JAVA_EMPTY_PAGES),
        notable_outputs=_JAVA_NOTABLE,
    )
```

Replace the `raise NotImplementedError` line in `describe_engine` with:

```python
    if engine == "java":
        return _describe_java(config)
    # Unknown engine — neutral, total fallback.
    return EngineCapability(
        engine=engine,
        can_run=False,
        block_reason=f"Unknown engine: {engine!r}",
        pages_populated=[],
        pages_empty=[],
        notable_outputs="",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_capabilities.py -v`
Expected: PASS (all Task 1–3 tests)

- [ ] **Step 5: Commit**

```bash
git add osmose/engine_capabilities.py tests/test_engine_capabilities.py
git commit -m "feat(capabilities): describe_engine for the Java engine + total fallback"
```

---

## Task 4: Remove the misleading engine tabs from the Run page

**Files:**
- Modify: `ui/pages/run.py` (`run_ui` ~185–234; `_sync_engine_tab` ~577–581)

The `run_engine_tabs` `navset_tab` looks like an engine chooser but is a read-only mirror of the header toggle (`_sync_engine_tab` pushes `engine_mode` → selected tab; clicking a tab does NOT change `engine_mode`). Remove both. The per-engine inputs move into a render slot in Task 6 — for THIS task, keep the input widgets by relocating them temporarily into a static `ui.div` so existing `input.java_opts()` / `input.py_threads()` references in `handle_run` / `_run_java_engine` keep resolving and tests stay green. Task 6 replaces that static div with the dynamic slot.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_run_capability.py
import ui.pages.run as run_page


def test_run_page_has_no_engine_tabs_navset():
    # The misleading read-only navset must be gone.
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "run_engine_tabs" not in text
    assert "_sync_engine_tab" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py -v`
Expected: FAIL — `run_engine_tabs` still present in the source.

- [ ] **Step 3: Make the change**

In `run_ui()`, replace the entire `ui.navset_tab(...)` block (the call beginning `ui.navset_tab(` and ending at `id="run_engine_tabs",\n                ),`) with a static settings div that preserves all input ids:

```python
                ui.div(
                    ui.output_ui("jar_selector"),
                    ui.input_text(
                        "java_opts",
                        "Java options",
                        value="-Xmx2g",
                        placeholder="-Xmx4g -Xms1g",
                    ),
                    ui.input_numeric(
                        "run_timeout", "Timeout (seconds)", value=3600, min=60, max=86400
                    ),
                    ui.input_text_area(
                        "param_overrides",
                        "Parameter overrides (key=value, one per line)",
                        rows=4,
                    ),
                    ui.input_numeric(
                        "py_threads", "Threads (Numba prange)", value=1, min=1, max=32
                    ),
                    ui.input_select(
                        "py_verbosity",
                        "Verbosity",
                        choices={"0": "Quiet", "1": "Normal", "2": "Verbose"},
                        selected="1",
                    ),
                    ui.input_text_area(
                        "py_param_overrides",
                        "Parameter overrides (key=value, one per line)",
                        rows=4,
                    ),
                    id="run_engine_settings_static",
                ),
```

In `run_server()`, delete the entire `_sync_engine_tab` effect:

```python
    @reactive.effect
    def _sync_engine_tab():
        mode = state.engine_mode.get()
        tab = "run_java_tab" if mode == "java" else "run_python_tab"
        ui.update_navset("run_engine_tabs", selected=tab, session=session)
```

- [ ] **Step 4: Run test + import smoke**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py -v && .venv/bin/python -c "import app"`
Expected: PASS and clean import (no exception).

- [ ] **Step 5: Commit**

```bash
git add ui/pages/run.py tests/test_ui_run_capability.py
git commit -m "refactor(run): remove read-only engine tabs + mirror observer"
```

---

## Task 5: Active-engine indicator slot

**Files:**
- Modify: `ui/pages/run.py` (`run_ui` + `run_server`)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ui_run_capability.py
def test_run_page_source_has_indicator_and_capability_slots():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert 'output_ui("engine_indicator")' in text
    assert 'output_ui("engine_capability")' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py::test_run_page_source_has_indicator_and_capability_slots -v`
Expected: FAIL — slots absent.

- [ ] **Step 3: Make the change**

In `run_ui()`, immediately after `body_collapse_header("Run Configuration", "run_config"),` and before the static settings div, insert:

```python
                ui.output_ui("engine_indicator"),
```

Then add the capability slot after the static settings div and before `ui.hr()`:

```python
                ui.output_ui("engine_capability"),
```

In `run_server()`, add the indicator render function (next to the other `@render.ui` defs, e.g. just before `run_status`):

```python
    @render.ui
    def engine_indicator():
        mode = state.engine_mode.get()
        label = "Python" if mode == "python" else "Java"
        return ui.p(
            ui.tags.strong("Active engine: "),
            label,
            ui.tags.span(
                " — change in the header toggle ↗", class_="text-muted"
            ),
            class_="mb-2",
        )
```

- [ ] **Step 4: Run test + import smoke**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py -v && .venv/bin/python -c "import app"`
Expected: PASS and clean import. (`engine_capability` slot is declared in UI but its server render is added in Task 6 — Shiny tolerates an unrendered `output_ui` as empty; the import smoke confirms no error.)

- [ ] **Step 5: Commit**

```bash
git add ui/pages/run.py tests/test_ui_run_capability.py
git commit -m "feat(run): active-engine indicator slot"
```

---

## Task 6: Dynamic engine-settings slot + capability panel

**Files:**
- Modify: `ui/pages/run.py` (`run_ui` + `run_server`)

Replace the static settings div with a dynamic slot that renders ONLY the active engine's inputs, and render the capability panel from `describe_engine`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ui_run_capability.py
def test_run_page_has_dynamic_settings_slot_and_imports_capabilities():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert 'output_ui("engine_settings")' in text
    assert "run_engine_settings_static" not in text
    assert "from osmose.engine_capabilities import" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py::test_run_page_has_dynamic_settings_slot_and_imports_capabilities -v`
Expected: FAIL — static div still present, no dynamic slot.

- [ ] **Step 3: Make the change**

Add the import near the top of `ui/pages/run.py` (with the other `osmose` imports):

```python
from osmose.engine_capabilities import describe_engine
```

In `run_ui()`, replace the entire `ui.div(..., id="run_engine_settings_static")` block with:

```python
                ui.output_ui("engine_settings"),
```

In `run_server()`, add the two render functions (next to `engine_indicator`):

```python
    @render.ui
    def engine_settings():
        if state.engine_mode.get() == "java":
            return ui.div(
                ui.output_ui("jar_selector"),
                ui.input_text(
                    "java_opts", "Java options", value="-Xmx2g", placeholder="-Xmx4g -Xms1g"
                ),
                ui.input_numeric(
                    "run_timeout", "Timeout (seconds)", value=3600, min=60, max=86400
                ),
                ui.input_text_area(
                    "param_overrides",
                    "Parameter overrides (key=value, one per line)",
                    rows=4,
                ),
            )
        return ui.div(
            ui.input_numeric("py_threads", "Threads (Numba prange)", value=1, min=1, max=32),
            ui.input_select(
                "py_verbosity",
                "Verbosity",
                choices={"0": "Quiet", "1": "Normal", "2": "Verbose"},
                selected="1",
            ),
            ui.input_text_area(
                "py_param_overrides",
                "Parameter overrides (key=value, one per line)",
                rows=4,
            ),
        )

    @render.ui
    def engine_capability():
        config = state.config.get()
        if not config:
            return ui.p("Load a configuration to see engine capabilities.", class_="text-muted")
        cap = describe_engine(state.engine_mode.get(), config)
        if not cap.can_run:
            return ui.div(
                ui.tags.strong("This engine can't run this configuration. "),
                cap.block_reason or "",
                class_="alert alert-warning",
            )
        populated = ", ".join(cap.pages_populated) or "—"
        empty = ", ".join(cap.pages_empty) or "—"
        return ui.div(
            ui.p(ui.tags.strong("Will populate: "), populated),
            ui.p(ui.tags.strong("Won't populate (this engine): "), empty, class_="text-muted"),
            ui.p(cap.notable_outputs, class_="small text-muted"),
        )
```

- [ ] **Step 4: Run test + full import smoke**

Run: `.venv/bin/python -m pytest tests/test_ui_run_capability.py -v && .venv/bin/python -c "import app"`
Expected: PASS and clean import.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/run.py tests/test_ui_run_capability.py
git commit -m "feat(run): dynamic engine-settings slot + capability panel"
```

---

## Task 7: Full suite, lint, format, pyright

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q -n auto`
Expected: all pass (existing Run-page tests + new capability/UI tests).

- [ ] **Step 2: Lint + format check (matches CI "lint" job — BOTH commands)**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/`
Expected: no errors. If format fails, run `.venv/bin/ruff format osmose/ ui/ tests/` and re-commit.

- [ ] **Step 3: Pyright against the clean dev venv**

Run: `.venv/bin/python -m pyright --pythonpath .venv/bin/python osmose/engine_capabilities.py ui/pages/run.py`
Expected: 0 errors. (Per the CI-pyright-reproduction gotcha, do not run bare `pyright` — it may pick up a pandas-blind sibling venv.)

- [ ] **Step 4: Commit any lint/format fixups**

```bash
git add -A
git commit -m "chore(run): lint/format/pyright fixups" || echo "nothing to commit"
```

---

## Self-Review notes (already applied)

- **Spec coverage:** Component 1 (pure core) → Tasks 1–3; Component 2 (Run-page changes: remove navset+observer, add indicator/settings/capability slots) → Tasks 4–6.
- **Type consistency:** `EngineCapability` field names (`pages_populated`, `pages_empty`, `notable_outputs`, `block_reason`, `can_run`) are identical across the dataclass def (Task 1), `describe_engine` (Tasks 2–3), and the render slot (Task 6). Page-name strings (`Results`, `Diagnostics`, `Genetics`, `Economic`, `Spatial Results`) match `app.py` nav values and are used identically in tests and impl.
- **No `app.py` change** → no nav / visual-baseline change (the header engine toggle is untouched).
- **Out of scope (per spec):** Python progress streaming, timeout-vs-failed label, post-run View-Results, run-history `duration_sec`, engine-filtered Results dropdown.
