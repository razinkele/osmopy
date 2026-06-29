# Loader: "Model selection" rename + per-model info modal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Grid-page example loader to "Model selection" / "— Select model —", and add an ℹ button next to Load that opens a modal listing all available models with their key facts, backed by a `DEMO_INFO` registry in `osmose/demo.py` (which also tidies the dropdown labels).

**Architecture:** A static metadata dict `DEMO_INFO` (keyed by `list_demos()` names) + a `demo_info()` accessor in the library; `ui/pages/grid.py` consumes it for the dropdown labels and a pure `_model_info_modal()` builder shown via `ui.modal_show` on the info button's reactive event. UI + library only.

**Tech Stack:** Python 3.12, Shiny for Python (`ui.input_select`, `ui.input_action_button`, `ui.modal`/`modal_show`/`modal_button`, `layout_columns`), pytest.

## Global Constraints

- **Run python via the MAIN venv ABSOLUTE path** `/home/razinka/osmose/osmose-python/.venv/bin/python` — do NOT create/symlink a `.venv` in the worktree. Tests: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest <path> -v`. Lint: `/home/razinka/osmose/osmose-python/.venv/bin/ruff check` + `ruff format --check`.
- `DEMO_INFO` keys MUST exactly cover `list_demos()` = `["baltic", "bay_of_biscay", "eec", "eec_full", "minimal"]`; each entry has all six fields (`title, region, species, resources, engine, summary`) non-empty. Engine = "Python only" for `baltic` + `eec_full` (they declare `simulation.nbackground > 0`, which Java can't load); "Java + Python" for the other three.
- The dropdown-choice builder is **defensive**: `(demo_info(d) or {}).get("title") or d.replace("_", " ").title()` (falls back to the auto-title if a key is missing).
- Select label = `"Model selection"`; placeholder = `"— Select model —"`.
- Info button id `btn_example_info`, left of Load; row `col_widths=[7, 2, 3]` (select · info · Load).
- `_model_info_modal()` is pure (no reactivity) and lists ALL models from `DEMO_INFO`.
- No engine-dynamics change; the Load flow (`handle_load_example`) is unchanged.
- Spec: `docs/superpowers/specs/2026-06-29-model-selection-info-design.md`.

---

### Task 1: `DEMO_INFO` registry + `demo_info()` accessor

**Files:**
- Modify: `osmose/demo.py` (add after `list_demos()`, ~line 105)
- Test: `tests/test_demo.py` (create or extend)

**Interfaces:**
- Produces: `DEMO_INFO: dict[str, dict[str, str]]`; `demo_info(name: str) -> dict[str, str] | None`.

- [ ] **Step 1: Write the failing tests** — `tests/test_demo.py` ALREADY EXISTS, so **append** these
  (do not overwrite):

```python
from osmose.demo import DEMO_INFO, demo_info, list_demos

_REQUIRED = ("title", "region", "species", "resources", "engine", "summary")


def test_demo_info_covers_all_demos_with_full_fields():
    for name in list_demos():
        assert name in DEMO_INFO, f"DEMO_INFO missing {name}"
        entry = DEMO_INFO[name]
        for field in _REQUIRED:
            assert entry.get(field), f"{name}.{field} empty"


def test_demo_info_accessor_and_engine_facts():
    assert demo_info("baltic")["engine"] == "Python only"
    assert demo_info("eec_full")["engine"] == "Python only"
    assert demo_info("bay_of_biscay")["engine"].startswith("Java")
    assert demo_info("eec")["title"] == "Eastern English Channel"
    assert demo_info("unknown") is None
```

- [ ] **Step 2: Run — verify it fails**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_demo.py -q`
Expected: FAIL (`cannot import name 'DEMO_INFO'`).

- [ ] **Step 3: Implement** — in `osmose/demo.py`, immediately after `list_demos()` (~line 105):

```python
# Per-model metadata for the UI model picker (title shown in the dropdown; the rest in the
# info modal). Keys MUST match list_demos(). Engine "Python only" = declares background species
# (simulation.nbackground > 0), which the bundled Java engine cannot load.
DEMO_INFO: dict[str, dict[str, str]] = {
    "bay_of_biscay": {
        "title": "Bay of Biscay",
        "region": "NE Atlantic (Bay of Biscay)",
        "species": "8 focal species",
        "resources": "6 LTL/plankton groups",
        "engine": "Java + Python",
        "summary": "The OSMOSE reference example (anchovy, sardine, hake, …); runs on both engines.",
    },
    "eec": {
        "title": "Eastern English Channel",
        "region": "English Channel (reduced)",
        "species": "6 focal species",
        "resources": "no LTL resources",
        "engine": "Java + Python",
        "summary": "A reduced Eastern English Channel configuration; quick to run.",
    },
    "eec_full": {
        "title": "Eastern English Channel (full)",
        "region": "English Channel",
        "species": "14 focal species",
        "resources": "10 LTL + 1 background group",
        "engine": "Python only",
        "summary": "The full 14-species EEC — the cross-engine parity benchmark; uses a background "
        "species, so it runs on the Python engine only.",
    },
    "baltic": {
        "title": "Baltic Sea",
        "region": "Central/Eastern Baltic",
        "species": "8 focal species",
        "resources": "6 LTL + 2 background groups",
        "engine": "Python only",
        "summary": "Cod, herring, sprat, flounder, perch, pike-perch, smelt, stickleback; uses "
        "background species + LTL forcing, so it runs on the Python engine only.",
    },
    "minimal": {
        "title": "Minimal",
        "region": "Toy configuration",
        "species": "2 focal species",
        "resources": "no LTL resources",
        "engine": "Java + Python",
        "summary": "A 2-species toy configuration for quick tests and smoke runs.",
    },
}


def demo_info(name: str) -> dict[str, str] | None:
    """Return the metadata dict for a demo model, or None if unknown."""
    return DEMO_INFO.get(name)
```

- [ ] **Step 4: Run — verify it passes**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_demo.py -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/demo.py tests/test_demo.py
git commit -m "feat(demo): DEMO_INFO registry + demo_info() accessor (per-model metadata)"
```

---

### Task 2: Grid-page rename + info button + modal

**Files:**
- Modify: `ui/pages/grid.py` (import, `grid_ui()` loader block, new `_model_info_modal()`, `grid_server()` handler)
- Test: `tests/test_ui_grid.py` (create or extend) + `import app`

**Interfaces:**
- Consumes: `DEMO_INFO`, `demo_info` (Task 1).
- Produces: `_model_info_modal()` (module-level, returns a `ui.modal`).

- [ ] **Step 1: Write the failing tests** — `tests/test_ui_grid.py` ALREADY EXISTS, so **append** these
  (do not overwrite):

```python
def test_model_info_modal_lists_all_models_with_engine_facts():
    from ui.pages.grid import _model_info_modal
    html = str(_model_info_modal())
    # all five model titles present
    for title in ["Bay of Biscay", "Eastern English Channel", "Eastern English Channel (full)",
                  "Baltic Sea", "Minimal"]:
        assert title in html
    # engine facts present
    assert "Python only" in html and "Java + Python" in html
    assert "Available models" in html  # modal title


def test_grid_loader_rename_and_labels():
    # The headline user-facing change: assert the rename + registry-driven labels on the rendered
    # loader (grid_ui() is a zero-arg module-level function that renders to HTML).
    from ui.pages.grid import grid_ui
    html = str(grid_ui())
    assert "Model selection" in html          # was "Example configuration"
    assert "— Select model —" in html         # was "— Select example —"
    assert "Eastern English Channel" in html  # eec registry title (auto-title "Eec" gone)
    assert "btn_example_info" in html          # the info button is present
```

- [ ] **Step 2: Run — verify it fails**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_ui_grid.py -k "model_info or rename" -q`
Expected: FAIL (the `model_info` test fails on `cannot import name '_model_info_modal'`; the `rename` test fails because the rendered loader still says "Example configuration" / "— Select example —").

- [ ] **Step 3a: Import** — in `ui/pages/grid.py` line 21, extend the demo import:
```python
from osmose.demo import demo_info, list_demos, osmose_demo
```

- [ ] **Step 3b: Add the modal-builder** at module scope in `grid.py` (e.g. just above `grid_ui()`):
```python
def _model_info_modal():
    """Modal listing all available demo models with their key facts (from DEMO_INFO)."""
    blocks = []
    for name in list_demos():
        info = demo_info(name)
        if not info:
            continue
        facts = " · ".join(
            [info["region"], info["species"], info["resources"], info["engine"]]
        )
        blocks.append(
            ui.div(
                ui.tags.h5(info["title"]),
                ui.tags.p(facts, class_="text-muted small mb-1"),
                ui.tags.p(info["summary"], class_="mb-0"),
                class_="mb-3",
            )
        )
    return ui.modal(
        *blocks,
        title="Available models",
        easy_close=True,
        size="l",
        footer=ui.modal_button("Close"),
    )
```

- [ ] **Step 3c: Rename + registry labels + info button** — in `grid_ui()`, replace the
  `demo_choices` block and the inner `layout_columns` (select + Load). Replace:
```python
    demo_choices = {
        "": "— Select example —",
        **{d: d.replace("_", " ").title() for d in list_demos()},
    }
```
with:
```python
    demo_choices = {
        "": "— Select model —",
        **{
            d: (demo_info(d) or {}).get("title") or d.replace("_", " ").title()
            for d in list_demos()
        },
    }
```
and replace the inner `ui.layout_columns(...)` (the one with the `load_example` select + the Load
button, `col_widths=[8, 4]`) with:
```python
                    ui.layout_columns(
                        ui.input_select(
                            "load_example",
                            "Model selection",
                            choices=demo_choices,
                            selected="",
                        ),
                        ui.div(
                            ui.input_action_button(
                                "btn_example_info",
                                "ℹ Info",
                                class_="btn-outline-secondary w-100",
                            ),
                            style="display: flex; align-items: flex-end; height: 100%;",
                        ),
                        ui.div(
                            ui.input_action_button(
                                "btn_load_example", "Load", class_="btn-primary w-100"
                            ),
                            style="display: flex; align-items: flex-end; height: 100%;",
                        ),
                        col_widths=[7, 2, 3],
                    ),
```

- [ ] **Step 3d: Server handler** — in `grid_server()`, next to `handle_load_example` (~line 811):
```python
    @reactive.effect
    @reactive.event(input.btn_example_info)
    def handle_example_info():
        """Show the per-model info modal."""
        ui.modal_show(_model_info_modal())
```

- [ ] **Step 4: Run — verify it passes + app imports**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_ui_grid.py -k "model_info or rename" -q`
Expected: PASS.
Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`.

- [ ] **Step 5: Lint + commit**

Run: `/home/razinka/osmose/osmose-python/.venv/bin/ruff check osmose/demo.py ui/pages/grid.py tests/test_demo.py tests/test_ui_grid.py && /home/razinka/osmose/osmose-python/.venv/bin/ruff format --check osmose/demo.py ui/pages/grid.py`
Expected: clean.
```bash
git add ui/pages/grid.py tests/test_ui_grid.py
git commit -m "feat(grid): Model-selection rename + per-model info modal (ℹ button)"
```

---

## Notes for the executor

- No engine-dynamics change — a static metadata dict + UI wiring only; the Load behaviour is untouched.
- `str(ui.modal(...))` renders the modal to HTML, so the Task-2 text assertions work without a browser.
- Do not create a `.venv` in the worktree; always use the main-venv absolute path.
- If `tests/test_demo.py` or `tests/test_ui_grid.py` already exist, append the new tests rather than overwriting.
