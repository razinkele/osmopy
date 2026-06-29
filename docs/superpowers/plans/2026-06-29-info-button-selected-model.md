# Info button: selected-model modal + visibility fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Grid-page ℹ info button open a modal dedicated to the model selected in the "Model selection" dropdown (falling back to the all-models overview when none is selected), and fix the button's clipped "ℹ Info" label.

**Architecture:** Parameterize the existing `_model_info_modal()` to take an optional `selected` model name; the handler passes `input.load_example()`. Widen the info button's column. `ui/pages/grid.py` only.

**Tech Stack:** Python 3.12, Shiny for Python (`ui.modal`/`modal_show`/`modal_button`, `layout_columns`, `input_action_button`), pytest.

## Global Constraints

- **Run python via the MAIN venv ABSOLUTE path** `/home/razinka/osmose/osmose-python/.venv/bin/python` — do NOT create/symlink a `.venv` in the worktree. Tests: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest <path> -v`. Lint: `/home/razinka/osmose/osmose-python/.venv/bin/ruff check` + `ruff format --check`.
- `_model_info_modal(selected: str | None = None)`: known `selected` (`demo_info(selected)` truthy) → single-model modal, `title=f"Model: {info['title']}"`; falsy/unknown → all-models overview, `title="Available models"` (unchanged behaviour). The no-arg call `_model_info_modal()` MUST still produce the overview (so existing tests pass).
- Single-model block layout identical to the overview block (title h5 · facts row `region · species · resources · engine` · summary). Same `ui.modal(..., easy_close=True, size="l", footer=ui.modal_button("Close"))`.
- Handler: `ui.modal_show(_model_info_modal(input.load_example()))`.
- Visibility fix: loader row `col_widths=[7, 2, 3]` → `[6, 3, 3]`; info button class `"btn-outline-secondary w-100"` → `"btn-outline-secondary w-100 text-nowrap"`. Keep the "ℹ Info" label.
- No change to `DEMO_INFO`/`demo_info`, the dropdown labels, or the Load flow (`handle_load_example`).
- Spec: `docs/superpowers/specs/2026-06-29-info-button-selected-model-design.md`.

---

### Task 1: Context-sensitive info modal + button visibility

**Files:**
- Modify: `ui/pages/grid.py` (`_model_info_modal`, the loader button block, `handle_example_info`)
- Test: `tests/test_ui_grid.py` (append 3 tests)

**Interfaces:**
- Produces: `_model_info_modal(selected: str | None = None)` (was zero-arg).

- [ ] **Step 1: Write the failing tests** — `tests/test_ui_grid.py` ALREADY EXISTS; **append** (do not overwrite):

```python
def test_model_info_modal_single_model_when_selected():
    from ui.pages.grid import _model_info_modal
    html = str(_model_info_modal("baltic"))
    assert "Model: Baltic Sea" in html  # single-model modal title
    assert "Baltic Sea" in html
    assert "Bay of Biscay" not in html  # NOT the all-models overview


def test_model_info_modal_none_shows_overview():
    from ui.pages.grid import _model_info_modal
    html = str(_model_info_modal(None))
    assert "Available models" in html
    for title in ["Bay of Biscay", "Baltic Sea", "Minimal"]:
        assert title in html


def test_model_info_modal_unknown_falls_back_to_overview():
    from ui.pages.grid import _model_info_modal
    html = str(_model_info_modal("bogus"))
    assert "Available models" in html
    assert "Baltic Sea" in html
```

- [ ] **Step 2: Run — verify it fails**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_ui_grid.py -k "single_model or overview or fall" -v`
Expected: FAIL — `_model_info_modal()` takes no argument (`TypeError: ... takes 0 positional arguments but 1 was given`).

- [ ] **Step 3a: Parameterize the modal builder** — replace the whole `_model_info_modal()` function (grid.py ~75-95) with:
```python
def _model_info_modal(selected: str | None = None):
    """Per-model info modal. With a known ``selected`` model, show only that model; otherwise
    list all available models (the overview)."""

    def _block(info):
        facts = " · ".join([info["region"], info["species"], info["resources"], info["engine"]])
        return ui.div(
            ui.tags.h5(info["title"]),
            ui.tags.p(facts, class_="text-muted small mb-1"),
            ui.tags.p(info["summary"], class_="mb-0"),
            class_="mb-3",
        )

    info = demo_info(selected) if selected else None
    if info is not None:
        return ui.modal(
            _block(info),
            title=f"Model: {info['title']}",
            easy_close=True,
            size="l",
            footer=ui.modal_button("Close"),
        )
    blocks = [_block(demo_info(name)) for name in list_demos() if demo_info(name)]
    return ui.modal(
        *blocks,
        title="Available models",
        easy_close=True,
        size="l",
        footer=ui.modal_button("Close"),
    )
```

- [ ] **Step 3b: Handler reads the dropdown** — in `handle_example_info` (grid.py ~849-851), replace the body line:
```python
        ui.modal_show(_model_info_modal())
```
with:
```python
        ui.modal_show(_model_info_modal(input.load_example()))
```

- [ ] **Step 3c: Button visibility** — in the loader `ui.layout_columns(...)`, change the info button class and the `col_widths`:
  - `"btn-outline-secondary w-100"` → `"btn-outline-secondary w-100 text-nowrap"` (the `btn_example_info` button).
  - `col_widths=[7, 2, 3]` → `col_widths=[6, 3, 3]`.

- [ ] **Step 4: Run — verify new + existing tests pass**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_ui_grid.py -q`
Expected: PASS — the 3 new tests + the existing `test_model_info_modal_lists_all_models_with_engine_facts` (still calls `_model_info_modal()` → overview via the `selected=None` default) and `test_grid_loader_rename_and_labels` (still finds `btn_example_info`).

- [ ] **Step 5: Verify app imports + lint**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`.
Run: `/home/razinka/osmose/osmose-python/.venv/bin/ruff check ui/pages/grid.py tests/test_ui_grid.py && /home/razinka/osmose/osmose-python/.venv/bin/ruff format --check ui/pages/grid.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/grid.py tests/test_ui_grid.py
git commit -m "feat(grid): info button opens the selected model's modal + fix clipped label"
```

---

## Notes for the executor

- No engine-dynamics change — UI wiring + a layout tweak only; `DEMO_INFO`, the dropdown labels, and the Load flow are untouched.
- `str(ui.modal(...))` renders to HTML, so the text assertions work without a browser.
- The single-model test asserts "Bay of Biscay" is ABSENT — this proves it's the single-model modal, not the overview. (Don't pick a `selected` whose title is a substring of another model's title; "baltic" → "Baltic Sea" is unique.)
- Do not create a `.venv` in the worktree; always use the main-venv absolute path.
