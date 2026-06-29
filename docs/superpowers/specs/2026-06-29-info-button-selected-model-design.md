# Info button: selected-model modal + visibility fix — design

> Status: design (awaiting review) · 2026-06-29
> Refine the Grid-page model-info button (shipped in `2fabefc`): when a model is selected in the
> "Model selection" dropdown, the ℹ button opens a modal dedicated to that one model; with nothing
> selected it falls back to the existing all-models overview. Also fix the button's clipped label by
> widening its column. `ui/pages/grid.py` only — no registry/engine change.

## 1. Problem

The ℹ "Info" button (`btn_example_info`) currently always shows `_model_info_modal()` — the all-models
overview — regardless of the dropdown selection. The user wants it to be **context-sensitive**: show
the selected model's details when one is chosen. Separately, the button sits in a 2/12 column
(`col_widths=[7, 2, 3]`), which clips/wraps the "ℹ Info" label.

## 2. Approach (`ui/pages/grid.py`)

### 2.1 Parameterize the modal builder
`_model_info_modal(selected: str | None = None)`:
- **`selected` is a known model** (`demo_info(selected)` truthy) → render a single block for that model
  only; modal `title=f"Model: {info['title']}"` (e.g. "Model: Baltic Sea").
- **`selected` falsy or unknown** → render all models (iterate `list_demos()`), `title="Available
  models"` — byte-for-byte today's behaviour.
- Same block layout (`title h5` · facts row `region · species · resources · engine` · summary) and the
  same `ui.modal(..., easy_close=True, size="l", footer=ui.modal_button("Close"))`. Factor the
  per-model block into a tiny local helper to avoid duplication between the two paths.

### 2.2 Handler reads the dropdown
`handle_example_info` (grid.py:849-851): `ui.modal_show(_model_info_modal(input.load_example()))`.
(`input.load_example()` is the dropdown's selected key — `""` when unselected, a model name otherwise.)

### 2.3 Button visibility fix
- Rebalance the loader row from `col_widths=[7, 2, 3]` to **`col_widths=[6, 3, 3]`** (select 6 · info 3 ·
  Load 3) so the "ℹ Info" label has room to render fully.
- Add `text-nowrap` to the info button's class (`"btn-outline-secondary w-100 text-nowrap"`) so the
  label never wraps. Keep the "ℹ Info" text (not icon-only).

## 3. Data flow

Click `btn_example_info` → handler reads `input.load_example()` → `_model_info_modal(selected)` →
single-model block if `selected` resolves, else the all-models overview → `ui.modal_show`. The dropdown,
the registry (`DEMO_INFO`/`demo_info`), and the Load flow are unchanged.

## 4. Edge cases

- **Unknown/stale `selected`** (a key not in `DEMO_INFO`) → `demo_info()` returns `None` → falls back to
  the all-models overview (defensive; never an empty modal).
- **`""` (placeholder selected)** → all-models overview (the confirmed no-selection behaviour).
- The all-models path is unchanged, so the existing modal/rename tests stay valid.

## 5. Testing (`tests/test_ui_grid.py`, append)

- `_model_info_modal("baltic")` → HTML contains "Model: Baltic Sea" and "Baltic Sea", and does **not**
  contain another model's title ("Bay of Biscay") — i.e. single-model.
- `_model_info_modal(None)` → contains all 5 titles + "Available models" (overview unchanged).
- `_model_info_modal("bogus")` → falls back: contains "Available models" + all titles (defensive).
- The existing `test_model_info_modal_lists_all_models_with_engine_facts` and
  `test_grid_loader_rename_and_labels` stay green (the latter only checks `btn_example_info` is present,
  not `col_widths`); a light assertion that `str(grid_ui())` still contains `btn_example_info` after the
  layout change. `import app` + ruff.

## 6. Out of scope

- Auto-updating the modal live as the dropdown changes (it's shown on button click, by design).
- Icon-only button / further restyling beyond the column rebalance + `text-nowrap`.
- Any change to `DEMO_INFO`, the Load flow, or the dropdown labels.
