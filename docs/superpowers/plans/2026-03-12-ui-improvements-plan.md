# UI Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve OSMOSE UI usability with 7 targeted enhancements: persistent config header, spreadsheet species/LTL tables, spatial overlay selector, smart results loading, populated species filters, and layered tooltips.

**Architecture:** Each improvement is a self-contained task modifying 1-3 files. The shared `AppState` class gains new reactive values (`config_name`, `species_names`, `results_loaded`). The spreadsheet table is a new `render_species_table()` function in `param_form.py` reused by both Setup and Forcing pages. Tooltips are layered: `(?)` hover popovers for quick reference + global "Show Help" toggle for inline descriptions.

**Tech Stack:** Shiny for Python, Bootstrap 5 popovers, deck.gl/shiny-deckgl, xarray, Plotly

---

## Chunk 1: State + Header + Example Loading

### Task 1: Add new reactive state fields

**Files:**
- Modify: `ui/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write tests for new state fields**

Add to `tests/test_state.py`:

```python
def test_appstate_has_config_name():
    state = AppState()
    with reactive.isolate():
        assert state.config_name.get() == ""


def test_appstate_has_species_names():
    state = AppState()
    with reactive.isolate():
        assert state.species_names.get() == []


def test_appstate_has_results_loaded():
    state = AppState()
    with reactive.isolate():
        assert state.results_loaded.get() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_state.py::test_appstate_has_config_name tests/test_state.py::test_appstate_has_species_names tests/test_state.py::test_appstate_has_results_loaded -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add the new fields to AppState**

In `ui/state.py`, add inside `__init__` after `self.load_trigger`:

```python
self.config_name: reactive.Value[str] = reactive.Value("")
self.species_names: reactive.Value[list[str]] = reactive.Value([])
self.results_loaded: reactive.Value[bool] = reactive.Value(False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ui/state.py tests/test_state.py
git commit -m "feat: add config_name, species_names, results_loaded to AppState"
```

---

### Task 2: Replace dirty_banner with persistent config header

**Files:**
- Modify: `app.py`
- Modify: `ui/styles.py`
- Modify: `www/osmose.css`

- [ ] **Step 1: Add header bar style constant to `ui/styles.py`**

```python
# Config header bar
STYLE_CONFIG_HEADER = (
    "display: flex; justify-content: space-between; align-items: center; "
    "padding: 8px 16px; border-bottom: 1px solid var(--osm-border, #2d3d50);"
)
```

- [ ] **Step 2: Replace `dirty_banner` with `config_header` in `app.py`**

In `app_ui`, replace `ui.output_ui("dirty_banner"),` with `ui.output_ui("config_header"),`.

In the `server()` function, **delete** the `dirty_banner` render function (lines 181-184) and replace it with:

```python
@render.ui
def config_header():
    name = state.config_name.get()
    if not name:
        return ui.div()
    cfg = state.config.get()
    n_species = int(cfg.get("simulation.nspecies", "0"))
    n_params = len(cfg)
    is_dirty = state.dirty.get()
    return ui.div(
        ui.div(
            ui.tags.span(name, style="color: #d4a017; font-weight: 600;"),
            ui.tags.span(
                f" {n_species} species \u2022 {n_params} parameters",
                style="color: #5a6a7a; font-size: 12px; margin-left: 8px;",
            ),
        ),
        ui.tags.span(
            "modified" if is_dirty else "",
            style="color: #e67e22; font-size: 11px; font-style: italic;",
        ),
        style=STYLE_CONFIG_HEADER,
    )
```

Add the import `from ui.styles import STYLE_CONFIG_HEADER` at the top of `app.py` (add to existing imports from `ui.styles` if any, otherwise add new import).

- [ ] **Step 3: Write test for config header rendering**

Add to `tests/test_state.py`:

```python
def test_config_header_shows_name_and_count():
    """Config header should display config name and param count."""
    state = AppState()
    with reactive.isolate():
        state.config_name.set("Eec Full")
        state.config.set({"a": "1", "b": "2"})
        assert state.config_name.get() == "Eec Full"
        assert len(state.config.get()) == 2


def test_config_header_empty_when_no_config():
    """Config header should be empty when no config loaded."""
    state = AppState()
    with reactive.isolate():
        assert state.config_name.get() == ""
```

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add app.py ui/styles.py tests/test_state.py
git commit -m "feat: replace dirty_banner with persistent config header bar"
```

---

### Task 3: Rework example loading with Load button + config_name

**Files:**
- Modify: `ui/pages/setup.py`

- [ ] **Step 1: Update `setup_ui()` — add Load button, keep dropdown**

Replace the example select div in `setup_ui()`:

```python
ui.div(
    ui.layout_columns(
        ui.input_select(
            "load_example",
            "Example configuration",
            choices=demo_choices,
            selected="",
        ),
        ui.input_action_button(
            "btn_load_example", "Load", class_="btn-primary mt-4"
        ),
        col_widths=[8, 4],
    ),
),
```

- [ ] **Step 2: Update `handle_load_example` to use action button + set config_name**

Change the handler to be triggered by the button instead of dropdown change:

```python
@reactive.effect
@reactive.event(input.btn_load_example)
def handle_load_example():
    """Load a bundled example config when Load button is clicked."""
    import tempfile

    from osmose.config.reader import OsmoseConfigReader

    example = input.load_example()
    if not example:
        ui.notification_show("Select an example first.", type="warning", duration=3)
        return

    try:
        tmp = Path(tempfile.mkdtemp(prefix="osmose_demo_"))
        result = osmose_demo(example, tmp)
    except ValueError as exc:
        ui.notification_show(str(exc), type="error", duration=5)
        return

    master = result["config_file"]
    if not master.exists():
        ui.notification_show(f"Example not found: {master}", type="error", duration=5)
        return

    config_dir = master.parent

    state.loading.set(True)
    try:
        reader = OsmoseConfigReader()
        cfg = migrate_config(reader.read(master))
        state.config.set(cfg)
        state.config_dir.set(config_dir)
        state.config_name.set(example.replace("_", " ").title())

        # Extract species names
        n_species = int(cfg.get("simulation.nspecies", "0"))
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n_species)]
        state.species_names.set(names)

        ui.update_numeric("n_species", value=n_species)

        with reactive.isolate():
            state.load_trigger.set(state.load_trigger.get() + 1)

        ui.notification_show(
            f"Loaded '{example}' ({len(cfg)} parameters).",
            type="message",
            duration=3,
        )
        state.dirty.set(False)
        # Do NOT reset dropdown — keep selection visible
    finally:
        state.loading.set(False)
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 4: Manual verification — launch app**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`
Verify: Select "Eec Full", click "Load". Header bar shows "Eec Full | 8 species | N parameters". Edit a value — "modified" appears.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/setup.py
git commit -m "feat: rework example loading with Load button and config header"
```

---

## Chunk 2: Spreadsheet Species Table

### Task 4: Build `render_species_table()` function

**Files:**
- Modify: `ui/components/param_form.py`
- Test: `tests/test_param_form.py`

- [ ] **Step 1: Write tests for render_species_table**

Add to `tests/test_param_form.py`:

```python
from ui.components.param_form import render_species_table


def test_render_species_table_zero_species():
    """0 species shows placeholder message."""
    from osmose.schema.species import SPECIES_FIELDS
    result = render_species_table(SPECIES_FIELDS, n_species=0, species_names=[])
    html = str(result)
    assert "Load a configuration" in html


def test_render_species_table_one_species():
    """1 species renders table with header + data column."""
    from osmose.schema.species import SPECIES_FIELDS
    result = render_species_table(
        SPECIES_FIELDS, n_species=1, species_names=["Anchovy"],
    )
    html = str(result)
    assert "Anchovy" in html
    assert "Growth" in html  # Category header


def test_render_species_table_multiple_species():
    """Multiple species render as columns."""
    from osmose.schema.species import SPECIES_FIELDS
    result = render_species_table(
        SPECIES_FIELDS, n_species=3,
        species_names=["Anchovy", "Sardine", "Hake"],
    )
    html = str(result)
    assert "Anchovy" in html
    assert "Sardine" in html
    assert "Hake" in html


def test_render_species_table_hides_advanced():
    """Advanced fields hidden by default."""
    from osmose.schema.species import SPECIES_FIELDS
    result_basic = render_species_table(
        SPECIES_FIELDS, n_species=1, species_names=["A"],
        show_advanced=False,
    )
    result_adv = render_species_table(
        SPECIES_FIELDS, n_species=1, species_names=["A"],
        show_advanced=True,
    )
    assert len(str(result_adv)) > len(str(result_basic))


def test_render_species_table_uses_spt_prefix():
    """Input IDs use spt_ prefix to avoid collision."""
    from osmose.schema.species import SPECIES_FIELDS
    result = render_species_table(
        SPECIES_FIELDS, n_species=1, species_names=["A"],
    )
    html = str(result)
    assert "spt_" in html


def test_render_species_table_with_start_idx():
    """start_idx offsets species indexing for LTL resources."""
    from osmose.schema.ltl import LTL_FIELDS
    indexed = [f for f in LTL_FIELDS if f.indexed]
    result = render_species_table(
        indexed, n_species=2, species_names=["Phyto", "Zoo"],
        start_idx=8,
    )
    html = str(result)
    # Should use sp8, sp9 in input IDs
    assert "spt_" in html
    assert "Phyto" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_param_form.py::test_render_species_table_zero_species -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `render_species_table()`**

Add to `ui/components/param_form.py`:

```python
def render_species_table(
    fields: list[OsmoseField],
    n_species: int,
    species_names: list[str],
    start_idx: int = 0,
    show_advanced: bool = False,
    config: dict[str, str] | None = None,
) -> ui.Tag:
    """Render a spreadsheet-style table: params as rows, species as columns.

    Args:
        fields: Schema fields to display (must be indexed).
        n_species: Number of species columns.
        species_names: Display names for each species.
        start_idx: Starting species index (for LTL resources offset by nspecies).
        show_advanced: Whether to include advanced fields.
        config: Optional config dict for initial values.
    """
    if n_species == 0:
        return ui.div(
            "Load a configuration to view species parameters.",
            style="padding: 20px; text-align: center; color: #5a6a7a;",
        )

    # Filter to indexed, non-advanced fields
    visible = [f for f in fields if f.indexed and (show_advanced or not f.advanced)]

    # Group by category
    categories: dict[str, list[OsmoseField]] = {}
    for f in visible:
        cat = f.category or "other"
        categories.setdefault(cat, []).append(f)

    # Build header row: Parameter | Species0 | Species1 | ...
    header_cells = [ui.tags.th("Parameter", style="position: sticky; left: 0; z-index: 2; background: var(--osm-bg-card, #162232); min-width: 200px; padding: 8px 12px;")]
    for i, name in enumerate(species_names):
        header_cells.append(ui.tags.th(name, style="text-align: center; min-width: 90px; padding: 8px;"))
    header = ui.tags.thead(ui.tags.tr(*header_cells, style="border-bottom: 2px solid var(--osm-border, #2d3d50);"))

    # Build body rows grouped by category
    rows = []
    for cat_name, cat_fields in categories.items():
        display_cat = cat_name.replace("_", " ").title()
        n_fields = len(cat_fields)
        # Category group header row (collapsible via JS)
        cat_id = f"spt_cat_{cat_name}"
        rows.append(
            ui.tags.tr(
                ui.tags.td(
                    ui.tags.span(
                        f"\u25bc {display_cat} ",
                        ui.tags.span(f"({n_fields} params)", style="color: #5a6a7a; font-weight: 400; font-size: 10px;"),
                        style="cursor: pointer;",
                    ),
                    colspan=str(n_species + 1),
                    style="padding: 6px 12px; font-weight: 700; color: #d4a017; background: var(--osm-bg-section, #1a2a3a);",
                ),
                **{"data-spt-cat": cat_id, "onclick": f"toggleSptCategory('{cat_id}')"},
                style="cursor: pointer;",
            )
        )

        # Parameter rows
        for field in cat_fields:
            label = field.description or field.key_pattern
            unit_text = f" ({field.unit})" if field.unit else ""
            param_cell = ui.tags.td(
                ui.tags.span(label),
                ui.tags.span(unit_text, style="color: #5a6a7a;"),
                style="padding: 5px 12px; position: sticky; left: 0; z-index: 1; background: var(--osm-bg-card, #0f1923);",
            )

            value_cells = []
            for i in range(n_species):
                sp_idx = start_idx + i
                config_key = field.resolve_key(sp_idx)
                # Input ID: spt_{key_without_sp_idx}_{species_idx}
                base_key = field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
                input_id = f"spt_{base_key}_{sp_idx}"

                # Resolve value from config or default
                val = field.default
                if config and config_key in config:
                    raw = config[config_key]
                    if field.param_type in (ParamType.FLOAT, ParamType.INT):
                        try:
                            val = float(raw) if field.param_type == ParamType.FLOAT else int(raw)
                        except (ValueError, TypeError):
                            val = field.default
                    elif field.param_type == ParamType.BOOL:
                        val = str(raw).lower() in ("true", "1", "yes")
                    else:
                        val = raw

                cell_style = "text-align: center; padding: 4px;"
                if field.param_type in (ParamType.FLOAT, ParamType.INT):
                    widget = ui.input_numeric(
                        input_id, "",
                        value=val if val is not None else 0,
                        min=field.min_val, max=field.max_val,
                        step=_guess_step(field) if field.param_type == ParamType.FLOAT else 1,
                        width="90px",
                    )
                elif field.param_type == ParamType.BOOL:
                    widget = ui.input_switch(input_id, "", value=bool(val) if val is not None else False)
                elif field.param_type == ParamType.ENUM:
                    choices = {c: c for c in (field.choices or [])}
                    widget = ui.input_select(input_id, "", choices=choices, selected=val, width="90px")
                elif field.param_type in (ParamType.FILE_PATH, ParamType.MATRIX):
                    widget = ui.tags.span("file", style="color: #5a6a7a; font-size: 11px;")
                else:
                    widget = ui.input_text(input_id, "", value=str(val or ""), width="90px")

                value_cells.append(ui.tags.td(widget, style=cell_style))

            rows.append(
                ui.tags.tr(param_cell, *value_cells, **{"data-spt-group": cat_id},
                    style="border-bottom: 1px solid var(--osm-border-dim, #1a2a3a);",
                )
            )

    body = ui.tags.tbody(*rows)
    table = ui.tags.table(
        header, body,
        class_="table table-sm",
        style="width: 100%; border-collapse: collapse; font-size: 12px;",
    )

    # Client-side JS for collapsing categories (no server round-trip)
    collapse_js = ui.tags.script("""
    function toggleSptCategory(catId) {
        var rows = document.querySelectorAll('[data-spt-group="' + catId + '"]');
        var header = document.querySelector('[data-spt-cat="' + catId + '"]');
        var visible = rows.length > 0 && rows[0].style.display !== 'none';
        rows.forEach(function(r) { r.style.display = visible ? 'none' : ''; });
        var span = header.querySelector('span');
        if (span) {
            var text = span.textContent;
            span.textContent = visible ? text.replace('\u25bc', '\u25b6') : text.replace('\u25b6', '\u25bc');
        }
    }
    """)

    return ui.div(
        ui.div(table, style="max-height: 600px; overflow: auto;"),
        collapse_js,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_param_form.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ui/components/param_form.py tests/test_param_form.py
git commit -m "feat: add render_species_table() spreadsheet component"
```

---

### Task 5: Wire spreadsheet table into Setup page

**Files:**
- Modify: `ui/pages/setup.py`

- [ ] **Step 1: Update imports and layout in setup_ui()**

Update the import in `ui/pages/setup.py`:

```python
from ui.components.param_form import render_category, render_species_table
```

The species card layout stays the same — the change is in the render function (Step 2).

- [ ] **Step 2: Replace `species_panels` render function**

```python
@render.ui
def species_panels():
    state.load_trigger.get()
    n = input.n_species()
    show_adv = input.show_advanced_species()
    with reactive.isolate():
        cfg = state.config.get()
    names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
    return render_species_table(
        SPECIES_FIELDS, n_species=n, species_names=names,
        show_advanced=show_adv, config=cfg,
    )
```

- [ ] **Step 3: Update `sync_species_inputs` for new spt_ IDs**

Replace the existing `sync_species_inputs`:

```python
@reactive.effect
def sync_species_inputs():
    """Auto-sync species table cells to state.config."""
    with reactive.isolate():
        if state.loading.get():
            return
    n = input.n_species()
    show_adv = input.show_advanced_species()
    state.update_config("simulation.nspecies", str(n))

    visible = [f for f in SPECIES_FIELDS if f.indexed and (show_adv or not f.advanced)]
    for i in range(n):
        for field in visible:
            config_key = field.resolve_key(i)
            base_key = field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
            input_id = f"spt_{base_key}_{i}"
            try:
                val = getattr(input, input_id)()
            except (AttributeError, TypeError):
                continue
            if val is not None:
                state.update_config(config_key, str(val))
```

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 5: Manual verification — launch app, load Eec Full**

Verify: Species are shown as columns, parameters as rows. Values editable. Categories collapse/expand on click.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/setup.py
git commit -m "feat: wire spreadsheet species table into Setup page"
```

---

### Task 6: Wire spreadsheet table into Forcing page (LTL)

**Files:**
- Modify: `ui/pages/forcing.py`

- [ ] **Step 1: Update imports**

```python
from ui.components.param_form import render_field, render_species_table
```

**Note:** The existing `forcing.py` uses `species_idx=i` (0-based) for LTL resources, which is wrong — OSMOSE numbers resource species after focal species (`sp8`, `sp9`, etc. for 8 focal species). This task fixes that bug by using `start_idx=n_focal`.

- [ ] **Step 2: Replace `resource_panels` with spreadsheet table**

```python
@render.ui
def resource_panels():
    state.load_trigger.get()
    n = input.n_resources()
    with reactive.isolate():
        cfg = state.config.get()
    n_focal = int(cfg.get("simulation.nspecies", "0"))
    indexed_fields = [f for f in LTL_FIELDS if f.indexed]
    names = [cfg.get(f"species.name.sp{n_focal + i}", f"Resource {i}") for i in range(n)]
    return render_species_table(
        indexed_fields, n_species=n, species_names=names,
        start_idx=n_focal, config=cfg,
    )
```

- [ ] **Step 3: Update `sync_resource_inputs` for new spt_ IDs**

```python
@reactive.effect
def sync_resource_inputs():
    n = input.n_resources()
    with reactive.isolate():
        if state.loading.get():
            return
        cfg = state.config.get()
    n_focal = int(cfg.get("simulation.nspecies", "0"))
    indexed_fields = [f for f in LTL_FIELDS if f.indexed]
    for i in range(n):
        sp_idx = n_focal + i
        for field in indexed_fields:
            config_key = field.resolve_key(sp_idx)
            base_key = field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
            input_id = f"spt_{base_key}_{sp_idx}"
            try:
                val = getattr(input, input_id)()
            except (AttributeError, TypeError):
                continue
            if val is not None:
                state.update_config(config_key, str(val))
```

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add ui/pages/forcing.py
git commit -m "feat: wire spreadsheet LTL table into Forcing page"
```

---

## Chunk 3: Results + Species Filter + Grid Overlay

### Task 7: Smart results directory + auto-load

**Files:**
- Modify: `ui/pages/results.py`

- [ ] **Step 1: Update results_ui() default output path**

Replace the hardcoded `value="output/"`:

```python
ui.output_ui("output_dir_input"),
```

Add render function in `results_server`:

```python
@render.ui
def output_dir_input():
    with reactive.isolate():
        out = state.output_dir.get()
    default_val = str(out) if out else ""
    return ui.div(
        ui.input_text("output_dir", "Output directory", value=default_val),
        ui.output_ui("output_dir_status"),
    )
```

- [ ] **Step 2: Add output directory validation status**

```python
@render.ui
def output_dir_status():
    path_str = input.output_dir()
    if not path_str:
        return ui.div()
    p = Path(path_str)
    if not p.is_dir():
        return ui.tags.small("Directory not found", style="color: #e74c3c;")
    csvs = list(p.glob("*.csv"))
    if not csvs:
        return ui.tags.small("No results found in this directory", style="color: #e67e22;")
    return ui.tags.small(f"Found {len(csvs)} output files", style="color: #2ecc71;")
```

- [ ] **Step 3: Add auto-load effect**

```python
@reactive.effect
def _auto_load_results():
    """Auto-load results when navigating to Results tab after a run."""
    nav = input.main_nav()
    if nav != "results":
        return
    with reactive.isolate():
        out = state.output_dir.get()
        loaded = state.results_loaded.get()
    if out and not loaded and Path(str(out)).is_dir():
        ui.update_text("output_dir", value=str(out))
        # Trigger the existing load handler
        _do_load_results(Path(str(out)))
```

- [ ] **Step 4: Extract load logic into reusable function**

Refactor `_load_results` so both the button handler and auto-load can call the same logic. Extract the body of the existing `_load_results` into `_do_load_results(out_dir: Path)`. Then:

```python
@reactive.effect
@reactive.event(input.btn_load_results)
def _load_results():
    out_dir = Path(input.output_dir())
    if not out_dir.is_dir():
        ui.notification_show(f"Directory not found: {out_dir}", type="error", duration=5)
        return
    _do_load_results(out_dir)
```

At the end of `_do_load_results`, add:

```python
state.results_loaded.set(True)
```

- [ ] **Step 5: Reset results_loaded when output_dir changes**

```python
@reactive.effect
def _reset_results_loaded():
    """Reset loaded flag when output directory changes."""
    state.output_dir.get()  # take dependency
    state.results_loaded.set(False)
```

- [ ] **Step 6: Write test for results_loaded flag**

Add to `tests/test_state.py`:

```python
def test_results_loaded_flag():
    """results_loaded flag should track load state."""
    state = AppState()
    with reactive.isolate():
        assert state.results_loaded.get() is False
        state.results_loaded.set(True)
        assert state.results_loaded.get() is True
        # Reset when output_dir changes
        state.results_loaded.set(False)
        assert state.results_loaded.get() is False
```

- [ ] **Step 7: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add ui/pages/results.py tests/test_state.py
git commit -m "feat: smart results directory defaults and auto-load on tab switch"
```

---

### Task 8: Populate species filter from config

**Files:**
- Modify: `ui/pages/setup.py`
- Modify: `ui/pages/results.py`

- [ ] **Step 1: Write test for species name extraction**

Add to `tests/test_state.py`:

```python
def test_species_names_extracted_from_config():
    """species_names should be extractable from config keys."""
    state = AppState()
    with reactive.isolate():
        state.config.set({
            "simulation.nspecies": "3",
            "species.name.sp0": "Anchovy",
            "species.name.sp1": "Sardine",
            "species.name.sp2": "Hake",
        })
        cfg = state.config.get()
        n = int(cfg.get("simulation.nspecies", "0"))
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        assert names == ["Anchovy", "Sardine", "Hake"]


def test_species_names_fallback_for_missing():
    """Missing species names should fall back to 'Species N'."""
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "2"})
        cfg = state.config.get()
        n = int(cfg.get("simulation.nspecies", "0"))
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        assert names == ["Species 0", "Species 1"]


def test_species_names_zero_species():
    """Zero species should produce empty list."""
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "0"})
        cfg = state.config.get()
        n = int(cfg.get("simulation.nspecies", "0"))
        names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
        assert names == []
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: All PASS

- [ ] **Step 3: Update species_names in sync_species_inputs**

In `ui/pages/setup.py`, at the end of `sync_species_inputs`:

```python
# Update global species names list
names = []
with reactive.isolate():
    cfg = state.config.get()
for i in range(n):
    names.append(cfg.get(f"species.name.sp{i}", f"Species {i}"))
state.species_names.set(names)
```

- [ ] **Step 2: Update results species filter to use state.species_names**

In `results_server`, update the species dropdown population in `_do_load_results` to also read from `state.species_names` as fallback, and update the dropdown after loading:

```python
# Discover species from biomass data, falling back to state
species_choices: dict[str, str] = {"all": "All species"}
bio_df = data.get("biomass", pd.DataFrame())
if not bio_df.empty and "species" in bio_df.columns:
    for sp in sorted(bio_df["species"].unique()):
        species_choices[sp] = sp
elif state is not None:
    with reactive.isolate():
        for sp in state.species_names.get():
            species_choices[sp] = sp
ui.update_select("result_species", choices=species_choices)
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add ui/pages/setup.py ui/pages/results.py
git commit -m "feat: populate species filter from config species names"
```

---

### Task 9: Grid preview spatial file overlay selector

**Files:**
- Modify: `ui/pages/grid.py`

- [ ] **Step 1: Add overlay dropdown to grid_ui()**

In `grid_ui()`, add a dropdown above the map:

```python
ui.card(
    ui.card_header("Grid Preview"),
    ui.output_ui("grid_overlay_selector"),
    ui.output_ui("grid_hint"),
    grid_map.ui(height="500px"),
),
```

- [ ] **Step 2: Add overlay selector render function**

In `grid_server`, add:

```python
@render.ui
def grid_overlay_selector():
    state.load_trigger.get()
    with reactive.isolate():
        cfg = state.config.get()
        cfg_dir = state.config_dir.get()
    choices: dict[str, str] = {"grid_extent": "Grid extent"}
    # Scan config for spatial file references
    from osmose.schema.base import ParamType
    for field in state.registry.all_fields():
        if field.param_type != ParamType.FILE_PATH:
            continue
        if field.indexed:
            n_sp = int(cfg.get("simulation.nspecies", "0"))
            n_res = int(cfg.get("simulation.nresource", "0"))
            for idx in range(n_sp + n_res):
                key = field.resolve_key(idx)
                val = cfg.get(key, "")
                if val and (val.endswith(".nc") or val.endswith(".csv")):
                    sp_name = cfg.get(f"species.name.sp{idx}", f"sp{idx}")
                    label = f"{field.description}: {sp_name}"
                    choices[key] = label
        else:
            val = cfg.get(field.key_pattern, "")
            if val and (val.endswith(".nc") or val.endswith(".csv")):
                choices[field.key_pattern] = field.description or field.key_pattern
    if len(choices) <= 1:
        return ui.div()
    return ui.input_select("grid_overlay", "Overlay data", choices=choices, selected="grid_extent")
```

- [ ] **Step 3: Add helper function for loading NetCDF overlay**

Add a new function in `grid.py`:

```python
def _load_netcdf_overlay(
    file_path: Path,
    fallback_lat: np.ndarray | None = None,
    fallback_lon: np.ndarray | None = None,
) -> list[dict] | None:
    """Load a NetCDF file and return overlay cell data for deck.gl.

    Returns list of cell dicts with 'polygon' and 'value' keys, or None on failure.
    """
    try:
        ds = xr.open_dataset(file_path)
        # Find first 2D+ numeric variable
        var_name = None
        for vn in ds.data_vars:
            if len(ds[vn].dims) >= 2:
                var_name = vn
                break
        if not var_name:
            ds.close()
            return None

        data_vals = ds[var_name].values
        if len(data_vals.shape) > 2:
            data_vals = data_vals[0]  # first time step

        # Get coordinates from the overlay file or fall back to grid coords
        olat = ds["lat"].values if "lat" in ds else fallback_lat
        olon = ds["lon"].values if "lon" in ds else fallback_lon
        ds.close()

        if olat is None or olon is None:
            return None

        ony, onx = data_vals.shape
        cells = []
        for r in range(min(ony, olat.shape[0] if olat.ndim > 1 else ony)):
            for c in range(min(onx, olon.shape[1] if olon.ndim > 1 else onx)):
                v = float(data_vals[r, c])
                if np.isnan(v):
                    continue
                cell_lat = float(olat[r, c] if olat.ndim == 2 else olat[r])
                cell_lon = float(olon[r, c] if olon.ndim == 2 else olon[c])
                # Infer cell size from neighboring coordinates
                if olat.ndim == 2:
                    dlat = abs(float(olat[min(r+1, ony-1), c] - olat[max(r-1, 0), c])) / 2
                    dlon = abs(float(olon[r, min(c+1, onx-1)] - olon[r, max(c-1, 0)])) / 2
                else:
                    dlat = abs(float(olat[min(r+1, len(olat)-1)] - olat[max(r-1, 0)])) / 2
                    dlon = abs(float(olon[min(c+1, len(olon)-1)] - olon[max(c-1, 0)])) / 2
                if r == 0 or r == ony - 1:
                    dlat *= 2
                if c == 0 or c == onx - 1:
                    dlon *= 2
                hlat, hlon = dlat / 2, dlon / 2
                cells.append({
                    "polygon": [
                        [cell_lon - hlon, cell_lat + hlat],
                        [cell_lon + hlon, cell_lat + hlat],
                        [cell_lon + hlon, cell_lat - hlat],
                        [cell_lon - hlon, cell_lat - hlat],
                    ],
                    "value": v,
                })
        return cells if cells else None
    except Exception as exc:
        _log.warning("Failed to load overlay %s: %s", file_path, exc)
        return None
```

- [ ] **Step 4: Handle overlay selection in update_grid_map**

At the end of `update_grid_map`, before `await _map.update(...)`, add:

```python
# Load overlay data if selected
try:
    overlay = input.grid_overlay()
except Exception:
    overlay = "grid_extent"

if overlay and overlay != "grid_extent":
    overlay_path_str = cfg.get(overlay, "")
    if overlay_path_str and cfg_dir:
        overlay_file = (cfg_dir / overlay_path_str).resolve()
        if not overlay_file.exists():
            ui.notification_show(f"File not found: {overlay_path_str}", type="warning", duration=3)
        elif overlay_file.suffix == ".nc":
            # Pass grid coordinates as fallback for overlay files without their own coords
            fb_lat = nc_data[0] if nc_data else None
            fb_lon = nc_data[1] if nc_data else None
            cells = _load_netcdf_overlay(overlay_file, fb_lat, fb_lon)
            if cells:
                layers.append(polygon_layer(
                    "grid-overlay",
                    data=cells,
                    get_polygon="@@=d.polygon",
                    get_fill_color=[255, 140, 0, 150],
                    get_line_color=[0, 0, 0, 0],
                    filled=True,
                    stroked=False,
                    pickable=True,
                ))
                legend_entries.append({
                    "layer_id": "grid-overlay",
                    "label": "Overlay Data",
                    "color": [255, 140, 0],
                    "shape": "rect",
                })
        elif overlay_file.suffix == ".csv":
            _log.info("CSV overlay support deferred: %s", overlay_path_str)
```

**Note:** CSV overlay rendering is deferred — most OSMOSE spatial data uses NetCDF. CSV support can be added later if needed.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add ui/pages/grid.py
git commit -m "feat: add spatial file overlay selector to grid preview"
```

---

## Chunk 4: Tooltips

### Task 10: Create tooltips module

**Files:**
- Create: `ui/tooltips.py`

- [ ] **Step 1: Create `ui/tooltips.py` with manual tooltip text**

```python
"""Tooltip text for non-schema UI fields."""

# Manual tooltips for fields not in the schema registry.
# Keys match the input IDs or descriptive labels used in the UI.
MANUAL_TOOLTIPS: dict[str, str] = {
    "jar_path": (
        "Path to the OSMOSE Java JAR file. "
        "Place JAR files in the osmose-java/ directory."
    ),
    "java_opts": (
        "JVM options passed to the Java process. "
        "Common: -Xmx4g (max heap 4 GB), -Xms1g (initial heap 1 GB)."
    ),
    "run_timeout": (
        "Maximum time in seconds before the simulation is killed. "
        "Increase for large grids or many species."
    ),
    "param_overrides": (
        "Override config parameters for this run only. "
        "One key=value per line. Example: simulation.nyear=100"
    ),
    "output_dir": (
        "Directory containing OSMOSE output CSV files. "
        "Set automatically after a run, or enter a path manually."
    ),
    "n_species": (
        "Number of focal (modeled) species in the simulation. "
        "Each species gets its own growth, reproduction, and mortality parameters."
    ),
    "n_resources": (
        "Number of lower trophic level (plankton) resource groups. "
        "These are forced from external data, not dynamically modeled."
    ),
    "load_example": (
        "Select a bundled example configuration to load. "
        "This replaces the current configuration."
    ),
}
```

- [ ] **Step 2: Commit**

```bash
git add ui/tooltips.py
git commit -m "feat: add manual tooltip text for non-schema UI fields"
```

---

### Task 11: Add tooltip markup to field rendering

**Files:**
- Modify: `ui/components/param_form.py`
- Modify: `www/osmose.css`
- Modify: `app.py`

- [ ] **Step 1: Add tooltip helper function to `param_form.py`**

Add after the imports:

```python
def _tooltip_content(field: OsmoseField) -> str:
    """Build tooltip HTML content from field metadata."""
    parts = [f"<strong>{field.description}</strong>"]
    if field.min_val is not None or field.max_val is not None:
        range_str = constraint_hint(field)
        if range_str:
            parts.append(f"<br>{range_str}")
    if field.default is not None:
        parts.append(f"<br>Default: {field.default}")
    parts.append(f"<br><code>{field.key_pattern}</code>")
    return "".join(parts)


def _wrap_with_tooltip(label: str, field: OsmoseField) -> ui.Tag:
    """Wrap a label string with a (?) tooltip icon."""
    content = _tooltip_content(field)
    return ui.tags.span(
        label,
        " ",
        ui.tags.span(
            "(?)",
            class_="osm-tooltip-icon",
            tabindex="0",
            **{
                "data-bs-toggle": "popover",
                "data-bs-trigger": "hover focus",
                "data-bs-html": "true",
                "data-bs-content": content,
                "data-bs-placement": "top",
            },
        ),
        # Hidden help text shown when "Show Help" toggle is active
        ui.tags.span(
            field.description or "",
            class_="field-help-text",
        ),
    )
```

- [ ] **Step 2: Update `render_field` to use tooltip labels**

In `render_field()`, replace the label construction:

```python
label = field.description or field.key_pattern
if field.unit:
    label = f"{label} ({field.unit})"
```

With:

```python
label_text = field.description or field.key_pattern
if field.unit:
    label_text = f"{label_text} ({field.unit})"
label = _wrap_with_tooltip(label_text, field)
```

**Note:** Shiny for Python's `input_numeric`, `input_text`, etc. accept `ui.TagChild` (which includes `ui.Tag`) for the `label` parameter. If any widget breaks with `ui.Tag` labels, fall back to using `str` labels and placing the tooltip icon as a sibling element after the widget instead.

- [ ] **Step 3: Update `render_species_table` param name cells with tooltips**

In the param cell generation within `render_species_table`, replace:

```python
param_cell = ui.tags.td(
    ui.tags.span(label),
    ui.tags.span(unit_text, style="color: #5a6a7a;"),
    ...
)
```

With:

```python
tooltip_html = _tooltip_content(field)
param_cell = ui.tags.td(
    ui.tags.span(label),
    ui.tags.span(unit_text, style="color: #5a6a7a;"),
    " ",
    ui.tags.span(
        "(?)",
        class_="osm-tooltip-icon",
        tabindex="0",
        **{
            "data-bs-toggle": "popover",
            "data-bs-trigger": "hover focus",
            "data-bs-html": "true",
            "data-bs-content": tooltip_html,
            "data-bs-placement": "right",
        },
    ),
    ui.tags.span(field.description or "", class_="field-help-text"),
    style="padding: 5px 12px; position: sticky; left: 0; z-index: 1; background: var(--osm-bg-card, #0f1923);",
)
```

- [ ] **Step 4: Add CSS for tooltip icon and help text**

Append to `www/osmose.css`:

```css
/* Tooltip (?) icon */
.osm-tooltip-icon {
    color: #5a6a7a;
    font-size: 11px;
    cursor: help;
    border: 1px solid #5a6a7a;
    border-radius: 50%;
    padding: 0 4px;
    margin-left: 4px;
    display: inline-block;
    line-height: 1.3;
}
.osm-tooltip-icon:hover {
    color: #d4a017;
    border-color: #d4a017;
}

/* Hidden help text — shown via "Show Help" toggle */
.field-help-text {
    display: none;
    color: #5a6a7a;
    font-size: 11px;
    margin-top: 2px;
}
body.show-all-help .field-help-text {
    display: block;
}
```

- [ ] **Step 5: Add popover initialization + Show Help toggle to app.py**

In `app.py`, add to the header actions div (inside `osmose-header-actions`), before the About link:

```python
ui.tags.button(
    "Show Help",
    class_="osmose-header-btn",
    id="helpToggle",
    onclick="toggleHelpMode()",
),
```

Add JS for popover init and help toggle, in the existing script block:

```javascript
// Initialize Bootstrap popovers on dynamic content
document.addEventListener('shiny:value', function() {
    document.querySelectorAll('[data-bs-toggle="popover"]').forEach(function(el) {
        if (!el._bsPopover) new bootstrap.Popover(el);
    });
});

// Show Help toggle
function toggleHelpMode() {
    document.body.classList.toggle('show-all-help');
    var active = document.body.classList.contains('show-all-help');
    localStorage.setItem('osmose-show-help', active ? '1' : '0');
    var btn = document.getElementById('helpToggle');
    if (btn) btn.textContent = active ? 'Hide Help' : 'Show Help';
}
// Restore help mode
(function() {
    if (localStorage.getItem('osmose-show-help') === '1') {
        document.body.classList.add('show-all-help');
    }
})();
```

- [ ] **Step 6: Add test for tooltip markup**

Add to `tests/test_param_form.py`:

```python
def test_tooltip_markup_in_render_field():
    """render_field should include tooltip (?) icon."""
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=100.0,
        min_val=1.0,
        max_val=500.0,
        description="L-infinity",
        unit="cm",
        indexed=True,
    )
    widget = render_field(field, species_idx=0)
    html = str(widget)
    assert "osm-tooltip-icon" in html
    assert "data-bs-toggle" in html
    assert "field-help-text" in html
```

- [ ] **Step 7: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add ui/components/param_form.py ui/tooltips.py www/osmose.css app.py tests/test_param_form.py
git commit -m "feat: add layered tooltip system with hover popovers and Show Help toggle"
```

---

### Task 12: Final integration test

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest --tb=short -q`
Expected: All 530+ tests pass

- [ ] **Step 2: Run linter**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: No errors

- [ ] **Step 3: Manual end-to-end verification**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`

Verify checklist:
1. Select "Eec Full" + click "Load" → header bar shows "Eec Full | 8 species | N params"
2. Edit a value → "modified" appears in header bar
3. Species tab shows spreadsheet table with collapsible categories
4. Forcing tab shows LTL resources in same spreadsheet format
5. Grid page shows overlay dropdown with spatial files
6. Run simulation → navigate to Results → auto-loads output
7. Species filter shows actual species names
8. Hover `(?)` icons → Bootstrap popover shows field details
9. Click "Show Help" → inline descriptions appear below all fields

- [ ] **Step 4: Commit any fixes from manual testing**

```bash
git add -A
git commit -m "fix: integration adjustments from manual testing"
```
