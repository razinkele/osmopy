# Live map: legend widget + empty-state hint removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the pre-run "Expand this card…" hint from the Live Movement status, and add a mode-aware deck.gl legend to the live map (per-species circle swatches in dots mode, a biomass-density gradient in heatmap mode) that refreshes as the species/stage/mode filters change.

**Architecture:** A pure `live_legend_widget` builder in `live_movement_render.py` produces a `layer_legend_widget` keyed off the rendered layer id; `_render_live_map` in `run.py` includes it in the widget set and refreshes it via `set_widgets` (or folds it into the layer `update` on an id change). Output-side only — no engine change.

**Tech Stack:** Python 3.12, shiny_deckgl (`layer_legend_widget`, `MapWidget.set_widgets`/`update`/`partial_update`), pytest, Shiny.

## Global Constraints

- **Run python via the MAIN venv ABSOLUTE path** `/home/razinka/osmose/osmose-python/.venv/bin/python` — do NOT create or symlink a `.venv` in the worktree (a worktree `.venv` symlink previously clobbered the real venv). Tests: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest <path> -v`. Lint: `/home/razinka/osmose/osmose-python/.venv/bin/ruff check` + `ruff format --check`.
- Legend keyed off the **actual rendered layer id**: `layer_id == _DOTS_ID` → per-species circle entries; else → biomass-density gradient. (So the >1500 dots→heatmap fallback shows the gradient.)
- Legend title: `f"{species_filter or 'All species'} · {stage_label}"`, `stage_label = "All stages" if stage_filter is None else STAGE_LABELS[stage_filter]`.
- `show_checkbox=False`, `placement="bottom-right"` (the four nav widgets occupy the other corners).
- `STAGE_LABELS` is imported from `osmose.live_movement` (verified present there; engine-side, no UI import → no cycle). `species_color`, `PALETTE_THERMAL`, `_DOTS_ID`, `_HEATMAP_ID` already live in `live_movement_render.py`.
- Render skeleton (verbatim shape): compare the PREVIOUS `_live_widget_sig`, assign at the BOTTOM on every path; fold the legend into the `update` on first-frame and id-change; standalone `set_widgets` only on the `partial_update` (same-id) path when the sig changed.
- `_live_widget_sig` reset at run start alongside `_live_framed`/`_live_layer_id`.
- No engine-dynamics change — output-side only; EEC/BoB parity untouched.
- Spec: `docs/superpowers/specs/2026-06-27-live-map-legend-design.md`.

---

### Task 1: `live_legend_widget` builder

**Files:**
- Modify: `ui/pages/live_movement_render.py` (imports + new helper after `choose_live_layer`)
- Test: `tests/test_live_movement_render.py` (4 new tests)

**Interfaces:**
- Produces: `live_legend_widget(snap: MovementSnapshot, species_filter: str | None, stage_filter: int | None, layer_id: str) -> dict` — returns a `layer_legend_widget` dict with keys incl. `entries`, `title`, `placement`, `showCheckbox`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_live_movement_render.py`:

```python
def test_live_legend_dots_lists_displayed_species():
    from ui.pages.live_movement_render import _DOTS_ID, live_legend_widget, species_color
    snap = _snap([0, 1], [0, 1], [0, 1], [1, 1], species=("cod", "sprat"))
    w = live_legend_widget(snap, None, None, _DOTS_ID)
    assert w["title"] == "All species · All stages"
    assert [e["label"] for e in w["entries"]] == ["cod", "sprat"]
    assert w["entries"][0]["color"] == list(species_color(0))
    assert w["entries"][0]["shape"] == "circle"


def test_live_legend_dots_species_filter_single_entry():
    from ui.pages.live_movement_render import _DOTS_ID, live_legend_widget
    snap = _snap([0, 1], [0, 1], [0, 1], [1, 1], species=("cod", "sprat"))
    w = live_legend_widget(snap, "cod", None, _DOTS_ID)
    assert [e["label"] for e in w["entries"]] == ["cod"]
    assert w["title"].startswith("cod · ")


def test_live_legend_stage_in_title():
    from ui.pages.live_movement_render import _DOTS_ID, live_legend_widget
    snap = _snap([0], [0], [0], [1], species=("cod",))
    w = live_legend_widget(snap, None, 2, _DOTS_ID)
    assert w["title"].endswith("· Adult")


def test_live_legend_heatmap_is_gradient():
    from shiny_deckgl import PALETTE_THERMAL
    from ui.pages.live_movement_render import _HEATMAP_ID, live_legend_widget
    snap = _snap([0, 1], [0, 1], [0, 1], [1, 1])
    w = live_legend_widget(snap, None, None, _HEATMAP_ID)
    assert len(w["entries"]) == 1
    assert w["entries"][0]["shape"] == "gradient"
    assert w["entries"][0]["colors"] == [list(c) for c in PALETTE_THERMAL]
```

- [ ] **Step 2: Run — verify it fails**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_live_movement_render.py -k live_legend -v`
Expected: FAIL (`live_legend_widget` not defined / cannot import).

- [ ] **Step 3: Implement** in `ui/pages/live_movement_render.py`:

(a) Add `layer_legend_widget` to the shiny_deckgl import (alphabetical, after `heatmap_layer`):
```python
from shiny_deckgl import (  # type: ignore[import-untyped]
    PALETTE_THERMAL,
    color_range,
    heatmap_layer,
    layer_legend_widget,
    scatterplot_layer,
)
```

(b) Add `STAGE_LABELS` to the osmose import:
```python
from osmose.live_movement import STAGE_LABELS, MovementSnapshot
```

(c) Append the helper at the end of the file (after `choose_live_layer`):
```python
def live_legend_widget(
    snap: MovementSnapshot,
    species_filter: str | None,
    stage_filter: int | None,
    layer_id: str,
) -> dict:
    """Legend reflecting the displayed features, keyed off the rendered layer id.

    Dots (``layer_id == _DOTS_ID``): one circle swatch per displayed species (filtered to
    ``species_filter`` when set). Heatmap (the >1500 dots fallback included): a single
    biomass-density gradient. Title carries the active species + stage filter.
    """
    stage_label = "All stages" if stage_filter is None else STAGE_LABELS[stage_filter]
    title = f"{species_filter or 'All species'} · {stage_label}"
    if layer_id == _DOTS_ID:
        entries = [
            {"label": name, "color": list(species_color(i)), "shape": "circle"}
            for i, name in enumerate(snap.species)
            if species_filter is None or name == species_filter
        ]
    else:
        entries = [
            {
                "label": "biomass density",
                "shape": "gradient",
                "colors": [list(c) for c in PALETTE_THERMAL],
            }
        ]
    return layer_legend_widget(
        entries=entries, title=title, placement="bottom-right", show_checkbox=False
    )
```

- [ ] **Step 4: Run — verify it passes**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_live_movement_render.py -q`
Expected: PASS (4 new + existing render tests).

- [ ] **Step 5: Commit**

```bash
git add ui/pages/live_movement_render.py tests/test_live_movement_render.py
git commit -m "feat(live): live_legend_widget — mode-aware legend (species swatches / biomass gradient)"
```

---

### Task 2: Wire the legend into the render effect + remove the hint

**Files:**
- Modify: `ui/pages/run.py` (imports, `live_movement_status`, `_render_live_map`, state init + run-start reset)
- Modify: `tests/test_e2e_live_movement.py` (add a same-id `set_widgets`-path assertion)
- Verify: `import app` smoke

**Interfaces:**
- Consumes: `live_legend_widget` (Task 1).

- [ ] **Step 1: Imports.** Add `layer_legend_widget`'s consumer + the builder:
  - Add `live_legend_widget` to the live_movement_render import (`run.py:38`):
    ```python
    from ui.pages.live_movement_render import choose_live_layer, live_legend_widget
    ```
  (No new shiny_deckgl import needed in run.py — the widget builders + `MapWidget` are already imported; the legend dict is produced by `live_legend_widget`.)

- [ ] **Step 2: Remove the hint.** In `live_movement_status` (`run.py`), replace the python-engine no-run return:
  ```python
            return ui.p(
                "Expand this card and run the model to stream movement, "
                "then filter by species / stage.",
                class_="text-muted",
            )
  ```
  with:
  ```python
            return ui.div()
  ```
  (Keep the non-Python `ui.p("Live view available for the Python engine.", ...)` branch + the running/done path unchanged.)

- [ ] **Step 3: State var.** Add next to `_live_layer_id` (`run.py:469`):
  ```python
    _live_widget_sig: list[tuple | None] = [None]  # (layer_id, species_filter, stage_filter)
  ```

- [ ] **Step 4: Run-start reset.** After `_live_layer_id[0] = None` (run start, ~`run.py:851`):
  ```python
            _live_layer_id[0] = None
            _live_widget_sig[0] = None
  ```

- [ ] **Step 5: Render skeleton.** In `_render_live_map`, replace the block from
  `layer, note = choose_live_layer(...)` through `_live_layer_id[0] = layer["id"]` with:
  ```python
            layer, note = choose_live_layer(snap, species_filter, mode, stage_filter=stage_filter)
            _live_note.set(note)
            legend = live_legend_widget(snap, species_filter, stage_filter, layer["id"])
            widgets = [
                fullscreen_widget(placement="top-left"),
                zoom_widget(placement="top-right"),
                compass_widget(placement="top-right"),
                scale_widget(placement="bottom-left"),
                legend,
            ]
            sig = (layer["id"], species_filter, stage_filter)
            if not _live_framed[0]:
                await _live_map.update(
                    session,
                    layers=[layer],
                    view_state={
                        "latitude": (snap.lat_min + snap.lat_max) / 2,
                        "longitude": (snap.lon_min + snap.lon_max) / 2,
                        "zoom": 5,
                    },
                    widgets=widgets,
                )
                _live_framed[0] = True
            elif layer["id"] != _live_layer_id[0]:
                # The active representation switched (heatmap <-> dots), distinct layer ids.
                # deck.gl cannot swap a layer's class under one id; a full update (no view_state,
                # to keep the camera) removes the old id and carries the fresh legend in one message.
                await _live_map.update(session, layers=[layer], widgets=widgets)
            else:
                await _live_map.partial_update(session, layers=[layer])
                if sig != _live_widget_sig[0]:
                    # same layer id, species/stage changed -> refresh the legend only.
                    await _live_map.set_widgets(session, widgets)
            _live_layer_id[0] = layer["id"]
            _live_widget_sig[0] = sig
  ```

- [ ] **Step 6: Verify app imports + no syntax/lint error**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`.
Run: `/home/razinka/osmose/osmose-python/.venv/bin/ruff check ui/pages/run.py ui/pages/live_movement_render.py && /home/razinka/osmose/osmose-python/.venv/bin/ruff format --check ui/pages/run.py ui/pages/live_movement_render.py`
Expected: clean.

- [ ] **Step 7: Extend the e2e to exercise the `set_widgets`-only (same-id) path.**
The existing `test_live_movement_renders_during_python_run` selects a species (all→cod), which
*flips* the layer id (heatmap fallback → dots) and so only hits the `update(widgets=...)` branch.
The new standalone `set_widgets` branch (same layer id, species/stage changed) is NOT covered.
Add a same-id change: a stage filter on cod stays in dots (cod+adult ⊂ cod, still under the dots
cap) → layer id unchanged, sig changes → the `set_widgets` branch. In
`tests/test_e2e_live_movement.py`, after the existing
`assert not deck_errors, f"deck.gl draw error on heatmap->dots swap: {deck_errors[:2]}"` line,
append:
```python
    # Same-id refresh path: a stage filter on cod stays in dots (so the layer id does NOT flip),
    # exercising the standalone set_widgets legend refresh. Must not crash deck.gl either.
    page.select_option("#live_movement_stage", "2")  # Adult
    page.wait_for_timeout(800)
    assert not deck_errors, f"deck.gl draw error on same-id legend refresh: {deck_errors[:2]}"
```

- [ ] **Step 8: Run the e2e (both the id-flip and the same-id set_widgets paths)**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_e2e_live_movement.py::test_live_movement_renders_during_python_run -m e2e -q`
Expected: PASS — Baltic → expand → Dots → select cod (id-flip `update(widgets=)` path) → select Adult
(same-id `set_widgets` path), with `deck_errors` empty at both checkpoints. Confirms both legend-push
paths render without crashing deck.gl.

- [ ] **Step 9: Commit**

```bash
git add ui/pages/run.py tests/test_e2e_live_movement.py
git commit -m "feat(run): wire mode-aware legend into live map + remove pre-run hint"
```

---

## Notes for the executor

- **No engine-dynamics change** — only `live_movement_render.py` (a pure builder) + `run.py` (the render effect). EEC/BoB parity untouched; `import app` must stay green.
- The legend swatches match the on-screen layer because the builder keys off `layer["id"]` (dots vs heatmap), and `enumerate(snap.species)` index == `sp_id` (the same index `species_color` uses for the dots), so colors agree with the drawn dots.
- `set_widgets` only fires on the `partial_update` (same-id) path when the sig changed — pure streaming frames (same filters) send no widget message; id changes carry the legend in the layer `update`.
- Do not create a `.venv` in the worktree; always use the main-venv absolute path.
