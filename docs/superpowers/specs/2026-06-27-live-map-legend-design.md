# Live map: remove empty-state hint + mode-aware legend widget — design

> Status: design (awaiting review) · 2026-06-27
> Remove the pre-run empty-state hint line from the Live Movement status, and add a mode-aware
> deck.gl legend widget to the live map that shows the features currently displayed (per-species
> color swatches in dots mode; a biomass-density gradient in heatmap mode). Output-side only; no
> engine change. Fullscreen + the other nav widgets already exist and stay.

## 1. Problem / goal

- The `live_movement_status` pre-run message ("Expand this card and run the model to stream
  movement, then filter by species / stage.") is unwanted — remove it.
- The live deck.gl map has nav widgets (fullscreen, zoom, compass, scale) but no legend, so the
  user can't tell which species/stage the colors on the map represent. Add a legend that reflects
  the currently displayed features and updates as the species/stage/mode filters change.

## 2. Approach

### 2.1 Remove the hint — `ui/pages/run.py` (`live_movement_status`)
The python-engine, no-run branch currently returns `ui.p("Expand this card and run the model to
stream movement, then filter by species / stage.", class_="text-muted")`. Replace that return with
an empty element — `return ui.div()` — so nothing renders pre-run. Keep the non-Python branch
(`"Live view available for the Python engine."`) and the running/done status path unchanged.

### 2.2 Legend builder — `ui/pages/live_movement_render.py`
Add a pure helper keyed off the **actual rendered layer** (so the dots→heatmap fallback at >1500
schools is reflected, not just the requested mode):

```
live_legend_widget(snap: MovementSnapshot, species_filter: str | None,
                   stage_filter: int | None, layer: dict) -> dict
```

- **Title** (both modes): `f"{species_filter or 'All species'} · {stage_label}"` where
  `stage_label = "All stages" if stage_filter is None else STAGE_LABELS[stage_filter]`.
- **layer is dots** (`layer["id"] == _DOTS_ID`): one entry per displayed species —
  `{"label": name, "color": list(species_color(i)), "shape": "circle"}` for `i, name in
  enumerate(snap.species)`, filtered to `species_filter` when set (else all focal species).
- **layer is heatmap** (otherwise): one entry
  `{"label": "biomass density", "shape": "gradient", "colors": [list(c) for c in PALETTE_THERMAL]}`.
- Returns `layer_legend_widget(entries=entries, title=title, placement="bottom-right",
  show_checkbox=False)` (checkboxes are off — all dots entries share one `_DOTS_ID`, so a per-entry
  visibility toggle would hide the whole layer; the legend is informational).
- `STAGE_LABELS` and `layer_legend_widget` are imported (from `osmose.live_movement` and
  `shiny_deckgl` respectively); `species_color`/`PALETTE_THERMAL`/`_DOTS_ID` already live in this file.

### 2.3 Wire into `_render_live_map` — `ui/pages/run.py`
The widget set becomes `[fullscreen, zoom, compass, scale, legend]`. The legend is refreshed
independently of the layer stream via `MapWidget.set_widgets` (updates widgets without resending
layers):

- Compute `legend = live_legend_widget(snap, species_filter, stage_filter, layer)` after
  `choose_live_layer`. Build `widgets = [fullscreen_widget(...), zoom_widget(...),
  compass_widget(...), scale_widget(...), legend]`.
- **First frame** (`not _live_framed[0]`): `update(..., layers=[layer], view_state=..., widgets=widgets)`
  — unchanged path, now including the legend.
- **Subsequent frames:** layers as today — `update(session, layers=[layer])` on a layer-id change
  (keeps camera + widgets), else `partial_update(session, layers=[layer])`.
- **Legend refresh:** track `_live_widget_sig = (layer_id, species_filter, stage_filter)`. After the
  first frame, when the signature changes, `await _live_map.set_widgets(session, widgets)` — refreshes
  the legend (and a dots↔heatmap-fallback flip) without touching layers or camera.
- Reset `_live_widget_sig[0] = None` at run start (alongside `_live_framed[0] = False` /
  `_live_layer_id[0] = None`).

## 3. Data flow

frame → `build_snapshot` → `_live_snapshot` → `_render_live_map`: `choose_live_layer` returns the
layer → `live_legend_widget` builds the legend from that layer + the filters → layers pushed via
`update`/`partial_update`; the legend pushed via `set_widgets` only when `(layer_id, species,
stage)` changes. The legend's swatches always match the on-screen layer (dots colors vs heatmap
gradient) because it keys off the returned `layer`, not the requested `mode`.

## 4. Edge cases

- **Empty / no schools:** `snap.species` is always the full focal-species list, so the dots legend
  lists the selected species even if a given species currently has zero located schools (it is a
  valid selection). Heatmap legend is a fixed gradient. No crash on an empty snapshot.
- **dots→heatmap fallback (>1500 schools):** `choose_live_layer` returns a heatmap layer; the legend
  keys off `layer["id"]` so it correctly shows the gradient (not species swatches). The id flip is
  captured by `_live_widget_sig`, so the legend refreshes.
- **Pre-run:** `_render_live_map` returns early when `_live_snapshot is None`, so no legend is built;
  the hint removal leaves the status blank (intended).

## 5. Testing

- **Unit (`live_movement_render.py`):** `live_legend_widget` —
  - dots layer, no species filter → one circle entry per focal species, colors match
    `species_color(i)`, title `"All species · All stages"`;
  - dots layer + `species_filter="cod"` → exactly one entry labelled `"cod"`;
  - dots layer + `stage_filter=2` → title ends with `"· Adult"`;
  - heatmap layer → a single `"gradient"` entry with `colors == [list(c) for c in PALETTE_THERMAL]`.
- **Wiring:** verified by `import app` + the existing `test_e2e_live_movement.py` (the legend widget
  renders without console error; the e2e already exercises run → dots-filter). No new engine path.
- No engine-dynamics change — output-side only; EEC/BoB parity untouched.

## 6. Out of scope

- Per-species visibility toggles (all dots share one layer id; checkboxes off).
- A biomass scale/colorbar with numeric breakpoints (the gradient is qualitative low→high).
- Legend for background species (the snapshot is focal-only, unchanged).
