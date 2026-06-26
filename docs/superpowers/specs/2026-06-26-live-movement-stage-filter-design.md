# Live Movement: species fix + life-stage filter + horizontal layout — design

> Status: design (awaiting review) · 2026-06-26
> Fix the Live Movement species selector (it never populates without live frames), add a
> life-stage filter (egg/larva · juvenile · adult), and lay the Mode/Species/Stage controls out
> horizontally to save vertical space. Python-engine live-movement viz only; no engine-dynamics change.

## 1. Problem

On the Run page's **Live Movement** card (`ui/pages/run.py:249-264`):
- **Species not selectable** — `live_movement_species` choices are populated only by `_populate_live_species`, which depends on `_live_snapshot` (live frames). Before/without a stream the dropdown is stuck on "All species", so a user can't pick a species.
- **No life-stage filter** — only whole-species filtering exists; users want to see where eggs/larvae vs juveniles vs adults move (nurseries vs spawning grounds).
- **Vertical sprawl** — Mode (radio) and Species (select) stack vertically; adding a third control worsens it.

## 2. Approach

Three coordinated changes, smallest unit each:

### 2.1 Snapshot carries life stage — `osmose/live_movement.py`
- `MovementSnapshot` gains `stage: NDArray[np.int8]` — one value per included school: **0 = egg/larva, 1 = juvenile, 2 = adult**.
- `build_snapshot` (already receives `config`) computes it on the existing selection `mask`, per school:
  - **egg/larva (0)** if `state.is_egg[mask]` (i.e. `age_dt < first_feeding_age_dt`);
  - **adult (2)** if mature — `state.length[mask] >= config.maturity_size[sp]` AND `state.age_dt[mask] >= config.maturity_age_dt[sp]` (the same conjunction `reproduction.py:101-104` / the SSB collector use);
  - **juvenile (1)** otherwise.
  - `stage = np.where(is_egg, 0, np.where(mature, 2, 1)).astype(np.int8)`.
- Add module constant `STAGE_LABELS = {0: "Egg/larva", 1: "Juvenile", 2: "Adult"}` (single source of truth for UI + filter).
- No signature change; pure; sampling/`dot_cap` slices `stage` alongside the other per-school arrays.

### 2.2 Stage filter in the renderer — `ui/pages/live_movement_render.py`
- `_filter_mask(snap, species_filter, stage_filter)` composes `species_mask & stage_mask`, where
  `stage_mask = (snap.stage == stage_filter)` when `stage_filter is not None`, else all-True.
- `choose_live_layer(snap, species_filter, stage_filter, mode)` threads `stage_filter` into both the
  heatmap (`_points_to_rows`) and dots paths (which already call `_filter_mask`).

### 2.3 UI + server — `ui/pages/run.py`
- **Species fix (the bug):** populate `live_movement_species` choices from the **config species names**
  as soon as a config is loaded (a reactive on `state.config` / the focal `species.name.sp*` keys),
  i.e. `{"__all__": "All species", <name>: <name>, ...}`. `_populate_live_species` (from live frames)
  can stay as a refresh but is no longer the only source — so the dropdown is usable immediately.
- **New stage selector** `live_movement_stage`: `{"__all__": "All stages", "0": "Egg/larva",
  "1": "Juvenile", "2": "Adult"}` (fixed, species-agnostic — derived from `STAGE_LABELS`).
- **Horizontal layout:** wrap Mode + Species + Stage in `ui.layout_columns(..., col_widths=[4, 4, 4])`
  so the three controls sit side-by-side. (Mode stays `input_radio_buttons` inline; Species/Stage are
  `input_select`.)
- **Render effect** (`_render_live_map`): read `live_movement_stage()` alongside species/mode, map
  `"__all__"/None → None` else `int(sel)`, and pass both filters to `choose_live_layer`. A species/stage
  change re-renders from the current snapshot (the effect already reacts to these inputs).

## 3. Data flow

config load → species choices (immediately). engine frame → `build_snapshot` (now with `stage`) →
`_live_snapshot` → `_render_live_map` reads (mode, species, stage) → `choose_live_layer` applies
`species_mask & stage_mask` → deck.gl layer.

## 4. Edge cases

- `"__all__"` species and/or stage → that filter is `None` (no masking).
- A species×stage combo with zero located schools → empty layer; the existing "no schools to show"
  note path handles it (mask all-False → 0 rows).
- Eggs that are freshly spawned and **unlocated** (`cell < 0`) are already excluded by `build_snapshot`'s
  mask — so "Egg/larva" shows only *located* egg/larva schools (correct for a map).
- Species with `maturity_size`/`maturity_age` of 0 (default) → `mature` is true for any non-egg school
  → that species is juvenile only at age/length 0, adult otherwise (acceptable; matches the engine's
  own maturity semantics).

## 5. Testing

- **Unit (`osmose/live_movement.py`):** `build_snapshot` assigns stage correctly — a state with an egg
  (`is_egg=True`), an immature non-egg (length < maturity_size), and a mature school (length ≥
  maturity_size & age ≥ maturity_age) → stage `[0, 1, 2]`; `stage` is sampled consistently with the
  other arrays under `dot_cap`.
- **Unit (`live_movement_render.py`):** `_filter_mask` with `stage_filter=1` returns only juvenile
  schools; species+stage compose (intersection); `None` filters select all.
- **No engine-dynamics change** — `build_snapshot` is an output-side read; EEC/BoB parity untouched.
- UI wiring (the 3-column layout + the two selects) is verified by `import app` + the existing
  run-page visual/e2e harness if it covers Live Movement; otherwise a manual check.

## 6. Out of scope

- Feeding/predation-stage or age-class filtering (chose ontogenetic life stages).
- Per-species custom stage labels (fixed egg/larva·juvenile·adult is species-agnostic by design).
- Background-species movement (the snapshot is focal-only, unchanged).
