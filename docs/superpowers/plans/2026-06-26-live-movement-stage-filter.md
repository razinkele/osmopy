# Live Movement: species fix + life-stage filter + horizontal layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Live Movement species selector (populate from config, not just live frames), add an egg/larva·juvenile·adult life-stage filter, and lay Mode/Species/Stage out in 3 columns.

**Architecture:** `MovementSnapshot` gains a per-school `stage` array computed in `build_snapshot` from `is_egg` + the maturity conjunction; the renderer composes a `species_mask & stage_mask`; the Run page populates species from `state.config`, adds a stage `input_select`, wraps the three controls in `layout_columns`, and owns the no-snapshot empty state.

**Tech Stack:** Python 3.12, NumPy, pytest, Shiny (reactive effects + `ui.update_select` + `layout_columns`), shiny_deckgl.

## Global Constraints

- Stage encoding: **0 = egg/larva, 1 = juvenile, 2 = adult**; `STAGE_LABELS = {0: "Egg/larva", 1: "Juvenile", 2: "Adult"}` is the single source of truth (UI choices + filter).
- Stage rule (per school): egg/larva if `state.is_egg`; adult if `state.length >= config.maturity_size[sp] AND state.age_dt >= config.maturity_age_dt[sp]` (matches `reproduction.py:101-104`); else juvenile.
- `stage` is computed on the selection `mask` then sliced by `[idx]` under `dot_cap` sampling, alongside `sp_id/cx/cy/bm`.
- `stage_filter` is threaded through ALL of `_filter_mask`, `_points_to_rows`, `heatmap_layer_from_points`, `dots_layer_from_points`, `choose_live_layer` (passed as a **keyword arg** to avoid breaking the positional call site).
- The dots-vs-heatmap fallback count in `choose_live_layer` MUST use the **composed** species+stage mask.
- Layout: `ui.layout_columns(col_widths=[3, 5, 4])` (narrow radio, wider selects).
- The species fix populates + makes the dropdown clickable, but selecting only changes the map when `_live_snapshot` is non-None (during a stream / retained final frame). With no run yet, show an **empty-state hint**.
- **No engine-dynamics change** — `build_snapshot` is output-side; EEC/BoB parity untouched. Run from worktree root with `PYTHONPATH=. .venv/bin/python`; lint `.venv/bin/ruff check` + `ruff format --check`.
- Spec: `docs/superpowers/specs/2026-06-26-live-movement-stage-filter-design.md`.

---

### Task 1: Snapshot carries life stage

**Files:**
- Modify: `osmose/live_movement.py` (the `MovementSnapshot` dataclass + `build_snapshot`)
- Test: `tests/test_live_movement.py` (extend the `_state`/`_config` fixtures + a new test)

**Interfaces:**
- Produces: `MovementSnapshot.stage: NDArray[np.int8]` (per-school, 0/1/2, same length+order as `sp_id`); module constant `STAGE_LABELS: dict[int, str]`.

- [ ] **Step 1: Write the failing test** — extend the fixtures + add a stage test in `tests/test_live_movement.py`. Replace the existing `_state` and `_config` helpers (top of file) with these (adds the fields `build_snapshot` will now read), then append the test:

```python
def _state(species_id, cell_x, cell_y, biomass, is_out=None,
           length=None, age_dt=None, is_egg=None):
    """A lightweight stand-in exposing only the fields build_snapshot reads."""
    n = len(species_id)
    return types.SimpleNamespace(
        species_id=np.array(species_id, dtype=np.int32),
        cell_x=np.array(cell_x, dtype=np.int32),
        cell_y=np.array(cell_y, dtype=np.int32),
        biomass=np.array(biomass, dtype=np.float64),
        is_out=np.array(is_out if is_out is not None else [False] * n, dtype=bool),
        length=np.array(length if length is not None else [10.0] * n, dtype=np.float64),
        age_dt=np.array(age_dt if age_dt is not None else [12] * n, dtype=np.int32),
        is_egg=np.array(is_egg if is_egg is not None else [False] * n, dtype=bool),
    )


def _config(n_species=2, n_steps=12, names=("cod", "sprat"),
            maturity_size=(5.0, 5.0), maturity_age_dt=(6, 6)):
    return types.SimpleNamespace(
        n_species=n_species, n_steps=n_steps, species_names=list(names),
        maturity_size=np.array(maturity_size, dtype=np.float64),
        maturity_age_dt=np.array(maturity_age_dt, dtype=np.int32),
    )


def test_build_snapshot_assigns_life_stage():
    from osmose.live_movement import build_snapshot, STAGE_LABELS
    g = Grid.from_dimensions(ny=3, nx=3)
    # 3 cod schools: egg (is_egg), juvenile (small/young, immature), adult (mature)
    st = _state(
        species_id=[0, 0, 0], cell_x=[0, 1, 2], cell_y=[0, 1, 2], biomass=[1.0, 1.0, 1.0],
        length=[0.1, 2.0, 50.0], age_dt=[0, 2, 30], is_egg=[True, False, False],
    )
    snap = build_snapshot(0, st, g, _config(maturity_size=(5.0, 5.0), maturity_age_dt=(6, 6)))
    assert list(snap.stage) == [0, 1, 2]  # egg/larva, juvenile, adult
    assert STAGE_LABELS == {0: "Egg/larva", 1: "Juvenile", 2: "Adult"}


def test_build_snapshot_stage_sliced_under_dot_cap():
    from osmose.live_movement import build_snapshot
    g = Grid.from_dimensions(ny=3, nx=3)
    n = 30
    st = _state(species_id=[0] * n, cell_x=[1] * n, cell_y=[1] * n, biomass=[1.0] * n,
                length=[50.0] * n, age_dt=[30] * n, is_egg=[False] * n)
    snap = build_snapshot(0, st, g, _config(), dot_cap=10)
    assert snap.stage.size == snap.sp_id.size == 10  # sampled in lockstep
    assert set(snap.stage.tolist()) == {2}  # all adult
```

- [ ] **Step 2: Run — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_live_movement.py -k "life_stage or stage_sliced" -v`
Expected: FAIL (`MovementSnapshot` has no `stage` / `STAGE_LABELS` not defined).

- [ ] **Step 3: Implement** in `osmose/live_movement.py`:

(a) Add the module constant near the top (after the imports):
```python
# Ontogenetic life stage per school for the live-movement filter.
STAGE_LABELS = {0: "Egg/larva", 1: "Juvenile", 2: "Adult"}
```

(b) Add a field to `MovementSnapshot` (after `biomass: NDArray[np.float64]`):
```python
    stage: NDArray[np.int8]  # per-school life stage: 0=egg/larva, 1=juvenile, 2=adult
```

(c) In `build_snapshot`, compute `stage` on the mask and sample it with the others. Replace the block from `sp_id = state.species_id[mask]` through the `if truncated:` slice with:
```python
    sp_id = state.species_id[mask]
    cx = state.cell_x[mask]
    cy = state.cell_y[mask]
    bm = state.biomass[mask]
    # Life stage: egg/larva if is_egg; adult if mature (length>=maturity_size AND
    # age>=maturity_age, per reproduction.py); else juvenile. Computed on the mask.
    is_egg_m = state.is_egg[mask]
    mature_m = (state.length[mask] >= config.maturity_size[sp_id]) & (
        state.age_dt[mask] >= config.maturity_age_dt[sp_id]
    )
    stage = np.where(is_egg_m, 0, np.where(mature_m, 2, 1)).astype(np.int8)
    n_total = int(sp_id.size)
    truncated = n_total > dot_cap
    if truncated:
        idx = np.linspace(0, n_total - 1, dot_cap).astype(np.intp)
        sp_id, cx, cy, bm, stage = sp_id[idx], cx[idx], cy[idx], bm[idx], stage[idx]
```

(d) In the `MovementSnapshot(...)` return, add (after `biomass=...`):
```python
        stage=np.asarray(stage, dtype=np.int8),
```

- [ ] **Step 4: Run — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_live_movement.py -q`
Expected: PASS (new tests + the existing build_snapshot tests still green — the fixtures gained fields with defaults).

- [ ] **Step 5: Commit**

```bash
git add osmose/live_movement.py tests/test_live_movement.py
git commit -m "feat(live): MovementSnapshot carries per-school life stage (egg/larva/juvenile/adult)"
```

---

### Task 2: Stage filter in the renderer

**Files:**
- Modify: `ui/pages/live_movement_render.py` (`_filter_mask`, `_points_to_rows`, `heatmap_layer_from_points`, `dots_layer_from_points`, `choose_live_layer`)
- Test: `tests/test_live_movement_render.py` (extend `_snap` with `stage` + new tests)

**Interfaces:**
- Consumes: `MovementSnapshot.stage` (Task 1).
- Produces: `_filter_mask(snap, species_filter, stage_filter)`, `choose_live_layer(snap, species_filter, mode, *, dots_max=1500, stage_filter=None)`.

- [ ] **Step 1: Write the failing test** — in `tests/test_live_movement_render.py`, replace the `_snap` helper's signature to carry `stage` and append tests:

```python
def _snap(sp_id, lon, lat, biomass, species=("cod", "sprat"), lon_step=1.0, lat_step=1.0,
          stage=None):
    lo, la = list(lon), list(lat)
    return MovementSnapshot(
        step=0, n_steps=12, status="running", species=list(species),
        sp_id=np.array(sp_id, dtype=np.int32),
        lon=np.array(lon, dtype=np.float64), lat=np.array(lat, dtype=np.float64),
        biomass=np.array(biomass, dtype=np.float64),
        stage=np.array(stage if stage is not None else [1] * len(sp_id), dtype=np.int8),
        truncated=False, n_total=len(sp_id),
        lon_min=float(min(lo)) if lo else 0.0, lon_max=float(max(lo)) if lo else 0.0,
        lat_min=float(min(la)) if la else 0.0, lat_max=float(max(la)) if la else 0.0,
        lon_step=lon_step, lat_step=lat_step,
    )


def test_filter_mask_stage_and_species_compose():
    from ui.pages.live_movement_render import _filter_mask
    snap = _snap(sp_id=[0, 0, 1], lon=[0, 1, 2], lat=[0, 1, 2], biomass=[1, 1, 1],
                 stage=[1, 2, 2])  # cod-juv, cod-adult, sprat-adult
    # species cod + stage adult -> only the 2nd school
    m = _filter_mask(snap, "cod", 2)
    assert list(m) == [False, True, False]
    # stage only (adult) -> schools 2 and 3
    assert list(_filter_mask(snap, None, 2)) == [False, True, True]
    # no filters -> all
    assert list(_filter_mask(snap, None, None)) == [True, True, True]


def test_choose_live_layer_fallback_uses_composed_count():
    from ui.pages.live_movement_render import choose_live_layer
    # 5 cod schools, only 1 adult; dots_max=2 -> species-only count (5) would fall back to
    # heatmap, but the composed (stage=adult) count is 1 -> must STAY in dots.
    snap = _snap(sp_id=[0, 0, 0, 0, 0], lon=[0, 1, 2, 3, 4], lat=[0, 1, 2, 3, 4],
                 biomass=[1, 1, 1, 1, 1], stage=[1, 1, 1, 1, 2])
    layer, note = choose_live_layer(snap, "cod", "dots", dots_max=2, stage_filter=2)
    assert note is None  # stayed in dots (composed count = 1 <= 2)
    assert layer["@@type"] == "ScatterplotLayer"
```

- [ ] **Step 2: Run — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_live_movement_render.py -k "stage or composed" -v`
Expected: FAIL (`_filter_mask` takes 2 args; `choose_live_layer` has no `stage_filter`).

- [ ] **Step 3: Implement** in `ui/pages/live_movement_render.py` — thread `stage_filter` through all five:

```python
def _filter_mask(
    snap: MovementSnapshot, species_filter: str | None, stage_filter: int | None = None
) -> np.ndarray:
    if species_filter is None:
        sp_mask = np.ones(snap.sp_id.size, dtype=bool)
    else:
        try:
            target = snap.species.index(species_filter)
        except ValueError:
            sp_mask = np.zeros(snap.sp_id.size, dtype=bool)
        else:
            sp_mask = snap.sp_id == target
    if stage_filter is None:
        return sp_mask
    return sp_mask & (snap.stage == stage_filter)


def _points_to_rows(
    snap: MovementSnapshot, species_filter: str | None, stage_filter: int | None = None
) -> list[dict]:
    """Base rows: position + weight + fill. Heatmap ignores fill (one builder, both modes)."""
    m = _filter_mask(snap, species_filter, stage_filter)
    sp_id, lon, lat, bm = snap.sp_id[m], snap.lon[m], snap.lat[m], snap.biomass[m]
    return [
        {"position": [float(lo), float(la)], "weight": float(b), "fill": species_color(s)}
        for s, lo, la, b in zip(sp_id, lon, lat, bm)
    ]


def heatmap_layer_from_points(
    snap: MovementSnapshot, species_filter: str | None, stage_filter: int | None = None
) -> dict:
    """Native deck.gl HeatmapLayer weighted by biomass, from un-jittered cell centers."""
    return heatmap_layer(
        _LAYER_ID,
        data=_points_to_rows(snap, species_filter, stage_filter),
        getPosition="@@=d.position",
        getWeight="@@=d.weight",
        colorRange=color_range(palette=PALETTE_THERMAL),
    )
```

In `dots_layer_from_points`, change the signature + the mask line:
```python
def dots_layer_from_points(
    snap: MovementSnapshot, species_filter: str | None, stage_filter: int | None = None
) -> dict:
    ...  # (docstring unchanged)
    m = _filter_mask(snap, species_filter, stage_filter)
    sp_id, lon, lat, bm = snap.sp_id[m], snap.lon[m], snap.lat[m], snap.biomass[m]
    ...  # (rest unchanged)
```

In `choose_live_layer`, add the keyword param + thread it (so the count + both builders are composed):
```python
def choose_live_layer(
    snap: MovementSnapshot, species_filter: str | None, mode: str, *,
    dots_max: int = 1500, stage_filter: int | None = None
) -> tuple[dict, str | None]:
    """... (docstring unchanged) ..."""
    if mode == "dots":
        n = int(_filter_mask(snap, species_filter, stage_filter).sum())
        if n > dots_max:
            note = f"Too many schools for dots ({n}) — showing heatmap"
            return heatmap_layer_from_points(snap, species_filter, stage_filter), note
        return dots_layer_from_points(snap, species_filter, stage_filter), None
    return heatmap_layer_from_points(snap, species_filter, stage_filter), None
```

- [ ] **Step 4: Run — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_live_movement_render.py -q`
Expected: PASS (new + existing render tests; existing positional calls `choose_live_layer(snap, sf, mode)` still work — `stage_filter` defaults to None).

- [ ] **Step 5: Commit**

```bash
git add ui/pages/live_movement_render.py tests/test_live_movement_render.py
git commit -m "feat(live): stage_filter threaded through the render call graph (composed mask + count)"
```

---

### Task 3: Run-page UI — species-from-config, stage selector, horizontal layout, empty-state

**Files:**
- Modify: `ui/pages/run.py` (the Live Movement card ~249-264, a new species-from-config effect, `_render_live_map` ~603-647, `live_movement_status` ~588-602)
- Test: `tests/test_ui_run.py` (a unit test for the pure species-choices helper) + `import app` smoke

**Interfaces:**
- Consumes: `STAGE_LABELS` (Task 1), `choose_live_layer(..., stage_filter=...)` (Task 2).
- Produces: module helper `_species_choices(config: dict[str, str]) -> dict[str, str]`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_ui_run.py`:

```python
def test_species_choices_from_config():
    from ui.pages.run import _species_choices
    cfg = {"simulation.nspecies": "2", "species.name.sp0": "Cod", "species.name.sp1": "Sprat"}
    assert _species_choices(cfg) == {"__all__": "All species", "Cod": "Cod", "Sprat": "Sprat"}
    # empty / missing config -> just the all-option
    assert _species_choices({}) == {"__all__": "All species"}
```

- [ ] **Step 2: Run — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ui_run.py -k species_choices -v`
Expected: FAIL (`_species_choices` not defined).

- [ ] **Step 3a: Add the pure helper** at module scope in `ui/pages/run.py` (near the top, after imports):

```python
def _species_choices(config: dict[str, str]) -> dict[str, str]:
    """Live-movement species dropdown choices from a flat config dict (focal species)."""
    choices = {"__all__": "All species"}
    try:
        n = int(float(config.get("simulation.nspecies", 0) or 0))
    except (ValueError, TypeError):
        n = 0
    for i in range(n):
        name = config.get(f"species.name.sp{i}")
        if name:
            choices[name] = name
    return choices
```

- [ ] **Step 3b: Horizontal layout + stage selector.** In the Live Movement `ui.card` (run.py ~249-264), replace the `input_radio_buttons(...mode...)` + `input_select(...species...)` block with a `layout_columns` wrapping mode + species + stage:

```python
        ui.card(
            body_collapse_header(
                "Live Movement (Python engine) — expand to stream during a run",
                "run_live_movement",
            ),
            ui.layout_columns(
                ui.input_radio_buttons(
                    "live_movement_mode", "Mode",
                    {"heatmap": "Heatmap", "dots": "Dots"}, selected="heatmap", inline=True,
                ),
                ui.input_select(
                    "live_movement_species", "Species", choices={"__all__": "All species"}
                ),
                ui.input_select(
                    "live_movement_stage", "Stage",
                    choices={"__all__": "All stages", **{str(k): v for k, v in STAGE_LABELS.items()}},
                ),
                col_widths=[3, 5, 4],
            ),
            ui.output_ui("live_movement_status"),
            live_map.ui(height="420px"),
        ),
```
Add the import at the top of `run.py` if not present: `from osmose.live_movement import STAGE_LABELS` (check existing imports — `live_movement` symbols may already be imported; add `STAGE_LABELS` to that import).

- [ ] **Step 3c: Populate species from config.** Add a new reactive effect in the server (next to `_populate_live_species`, ~569):

```python
    @reactive.effect
    def _populate_species_from_config():
        if not _session_alive[0]:
            return
        ui.update_select("live_movement_species", choices=_species_choices(state.config.get()))
```

- [ ] **Step 3d: Empty-state hint.** In `live_movement_status` (~588-602), update the no-run python-engine message to mention filtering. Replace:
```python
        return ui.p("Expand this card before running to stream movement.", class_="text-muted")
```
with:
```python
        return ui.p(
            "Expand this card and run the model to stream movement, "
            "then filter by species / stage.",
            class_="text-muted",
        )
```

- [ ] **Step 3e: Thread stage_filter into the render effect.** In `_render_live_map` (~603-647), after the `species_filter` line, add the stage read, and pass it to `choose_live_layer`:
```python
            sel = input.live_movement_species()
            species_filter = None if sel in ("__all__", None) else sel
            stage_sel = input.live_movement_stage()
            stage_filter = None if stage_sel in ("__all__", None) else int(stage_sel)
            ...
            layer, note = choose_live_layer(snap, species_filter, mode, stage_filter=stage_filter)
```

- [ ] **Step 4: Run — verify it passes + app imports**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ui_run.py -k species_choices -q`
Expected: PASS.
Run: `PYTHONPATH=. .venv/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`.

- [ ] **Step 5: Full check + commit.** Run the touched suites + scoped lint:
Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_live_movement.py tests/test_live_movement_render.py tests/test_ui_run.py -q && .venv/bin/ruff check osmose/live_movement.py ui/pages/live_movement_render.py ui/pages/run.py tests/test_live_movement.py tests/test_live_movement_render.py tests/test_ui_run.py && .venv/bin/ruff format --check osmose/live_movement.py ui/pages/live_movement_render.py ui/pages/run.py`
Expected: pass + clean.
```bash
git add ui/pages/run.py tests/test_ui_run.py
git commit -m "feat(run): Live Movement species-from-config + stage selector + 3-col layout + empty-state"
```

---

## Notes for the executor

- **No engine-dynamics change** — only `build_snapshot` (output-side read) + the UI/renderer change; EEC/BoB parity is untouched (no parity run needed, but `import app` must stay green).
- The species fix makes the dropdown usable immediately; **selecting only changes the map when a snapshot exists** (during a stream / retained final frame, card expanded). The empty-state hint (Step 3d) owns the no-run case — this is intended, not a bug.
- Existing positional callers of `choose_live_layer`/`_filter_mask` keep working because `stage_filter` defaults to `None`.
- If `run.py` already imports from `osmose.live_movement`, extend that import with `STAGE_LABELS` rather than adding a second import line.
