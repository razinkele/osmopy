# Live Movement pane: egg/larva on spawning grounds + run button + phenology dates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the "Egg/larva" filter show the current spawning cloud (place unlocated eggs on the species' egg-stage map), add a Run button in the Live Movement pane, and show a phenology date in the execution status.

**Architecture:** `build_snapshot` places unlocated `is_egg` schools onto the species' egg-stage `MovementMapSet` map (probability-weighted, ocean fallback) and computes a date label; `map_sets` is threaded to the live observer (output-side). `run.py` adds the button + the date in the status. No engine-dynamics change.

**Tech Stack:** Python 3.12, NumPy, pytest, Shiny, shiny_deckgl, `osmose.engine.movement_maps.MovementMapSet`.

## Global Constraints

- **Run python via the MAIN venv ABSOLUTE path** `/home/razinka/osmose/osmose-python/.venv/bin/python` — do NOT create/symlink a `.venv` in the worktree. Tests: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest <path> -v`. Lint: `/home/razinka/osmose/osmose-python/.venv/bin/ruff check` + `ruff format --check`.
- **Eggs are a per-frame spawning CLOUD**, re-sampled each frame — NOT stable per-egg dots (eggs are unlocated ~1 step; `state.compact()` re-indexes between frames). Determinism is **within a single `build_snapshot` call only**; make no cross-frame stability claim/test.
- Egg placement is **probability-weighted** over `m>0` cells (concentrate where the engine concentrates spawners); ocean (`grid.ocean_mask`) uniform fallback when no/empty map. Deterministic per-egg seed `(i * 2654435761)`, no RNG. Guard `len(cells) > 0`.
- `build_snapshot` is **read-only** on the frozen `state` — copy `cell_x/cell_y/is_out` into locals; never mutate `state`.
- Observer `map_sets` is passed **positionally**: `step_observer(step, state, grid, config, map_sets)`. Factory observers get `map_sets=None` defaults; the 4-arg test lambda at `tests/test_engine_simulate.py:350` must accept it.
- Date: `ndt = max(1, int(config.n_dt_per_year))` (the field **already exists** — config.py:1225; do NOT derive from `n_steps//n_year`). `year = step // ndt + 1` (**1-based**). `date_label = f"Y{year} · {dt:%d %b}"`, `dt = datetime(2001,1,1)+timedelta(days=int((step%ndt)/ndt*365))`. `MovementSnapshot.date_label: str = ""` (defaulted → existing `_snap` test fixtures unaffected).
- Status shows `date_label` from `snap`, guarded by the existing `snap is None` check; keep `prog`.
- Run button: `btn_run_live` is the FIRST child of the pane's `layout_columns` (`col_widths=[2,2,4,4]` → Run·Mode·Species·Stage); `handle_run` fires on `@reactive.event(input.btn_run, input.btn_run_live)`; both buttons toggled via a `_set_run_buttons(disabled)` helper.
- No engine-dynamics change — observer is output-side; EEC/BoB parity untouched.
- Spec: `docs/superpowers/specs/2026-06-27-live-pane-eggs-runbtn-dates-design.md`.

---

### Task 1: `build_snapshot` egg placement + date label + observer `map_sets` plumbing

**Files:**
- Modify: `osmose/live_movement.py` (`MovementSnapshot`, `build_snapshot`, `make_step_observer`, `make_run_observer`, module imports)
- Test: `tests/test_live_movement.py` (extend `_config` + new tests)

**Interfaces:**
- Produces: `build_snapshot(step, state, grid, config, *, map_sets=None, status="running", dot_cap=2000)`; `MovementSnapshot.date_label: str`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_live_movement.py`, extend the `_config` fixture to add `n_dt_per_year` and append the tests:

```python
# in _config(...), add a kwarg `n_dt_per_year=12` and include it in the SimpleNamespace:
#     n_dt_per_year=12,  (kwarg)
#     n_dt_per_year=n_dt_per_year,  (in the returned SimpleNamespace)


class _StubMap:
    def __init__(self, grid):
        self._grid = grid

    def get_map(self, age_dt, step):
        return self._grid


def test_unlocated_egg_placed_on_spawning_map():
    from osmose.live_movement import build_snapshot
    g = Grid.from_dimensions(ny=3, nx=3)
    # one located adult (idx0) + one unlocated egg of sp0 (idx1, cell=-1, is_egg)
    st = _state(
        species_id=[0, 0], cell_x=[1, -1], cell_y=[1, -1], biomass=[5.0, 0.1],
        is_egg=[False, True], length=[50.0, 0.1], age_dt=[30, 0],
    )
    m = np.zeros((3, 3), dtype=np.float64)
    m[2, 0] = 1.0  # only (row=2, col=0) is a spawning cell
    snap = build_snapshot(0, st, g, _config(n_species=1), map_sets={0: _StubMap(m)})
    # the egg is now in the snapshot at the spawning cell with stage 0
    assert 0 in snap.stage.tolist()  # egg/larva present
    egg_i = snap.stage.tolist().index(0)
    lat_arr, lon_arr = (np.arange(3.0), np.arange(3.0))
    assert snap.lon[egg_i] == lon_arr[0] and snap.lat[egg_i] == lat_arr[2]  # placed at (col0,row2)
    # within-build deterministic
    snap2 = build_snapshot(0, st, g, _config(n_species=1), map_sets={0: _StubMap(m)})
    assert snap2.lon.tolist() == snap.lon.tolist()


def test_egg_placement_probability_weighted():
    from osmose.live_movement import build_snapshot
    g = Grid.from_dimensions(ny=1, nx=3)
    n = 60
    st = _state(
        species_id=[0] * n, cell_x=[-1] * n, cell_y=[-1] * n, biomass=[0.1] * n,
        is_egg=[True] * n, length=[0.1] * n, age_dt=[0] * n,
    )
    m = np.array([[0.01, 0.01, 5.0]], dtype=np.float64)  # cell (0,2) dominates
    snap = build_snapshot(0, st, g, _config(n_species=1), map_sets={0: _StubMap(m)})
    # most eggs land on the high-proba cell (col=2)
    from collections import Counter
    modal_col = Counter(snap.lon.tolist()).most_common(1)[0][0]
    assert modal_col == 2.0


def test_egg_random_fallback_no_map():
    from osmose.live_movement import build_snapshot
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[-1], cell_y=[-1], biomass=[0.1],
                is_egg=[True], length=[0.1], age_dt=[0])
    snap = build_snapshot(0, st, g, _config(n_species=1), map_sets=None)
    assert snap.stage.tolist() == [0]  # placed on an ocean cell, stage 0


def test_date_label_one_based_year():
    from osmose.live_movement import build_snapshot
    g = Grid.from_dimensions(ny=2, nx=2)
    st = _state(species_id=[0], cell_x=[0], cell_y=[0], biomass=[5.0],
                is_egg=[False], length=[50.0], age_dt=[30])
    cfg = _config(n_species=1, n_dt_per_year=24)
    assert build_snapshot(0, st, g, cfg).date_label == "Y1 · 01 Jan"
    assert build_snapshot(24, st, g, cfg).date_label.startswith("Y2 · 01 Jan")
    assert build_snapshot(12, st, g, cfg).date_label.startswith("Y1 · ")  # mid-year
```

- [ ] **Step 2: Run — verify it fails**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_live_movement.py -k "egg or date_label" -v`
Expected: FAIL (`build_snapshot` has no `map_sets`; `date_label` missing).

- [ ] **Step 3: Implement** in `osmose/live_movement.py`:

(a) Add the import near the top (with the other stdlib imports):
```python
from datetime import datetime, timedelta
```

(b) Add the field to `MovementSnapshot` as the LAST field (defaulted, so existing constructions are unaffected) — after `lat_step: float`:
```python
    date_label: str = ""  # phenology label e.g. "Y2 · 11 Mar"; "" if not computed
```

(c) Change `build_snapshot`'s signature to accept `map_sets`:
```python
def build_snapshot(
    step: int, state, grid, config, *, map_sets=None, status: str = "running", dot_cap: int = 2000
) -> MovementSnapshot:
```

(d) Replace the body from `lat_arr, lon_arr = resolve_grid_latlon(grid)` through the `stage = np.where(...)` line with (places eggs, then builds the mask on the local arrays):
```python
    lat_arr, lon_arr = resolve_grid_latlon(grid)
    # Place unlocated eggs (cell=-1) on the species' egg-stage spawning map so "Egg/larva"
    # shows the current spawning cloud (eggs are created unlocated and were always dropped).
    # Read-only on the frozen state: work on local copies.
    cx_full = state.cell_x.copy()
    cy_full = state.cell_y.copy()
    is_out_full = state.is_out.copy()
    egg = (
        (state.species_id < config.n_species)
        & state.is_egg
        & (state.cell_x < 0)
        & (state.biomass > 0.0)
    )
    if egg.any():
        ocean = getattr(grid, "ocean_mask", None)
        ocean_cells = np.argwhere(ocean) if ocean is not None else np.empty((0, 2), dtype=np.intp)
        for i in np.nonzero(egg)[0]:
            i = int(i)
            sp = int(state.species_id[i])
            m = None
            if map_sets is not None and sp in map_sets:
                m = map_sets[sp].get_map(int(state.age_dt[i]), int(step))
            if m is not None and m.max() > 0:
                cells = np.argwhere(m > 0)
                w = m[m > 0].astype(np.float64)
                cdf = np.cumsum(w)
                cdf /= cdf[-1]
                frac = ((i * 2654435761) % 10_000) / 10_000.0
                k = min(int(np.searchsorted(cdf, frac)), len(cells) - 1)
                j, ix = cells[k]
            elif len(ocean_cells) > 0:
                j, ix = ocean_cells[(i * 2654435761) % len(ocean_cells)]
            else:
                continue  # no valid cell — leave dropped
            cx_full[i] = ix
            cy_full[i] = j
            is_out_full[i] = False
    mask = (
        (state.species_id < config.n_species)
        & ~is_out_full
        & (cx_full >= 0)
        & (cy_full >= 0)
        & (state.biomass > 0.0)
    )
    sp_id = state.species_id[mask]
    cx = cx_full[mask]
    cy = cy_full[mask]
    bm = state.biomass[mask]
    # Life stage: egg/larva if is_egg; adult if mature; else juvenile.
    is_egg_m = state.is_egg[mask]
    mature_m = (state.length[mask] >= config.maturity_size[sp_id]) & (
        state.age_dt[mask] >= config.maturity_age_dt[sp_id]
    )
    stage = np.where(is_egg_m, 0, np.where(mature_m, 2, 1)).astype(np.int8)
```

(e) Compute `date_label` just before the `return MovementSnapshot(`:
```python
    ndt = max(1, int(getattr(config, "n_dt_per_year", 0)) or 1)
    year = step // ndt + 1
    doy = int((step % ndt) / ndt * 365)
    date_label = f"Y{year} · {datetime(2001, 1, 1) + timedelta(days=doy):%d %b}"
```

(f) Pass `date_label=date_label` in the `MovementSnapshot(...)` return (add after `lat_step=lat_step,`):
```python
        date_label=date_label,
```

(g) Thread `map_sets` through the two observers. In `make_step_observer`'s inner `observer`:
```python
    def observer(step: int, state, grid, config, map_sets=None) -> None:
```
and its `build_snapshot` call:
```python
            snap = build_snapshot(step, state, grid, config, map_sets=map_sets, dot_cap=dot_cap)
```
In `make_run_observer`'s inner `observer`:
```python
    def observer(step: int, state, grid, config, map_sets=None) -> None:
```
and its delegate call:
```python
            if live_observer is not None:
                live_observer(step, state, grid, config, map_sets)
```

- [ ] **Step 4: Run — verify it passes**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_live_movement.py -q`
Expected: PASS (new egg/date tests + existing build_snapshot/observer tests; the defaulted `date_label` keeps other snapshot constructions valid).

- [ ] **Step 5: Commit**

```bash
git add osmose/live_movement.py tests/test_live_movement.py
git commit -m "feat(live): place unlocated eggs on spawning grounds + phenology date_label + map_sets observer arg"
```

---

### Task 2: thread `map_sets` from the engine to the observer

**Files:**
- Modify: `osmose/engine/simulate.py` (the `step_observer(...)` call)
- Modify: `tests/test_engine_simulate.py` (the 4-arg observer lambda at line 350)

**Interfaces:**
- Consumes: the observer signature `(step, state, grid, config, map_sets)` (Task 1).

- [ ] **Step 1: Change the observer call (introduces the break).** In `osmose/engine/simulate.py`, the observer call (~line 1710):
```python
        if step_observer is not None:
            step_observer(step, state, grid, config, map_sets)
```
(`map_sets` is the `dict[int, MovementMapSet]` already in scope, built ~line 1427.)

- [ ] **Step 2: Run — see the strict lambda break**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_engine_simulate.py -k observer -q`
Expected: FAIL — the 4-arg lambda at `tests/test_engine_simulate.py:350` raises `TypeError: <lambda>() takes 4 positional arguments but 5 were given` (the `lambda *a: None` observers are unaffected).

- [ ] **Step 3: Fix the strict observer lambda** at `tests/test_engine_simulate.py:350` to accept the new positional arg:
```python
        step_observer=lambda step, state, g, c, map_sets=None: calls.append((step, c.n_steps)),
```

- [ ] **Step 4: Run — verify the suite passes**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_engine_simulate.py -q`
Expected: PASS — all observer lambdas now accept the 5th positional arg; no `TypeError`.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_simulate.py
git commit -m "feat(engine): pass map_sets to the step observer (output-side; enables egg placement)"
```

---

### Task 3: Run button in the pane + phenology date in the status

**Files:**
- Modify: `ui/pages/run.py` (live card `layout_columns`, `handle_run` decorator, `_set_run_buttons` helper + button toggles, `live_movement_status`)
- Test: `import app` + extend `tests/test_e2e_live_movement.py`

**Interfaces:**
- Consumes: `MovementSnapshot.date_label` (Task 1).

- [ ] **Step 1: Add the Run button to the pane** — in the Live Movement card's `layout_columns` (run.py ~269), insert as the FIRST child (before the Mode `input_radio_buttons`) and change `col_widths`:
```python
            ui.layout_columns(
                ui.input_action_button("btn_run_live", "▶ Run", class_="btn-success"),
                ui.input_radio_buttons(
                    "live_movement_mode",
                    "Mode",
                    {"heatmap": "Heatmap", "dots": "Dots"},
                    selected="heatmap",
                    inline=True,
                ),
                ui.input_select(
                    "live_movement_species", "Species", choices={"__all__": "All species"}
                ),
                ui.input_select(
                    "live_movement_stage",
                    "Stage",
                    choices={
                        "__all__": "All stages",
                        **{str(k): v for k, v in STAGE_LABELS.items()},
                    },
                ),
                col_widths=[2, 2, 4, 4],
            ),
```

- [ ] **Step 2: Fire `handle_run` from either button** — change the decorator (run.py ~795):
```python
    @reactive.event(input.btn_run, input.btn_run_live)
    async def handle_run():
```

- [ ] **Step 3: Add the `_set_run_buttons` helper + toggle both buttons.** Define the helper in the server scope (e.g. just above `handle_run`):
```python
    def _set_run_buttons(disabled: bool) -> None:
        ui.update_action_button("btn_run", disabled=disabled, session=session)
        ui.update_action_button("btn_run_live", disabled=disabled, session=session)
```
Then replace the existing `btn_run` toggles (both occurrences-by-value) so `btn_run_live` is kept in sync:
- Replace ALL `ui.update_action_button("btn_run", disabled=False, session=session)` → `_set_run_buttons(False)` (4 sites: run.py 355, 370, 403, 536).
- Replace `ui.update_action_button("btn_run", disabled=True, session=session)` → `_set_run_buttons(True)` (1 site: run.py 837).
(The `btn_cancel` toggles are unchanged.)

- [ ] **Step 4: Show the phenology date in the status** — in `live_movement_status`, change the final return (run.py ~637):
```python
        note = _live_note.get()
        suffix = f" · {note}" if note else ""
        date = f" · {snap.date_label}" if snap is not None and snap.date_label else ""
        return ui.p(f"{status_v}{date} {prog}{extra}{suffix}".strip())
```

- [ ] **Step 5: Verify app imports + lint**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -c "import app; print('app imports OK')"`
Expected: `app imports OK`.
Run: `/home/razinka/osmose/osmose-python/.venv/bin/ruff check ui/pages/run.py && /home/razinka/osmose/osmose-python/.venv/bin/ruff format --check ui/pages/run.py`
Expected: clean.

- [ ] **Step 6: Extend the e2e to click the pane Run button.** In `tests/test_e2e_live_movement.py`, in `test_live_movement_renders_during_python_run`, replace the existing `page.locator("#btn_run").click()` with the pane button (it triggers the same run):
```python
    page.locator("#btn_run_live").click()
```
(The rest of the test — status reaching "running"/"done", the dots/legend assertions — is unchanged and now also exercises `btn_run_live` + the date label in the status.)

- [ ] **Step 7: Run the e2e**

Run: `PYTHONPATH=. /home/razinka/osmose/osmose-python/.venv/bin/python -m pytest tests/test_e2e_live_movement.py::test_live_movement_renders_during_python_run -m e2e -q`
Expected: PASS — the pane Run button starts the run; status reaches "running"/"done"; zero deck errors. (Requires the `viztest` extra: `pip install -e ".[viztest]"` if playwright is absent.)

- [ ] **Step 8: Commit**

```bash
git add ui/pages/run.py tests/test_e2e_live_movement.py
git commit -m "feat(run): Run button in the Live Movement pane + phenology date in the status"
```

---

## Notes for the executor

- **No engine-dynamics change** — Task 2 only adds an arg to the output-side observer call; `build_snapshot` copies arrays and never mutates `state`. `import app` + parity stay green.
- Eggs are a **per-frame cloud** — do not add a cross-frame stability test; the within-build determinism test is the correct guard.
- Do not create a `.venv` in the worktree; always use the main-venv absolute path.
- If `ui.input_action_button("btn_run_live", …)` width looks cramped at `col_widths=[2,2,4,4]`, that is acceptable for this pass (a compact button); do not redesign the layout.
