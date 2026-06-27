# Live Movement pane: egg/larva on spawning grounds + run button + phenology dates — design

> Status: design (awaiting review) · 2026-06-27
> Three improvements to the Live Movement pane: (A) make the "Egg/larva" stage filter show the
> spawning grounds (eggs are created unlocated and were always dropped); (B) add a Run button inside
> the pane, left of the Mode selector; (C) show a calendar date in the execution info so frames can
> be correlated with spawning/recruitment phenology. (A) plumbs `map_sets` to the live observer
> (output-side only — no engine-dynamics change); B/C are UI/output-side.

## 1. Problem

- **(A)** Eggs/larvae are created at `cell=-1` (unlocated — reproduction.py:184, simulate.py:599) and only
  get a cell after aging past first-feeding into juveniles. `build_snapshot`'s mask drops `cell<0`, so the
  "Egg/larva" filter (`stage==0`, `is_egg`) is **structurally empty for every species** (verified: a 2-yr
  Baltic run has 0 located egg/larva schools at every step, while 25–50 cod eggs sit unlocated). Selecting
  "cod" + "Egg/larva" → blank.
- **(B)** The only Run control is at the top of the Run page; while watching the Live Movement card the user
  must scroll up to start a run.
- **(C)** The status shows `step 49/96` — no calendar context, so the user can't tell which season a frame is
  in to follow phenological events (spawning, recruitment).

## 2. Approach

### 2.1 (A) Place unlocated eggs on the spawning grounds

**Plumb `map_sets` to the live observer (output-side).** `simulate()` already builds
`map_sets: dict[int, MovementMapSet]`. Thread it to the observer:
- `osmose/engine/simulate.py:1709`: `step_observer(step, state, grid, config)` → `step_observer(step, state, grid, config, map_sets)`.
- `make_run_observer`'s inner `observer(step, state, grid, config)` → `(…, map_sets=None)`, delegating
  `live_observer(step, state, grid, config, map_sets)`.
- `make_step_observer`'s inner `observer(step, state, grid, config)` → `(…, map_sets=None)`, calling
  `build_snapshot(step, state, grid, config, map_sets=map_sets, …)`.
- All gain a `map_sets=None` default → backward-compatible; the engine dynamics path is untouched (the
  observer is purely the live-viz hook).

**`build_snapshot` egg placement (pure, before the existing mask):**
- Copy `cx = state.cell_x.copy()`, `cy = state.cell_y.copy()`, `is_out_local = state.is_out.copy()`.
- Select unlocated focal living eggs: `egg = (state.species_id < n_species) & state.is_egg & (state.cell_x < 0)
  & (state.biomass > 0.0)`.
- For each such school `i` (species `sp = species_id[i]`): pick a target cell deterministically (no RNG,
  seeded by `i` — same style as the existing dots jitter):
  - if `map_sets` and `sp in map_sets` and `m = map_sets[sp].get_map(int(state.age_dt[i]), step)` is not
    None and has a positive cell → `valid = np.argwhere(m > 0)`; `j, i_x = valid[(i * 2654435761) % len(valid)]`.
    (Unlocated eggs are age≈0, so this is the egg/larval-stage map; using the school's own `age_dt` is robust
    if an early larva is still unlocated.)
  - else (random-movement species / no map / empty map) → `valid = np.argwhere(grid.ocean_mask)`; same pick.
  - set `cx[i]=i_x`, `cy[i]=j`, `is_out_local[i]=False`.
- The existing selection mask then uses `cx`/`cy`/`is_out_local` instead of `state.cell_*`/`state.is_out`, so
  the placed eggs pass it; their `stage` is `0` (`is_egg`). So **"Egg/larva" shows the spawning area** during the
  spawning season, and the "All species/All stages" view now also includes the (tiny-biomass) eggs there.
- Determinism: index-seeded selection, no RNG (reproducible frames, matches the file's existing convention).
  Vectorize where reasonable, but a Python loop over the (≤ low-hundreds) unlocated eggs is acceptable.

### 2.2 (B) Run button in the pane

- Add `ui.input_action_button("btn_run_live", "▶ Run", class_="btn-success")` as the FIRST child of the Live
  Movement card's `layout_columns`, so the row is **Run · Mode · Species · Stage** with `col_widths=[2, 2, 4, 4]`.
- Wire it to the existing run: `@reactive.event(input.btn_run, input.btn_run_live)` on `handle_run`.
- Mirror the button disable/enable: every `ui.update_action_button("btn_run", disabled=…, session=session)`
  in run.py (run start → disabled=True; completion/cancel/error → disabled=False) gets a sibling call for
  `"btn_run_live"`. (A 2-line `_set_run_buttons(disabled, session)` helper keeps the spots DRY.)

### 2.3 (C) Phenology date in the execution info

- `build_snapshot` computes a pure `date_label: str` and the snapshot carries it (new field).
- `ndt = max(1, config.n_steps // max(1, config.n_year))` (Baltic = 1200/50 = 24; stays correct under a
  `nyear` override since both scale). `year = step // ndt` (0-based index); within-year day-of-year
  `doy = int((step % ndt) / ndt * 365)`; map to a month/day via a fixed non-leap reference year
  (`datetime(2001,1,1) + timedelta(days=doy)`), formatted `"%d %b"`.
- `date_label = f"Y{year} · {dt:%d %b}"` → e.g. `"Y2 · 11 Mar"`.
- `live_movement_status` shows it: `f"{status_v} · {date_label} · step {step+1}/{n_steps}{extra}{suffix}"`.
- No real calendar base year exists in the bundled configs (`output.start.year` absent) → the year is a
  0-based run-relative index; the month is what matters for phenology.

## 3. Data flow

frame → `simulate` calls `step_observer(step, state, grid, config, map_sets)` → `make_run_observer` (progress
+ delegate) → `make_step_observer` → `build_snapshot(…, map_sets)`: places unlocated eggs on the species'
egg-stage map (or ocean fallback), computes `stage` (placed eggs = 0) + `date_label` → `MovementSnapshot`.
The render/legend/filter are unchanged — "Egg/larva" now has stage-0 schools to show; the status shows the date.

## 4. Edge cases

- **No `map_sets`** (e.g. a caller that doesn't pass it / Java path doesn't use this) → `map_sets=None` →
  eggs placed on ocean cells (still visible). Backward-compatible.
- **A species' egg map is None / all-zero** at this step (out of spawning season for a seasonal map) → ocean
  fallback (rare; most unlocated eggs only exist during spawning, when the map is non-empty).
- **No unlocated eggs** (off-season) → nothing placed; "Egg/larva" legitimately empty (the spawning event is
  over). The date label tells the user it's off-season.
- **`grid.ocean_mask` empty** (degenerate grid) → no valid cell; skip placement for that school (it stays
  dropped). Guard `len(valid) > 0`.
- **`n_year == 0`** (shouldn't happen) → `max(1, …)` guards divide-by-zero; `date_label` degrades gracefully.

## 5. Testing

- **Unit (`osmose/live_movement.py`):**
  - egg placement: a snapshot with an unlocated `is_egg` school of a maps-species + a stub `map_sets` whose
    `get_map(0, step)` returns a grid with one positive cell → the school appears in the snapshot at that cell
    with `stage==0`; deterministic (two builds → identical placement).
  - random fallback: unlocated egg with `map_sets=None` (or no map) → placed on an `ocean_mask` cell, `stage==0`.
  - off-season: no unlocated eggs → snapshot has no stage-0 schools (no crash).
  - `date_label`: `step=0 → "Y0 · 01 Jan"`; `step = ndt//2 → "Y0 · ~02 Jul"` (mid-year); `step = ndt → "Y1 · 01 Jan"`
    (year rollover). (Assert the year + month, tolerant of day rounding.)
- **Wiring:** `import app`; the existing `tests/test_e2e_live_movement.py` still passes (legend/filter unchanged).
- **(B) run button:** an e2e or import-app check that `#btn_run_live` exists and triggers a run (reuse/extend
  the existing e2e: click `#btn_run_live` instead of `#btn_run`, assert the run starts).
- **No engine-dynamics change** — only the observer call gains an arg (output-side); `build_snapshot` is a
  read-only state consumer (copies arrays, never mutates `state`). EEC/BoB parity untouched.

## 6. Out of scope

- Changing where the engine actually places eggs (a dynamics change) — placement is snapshot-only (viz).
- Distributing eggs by map probability weight (uniform pick over `>0` cells is enough for a qualitative view).
- A real calendar base year / leap-year handling (run-relative `Y{n}` + month is sufficient for phenology).
- Per-species spawning-season overlays beyond the egg dots themselves.
