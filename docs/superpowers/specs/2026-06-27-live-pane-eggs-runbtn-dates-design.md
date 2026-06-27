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
- `osmose/engine/simulate.py:1710`: `step_observer(step, state, grid, config)` → `step_observer(step, state, grid, config, map_sets)` (pass **positionally**).
- `make_run_observer`'s inner `observer(step, state, grid, config)` → `(…, map_sets=None)`, delegating
  `live_observer(step, state, grid, config, map_sets)`.
- `make_step_observer`'s inner `observer(step, state, grid, config)` → `(…, map_sets=None)`, calling
  `build_snapshot(step, state, grid, config, map_sets=map_sets, …)`.
- The factory observers gain a `map_sets=None` default. **But any user-supplied observer now receives a 5th
  positional arg:** the strict 4-arg lambda at `tests/test_engine_simulate.py:350`
  (`lambda step, state, g, c: …`) must be updated to `lambda step, state, g, c, map_sets=None: …` (the
  `lambda *a: None` observers at :362/:385 already accept it). The engine dynamics path is untouched — the
  observer is purely the live-viz hook.

**Framing — a per-frame spawning *cloud*, NOT stable per-egg dots.** Eggs are created at `cell=-1`, the
observer fires, then `state.compact()` (simulate.py:~1723) removes dead schools and re-indexes the array; an
egg is unlocated+`is_egg` for only ~1 step before it ages/hatches or `movement` assigns it a real cell. So
each observed frame shows a *different cohort* of just-spawned eggs, and a school's array index is NOT stable
across frames. The feature therefore renders the **current spawning cloud** (the egg-stage map lit up where
spawning is happening this step), re-sampled each frame — it does NOT track individual eggs across frames, and
we make no cross-frame reproducibility claim. Within a single `build_snapshot` call the placement is
deterministic (so one frame is stable to re-render under filter/mode changes), which is all the render needs.

**`build_snapshot` egg placement (pure, before the existing mask):**
- Copy `cx = state.cell_x.copy()`, `cy = state.cell_y.copy()`, `is_out_local = state.is_out.copy()` (never
  mutate the frozen `state`; its `__post_init__` also forbids `cell<0` on a rebuilt state).
- Select unlocated focal living eggs: `egg = (state.species_id < n_species) & state.is_egg & (state.cell_x < 0)
  & (state.biomass > 0.0)`.
- For each such school at array index `i` (species `sp = species_id[i]`): pick a target cell deterministically
  (no RNG, seeded by `i`), **probability-weighted** so the cloud concentrates where the engine concentrates
  spawners (movement.py accepts a cell with prob ∝ its map value — uniform-over-positive-cells would skew a
  peaked map toward its low-proba tail):
  - if `map_sets` and `sp in map_sets` and `m = map_sets[sp].get_map(int(state.age_dt[i]), step)` is not None
    and `m.max() > 0` → `cells = np.argwhere(m > 0)`; `w = m[m > 0]`; `cdf = np.cumsum(w); cdf /= cdf[-1]`;
    `frac = ((i * 2654435761) % 10_000) / 10_000.0`; `k = min(int(np.searchsorted(cdf, frac)), len(cells)-1)`;
    `j, i_x = cells[k]`. (Unlocated eggs are age≈0 → the egg/larval-stage map; the school's own `age_dt` is robust
    if an early larva is still unlocated.)
  - else (random-movement species / no map / empty map) → `cells = np.argwhere(grid.ocean_mask)` (uniform, no
    proba); `j, i_x = cells[(i * 2654435761) % len(cells)]`. Guard `len(cells) > 0` (else skip this school).
  - set `cx[i]=i_x`, `cy[i]=j`, `is_out_local[i]=False`.
- The existing selection mask then uses `cx`/`cy`/`is_out_local` instead of `state.cell_*`/`state.is_out`, so
  the placed eggs pass it; their `stage` is `0` (`is_egg`). So **"Egg/larva" shows the current spawning cloud**.
- A Python loop over the (≤ low-hundreds) unlocated eggs is acceptable; no RNG.

**Interaction with the existing views (eggs stay in all stages — they are real schools):** placed eggs also
appear in the "All stages" dots/heatmap, which is correct (they are part of the population, shown tiny because
the heatmap is biomass-weighted and dots are biomass-sized, and eggs have `egg_size` biomass). Cap interplay,
clarified (these are two different caps): the dots-vs-heatmap decision uses **`dots_max=1500`** on the
*filtered* count (`live_movement_render.py:choose_live_layer`); `build_snapshot`'s **`dot_cap=2000`** is the
per-frame sampling bound. All-species already exceeds 1500 (→ heatmap) so adding tens–hundreds of eggs does not
flip it; a single-species view stays well under 1500, so eggs render as dots there without a fallback flip. The
only effect is that eggs consume a few of the 2000 `dot_cap` sample slots in the dense All view — negligible.

### 2.2 (B) Run button in the pane

- Add `ui.input_action_button("btn_run_live", "▶ Run", class_="btn-success")` as the FIRST child of the Live
  Movement card's `layout_columns`, so the row is **Run · Mode · Species · Stage** with `col_widths=[2, 2, 4, 4]`.
- Wire it to the existing run: `@reactive.event(input.btn_run, input.btn_run_live)` on `handle_run`.
- Mirror the button disable/enable: every `ui.update_action_button("btn_run", disabled=…, session=session)`
  in run.py (run start → disabled=True; completion/cancel/error → disabled=False) gets a sibling call for
  `"btn_run_live"`. (A 2-line `_set_run_buttons(disabled, session)` helper keeps the spots DRY.)

### 2.3 (C) Phenology date in the execution info

- Add a `date_label: str` field to `MovementSnapshot` (the dataclass at `live_movement.py:28-51`);
  `build_snapshot` computes it (pure).
- `ndt = max(1, int(config.n_dt_per_year))` — **use the existing `EngineConfig.n_dt_per_year`** (config.py:1225;
  Baltic = 24); do NOT re-derive from `n_steps // n_year` (it already exists and doesn't assume exact
  divisibility). `year = step // ndt + 1` (**1-based**, matching the existing `format_progress_label`'s
  "Year 1/1" convention — `live_movement.py:219`); within-year day-of-year `doy = int((step % ndt) / ndt * 365)`
  (0..364); map to a month/day via a fixed non-leap reference year (`datetime(2001, 1, 1) + timedelta(days=doy)`),
  formatted `"%d %b"`.
- `date_label = f"Y{year} · {dt:%d %b}"` → e.g. `"Y2 · 11 Mar"`.
- `live_movement_status` shows it **from the snapshot, guarded like the existing `prog`** (which is only built
  when `snap is not None`): set `date = f" · {snap.date_label}" if snap is not None else ""` and fold it into
  the existing return — `f"{status_v}{date} {prog}{extra}{suffix}".strip()` (do not drop the current
  `snap is None` guard or the separately-computed `prog`).
- No real calendar base year exists in the bundled configs (`output.start.year` absent) → the year is a 1-based
  run-relative index; the month is what matters for phenology.

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
- **Retained final frame after a run** is one moment (the last step). If the run ends off-season, "Egg/larva" on
  that retained frame is empty even though spawning happened earlier — expected; the date label shows the season,
  and watching the live stream (or the dots through the run) shows the spawning cloud appear/disappear.
- **`grid.ocean_mask` empty** (degenerate grid) → no valid cell; skip placement for that school (it stays
  dropped). Guard `len(cells) > 0`.
- **`n_dt_per_year == 0`** (shouldn't happen) → `max(1, int(config.n_dt_per_year))` guards divide-by-zero;
  `date_label` degrades gracefully.

## 5. Testing

- **Unit (`osmose/live_movement.py`):**
  - egg placement on a map: a stub `state` with an unlocated `is_egg` school + a stub `map_sets` whose
    `get_map(...)` returns a grid with a positive cell → the school appears in the snapshot at a `m>0` cell with
    `stage==0`. Determinism is **within a single build** (re-running the SAME build gives the SAME placement) —
    do NOT assert cross-frame stability (eggs are a per-frame cloud; §2.1).
  - proba-weighting: a stub map with one high-proba cell and several low-proba cells, many unlocated eggs → the
    placed cells concentrate on the high-proba cell (assert the modal cell is the high-proba one).
  - random fallback: unlocated egg with `map_sets=None` (or no/empty map) → placed on a `grid.ocean_mask` cell,
    `stage==0`; `ocean_mask` all-False → school skipped (no crash).
  - off-season: no unlocated eggs → snapshot has no stage-0 schools (no crash).
  - `date_label` (1-based year): `step=0 → "Y1 · 01 Jan"`; `step = ndt//2 → "Y1 · ~02 Jul"` (mid-year);
    `step = ndt → "Y2 · 01 Jan"` (year rollover). (Assert the year + month, tolerant of day rounding.)
- **Observer signature:** update the 4-arg observer lambda at `tests/test_engine_simulate.py:350` to accept
  `map_sets=None`; the `lambda *a: None` observers already pass. Run `tests/test_engine_simulate.py` green.
- **Wiring:** `import app`; the existing `tests/test_e2e_live_movement.py` still passes (legend/filter unchanged).
- **(B) run button:** extend the existing e2e to click `#btn_run_live` (instead of/in addition to `#btn_run`) and
  assert the run starts (status reaches "running"/"done").
- **No engine-dynamics change** — the observer call gains a positional arg (output-side); `build_snapshot` is a
  read-only state consumer (copies arrays into locals, never mutates the frozen `state`). EEC/BoB parity untouched.

## 6. Out of scope

- Changing where the engine actually places eggs (a dynamics change) — placement is snapshot-only (viz).
- Distributing eggs by map probability weight (uniform pick over `>0` cells is enough for a qualitative view).
- A real calendar base year / leap-year handling (run-relative `Y{n}` + month is sufficient for phenology).
- Per-species spawning-season overlays beyond the egg dots themselves.
