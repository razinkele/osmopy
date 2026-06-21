# Run-Page Rework Implementation Plan (Python default · compact UI · live-stream crash fix)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Python the default engine; compact the Run page (slim busy indicator, Live Movement collapsed by default); and make the live-stream crash impossible (bound client load + heatmap fallback + session-teardown hardening).

**Architecture:** The crash root cause (systematic-debugging, prod stack trace) is unbounded client dots-streaming + no session-teardown guard (`DestroyedReactiveError` cascade). Fix = card-expanded becomes the server-readable stream gate, dots auto-fall-back to heatmap above a threshold, lower `dot_cap`/throttle, and `session.on_ended` cancels the run + `_session_alive` guards stop the polls. Server build is already proven robust.

**Tech Stack:** Python 3.12, Shiny for Python, shiny_deckgl, pytest + Playwright. No new deps.

---

## File Structure

- **Modify:** `osmose/live_movement.py` (`dot_cap` 5000→2000), `ui/pages/live_movement_render.py` (`choose_live_layer` helper), `ui/state.py` (engine default), `app.py` (localStorage default; `loading_overlay` compaction; `toggleCardBody`/`restoreCardBodies` → `live_view_expanded` input + default-collapsed `run_live_movement`), `www/osmose.css` (compact busy indicator), `ui/pages/run.py` (remove switch + auto-enable; `live_view_expanded` gate + 0.5s throttle; `choose_live_layer`; `session.on_ended` + `_session_alive` guards).
- **Tests:** `tests/test_live_movement_render.py`, `tests/test_state_engine.py` (engine default), `tests/test_ui_run_capability.py` (delete the stale auto-enable test), `tests/test_ui_run.py` (drop the `live_movement_view` source assertion), `tests/test_e2e_live_movement.py`, `tests/test_e2e_baltic.py`.

Run tests with `.venv/bin/python -m pytest`. Lint `.venv/bin/ruff check` + `--check` format.

---

## Task 1: `choose_live_layer` + lower `dot_cap` (pure, testable)

**Files:** Modify `ui/pages/live_movement_render.py`, `osmose/live_movement.py`; Test `tests/test_live_movement_render.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_movement_render.py
import numpy as np

from osmose.live_movement import MovementSnapshot
from ui.pages.live_movement_render import choose_live_layer


def _snap(n, n_species=8):
    rng = np.random.default_rng(0)
    sp = rng.integers(0, n_species, n).astype(np.int32)
    sp[: n // 2] = 0  # half are species 0 ("cod")
    return MovementSnapshot(
        step=1, n_steps=10, status="running",
        species=["cod", "herring", "sprat", "flounder", "perch", "pikeperch", "smelt", "stickleback"][:n_species],
        sp_id=sp,
        lon=rng.uniform(10, 30, n).astype(np.float64),
        lat=rng.uniform(54, 66, n).astype(np.float64),
        biomass=rng.uniform(1e-3, 1e3, n).astype(np.float64),
        truncated=False, n_total=n,
        lon_min=10.0, lon_max=30.0, lat_min=54.0, lat_max=66.0, lon_step=0.4, lat_step=0.3,
    )


def test_dots_below_threshold_renders_dots():
    layer, note = choose_live_layer(_snap(200), None, "dots", dots_max=1500)
    assert layer["type"] == "ScatterplotLayer"
    assert note is None


def test_dots_above_threshold_falls_back_to_heatmap():
    # 2000 points, all species (filter None) -> > 1500 -> heatmap + note
    layer, note = choose_live_layer(_snap(2000), None, "dots", dots_max=1500)
    assert layer["type"] == "HeatmapLayer"
    assert note is not None and "heatmap" in note.lower()


def test_heatmap_mode_always_heatmap():
    layer, note = choose_live_layer(_snap(3000), None, "heatmap", dots_max=1500)
    assert layer["type"] == "HeatmapLayer"
    assert note is None


def test_filter_reduces_count_so_dots_stays_dots():
    # 2400 pts but only ~half are cod (~1200 < 1500) -> dots kept for the cod filter
    layer, note = choose_live_layer(_snap(2400), "cod", "dots", dots_max=1500)
    assert layer["type"] == "ScatterplotLayer"
    assert note is None


def test_dot_cap_default_is_2000():
    # The LIVE path's cap is make_step_observer's default, which is what reaches
    # build_snapshot at runtime (it passes dot_cap through). Assert THAT default.
    from osmose.live_movement import make_step_observer
    import inspect

    assert inspect.signature(make_step_observer).parameters["dot_cap"].default == 2000
```

(Verified: shiny_deckgl layer dicts carry the layer kind under the key `"type"` — values `"ScatterplotLayer"`/`"HeatmapLayer"` — NOT `@@type`. The `@@=` prefix is only for accessor strings like `getPosition`.)

- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_live_movement_render.py -q` → ImportError (`choose_live_layer` absent).

- [ ] **Step 3: Implement.**

In `ui/pages/live_movement_render.py`, add (it can use the module-private `_filter_mask`):

```python
def choose_live_layer(
    snap: MovementSnapshot, species_filter: str | None, mode: str, *, dots_max: int = 1500
) -> tuple[dict, str | None]:
    """Pick the live layer, returning (layer, note).

    Dots (per-point ScatterplotLayer) only when the FILTERED point count <= dots_max;
    above that, fall back to the aggregated heatmap, because thousands of per-point dots
    streamed every frame over a long run exhaust the browser's WebGL/memory (the diagnosed
    live-stream crash). Heatmap mode is always heatmap.
    """
    if mode == "dots":
        n = int(_filter_mask(snap, species_filter).sum())
        if n > dots_max:
            note = f"Too many schools for dots ({n}) — showing heatmap"
            return heatmap_layer_from_points(snap, species_filter), note
        return dots_layer_from_points(snap, species_filter), None
    return heatmap_layer_from_points(snap, species_filter), None
```

In `osmose/live_movement.py`, change `build_snapshot`'s `dot_cap` default `5000` → `2000` (line ~63), AND — critically — change **`make_step_observer`'s own `dot_cap` default `5000` → `2000` (line ~113)**: `make_step_observer` passes its `dot_cap` explicitly into `build_snapshot(..., dot_cap=dot_cap)` (line ~132), so it *overrides* build_snapshot's signature default on the live path. Changing only build_snapshot's default would have **no runtime effect** (the streamed cap would stay 5000). Update both docstrings to say 2000. (The run.py call site `make_step_observer(_live_queue, throttle_s=0.5)` then inherits the new 2000 default — no extra arg needed.)

- [ ] **Step 4: Run, verify PASS.** `.venv/bin/python -m pytest tests/test_live_movement_render.py -q`. ruff + pyright clean on both files.

- [ ] **Step 5: Commit.**
```bash
git add ui/pages/live_movement_render.py osmose/live_movement.py tests/test_live_movement_render.py
git commit -m "feat(run): choose_live_layer (dots->heatmap fallback) + dot_cap 5000->2000"
```

---

## Task 2: Python default engine

**Files:** Modify `ui/state.py`, `app.py`; Test `tests/test_state_engine.py`.

- [ ] **Step 1: Update the failing test.** The engine-default assertion lives in **`tests/test_state_engine.py:8-11`** (`test_engine_mode_default_is_java` → `assert state.engine_mode.get() == "java"`), NOT in `tests/test_state.py` (which has no engine_mode assertion). Flip it to `== "python"` and rename it `test_engine_mode_default_is_python`; keep `test_engine_mode_can_be_set_to_python` as-is. Run it → FAIL (still "java" in source).

- [ ] **Step 2: Implement.**
- `ui/state.py:64`: `self.engine_mode: reactive.Value[str] = reactive.Value("java")` → `reactive.Value("python")`.
- `app.py` (engine init, ~line 480): `var savedEngine = localStorage.getItem('osmose-engine') || 'java';` → `|| 'python';`. (The button-active logic in `setEngineMode` already follows the resolved mode — no other change.)

- [ ] **Step 3: Run, verify PASS** + `.venv/bin/python -c "import app"`. Grep for any other `'java'`-default assumption in tests (`grep -rn "engine_mode" tests/`) and fix.

- [ ] **Step 4: Commit.**
```bash
git add ui/state.py app.py tests/test_state_engine.py
git commit -m "feat(run): default the UI engine to Python (Java still selectable)"
```

---

## Task 3: Compact the busy indicator

**Files:** Modify `app.py` (`loading_overlay`), `www/osmose.css`.

- [ ] **Step 1: Implement** (CSS/markup only — no behavior change; verify by `import app` + a visual glance).

Replace `loading_overlay` (`app.py:632-640`) markup with a compact corner pill:

```python
    @render.ui
    def loading_overlay():
        msg = state.busy.get()
        if msg is None:
            return ui.div()
        return ui.div(
            ui.div(class_="osm-spinner osm-spinner-sm"),
            ui.span(msg),
            class_="osm-busy-pill",
        )
```

In `www/osmose.css`, add a compact bottom-right pill (replacing reliance on the large `osm-loading-overlay`):

```css
/* Compact busy indicator (replaces the full-screen loading overlay) */
.osm-busy-pill {
  position: fixed;
  bottom: 1rem;
  right: 1rem;
  z-index: 1080;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  background: rgba(20, 30, 45, 0.92);
  color: #e8eef5;
  border-radius: 999px;
  font-size: 0.85rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
}
.osm-spinner-sm { width: 1rem; height: 1rem; border-width: 2px; }
```

(Keep the existing `.osm-spinner` keyframes; `osm-spinner-sm` just shrinks it. Leave the old `.osm-loading-overlay` rule in place — now unused — or remove it; do not break other callers, so grep `osm-loading-overlay` first: it should be only this render.)

- [ ] **Step 2: Verify** `.venv/bin/python -c "import app"` clean; `grep -rn "osm-loading-overlay" ui/ app.py www/` shows no remaining user besides the (now-removed) render.

- [ ] **Step 3: Commit.**
```bash
git add app.py www/osmose.css
git commit -m "feat(run): compact busy indicator (corner pill, not full overlay)"
```

---

## Task 4: Live card = stream gate (remove switch/auto-enable; expanded-gated; heatmap fallback; throttle)

**Files:** Modify `app.py` (`toggleCardBody`/`restoreCardBodies`), `ui/pages/run.py`.

- [ ] **Step 1: Update the tests (delete the now-stale ones; add the new gate test).**

  **(i) DELETE** the obsolete `test_run_page_auto_enables_live_for_spatial` (`tests/test_ui_run_capability.py:35-39`) — it asserts `"def _auto_enable_live_for_spatial" in text` and `'update_switch("live_movement_view"' in text`, both of which this task removes from run.py (it would otherwise turn the Task-7 full-suite gate red and directly contradict the new test below).

  **(ii) FIX** `tests/test_ui_run.py:109` (`assert "live_movement_view" in src` inside the wired-into-run-page test) — the `live_movement_view` switch is removed; replace that assertion with `assert "live_view_expanded" in src` (the new gate), or drop the `live_movement_view` line. (The plan must touch this file — it's in the default suite and would otherwise go red.)

  **(iii) APPEND** to `tests/test_ui_run_capability.py`:
```python
def test_live_view_uses_expand_gate_not_switch():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "input.live_movement_view" not in text       # switch gate removed
    assert "live_view_expanded" in text                 # expand gate present
    assert "_auto_enable_live_for_spatial" not in text  # superseded
    assert "choose_live_layer" in text                  # heatmap fallback wired
```

- [ ] **Step 2: Run, verify FAIL.** (The new test fails; the deleted ones are gone.)

- [ ] **Step 3: Implement.**

(a) `app.py` — publish `live_view_expanded` from the live card's collapse toggle. In `toggleCardBody` (after computing `collapsed`), add:
```javascript
            if (panelId === 'run_live_movement' && typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                Shiny.setInputValue('live_view_expanded', !collapsed);
            }
```
In `restoreCardBodies` (after `card.classList.toggle('osm-body-collapsed', want)`), add — so the server has the value on connect AND the live card defaults collapsed when unset:
```javascript
                if (panelId === 'run_live_movement' && typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                    Shiny.setInputValue('live_view_expanded', !want);
                }
```
Default-collapsed for `run_live_movement`: in `restoreCardBodies`, when no stored preference exists for that panel, treat it as collapsed:
```javascript
                var stored = localStorage.getItem('osmose-card-collapsed-' + panelId);
                var want = stored === null ? (panelId === 'run_live_movement') : (stored === '1');
```
(Replaces the existing `var want = localStorage.getItem(...) === '1';` line.)

(b) `ui/pages/run.py` `run_ui()` — in the Live Movement card, **remove** the `ui.input_switch("live_movement_view", …)` line. Keep mode/species/status/map. (The card header's collapse button is the on/off control now.) **Discoverability:** since the card defaults collapsed, the body (and any hint inside it) is hidden — so put the "expand to stream" hint in the **card header title** where it's always visible: change `body_collapse_header("Live Movement (Python engine)", "run_live_movement")` to `body_collapse_header("Live Movement (Python engine) — expand to stream during a run", "run_live_movement")`. (A hint `ui.p(...)` inside the body would be invisible when collapsed, defeating the point.)
Also fix the now-stale guidance text in `live_movement_status` (run.py:~556): change `"Enable the toggle before running to stream movement."` → `"Expand this card before running to stream movement."`

(c) `ui/pages/run.py` `run_server()` — **remove** the `_auto_enable_live_for_spatial` effect and the `_last_spatial` cell (added in the prior feature). Remove the now-unused `config_is_spatial` import if nothing else uses it (grep first).

(d) `handle_run` — change the live-stream gate from the switch to the expand input, and raise the throttle. Replace the `if input.live_movement_view() and engine_mode == "python":` block's gate. The block currently (lines ~739-754) drains `_live_queue`, resets snapshot/framed/species, sets status "running", and `live_observer = make_step_observer(_live_queue)`. Change:
```python
        live_observer = None
        try:
            _live_expanded = bool(input.live_view_expanded())
        except Exception:
            _live_expanded = False  # input unset (card never toggled) -> collapsed
        if _live_expanded and engine_mode == "python":
            while True:
                try:
                    _live_queue.get_nowait()
                except queue.Empty:
                    break
            _live_snapshot.set(None)
            _live_framed[0] = False
            _last_live_species[0] = None
            _live_status_val.set("running")
            live_observer = make_step_observer(_live_queue, throttle_s=0.5)  # ≤2 fps (was 0.2)
```

(e) `_render_live_map` — gate on expanded, use `choose_live_layer`, surface the note. Add a reactive note value near the other live values:
```python
    _live_note: reactive.Value = reactive.Value(None)
```
At the TOP of `_render_live_map`, add an expand gate so collapsing the card *mid-run* also pauses client rendering (symmetric with the run-start gate; avoids streaming to a hidden canvas):
```python
        try:
            if not input.live_view_expanded():
                return
        except Exception:
            return  # input unset -> collapsed -> nothing to render
```
Update the import to add `choose_live_layer`, and replace the `layer = (dots… if mode=="dots" else heatmap…)` block with:
```python
        layer, note = choose_live_layer(snap, species_filter, mode)
        _live_note.set(note)
```
Show the note in `live_movement_status` (append when set):
```python
        note = _live_note.get()
        suffix = f" · {note}" if note else ""
        return ui.p(f"{status_v} {prog}{extra}{suffix}".strip())
```

- [ ] **Step 4: Run, verify PASS** (`tests/test_ui_run_capability.py`) + `.venv/bin/python -c "import app"`.

- [ ] **Step 5: Commit.**
```bash
git add app.py ui/pages/run.py tests/test_ui_run_capability.py tests/test_ui_run.py
git commit -m "feat(run): live card collapse = stream gate; heatmap fallback; 0.5s throttle; drop switch/auto-enable"
```

---

## Task 5: Server hardening against session teardown

**Files:** Modify `ui/pages/run.py`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_ui_run_capability.py`:
```python
def test_run_server_hardens_against_session_teardown():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "session.on_ended" in text
    assert "_session_alive" in text
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement** in `run_server()`.

(a) Near the top of `run_server` (with the other plain cells), add:
```python
    _session_alive = [True]
    _active_cancel_token: list = [None]  # plain ref so on_ended needn't read a reactive
```
(b) In `handle_run`, where `cancel_token = threading.Event()` is created, also record it:
```python
            cancel_token = threading.Event()
            state.run_cancel_token.set(cancel_token)
            _active_cancel_token[0] = cancel_token
```
(c) Register the teardown handler (anywhere in `run_server` body, e.g. just after the cells):
```python
    def _on_session_end():
        _session_alive[0] = False
        tok = _active_cancel_token[0]
        if tok is not None:
            tok.set()  # stop the daemon engine thread instead of running against a dead session

    session.on_ended(_on_session_end)
```
(d) Guard the reactive consumers: `_drain_run_done`, `_drain_live_queue`, `_drain_progress`, `_render_live_map`, **and `_populate_live_species`** (it also calls `ui.update_select(...)`, a session op — guard it for symmetry). At the TOP of each: `if not _session_alive[0]: return`. Then wrap the body so a teardown **race** (flag flips after the early-return, mid-body) can't cascade — but **do NOT silently swallow real bugs during a live session.** Two precise rules:

1. **Catch `BaseException`, not `Exception`** — `asyncio.CancelledError` (the cascade named in the diagnosis) derives from `BaseException`, so `except Exception` would NOT catch it. Use `except BaseException` (or `(Exception, asyncio.CancelledError)`; `import asyncio`).
2. **Only swallow when the session is actually ending; otherwise re-raise** — so a genuine bug in `_handle_result` (which `_drain_run_done` calls — it sets `state.output_dir` etc.) is NOT masked during a normal run.

Pattern (apply to each; for `_render_live_map` wrap the ENTIRE body after the guard, **including the `await _live_map.set_style(...)` in the style branch** — that await precedes the `if snap is None: return` and can itself hit the teardown race):
```python
    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_live_queue():
        if not _session_alive[0]:
            return
        try:
            ... existing body ...
        except BaseException:  # noqa: BLE001
            if _session_alive[0]:
                raise  # genuine bug during a live session — surface it
            _log.debug("live poll skipped (session ending)", exc_info=True)
```
(`import asyncio` at the top of run.py if using the `(Exception, asyncio.CancelledError)` form.)

- [ ] **Step 4: Run, verify PASS** + `import app` clean. (The guards are source-asserted; behavior is exercised by the e2e + manual teardown.)

- [ ] **Step 5: Commit.**
```bash
git add ui/pages/run.py tests/test_ui_run_capability.py
git commit -m "fix(run): session.on_ended cancels the run + _session_alive guards (no DestroyedReactiveError cascade)"
```

---

## Task 6: Update e2e (expand card instead of the removed switch)

**Files:** Modify `tests/test_e2e_live_movement.py`, `tests/test_e2e_baltic.py`.

The `#live_movement_view` switch no longer exists; expanding the Live Movement card is now the stream gate. The card defaults collapsed, so the live tests must expand it (click its collapse toggle) BEFORE Start Run.

- [ ] **Step 1: Update the live tests.** In both `test_live_movement_renders_during_python_run`/`test_live_movement_cancel_path` (`test_e2e_live_movement.py`) and the Baltic live case (`test_e2e_baltic.py`): the only `#live_movement_view` references are `expect(...).to_be_checked()` assertions (lines 58/96 and baltic:57 — there is NO `.click()` to remove). **Replace each `to_be_checked` assertion** with a robust **expand** of the live card. The collapse button toggles state, so a blind `.click()` could COLLAPSE it (race with `restoreCardBodies`); instead click only if not already expanded, then assert it's expanded via the published input:
```python
    # Expand the Live Movement card (collapsed by default) to enable streaming.
    card = page.locator('.card:has(button[data-osm-card-toggle="run_live_movement"])')
    btn = page.locator('button[data-osm-card-toggle="run_live_movement"]')
    if "osm-body-collapsed" in (card.get_attribute("class") or ""):
        btn.click()
    expect(card).not_to_have_class("osm-body-collapsed")  # expanded
    page.wait_for_timeout(250)  # let Shiny.setInputValue('live_view_expanded', true) round-trip
    page.locator("#btn_run").click()
```
Keep the running→done assertions on `#live_movement_status`. The default engine is now Python, so the `#engineBtnPython` click is harmless/defensive — leave it. **Also refresh the now-stale comments** in `test_e2e_live_movement.py:44-49` ("engine defaults to Java", "REQUIRED not defensive", wrong app.py:195/201 button cites) and `test_e2e_baltic.py:56` to reflect the Python default + the expand-gate (button ids are at `app.py:279`/`:285`).

- [ ] **Step 2: The plain-run progress test** (`test_run_progress_shows_during_python_run`) is unaffected (it never touches the live view; progress is independent of the live gate) — leave it, but it now runs on the Python default automatically (the `#engineBtnPython` click is still fine).

- [ ] **Step 3: Run the e2e.** `.venv/bin/python -m pytest tests/test_e2e_live_movement.py tests/test_e2e_baltic.py -m e2e -v`. Expect PASS (expanding the card enables streaming → running→done). If Playwright unavailable, note it and fall back to source-edit confirmation.

- [ ] **Step 4: Commit.**
```bash
git add tests/test_e2e_live_movement.py tests/test_e2e_baltic.py
git commit -m "test(e2e): expand live card (new stream gate) instead of the removed switch"
```

---

## Task 7: Full suite + lint/format/pyright + e2e smoke

**Files:** none (verification only).

- [ ] **Step 1:** `.venv/bin/python -m pytest -q -n auto` — all pass.
- [ ] **Step 2:** `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/` — clean (format + recommit if needed).
- [ ] **Step 3:** `.venv/bin/python -m pyright --pythonpath .venv/bin/python osmose/live_movement.py ui/pages/run.py ui/pages/live_movement_render.py` — 0 errors.
- [ ] **Step 4:** `.venv/bin/python -m pytest tests/test_e2e_live_movement.py -m e2e -v` — the live stream + progress work end-to-end.
- [ ] **Step 5:** Commit any fixups.

---

## Self-Review notes (applied)

- **Spec coverage:** A (Python default) → Task 2; B (compact busy + collapsed live card) → Task 3 + Task 4(a); C (crash fix) → Task 1 (dot_cap + heatmap fallback) + Task 4(d,e) (expand gate + throttle) + Task 5 (session hardening). e2e → Task 6.
- **Supersedes same-day work:** Tasks 4(b,c)/6 remove the `live_movement_view` switch + `_auto_enable_live_for_spatial` effect shipped earlier today (`f1682ae`) and re-point the e2e — intentional, per the spec.
- **No private import:** the teardown guard uses `_session_alive` + broad `except` (NOT `DestroyedReactiveError`, which isn't publicly exported).
- **`live_view_expanded` default:** read via try/except → False when unset (card never toggled), so a run with the card collapsed simply doesn't stream (progress still does).
- **Out of scope:** expanding the live card *mid-run to start* streaming (start-gate is evaluated at run start; collapsing mid-run DOES pause render via the `_render_live_map` expand-gate). Java live path (none exists).
- **Documented residual (acceptable):** an explicitly-expanded, long-running dots view of a *small* filtered set (≤1500 pts, so the heatmap fallback doesn't fire) is still bounded only by the 0.5s throttle + the 2000 dot_cap (≈6× less cumulative geometry than before). The reported crash is fully covered regardless: the live card is collapsed-by-default (no stream unless opened), and `session.on_ended` + the `_session_alive`/`BaseException` guards make the `DestroyedReactiveError`/`CancelledError` cascade impossible even if the tab does die. A hard per-run frame cap is a possible future belt-and-suspenders, not needed here.
- **Round-2 workflow-review fixes folded in:** `make_step_observer` dot_cap (not just build_snapshot) lowered to 2000; layer test key `@@type`→`type`; Task 2 retargeted to `tests/test_state_engine.py`; stale `test_run_page_auto_enables_live_for_spatial` deleted + `test_ui_run.py:109` updated; `CancelledError` caught (BaseException); poll except re-raises when session alive (no masking `_handle_result`); `_populate_live_species` guarded; e2e expand made race-safe; header-title hint for discoverability.
