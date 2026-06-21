# Run-Page Rework Implementation Plan (Python default · compact UI · live-stream crash fix)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Python the default engine; compact the Run page (slim busy indicator, Live Movement collapsed by default); and make the live-stream crash impossible (bound client load + heatmap fallback + session-teardown hardening).

**Architecture:** The crash root cause (systematic-debugging, prod stack trace) is unbounded client dots-streaming + no session-teardown guard (`DestroyedReactiveError` cascade). Fix = card-expanded becomes the server-readable stream gate, dots auto-fall-back to heatmap above a threshold, lower `dot_cap`/throttle, and `session.on_ended` cancels the run + `_session_alive` guards stop the polls. Server build is already proven robust.

**Tech Stack:** Python 3.12, Shiny for Python, shiny_deckgl, pytest + Playwright. No new deps.

---

## File Structure

- **Modify:** `osmose/live_movement.py` (`dot_cap` 5000→2000), `ui/pages/live_movement_render.py` (`choose_live_layer` helper), `ui/state.py` (engine default), `app.py` (localStorage default; `loading_overlay` compaction; `toggleCardBody`/`restoreCardBodies` → `live_view_expanded` input + default-collapsed `run_live_movement`), `www/osmose.css` (compact busy indicator), `ui/pages/run.py` (remove switch + auto-enable; `live_view_expanded` gate + 0.5s throttle; `choose_live_layer`; `session.on_ended` + `_session_alive` guards).
- **Tests:** `tests/test_live_movement_render.py`, `tests/test_state.py`, `tests/test_ui_run_capability.py`, `tests/test_e2e_live_movement.py`, `tests/test_e2e_baltic.py`.

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
    assert layer["@@type"] == "ScatterplotLayer" or "Scatterplot" in str(layer.get("@@type", ""))
    assert note is None


def test_dots_above_threshold_falls_back_to_heatmap():
    # 2000 points, all species (filter None) -> > 1500 -> heatmap + note
    layer, note = choose_live_layer(_snap(2000), None, "dots", dots_max=1500)
    assert "Heatmap" in str(layer.get("@@type", ""))
    assert note is not None and "heatmap" in note.lower()


def test_heatmap_mode_always_heatmap():
    layer, note = choose_live_layer(_snap(3000), None, "heatmap", dots_max=1500)
    assert "Heatmap" in str(layer.get("@@type", ""))
    assert note is None


def test_filter_reduces_count_so_dots_stays_dots():
    # 2400 pts but only ~half are cod (~1200 < 1500) -> dots kept for the cod filter
    layer, note = choose_live_layer(_snap(2400), "cod", "dots", dots_max=1500)
    assert "Scatterplot" in str(layer.get("@@type", ""))
    assert note is None


def test_dot_cap_default_is_2000():
    from osmose.live_movement import build_snapshot
    import inspect

    assert inspect.signature(build_snapshot).parameters["dot_cap"].default == 2000
```

(Note: `@@type` is shiny_deckgl's layer-type key. If the actual key differs, assert on the builder identity instead — the test's intent is "dots below threshold, heatmap above".)

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

In `osmose/live_movement.py`, change `build_snapshot`'s `dot_cap` default `5000` → `2000` (line ~63) and update the docstring's "Samples to ``dot_cap``" note to mention 2000.

- [ ] **Step 4: Run, verify PASS.** `.venv/bin/python -m pytest tests/test_live_movement_render.py -q`. ruff + pyright clean on both files.

- [ ] **Step 5: Commit.**
```bash
git add ui/pages/live_movement_render.py osmose/live_movement.py tests/test_live_movement_render.py
git commit -m "feat(run): choose_live_layer (dots->heatmap fallback) + dot_cap 5000->2000"
```

---

## Task 2: Python default engine

**Files:** Modify `ui/state.py`, `app.py`; Test `tests/test_state.py`.

- [ ] **Step 1: Update the failing test.** In `tests/test_state.py`, the engine-default assertion (currently expects `"java"`) → expect `"python"`. Run it → FAIL (still "java").

- [ ] **Step 2: Implement.**
- `ui/state.py:64`: `self.engine_mode: reactive.Value[str] = reactive.Value("java")` → `reactive.Value("python")`.
- `app.py` (engine init, ~line 480): `var savedEngine = localStorage.getItem('osmose-engine') || 'java';` → `|| 'python';`. (The button-active logic in `setEngineMode` already follows the resolved mode — no other change.)

- [ ] **Step 3: Run, verify PASS** + `.venv/bin/python -c "import app"`. Grep for any other `'java'`-default assumption in tests (`grep -rn "engine_mode" tests/`) and fix.

- [ ] **Step 4: Commit.**
```bash
git add ui/state.py app.py tests/test_state.py
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

- [ ] **Step 1: Write the failing test.** Append to `tests/test_ui_run_capability.py`:
```python
def test_live_view_uses_expand_gate_not_switch():
    text = open(run_page.__file__, encoding="utf-8").read()
    assert "input.live_movement_view" not in text       # switch gate removed
    assert "live_view_expanded" in text                 # expand gate present
    assert "_auto_enable_live_for_spatial" not in text  # superseded
    assert "choose_live_layer" in text                  # heatmap fallback wired
```

- [ ] **Step 2: Run, verify FAIL.**

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

(b) `ui/pages/run.py` `run_ui()` — in the Live Movement card, **remove** the `ui.input_switch("live_movement_view", …)` line. Keep mode/species/status/map. (The card header's collapse button is the on/off control now.) Add a short hint line where the switch was:
```python
            ui.p("Expand this card before running to stream movement (Python engine).",
                 class_="text-muted small"),
```

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

(e) `_render_live_map` — use `choose_live_layer` and surface the note. Add a reactive note value near the other live values:
```python
    _live_note: reactive.Value = reactive.Value(None)
```
Update the import to add `choose_live_layer`, and in `_render_live_map` replace the `layer = (dots… if mode=="dots" else heatmap…)` block with:
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
git add app.py ui/pages/run.py tests/test_ui_run_capability.py
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
(d) Guard the three poll functions and the render. At the TOP of `_drain_run_done`, `_drain_live_queue`, `_drain_progress`, and `_render_live_map`, add the early-return; wrap each body's reactive work in a broad try/except so a teardown race can't cascade. Pattern (apply to each):
```python
    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_live_queue():
        if not _session_alive[0]:
            return
        try:
            ... existing body ...
        except Exception:  # noqa: BLE001 — post-teardown reactive access is best-effort
            _log.debug("live poll skipped (session ending)", exc_info=True)
```
For the async `_render_live_map`, the same `if not _session_alive[0]: return` at the top and a `try/except Exception` around the `await` calls.

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

- [ ] **Step 1: Update the live tests.** In both `test_live_movement_renders_during_python_run`/`test_live_movement_cancel_path` (`test_e2e_live_movement.py`) and the Baltic live case (`test_e2e_baltic.py`): replace the `expect(#live_movement_view).to_be_checked()` / removed `.click()` lines with expanding the live card and asserting the input. The live card's collapse button is `button.osm-collapse-btn[data-osm-card-toggle="run_live_movement"]`:
```python
    # Expand the Live Movement card (collapsed by default) to enable streaming.
    page.locator('button[data-osm-card-toggle="run_live_movement"]').click()
    page.wait_for_timeout(200)  # let Shiny.setInputValue('live_view_expanded', true) round-trip
    page.locator("#btn_run").click()
```
Keep the running→done assertions on `#live_movement_status`. The default-engine is now Python, so the `#engineBtnPython` click becomes a no-op/defensive (leave it — harmless).

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
- **Out of scope:** expanding the live card *mid-run* (gate is evaluated at run start); Java live path (none exists).
