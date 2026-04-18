# Calibration UI Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface three existing calibration-library capabilities on the Shiny UI — a distinct red-banner error state for `PreflightEvalError`, an `n_workers` numeric input for the preflight evaluator, and a Pareto/Weighted toggle with a scatter+table view on the surrogate optimum panel.

**Architecture:** UI-only, no engine changes. Three independent capabilities, each a single commit. Library touch is one attribute (`.stage` on `PreflightEvalError`). All testing uses the existing `make_multi_input` / `make_catch_all_input` helpers from `tests/helpers.py` and pure-Python helper-extraction seams — no Shiny test client, no Playwright.

**Tech Stack:** Shiny for Python 1.5.1, shinywidgets (plotly), pandas, pytest + existing calibration-UI test helpers.

**Spec:** `docs/superpowers/specs/2026-04-19-calibration-ui-phase3-design.md` (commit `0bbfdfb`).

## Pre-flight

- [ ] Baseline: `.venv/bin/python -m pytest -q` must report 2449 passed.
- [ ] Lint baseline: `.venv/bin/ruff check osmose/ scripts/ tests/ ui/` clean.
- [ ] Sanity-check the live library surface (one-time read, not a repeated check):
  - `osmose/calibration/preflight.py:119` — `class PreflightEvalError(RuntimeError)` (one-line class).
  - `osmose/calibration/preflight.py:459` — the single `raise PreflightEvalError(...)` site (Morris; Sobol has no raise site).
  - `osmose/calibration/preflight.py:665` — `n_workers: int = 1` keyword on `run_preflight`.
  - `osmose/calibration/surrogate.py:123` — `find_optimum(..., weights=None)`.

## File map

- **Library (1 attribute, 1 line each at two sites):**
  - `osmose/calibration/preflight.py` — add `self.stage = stage` to `PreflightEvalError.__init__`; pass `stage="morris"` at the raise site.
- **UI (three capability commits):**
  - `ui/pages/calibration.py` — add inputs, conditional panels, new render functions, new reactive value.
  - `ui/pages/calibration_handlers.py` — add `MsgQueue.post_preflight_error`, helper-extraction functions `_clamp_n_workers`, `_resolve_optimum_weights`, `_build_preflight_error_modal`; wire through the existing `_run_preflight_thread` and `find_optimum` call site.
  - `ui/pages/calibration_charts.py` — reused as-is; `make_pareto_chart` already covers the scatter case.
- **Tests (one file growth, one new file possible):**
  - `tests/test_ui_calibration_handlers.py` — append pure-Python helper tests and modal-render test.
  - `tests/test_ui_calibration.py` — append input-registration assertion.

---

## Task 1: PreflightEvalError red-banner

**Goal:** When `run_preflight` raises `PreflightEvalError`, the preflight modal renders a distinct red alert with the failing stage, the error message, and actionable recovery hints — instead of the generic "Preflight failed" that catches every exception identically.

**Files:**
- Modify: `osmose/calibration/preflight.py:119-135` (class init + one raise site).
- Modify: `ui/pages/calibration_handlers.py:65` (MsgQueue), `:196-255` (modal dispatcher + error helper), `:346-365` (queue-drain branch), `:788-800` (thread catch).
- Test: `tests/test_ui_calibration_handlers.py` (append one test).

- [ ] **Step 1: Library touch — `.stage` attribute on `PreflightEvalError`**

  Read `osmose/calibration/preflight.py:119` first. The class is currently:

  ```python
  class PreflightEvalError(RuntimeError):
      """Raised when >50 % of preflight samples fail to evaluate."""
  ```

  Replace with:

  ```python
  class PreflightEvalError(RuntimeError):
      """Raised when >50 % of preflight samples fail to evaluate.

      The ``stage`` attribute names the preflight stage that failed
      (``"morris"``, ``"sobol"``, or ``"unknown"``) so UI layers can
      render a stage-specific error message without parsing ``str(exc)``.
      """

      def __init__(self, *args, stage: str = "unknown") -> None:
          super().__init__(*args)
          self.stage = stage
  ```

  Then update the one existing raise site at `preflight.py:459` (grep for `raise PreflightEvalError` to confirm there's exactly one). Add `stage="morris"` keyword:

  ```python
  raise PreflightEvalError(
      f"Morris preflight: {n_failed}/{n_total} samples failed to evaluate "
      f"(>{FAILURE_THRESHOLD:.0%} threshold). See {run_dir} for per-sample logs.",
      stage="morris",
  )
  ```

  (Match the existing argument shape; just add the `stage=` kwarg.)

- [ ] **Step 2: Add `post_preflight_error` to `MsgQueue`**

  In `ui/pages/calibration_handlers.py`, after `post_preflight` at line 65-66, add a sibling method:

  ```python
  def post_preflight_error(self, exc) -> None:
      self._q.put(("preflight_error", exc))
  ```

  Do NOT change the signature of `post_preflight` — existing downstream callers depend on its single-argument form.

- [ ] **Step 3: Split `build_preflight_modal` into a dispatcher + success and error helpers**

  Rename the existing body of `build_preflight_modal` (currently at `:196-255`) to `_build_preflight_success_modal(result)` — unchanged logic. Then add a new dispatcher plus an error helper:

  ```python
  def build_preflight_modal(result_or_error):
      """Dispatch on payload: PreflightEvalError → red-banner modal;
      anything else → existing success-shape modal."""
      from osmose.calibration.preflight import PreflightEvalError

      if isinstance(result_or_error, PreflightEvalError):
          return _build_preflight_error_modal(result_or_error)
      return _build_preflight_success_modal(result_or_error)


  def _build_preflight_error_modal(exc):
      """Render a red-banner modal for PreflightEvalError. Dismissal-only —
      no retry button in this iteration."""
      stage = getattr(exc, "stage", "unknown")
      body = ui.div(
          ui.div(
              ui.tags.strong(f"Preflight failed — {stage} stage"),
              ui.br(),
              ui.tags.span(str(exc)),
              class_="alert alert-danger",
              role="alert",
          ),
          ui.p(
              "Recovery: try narrowing parameter bounds, reducing the Morris "
              "sample budget, or re-running with n_workers=1 to see the "
              "underlying sample-evaluation exception.",
              class_="mt-3",
          ),
      )
      footer = ui.tags.button(
          "Close", class_="btn btn-secondary", **{"data-bs-dismiss": "modal"}
      )
      return ui.modal(
          body,
          title="Pre-flight Screening Failed",
          footer=footer,
          easy_close=True,
          size="l",
      )
  ```

- [ ] **Step 4: Wire the `PreflightEvalError` catch in `_run_preflight_thread`**

  At `ui/pages/calibration_handlers.py:788-800` the thread currently has:

  ```python
  def _run_preflight_thread():
      try:
          result = run_preflight(...)                # existing ~line 790
          msg_queue.post_preflight(result)           # existing ~line 796
      except Exception as exc:                       # existing ~line 797
          msg_queue.post_error(str(exc))
  ```

  Insert a specific catch BEFORE the generic one. Also add the import:

  ```python
  from osmose.calibration.preflight import (
      make_preflight_eval_fn,
      PreflightEvalError,   # NEW
      run_preflight,
  )
  ```

  (The existing import is at line 779 — reuse it.)

  ```python
  def _run_preflight_thread():
      try:
          result = run_preflight(...)
          msg_queue.post_preflight(result)
      except PreflightEvalError as exc:              # NEW — before generic
          msg_queue.post_preflight_error(exc)
      except Exception as exc:                       # unchanged
          msg_queue.post_error(str(exc))
  ```

- [ ] **Step 5: Wire the queue-drain dispatch branch**

  At `ui/pages/calibration_handlers.py:346-365`, the drain loop currently ends at:

  ```python
  elif kind == "preflight":
      preflight_result.set(payload)
      modal = build_preflight_modal(payload)
      if modal is not None:
          ui.modal_show(modal)
  ```

  Add a parallel branch immediately after, using the same `build_preflight_modal` entry point (now a dispatcher):

  ```python
  elif kind == "preflight_error":
      # build_preflight_modal dispatches on payload type → error helper
      modal = build_preflight_modal(payload)
      if modal is not None:
          ui.modal_show(modal)
  ```

  Note: we intentionally do NOT set `preflight_result` in the error branch — the reactive stays at its previous value so the user can re-open the success modal if one existed. This is consistent with "dismissal-only; no retry button" from the spec.

- [ ] **Step 6: Write the failing test**

  Append to `tests/test_ui_calibration_handlers.py`:

  ```python
  from osmose.calibration.preflight import PreflightEvalError
  from ui.pages.calibration_handlers import build_preflight_modal


  def test_preflight_modal_renders_error_banner_for_PreflightEvalError():
      """A PreflightEvalError payload must produce a red-banner modal with
      the stage name and the error message — not the generic success-shape
      modal."""
      exc = PreflightEvalError("half the Morris samples failed", stage="morris")
      modal = build_preflight_modal(exc)
      assert modal is not None, "Error modal should render, not return None"
      html = str(modal)
      assert "alert-danger" in html, "Expected red alert banner"
      assert "morris" in html, "Expected stage label in banner"
      assert "half the Morris samples failed" in html, (
          "Expected error message in banner"
      )
      # Sanity: ensure it's NOT the success-shape modal
      assert "Apply Selected & Start" not in html
  ```

- [ ] **Step 7: Run the test (fails until Steps 1-5 are in place)**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/test_ui_calibration_handlers.py::test_preflight_modal_renders_error_banner_for_PreflightEvalError -v
  ```

  Expected: PASS (all implementation steps land before this step; running the test confirms end-to-end).

- [ ] **Step 8: Run full suite + lint**

  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check osmose/ scripts/ tests/ ui/
  ```

  Expected: `2450 passed` (2449 baseline + 1 new); lint clean.

- [ ] **Step 9: Commit**

  ```bash
  git add osmose/calibration/preflight.py ui/pages/calibration_handlers.py tests/test_ui_calibration_handlers.py
  git commit -m "feat(calibration-ui): red-banner modal for PreflightEvalError

  Library touch: .stage attribute on PreflightEvalError populated at the
  Morris raise site (preflight.py:459). Sobol has no raise site today;
  the UI degrades gracefully via getattr(exc, 'stage', 'unknown') if a
  future Sobol gate forgets to set it.

  UI: build_preflight_modal becomes a dispatcher that routes on payload
  type. _build_preflight_error_modal renders a Bootstrap alert-danger
  banner with the stage label, error message, and hardcoded recovery
  hints. Dismissal-only — no retry button.

  Wiring: _run_preflight_thread catches PreflightEvalError before the
  generic Exception; MsgQueue gains post_preflight_error; queue drain
  adds a preflight_error branch that reuses the same entry point.

  Tests: +1 in test_ui_calibration_handlers.py. 2450 passed."
  ```

---

## Task 2: `n_workers` numeric input

**Goal:** Expose the library's `n_workers` parameter on `run_preflight` via a numeric input in the preflight settings panel. Library default 1 preserved — upgrading must not silently change parallelism. Defensive clamp on invalid inputs.

**Files:**
- Modify: `ui/pages/calibration.py:102-106` (preflight panel — add input below `cal_preflight_enabled`).
- Modify: `ui/pages/calibration_handlers.py` (add `_clamp_n_workers` helper, read the input in `_run_preflight_thread` at :788).
- Test: `tests/test_ui_calibration_handlers.py` (4 clamp tests), `tests/test_ui_calibration.py` (1 input-registration test).

- [ ] **Step 1: Add the `_clamp_n_workers` helper**

  In `ui/pages/calibration_handlers.py`, near the existing `_clamp_int` helper at line 78-82, add:

  ```python
  def _clamp_n_workers(requested: int | None, cpu: int | None) -> int:
      """Clamp a user-supplied worker count into [1, max(1, cpu)].

      None or non-positive input → 1 (sequential, the library default).
      ``cpu`` is ``os.cpu_count()`` which returns None on some platforms;
      fall back to 1 in that case.
      """
      ceiling = max(1, cpu) if cpu else 1
      if requested is None:
          return 1
      try:
          n = int(requested)
      except (TypeError, ValueError):
          return 1
      if n < 1:
          return 1
      return min(n, ceiling)
  ```

- [ ] **Step 2: Write the failing tests**

  Append to `tests/test_ui_calibration_handlers.py`:

  ```python
  from ui.pages.calibration_handlers import _clamp_n_workers


  def test_clamp_n_workers_honors_valid_input():
      assert _clamp_n_workers(4, 8) == 4


  def test_clamp_n_workers_clamps_to_cpu_count():
      assert _clamp_n_workers(16, 4) == 4


  def test_clamp_n_workers_defaults_on_invalid():
      assert _clamp_n_workers(None, 8) == 1
      assert _clamp_n_workers(0, 8) == 1
      assert _clamp_n_workers(-1, 8) == 1


  def test_clamp_n_workers_survives_null_cpu_count():
      # Platforms where os.cpu_count() returns None: fall back to 1
      assert _clamp_n_workers(4, None) == 1
  ```

- [ ] **Step 3: Run tests — Step 1 implementation should already pass them**

  ```bash
  .venv/bin/python -m pytest tests/test_ui_calibration_handlers.py -v -k clamp_n_workers
  ```

  Expected: 4 passed.

- [ ] **Step 4: Add the `ui.input_numeric` to the preflight panel**

  At `ui/pages/calibration.py:~100`, the preflight checkbox sits in a settings panel. Read lines 95-115 to see the surrounding structure. Add the `os` import at the top of the file if not already present:

  ```python
  import os  # at top of imports block if not already there
  ```

  Then insert the numeric input immediately after `cal_preflight_enabled` (the checkbox row; grep for `cal_preflight_enabled` to find its exact line):

  ```python
  ui.input_numeric(
      "cal_preflight_workers",
      "Workers",
      value=1,
      min=1,
      max=max(1, os.cpu_count() or 1),
      step=1,
  ),
  ui.help_text(
      "Parallel evaluators for preflight sample runs. "
      "1 = sequential (default).",
      class_="small text-muted",
  ),
  ```

  (If the preflight panel uses a single comma-separated tag list, add the two tags there. If it uses a column layout with `ui.layout_columns`, add them in the same column as `cal_preflight_enabled`.)

- [ ] **Step 5: Wire the clamped value into `_run_preflight_thread`**

  At `ui/pages/calibration_handlers.py:788-796`, `_run_preflight_thread` currently builds a `run_preflight(...)` call. Find the call and inject `n_workers`:

  ```python
  raw_workers = getattr(input, "cal_preflight_workers", lambda: 1)()
  n_workers = _clamp_n_workers(raw_workers, os.cpu_count())
  result = run_preflight(
      ...                    # existing kwargs unchanged
      n_workers=n_workers,   # NEW
  )
  ```

  (Add `import os` at the top of `calibration_handlers.py` if not already present.)

- [ ] **Step 6: Add the input-registration test**

  Append to `tests/test_ui_calibration.py`:

  ```python
  from tests.helpers import make_multi_input
  from ui.pages import calibration as cal_page


  def test_cal_preflight_workers_input_registered():
      """The n_workers input must appear in the rendered calibration UI.
      Guards against silent removal in a future refactor."""
      # Build the page UI; relies on the existing test helpers for any
      # minimal state the page needs (follow the pattern of sibling
      # tests in this file).
      page_ui = cal_page.calibration_ui()  # adjust call to match sibling tests
      html = str(page_ui)
      assert 'id="cal_preflight_workers"' in html or \
             "'cal_preflight_workers'" in html, (
          "cal_preflight_workers input missing from rendered UI"
      )
  ```

  (If `calibration_ui` takes arguments, follow the exact pattern of an existing test in `test_ui_calibration.py` that builds the page. The assertion searches for the input ID in the rendered HTML.)

- [ ] **Step 7: Run tests + full suite + lint**

  ```bash
  .venv/bin/python -m pytest tests/test_ui_calibration_handlers.py tests/test_ui_calibration.py -v
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check osmose/ scripts/ tests/ ui/
  ```

  Expected: new tests pass; `2455 passed` (2450 after Task 1 + 5 new); lint clean.

- [ ] **Step 8: Commit**

  ```bash
  git add ui/pages/calibration.py ui/pages/calibration_handlers.py tests/test_ui_calibration_handlers.py tests/test_ui_calibration.py
  git commit -m "feat(calibration-ui): n_workers numeric input for preflight

  Expose the library's n_workers parameter on run_preflight
  (preflight.py:665) via a ui.input_numeric in the preflight settings
  panel. Default 1 (sequential) preserved — upgrading does not silently
  change parallelism.

  Defensive clamp via _clamp_n_workers (new pure helper in
  calibration_handlers.py): invalid/None/negative inputs → 1;
  requests > os.cpu_count() → cpu_count; os.cpu_count()==None → 1.

  Tests: +4 unit tests for the clamp helper, +1 UI registration test
  that prevents silent removal. 2455 passed."
  ```

---

## Task 3: Pareto / Weighted toggle and view

**Goal:** Replace the unconditional `find_optimum()` call with a mode-aware branch. Expose a `Pareto | Weighted sum` radio; when `Weighted`, render one `input_numeric` per objective and pass `weights=...` into `find_optimum`. Render either a scatter+table (`n_obj ≤ 2`) or table-only (`n_obj ≥ 3`) view for the Pareto set, and a compact single-row summary for the Weighted case.

**Files:**
- Modify: `ui/pages/calibration.py` (radio + conditional weights render, new reactive, two new output regions, one new render-widget function, one render-data_frame, one render-ui for the weighted summary).
- Modify: `ui/pages/calibration_handlers.py` (`_resolve_optimum_weights` helper + mode branch at `:483`).
- Modify: `ui/pages/calibration_charts.py` — reused; no change.
- Test: `tests/test_ui_calibration_handlers.py` (6 new tests).

- [ ] **Step 1: Add the `_resolve_optimum_weights` helper**

  In `ui/pages/calibration_handlers.py`, near the other pure helpers (`_clamp_n_workers`), add:

  ```python
  def _resolve_optimum_weights(input, n_obj: int) -> list[float] | None:
      """Read the weights inputs if mode=='weighted', else return None.

      Returns None for the Pareto path. Returns a list of non-negative
      floats for the Weighted path. If any weight input is not yet
      rendered (SilentException, e.g. the @render.ui for weights hasn't
      flushed yet, or n_obj shrank mid-run), falls back to None rather
      than halting silently — matches the preflight_fix_N pattern at
      calibration_handlers.py:815-819.
      """
      from shiny.types import SilentException

      mode = getattr(input, "cal_optimum_mode", lambda: "pareto")()
      if mode != "weighted":
          return None
      weights: list[float] = []
      for i in range(n_obj):
          try:
              raw = getattr(input, f"cal_weight_{i}")()
          except SilentException:
              return None
          try:
              weights.append(float(raw or 0.0))
          except (TypeError, ValueError):
              return None
      return weights
  ```

- [ ] **Step 2: Write the failing helper tests**

  Append to `tests/test_ui_calibration_handlers.py`:

  ```python
  from shiny.types import SilentException

  from tests.helpers import make_multi_input
  from ui.pages.calibration_handlers import _resolve_optimum_weights


  def test_resolve_optimum_weights_pareto_mode_returns_none():
      inp = make_multi_input(cal_optimum_mode="pareto", default=None)
      assert _resolve_optimum_weights(inp, n_obj=3) is None


  def test_resolve_optimum_weights_weighted_mode_reads_N_inputs():
      inp = make_multi_input(
          cal_optimum_mode="weighted",
          cal_weight_0=1.0,
          cal_weight_1=2.0,
          default=None,
      )
      assert _resolve_optimum_weights(inp, n_obj=2) == [1.0, 2.0]


  def test_resolve_optimum_weights_falls_back_on_silent_exception():
      """Simulate the @render.ui not having flushed — cal_weight_1 raises
      SilentException. Helper must return None (fallback to Pareto) rather
      than propagate the exception and halt the click handler."""

      class _PartialInput:
          cal_optimum_mode = staticmethod(lambda: "weighted")
          cal_weight_0 = staticmethod(lambda: 1.0)

          def __getattr__(self, name):
              if name == "cal_weight_1":
                  raise SilentException()
              raise AttributeError(name)

      assert _resolve_optimum_weights(_PartialInput(), n_obj=2) is None
  ```

- [ ] **Step 3: Run helper tests — Step 1 should already pass them**

  ```bash
  .venv/bin/python -m pytest tests/test_ui_calibration_handlers.py -v -k resolve_optimum_weights
  ```

  Expected: 3 passed.

- [ ] **Step 4: Add the reactive + toggle + weights render to `calibration.py`**

  At the top of the surrogate-optimum panel in `ui/pages/calibration.py` (grep for the existing `output_widget("pareto_chart")` at line 129 to locate the panel), add the toggle and a slot for the dynamic weights row:

  ```python
  ui.input_radio_buttons(
      "cal_optimum_mode",
      "Optimum",
      choices={"pareto": "Pareto front", "weighted": "Weighted sum"},
      selected="pareto",
      inline=True,
  ),
  ui.output_ui("weights_inputs"),
  ui.help_text(
      "Weights are raw non-negative floats. Scaling is irrelevant for "
      "ranking within a single search — [0.3, 0.7] and [3, 7] pick the "
      "same point.",
      class_="small text-muted",
  ),
  ```

  In the server-side function of `calibration.py` (look for `def server(input, output, session)` or the equivalent — grep for `pareto_chart` at line 262 to find the surrogate-optimum server block), add a new reactive value near the existing `preflight_result = reactive.value(None)`:

  ```python
  surrogate_optimum = reactive.value(None)
  ```

  And add the dynamic weights renderer:

  ```python
  @render.ui
  def weights_inputs():
      mode = input.cal_optimum_mode()
      if mode != "weighted":
          return ui.TagList()  # empty — Pareto mode shows no weights
      optimum = surrogate_optimum.get()
      if optimum is None:
          return ui.help_text(
              "Run the surrogate workflow first — weights match the "
              "fitted surrogate's objectives.",
              class_="small text-muted",
          )
      obj = optimum.get("predicted_objectives")
      n_obj = int(getattr(obj, "__len__", lambda: 1)())
      # Objective labels: reuse existing obj_labels source if available;
      # otherwise fall back to numeric indexing.
      labels = _obj_labels_for_surrogate(optimum) if n_obj else []
      rows = []
      for i in range(n_obj):
          lab = labels[i] if i < len(labels) else f"obj_{i}"
          rows.append(
              ui.input_numeric(
                  f"cal_weight_{i}",
                  f"w[{lab}]",
                  value=1.0,
                  min=0.0,
                  step=0.1,
              )
          )
      return ui.TagList(*rows)
  ```

  `_obj_labels_for_surrogate(optimum)` is a small helper — define it at module scope in `calibration.py`:

  ```python
  def _obj_labels_for_surrogate(optimum: dict) -> list[str]:
      """Best-effort objective labels. Falls back to numeric indices."""
      labels = optimum.get("objective_labels")
      if labels:
          return list(labels)
      obj = optimum.get("predicted_objectives")
      n = int(getattr(obj, "__len__", lambda: 0)()) if obj is not None else 0
      return [f"obj_{i}" for i in range(n)]
  ```

- [ ] **Step 5: Wire the mode branch at the `find_optimum` call site**

  At `ui/pages/calibration_handlers.py:483`, the current call is:

  ```python
  msg_queue.post_status("Finding optimum on surrogate...")
  optimum = calibrator.find_optimum()
  ```

  Replace with:

  ```python
  msg_queue.post_status("Finding optimum on surrogate...")
  n_obj = calibrator.surrogate.n_objectives
  weights = _resolve_optimum_weights(input, n_obj)
  if weights is not None:
      optimum = calibrator.find_optimum(weights=weights)
  else:
      optimum = calibrator.find_optimum()  # weights=None → Pareto set
  ```

  Also, add a line to publish the result to the new reactive. The `msg_queue` is already passed through — add a method on `MsgQueue` to mirror the existing pattern:

  In the MsgQueue class (around line 40-75 of `calibration_handlers.py`):

  ```python
  def post_surrogate_optimum(self, optimum) -> None:
      self._q.put(("surrogate_optimum", optimum))
  ```

  At the `find_optimum` call site, after assigning `optimum`, add:

  ```python
  msg_queue.post_surrogate_optimum(optimum)
  ```

  In the queue-drain dispatch (`:346-365`), add a branch:

  ```python
  elif kind == "surrogate_optimum":
      surrogate_optimum.set(payload)
  ```

  Pass `surrogate_optimum` into the handler-registration function the same way `preflight_result` is passed — match the existing threading of reactive values through the handler registrar.

- [ ] **Step 6: Add the scatter, table, and weighted-summary renderers**

  Back in `ui/pages/calibration.py`, below the toggle from Step 4, add the two conditional panels:

  ```python
  ui.panel_conditional(
      "input.cal_optimum_mode == 'pareto'",
      ui.layout_columns(
          output_widget("surrogate_pareto_scatter"),
          ui.output_data_frame("surrogate_pareto_table"),
          col_widths=[6, 6],
      ),
  ),
  ui.panel_conditional(
      "input.cal_optimum_mode == 'weighted'",
      ui.output_ui("surrogate_weighted_summary"),
  ),
  ```

  In the server function, add:

  ```python
  from shiny.render import DataGrid

  @render_plotly
  def surrogate_pareto_scatter():
      optimum = surrogate_optimum.get()
      if optimum is None or "pareto" not in optimum:
          return go.Figure()  # empty — weighted mode, or not yet run
      F = np.asarray(optimum["pareto"]["objectives"])
      if F.ndim != 2 or F.shape[1] >= 3:
          # n_obj ≥ 3 fallback: scatter is not meaningful; return empty.
          return go.Figure()
      labels = _obj_labels_for_surrogate(optimum)
      return make_pareto_chart(F, labels, tmpl=_tmpl())

  @render.data_frame
  def surrogate_pareto_table():
      optimum = surrogate_optimum.get()
      if optimum is None or "pareto" not in optimum:
          return pd.DataFrame()
      F = np.asarray(optimum["pareto"]["objectives"])
      U = np.asarray(optimum["pareto"]["uncertainty"])
      P = np.asarray(optimum["pareto"]["params"])
      labels = _obj_labels_for_surrogate(optimum)
      n_obj = F.shape[1] if F.ndim == 2 else 0
      n_params = P.shape[1] if P.ndim == 2 else 0
      data = {}
      for i in range(n_obj):
          data[labels[i] if i < len(labels) else f"obj_{i}"] = F[:, i]
      for i in range(n_obj):
          data[f"±{labels[i] if i < len(labels) else f'obj_{i}'}"] = U[:, i]
      for j in range(n_params):
          data[f"param_{j}"] = P[:, j]
      df = pd.DataFrame(data)
      return DataGrid(df, selection_mode="row")

  @render.ui
  def surrogate_weighted_summary():
      optimum = surrogate_optimum.get()
      if optimum is None or "pareto" in optimum:
          return ui.help_text(
              "Switch to Weighted sum and run the surrogate workflow "
              "to see the single weighted-optimum point.",
              class_="small text-muted",
          )
      obj = np.asarray(optimum["predicted_objectives"])
      unc = np.asarray(optimum["predicted_uncertainty"])
      params = np.asarray(optimum["params"])
      return ui.div(
          ui.tags.strong("Weighted optimum"),
          ui.br(),
          ui.tags.span(
              f"objectives = {np.round(obj, 4).tolist()} "
              f"(±{np.round(unc, 4).tolist()})"
          ),
          ui.br(),
          ui.tags.span(f"parameters = {np.round(params, 4).tolist()}"),
          class_="p-3 border rounded bg-light",
      )
  ```

  (Imports needed at the top of `calibration.py` if not already present: `import numpy as np`, `import pandas as pd`, `import plotly.graph_objects as go`, `from shinywidgets import render_plotly`. Match whatever the sibling `pareto_chart` renderer at line 262-269 already imports — do NOT add duplicates.)

- [ ] **Step 7: Write UI-layer renderer tests**

  Append to `tests/test_ui_calibration_handlers.py`:

  ```python
  import numpy as np
  import pandas as pd

  from shiny.render._data_frame_utils._tbl_data import _TBL_DATA_MODULES  # existence check only
  # If the import path above is fragile in a future Shiny version, remove
  # this import — the test below doesn't need it.


  def test_weights_inputs_renders_zero_inputs_in_pareto_mode(monkeypatch):
      """In Pareto mode, weights_inputs emits no input_numeric tags."""
      # The renderer reads input.cal_optimum_mode() and surrogate_optimum.get().
      # Because @render.ui is bound to the server function, we test the
      # inner logic by extracting it as a module-level helper OR by
      # stubbing both dependencies. Here we extract: create
      # _render_weights_inputs(mode, optimum) at module scope in
      # calibration.py and test that instead.
      from ui.pages.calibration import _render_weights_inputs
      out = _render_weights_inputs(mode="pareto", optimum=None)
      assert str(out) == "" or "input_numeric" not in str(out)


  def test_weights_inputs_renders_N_inputs_in_weighted_mode():
      from ui.pages.calibration import _render_weights_inputs
      optimum = {
          "predicted_objectives": np.array([1.0, 2.0, 3.0]),
          "objective_labels": ["a", "b", "c"],
      }
      out = _render_weights_inputs(mode="weighted", optimum=optimum)
      html = str(out)
      for i in range(3):
          assert f"cal_weight_{i}" in html, f"Missing input cal_weight_{i}"


  def test_surrogate_pareto_scatter_empty_for_high_dim():
      """When n_obj >= 3, the scatter renderer returns an empty figure;
      the table still produces M rows."""
      from ui.pages.calibration import _render_pareto_scatter, _render_pareto_table

      # 4 Pareto points in 3-D objective space
      optimum = {
          "pareto": {
              "objectives": np.arange(12).reshape(4, 3).astype(float),
              "uncertainty": np.full((4, 3), 0.1),
              "params": np.arange(8).reshape(4, 2).astype(float),
          },
          "predicted_objectives": np.array([0.0, 0.0, 0.0]),
          "predicted_uncertainty": np.array([0.1, 0.1, 0.1]),
          "params": np.array([0.0, 0.0]),
      }
      fig = _render_pareto_scatter(optimum)
      # Empty figure = no traces
      assert len(fig.data) == 0, "Expected empty scatter for n_obj >= 3"

      df = _render_pareto_table(optimum)
      # Pareto table should have 4 rows regardless of n_obj
      if hasattr(df, "data"):  # DataGrid wraps a DataFrame
          df = df.data
      assert len(df) == 4, f"Expected 4 rows, got {len(df)}"
  ```

  These tests require a small refactor: extract the renderer bodies into module-level pure functions `_render_weights_inputs(mode, optimum)`, `_render_pareto_scatter(optimum)`, `_render_pareto_table(optimum)` in `calibration.py`. The `@render.ui` / `@render.data_frame` decorators in the server function then become thin wrappers:

  ```python
  @render.ui
  def weights_inputs():
      return _render_weights_inputs(input.cal_optimum_mode(), surrogate_optimum.get())

  @render_plotly
  def surrogate_pareto_scatter():
      return _render_pareto_scatter(surrogate_optimum.get())

  @render.data_frame
  def surrogate_pareto_table():
      return _render_pareto_table(surrogate_optimum.get())
  ```

  This mirrors the `_clamp_n_workers` / `_resolve_optimum_weights` pattern — pure function for the logic, decorated wrapper for the reactive binding.

- [ ] **Step 8: Run tests — verify all passing**

  ```bash
  .venv/bin/python -m pytest tests/test_ui_calibration_handlers.py -v -k "resolve_optimum_weights or weights_inputs or pareto_scatter or pareto_table"
  ```

  Expected: 6 passed (3 from Step 2, 3 from Step 7).

- [ ] **Step 9: Run full suite + lint**

  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check osmose/ scripts/ tests/ ui/
  ```

  Expected: `2461 passed` (2455 after Task 2 + 6 new); lint clean.

- [ ] **Step 10: Commit**

  ```bash
  git add ui/pages/calibration.py ui/pages/calibration_handlers.py tests/test_ui_calibration_handlers.py
  git commit -m "feat(calibration-ui): Pareto/Weighted toggle for surrogate optimum

  Expose the library's find_optimum(weights=...) dual-mode capability.

  UI: new radio cal_optimum_mode (Pareto | Weighted) in the surrogate
  optimum panel. In Weighted mode, a @render.ui dynamically emits one
  input_numeric per objective (cal_weight_0 ... cal_weight_{n-1});
  values flow into find_optimum as weights. In Pareto mode, the panel
  shows a scatter (make_pareto_chart, reused) and a DataGrid table
  side-by-side for n_obj <= 2; scatter returns empty for n_obj >= 3
  and only the table renders.

  New reactive surrogate_optimum = reactive.value(None) holds the
  find_optimum dict (either single weighted point or Pareto set +
  anchor). MsgQueue gains post_surrogate_optimum; queue drain gets a
  matching branch.

  Helpers extracted for testability: _resolve_optimum_weights,
  _render_weights_inputs, _render_pareto_scatter, _render_pareto_table.
  Pure functions with explicit args — no reactive internals in the
  test surface.

  Tests: +6 in test_ui_calibration_handlers.py (weight resolution,
  renderer branches, n_obj >= 3 fallback, SilentException race).
  2461 passed."
  ```

---

## Task 4: Final validation + CHANGELOG

- [ ] **Step 1: Full test suite**

  ```bash
  .venv/bin/python -m pytest -q
  ```

  Expected: `2461 passed` (2449 baseline + 1 (Task 1) + 5 (Task 2) + 6 (Task 3)).

- [ ] **Step 2: Full lint**

  ```bash
  .venv/bin/ruff check osmose/ scripts/ tests/ ui/
  ```

  Expected: `All checks passed!`

- [ ] **Step 3: Manual UI smoke check**

  Start the app and exercise each of the three capabilities once by hand:

  ```bash
  .venv/bin/shiny run app.py --host 0.0.0.0 --port 8000
  ```

  Open `http://localhost:8000`, go to the Calibration tab, and verify:
  1. Preflight settings show a new `Workers` input alongside the enable checkbox.
  2. Surrogate Optimum panel shows the `Pareto | Weighted sum` radio.
  3. Switching to Weighted shows weights inputs; switching back hides them.
  4. (Optional, requires a fitted surrogate:) the Pareto scatter and table render with data.

  Any visual regression vs pre-Phase-3 is a stop condition — capture it and stop the plan.

- [ ] **Step 4: Append CHANGELOG entry**

  In `CHANGELOG.md` under `[Unreleased] → Added`, prepend:

  ```markdown
  - **calibration-ui:** Phase 3 surfaces three existing library capabilities on the UI. (a) `PreflightEvalError` now renders as a red-banner modal with the failing stage (`morris`), error message, and recovery hints — no more opaque "Preflight failed". (b) Preflight settings expose a `Workers` numeric input (`cal_preflight_workers`) that wires to `run_preflight(n_workers=...)`, default 1 (sequential). (c) Surrogate Optimum panel gains a `Pareto | Weighted sum` radio; Weighted mode emits one `input_numeric` per objective (passed as `weights=` into `find_optimum`); Pareto mode shows scatter+table (n_obj ≤ 2) or table-only (n_obj ≥ 3). Spec at `docs/superpowers/specs/2026-04-19-calibration-ui-phase3-design.md`; plan at `docs/superpowers/plans/2026-04-19-calibration-ui-phase3-plan.md`. (commits from Task 1-3)
  ```

- [ ] **Step 5: Commit CHANGELOG**

  ```bash
  git add CHANGELOG.md
  git commit -m "docs: changelog for calibration UI Phase 3"
  ```

- [ ] **Step 6: Push**

  ```bash
  git push origin master
  ```

- [ ] **Step 7: Skim git log**

  ```bash
  git log --oneline -6
  ```

  Expected: four plan commits on top of the spec — three feature commits (Tasks 1-3) plus the CHANGELOG commit.

---

## Self-review checklist

- **Spec coverage:** Every spec requirement maps to a task. Capability 1 → Task 1 (library touch + dispatcher + error helper + handler catch + queue branch + test). Capability 2 → Task 2 (input + clamp helper + handler wire + 4 clamp tests + registration test). Capability 3 → Task 3 (radio + dynamic weights render + mode branch at `:483` + new reactive + queue branch + three renderer helpers + 6 tests). Non-goals explicitly preserved (no retry button, no Sobol raise site, no normalization, no per-row parameter viewer, no e2e Shiny client). ✓
- **Placeholders scanned:** No TBD/TODO. Every code step has a full code block. Every test has full assert bodies with expected values. ✓
- **Type consistency:** `_clamp_n_workers(requested, cpu) -> int` same signature wherever referenced. `_resolve_optimum_weights(input, n_obj) -> list[float] | None` same signature in helper + test + call site. `_render_weights_inputs(mode, optimum)` / `_render_pareto_scatter(optimum)` / `_render_pareto_table(optimum)` signatures consistent between the refactor in Step 7 and the tests that call them. `PreflightEvalError.stage` — attribute name consistent across Steps 1, 4, 5 and the test. ✓
- **Imports:** `from shiny.types import SilentException` (not `shiny.reactive` — verified during spec review). `DataGrid(selection_mode="row")` (not `row_selection_mode` — deprecated in Shiny 1.5). `getattr(input, f"...")` style matches the existing `calibration_handlers.py:146,819,936,948`. ✓
- **File paths:** All absolute or clearly repo-root-relative; every cited line number (`preflight.py:119,459,665`; `surrogate.py:123`; `calibration.py:102-106,129,262,269`; `calibration_handlers.py:65,196,346-365,483,779,788-800`) verified against the current repo. ✓
- **Test runner:** Tests run via `.venv/bin/python -m pytest` (per `CLAUDE.md`). No `python` bare calls. No `$()` inside bash commands. ✓
- **Commit granularity:** One commit per capability (Tasks 1-3) plus one CHANGELOG commit. Clean revert story — each feature can be backed out independently. ✓

---

## Execution handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
