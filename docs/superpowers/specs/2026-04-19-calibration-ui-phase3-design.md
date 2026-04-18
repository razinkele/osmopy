# Calibration UI Phase 3 — Design

> **Front 3** of the 2026-04-18 post-session roadmap. Spec for the plan at (forthcoming) `docs/superpowers/plans/2026-04-19-calibration-ui-phase3-plan.md`.

**Goal:** Surface three library capabilities that don't currently have any UI. All three exist and are library-tested; Phase 3 is pure UI plumbing, no engine change.

1. `PreflightEvalError` (`osmose/calibration/preflight.py:119`) — raised when >50 % of Morris samples fail; currently surfaces as an opaque "Preflight failed" with no detail.
2. `n_workers` on `make_preflight_eval_fn` (`preflight.py:665` — the factory that builds the evaluator fed into `run_preflight`) — the parallel evaluator is fully wired library-side but UI hardcodes 1. Note: `n_workers` is NOT a kwarg of `run_preflight` itself (`preflight.py:565`); passing it there raises `TypeError`.
3. `find_optimum(weights=...)` (`osmose/calibration/surrogate.py:123`) — returns either a Pareto set (weights=None) or a weighted-scalarized single point (weights given); UI today only shows the unweighted sum.

**Out of scope:** new library features; changes to NSGA-II or the schema registry; UI theming/accessibility beyond what existing components already do; the "View parameters for a selected Pareto point" side panel (nice-to-have, deferred).

## Architecture

Three independent UI additions, all in `ui/pages/calibration.py` + `ui/pages/calibration_handlers.py` + `ui/pages/calibration_charts.py` (existing files). No schema changes; no new reactive values beyond one new `surrogate_optimum = reactive.value(None)` for the Pareto render path. Each capability is independently testable and independently shippable behind `if`-gates in the existing reactive graph.

```
┌──────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────┐
│ Preflight settings   │   │ Preflight worker (handler)   │   │ Preflight modal      │
│ ┌──────────────────┐ │   │ ┌──────────────────────────┐ │   │ ┌──────────────────┐ │
│ │ Enable ✓         │ │──▶│ │ eval_fn = make_preflight_│ │──▶│ │ success → checkbox│ │
│ │ Workers [n_w]    │ │   │ │   eval_fn(...,           │ │   │ │ error  → red alert│ │
│ │                  │ │   │ │   n_workers=n_w)         │ │   │ │                  │ │
│ └──────────────────┘ │   │ │ run_preflight(eval_fn=...│ │   │ └──────────────────┘ │
└──────────────────────┘   │ └──────────────────────────┘ │   └──────────────────────┘
                           │ except PreflightEvalError: │
                           │   post_preflight_error(exc) │
                           └──────────────────────────────┘

┌──────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────┐
│ Optimum panel        │   │ find_optimum(weights?)       │   │ Render                   │
│ ┌──────────────────┐ │   │                              │   │ Pareto:  scatter + table │
│ │ ○ Pareto         │ │──▶│ weights=None → Pareto set    │──▶│  n_obj≥3 → table only    │
│ │ ○ Weighted       │ │   │ weights given → single point │   │ Weighted: one-row summary│
│ │ [w0 w1 …]        │ │   │                              │   │                          │
│ └──────────────────┘ │   └──────────────────────────────┘   └──────────────────────────┘
└──────────────────────┘
```

---

## Capability 1 — PreflightEvalError red-banner

### Scope

`build_preflight_modal(result)` (`calibration_handlers.py:196`) currently reads `result.issues` (line 202), `result.elapsed_seconds` (208), `len(result.survivors)` and `len(result.screening)` (209) — all the success shape. When `run_preflight` raises `PreflightEvalError` (`preflight.py:459` — *only Morris has a raise site today*; Sobol has no equivalent gate in this version), the handler's generic `except Exception` at line 797 catches it indistinguishably from any other exception and the user gets an opaque "Preflight failed" message.

### UI changes

- `calibration_handlers.py:65-66` — `post_preflight` currently does `self._q.put(("preflight", result))` (hardcoded tag). Add a sibling `post_preflight_error(exc)` that does `self._q.put(("preflight_error", exc))`. Do NOT change the existing `post_preflight` signature — downstream callers rely on the tag format.
- `calibration_handlers.py:788-800` (the `_run_preflight_thread` closure inside `register_calibration_handlers`) — the existing `except Exception as exc` at line 797 catches everything generic. Insert a specific `except PreflightEvalError` *before* the generic catch so the structured error flows through its own tag:

  ```python
  try:
      result = run_preflight(...)          # existing, line 790
      msg_queue.post_preflight(result)     # existing, line 796
  except PreflightEvalError as exc:        # NEW — before the generic Exception
      msg_queue.post_preflight_error(exc)
  except Exception as exc:                 # existing, line 797 — unchanged
      msg_queue.post_error(str(exc))
  ```

- `calibration_handlers.py:364` — the queue-drain dispatch already handles `kind == "preflight"` (sets `preflight_result.set(payload)` at line 365). Add a parallel branch `elif kind == "preflight_error": modal = build_preflight_modal(payload)` — same `build_preflight_modal` entry point, which now dispatches on payload type.

- `calibration_handlers.py:196` — `build_preflight_modal` becomes a dispatcher:

  ```python
  def build_preflight_modal(result_or_error):
      if isinstance(result_or_error, PreflightEvalError):
          return _build_preflight_error_modal(result_or_error)
      return _build_preflight_success_modal(result_or_error)
  ```

- New helper `_build_preflight_error_modal(exc)` renders a Bootstrap `alert-danger` banner with:
  - **Stage name** (from `exc.stage` — see library touch below): "Morris" or "Sobol".
  - **Error message**: `str(exc)`.
  - **Recovery hints** (hardcoded text, not computed): "Try narrowing parameter bounds, reducing the Morris sample budget, or re-running with `n_workers=1` to see the underlying exception."
  - Footer: single "Close" button. Dismissal-only — no retry button in this iteration.

### Library touch

`PreflightEvalError` gains a `.stage` string attribute populated at the single current raise site (`preflight.py:459`, Morris). The modal reads it via `getattr(exc, "stage", "unknown")` so older exception instances (or a future Sobol raise site that forgot to set `.stage`) degrade gracefully to "unknown". Adding a Sobol raise site is out of scope for this phase — flagged in Non-goals.

### Testing

Addition to `tests/test_ui_calibration_handlers.py` (uses the existing `make_multi_input` / `make_catch_all_input` helpers from `tests/helpers.py` — no Shiny test client, no spy library):

- `test_preflight_modal_renders_error_banner_for_PreflightEvalError`: construct `PreflightEvalError("half the samples failed", stage="morris")`, pass through `build_preflight_modal`, assert the returned modal tree (or serialized HTML) contains the `alert-danger` class, the stage label "morris", and the error message. `build_preflight_modal` is a pure function — no Shiny inputs needed.

Not testing the exception-catch path end-to-end — that belongs to the library side (which already has its own tests) and to the queue-drain dispatch, which is covered indirectly by existing preflight integration tests.

---

## Capability 2 — `n_workers` numeric input

### Scope

Expose `n_workers` on `make_preflight_eval_fn` (the evaluator factory fed into `run_preflight`) via a numeric input. Library default 1 (sequential) preserved — upgrading should not silently change parallelism. **`run_preflight` itself does NOT accept `n_workers`** — the parallelism knob is on the factory that produces its `evaluation_fn`.

### UI changes

- `ui/pages/calibration.py:~100` — add in the preflight settings panel, after `cal_preflight_enabled`:

  ```python
  ui.input_numeric(
      "cal_preflight_workers",
      "Workers",
      value=1,
      min=1,
      max=os.cpu_count() or 1,
      step=1,
  ),
  ```

  Help text: "Parallel evaluators for preflight sample runs. 1 = sequential (default)."

- `ui/pages/calibration_handlers.py:781-794` — `_run_preflight_thread` builds `eval_fn = make_preflight_eval_fn(...)` (line 781) and then feeds it to `run_preflight(evaluation_fn=eval_fn, ...)` (line 790). The `n_workers` kwarg goes on the **factory at line 781**, not the `run_preflight` call. Add `n_workers=getattr(input, "cal_preflight_workers", lambda: 1)()` clamped via `_clamp_n_workers(...)` into the factory-call kwargs. `getattr` with a fallback lambda protects against the input ID being absent during tests that stub only some inputs.

- `import os` at the top of `calibration.py` if not already present.

### No library touch

`n_workers` already exists and is library-tested.

### Testing

Extract a small pure-Python seam from `_run_preflight_thread` — a helper `_clamp_n_workers(requested: int | None, cpu: int | None) -> int` — so the clamping logic is testable without threading / Shiny mocks. Add to `tests/test_ui_calibration_handlers.py`:

- `test_clamp_n_workers_honors_valid_input`: `(4, 8) -> 4`.
- `test_clamp_n_workers_clamps_to_cpu_count`: `(16, 4) -> 4`.
- `test_clamp_n_workers_defaults_on_invalid`: `(None, 8) -> 1`, `(0, 8) -> 1`, `(-1, 8) -> 1`.
- `test_clamp_n_workers_survives_null_cpu_count`: `(4, None) -> 1` (platforms where `os.cpu_count()` returns None).

Addition to `tests/test_ui_calibration.py`:

- `test_cal_preflight_workers_input_registered`: build the calibration UI module with `make_multi_input(default=None)` and a minimal scenario; assert the rendered tag tree contains an input with ID `cal_preflight_workers`. This guards against silent removal.

### Edge case

`os.cpu_count()` can return `None` on some platforms. Fall back to `1` for `max=`. Sandboxed CI may report fewer cores than the host has — don't assume ≥ 4.

---

## Capability 3 — Pareto / Weighted toggle and view

### 3a. Mode toggle + weights row

Additions in the Surrogate Optimum panel of `ui/pages/calibration.py`:

```python
ui.input_radio_buttons(
    "cal_optimum_mode",
    "Optimum",
    choices={"pareto": "Pareto front", "weighted": "Weighted sum"},
    selected="pareto",
    inline=True,
),
ui.output_ui("weights_inputs"),
```

Server-side `@render.ui`-decorated `weights_inputs` returns empty (`ui.TagList()`) when mode=="pareto" and renders one `ui.input_numeric(f"cal_weight_{i}", label=f"w[{obj_labels[i]}]", value=1.0, min=0.0, step=0.1)` per objective when mode=="weighted". `n_objectives` and `obj_labels` come from the fitted surrogate, already reachable via the existing calibration reactive chain.

Help text under the weights row: "Non-negative floats. Scaling is irrelevant for ranking within one search — `[0.3, 0.7]` and `[3, 7]` pick the same point."

### 3b. Handler change — branch on mode

`ui/pages/calibration_handlers.py:483` currently:

```python
optimum = calibrator.find_optimum()
```

Becomes:

```python
from shiny import req
from shiny.types import SilentException

mode = getattr(input, "cal_optimum_mode", lambda: "pareto")()
if mode == "weighted":
    n_obj = calibrator.surrogate.n_objectives
    weights: list[float] = []
    for i in range(n_obj):
        try:
            weights.append(float(getattr(input, f"cal_weight_{i}")() or 0.0))
        except SilentException:
            # The @render.ui for weights hasn't flushed yet, or n_obj
            # shrank mid-run. Fall through to the Pareto path rather
            # than halt silently (matches the preflight_fix_N pattern
            # at calibration_handlers.py:815-819).
            weights = []
            break
    optimum = (calibrator.find_optimum(weights=weights)
               if weights else calibrator.find_optimum())
else:
    optimum = calibrator.find_optimum()  # weights=None → Pareto set
```

The `getattr` form (over `input[f"..."]()` subscripting) matches the existing style at `calibration_handlers.py:146, 819, 936, 948`. Both work; `getattr` is the house convention.

Result dict shape (per `surrogate.py:123` docstring):

- Weighted: `{params, predicted_objectives, predicted_uncertainty}` — single point.
- Pareto: same keys (anchor point) **plus** `pareto: {params, objectives, uncertainty}` of shape `(M, k)` / `(M, n_obj)`.

A new reactive `surrogate_optimum = reactive.value(None)` holds the whole dict. Render layer branches on `"pareto" in optimum` to decide which views to populate.

### 3c. Render layer

New panel below the toggle:

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

- **Scatter (`surrogate_pareto_scatter`).** Reuses `make_pareto_chart(F, obj_labels, tmpl=_tmpl())` from `calibration_charts.py`. Fed `optimum["pareto"]["objectives"]` — the surrogate Pareto set, not the NSGA-II `pareto_F` that the existing widget at `calibration.py:262` currently shows. (The existing NSGA-II widget stays — it's a different data source.) Returns an empty widget when `n_obj ≥ 3`.
- **Table (`surrogate_pareto_table`).** New `@render.data_frame`. Columns: `obj_0, …, obj_{n-1}, ±unc_0, …, ±unc_{n-1}, param_0, …, param_{k-1}`. Rows = Pareto points. Pandas-rendered via `DataGrid(df, selection_mode="row")` (note: `row_selection_mode` is deprecated in Shiny ≥1.5; use `selection_mode` with values `"row"`/`"rows"`/`"none"`). Cell-selection accessor for the future "view params for selected row" extension will be `input.surrogate_pareto_table_cell_selection()["rows"]` — out of scope for MVP, pinned here so the future implementer doesn't guess.
- **Weighted summary (`surrogate_weighted_summary`).** Compact `ui.output_ui` block: "Best point (weighted sum = *w·means* = X): objectives = [a, b, …], parameters = [p, q, …]".

### 3d. n_obj ≥ 3 fallback

Shiny's `ui.panel_conditional` can't inspect `n_obj` (server-side computed), so the fallback is entirely server-side. The scatter renderer returns `None`/empty-figure when the fitted surrogate reports `n_obj ≥ 3`; the table still renders for all `n_obj`. No UI-side toggle needed.

### 3e. Testing

Extract the mode-branching logic from the handler into a pure helper `_resolve_optimum_weights(input, n_obj) -> list[float] | None` (returns None for Pareto path, list for Weighted). Addition to `tests/test_ui_calibration_handlers.py` using `make_multi_input` / `make_catch_all_input`:

- `test_resolve_optimum_weights_pareto_mode_returns_none`: `make_multi_input(cal_optimum_mode="pareto", default=None)`, `n_obj=3`, assert returns `None`.
- `test_resolve_optimum_weights_weighted_mode_reads_N_inputs`: `make_multi_input(cal_optimum_mode="weighted", cal_weight_0=1.0, cal_weight_1=2.0, default=None)`, `n_obj=2`, assert returns `[1.0, 2.0]`.
- `test_resolve_optimum_weights_falls_back_on_silent_exception`: stub input such that `cal_weight_1` raises `SilentException` (simulating the render-hasn't-flushed race); `n_obj=2`, assert the helper returns `None` (fallback to Pareto) rather than raising.

Addition for the UI-layer branches (same file):

- `test_weights_inputs_renders_zero_inputs_in_pareto_mode`: call the `weights_inputs` renderer with `make_multi_input(cal_optimum_mode="pareto")` and a stub `n_objectives=3`; assert the returned `TagList` contains zero `input_numeric` tags.
- `test_weights_inputs_renders_N_inputs_in_weighted_mode`: same with `cal_optimum_mode="weighted"`; assert three `input_numeric` tags with IDs `cal_weight_0`, `cal_weight_1`, `cal_weight_2`.
- `test_surrogate_pareto_scatter_empty_for_high_dim`: pass a stub surrogate with `n_objectives=3` and a Pareto result dict with an `(M, 3)` objectives array into the scatter renderer; assert it returns an empty figure. Table renderer with the same input should still produce `M` rows.

No end-to-end Shiny test-client test. `pyproject.toml` sets `addopts = "-m 'not e2e'"` by default, radio-button interactions need Playwright, and none of the existing calibration tests go through that path. Radio-toggle behavior is exercised indirectly through the three helper tests above.

---

## Non-goals and explicit trade-offs

- **No Sobol `PreflightEvalError` raise site.** The library currently raises only in the Morris stage (`preflight.py:459`). Adding a parallel gate in Sobol is a library-side decision about when "Sobol quality is unacceptable," out of scope here. The modal degrades gracefully via `getattr(exc, "stage", "unknown")` — a future Sobol raise site can populate `.stage` with no UI change.
- **No retry button** on the preflight error modal. If the pattern emerges, add later.
- **No weight normalization** — library accepts raw non-negative floats and ranking is invariant to positive scaling, so normalization would add UI code without behavior change.
- **No per-row parameter view** in the Pareto table for MVP. Single-row-selection capability is present for future addition; the viewer panel is deferred.
- **Existing `pareto_chart`** at `ui/pages/calibration.py:262` stays in place — but note that its backing reactive `cal_F` is already written to by *both* the NSGA-II path (at `calibration_handlers.py:~576`, `res.F`) and the surrogate workflow (at `:485`, full Y matrix); so the existing widget today shows whichever path ran last. The new `surrogate_optimum` reactive (for Capability 3) is distinct — it holds the `find_optimum` dict (single point or Pareto set + anchor), not sample data. The two widgets will read from different reactives and can display different content simultaneously.
- **No phased rollout** across sessions — single implementation plan. Each of the three capabilities lands in its own commit for clean revert if needed.

## Implementation order (informational — plan is authoritative)

1. Library touch: add `.stage` to `PreflightEvalError` (one line at raise sites).
2. Capability 1 (error modal): smallest surface, validates the test pattern, commit.
3. Capability 2 (n_workers): one input + handler pass-through, commit.
4. Capability 3a+3b (toggle + handler branch): mode affects data shape downstream.
5. Capability 3c (render layer): scatter, table, weighted summary.
6. Capability 3d (n_obj≥3 fallback): single server-side guard.
7. Tests at the end of each numbered step. Final full pytest + ruff sweep.

## References

- Library surfaces: `osmose/calibration/preflight.py:119` (class), `:459` (Morris raise site — the only one), `:665` (`n_workers` kw); `osmose/calibration/surrogate.py:123` (`find_optimum(weights=...)`).
- UI entry points: `ui/pages/calibration.py:102-106` (preflight checkbox panel), `:262,269` (existing `pareto_chart`); `ui/pages/calibration_handlers.py:65` (`post_preflight`), `:196` (`build_preflight_modal`), `:346-365` (queue-drain dispatch), `:483` (`find_optimum` call), `:788-800` (`_run_preflight_thread` closure with try/except on line 797).
- Existing test style: `tests/test_ui_calibration_handlers.py`, `tests/test_ui_calibration.py`, `tests/test_ui_calibration_preflight.py`, using `make_multi_input` / `make_catch_all_input` helpers from `tests/helpers.py:81,91`.
- `getattr(input, f"...")` reader pattern: `calibration_handlers.py:146, 819, 936, 948`.
- Prior phase: `docs/superpowers/plans/2026-04-15-calibration-ui-phase2-plan.md` (STATUS-COMPLETE).
- Front 3 roadmap entry: `docs/superpowers/plans/2026-04-18-post-session-roadmap.md`.
