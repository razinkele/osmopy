# Calibration UI Phase 3 — Design

> **Front 3** of the 2026-04-18 post-session roadmap. Spec for the plan at (forthcoming) `docs/superpowers/plans/2026-04-19-calibration-ui-phase3-plan.md`.

**Goal:** Surface three library capabilities that don't currently have any UI. All three exist and are library-tested; Phase 3 is pure UI plumbing, no engine change.

1. `PreflightEvalError` (`osmose/calibration/preflight.py:119`) — raised when >50 % of Morris samples fail; currently surfaces as an opaque "Preflight failed" with no detail.
2. `n_workers` on `run_preflight` (`preflight.py:665`) — the parallel evaluator is fully wired library-side but UI hardcodes 1.
3. `find_optimum(weights=...)` (`osmose/calibration/surrogate.py:123`) — returns either a Pareto set (weights=None) or a weighted-scalarized single point (weights given); UI today only shows the unweighted sum.

**Out of scope:** new library features; changes to NSGA-II or the schema registry; UI theming/accessibility beyond what existing components already do; the "View parameters for a selected Pareto point" side panel (nice-to-have, deferred).

## Architecture

Three independent UI additions, all in `ui/pages/calibration.py` + `ui/pages/calibration_handlers.py` + `ui/pages/calibration_charts.py` (existing files). No schema changes; no new reactive values beyond one new `surrogate_optimum = reactive.value(None)` for the Pareto render path. Each capability is independently testable and independently shippable behind `if`-gates in the existing reactive graph.

```
┌──────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────┐
│ Preflight settings   │   │ Preflight worker (handler)   │   │ Preflight modal      │
│ ┌──────────────────┐ │   │ ┌──────────────────────────┐ │   │ ┌──────────────────┐ │
│ │ Enable ✓         │ │──▶│ │ run_preflight(           │ │──▶│ │ success → checkbox│ │
│ │ Workers [n_w]    │ │   │ │   ..., n_workers=n_w,   │ │   │ │ error  → red alert│ │
│ └──────────────────┘ │   │ │ )                        │ │   │ └──────────────────┘ │
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

`build_preflight_modal(result)` (`calibration_handlers.py:196`) currently assumes `result` has `.issues` and `.free_params` (the success shape). When `run_preflight` raises `PreflightEvalError` (`preflight.py:459`), the handler's generic `except` block drops to a bland "Preflight failed" message with no detail, so users can't tell which stage failed or why.

### UI changes

- `calibration_handlers.py:65-66` — `post_preflight` currently does `self._q.put(("preflight", result))` (hardcoded tag). Add a sibling `post_preflight_error(exc)` that does `self._q.put(("preflight_error", exc))`. Do NOT change the existing `post_preflight` signature — downstream callers rely on the tag format.
- `calibration_handlers.py:~330` — the `except` block around `run_preflight` catches `PreflightEvalError` explicitly:

  ```python
  try:
      result = run_preflight(...)
      post_preflight(result)
  except PreflightEvalError as exc:
      post_preflight_error(exc)
  ```

- `calibration_handlers.py:~364` — the queue-drain dispatch already handles `kind == "preflight"`. Add a parallel branch `elif kind == "preflight_error": modal = build_preflight_modal(payload)` — same `build_preflight_modal` entry point, which now dispatches on payload type.

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

`PreflightEvalError` gains a `.stage` string attribute populated at the raise site (`preflight.py:459`). Both raise sites (Morris and Sobol) set it to the matching stage name. Existing callers tolerate the additional attribute — it's read via `getattr(exc, "stage", "unknown")` in the modal.

### Testing

Addition to `tests/test_calibration_handlers.py`:

- `test_preflight_modal_renders_error_banner_for_PreflightEvalError`: construct `PreflightEvalError("half the samples failed", stage="morris")`, pass through `build_preflight_modal`, assert the returned HTML contains `alert-danger` and both the stage label "morris" and the error message.

Not testing the exception-catch path end-to-end — that belongs to the library side, which already has its own tests.

---

## Capability 2 — `n_workers` numeric input

### Scope

Expose `n_workers` on `run_preflight` via a numeric input. Library default 1 (sequential) preserved — upgrading should not silently change parallelism.

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

- `ui/pages/calibration_handlers.py:~333` — the preflight dispatcher reads `input.cal_preflight_workers()`, clamps to `[1, max(1, os.cpu_count())]` as a defense against None/invalid inputs, and passes as `n_workers=`.

- `import os` at the top of `calibration.py` if not already present.

### No library touch

`n_workers` already exists and is library-tested.

### Testing

Addition to `tests/test_calibration_handlers.py`:

- `test_preflight_handler_passes_n_workers`: stub `input.cal_preflight_workers()` returning 4, mock `run_preflight` as a sentinel-recording stub, trigger the preflight dispatch, assert the stub was called with `n_workers=4`.
- `test_preflight_handler_clamps_n_workers`: stub returns 0 or None, assert the stub is called with `n_workers=1`.

Addition to `tests/test_calibration_ui.py`:

- `test_cal_preflight_workers_input_registered`: assert the input ID `cal_preflight_workers` appears in the rendered UI (prevents silent removal regression).

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

`ui/pages/calibration_handlers.py:~483` currently:

```python
optimum = calibrator.find_optimum()
```

Becomes:

```python
mode = input.cal_optimum_mode()
if mode == "weighted":
    weights = [input[f"cal_weight_{i}"]() for i in range(n_obj)]
    optimum = calibrator.find_optimum(weights=weights)
else:
    optimum = calibrator.find_optimum()  # weights=None → Pareto set
```

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
- **Table (`surrogate_pareto_table`).** New `@render.data_frame`. Columns: `obj_0, …, obj_{n-1}, ±unc_0, …, ±unc_{n-1}, param_0, …, param_{k-1}`. Rows = Pareto points. Pandas-rendered; `DataGrid(row_selection_mode="single")` for future "view params for selected row" extension (out of scope for MVP).
- **Weighted summary (`surrogate_weighted_summary`).** Compact `ui.output_ui` block: "Best point (weighted sum = *w·means* = X): objectives = [a, b, …], parameters = [p, q, …]".

### 3d. n_obj ≥ 3 fallback

Shiny's `ui.panel_conditional` can't inspect `n_obj` (server-side computed), so the fallback is entirely server-side. The scatter renderer returns `None`/empty-figure when the fitted surrogate reports `n_obj ≥ 3`; the table still renders for all `n_obj`. No UI-side toggle needed.

### 3e. Testing

Additions to `tests/test_calibration_handlers.py`:

- `test_find_optimum_passes_weights_when_mode_weighted`: stub `input.cal_optimum_mode()` → "weighted", `input.cal_weight_{i}()` → `[1.0, 2.0]`, mock `calibrator.find_optimum` as a spy, assert it's called with `weights=[1.0, 2.0]`.
- `test_find_optimum_passes_no_weights_when_mode_pareto`: stub mode → "pareto", assert `find_optimum` called with no `weights` kwarg and the result dict has a `"pareto"` key.
- `test_weights_inputs_renders_zero_inputs_in_pareto_mode`: stub mode → "pareto", `n_objectives = 3`, assert `weights_inputs` produces no `input_numeric` rows.
- `test_weights_inputs_renders_N_inputs_in_weighted_mode`: stub mode → "weighted", `n_objectives = 3`, assert three `input_numeric` rows.
- `test_surrogate_pareto_scatter_empty_for_high_dim`: mock `surrogate.n_objectives = 3`, construct a Pareto result dict with an `(M, 3)` objectives array, assert the scatter widget is empty while the table widget produces `M` rows.

Plus one end-to-end Shiny test-client check if the harness supports it (some radio-button events are finicky in the test client; skip with `@pytest.mark.skipif` if needed).

---

## Non-goals and explicit trade-offs

- **No retry button** on the preflight error modal. If the pattern emerges, add later.
- **No weight normalization** — library accepts raw non-negative floats and ranking is invariant to positive scaling, so normalization would add UI code without behavior change.
- **No per-row parameter view** in the Pareto table for MVP. Single-row-selection capability is present for future addition; the viewer panel is deferred.
- **Existing NSGA-II `pareto_chart`** stays in place — it's a different data source (the real optimizer run) from the surrogate Pareto front being added here. Both can coexist.
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

- Library surfaces: `osmose/calibration/preflight.py:119,459,665`; `osmose/calibration/surrogate.py:123`.
- UI entry points: `ui/pages/calibration.py:100,262,269`; `ui/pages/calibration_handlers.py:196,333,483`; `ui/pages/calibration_charts.py`.
- Prior phase: `docs/superpowers/plans/2026-04-15-calibration-ui-phase2-plan.md` (STATUS-COMPLETE).
- Front 3 roadmap entry: `docs/superpowers/plans/2026-04-18-post-session-roadmap.md`.
