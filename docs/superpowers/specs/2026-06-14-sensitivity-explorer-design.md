# Parameter Sensitivity Explorer — design

**Date:** 2026-06-14
**Status:** Approved (design phase)
**Feature:** A dedicated top-level Shiny page that loads persisted Sobol sensitivity results and
visualizes them (ranked S1/ST tornado + table + export), plus a small shared save/load helper so
the *existing* live calibration Sobol run persists an artifact the page can discover.

## Motivation

OSMOSE already computes Sobol sensitivity (`osmose/calibration/sensitivity.py` `SensitivityAnalyzer`
— S1/ST + 95% CIs, single- and multi-objective) and exposes a **live** run inside the Calibration
page (Results → Sensitivity sub-tab: a button runs samples and draws a bar chart). But two gaps make
that result effectively un-explorable:

1. **No persistence** — the live result lives only in a `reactive.value` (`calibration.py:389`); it
   is lost on tab refresh / session end and never written to disk.
2. **No browsing surface** — the only persistent producer is the multi-hour batch script
   `scripts/sensitivity_phase12.py`, whose JSON/CSV output has no home in the UI.

So the backlog item *"Parameter sensitivity explorer (surface Sobol output as a Shiny page)"* is an
**enhancement, not a rebuild**: keep the analyzer as-is, make results persistent, and give them a
dedicated read/explore page. Confirmed by recon: `data/**/sensitivity/` is currently empty (nothing
has ever been persisted), so the feature must also create its own data source via the persist hook.

## Reuse (do not rebuild)

- `osmose/calibration/sensitivity.py` `SensitivityAnalyzer.analyze(Y, objective_names=None)` returns
  `{"S1","ST","S1_conf","ST_conf","param_names"(, "objective_names","n_objectives")}` (numpy arrays;
  1-D for single objective, 2-D `(n_obj, n_params)` for multi). **Unchanged.**
- The live calibration handler `ui/pages/calibration_handlers.py:handle_sensitivity` (the producer we
  hook) computes `sens_result = analyzer.analyze(Y_1d)` at **line 1789** then
  `msg_queue.post_sensitivity(sens_result)`. The live run is always single-objective (`Y_1d` is the
  summed objective).
- `osmose/history.py` is the pattern to mirror for the artifact store (`RUN_HISTORY_DIR`,
  `default_run_history`, `list_runs`/`load_run`, path-safety guards).
- `app.py` page registration: `from ui.pages.X import X_ui, X_server` → `ui.nav_panel("Label",
  X_ui(), value="x")` in the `navset_pill_list` (id `main_nav`, line 286) → `X_server(...)` in the
  server fn. Pages gate page-scoped effects on `input.main_nav() == "<value>"`.
- `get_calibratable_params` (`calibration_handlers.py`) is how param keys/bounds are sourced (already
  used by the live run; the explorer does not re-derive them — it reads what the artifact stored).
- `ui/pages/calibration_charts.py:79` `make_sensitivity_chart(result, tmpl, selected_objective)`
  already renders a **vertical grouped** S1/ST bar from the identical `result` dict, with the 1-D/2-D
  dispatch the explorer also needs (`if "objective_names" in result: s1 = result["S1"][selected_objective]`,
  else `result["S1"]`). The explorer needs a **horizontal tornado** with sort-by-key, threshold
  highlighting, and `error_x` CIs — none of which that vertical helper does — so `_make_tornado` is a
  **new** builder rather than an extension; but it MUST keep the same objective-dispatch convention
  (drive 1-D vs 2-D off `n_objectives`/`objective_idx`) so the two stay consistent.

## Architecture

Three units with clean boundaries.

### 1. `osmose/calibration/sobol_io.py` (pure I/O + pure view helpers)

Artifact JSON schema (one file per result, `sobol_<safe_ts>.json`):

```json
{
  "timestamp": "2026-06-14T12:00:00",
  "source": "calibration-live",
  "n_base": 64,
  "param_names": ["species.linf.sp0", "species.k.sp0"],
  "param_bounds": [[20.0, 60.0], [0.1, 0.5]],
  "objective_names": ["Biomass RMSE"],
  "n_objectives": 1,
  "S1": [...], "ST": [...], "S1_conf": [...], "ST_conf": [...]
}
```

`S1`/`ST`/`S1_conf`/`ST_conf` are stored as nested lists (numpy → `.tolist()`); 1-D `[n_params]` for
single objective, 2-D `[n_obj][n_params]` for multi.

```python
SENSITIVITY_DIR = _PROJECT_ROOT / "data" / "history" / "sensitivity"  # mirrors RUN_HISTORY_DIR

def save_sobol_result(result: dict, *, metadata: dict, directory: Path | None = None) -> Path
def load_sobol_result(timestamp: str, directory: Path | None = None) -> dict
def list_sobol_results(directory: Path | None = None) -> list[dict]   # summaries, newest-first
def rank_rows(result: dict, objective_idx: int = 0, sort: str = "ST") -> list[dict]
def influential_keys(rows: list[dict], threshold: float) -> list[str]
def rows_to_csv(rows: list[dict]) -> str
```

- **`save_sobol_result`**: merges `result` (indices + `param_names` + optional `objective_names`/
  `n_objectives`) with `metadata` (`source`, `n_base`, `param_bounds`, and — for the single-objective
  live run, whose `analyze()` output has **no** `objective_names` key — an optional `objective_names`).
  `objective_names` is sourced `result.get("objective_names") or metadata.get("objective_names")`
  (may be `None`); fills `timestamp` = `datetime.now().isoformat()` if absent and `n_objectives` =
  `result.get("n_objectives", 1)`; converts any numpy arrays via `np.asarray(x).tolist()`; writes JSON
  to `directory or SENSITIVITY_DIR` (mkdir parents), file `sobol_<timestamp with ':'→'-'>.json`;
  returns the path.
- **`load_sobol_result`**: same path-safety as `history.load_run` (reject `..`/absolute; assert the
  resolved path stays under the dir); returns the parsed dict (arrays stay lists).
- **`list_sobol_results`**: glob `sobol_*.json`, parse each, **skip** corrupt/unparseable files
  (`# noqa: BLE001` log+continue), return lightweight summaries
  `{"timestamp","source","n_base","n_params","n_objectives","objective_names"}` (using `.get` with
  safe defaults for optional fields — `source`→`"unknown"`, `n_base`→`None`, `objective_names`→`None`,
  `n_params`→`len(param_names)`) sorted by `timestamp` descending.
- **`rank_rows`** (pure): for the chosen objective (1-D ignores `objective_idx`; 2-D indexes
  `S1[objective_idx]` etc. via `n_objectives`), build one row per param
  `{"param","s1","s1_conf","st","st_conf"}`; sort `"ST"`/`"S1"` descending or `"name"` ascending.
  Deterministic; tolerates list **or** ndarray inputs.
- **`influential_keys`** (pure): `[r["param"] for r in rows if r["st"] >= threshold]` (boundary
  `st == threshold` is included).
- **`rows_to_csv`** (pure): header `param,S1,S1_conf,ST,ST_conf` + one line per row.

### 2. Persist hook (one additive call, parity-safe)

In `handle_sensitivity` (`calibration_handlers.py`), immediately after line 1789
(`sens_result = analyzer.analyze(Y_1d)`), add a wrapped save:

```python
try:
    save_sobol_result(
        sens_result,
        metadata={
            "source": "calibration-live",
            "n_base": 64,
            "param_bounds": [[lo, hi] for lo, hi in param_bounds],
            "objective_names": obj_names_sens or None,
        },
    )
except Exception:  # noqa: BLE001 — persistence is additive; never break the live run
    _log.warning("Failed to persist sensitivity result", exc_info=True)
```

`param_names` already lives inside `sens_result`; `objective_names` is supplied via metadata because
the single-objective `analyze(Y_1d)` output omits it (`obj_names_sens` and `param_bounds` are both in
closure scope at line 1789). A save failure logs and is swallowed — the live in-memory chart
(`post_sensitivity`) is unaffected.

### 3. `ui/pages/sensitivity_explorer.py` (new top-level page)

`sensitivity_explorer_ui()` + `sensitivity_explorer_server(input, output, session, state)`,
registered in `app.py` as `ui.nav_panel("Sensitivity", sensitivity_explorer_ui(), value="sensitivity")`
placed just after the Calibration panel, with `sensitivity_explorer_server(...)` added to the server
body.

**UI (static widgets; render returns empty-state figures when no data — the shinywidgets
bind-on-insertion rule, same as `scenario_diff.py`):**
- `input_select("sens_run", ...)` — discovered artifacts; label `"<ts[:19]> (<source>, n_base=N)"`.
- `output_ui("sens_objective_ui")` — renders an `input_select("sens_objective", ...)` only when the
  selected artifact has `n_objectives > 1` (dynamic `input_select` inside `@render.ui` is fine; only
  `output_widget` must be static).
- `input_radio_buttons("sens_index", choices=["Both","S1","ST"], selected="Both")`.
- `input_slider("sens_threshold", "Influence threshold (ST)", min=0, max=1, value=0.05, step=0.01)`.
- `input_select("sens_sort", choices={"ST":"Total (ST)","S1":"First-order (S1)","name":"Name"})`.
- `output_widget("sens_tornado")` (static).
- `output_ui("sens_table")`.
- `download_button("sens_export_csv", "Download ranked CSV")` and
  `download_button("sens_export_keys", "Download influential keys")`.

**Server:**
- `_populate_runs` `@reactive.effect` gated on `input.main_nav() == "sensitivity"`: calls
  `list_sobol_results()` and `ui.update_select("sens_run", choices=...)` with a changed-only guard
  (a `reactive.Value` of the last choices, like `scenario_diff`'s `_last_choices` at
  `scenario_diff.py:119`).
- `_result` `@reactive.calc`: `load_sobol_result(_safe(input.sens_run))` → dict or `None`
  (broad-except degrade). A `_safe(getter, default=None)` helper mirrors `scenario_diff.py:126`
  (catches `SilentException`/`AttributeError`).
- `sens_objective_ui` `@render.ui`: dropdown when multi-objective, else `ui.div()`.
- `sens_tornado` `@render_plotly`: from `_result` + objective + index toggle + threshold + sort,
  build a horizontal grouped bar via a local `_make_tornado(rows, indices, threshold, template)`
  (params on y sorted by the chosen key; S1/ST bars; `*_conf` as `error_x`; `_tpl(input)` template).
  Returns an empty-state figure ("Select a result" / "No sensitivity results yet …") when `_result`
  is `None`.
- `sens_table` `@render.ui`: Bootstrap table (`table table-sm table-striped`, `STYLE_MONO_KEY` for
  the param column, like `config_diff_table`) — columns `Param | S1 | ST | Influential`, influential
  rows badged (`badge bg-success`); empty state when no data.
- `sens_export_csv` `@render.download(filename="sensitivity_ranked.csv")`: yields
  `rows_to_csv(rank_rows(...))` (static filename, matching the `@render.download(filename=...)` idiom
  at `calibration.py:607`).
- `sens_export_keys` `@render.download(filename="influential_keys.txt")`: yields newline-joined
  `influential_keys(...)`.

## Data flow

```
list_sobol_results()  ──► sens_run selector
        │ (selected timestamp)
        ▼
load_sobol_result(ts) ──► _result (reactive.calc: dict | None)
        │
        ▼
rank_rows(result, objective_idx, sort) ──► rows  (PURE)
        ├─► _make_tornado(rows, indices, threshold, template)  → go.Figure  (sens_tornado)
        ├─► sens_table (render.ui)
        ├─► rows_to_csv(rows)              → sens_export_csv
        └─► influential_keys(rows, thr)    → sens_export_keys
```

No NetCDF/engine involvement; config-independent. Producer (persist hook) and consumer (page) are
decoupled by the on-disk artifact, exactly like run `history.py`.

## Error handling

- Discovery/load: `list_sobol_results` skips corrupt files (`# noqa: BLE001` log+continue);
  `_result` degrades to `None` on any load error → muted empty state; the page never crashes
  (matches `_populate_diff_runs`/`config_diff_table`).
- Persist hook: wrapped, swallowed, logged — never breaks the live calibration run.
- Malformed/partial artifact (missing index keys) → unloadable → "Could not load this result."
- Multi-objective `objective_idx` clamped to `[0, n_objectives)`.
- numpy↔list handled in `sobol_io` (`rank_rows` accepts either; save stores lists).
- Threshold fixed to `[0, 1]` (Sobol indices live in ~[0,1]).
- `load_sobol_result` rejects unsafe timestamps (`..`/absolute), like `history.load_run`.

## Testing

1. **Unit `tests/test_sobol_io.py`** (pure):
   - save→load round-trip for **1-D single-objective** and **2-D multi-objective**; numpy inputs
     stored as lists; metadata (`source`/`n_base`/`param_bounds`/`timestamp`) preserved.
   - `list_sobol_results` returns newest-first summaries and **skips a deliberately corrupt** file.
   - `rank_rows`: correct rows; sort by ST (default), S1, name; multi-objective `objective_idx`
     selects the right row; accepts list and ndarray inputs.
   - `influential_keys`: threshold filter with boundary `st == threshold` included.
   - `rows_to_csv`: header + values.
   - `load_sobol_result` raises on an unsafe timestamp (`"../x"`).
   - All file tests use a `tmp_path` `directory=` argument (no reliance on the real dir).
2. **Structure tests** (`tests/test_ui_*` — source-grep, the repo's wiring-test idiom):
   - `app.py` imports `sensitivity_explorer_ui`/`sensitivity_explorer_server`, registers the
     `value="sensitivity"` nav panel, and calls the server.
   - `calibration_handlers.py` calls `save_sobol_result` (persist hook wired).
   - `str(sensitivity_explorer_ui())` contains the **full** widget id set — `sens_run`,
     `sens_objective_ui`, `sens_index`, `sens_threshold`, `sens_sort`, `sens_tornado`, `sens_table`,
     `sens_export_csv`, `sens_export_keys` — so a future edit dropping any control fails the test.
     (A top-level page returns tags, so `str(...)` tagifies — unlike a bare `NavPanel`.)
3. **e2e `tests/test_e2e_sensitivity_explorer.py`**: write a synthetic `sobol_<ts>.json` into the
   real `data/history/sensitivity/` (same on-disk substrate approach as the scenario-diff e2e),
   navigate to the Sensitivity page, select the run, assert the tornado widget (`#sens_tornado`) is
   visible and the table (`#sens_table`) **contains** a known param key (`to_contain_text`, not
   visibility — the bare `@render.ui` is zero-height until it recomputes), and the export buttons are
   present. The fixture mirrors `test_e2e_scenario_diff.py:73-82` **with one difference** —
   `data/history/sensitivity/` does **not** exist yet (it is gitignored under `data/history/` and is
   currently empty), so the fixture MUST: (a) `SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)`
   before writing; (b) use the `yield`-fixture form with cleanup **after** the yield so teardown runs
   even when an assertion fails; (c) `path.unlink(missing_ok=True)` the written `sobol_*.json` — so a
   failed test never leaks a stale artifact that `list_sobol_results` would surface in a later human
   session. (Leaving the now-empty `sensitivity/` dir is fine — it is gitignored.)

**Substrate:** synthetic JSON with real-looking keys (`species.linf.sp0`, `species.k.sp0`,
`predation.efficiency.sp0`) in single- and multi-objective variants; real data arrives via the
persist hook going forward. **Gates:** every task runs `ruff check`, `ruff format --check`, **and**
`pyright`. **CHANGELOG** entry included.

## Out of scope (YAGNI)

- **Side-by-side comparison** of two sensitivity runs (v2 — doubles layout/state for marginal v1
  value; the selector + single-view is enough to "surface Sobol output").
- **Run-it-here** from this page (the Calibration sub-tab already runs Sobol live; duplicating the
  run machinery is exactly what this design avoids).
- **Batch-script alignment** — `scripts/sensitivity_phase12.py` keeps its own outputs; teaching it to
  also call `save_sobol_result` is a trivial future follow-up, not part of v1.
- **Morris preflight UI** (`preflight.py`) — separate method, separate feature.
- **Configurable `n_base`/seed** in the live calibration run — orthogonal to the explorer.
- Any change to `SensitivityAnalyzer`.
