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
  X_ui(), value="x")` in the `navset_pill_list` (id kwarg `id="main_nav"` at `app.py:286`; the new
  panel goes just after the Calibration `nav_panel` at `app.py:280`) → `X_server(...)` in the server
  fn. Pages gate page-scoped effects on `input.main_nav() == "<value>"`.
- `get_calibratable_params` (`calibration_handlers.py`) is how param keys/bounds are sourced (already
  used by the live run; the explorer does not re-derive them — it reads what the artifact stored).
- `ui/pages/calibration_charts.py:79` `make_sensitivity_chart(result, tmpl, selected_objective)`
  already renders a **vertical grouped** S1/ST bar from the identical `result` dict, with the 1-D/2-D
  dispatch (`if "objective_names" in result: s1 = result["S1"][selected_objective]`, else
  `result["S1"]`). The explorer needs a **horizontal tornado** with sort-by-key, threshold
  highlighting, and `error_x` CIs — none of which that vertical helper does — so a **new**
  `make_sobol_tornado` builder is added alongside it (see §4), and `rank_rows` (in `sobol_io`) is new.
  **Critically, the explorer dispatches 1-D vs 2-D on `int(n_objectives) > 1`, NOT on
  `make_sensitivity_chart`'s `"objective_names" in result` check** — these are *deliberately different*
  predicates. A persisted single-objective artifact carries `objective_names` (the live hook supplies
  it via metadata, see §2) **yet is 1-D**, so the key-presence check would wrongly try to index a flat
  `S1` by objective and return a scalar. The explorer does its 1-D/2-D selection in `rank_rows`; the
  tornado builder consumes the already-selected rows and never re-dispatches.

## Architecture

Four units with clean boundaries (the persist hook is a one-line addition to existing code; §1/§3/§4
are new).

### 1. `osmose/calibration/sobol_io.py` (pure I/O + pure view helpers)

Artifact JSON schema (one file per result, `sobol_<safe_ts>.json`):

```json
{
  "timestamp": "2026-06-14T12:00:00.123456",
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
single objective, 2-D `[n_obj][n_params]` for multi. `param_bounds` is **provenance only** — stored
for the record of which params/bounds produced the result; **no v1 widget reads it** (don't hunt for
a consumer). The example `timestamp` includes microseconds deliberately (see the collision note under
`save_sobol_result`).

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
  `int(result.get("n_objectives", 1))`; converts any numpy arrays via `np.asarray(x).tolist()`;
  `param_bounds` and any other metadata are stored **verbatim** (a list of `(lo, hi)` tuples
  serializes to a list of 2-lists — round-trips as lists, see Testing). Writes JSON to
  `directory or SENSITIVITY_DIR` (mkdir parents), file `sobol_<timestamp with ':'→'-'>.json`.
  **Collision-safe:** if that path already exists (two saves in the same second when `isoformat()`
  drops the microsecond field at `microsecond == 0`), append a `-<n>` counter before `.json` until the
  name is free, so a save never silently overwrites an earlier result. Returns the written path.
- **`load_sobol_result`**: **validate the raw `timestamp` BEFORE building the filename** — reject any
  value containing `/`, `\`, or `..` (raise `ValueError`). Only then apply the `:`→`-` transform
  (matching `history.load_run:63`) and build `sobol_<safe>.json`. (Do NOT rely on
  `history.load_run`'s `any(part == "..")` / `is_relative_to` checks alone: the `sobol_` prefix fuses
  with a leading `..`, so `parts` becomes `("sobol_..", "x.json")` and the bare-`..` check never fires
  — `load_sobol_result("../x")` would NOT raise. Up-front validation is required and is what the unit
  test asserts.) Returns the parsed dict (arrays stay lists).
- **`list_sobol_results`**: glob `sobol_*.json`, parse each, **skip** corrupt/unparseable files
  (`# noqa: BLE001` log+continue), return lightweight summaries
  `{"timestamp","source","n_base","n_params","n_objectives","objective_names"}` (using `.get` with
  safe defaults for optional fields — `source`→`"unknown"`, `n_base`→`None`, `objective_names`→`None`,
  `n_params`→`len(param_names)`) sorted by `timestamp` descending.
- **`rank_rows`** (pure): the result is **2-D iff `int(result.get("n_objectives", 1)) > 1`**; in that
  case index `S1[objective_idx]` / `ST[...]` / `*_conf[...]` with `objective_idx` clamped to
  `[0, n_objectives)`; otherwise use the arrays directly and **ignore** `objective_idx`. Apply
  `np.asarray(...)` first so list and ndarray inputs index uniformly. Build one row per param
  `{"param","s1","s1_conf","st","st_conf"}` (floats). Sort `"ST"`/`"S1"` **descending** or `"name"`
  ascending, with a **NaN-safe key**: treat NaN indices as `-inf` for sorting so NaN params sink
  deterministically to the bottom (SALib can emit NaN when the summed objective is contaminated by the
  live run's `inf` failure path). Deterministic.
- **`influential_keys`** (pure): `[r["param"] for r in rows if r["st"] >= threshold]` (boundary
  `st == threshold` is included; `NaN >= threshold` is naturally `False`, so NaN-ST params are not
  flagged influential — intentional).
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
body. (`state` is kept for the standard `*_server(input, output, session, state)` call signature but
is **unused** — the page is config-independent; don't wire it.)

**UI (static widgets; render returns empty-state figures when no data — the shinywidgets
bind-on-insertion rule, same as `scenario_diff.py`):**
- `input_select("sens_run", ...)` — discovered artifacts; label `"<ts[:19]> (<source>, n_base=N)"`.
- `output_ui("sens_objective_ui")` — renders `input_select("sens_objective", choices=...)` only when
  the selected artifact has `int(n_objectives) > 1` (dynamic `input_select` inside `@render.ui` is
  fine; only `output_widget` must be static). Choices:
  `{str(i): (names[i] if names and i < len(names) else f"obj_{i}") for i in range(n_objectives)}`
  where `names = result.get("objective_names")` (handles `names is None`), `selected="0"`.
- `input_radio_buttons("sens_index", choices=["Both","S1","ST"], selected="Both")`.
- `input_slider("sens_threshold", "Influence threshold (ST)", min=0, max=1, value=0.05, step=0.01)`.
  (Sobol `ST` can slightly exceed 1 at low `n_base`/noisy objectives; the `[0,1]` slider is a
  deliberate v1 simplification — a param with `ST > 1` is simply always flagged influential, which is
  the intent. Not a hard cap on the *index*, only on the chosen *threshold*.)
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
- `_result` `@reactive.calc`: `load_sobol_result(_safe(input.sens_run))` → dict or `None` (broad-except
  degrade; also `None` when nothing is selected). A `_safe(getter, default=None)` helper mirrors
  `scenario_diff.py:126` (catches `SilentException`/`AttributeError`).
- **`_rows` `@reactive.calc`** — the single source of ranked rows reused by the chart, table, and BOTH
  downloads (so they never re-read raw inputs in a download generator, which would risk
  `SilentException`): `r = _result(); return [] if r is None else rank_rows(r, objective_idx=_obj_idx(r), sort=_safe(input.sens_sort, "ST"))`,
  where `_obj_idx(r) = max(0, min(int(_safe(input.sens_objective, "0") or 0), int(r.get("n_objectives", 1)) - 1))`
  (`clamp` is not a Python builtin — inline the `max(0, min(...))`).
- `sens_objective_ui` `@render.ui`: the dropdown (choices per the UI section) when `_result()` is
  multi-objective, else `ui.div()`.
- `sens_tornado` `@render_plotly`: if `_result() is None` → empty-state figure (one message, see
  below); else `make_sobol_tornado(_rows(), indices=_safe(input.sens_index,"Both"), threshold=_safe(input.sens_threshold,0.05), template=_tpl(input))`
  (the builder is specified in §4).
- `sens_table` `@render.ui`: if `_result() is None` → empty-state `ui.p(..., class_="text-muted")`;
  else a Bootstrap table (`table table-sm table-striped`, `STYLE_MONO_KEY` for the param column, like
  `config_diff_table`) from `_rows()` — columns `Param | S1 | ST | Influential`, rows whose key is in
  `influential_keys(_rows(), _safe(input.sens_threshold,0.05))` badged `badge bg-success`.
- **Empty state** is a single message (both "nothing selected" and "load failed" collapse to
  `_result() is None`, since a load error is logged then degraded to `None`): *"No sensitivity result
  to display — pick a saved run above, or run one from Calibration → Results → Sensitivity."* (No
  separate "could not load" copy — there is no reactive path to distinguish it, and load failures are
  logged.)
- `sens_export_csv` `@render.download(filename="sensitivity_ranked.csv")`: `yield rows_to_csv(_rows())`
  (static filename, matching the `@render.download(filename=...)` idiom at `calibration.py:607`).
- `sens_export_keys` `@render.download(filename="influential_keys.txt")`:
  `yield "\n".join(influential_keys(_rows(), _safe(input.sens_threshold, 0.05)))`.

### 4. `make_sobol_tornado` (new figure builder in `ui/pages/calibration_charts.py`)

Lives next to `make_sensitivity_chart` (so the two Sobol chart builders are co-located and tested
together). It consumes **already-ranked rows** (from `rank_rows`) — it does **no** 1-D/2-D dispatch
and no I/O, which makes it pure and unit-testable:

```python
def make_sobol_tornado(
    rows: list[dict],            # [{param, s1, s1_conf, st, st_conf}, ...] in display order
    *,
    indices: str = "Both",       # "Both" | "S1" | "ST"
    threshold: float = 0.05,
    template: str = "osmose",
) -> go.Figure
```

Behavior:
- **Orientation `"h"`**, `y = [r["param"] for r in rows]`. Plotly draws the first y last, so reverse
  the row order (or set `yaxis autorange="reversed"`) so the top-ranked param appears at the top.
- **Traces by `indices`**: `"Both"` → two `go.Bar` (S1 then ST); `"S1"` → only the S1 bar; `"ST"` →
  only the ST bar. Each bar's `x` is the matching value list, `error_x=dict(type="data", array=<the
  matching *_conf list>)`.
- **Threshold highlighting**: params with `st >= threshold` are visually marked — use a per-point
  marker color on the **ST** bar (influential = a highlight color, others = the muted default) AND an
  `add_vline(x=threshold, line_dash="dash")` reference line. (NaN `st` is treated as not influential,
  consistent with `influential_keys`.)
- `update_layout(barmode="group", template=template, title="Sobol sensitivity")`. Returns the
  `go.Figure`. Empty `rows` → a figure with no bars (the page only calls it for non-empty `_rows`,
  but it must not raise on `[]`).

## Data flow

```
list_sobol_results()  ──► sens_run selector
        │ (selected timestamp)
        ▼
load_sobol_result(ts) ──► _result (reactive.calc: dict | None)
        │
        ▼
_rows (reactive.calc) = rank_rows(_result(), objective_idx, sort)   (PURE rank_rows; [] when None)
        ├─► _make_tornado(_rows(), indices, threshold, template)  → go.Figure  (sens_tornado)
        ├─► sens_table (render.ui)
        ├─► rows_to_csv(_rows())              → sens_export_csv
        └─► influential_keys(_rows(), thr)    → sens_export_keys
```

No NetCDF/engine involvement; config-independent. Producer (persist hook) and consumer (page) are
decoupled by the on-disk artifact, exactly like run `history.py`.

## Error handling

- Discovery/load: `list_sobol_results` skips corrupt files (`# noqa: BLE001` log+continue);
  `_result` degrades to `None` on any load error (logged) AND when nothing is selected → both render
  the **single** muted empty state; the page never crashes (matches `_populate_diff_runs`/
  `config_diff_table`). There is no separate "could not load" message (no reactive path distinguishes
  it from "nothing selected").
- Persist hook: wrapped, swallowed, logged — never breaks the live calibration run.
- Malformed/partial artifact (missing index keys) → `load_sobol_result`/`rank_rows` raise → caught by
  `_result`'s broad-except → `None` → empty state.
- `objective_idx` clamped to `[0, n_objectives)` (in `_obj_idx` and again defensively in `rank_rows`).
- NaN indices: `rank_rows` sorts NaN to the bottom (NaN→`-inf` key); `influential_keys` excludes them.
- numpy↔list handled in `sobol_io` (`rank_rows` `np.asarray`s first; save stores lists; tuples in
  metadata like `param_bounds` serialize to lists).
- Threshold slider is `[0, 1]` by design (see UI note); `ST > 1` params are always flagged influential.
- `load_sobol_result` validates the raw timestamp (reject `/`, `\`, `..`) **before** prefixing — not
  via `history.load_run`'s post-prefix part-check, which the `sobol_` prefix defeats for a leading `..`.

## Testing

1. **Unit `tests/test_sobol_io.py`** (pure):
   - save→load round-trip for **1-D single-objective** and **2-D multi-objective**; numpy inputs
     stored as lists; metadata preserved — assert `param_bounds` against the **list-of-lists** form
     (e.g. `[[20.0, 60.0], ...]`), since tuples passed in serialize to lists through JSON (do NOT
     assert against the tuple input).
   - **collision-safety**: two saves with the same `timestamp` (pass an explicit second-resolution
     timestamp) produce two distinct files (the second gets a `-1` suffix), neither overwritten.
   - `list_sobol_results` returns newest-first summaries and **skips a deliberately corrupt** file;
     summaries tolerate a single-objective artifact whose `objective_names` is `None`.
   - `rank_rows`: correct rows; sort by ST (default), S1, name; **dispatch on `n_objectives`** — a 1-D
     artifact that *also* carries `objective_names` (the live-run shape) is treated as 1-D (ignores
     `objective_idx`), and a 2-D artifact indexes the right objective; `objective_idx` out of range is
     clamped; accepts list and ndarray inputs.
   - **NaN handling**: a result with a NaN `ST` ranks that param last (deterministically) and
     `influential_keys` excludes it.
   - `influential_keys`: threshold filter with boundary `st == threshold` included.
   - `rows_to_csv`: header + values.
   - `load_sobol_result` raises `ValueError` on an unsafe timestamp **`"../x"`** (the single-`..`
     case the prefix would otherwise mask) and on `"/abs"`; and round-trips a real colon-bearing
     timestamp (proving the `:`→`-` transform is applied on both save and load).
   - All file tests use a `tmp_path` `directory=` argument (no reliance on the real dir).
2. **Unit `tests/test_sobol_tornado.py`** (pure figure builder — `make_sobol_tornado` returns a
   `go.Figure`, no browser needed; build `rows` by hand):
   - `indices="Both"` → `len(fig.data) == 2`; `"S1"` → 1 (the S1 bar only); `"ST"` → 1 (the ST bar
     only).
   - every bar has `orientation == "h"` and a populated `error_x.array` from the matching `*_conf`.
   - threshold highlighting marks exactly the params with `st >= threshold` (assert the ST bar's
     per-point marker colors, or the influential set) and adds the `threshold` reference line.
   - `make_sobol_tornado([])` returns a figure with no bars (does not raise).
3. **Structure tests** (`tests/test_ui_*` — source-grep, the repo's wiring-test idiom):
   - `app.py` imports `sensitivity_explorer_ui`/`sensitivity_explorer_server`, registers the
     `value="sensitivity"` nav panel, and calls the server.
   - `calibration_handlers.py` calls `save_sobol_result` (persist hook wired).
   - `str(sensitivity_explorer_ui())` contains the **full** widget id set — `sens_run`,
     `sens_objective_ui`, `sens_index`, `sens_threshold`, `sens_sort`, `sens_tornado`, `sens_table`,
     `sens_export_csv`, `sens_export_keys` — so a future edit dropping any control fails the test.
     (A top-level page returns tags, so `str(...)` tagifies — unlike a bare `NavPanel`.)
4. **e2e `tests/test_e2e_sensitivity_explorer.py`**: write a synthetic `sobol_<ts>.json` into the
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
