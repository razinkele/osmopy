# Backlog Plan: CI Matrix, Narrative Docs, Per-Cell NetCDF Viewer, Pareto Picker

**Date:** 2026-06-13
**Status:** Proposed — revised after Review Rounds 1 & 2; strategic decisions settled
**Author:** planning session (master `02c43d3`)

**Settled decisions (post-Round-2):** (1) **Sphinx API docs → replaced with narrative-doc work**
(README usage + tutorial polish for the scientist audience) — no autodoc, no `sphinx`/`furo`/`myst`
deps, no `-W` gate, no `cma` issue. (2) Per-cell viewer → **full time-series panel** (Stage 3 as
specified). (3) Scenario-diff view → **kept in original order** (next-up after this plan; not inserted
here), but Stage 3's `spatial_series.py` + Playwright harness are designed for its later reuse.

Organises four backlog items into a dependency-ordered, multistage implementation
plan. Each stage ships as its own fast-forward PR following the repo's established
methodology: spec → plan-review (in-loop) → TDD / subagent-driven build →
whole-feature review → (UI: Playwright validation) → CHANGELOG entry → merge.

## Review Round 1 — findings incorporated (2026-06-13)

Four parallel executing reviewers (factual / scope-sequencing / architecture /
testing) opened the actual code, configs, and `.nc` files. Two BLOCKERs and several
MAJORs were confirmed by 3–4 reviewers each. Changes folded into this revision:

- **B1 (Stage 4):** The NSGA-II *front* is **not** in `CalibrationCheckpoint` (it stores
  only the single best `best_x_log10`/`best_parameters`/`best_fun`). No production NSGA-II
  runner persists a population; the lone `NSGA2` call (`scripts/benchmark_calibration.py:29-58`)
  discards its result. → **Rescoped:** load the front from the **history run record**
  (`results.pareto_X`/`pareto_F`/`objective_names` written by `save_run` at
  `ui/pages/calibration_handlers.py:1278-1300`), and from the surrogate `optimum["pareto"]`.
  Fixture = hand-built history JSON, not a `minimize()` result. Effort raised.
- **B2 (Stage 3):** `output.spatial.enabled=false` in **both** EEC and Baltic
  (`eec_param-output.csv:51`, `baltic_param-output.csv:52`); the only shipped `.nc` files are
  `*_yieldByFishery` with dims `(time, species, fishery)` — **no lat/lon**. There is no
  per-cell spatial NetCDF to demo against. → **Rescoped:** both fixtures are BYO; added an
  explicit fixture phase (synthetic `(time, species, lat, lon)` NetCDF for units + a
  documented spatial-enabled re-run for integration/Playwright). Backend stays feasible.
- **Sphinx (MAJOR):** package-wide `sphinx-build -W` would explode (~648 funcs/classes, 99
  files, mixed Google+NumPy docstrings). Autodoc *imports* modules, and several public
  modules import `pymoo`/`SALib`/`cma`/`xarray` at module scope. → scope autodoc to a
  curated public-module allowlist (exclude `engine/`); docs CI installs `.[dev,docs]`;
  enable **both** napoleon styles; apply `-W` only to the curated set.
- **Coverage (MAJOR):** CI runs `--cov=osmose` — `ui/` is **not measured**. → the ≥90% gate
  binds only the new backend modules; UI panels are validated by Playwright, not coverage.
- **Playwright harness (MAJOR):** the only spatial e2e test asserts the *disabled* pill; no
  harness loads a completed run and interacts. → both UI stages need a new fixture-injection
  path into the running app — now an explicit sub-task, not an afterthought.
- **deck.gl click (MINOR, good news):** `shiny_deckgl` 1.6.1 exposes `click_input_id`; cells
  already carry `row`/`col` and `pickable=True`. Click is **supported**, not uncertain →
  Stage 3 risk downgraded; numeric pickers demoted to optional fallback.
- **Reuse (MINOR):** `_non_dominated_indices` (`surrogate.py:12`) and a parallel-coords chart
  (`make_correlation_chart`) already exist → reuse, don't rebuild.
- **Sequencing (MINOR):** infra-first rationale was oversold (test job already runs
  3.12+3.13; only lint/type-check extend). Floor decision settled: **keep 3.12**.
  "Parallel" stages are authoring-parallel only — fast-forward merges are serial.
- **CHANGELOG (MAJOR):** added as a cross-cutting per-stage gate (project commits one per feature).

## Review Round 2 — findings incorporated (2026-06-13)

Four NEW angles (adversarial-verifier / product-value / maintenance-cost / cross-feature-integration),
each executing against the code. Two findings are scope-changing:

- **Stage 4 is ~70% already shipped (BLOCKER-equivalent, integration reviewer).** The calibration
  page ALREADY loads a persisted front from `calibration.history` into `cal_F`/`cal_X`
  (`calibration.py:371-372`), driving `pareto_chart` (`:486-494`), parallel-coords
  `correlation_chart` (`:496-505`, handles ≥3 obj), and `best_params_table` (`:507-519`). The
  History tab's `btn_load_run_{i}` handler already does `cal_X.set(np.array(data["results"]
  ["pareto_X"]))` etc. (`calibration_handlers.py:1646-1655`). → **Stage 4 rescoped to EXTEND this
  existing UI** (per-solution picker + N-objective scatter + config export), NOT a parallel
  loader/panel. The standalone `osmose/calibration/pareto.py` loader is largely redundant —
  reuse `load_run` + the existing `cal_F`/`cal_X` setter path. This roughly halves Stage 4.
- **`cma` autodoc import break (adversarial verifier — repeats the v1 mistake).** `cmaes_runner.py:34`
  does `import cma` at module scope, but `cma` is declared NOWHERE in `pyproject.toml`. The curated
  allowlist includes `calibration/*` → autodoc imports `cmaes_runner` → `ModuleNotFoundError` on
  clean CI. → Stage 2 fix: add `cma` to the `docs`/base extra OR exclude `cmaes_runner` from the
  autodoc allowlist. Also corrected: the surrogate `optimum["pareto"]` is never persisted, and the
  surrogate run's saved `pareto_X/F` is the full LHS sample pool, not a front — so "load from the
  surrogate dict" is dropped; only the NSGA-II history record is a real loadable front.

Folded-in corrections (non-scope-changing):
- **Sphinx `-W` → non-blocking (maintenance MAJOR).** Land `sphinx-build` without `-W` and keep it
  non-blocking; a warnings-as-errors docstring gate over ~648 mixed-dialect members will be
  resented/disabled on a science codebase. `pdoc` (zero-config, no stub/toctree upkeep) noted as a
  lighter alternative. (Moot after the settled decision: Sphinx replaced with narrative docs.)
- **Drop multi-version lint (maintenance MAJOR).** `ruff` is version-independent → multi-version lint
  is pure recurring runner-minutes (worsened by the two git-URL deps rebuilding from source, no wheel
  cache). Keep `lint` single-version; multi-version `type-check` is marginal — include only if 3.13
  stub drift has actually bitten.
- **Fixtures = programmatic factories, NOT hand-authored JSON (maintenance MINOR).** The repo's
  universal pattern is factory construction (`_make_ltl_nc()` in `test_overlay_display.py`,
  `_valid_checkpoint_kwargs()` in `test_calibration_checkpoint.py`). Build the synthetic NetCDF and
  the calibration-history fixtures via factories routed through `save_run`/`write_checkpoint` so a
  schema change breaks at the source, not in a stale literal.
- **Stage 3 empty-state guard (integration MAJOR).** The dominant runtime state is "run completed +
  `output.spatial.enabled=false`" → no lat/lon NetCDF. The per-cell panel must reuse the existing
  lat/lon-var filter (`spatial_results.py:58,241`) and show a "no per-cell spatial data — enable
  `output.spatial.enabled`" empty state, not render against `yieldByFishery` (species/fishery dims).
- **Theme dependency in the render body (integration NOTE).** New charts must read
  `get_theme_mode(input)` inside the `@render_plotly`/effect body (like `spatial_flat_chart`
  `:299-301`), not a precomputed value, or they desync from the deck.gl basemap on theme toggle.
- **Coverage: sanction `# pragma: no cover` / omit (maintenance MINOR).** For genuinely untestable
  NetCDF/JSON I/O error branches, use a commented pragma/omit (precedent: `pyproject.toml:68-78`),
  not contrived tests, to hold the `--cov=osmose` ≥90% gate.
- **Shared substrate (integration MINOR).** `osmose/spatial_series.py` (NetCDF open + dim detection +
  cell selection) and the new Playwright fixture-injection harness are the same substrate the queued
  **scenario-diff view** will need for side-by-side spatial maps — build once, design for reuse.

Out-of-scope but flagged: **`MEMORY.md` is 38.7 KB vs a 24.4 KB limit** → future sessions load an
incomplete index. (RESOLVED 2026-06-13: compacted to ~9 KB.)

> **Strategic decisions — RESOLVED below** (see "Strategic decisions — SETTLED"): whether to keep
> Sphinx (→ replaced with narrative docs), simplify the per-cell viewer (→ kept full), and promote
> the scenario-diff view (→ kept next-up).

## Reconnaissance — what already exists (target the gap, not greenfield)

| Item | Existing | Real gap |
|------|----------|----------|
| **CI Python matrix** | `test` job already runs `["3.12","3.13"]` (`ci.yml:41-43`); coverage upload pinned to 3.12 (`:64`). | `lint` (`:17`), `type-check` (`:35`), `docker` (`:70-75`, no setup-python) are single-version. `requires-python=">=3.12"` (`pyproject.toml:5`); no syntax blocks a lower floor but no consumer needs one. |
| **Pareto explorer** | `make_pareto_chart(F, obj_names)` 2D scatter (`calibration_charts.py:58-76`); `make_correlation_chart` parallel-coords (`:119-124`); `_non_dominated_indices` (`surrogate.py:12`); a pymoo `Problem` subclass (`problem.py:127`). The front is persisted by the UI run path as `results.pareto_X/pareto_F/objective_names` via `save_run` (`calibration_handlers.py:1278-1300`). | No *interactive* explorer: no front→solution selection, no param/objective inspector, no ≥3-objective view, no config export. **The checkpoint does NOT hold the front** — load from the history run record / surrogate dict instead. |
| **Per-cell NetCDF viewer** | `spatial_results.py`/`map_viewer.py` render spatial **snapshots**; `grid_helpers.py` has `load_netcdf_overlay`/`list_nc_overlay_variables` + `_NC_COORD_NAMES`/`_NC_TIME_DIM_NAMES` dim detection (`:911-937`); `OsmoseResults.read_netcdf` opens lazily (`results.py:387-390`). Engine can write `(time, species, lat, lon)` via `write_outputs_netcdf_spatial` (`output.py:689`). | No **per-cell time-series** extraction. **No shipped config enables spatial output** (`output.spatial.enabled=false` in EEC & Baltic) → no real `.nc` to test against; fixtures must be made. |
| **Narrative docs** *(was Sphinx API docs)* | `docs/` has tutorials (`tutorials/30-minute-ecosystem.md`), `baltic_example.md`, and many diagnostic write-ups; no top-level "how do I script a run / read outputs" usage guide; README is the entry point. | A task-oriented **usage guide** for the scientist audience (run a config, read outputs, calibrate, compare runs) + tutorial/README polish — NOT autodoc of internal APIs. |

## Dependency graph & recommended order

```
Stage 1  CI Python matrix       (infra, ~1 file)               ─┐ infra/docs-first:
Stage 2  Narrative usage docs   (docs, scientist-facing)       ─┘ small + low-risk, get them done
Stage 3  Per-cell NetCDF viewer (feature, new backend + UI)    ─┐ ordered by user-facing breadth
Stage 4  Pareto picker          (feature, extends existing UI) ─┘ (mostly reuse)
```

Order retained per the chosen infra-first preference, but the **honest** rationale is
"infra is small, low-risk, and worth clearing first" — not "it protects the features"
(the high-value `test` matrix already covers 3.13). Stages 1 and 2 are independent and can
be *authored* in parallel, but their PRs **merge serially** (fast-forward requires rebasing
the second onto the first). Stage 4 now has a prerequisite (front persistence) — see below.

---

## Stage 1 — CI Python matrix completion

**Goal:** every CI job runs the supported Python range with a clearly declared floor.

**Settled decisions** (no spec debate needed):
- **Keep the 3.12 floor.** No PEP-695 `type` aliases or other 3.12-only syntax in `osmose/`,
  so 3.11 is *syntactically* viable, but `requires-python=">=3.12"` and the stated "3.12+"
  convention stand and no consumer needs 3.11. Revisit only on concrete demand.
- **No OS matrix yet.** macOS adds Java + numba + netCDF4 + git-dep source-build risk for
  little gain; note as explicitly deferred.

**Phases**
1. **Keep `lint` single-version** (Round 2: `ruff` is version-independent → multi-version lint is pure
   recurring runner-minutes for zero signal, worsened by the git-URL deps rebuilding from source with
   no wheel cache). Lift `type-check` to the matrix **only if** 3.13 pyright stub drift has actually
   bitten — otherwise the 3.13 `test` leg already exercises 3.13 at runtime. `docker` stays
   single-version (image pins its runtime).
2. Run on a branch. Note the two **git-URL deps** (`shiny_deckgl`, `pyvis`) build from source on every
   leg (no wheel cache); a 3.13 source-build failure is possible but unproven. Surface any 3.13
   diagnostic, don't suppress.
3. Respect existing intentional pins (`shiny<1.6`, the pyright pin — both from the recent CI-red fix).

> Round 2 reframe: this stage is **housekeeping, not value** — the high-value `test` matrix already
> covers 3.13. Its merit is that it's nearly free; do not let "it's first and easy" anchor priority.

**Deliverables:** updated `.github/workflows/ci.yml`; CHANGELOG entry.
**Acceptance:** all matrix legs (lint × {3.12,3.13}, type-check × {3.12,3.13}, test × {3.12,3.13},
docker) green on the PR branch; coverage gate still met on the pinned 3.12 leg. **Risk:** low.

---

## Stage 2 — Narrative usage docs (replaces Sphinx API docs)

**Decision (settled):** the product + maintenance reviews rated autodoc of internal APIs weakest for
this project's audience (UI-using fisheries/ecosystem scientists, not library importers). Replaced
with **task-oriented narrative documentation** that serves real users — no `sphinx`/`furo`/`myst`
deps, no autodoc, no `-W` doc-rot gate, no `cma` import problem.

**Goal:** a top-level **usage guide** answering the questions a scientist actually has, plus polish of
the existing tutorial/README, all in plain Markdown (renders on GitHub today; no build step).

**Scope — write/refresh these:**
- **`docs/usage-guide.md`** (new): end-to-end task recipes — (a) run a config from the CLI and from
  the UI; (b) read outputs (`OsmoseResults`: biomass/abundance/yield CSV + NetCDF) into pandas/xarray;
  (c) run a calibration and read the result; (d) compare two runs (Compare-Runs + `run_delta`);
  (e) the dual-engine choice (Python vs Java, parity caveats from the RNG note in `engine/rng.py`).
  Each recipe is a copy-pasteable snippet verified against the real API.
- **README polish:** ensure the entry path (install → run example → open UI → where docs live) is
  current and points to the usage guide and the 30-minute tutorial.
- **`docs/` index/README:** a short map of the existing docs (tutorials, `baltic_example`, the
  diagnostic write-ups) so the growing `docs/` folder is navigable.

**Phases**
1. Inventory existing docs and the public entry points actually used (`osmose.runner`, `OsmoseResults`,
   `osmose.scenarios`, the calibration entry points) so recipes match real signatures.
2. Draft `docs/usage-guide.md` recipes; **verify every snippet runs** against the current API (don't
   ship aspirational code) — use the EEC/Baltic example configs.
3. Polish README + add the `docs/` index map.
4. **scientific-validation** pass on any cited methods/claims in the guide (the project has a skill
   for this); no CI build job needed.

**Deliverables:** `docs/usage-guide.md`, README edits, `docs/` index, CHANGELOG entry.
**Acceptance:** every code snippet in the usage guide executes successfully against the example
configs (objective check — run them); README links resolve. **Risk:** low — pure docs, no deps, no
build gate, no parity surface. **Effort:** ~1 day.

---

## Stage 3 — Per-cell spatial NetCDF viewer

**Goal:** select a grid cell on a spatial map and see that cell's value over time for a chosen
NetCDF variable; optional CSV export of the series.

**Decisions:**
- **Click-to-select is the primary UX and is supported** — `shiny_deckgl` 1.6.1 exposes
  `MapWidget.click_input_id` → `{id}_click` with `{object:{row,col,...}}`, and the existing
  spatial polygon cells already set `row`/`col` + `pickable=True` (`spatial_results.py:399-417`).
  Numeric row/col pickers become an optional accessibility fallback, not the primary path.
- Extra dims (species, size class): expose a reduce selector (pick-one / sum / mean).
- Add as a **third `nav_panel` inside the existing `navset_card_tab`** in `spatial_results.py`
  — NOT a sibling div inside `osm-split-layout` (that flex seam squished a prior feature; MEMORY).

**Phases**
1. **Fixtures first (new, was missing):** author a synthetic `(time, species, lat, lon)` NetCDF
   fixture for unit tests; document a spatial-enabled re-run (`output.spatial.enabled=true`)
   producing `{prefix}_spatial_biomass_Simu0.nc` for the integration/Playwright leg. Guard any
   real-artifact test with the `tests/_data_guards.py` skip pattern (BYO-run for EEC & Baltic).
2. **Backend (pure, fully testable):** `osmose/spatial_series.py` —
   `cell_timeseries(nc_path, variable, x, y, *, reduce=...) -> (times, values)`. Must
   **select-then-materialize** (`ds[var].isel(lat=y, lon=x, ...).values`), never `.values` on the
   full cube — assert this in a test. Reuse `_NC_COORD_NAMES`/`_NC_TIME_DIM_NAMES` for dim detection.
3. **Backend tests (TDD):** known cell trajectory; out-of-mask cell → NaN/empty; multi-dim
   reduction correctness; bad coords raise cleanly; lazy-selection assertion.
4. **UI panel:** cell selection (click + optional pickers) + variable + reduce selector →
   plotly line chart via `@render_plotly`; theme via `get_theme_mode`.
5. **Playwright (needs new harness sub-task):** build a fixture-injection path that starts the app
   with a spatial-output run loaded, then assert clicking a cell renders the series equal to the
   fixture's known trajectory for that `(x,y)`.

**Deliverables:** `osmose/spatial_series.py`, fixtures, panel in `spatial_results.py`, tests,
CHANGELOG entry. **Acceptance:** backend units (incl. edge cases + lazy-selection) pass and hold
the `--cov=osmose` ≥90% gate; Playwright shows the series for a clicked in-mask cell equals the
fixture trajectory. **Risk:** medium — concentrated in the new Playwright fixture-injection harness,
not the click API (supported) nor the backend (feasible).

---

## Stage 4 — Pareto-front explorer (EXTEND the existing UI)

**Round 2 rescope — most of this already exists.** The calibration page already loads a persisted
front from history into `cal_F`/`cal_X` (`calibration.py:371-372`) and renders it via `pareto_chart`
(`:486-494`), parallel-coords `correlation_chart` for ≥3 objectives (`:496-505`), and
`best_params_table` (`:507-519`). The History tab's `btn_load_run_{i}` handler already calls
`osmose.calibration.history.load_run(...)` and does `cal_X.set(np.array(data["results"]["pareto_X"]))`
/ `cal_F.set(...)` / `cal_param_names.set(...)` (`calibration_handlers.py:1646-1655`). So a standalone
loader + a new "Explorer" panel would DUPLICATE shipped infrastructure.

**Goal (narrowed):** add **per-solution selection** to the existing Results→Pareto/Best-Parameters
sub-tabs — pick one non-dominated solution, inspect its param vector + per-objective breakdown, and
**export its config** — reusing `cal_F`/`cal_X` and the existing History loader.

**The real gap (everything else is reuse):**
- The existing `best_params_table` shows the **top-10 by `F.sum`**, not a user-selected single
  solution → add a **picker** (table-row select or `input_select` over the front's solutions).
- No **config export** for a chosen solution → wire the existing config writer to the picked params.
- N-objective scatter: `make_pareto_chart` is 2D-only; parallel-coords already exists
  (`make_correlation_chart`) → generalise the scatter only if >2 objectives need it.

**Decisions:**
- **Reuse `cal_F`/`cal_X` and `calibration.history.load_run` — do NOT build `osmose/calibration/pareto.py`
  as a parallel loader.** If non-dominated filtering is needed, extract/expose the existing
  `_non_dominated_indices` (`surrogate.py:12`); don't reimplement.
- **Picker = table-row / `input_select`** (zero new plumbing). A clickable plotly trace needs the
  `FigureWidget.on_click`+`register_widget` pattern absent everywhere in this repo — optional, not baseline.
- **Only NSGA-II UI runs write a real front.** Single-objective DE/CMA-ES runs (the documented Baltic
  workhorse) and surrogate runs do NOT (`optimum["pareto"]` is in-memory only; the surrogate's saved
  `pareto_X/F` is the LHS sample pool, not a front). The picker must show an empty/disabled state when
  the loaded run has no multi-objective front — reuse the existing `F is None` guard (`calibration.py:489`).

**Phases**
1. **Selection backend (small, testable):** a pure helper that, given `F` (+ optional non-dominated
   mask via the reused `_non_dominated_indices`), maps a picker index → that solution's `X` row,
   `F` row, and param-key mapping. No new file required if it fits `calibration_charts.py`/helpers.
2. **Tests (TDD):** index→solution mapping on a **factory-built** history record (routed through
   `save_run`, not a hand-authored literal — Round 2); degenerate cases (single objective → empty
   state, 1 solution, all-dominated).
3. **UI:** add a solution picker to the existing Results sub-tab; render the selected solution's param
   table + objective breakdown from `cal_X`/`cal_F`; add an "export config" button using the config writer.
4. **Charts:** generalise `make_pareto_chart` to N objectives only if needed (parallel-coords already exists).
5. **Playwright (reuses Stage 3's harness):** load a factory-built NSGA-II history fixture via the
   existing History-load path, pick a solution, assert the param-table row count == n_params and the
   shown objective values equal the chosen solution's `F` row; assert empty state for a single-objective run.

**Deliverables:** selection helper + tests, picker + export-config UI on the existing calibration
Results tab, optional N-objective scatter, CHANGELOG entry. **Acceptance:** selection units pass and
hold the `--cov=osmose` gate (UI uncovered); Playwright selection populates the inspector with values
equal to the fixture solution and shows the empty state for single-objective runs. **Risk:** medium —
lower than v2 now that the loader/charts are reuse; risk concentrates in the picker wiring + config
export. **Effort revised DOWN to ~1.5 days** given the extensive reuse.

---

## Cross-cutting conventions (all stages)

- **TDD / subagent-driven build**; pure backend functions first, fully unit-tested before any
  Shiny wiring.
- **In-loop plan review** before building each stage; **whole-feature review** after (history shows
  the final review repeatedly catches seams per-task reviews miss — e.g. the `osm-split-layout`
  flex squish, the double-threshold bug).
- **Playwright** validation is mandatory for Stages 3–4; both depend on a **new app
  fixture-injection harness** (no existing e2e test loads a run and interacts — the lone spatial e2e
  only checks the disabled pill). Treat the harness as shared Stage-3 work reused by Stage 4.
- **Coverage:** CI runs `pytest --cov=osmose` — **`ui/` is not measured**. The ≥90% gate binds only
  new code under `osmose/` (Stage 3's `spatial_series.py`; Stage 4's selection helper if placed there);
  UI panels are validated by Playwright, not coverage. For genuinely untestable NetCDF/JSON I/O error
  branches, use a commented `# pragma: no cover` or `[tool.coverage.run] omit` (precedent:
  `pyproject.toml:68-78`) — not contrived tests.
- **Fixtures = programmatic factories**, NOT hand-authored static files. Build the synthetic NetCDF via
  an `xr.Dataset(...).to_netcdf()` factory (precedent: `_make_ltl_nc()` in `test_overlay_display.py`)
  and the calibration-history fixture via `save_run`/`write_checkpoint` (precedent:
  `_valid_checkpoint_kwargs()` in `test_calibration_checkpoint.py`) so a schema change breaks the
  factory at its source, not a stale literal.
- **CHANGELOG:** every stage lands a `CHANGELOG.md` entry (established per-feature gate).
- **CI gates to satisfy:** `ruff check` AND `ruff format --check` on `osmose/ ui/ tests/`; `pyright`;
  the `docker` build smoke test (Stage 1's matrix can affect it — verify). Stage 2 adds no deps/build.
- Ship each stage as a **fast-forward PR**, branch deleted after merge; keep changes
  **additive/parity-safe** — none touch `osmose/engine/`; Stage 3 only *reads* finished output.
- Make acceptance criteria **objective equality checks** against fixture-known values (avoid
  "spot-check" / "renders" phrasing in the per-stage specs).

## Suggested sequencing summary (effort re-estimated after Round 2 + decisions)

1. **Stage 1 (CI matrix)** — ~half-day; housekeeping (drop multi-version lint → smaller than v1).
2. **Stage 2 (narrative usage docs)** — ~1 day; pure Markdown, no deps/build, verify snippets run.
3. **Stage 3 (per-cell viewer, full)** — ~2.5–3 days incl. the synthetic NetCDF factory **and** the
   reusable Playwright fixture-injection harness (also serves the later scenario-diff view).
4. **Stage 4 (Pareto picker)** — **~1.5 days**: extends the existing `cal_F`/`cal_X` history-load UI;
   the real new work is the per-solution picker + config export.

Total ~5–6 days across four fast-forward PRs.

## Strategic decisions — SETTLED (2026-06-13)

1. **Sphinx → REPLACED with narrative usage docs** (Stage 2 rewritten above). Autodoc of internal
   APIs judged weakest value for the scientist audience; a task-oriented usage guide serves real users
   without the doc-build/dep/`cma`/`-W` tax.
2. **Per-cell viewer → FULL time-series panel** (Stage 3 as specified; not the tooltip-only shortcut).
3. **Scenario-diff view → KEPT in original order** (next-up after this plan, its own effort). Stage 3
   is designed so its `spatial_series.py` loader + Playwright fixture-injection harness are reusable by
   the scenario-diff spatial maps — avoiding a double build when it lands.

Separately flagged (outside plan scope): compact **`MEMORY.md`** (38.7 KB vs 24.4 KB limit) — move
per-feature detail into the referenced `project_*.md` topic files.
