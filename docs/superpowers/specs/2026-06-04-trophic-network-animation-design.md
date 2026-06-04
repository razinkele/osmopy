# Trophic-network animation (pyvis) — Design

**Date:** 2026-06-04
**Status:** Approved direction (brainstormed; codebase- + literature-grounded). New feature
(UI + analysis). **Representation = pyvis node-link graph** (revised after an in-loop review
killed the plotly-Sankey approach; see "Why pyvis").

## Motivation

OSMOSE emits a per-timestep diet matrix (who-eats-whom), but the UI shows only a **time-averaged**
diet heatmap (and that view is latently broken — see below). This feature surfaces the trophic
network and adds the time dimension: step (and optionally auto-play) through the run's diet matrix
as an interactive **node-link graph** where nodes are species, directed edges are predator→prey
diet links, and self-loops are cannibalism.

## Why pyvis (and why NOT a Sankey) — the load-bearing decision

An in-loop review (incl. the **scientific-validation** skill / scite) established:
- **Food webs are inherently cyclic.** Intraguild predation is "ubiquitous" and "uni- or
  bidirectional," "often associated with cannibalism" (Hatcher et al., 2008; Sieber & Hilker,
  2010). The real EEC diet network at one timestep has **145 mutual 2-cycles** + **13 self-loops**.
- **A plotly Sankey (DAG-only) cannot represent cycles** — it returns a `go.Figure` (so a naive
  test passes) but the browser layout engine breaks on cycles. The orphaned `make_food_web` Sankey
  is orphaned *partly because* a Sankey is the wrong tool for a cyclic food web. **Abandoned.**
- **The standard food-web representation is a node-link network graph.** **pyvis** (vis.js) renders
  arbitrary directed graphs including cycles and self-loops — confirmed by execution
  (cod↔herring + cod→cod render fine).

## Verified context (audit — confirmed by execution)

- Data: `output/Trophic/{prefix}_dietMatrix_Simu0.csv` — 1-line title preamble, header
  `"Time","Prey",<predator×size-class cols>`; rows keyed by `(Time, Prey)`; value = **% of that
  prey-stage in that predator-stage's diet** (each predator-stage **column sums to ~100** — values
  are **PERCENT, 0–100**, not fractions). EEC: 70 timesteps. Predators (cols) are all size-split;
  **prey (rows) are size-split for the 12 fish but single-row for the 10 plankton/benthos resource
  species** (no ` in [` suffix). NaN cells are real (416 at t=1). Self-loops (cannibalism) are real
  (33 at t=1).
- **`OsmoseResults.diet_matrix()` does NOT read this file** — `_find_output_files` searches root +
  `_ENGINE_SUBDIRS = ("Mortality","Bioen")`, NOT `Trophic/`, AND defaults `prefix="osm"` (files are
  `eec_`/`baltic_`). It **raises `FileNotFoundError`** on committed outputs (so the existing Results
  `diet_chart` is latently broken — a separate pre-existing issue, NOT fixed here). The feature
  reads the file **directly** via `rglob("{prefix}_dietMatrix*.csv")` + `_read_output_csv`
  (preamble-safe). **No `OsmoseResults`/engine change.**
- **pyvis 4.2** (confirmed installed + working):
  - Install: `pyvis @ git+https://github.com/razinkele/pyvis.git@v4.2` (pip-installable wheel;
    pulls `networkx`, `jsonpickle`) — mirrors the existing `shiny_deckgl @ git+…` dep. (PyPI pyvis
    is 0.3.x; 4.2 is the razinkele build — git+ is the pip path; the `razinka` conda channel also
    has it but the project is pip/`.venv`.)
  - API: `Network(height, width, directed=False, …, cdn_resources='local'|'in_line'|'remote')`;
    `add_node(id, label=…, …)`, `add_edge(src, dst, value=…)`, `generate_html() -> str`. Renders
    mutual cycles + self-loops (verified).
  - **Self-contained HTML for Shiny:** `Network(directed=True, cdn_resources='in_line')` →
    `generate_html()` inlines vis.js (no `lib/` refs, no CDN) → ~658 KB self-contained HTML →
    embed via `ui.tags.iframe(srcdoc=html, …)`. (Default `cdn_resources='local'` writes relative
    `lib/vis-10.0.2/...` refs that need served assets — we use `in_line` to avoid static-serving.)
- Results-page slider+render pattern exists (`compare_window_years` slider; sliders populated in
  `_do_load_results`; `results_obj.get().prefix` carries the run prefix). `_read_output_csv` is a
  private cross-module helper (acknowledged coupling).
- CI lints `osmose/ ui/ tests/` (NOT `scripts/`).

## Dependency

Add to `pyproject.toml` `dependencies`: `"pyvis @ git+https://github.com/razinkele/pyvis.git@v4.2"`
(alongside `shiny_deckgl`). The render fn **lazy-imports** pyvis so a missing install degrades to a
message (`"Install pyvis to view the trophic network"`) rather than crashing the Results page; the
analysis module stays pyvis-free at import time.

## Architecture

### 1. `osmose/trophic_network.py` (analysis layer — pyvis-free, fully testable)

- `_read_diet_matrix(output_dir) -> pd.DataFrame` — `rglob("*_dietMatrix*.csv")` (WILDCARD prefix —
  any subdir incl. `Trophic/`; there is exactly one dietMatrix file per run dir) + `_read_output_csv`;
  raise `FileNotFoundError` if absent. Returns wide `Time, Prey, <predator-stage cols>`. **Do NOT
  depend on `res.prefix`** — review found `_do_load_results` constructs `OsmoseResults(out_dir,
  strict=False)` with the default `prefix="osm"`, so `res.prefix` is `"osm"` on every real
  (`eec_`/`baltic_`) run; a prefixed glob would find nothing. The wildcard glob is the fix.
- `_split_species(label) -> str` — strip ` in [lo, hi[` suffix; pass through labels without it
  (the 10 resource species). Handles both prey-row and predator-col labels.
- `available_times(output_dir) -> list[float]` — sorted unique `Time` (wildcard reader; no prefix).
- `species_layout(species: list[str]) -> dict[str, tuple[float, float]]` — compute **fixed** (x, y)
  positions ONCE for the **union of all species across all timesteps** (e.g.
  `networkx.spring_layout(union_graph, seed=42)` scaled, or a deterministic circular layout —
  networkx is already a pyvis dep). Deterministic (fixed seed) so the layout is identical every
  render. Used so the graph does NOT re-jiggle per timestep.
- `diet_network_at(output_dir, *, time, threshold=5.0, predator_level="species")
  -> pd.DataFrame` — filter `Time == time` (raise `ValueError` if absent); drop NaN; melt predator
  cols; `_split_species` both axes; **prey-stage rows SUM to prey-species** (exact additive
  within a predator); **predator level:** `"species"` → MEAN the species' stage columns
  **excluding 0-sum dead stages** (documented unweighted approximation); `"stage"` → keep predator
  at stage granularity (no predator aggregation, exact). Return long `predator, prey, proportion`
  (proportion in **percent**), keep links `>= threshold`. **Default threshold `5.0` (5%)** — review
  measured 97 edges / 23 nodes at 1% (a hairball) vs 63 at 5% (readable); the slider lets the user
  lower it. Values stay percent.

### 2. `make_trophic_network_html(diet_df, *, positions, threshold=5.0, height="600px") -> str` (in `trophic_network.py`)

Lazy `from pyvis.network import Network`. Build `Network(directed=True, cdn_resources='in_line',
height=height, width='100%')`; **disable physics** (`net.set_options('{"physics": {"enabled":
false}}')` — or `net.toggle_physics(False)`); add a node per species with its **fixed
`x`/`y` from `positions`** (the `species_layout` dict, computed once over the all-timestep union)
and `physics=False` so **the layout is identical every render** (no per-tick re-jiggle — the
review's BLOCKER B1); add a directed edge predator→prey with `value=proportion` (width) + a hover
title (`"X% of P's diet"`) for links `>= threshold`; self-loops (predator==prey) render as
cannibalism (vis.js draws a loop arc — confirmed). Return `generate_html()` (self-contained HTML).
Testable: assert species labels appear, HTML is self-contained (no `lib/` src), the emitted JSON
carries `"x"`/`"y"` + `"physics": {"enabled": false}` (proves fixed layout), cycles/self-loops
survive. Skipped if pyvis absent.

### 3. UI — Results page (`ui/pages/results.py`)

Add a **"Trophic Network"** sub-tab beside the diet view:
- `ui.input_slider("trophic_time", …)` + `ui.input_radio_buttons("trophic_predator_level", …,
  {"species","stage"})` + `ui.input_slider("trophic_threshold", …, value=5.0)` +
  `ui.output_ui("trophic_network")`.
- **Slider population (review finding):** `_do_load_results` currently populates only a *select*
  (`ui.update_select`) — no slider is dynamically populated there. Add an explicit
  `ui.update_slider("trophic_time", min=…, max=…, value=…)` in `_do_load_results` from
  `available_times(out_dir)` (guarded so a run without diet output just leaves the slider default).
  This is the first dynamically-populated slider in the module.
- `@render.ui def trophic_network()`: lazy-import pyvis (missing → `ui.div("Install pyvis …")`);
  read the loaded output dir (NOT `res.prefix` — the reader globs a wildcard), the slider time,
  level, threshold → `make_trophic_network_html(diet_network_at(dir, time=…, threshold=…,
  predator_level=…), positions=<cached layout>)` → `ui.tags.iframe(srcdoc=html, style="width:100%;
  height:620px; border:0;")`. Wrap in try/except → `FileNotFoundError`/empty → `ui.div("No
  diet-matrix output found")` (degrade, never crash — the run-history-fix pattern).
- **Caching + debounce:** a `@reactive.calc` keyed on the output dir holds (a) the wide df and
  (b) `species_layout(union species)` — both computed once per run, so dragging the slider slices an
  in-memory df + reuses the fixed layout (no CSV re-read, no re-layout). **Debounce/throttle the
  slider** via `shiny.reactive.debounce`/`reactive.throttle` (NOT `@reactive.event`, which fires on
  every value change, not only on release) so the ~710 KB self-contained iframe `srcdoc` transfers
  at a rate-limit, not on every continuous drag step
  (review: in_line is ~710 KB after srcdoc-escaping; CPU per render ~95 ms is fine — wire size is
  the cost).
- **Optional auto-play** (`reactive.invalidate_later` advancing `trophic_time`): **cut for v1** —
  ~710 KB/frame is too heavy for auto-advance; revisit only after the `local`+`www` optimization.
- **Documented optimization (out of scope for v1):** `cdn_resources='local'` writing vis.js once to
  the already-mounted `www/` static dir (`app.py:543 static_assets=_WWW`) + `iframe src=` drops the
  per-tick payload from ~710 KB to ~5.6 KB. Kept as a follow-on; v1 uses `in_line` + debounce for
  simplicity (no per-tick static-file churn).

## Data flow

Load `OsmoseResults` → `available_times` sets slider bounds (cached wide df) → on slider/level/
threshold change → `diet_network_at(...)` (slice cached df, aggregate one timestep) →
`make_trophic_network_html(...)` (pyvis, in_line) → `ui.tags.iframe(srcdoc=…)`.

## Error handling

- No `dietMatrix` file → `FileNotFoundError` in `_read_diet_matrix`; render fn catches → "No
  diet-matrix output found".
- pyvis not installed → render fn lazy-import fails → "Install pyvis to view the trophic network".
- `time` not in `available_times` → `ValueError` (slider only offers valid times; guards direct
  callers/tests).
- All links below threshold → an empty/near-empty graph (acceptable; threshold is user-tunable).
- NaN cells dropped before aggregation; 0-sum dead predator stages excluded from the species mean.

## Testing (`tests/test_trophic_network.py`)

- `_split_species`: `"cod in [10.000000, 30.000000["` → `"cod"`; `"Diatoms"` → `"Diatoms"`.
- `diet_network_at` on a **synthetic** wide dietMatrix (Time, fish prey-stage rows + a resource row,
  predator-stage cols incl. **a species with a 0-sum dead stage** — model the herring case, NOT cod
  which has no dead stage, so the exclusion path is actually exercised — a NaN, a self-loop) with
  known values: prey stages SUM to prey-species (exact); `species` level MEANs predator stages
  **excluding the dead stage** (assert the ×N/(N−dead) scaling, e.g. the herring before/after);
  `stage` level keeps stages; threshold filters; NaN dropped; self-loop preserved; bad `time` raises
  `ValueError`. Assert proportions are in percent (0–100).
- `available_times` sorted-unique (wildcard reader, no prefix arg).
- `species_layout` is deterministic (same input → identical coords) and covers the union species.
- Real-EEC smoke: `diet_network_at("data/eec_full/output", time=1.0)` (NO prefix — wildcard read) →
  non-empty long df, 3 columns, species names (no ` in [` at `species` level), includes a resource
  prey (e.g. `Mesozoo`) and a self-loop predator.
- `make_trophic_network_html` (skip if pyvis absent): on a small df with a **self-loop** + a mutual
  cycle (both present only at low threshold — pass `threshold=0.0` for this test) + a `positions`
  dict → returns a self-contained HTML string (`'src="lib/'` NOT present) containing the species
  labels AND `"physics": {"enabled": false}` + `"x"`/`"y"` (proves the fixed-layout BLOCKER fix).
- (Render fn / slider / iframe / play not unit-tested — convention; manual UI run-through covers
  wiring: launch app, load a run with diet output, open Trophic Network, drag the slider.)

## Scope / YAGNI

- **In:** the pyvis dep; `trophic_network.py` (direct Trophic reader + per-timestep species
  aggregation + `available_times` + the pyvis HTML builder); the Results "Trophic Network" sub-tab
  (slider + level + threshold + iframe, cached, optional play); tests; docs.
- **Out:** the plotly Sankey / `make_food_web` (cyclic-graph trap — abandoned); the heatmap option;
  fixing `diet_matrix()`/`_ENGINE_SUBDIRS`/the broken existing diet_chart (read directly,
  pre-existing, out of scope); consumption/biomass-weighted predator roll-up (data not in the
  matrix — documented unweighted approximation, or use `stage` level); per-cell spatial diet;
  changing `make_diet_heatmap`.

## Honest limitations (incl. scientific-validation)

- The graph shows **diet composition (proportions)**, NOT consumption-weighted trophic *flow*
  (flow = composition × consumption Q/B × biomass — Ecopath-style; Mensah et al., 2019). Predator
  species-aggregation is an **unweighted** stage mean (documented; `stage` level avoids it). The UI
  labels it "diet composition, predator stages averaged unweighted — not consumption-weighted
  flow." **(Validated against the trophic-ecology literature; the node-link representation is the
  ecologically-correct choice for a cyclic web.)**
- Self-contained pyvis HTML is ~673 KB raw / ~710 KB as an escaped iframe `srcdoc`; slider-stepping
  is debounced (transfer on commit); auto-play is cut for v1. The `cdn_resources='local'`+`www/`
  served path (per-tick ~5.6 KB) is the documented follow-on optimization.
- Node layout is **fixed** (computed once over the all-timestep species union) so the graph is
  stable across timesteps; the trade-off is the layout isn't re-optimised per timestep (correct
  for "watch it shift over time" — edges change, nodes hold still).
- **Cycle visibility:** the "145 mutual 2-cycles + self-loops" are the *raw* (threshold≈0) counts;
  at the default 5% threshold mutual 2-cycles largely vanish and a handful of self-loops remain.
  The pyvis-over-Sankey justification still holds (pyvis *can* render cycles/self-loops, which a
  Sankey can't, and they appear as the user lowers the threshold) — but the default view is a
  readable thresholded network, not the full cyclic tangle.
- Reads the Trophic file directly (existing `diet_matrix()` accessor can't — unfixed, out of scope).

## Delivery

Single PR: `pyproject.toml` (pyvis dep), `osmose/trophic_network.py`, `ui/pages/results.py`
(sub-tab + render fn + slider + cache), `tests/test_trophic_network.py`, a docs/CHANGELOG note.
No engine / `OsmoseResults` change.

## References (scientific-validation)

- Hatcher, M. J., Dick, J. T. A., & Dunn, A. M. (2008). A keystone effect for parasites in
  intraguild predation? *Biology Letters, 4*(5), 534–537. https://doi.org/10.1098/rsbl.2008.0178
- Sieber, M., & Hilker, F. M. (2010). Prey, predators, parasites: intraguild predation or simpler
  community modules in disguise? *Journal of Animal Ecology, 80*(2), 414–421.
  https://doi.org/10.1111/j.1365-2656.2010.01788.x
- Mensah, E. T.-D., Dankwa, H. R., & Lauridsen, T. L. (2019). Mass balance model of Lake Volta
  fisheries: The use of Ecopath model. *Lakes & Reservoirs, 24*(3), 246–254.
  https://doi.org/10.1111/lre.12276
