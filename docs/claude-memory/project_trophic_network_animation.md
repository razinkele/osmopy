---
name: project_trophic_network_animation
description: "Trophic-network animation (pyvis node-link diet-matrix graph in Results UI) SHIPPED to origin/master 2026-06-04. Per-timestep predator→prey graph, fixed layout, time slider."
metadata: 
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

**Trophic-network animation** — a Results → "Trophic Network" sub-tab that renders the per-timestep diet matrix as an interactive **pyvis node-link graph** (predator→prey, cannibalism self-loops), with a fixed layout + a time slider to step through time. **SHIPPED to origin/master 2026-06-04** (`d335964..5197f15`, fast-forward, branch `feature/trophic-network-animation` deleted, pushed). The 1st of the user's queued trio: config presets (found ALREADY-EXISTS, not built) → **trophic-network** (this) → property-based tests via Hypothesis (NEXT).

## What shipped
- **`osmose/trophic_network.py`** (new, 162 lines): `_read_diet_matrix` (WILDCARD `rglob("*_dietMatrix*.csv")` — `OsmoseResults.diet_matrix()` can't find it; reads first replicate Simu0 via `osmose.results._read_output_csv`), `_split_species` (strip ` in [lo,hi[` suffix), `available_times`, `network_node_universe`, `diet_network_at` (one-timestep aggregation), `species_layout` (fixed `nx.spring_layout(seed=42)`×600), `make_trophic_network_html` (pyvis `Network(directed=True, cdn_resources="in_line")`, physics off, fixed x/y, `generate_html()`).
- **`ui/pages/results.py`** (+87): "Trophic Network" nav_panel after "Diet Composition" (index slider + predator-level radio + threshold slider + `output_ui`); `_trophic_cache` `@reactive.calc` (keyed on LOADED dir = `results_obj`+`state.output_dir`, holds `{dir,times,layouts}`); `trophic_network` `@render.ui` returns `ui.tags.iframe(srcdoc=html, sandbox="allow-scripts")` + `Time N` caption; slider populated in `_do_load_results`.
- pyvis git+ dep in pyproject.toml; `tests/test_trophic_network.py` (14 tests). pyvis 4.2 + networkx in `.venv`.

## Key design decisions (semantics)
- Shows **diet COMPOSITION** (% of predator's diet, sums ~100/predator-stage), NOT consumption-weighted flow. Prey size-stages SUM to prey-species (exact); predator size-stages averaged UNWEIGHTED over LIVE stages (a 0-sum dead stage excluded = sum÷n_live); `predator_level="stage"` keeps stages (exact). NaN cells dropped.
- **Node-link (pyvis), NOT Sankey** — the in-loop review + scientific-validation (scite) killed the original Sankey design: food webs are inherently CYCLIC (intraguild predation ubiquitous/bidirectional + cannibalism); real EEC diet network has 145 mutual cycles + 13 self-loops at one timestep → a DAG/Sankey can't render it. (Hatcher et al. 2008; Sieber & Hilker 2010; Mensah et al. 2019.)
- **Fixed layout once** over the all-timestep node universe + `physics=False` so nodes hold position as the slider moves (Playwright-verified byte-identical coords across timesteps).
- **Index slider** (0..n-1 over discrete `available_times`), render maps index→Time — so fractional/sub-annual Time is addressable. **NO debounce** (Shiny 1.5.1 has no `reactive.debounce`/`throttle`; `input_slider` has no client rate policy — both verified live). Per-render ~660KB `in_line` iframe is an accepted self-containment trade-off; `cdn_resources="remote"` is the documented follow-on.

## In-loop reviews caught the substance (spec 2 rounds + plan 2 rounds + final whole-feature)
- **Spec:** Sankey-can't-do-cycles BLOCKER (scientific-validation) → pivot to pyvis; prefix BLOCKER (`res.prefix`="osm" → prefixed glob finds nothing on eec_/baltic_) → wildcard; per-tick re-layout BLOCKER → fixed positions.
- **Plan (2 executing reviewers vs live env):** `reactive.debounce` DOESN'T EXIST in Shiny 1.5.1 (BLOCKER — would crash the page at import) → removed; integer slider can't hit fractional Time (MAJOR) → index slider+index→Time map; malformed noqa + NaN-in-live-stage docstring (minor). All folded before build.
- **Final whole-feature review (100% conf seam bug the per-task reviews missed):** the render filtered via `diet_network_at(threshold=user)` then passed the filtered df to `make_trophic_network_html` WITHOUT threshold → its default 5.0 RE-filtered → any slider **below 5% silently clamped to 5%**. Playwright tested 5→20 so never hit it. Fix `5197f15`: pass `threshold=0.0` at the UI call site. **Lesson: per-task reviews pass each side of a seam; the cross-module double-filter only shows when reviewing the whole diff.**

## Validation
- 14 trophic unit tests + page-build smoke; 3133 passed full suite (1 PRE-EXISTING unrelated failure: `test_tutorial_3species::test_markdown_code_block_parses_and_runs` — `osmose` not pip-installed in `.venv` so a subprocess `import osmose` from a tmp cwd fails; independent of any branch).
- **Playwright live run-through (EEC):** graph renders 20 nodes / 63 edges / 3 self-loops at Time 1; `Time N` caption updates; node positions byte-identical Time 1↔36 (fixed layout works); threshold 5→20 thins edges 63→28; no new console errors.

## Gotchas
- `_read_output_csv` private-import from `osmose.results` is an established intra-package convention (`osmose/size_spectrum.py` does it too).
- The render fn's `except ImportError` degrade ("Install pyvis…") is effectively dead — pyvis is a HARD dep and the import of the two functions never fails (pyvis import is lazy inside `make_trophic_network_html`). Harmless belt-and-suspenders; left as-is.
- The diet matrix the tab reads is NOT in `results_obj` (that's why the wildcard reader exists); `_trophic_cache` reads the CSV 3× per load (1 times + 2 node-universe) — bounded + memoized by `@reactive.calc`.

**NEXT: property-based tests via Hypothesis** (the 3rd queued item). See [[project_feature_improvements_backlog]]. Subagent-driven build (5 tasks, two-stage review each) worked cleanly again — see [[feedback_in_loop_review_pattern]].
