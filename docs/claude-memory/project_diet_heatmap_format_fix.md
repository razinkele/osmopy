---
name: project-diet-heatmap-format-fix
description: "Results Diet heatmap showed (no prey data) for every real run — engine writes <predator>_<prey> wide cols, code only knew legacy prey_<name>; fixed 2026-06-16 (9d1359f)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**Diet-heatmap format fix — SHIPPED 2026-06-16** (master `9d1359f`, pushed direct; CI green).

The Results "Diet Composition" heatmap showed "(no prey data)" for EVERY real run: `make_diet_heatmap` (`ui/pages/results.py`) only recognized legacy `prey_<name>` columns, but the engine writes the diet matrix WIDE as **`<predator>_<prey>`** columns (predator-major; focal `species_names` × all-species prey; values = **biomass eaten in tonnes**; `osmose/engine/output.py` `_build_diet_dataframe`/`write_diet_csv`; reader `OsmoseResults.diet_matrix()` adds a constant `species="all"` col).

Fix: added the `<predator>_<prey>` layout — split each non-meta col on the FIRST `_` (predator names carry no underscore), mean over time, **normalize per predator row → diet proportions**, render predators×prey; legacy `prey_` path preserved. Verified vs the real Baltic run matrix (8 predators × 10 prey, row sums 1.0). The empty-output recalculation was ALSO the source of transient `diet_chart` Shiny client state-sync errors.

**▶▶ LESSON: a full e2e Baltic run driven through Playwright (live-movement + graphical outputs) surfaced this pre-existing bug that the whole unit suite never caught** — no shipped/test path had fed a real engine dietMatrix into `make_diet_heatmap`. Run the WHOLE thing end-to-end, not just units.
