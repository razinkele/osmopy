---
name: project_result_delta_tracking
description: Result delta-tracking (per-species output delta between two runs) SHIPPED to origin/master 2026-06-03; hardened across TWO deep in-loop plan-review rounds.
metadata: 
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

Per-species **output delta-tracking** between a baseline and a variant run MERGED + PUSHED to origin/master 2026-06-03 (`6d8b2f5..dfcb422`, branch deleted). Built via brainstorming→writing-plans→**two deep in-loop plan-review rounds**→subagent-driven (DT1–DT6). Spec `docs/superpowers/specs/2026-06-03-result-delta-tracking-design.md`, plan `…/plans/2026-06-03-result-delta-tracking-plan.md`, doc `docs/baltic_example.md`. 15 tests; purely additive (+1369/−0, no engine changes).

Deliverables: `osmose/analysis.py` (`_trailing_window`, `_per_species_window_mean`, `SpeciesDelta`, `run_delta`, `format_delta_report`), `osmose/plotting.py::make_run_delta_chart` (diverging bar), `scripts/compare_runs.py` CLI. Real-data validated: cod-halved variant → cod surfaces at exactly −50% top mover. Computes per-species abs + % change in biomass/yield/abundance over a trailing window, ranked by |Δ%|.

## Why two review rounds (the lesson — same "execute don't infer" theme as fisheries)
The spec/plan initially trusted reconnaissance that **claimed the accessors return long-form `time,species,value`**. I caught by execution that `biomass()/yield_biomass()/abundance()` actually return a **WIDE** frame (`Time` + per-species columns + a constant `species="all"`; `biomass(species="cod")` → 0 rows, no `value` col — and PR #46's `model_biomass_window_mean` is latently wrong on this format). Then two deep review rounds (executing reviewers) found **silent-wrong-result bugs the synthetic tests + the recordfreq==ndtPerYear configs masked**:
- ROUND 1: (BLOCKER) `df.tail(window_years)` assumed 1 row/year → wrong on sub-annual output (same class as the fisheries `steps_per_year` bug) → fixed to **window by the Time column (years)**. (BLOCKER) `pct_delta None→inf` ranked a **0→0 dead species above real movers** → `inf` only for genuine `from_zero`, `0.0` for 0→0. (MAJOR) `species.nunique()>1` long-detector → phantom `"value"` species → detect by `value`+`species` columns. + test tautology (`top_n` alpha==pct order), CLI asserting only rc==0.
- ROUND 2: round-1 fixes all verified clean; fresh adversarial found 2 NEW (disjoint) issues — (BLOCKER) the **fixture test crashed** (`cp` kept the CSV title preamble → bare `pd.read_csv` read it as header → KeyError) → generate fixture via `OsmoseResults().biomass().to_csv`; (MAJOR) **`window_years<=0` → empty window → NaN → corrupt sort + invalid JSON** → guard `window_years>=1` + CLI rejects non-positive. Convergence: BLOCKER class shifted logic→harness/input-validation across rounds.

## Gotchas (carry forward)
- The 1D global outputs (biomass/yield/abundance) are **WIDE** (Time + per-species cols + species="all"), NOT long-form; per-species values are COLUMNS. The `species=` filter is useless for them (matches the constant "all"). Window by the Time column, never by row count (robust to sub-annual cadence).
- `scripts/compare_runs.py` needs `PYTHONPATH=.` to run directly (the recurring osmose-not-editable-installed gotcha).
- Deferred follow-ons: per-period (which year moved) deltas, per-cell spatial deltas (needs output.spatial.enabled). **UI "Compare Runs" tab follow-on SHIPPED 2026-06-03** (`dfcb422..110b011`): `ui/pages/results.py` now shows the per-species output delta (ranked table + diverging bar) when exactly 2 runs are selected — `_delta_for_selected` helper + `compare_window_years` slider + `run_delta_chart`/`run_delta_table` render fns wrapping the shipped `run_delta`/`make_run_delta_chart`/`format_delta_report`. Spec/plan `…2026-06-03-compare-runs-delta-ui-*`. In-loop reviewed (2 executing reviewers, no blocker — selectize click-order preserves baseline/variant; live app launches HTTP 200). UI render fns aren't unit-tested by convention → logic in the testable `_delta_for_selected`; verified by page-builds smoke + live launch. Gotcha confirmed: the run selector populates only when run history exists at `<output_dir>/../.osmose_history` (a pre-existing run.py-vs-results.py history-dir mismatch, left as a separate follow-on).

See [[project_fisheries_fm_diagnostics]], [[project_predator_functional_response]] (same session). The "execute the readers, don't infer from docstrings/headers" lesson recurred in BOTH fisheries and this feature.
