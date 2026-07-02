---
name: project-diet-chart-recalc-and-trophic-format-fix
description: PR
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

PR #72 (master `888a482` trophic, `578879d` diet) — SHIPPED 2026-06-17, merged + prod-verified live (full Baltic browser smoke at `https://laguna.ku.lt/osmose/`: load→Python 1-yr run+live movement→Diet Composition heatmap (real matrix, not empty-state)→Trophic Network iframe, **0 console errors** on the prod origin).

Two fixes, both surfaced ONLY by the full Baltic e2e ([[project-visual-regression-tests]] sibling work), unit-test fixtures missed them:

1. **diet_chart "unexpected state" client errors** — `_get_result_data` (ui/pages/results.py) both *read* AND *wrote* `results_data` (a lazy memo cache), so a reader self-invalidated → double recalc → Shiny `OutputProgressReporter` desync. Fix: wrap the cache read/write in `reactive.isolate()`, depend only on live `results_obj` → recompute exactly once.
2. **trophic_network `KeyError: 'Prey'`** — diet matrix has TWO layouts. Java: `Time, Prey, <pred-stage cols>`. Python engine: `Time, <pred>_<prey>` (predator-major, species-level, biomass eaten; see [[project-diet-heatmap-format-fix]]). `osmose/trophic_network.py` assumed Java. Fix: `_is_engine_diet` (= `"Prey" not in df.columns`) + `_engine_pairs` (split on first `_`), parallel branch in `network_node_universe`/`diet_network_at`; stage-level falls back to species for engine output.

**Why:** real Baltic data exposes interaction bugs that synthetic fixtures don't — the e2e is the regression guard going forward (`tests/test_e2e_baltic.py` asserts both: console-error guard for diet_chart, no `.shiny-output-error` on Trophic tab).
**How to apply:** any new diet-matrix reader must accept BOTH formats; any reactive memo-cache that's also a reader MUST isolate its writes.

GOTCHA hit this PR: a **CONFLICTING/DIRTY PR silently gets NO CI** (only advisory checks like CodeRabbit show) — GitHub skips `pull_request` workflows on non-mergeable PRs. Cause here: #71 was *rebase*-merged → master's `test_e2e_baltic.py` got a new SHA → my sibling branch (based on the pre-rebase tip) conflicted on that file. Fix: rebase sibling branches onto current master (don't merge); check `mergeStateStatus`/`mergeable` before debugging "why no CI".
