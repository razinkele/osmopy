---
name: project_property_based_tests
description: "Property-based tests via Hypothesis (config round-trip, preamble detection, diet aggregation, size-spectrum helpers) SHIPPED to origin/master 2026-06-08. The 3rd/final queued item."
metadata: 
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

**Property-based tests via Hypothesis** — 13 property tests over 4 pure-Python targets. **SHIPPED to origin/master 2026-06-08** (`5197f15..e7268bb`, fast-forward, branch `feature/property-based-tests` deleted, pushed). The LAST of the user's queued trio (config presets = already-exists; trophic-network = shipped; this = shipped). Purely additive (no production code changed). **Next: pick a fresh backlog item** — see [[project_feature_improvements_backlog]].

## What shipped
- `hypothesis>=6.100` in the `dev` extra; a deterministic **`ci` Hypothesis profile** in `tests/conftest.py` (`max_examples=150, deadline=None, derandomize=True, database=None`), **`find_spec`-guarded NOT `pytest.importorskip`** (importorskip at conftest module scope raises Skipped → fails the WHOLE suite's collection — the round-2 BLOCKER); `.hypothesis/` gitignored.
- `tests/strategies.py` (9 strategies): `config_keys`/`config_values`/`config_kv_dicts` (st.dictionaries), `csv_texts`/`csv_text_pairs`, `diet_matrices`, `edges_and_values`, `shuffled_bin_edges`, `time_value_frames`.
- 4 property files (13 tests): `test_config_roundtrip_properties` (round-trip survival + substantive-keyset equality + separator-invariance), `test_results_preamble_properties` (detect-k, never-raise, cache-invalidation), `test_trophic_network_properties` (non-neg+clean-names, threshold-monotonicity, prey-sum-exactness, dead-stage; `@settings(max_examples=50)`), `test_size_spectrum_properties` (mean-size convexity, LFI edge==threshold boundary, bin-width order-invariance, two-sided window).
- Verified: 3146 passed (+13), deterministic across 2 runs, ruff-clean (365 files). The 1 suite failure is PRE-EXISTING/environmental (`test_tutorial_3species` subprocess `import osmose` — osmose not pip-installed in `.venv`; fails identically on master).

## Hard-won facts (verified by executing reviewers)
- **ruff exempts imports after `pytest.importorskip(...)` from E402** (precedent: `tests/test_ui_state.py`). So `import pytest; pytest.importorskip("hypothesis"); from hypothesis import ...` is clean. But importorskip at CONFTEST scope is the suite-killing BLOCKER → use `find_spec` there, `importorskip` per-file.
- **`tmp_path` fixture CANNOT be used under `@given`** → `HealthCheck.function_scoped_fixture` hard error. Use `with tempfile.TemporaryDirectory() as td:` INSIDE the test body.
- **`_detect_preamble_lines` caches on `(mtime_ns, st_size)`** and counts fields with **`csv.reader` (comma)**. A same-path, same-byte-size in-place rewrite within one mtime_ns tick (coarse tmpfs clock) does NOT invalidate the cache → the cache-invalidation property MUST force a byte-size difference (`csv_text_pairs` appends a trailing line if sizes match). This is a real (low-impact, out-of-scope) latent cache limitation.
- **config round-trip:** reader injects `_osmose.config.dir`, writer regenerates `osmose.configuration.*` reference keys → the property compares only substantive keys (strip both). Strategy must exclude `osmose.configuration.*` keys (writer clobbers the 9 known suffixes) and value first+last char must be non-separator (writer frames `key ; value`; reader splits maxsplit=1 + rstrips `;,:\t =`). Internal separators round-trip fine.
- **size-spectrum:** `_large_fish_indicator` returns **`0.0` not NaN** on total≤0; compares lower `edge >= threshold`. `_mean_size` returns NaN on total≤0. `_infer_bin_width` does `sorted(set(edges))`. Strategy floats need `min_value`/`max_value` finite bounds + a `1e-3` floor (denormal `5e-324` underflows the mean numerator → false-fail).

## Methodology note — 3 spec rounds + 1 plan round, escalating rigor
- Round 1 (correctness): prototyped invariants hold. Round 2 (teeth-by-reasoning + ops): caught the importorskip BLOCKER, debounce-style ops, cut tautological props. Round 3 (**empirical mutation + post-edit consistency**): injected each property's claimed bug and confirmed RED — **caught that round-2's "threshold-monotonicity catches filter-before-aggregation" reasoning was WRONG** (0/150 RED; an earlier filter only removes edges → subset preserved; the reorder is caught by PREY-SUM-EXACTNESS). Also instrumented strategies for vacuous-pass (forced explicit dead-stage/NaN/multi-stage-prey biasing). Plan round (**executed the plan's code**): both reviewers hit the cache same-size BLOCKER by running it. Final whole-feature review: caught the one-sided window property → made two-sided. **Lesson reinforced: mutation-test the teeth + run the actual code; reasoning about "does this test catch X" is unreliable.** See [[feedback_in_loop_review_pattern]].

## Gotcha
- `_community_long`/`compute_size_spectrum` end-to-end NOT fuzzed (invariants live in the helpers, which ARE fuzzed; end-to-end stays on the existing real-EEC example test). Scenarios `fork`-independence evaluated + REJECTED (flat `dict[str,str]` config → shallow copy already independent → tautological).
