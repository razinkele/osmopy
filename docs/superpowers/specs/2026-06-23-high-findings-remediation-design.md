# Deep-Review High-Findings Remediation — Design Spec (v2, hot-path-corrected)

**Date:** 2026-06-23
**Status:** Draft for review (v2 — reworked against the production dispatch path
after the 2026-06-23 spec-review workflow; see `docs/reviews/2026-06-23-spec-review-high-remediation.md`)
**Source:** 2026-06-22 deep review (`docs/reviews/2026-06-22-deep-review-v1.0.0.md`)

## Dispatch-path context (READ FIRST)

The engine has **two parallel mortality implementations**, and the v1 spec
targeted the wrong one:

- **Production / CI hot path** (`_HAS_NUMBA=True`): `mortality()` →
  `_mortality_all_cells_numba` / `_mortality_all_cells_parallel`
  (`mortality.py:1923` dispatch, **no bioen check**) → `_apply_predation_numba`,
  with per-school rates from `_precompute_effective_rates`.
- **Fallback path** (`_HAS_NUMBA=False` only): `mortality()` →
  `_mortality_in_cell` → either `_mortality_in_cell_numba` (still uses the same
  `_apply_predation_numba` kernel and `eff_*` rates) or the innermost pure-Python
  interleaved loop (`_get_mortality_causes` + `_apply_*_for_school`).

Every fix below states which path it targets. **The Numba hot path is primary.**

## Already fixed (the critical)

The review's critical (bioen double-starvation) is **fully fixed on both paths**:
- `457ac55` — exclude STARVATION from `_get_mortality_causes` (innermost
  pure-Python loop, fallback only).
- `c273c8f` — zero `eff_starv` when `config.bioen_enabled` in
  `_precompute_effective_rates` (all Numba paths = production). Empirically
  confirmed the double-count (`eff_starv` was non-zero for under-fed bioen
  schools while `_bioen_step` also applied bioen starvation).

This spec covers the remaining work: HIGH-1 (egg-retention), HIGH-2
(fleet-effort), HIGH-3-narrow (`_worker_eval`).

## Goals / Non-goals

**Goals:** fix the three findings below, each test-first; keep the BoB parity
baseline within tolerance.

**Non-goals (deferred):** the 14 `_shared_*` calibration nonlocal refactor; all
medium/low review findings; a Java/EEC cross-parity harness (see Open Questions).

---

## Fix 1 — Egg-retention: gate egg predation on the released fraction (HOT PATH)

### Bug (production-affecting)

`mortality()` advances a graduated egg-release schedule each sub-dt
(`work_state.egg_retained` seeded full at `:1795`, decremented by
`abundance/n_subdt` at `:1915-1920`), but **predation never reads it**. The
production kernel `_apply_predation_numba` reads `inst_abd[q_idx]` as egg prey
availability (`mortality.py:886`) with no retention term, so a predator can eat
an entire egg cohort in sub-dt 1. The pure-Python predation
(`_apply_predation_for_school`, `:397`) has the same gap.

### Model (unchanged from v1 — verified correct)

At sub-dt *k* the **eatable** egg count is the released-and-alive portion:
`max(inst_abd[q] − egg_retained[q], 0)`. For non-eggs `egg_retained == 0`, so
this is identically `inst_abd[q]`. The release loop runs *before* predation each
sub-dt, so `egg_retained` is current at read time. Deaths still decrement
`inst_abd` (the retained portion is protected, not consumed). *(Note: the release
quantum is `work_state.abundance / n_subdt` computed from step-start abundance,
not the declining `inst_abd` — a pre-existing approximation, not introduced
here; `egg_retained` can reach the 0-clamp early under heavy initial predation.)*

### Change — Numba hot path (PRIMARY)

`_apply_predation_numba` is the shared prey-consumption kernel, called from all
three driver kernels: `_mortality_in_cell_numba` (`:1139`),
`_mortality_all_cells_numba` (`:1306`), `_mortality_all_cells_parallel`
(`:1485`).

1. Add an `egg_retained` float64 array parameter to `_apply_predation_numba`
   (def `:810`). At the prey-availability read (`abd_q = inst_abd[q_idx]`,
   `:886`) use `abd_q = inst_abd[q_idx] - egg_retained[q_idx]` with an inline
   `if abd_q < 0.0: abd_q = 0.0` clamp (Numba can't call a Python helper).
2. Add `egg_retained` to the signatures of `_mortality_in_cell_numba`,
   `_mortality_all_cells_numba`, `_mortality_all_cells_parallel`, and pass it
   through to each `_apply_predation_numba` call site.
3. At the production dispatch in `mortality()` (`_batch_fn(...)` call,
   `mortality.py:~1939`), pass `work_state.egg_retained`. For the
   `_mortality_in_cell` fallback dispatch, pass it there too.
4. `@njit(cache=True)` signature changes invalidate the Numba cache
   automatically on first run; no manual action, but the test run will recompile.

### Change — pure-Python fallback (SECONDARY, for behavioural parity)

`_apply_predation_for_school` already receives `state: SchoolState`, and
`state.egg_retained` is current at call time — **no new parameter needed**
(v1 over-specified this). At the school-prey availability read
(`inst_abd_q = inst_abd[q_idx]`, `:397`) subtract `state.egg_retained[q_idx]`
with a ≥0 clamp. Resource prey are never eggs → unchanged.

### Parity gate (REQUIRED)

This changes egg predation for **every** config. After implementing, re-run
`tests/test_engine_parity.py` (BoB Python-vs-stored-baseline; **note this file is
BoB-only — no EEC, no Java cross-check**, see Open Questions). If it shifts
beyond tolerance: **stop and report the deltas** for a human decision — do not
loosen the tolerance or re-bless the baseline without sign-off.

### Test

`tests/test_engine_egg_retention.py` — runs on the default (Numba) path. Build a
single cell with one hungry predator and one egg-cohort prey, `n_subdt` (e.g. 4),
and a predator whose appetite is **strictly between `egg_abundance/n_subdt` and
`egg_abundance`** (so the buggy code eats the whole cohort while the fix can only
eat the sub-dt-1 released slice). Run `mortality(...)`; assert the egg cohort's
surviving abundance is `> 0` (and ≳ the retained fraction). Fails today (cohort
wiped), passes after. *(A bare "≥ retained fraction" with default appetite is
non-falsifiable because post-run `egg_retained` is 0 and abundance is clamped at
≥0 — the appetite window is what makes it falsifiable.)*

---

## Fix 2 — Fleet-effort on the Python fishing path (FALLBACK PATH only)

### Bug (scope corrected: `_HAS_NUMBA=False` only)

Fleet-effort scaling lives in `_precompute_effective_rates` (`:778-794`), which
feeds **the Numba path** — so production already applies it. The gap is in the
pure-Python fallback: `_apply_fishing_for_school` (`:180`) computes `f_rate` with
year/season overrides but **no fleet-effort term** and is never passed
`fleet_state`. This path runs **only when `_HAS_NUMBA=False`** (v1 wrongly said
"whenever `bioen_enabled` or Numba absent"; the `bioen_enabled` gate lives inside
`_mortality_in_cell`, which is itself the no-Numba fallback). So this is a
low-production-impact correctness/consistency fix for Numba-absent installs.

### Change

Extract the scaling at `:778-794` into a shared helper:

```python
def _fleet_effort_factor(sp_id, cell_y, cell_x, fleet_state) -> float:
    """Multiplicative fishing-effort factor. 1.0 when fleet_state is None OR
    sp_id isn't targeted by any fleet (base F unchanged). For a TARGETED species:
    sum of effort_map across fleets at (cell_y, cell_x); 0.0 if the cell is out
    of bounds (no fishing there). Mirrors mortality.py:778-794 exactly."""
```

- The cell comes from `state.cell_y[i] / cell_x[i]` (`mortality.py:787`); no
  `grid` arg. Targeted set = `union(f.target_species for f in fleet_state.fleets)`.
- `_precompute_effective_rates` replaces its inline block with per-school helper
  calls — behaviour-preserving for the Numba path.
- `_apply_fishing_for_school` gains a `fleet_state` parameter and multiplies its
  `f_rate` by `_fleet_effort_factor(sp, state.cell_y[idx], state.cell_x[idx],
  fleet_state)`; the Python fishing call site (`:1737`) passes `ctx.fleet_state`.

### Test

`tests/test_engine_fishing_fleet_python_path.py` — **must force the fallback**
with `mock.patch("osmose.engine.processes.mortality._HAS_NUMBA", False)` (the
default dispatch routes through Numba, where this code is never reached; without
the patch the test is vacuous). A minimal `fishing_enabled` config + a
`fleet_state` whose `effort_map` doubles effort in the school's cell; assert the
Python-path fishing deaths reflect the ~2× effort-scaled F. Companion assert:
`fleet_state=None` leaves F unchanged.

---

## Fix 3 — `_worker_eval`: stop swallowing programming errors

### Bug (corrected per L1)

`_evaluate_candidate` (`problem.py:245`) already catches `_python_engine_errors`
internally (`:266`) and returns `[inf]*n_obj` for *expected* model failures.
Therefore those errors **never reach** `_worker_eval` — its `except Exception:
return [inf]` (`problem.py:98`) only ever catches **unexpected** programming
errors (TypeError/AttributeError), silently turning a real bug into `inf` and
poisoning the Pareto front. (Adding `except _python_engine_errors` would be dead
code; that's not the fix.)

### Change

**Remove** the `try/except Exception` wrapper in `_worker_eval` (`:96-99`) — call
`_WORKER_PROBLEM._evaluate_candidate(run_id, params)` directly and let unexpected
exceptions propagate (the `pool.submit` future re-raises; the existing
`BrokenProcessPool`/result handling at `:284-302` surfaces worker death).
Update the docstring (`:94`, currently "never raise into the pool") to: returns
`[inf]*n_obj` for expected model errors **via `_evaluate_candidate`**; raises for
unexpected programming errors.

### Test

`tests/test_calibration_worker_eval.py` — `_worker_eval` asserts
`_WORKER_PROBLEM is not None` (`:95`), so patch the module global:
`monkeypatch.setattr("osmose.calibration.problem._WORKER_PROBLEM", stub)` where
`stub = MagicMock(n_obj=1)` and `stub._evaluate_candidate.side_effect =
TypeError("bug")`. Assert `_worker_eval(0, params)` **raises `TypeError`** (does
not return `[inf, …]`). *(No "ValueError → inf" case here — that's
`_evaluate_candidate`'s job, not `_worker_eval`'s; testing it here would be
vacuous.)*

---

## Testing & gates

- New tests above (each red-first; egg + fleet tests target the correct path per
  their notes).
- Per-fix: `.venv/bin/ruff check` + `format --check` + `.venv/bin/pyright` on
  changed files.
- **Fix 1:** re-run `tests/test_engine_parity.py` (stop-and-report on a
  tolerance breach).
- Engine regression: `tests/test_engine_mortality*.py`,
  `tests/test_engine_predation*.py`, `tests/test_engine_bioen_*.py`,
  `tests/test_vectorized_rates.py`.
- Calibration regression: `tests/test_calibration_problem.py`.

## Files touched

- **Mod** `osmose/engine/processes/mortality.py` — `_apply_predation_numba` +
  `_mortality_in_cell_numba` + `_mortality_all_cells_numba` +
  `_mortality_all_cells_parallel` signatures & call sites (egg_retained);
  `_apply_predation_for_school` (read `state.egg_retained`); `mortality()`
  dispatch call sites; `_fleet_effort_factor` helper +
  `_precompute_effective_rates` + `_apply_fishing_for_school` + Python fishing
  call site.
- **Mod** `osmose/calibration/problem.py` — remove `_worker_eval` swallow +
  docstring.
- **New** `tests/test_engine_egg_retention.py`,
  `tests/test_engine_fishing_fleet_python_path.py`,
  `tests/test_calibration_worker_eval.py`.

## Risks

- **Numba signature ripple (Fix 1)** is the highest-risk edit: `egg_retained`
  must be threaded through all three driver kernels and the kernel itself, with
  matching positional argument order at all call sites (`:1139`, `:1306`,
  `:1485`, and the `mortality()` dispatch). A missed site fails to compile or
  diverges — the parity gate is the backstop.
- **Parity shift (Fix 1)** — addressed by the stop-and-report gate.
- **Fix 3** turns a previously-`inf` candidate into a hard run failure if an
  objective raises something outside `_python_engine_errors` — intended (surface
  real bugs); confirm the bundled objectives only raise expected types.

## Open Questions (for human decision)

1. **Parity gate scope.** `tests/test_engine_parity.py` is **BoB-only,
   Python-vs-stored-baseline** — no EEC, no Python-vs-Java cross-check (the
   "14/14 EEC" cited in CLAUDE.md lives elsewhere / is historical). Fix 1 changes
   egg predation in *all* configs. Is the BoB baseline gate sufficient, or should
   an EEC and/or Java cross-check be added before Fix 1 lands?
2. **Fix 2 priority.** Given the fleet-effort bug is `_HAS_NUMBA=False`-only
   (uncommon in production), keep it in this remediation or defer it?
3. **Fix 1 fallback completeness.** Is fixing the pure-Python
   `_apply_predation_for_school` (the SECONDARY change) required now for
   behavioural parity, or acceptable as a documented follow-up while landing the
   Numba hot-path fix?
