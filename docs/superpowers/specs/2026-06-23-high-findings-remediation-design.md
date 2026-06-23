# Deep-Review High-Findings Remediation — Design Spec

**Date:** 2026-06-23
**Status:** Draft for review
**Source:** 2026-06-22 deep review (`docs/reviews/2026-06-22-deep-review-v1.0.0.md`)
**Predecessor:** the review's one **critical** (bioen double-starvation) is already fixed — commit `457ac55`.

## Problem

Three high-severity findings from the deep review, each re-verified against the
current source:

1. **Egg-retention is a no-op (correctness).** `_mortality_in_cell` computes a
   graduated egg-release schedule — `egg_retained` is seeded to full egg
   abundance (`mortality.py:1795`) and decremented by `abundance/n_subdt` at the
   start of each sub-dt (`mortality.py:1915-1920`) — but **predation never reads
   it**. Both predation paths use `inst_abd[q_idx]` as egg prey availability
   (Python `mortality.py:397`; Numba `mortality.py:886`), and `inst_abd` starts
   at full abundance (`work_state.abundance.copy()`). A predator can therefore
   eat an entire egg cohort in sub-dt 1, defeating the design and inflating egg
   mortality / depressing recruitment.

2. **Fleet-effort (DSVM economics) dropped on the Python fishing path
   (correctness divergence).** Fleet-effort scaling lives only in
   `_precompute_effective_rates` (`mortality.py:778-790`, gated on
   `fleet_state is not None`), which feeds the **Numba** cell loop. The Python
   fallback path — taken whenever `bioen_enabled` or Numba is absent
   (`mortality.py:1611-1615`) — applies fishing via `_apply_fishing_for_school`
   (`mortality.py:180`), which computes `f_rate` with year/season overrides but
   **no fleet-effort term** and is never passed `fleet_state`. DSVM fleet
   economics thus has zero effect on the Python path. Same root cause as the
   fixed critical: the Python fallback diverged from the precompute/Numba path.

3. **`_worker_eval` swallows programming errors to `inf` (narrowed).** The
   calibration backend asymmetry the review flagged is *mostly already
   mitigated* — `_python_engine_errors` widens the thread/serial catch
   (`problem.py:47`, with explanatory comments at `:252-268`). The residual real
   issue: the process-pool worker `_worker_eval` still does
   `except Exception: return [inf]*n_obj` (`problem.py:98`), so a genuine
   `TypeError`/`AttributeError` in an objective function silently becomes `inf`
   in worker processes — hiding the bug and poisoning the Pareto front.

## Goals / Non-goals

**Goals:** fix findings 1-3 above, each with a focused failing test first; keep
the 12/12 EEC + 8/8 BoB parity within the existing "within 1 OoM" tolerance.

**Non-goals (deferred):** packing the 14 `_shared_*` calibration nonlocals into a
frozen `CalibrationRun` dataclass (a larger, riskier refactor of live handlers);
all medium/low review findings.

## Fix 1 — Egg-retention: gate egg predation on the released fraction

### Model

At sub-dt *k*, the **eatable** egg count is the released-and-still-alive
portion: `inst_abd[q] − egg_retained[q]` (clamped ≥ 0). For non-eggs
`egg_retained == 0`, so this is identically `inst_abd[q]` — no behavior change.
The release loop already advances `work_state.egg_retained` *before* the cell
loop each sub-dt, so the current value is correct at predation time. Deaths still
decrement `inst_abd` (the retained portion is protected, not consumed).

### Change

A small pure helper expresses the cap once:

```python
def _eatable_abundance(inst_abd_q: float, egg_retained_q: float) -> float:
    """Prey availability seen by predation: total instantaneous abundance minus
    the still-retained (not-yet-released) egg fraction. egg_retained is 0 for
    non-eggs, so this equals inst_abd_q there."""
    e = inst_abd_q - egg_retained_q
    return e if e > 0.0 else 0.0
```

- **Python predation** (`_apply_predation_for_school`, `mortality.py:~355-520`):
  where it reads `inst_abd_q = inst_abd[q_idx]` (line 397) for a *school* prey,
  use `_eatable_abundance(inst_abd[q_idx], egg_retained[q_idx])`. The function
  must receive the `egg_retained` array (add a parameter; the caller has
  `work_state.egg_retained`). Resource prey are never eggs → unchanged. Deaths
  continue to decrement `inst_abd[q_idx]`.
- **Numba predation** — the prey-availability read is in `_apply_predation_numba`
  (`mortality.py:801`, read at `:886`), called by `_mortality_in_cell_numba`
  (`:1069`) and the all-cells/parallel variants (`:1206`, `:1373`). Add an
  `egg_retained` float64 array parameter to `_apply_predation_numba` and thread
  it down from `_mortality_in_cell` (which already holds
  `work_state.egg_retained`) through every Numba kernel in that call chain. At
  the prey-availability read, subtract `egg_retained[q_idx]` with an inline ≥0
  clamp (Numba can't call the Python helper — mirror its two-line logic). Grep
  every `inst_abd[q` / `abd_q = inst_abd` read in the kernels and treat all.

### Parity gate (REQUIRED)

This changes egg predation for **every** config, so it will shift Python-engine
outputs. Java implements graduated release, so the fix should move Python
*closer* to Java. After implementing, **re-run the full parity suite** (12/12 EEC
+ 8/8 BoB). If any case exceeds the existing tolerance: **stop and investigate**
(report the deltas) — do **not** loosen the tolerance to make it pass. A genuine
improvement or a within-tolerance shift is acceptable; a regression past
tolerance is a blocker requiring human decision.

### Test

`tests/test_engine_egg_retention.py`: build a single cell with one hungry
predator and one egg-cohort prey, `n_subdt > 1`, predator able to consume far
more than `abundance/n_subdt`. Run one mortality step (`mortality(...)`); assert
the egg cohort's surviving abundance ≥ the retained fraction at sub-dt 1 (i.e.
the predator could NOT eat the whole cohort immediately) — fails today (eggs
fully eaten), passes after. Cover both the Numba and Python paths (parametrize on
`bioen_enabled` / a Numba-availability toggle if practical, else assert the
shared invariant the helper enforces).

## Fix 2 — Fleet-effort: shared effective-F across both paths

### Change

Extract the fleet-effort scaling currently inline in
`_precompute_effective_rates` (`mortality.py:778-790`) into a small shared helper:

```python
def _fleet_effort_factor(sp_id, cell_y, cell_x, fleet_state) -> float:
    """Multiplicative fishing-effort factor for a school. Returns 1.0 when
    fleet_state is None OR sp_id is not targeted by any fleet (base F unchanged).
    For a TARGETED species: the sum of effort_map across fleets at (cell_y,
    cell_x), or 0.0 if that cell is out of bounds (no fishing there). Mirrors the
    exact semantics of mortality.py:778-794 — only targeted species are scaled,
    non-targeted keep base F."""
```

The cell comes from `work_state.cell_y[i] / cell_x[i]` (NOT a grid lookup — this
is exactly what `:786` uses), so no `grid` arg is needed. The targeted-species
set is `union(f.target_species for f in fleet_state.fleets)`.

- `_precompute_effective_rates` (`mortality.py:778-794`) replaces its inline
  two-loop block with per-school calls to the helper — behavior-preserving for
  the Numba path. (The helper rebuilds the small `targeted` set per call;
  negligible — called per-school per-timestep, not per-sub-dt, and `n_fleets` is
  tiny. If profiling later flags it, hoist the set.)
- `_apply_fishing_for_school` (`mortality.py:180`) gains a `fleet_state`
  parameter and multiplies its computed `f_rate` by
  `_fleet_effort_factor(sp, state.cell_y[idx], state.cell_x[idx], fleet_state)`.
  The Python loop call site (`mortality.py:1737`) passes `ctx.fleet_state`
  (available — the enclosing `_mortality_in_cell` already receives `ctx`,
  `mortality.py:~1590`).

Non-fleet runs (`fleet_state is None`) are unchanged (factor 1.0). The Numba path
is behavior-preserving (same math, now via the helper).

### Test

`tests/test_engine_fishing_fleet_python_path.py`: a minimal config with
`fishing_enabled`, one species, and a `fleet_state` whose `effort_map` doubles
effort in the school's cell. Drive the **Python** path (set `bioen_enabled=True`
or otherwise force the fallback) and assert the fishing deaths reflect the
effort-scaled F (≈2× the no-fleet deaths), not the unscaled rate — fails today,
passes after. A companion assert that `fleet_state=None` leaves F unchanged.

## Fix 3 — `_worker_eval`: narrow the exception swallow

### Change

In `osmose/calibration/problem.py:93-99`, change
`except Exception:  # noqa: BLE001` to `except _python_engine_errors as exc:` —
the same expected-error set the in-process backends use — returning
`[inf]*n_obj` for *expected* model/objective failures. Let unexpected exceptions
(TypeError/AttributeError/etc.) propagate so the `pool.submit` future raises and
the run surfaces the bug (the existing `BrokenProcessPool`/result handling at
`:284-302` already deals with worker death). Optionally log the unexpected
exception before it propagates.

### Test

`tests/test_calibration_worker_eval.py`: a stub problem whose objective raises a
`TypeError` (a programming bug, NOT in `_python_engine_errors`); assert
`_worker_eval` **re-raises** (does not return `[inf, …]`). A second case: an
objective raising a `_python_engine_errors` member (e.g. `ValueError`) still
returns `[inf]*n_obj`.

## Testing & gates

- New unit tests above (each red-first).
- Per-fix: `.venv/bin/ruff check` + `format --check` + `.venv/bin/pyright` on
  changed files.
- **Fix 1 only:** full parity suite re-run — `tests/test_engine_parity.py` (the
  12/12 EEC + 8/8 BoB cases; verify the exact case list in that file) must stay
  within tolerance, else stop-and-investigate.
- Engine regression: `tests/test_engine_mortality*.py`,
  `tests/test_engine_predation*.py`, `tests/test_engine_bioen_*.py`.
- Calibration regression: `tests/test_calibration_problem.py`.

## Files touched

- **Mod** `osmose/engine/processes/mortality.py` — `_eatable_abundance` +
  `_fleet_effort_factor` helpers; Python + Numba predation prey-availability;
  `_apply_fishing_for_school` + `_precompute_effective_rates`; Python-loop call
  sites.
- **Mod** `osmose/calibration/problem.py` — `_worker_eval` except narrowing.
- **New** `tests/test_engine_egg_retention.py`,
  `tests/test_engine_fishing_fleet_python_path.py`,
  `tests/test_calibration_worker_eval.py`.

## Risks

- **Numba kernel signature change (Fix 1)** is the highest-risk edit: every
  Numba predation variant that reads prey availability must get `egg_retained`
  and the clamp, or the single/all-cells/parallel paths diverge. Mitigation:
  grep every `inst_abd[q` / `abd_q = inst_abd` read in the kernel and treat all;
  the parity gate is the backstop.
- **Parity shift (Fix 1)** — addressed by the stop-and-investigate gate above.
- **Cell lookup in `_apply_fishing_for_school` (Fix 2)** must use
  `state.cell_y[idx] / cell_x[idx]` — the same basis `_precompute_effective_rates`
  uses (`mortality.py:786`) — to avoid an off-by-cell effort factor.
- **Fix 3** could turn a previously-`inf` candidate into a hard run failure if an
  objective legitimately raises something outside `_python_engine_errors` —
  that's the intended behavior (surface real bugs), but the implementer should
  confirm the bundled objectives only raise expected types.
