# SchoolState validation perf refactor — honest negative (NO-GO)

**Date:** 2026-07-14 · **Branch:** `perf/schoolstate-validation` · **Verdict:** do not merge for performance.

## Hypothesis (from the backlog)

`SchoolState.__post_init__` runs an O(30-field) structural-validation loop on every construction
(thousands per step). The backlog flagged removing it as "the biggest remaining perf lever
(~6–12%)." A quick best-of-5 A/B during brainstorming (no-op'd `__post_init__`, multi-threaded)
appeared to confirm it: baseline 3.672 s vs no-op 3.236 s → 436 ms (11.9%).

## What we built

Per the (in-loop-reviewed) spec + plan: deleted `__post_init__`, relocated the structural check to
an always-on `structural_check` called after each of the 12 step-loop `state = …` reassignments
(`_checked` tripwire), and hoisted `dataclasses.fields()` to a module-level `_FIELD_NAMES`. All
bit-identical (golden hash guard green throughout); the engine test suite stayed green.

## Verification gate (Task 5) — the phantom collapses

Robust measurement, Baltic `nyear=5`, **20 reps, median**, warm Numba:

| Variant | thread-pinned median |
|---|---|
| **master** (per-construction validation) | **6.674 s** |
| **branch, tripwire no-op'd** (NO validation anywhere) | **6.675 s** |
| **branch** (per-reassignment tripwire) | **6.710 s** |

Multi-threaded (noisier): master 3.492 s vs branch 3.633 s — branch again *slower*.

**Conclusion:** removing *all* validation is **identical to master** (6.675 vs 6.674 s — a 1 ms /
0.01% difference, pure noise). The per-construction `SchoolState` validation **does not cost
anything measurable.** The refactor delivers no win and the branch's tripwire adds a ~0.5%
regression.

## Why the brainstorming A/B was wrong

The 436 ms "win" was the **best-of-5 minimum** statistic on a noisy multi-threaded run — exactly the
optimistic-tail artifact the in-loop reviewers flagged when they hardened the Task 5 gate to
median-of-N. The construction count is also far lower than "millions": cProfile showed
`__post_init__` fired ~2,509×/3-yr run, so even the theoretical loop-count reduction was small, and
the per-loop cost (a few `getattr` + `len` metadata reads, O(1) per field) is negligible against the
Numba mortality kernel that dominates at ~57%.

## Recommendation

**NO-GO.** Do not merge the refactor for performance. master's existing per-construction validation
is effectively free *and* a safety net; the refactor removes that free net and adds a costlier,
weaker one. Cross the "`SchoolState` refactor / `__slots__` / remove per-construction validation"
item off the perf backlog — it is not a lever. The remaining real cost is the mortality cell-loop
(~57%, boundary-bound), which is a separate, harder effort.

**Process note:** the "re-profile before any perf plan" rule fired, but a quick best-of-5 A/B was
insufficient rigor and gave a false positive. The robust median-of-N gate (Task 5) caught it before
merge. Next time, run the robust measurement at the brainstorming gate, not just at verification.
