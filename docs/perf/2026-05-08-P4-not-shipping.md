# P4 — not shipping

## Measured delta

Implemented field-name caching on `SchoolState` (added `_FIELD_NAMES` class var, refactored `replace`, `append`, `compact`, `__post_init__` to iterate the cache instead of calling `dataclasses.fields()` each invocation), then benchmarked:

- Baseline (post-P2 master): 2.957 s eec_full 5-yr 7-repeat median
- After P4: 2.952 s
- Delta: **−0.17 % (5 ms — well within the 100 ms / 2 % noise floor)**

12/12 parity tests bit-exact (the cache change is purely structural — same field iteration order, same per-field operations).

## Why it didn't hit the gate

The plan-r2 ceiling estimate for the caching-only path was **1.3-2 %**, with the explicit caveat that `__post_init__`'s validation loop (a separate ~0.05 s in cProfile) is not eliminated by name caching alone. Measured 0.17 % confirms the gate-edge framing — the actual recoverable cost from `dataclasses.fields()` calls is much smaller in practice than the cProfile attribution suggested:

- `state.replace`: 0.036 s tottime / 3 966 calls = 9 µs / call. Caching saves ~2-3 µs / call (the dict-build's tuple-walk vs the fields()-walk). Aggregate: ~10 ms.
- `__post_init__` and `compact` get the same per-call shave but are called less frequently.

The field-name caching is the right technique for the right problem — it just measures below the gate on this fixture.

## P4b path (deferred — separate plan if pursued)

The plan's ~1.5 % unrecoverable share comes from `__post_init__`'s validation loop (`val.shape`, `val.ndim`, `len(val)` checks per field). Bypassing this via `object.__setattr__` on a pre-validated internal-construction path could potentially recover that — but at the cost of:
- Adding an `_internal` classmethod that skips the `__post_init__` shape checks.
- Auditing every `replace()` / `append()` / `compact()` caller to verify they're producing shape-correct outputs (the existing tests cover most of this implicitly via end-to-end runs, but a dedicated invariant test sweep is needed).

This is a meaningfully larger refactor than P4's caching change. **Do not pursue without a plan that establishes the invariant-coverage test sweep first.** The 1.5 % expected gain doesn't justify the refactor by itself; combine with another related cleanup (e.g. splitting `SchoolState` into a hot-path-frozen vs editable variant) if such a refactor is on the agenda.

## Do not retry without

- A scenario where `replace`-machinery dominates (long simulations with frequent re-binning, tight calibration loops with many short configs). On 5-yr eec_full the per-call overhead is too small.
- The invariant-coverage refactor described above as P4b — without `__post_init__` bypass the ceiling stays at ~0.17 % as measured.

## Reference

- Plan: `docs/plans/2026-05-08-post-v012-perf-plan.md` § P4
- Profile data: `/tmp/post_v012.prof` (post-P2 master, eec_full 5-yr warm cache)
