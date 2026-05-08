# P4b — not shipping

Combined `_FIELD_NAMES` cache + `_construct_unchecked` (`__post_init__` bypass via `object.__setattr__` on `cls.__new__(cls)`) implemented and benchmarked. **Below the 2 % gate on both eec_full and baltic.**

## Measured delta

| Fixture | Baseline | After P4b | Δ ms | Δ % |
|---|---:|---:|---:|---:|
| eec_full 5-yr | 2.975 s | 2.922 s | 53 ms | **1.8 %** |
| baltic 5-yr | 2.613 s | 2.590 s | 23 ms | **0.9 %** |

Plan-r2 ship gate: ≥ 2 % wall-time improvement on at least one of {eec_full, baltic}. Both fixtures fall below the gate. **Drop with not-shipping post-mortem.**

12/12 Java parity tests bit-exact + 6 new acceptance tests pass (`test_engine_state.py::TestPostInitBypass`):
- `test_field_names_cache_matches_dataclass_fields`
- `test_construct_unchecked_skips_validation` (verifies public `SchoolState(...)` still validates while the internal bypass does not)
- `test_replace_preserves_validation_invariants`
- `test_replace_returns_distinct_instance`
- `test_compact_unchecked_yields_valid_state`
- `test_append_unchecked_yields_valid_state`

The implementation is sound; the gate is the issue.

## Why it didn't hit

Plan-r2 ceiling estimate was 2-4 % on baltic, with eec_full as "borderline." Reviewer-r1 pre-flagged that the cited 4 191-call cProfile total includes calls from `reproduction.py:191` and `simulate.py:622` — which the r2 patch added to scope. Even so, the realised saving on baltic was 23 ms / 0.9 %, well below the floor.

Likely explanations:
1. **`__post_init__`'s 11 µs/call cost was a profile-attribution artefact.** cProfile attributes setup overhead (frame allocation, bytecode preamble) to the function being entered, not just its body. A function whose body is a fast loop over ~26 fields with branching is dominated by the loop body itself; the bypass eliminates the body but a residual frame-entry cost remains because hot paths still construct objects (just via `__new__` + N `setattr` calls instead of `__init__`).
2. **`object.__setattr__` per field has its own per-call cost.** ~0.5-1 µs for a Python-level setattr on a frozen dataclass, × 26 fields = 13-26 µs/object. Comparable in magnitude to the `__post_init__` cost we eliminated. Net win is the difference between two similar-cost paths, not the full `__post_init__` cost.
3. **The post-P2 hot path is dominated by the JIT'd kernel.** `mortality()` Python wrapper at 1.0-1.2 s tottime is mostly waiting for the Numba kernel; reducing wrapper-side Python overhead by 50 ms shaves the wrapper's intrinsic cost but doesn't change the kernel's wall-clock contribution. The denominator effect makes any wrapper-side win look smaller than the absolute saving suggests.

## Implementation correctness

The implementation is clean and the contract is sound — keep the patch as a reference if anyone re-attempts. Diff summary:

- `osmose/engine/state.py` — `_FIELD_NAMES: ClassVar[tuple[str, ...]]` declared, `_construct_unchecked` classmethod added, `replace` / `append` / `compact` / `__post_init__` switched to use the cache + bypass.
- `osmose/engine/processes/reproduction.py:175-191` — spawning-merge switched to `SchoolState._FIELD_NAMES` + `_construct_unchecked`.
- `osmose/engine/simulate.py:611-622` — `_strip_background` switched to `SchoolState._FIELD_NAMES` + `_construct_unchecked`.

All 12/12 parity tests + 20 state tests + 46 cross-process tests passed. No correctness regression. Diff was reverted; only the plan and this post-mortem ship.

## Do not retry without

- A scenario where SchoolState construction frequency is materially higher than on eec_full / baltic (e.g., ensemble simulations with many short configs, long calibration loops). On 5-yr single-config runs the construction cost is ~50 ms — too small to clear the gate.
- A combined refactor that pairs the bypass with `__slots__` on SchoolState. `__slots__` would eliminate the per-field `__dict__` write entirely, possibly recovering more time. But it's a meaningful refactor that needs its own plan + invariant-coverage test sweep (the SchoolState fields list is 26 entries; getting the slot declaration right + handling serialization/deepcopy/match-args properly is non-trivial).
- A move away from the per-construction validation idiom entirely. If `SchoolState` becomes a typed-but-mutable container with explicit invariant-check call sites, the bypass becomes default behaviour and the validation cost disappears.

## Reference

- Plan: `docs/plans/2026-05-08-p4b-post-init-bypass-plan.md`
- Predecessor: `docs/perf/2026-05-08-P4-not-shipping.md` (caching-only path measured 0.17 %)
- Profile data: `/tmp/post_v012.prof` (post-P2 master)
- Benchmark JSONs: `/tmp/master_eec_post.json`, `/tmp/p4b_eec.json`, `/tmp/master_baltic_post.json`, `/tmp/p4b_baltic.json`
