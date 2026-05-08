# P4b — `__post_init__` bypass on SchoolState hot paths

> Created: 2026-05-08
> Branch: `feature/p4b-post-init-bypass`
> Status: r2 (post-review)
>
> **Revision log:**
> - 2026-05-08 r1 — initial draft.
> - 2026-05-08 r2 — applied review finding: scope extended to two additional internal hot-path callers that construct `SchoolState(**...)` directly (`reproduction.py:175-191` for spawning-merge, `simulate.py:611-622` for `_strip_background`). They use the same shape-correct construction pattern as `append`/`compact`, so the bypass is safe and the savings estimate now includes their contribution to the 4 191-call profile total. Also added a `__slots__` future-proofing docstring note on `_construct_unchecked`.
> Predecessor: [`docs/perf/2026-05-08-P4-not-shipping.md`](../perf/2026-05-08-P4-not-shipping.md) — P4 caching-only path measured 0.17 %

## What this changes

P4 (caching `dataclasses.fields()` result) measured below the gate because the dominant cost of `replace()` / `append()` / `compact()` is not the `fields()` call itself but the `__post_init__` shape-validation loop that runs on **every** `SchoolState` construction. Per the post-P2 cProfile:

```
state.py:141(replace)        0.031 s tottime  3 280 calls  (cumtime 0.180 s on baltic)
state.py:83(__post_init__)   0.044 s tottime  4 191 calls
state.py:204(compact)        0.034 s tottime    120 calls
state.py:150(append)         0.025 s tottime  varies
dataclasses.fields()         0.038 s tottime  7 931 calls (most from replace/append/compact)
```

P4b combines two complementary optimisations:
1. **Field-name cache** (formerly P4) — `_FIELD_NAMES: ClassVar[tuple[str, ...]]` populated at module load; replace/append/compact/__post_init__ iterate the cache instead of `dataclasses.fields(self)`.
2. **`__post_init__` bypass** — internal `_construct_unchecked` classmethod uses `cls.__new__(cls) + object.__setattr__` to build a `SchoolState` without triggering `__post_init__` validation. The hot paths (`replace`, `append`, `compact`) use it because they consume already-validated arrays from `self`.

External callers (anything using `SchoolState(...)` directly or `SchoolState.create(...)`) still go through `__post_init__` — validation is preserved at the entry-points.

## Safety contract

`_construct_unchecked` is **internal-only**:
- `replace(**kwargs)` — kwargs override specific fields; the shapes match `self`'s n_schools because callers pass arrays of the same length (predation outputs, mortality updates, growth recomputations).
- `append(other)` — np.concatenate produces correctly-shaped outputs by construction.
- `compact()` — boolean-indexed views preserve dtype + 2D-shape; new length is sum(alive_mask) which is consistent across every field.

If a caller passes a wrong-shape array via `kwargs`, the bug surfaces later (typically a numpy broadcasting error in the next process). The validation hop on the public `SchoolState(...)` path is preserved precisely so this can't slip through external code.

## Implementation surface

`osmose/engine/state.py`:
- Add `_FIELD_NAMES: ClassVar[tuple[str, ...]]` declaration inside the class.
- Add `_construct_unchecked(cls, **values)` classmethod with a docstring note that it assumes `__dict__`-backed fields (no `__slots__`); if `__slots__` is later added, the method's setattr loop continues to work but the safety analysis must be re-checked.
- Refactor `__post_init__` to iterate `self._FIELD_NAMES` instead of `fields(self)`.
- Refactor `replace` / `append` / `compact` to (a) iterate the cache, (b) use `cls._construct_unchecked(**values)` instead of `cls(**values)`.
- Set `SchoolState._FIELD_NAMES = tuple(f.name for f in fields(SchoolState))` at module bottom.

`osmose/engine/processes/reproduction.py:175-191`:
- The spawning-merge block iterates `fields(state)` directly, builds `merged_fields`, then calls `SchoolState(**merged_fields)`. Replace with `SchoolState._FIELD_NAMES` iteration + `SchoolState._construct_unchecked(**merged_fields)`. The arrays are produced by `np.concatenate(parts)` — same shape-correct contract as `append`.

`osmose/engine/simulate.py:611-622`:
- `_strip_background` slices each field to `[:n_focal]` and constructs a new `SchoolState`. Same pattern: replace with `SchoolState._FIELD_NAMES` iteration + `SchoolState._construct_unchecked(**sliced)`. Slice operations preserve dtype and produce correctly-sized arrays by construction — same contract as `compact`.

No public API change. External callers using `SchoolState(...)` directly or `SchoolState.create(...)` still go through `__post_init__`.

## Measurement protocol

Same as A1/A2/K1: 7-repeat median, discard run 1, side-by-side compare via `--compare`. Run on **both eec_full and baltic** (per the P4 plan's "ship gate is on eec_full" caveat — but P4b is a state-machinery change that should help on any fixture, and baltic shows a higher relative cost surface).

**Ship gate:** ≥ 2 % wall-time improvement on **at least one** of {eec_full, baltic} 5-yr 7-repeat median, AND 12/12 parity tests bit-exact, AND no shape-mismatch test regressions in the existing suite.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Wrong-shape array slips through `kwargs` to a hot path | Low — current callers all produce correctly-sized arrays from the existing implementation | The 12/12 parity tests + 2740-test suite catch shape-mismatches downstream. Add an explicit shape-mismatch unit test that constructs a SchoolState via the **public** `SchoolState(...)` path with a wrong-shape array and verifies `__post_init__` still raises. |
| `__post_init__` is a public hook that some test depends on running | Low — only constructed paths are: `cls(...)`, `cls.create(...)` (both keep validation), `cls._construct_unchecked(...)` (bypass) | Tests using `SchoolState(...)` directly remain validated. Audit `tests/` for direct construction calls and confirm none rely on `__post_init__` running on the hot-path output of `replace/append/compact` (validation already happened on the inputs). |
| `object.__setattr__` on a frozen dataclass is dis-allowed | None | Frozen dataclass `__setattr__` raises FrozenInstanceError, but `object.__setattr__` bypasses Python's MRO and writes directly. Standard pattern. |
| `cls.__new__(cls)` skips dataclass-generated `__init__` (which sets up the frozen fields) | Need to verify | Frozen dataclasses generate an `__init__` that calls `object.__setattr__` for each field. `__new__` allocates the instance without running `__init__`; we then explicitly setattr each field via `object.__setattr__`. Net result: same memory layout. |

## Acceptance

1. New unit test: `tests/test_engine_state.py::test_construct_unchecked_skips_validation` — constructs a SchoolState via `_construct_unchecked` with a deliberately wrong-shape array; verifies it does NOT raise (shape validation is bypassed). Public `SchoolState(...)` path with the same wrong-shape array MUST raise.
2. New unit test: `tests/test_engine_state.py::test_replace_preserves_validation_invariants` — calls `replace(...)` with various correctly-shaped overrides; verifies output passes `validate()` (the existing opt-in invariant check).
3. eec_full 5-yr OR baltic 5-yr benchmark median improves ≥ 2 % vs post-P3/P4 master.
4. 12/12 parity tests bit-exact.
5. Full suite (2740 tests) green; ruff clean.

## Estimated saving

Per the post-P2 baltic profile:
- `__post_init__`: 44 ms tottime / 4 191 calls = ~11 µs per call
- `state.replace`: 180 ms cumtime — most is downstream `__post_init__`
- Combined elimination: ~50-100 ms savings on baltic = **2-4 %** of 2.6 s baseline

On eec_full the absolute saving is similar (~50 ms), so 1.5-2 % of 2.9 s — borderline. **Likely passes gate on baltic, borderline on eec_full.**

## What NOT to do

- Do not bypass validation on `cls.create(...)` — that's a one-shot factory and the validation cost is amortised; keeping it preserves the contract for new-state construction.
- Do not export `_construct_unchecked` as public API — the underscore prefix is the contract; only state.py methods call it.
- Do not extend the bypass to the `validate()` invariant checker — that's a different (opt-in) function called from tests.
- Do not bundle other state.py refactors (e.g. switching to `__slots__`) into this PR — separate concerns, separate review surface.
