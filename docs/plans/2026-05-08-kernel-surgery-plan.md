# Kernel Surgery Plan — remaining Phase 4 perf wins

> Created: 2026-05-08
> Branch: `docs/kernel-surgery-plan`
> Status: predecessor plan
> [`docs/plans/2026-05-05-deep-review-remediation-plan.md`](2026-05-05-deep-review-remediation-plan.md)
> shipped 31 PRs (Phases 1, 2, 3, 5a, 5b complete; Phase 4 partially shipped: H7 #29, bench infra #30, H6 cleanup half #31). The kernel-level perf items the original plan listed were deferred when measured deltas came in below the 2 % noise floor under the bench fixtures available at the time; this plan picks them up with a tighter measurement protocol and explicit per-item parity-preservation strategy.
>
> **Anchor convention.** All file:line refs in this plan are relative to master at the time of writing (post-#31, master `e0e1d94`). Implementing K-items in any order other than the recommended sequence will shift line numbers — anchor by **function name + symbol** (e.g., `_apply_predation_numba::Phase 1 prey-scan loop`) when re-applying after a sibling K-item merges. This is called out per-K below.
>
> **Revision log:**
> - 2026-05-08 r1 — initial draft.
> - 2026-05-08 r2 — applied review findings: tightened K2 parallel-path safety language; corrected cache-key mitigation (keyed on `__version__`, not RNG output); reframed K4 as a gate not a recommendation; defined per-K parity contracts (K1/K3 bit-exact, K2 within-Python-deterministic + tolerance band only); replaced unimplementable `nonlocal counter` acceptance check with Python-side wrapper; added mid-run trajectory-drift risk row; corrected K1 function range and K3 inst_abd line number.

## What this plan covers

Three kernel-surgery items the predecessor plan deferred:

- **K1 (H5):** hoist scratch buffers in `_apply_predation_numba` (`mortality.py:846-848`).
- **K2 (M14):** pool `cause_orders` per-cell allocation + inline a 4-element Fisher-Yates instead of `np.random.shuffle` (`mortality.py:1220-1226`).
- **K3 (full H6):** pool `n_dead` / `pred_success_rate` / `preyed_biomass` / `inst_abd` scratch buffers via `SimulationContext` with size-resize logic (`mortality.py:1700-1810`).

Each is a real optimization the original plan listed. None ship without (a) a measured ≥2 % wall-time improvement on the established eec_full 5-year benchmark, (b) bit-exact parity with the existing 12/12 Java parity tests, and (c) a regression test that pins the new behaviour.

## Pre-requisites

### Baseline (already established by predecessor plan PR #30)

```
.venv/bin/python scripts/benchmark_engine.py --config eec_full --years 5 --repeats 7
```

Current master median: **5.025 s** (1.005 s/yr). 2 % noise floor: 100 ms.

### Measurement protocol (same for every K-item)

1. **Before the change, on the working branch:** capture a 7-repeat median (`--output baseline.json`).
2. **After the change:** repeat with `--output current.json`.
3. **Discard runs 1 of each set** to amortise Numba JIT compilation; the median over the remaining 6 is the comparison.
4. Use `scripts/benchmark_engine.py --compare baseline.json current.json` for a side-by-side report (already built in PR #30; reports per-species final-biomass parity alongside timing).
5. **Ship gate:** post-change median must be ≤ 0.98 × pre-change median (≥ 2 % faster) AND species-final-biomass exact parity (every species ratio == 1.0 in the compare output).

### Parity gate (same for every K-item)

```
.venv/bin/python -m pytest tests/test_engine_parity.py -q
```

12/12 must remain bit-exact. Numba-kernel changes that touch the predation hot path are the most likely to drift parity due to non-associative float reductions (the H9 cross-thread test demonstrates the engine is currently deterministic across thread counts; the K-items must preserve that).

If a K-item only goes parity-tight on a subset of fixtures, that's a meaningful regression — defer.

## K1 — Predation scratch buffer hoisting

**Files.** `osmose/engine/processes/mortality.py:786+` (`_apply_predation_numba` — the function spans well past line 948; the three scratch allocs are at lines 846-848 and the prey-scan / phase-2 / TL paths continue beyond), `mortality.py:1156` (`_mortality_all_cells_numba` caller), `mortality.py:1308` (`_mortality_all_cells_parallel` caller). Anchor by symbol when re-applying after K2 or K3 lands first.

**Symptom.** `_apply_predation_numba` allocates three arrays at every call (`mortality.py:846-848`):
```python
prey_type = np.zeros(max_prey, dtype=np.int32)
prey_id = np.zeros(max_prey, dtype=np.int32)
prey_eligible = np.zeros(max_prey, dtype=np.float64)
```
Called per-school per-cell per-subdt. On eec_full at ~3 k schools × 24 dt × 100 cells × 1 subdt-per-call ≈ 7 M `np.zeros` calls per year — even at low μs each, that's 1–10 % of wall-time.

**Sketch.**
1. Allocate the three arrays **once per cell** in the caller (`_mortality_all_cells_numba` and `_mortality_all_cells_parallel`), sized `max_n_local + n_resources` for that cell. Pass into `_apply_predation_numba` as additional args.
2. In the parallel kernel (`prange`), allocate one set per `prange` iteration — each thread works on a disjoint cell, so there's no cross-thread aliasing.
3. Inside `_apply_predation_numba`, replace the `np.zeros(...)` lines with a zero-fill of the prefix `[:max_prey]` of the passed buffer. Numba will lower this to a `memset`.
4. Update both Numba callsites and the Python reference path that calls `_apply_predation_numba` (if any — verify with grep).

**Parity-preservation strategy.**
- The three arrays are **input scratch only** — they're populated then read inside `_apply_predation_numba`, then discarded. No state escapes through them. Reusing the buffer (zero-filled) is safe.
- Numba `prange` threads work on disjoint cells (`cell_indices` is a per-cell slice, so school indices in cell A do not appear in cell B). One set of buffers per `prange` iteration is the correct ownership.
- The zero-fill must run on every call so previous-call leftovers don't leak into `total_available` accumulation. Skipping the zero-fill (relying on `n_prey` as a high-water mark) is **not safe** because `prey_id`/`prey_eligible` are read by index in Phase 3 (`mortality.py:944-948`) and a stale entry past `n_prey` could be picked up if the algorithm changes.

**Acceptance.**
- Per-cell allocation count drops from `O(n_local × 3)` to `3` (per cell, per subdt). Verify by wrapping the kernel call site in a Python-side test harness that monkey-patches `np.zeros` with a counter (Numba calls `np.zeros` through a lowered intrinsic; the Python-side wrapper sees only the per-call entry. For an in-kernel allocation count, run the test with `_apply_predation_numba.py_func(...)` — the un-jitted reference — to count zeros calls in pure Python). Do NOT use `nonlocal` inside `@njit`; Numba won't compile that.
- eec_full 5-year median ≤ 0.98 × baseline (i.e. ≥ 100 ms faster).
- **Parity contract: bit-exact (see above).** Java parity tests 12/12 pass with no tolerance widening.
- New test: `tests/test_predation_scratch_buffer_reuse.py` — call `_apply_predation_numba` twice with deliberately-stale buffer contents from a prior call; assert the second call's `total_available` is correct (i.e. zero-fill happens).

**Risks.**
- Numba's `prange` may not allow per-iteration array allocation on the heap — Numba's own docs note that NumPy allocations inside `prange` body work but force per-thread heap traffic. The hoisted buffers should be allocated **outside** `prange`, then sliced per cell inside. This adds complexity (per-thread vs per-cell ownership). If the parallel path becomes too tangled, ship the sequential-path optimization only and keep `prange` allocations as-is.
- The `max_prey = max_n_local + n_resources` sizing is per-cell. If `max_n_local` varies across cells, the hoisted buffer must be sized to the **global** max — adds a precompute pass over `boundaries`. Trivial but easy to forget.

## K2 — Cause-order pooling + 4-element Fisher-Yates

**Files.** `osmose/engine/processes/mortality.py:1216-1226` (sequential), `mortality.py:1366+` (parallel).

**Symptom.** Per-cell, per-subdt:
```python
seq_pred = np.random.permutation(n_local).astype(np.int32)
seq_starv = np.random.permutation(n_local).astype(np.int32)
seq_fish = np.random.permutation(n_local).astype(np.int32)
seq_nat = np.random.permutation(n_local).astype(np.int32)
causes = np.array([0, 1, 2, 3], dtype=np.int32)
cause_orders = np.empty((n_local, 4), dtype=np.int32)
for ii in range(n_local):
    np.random.shuffle(causes)
    cause_orders[ii, 0] = causes[0]
    ...
```

Two independent costs:
- 5 array allocations per cell-subdt (4× `permutation(n_local)`, 1× `cause_orders((n_local, 4))`).
- `n_local` calls to `np.random.shuffle(causes)` where `causes` is length-4. `np.random.shuffle` has Python-level overhead even under Numba JIT; for a length-4 array, an inlined Fisher-Yates is dramatically cheaper.

**Sketch.**
1. Hoist `cause_orders` and the four `seq_*` buffers. Ownership rule (parallel kernel): allocate **one set per `prange` iteration**, not outside `prange`; threads work on disjoint cells (verified by the kernel docstring at `mortality.py:1352-1363`). Sequential kernel: a single set hoisted to the per-call frame is fine. Reuse across subdt within a cell either way. Size them to the **global** max `n_local` (one precompute pass over `boundaries` to find it). **Same caveat as K1's parallel path** — sharing a single buffer across `prange` threads is a data race.
2. Replace `np.random.shuffle(causes)` with an inlined Fisher-Yates of 4 ints:
   ```python
   # Length-4 inline Fisher-Yates: 3 random integer draws, 3 conditional swaps
   for j in range(3, 0, -1):
       k = np.random.randint(0, j + 1)  # Numba-safe: int from [0, j]
       tmp = causes[j]
       causes[j] = causes[k]
       causes[k] = tmp
   ```
   The bounded length lets Numba unroll; the call-overhead saving is the main win.

**Parity-preservation strategy.**
- The Fisher-Yates above produces the same permutation distribution as `np.random.shuffle` because both use the standard knuth-shuffle algorithm. **But the RNG draw order differs** if `np.random.shuffle` doesn't inline a length-4 specialisation — meaning the per-call RNG state advances differently.
- This breaks bit-exact parity with master (the 12/12 Java parity tests will fail because Python-side numbers will diverge from the previous Python-side numbers, even if Java-vs-Python parity is preserved).
- **Mitigation:** because Numba's RNG already differs from Java's MT19937, "Java parity" here means "tolerances pass within the documented 1 OoM". Bit-exactness is between-Python-runs only, not Python-vs-Java. So K2's RNG-stream change is acceptable **if** within-Python-determinism is maintained (same seed → same output), which the H9 single-thread determinism test pins.
- Concrete check: after K2, run `tests/test_jit_determinism.py::test_single_thread_is_deterministic` — must pass byte-equal.
- Concrete check: re-run `tests/test_engine_parity.py` — tolerances must still pass (they have headroom; M2 already established the parity tolerance has slack at the boundary).
- The bit-exact-master regression that K2 will introduce is a **breaking change** to between-version reproducibility. Document it in CHANGELOG and the per-PR description.

**Acceptance.**
- Allocation count per cell drops from `5 × n_subdt` to `5` (hoisted to per-cell scratch reused across subdt).
- 4-element Fisher-Yates inlined; `np.random.shuffle(length-4-array)` no longer in the hot loop.
- eec_full 5-year median ≤ 0.98 × baseline.
- `test_jit_determinism::test_single_thread_is_deterministic` passes byte-equal post-change.
- `tests/test_engine_parity.py` 12/12 within tolerance (NB: numbers will differ from master).
- New test: `tests/test_cause_order_distribution.py` — repeat the inlined Fisher-Yates 100 k times on `[0, 1, 2, 3]`, assert each of the 24 permutations appears within 5 % of the expected uniform frequency. Pins the algorithmic correctness.

**Risks.**
- Cross-version reproducibility: pre-K2 saved calibration cache files were generated with the previous RNG stream. The cache key at `osmose/calibration/problem.py:341-354` is keyed on `f"python-{__version__}"` — the cache invalidates **automatically** as long as the K2 PR also bumps `osmose/__version__.py` (the canonical source of truth, per `osmose/__version__.py`). Concrete mitigation: bump `osmose/__version__.py` in the K2 PR itself; `_cache_key` then naturally rejects pre-K2 cached evaluations. No additional cache-version constant needed.
- If `np.random.randint` inside `@njit` doesn't compile or has unexpected overhead, fall back to `np.random.uniform()` thresholding (`int(np.random.uniform(0, j+1))`).

## K3 — Full H6: ctx-pool work_state scratch

**Files.** `osmose/engine/processes/mortality.py:1665-1945` (the `mortality()` Python entry-point), `osmose/engine/simulate.py:33-63` (`SimulationContext` definition).

**Symptom.** Per simulation step, `mortality()` allocates four function-local arrays of size `n_schools` (or `n_schools × n_mortality_causes`):
- `larva_deaths = state.n_dead.copy()` (line 1700, kept after PR #31)
- `pred_success_rate = state.pred_success_rate.copy()` (line 1709)
- `preyed_biomass = state.preyed_biomass.copy()` (line 1710)
- `inst_abd = work_state.abundance.copy()` (line 1811)

`pred_success_rate` and `preyed_biomass` **escape to the caller** via `state.replace(...)` at line 1938 (which assigns them as the new state's arrays), so they cannot be naively pooled. `larva_deaths` and `inst_abd` are function-local and pool-able directly. (`n_dead` also escapes via `combined_n_dead` at line 1936 → 1938; not in scope for K3.)

**Sketch.**
1. Add scratch buffers to `SimulationContext`:
   ```python
   _h6_larva_deaths_scratch: NDArray[np.float64] | None = None  # (n_schools, n_causes)
   _h6_inst_abd_scratch: NDArray[np.float64] | None = None      # (n_schools,)
   ```
2. Add a `_get_or_resize(scratch, target_shape, dtype) -> NDArray` helper (in `mortality.py` or a new `osmose/engine/_scratch.py`) that returns the scratch array if shape matches, else allocates new and replaces. Schools count grows monotonically across a simulation (eggs hatch → cohorts), so resize is rare.
3. In `mortality()`, replace:
   ```python
   larva_deaths = state.n_dead.copy()
   ...
   inst_abd = work_state.abundance.copy()
   ```
   with:
   ```python
   larva_deaths = _get_or_resize(ctx._h6_larva_deaths_scratch, state.n_dead.shape, np.float64)
   np.copyto(larva_deaths, state.n_dead)
   ctx._h6_larva_deaths_scratch = larva_deaths
   ...
   inst_abd = _get_or_resize(ctx._h6_inst_abd_scratch, work_state.abundance.shape, np.float64)
   np.copyto(inst_abd, work_state.abundance)
   ctx._h6_inst_abd_scratch = inst_abd
   ```
4. **Do not** pool `pred_success_rate` / `preyed_biomass` / `n_dead`. These escape to the caller; the per-step copy from `state` is load-bearing for ownership separation.
5. (Optional, future) consider extending K3 to remove the *escape* by restructuring `state.replace(...)` at line 1938 to write into a pooled buffer that's then copied into the caller's state at the end. This is a bigger refactor that doubles the surgery risk; defer to a K3.5 if K3 alone misses the 2 % gate.

**Parity-preservation strategy.**
- `larva_deaths` and `inst_abd` are read-once / write-once buffers. Pooling them is safe iff:
  - The previous step's `larva_deaths` is fully consumed before the next step writes (it is — line 1936's `combined_n_dead = work_state.n_dead + larva_deaths` is the last read).
  - No external code holds a reference to either after `mortality()` returns. Verify with grep on `larva_deaths` / `inst_abd` outside `mortality.py`.
- `np.copyto` produces bit-exact output identical to `.copy()`. Java parity tests should pass unchanged.

**Acceptance.**
- Per-step allocation count drops by 2.
- eec_full 5-year median ≤ 0.98 × baseline (probably below — these are the smallest of the per-step allocations).
- Java parity tests 12/12 bit-exact.
- New test: `tests/test_mortality_ctx_scratch_pooling.py` — call `mortality()` twice on the same `state` with the same `ctx`; assert (a) `ctx._h6_larva_deaths_scratch` is the same array object across calls (pooling works), (b) results are bit-equal to a reference implementation that uses fresh `.copy()` calls.

**Risks.**
- If `len(state)` grows mid-simulation (eggs hatch into new schools), the scratch must be resized. The resize path is exercised every time a cohort is added — verify it triggers under the eec_full 5-year run by adding a temporary `print(scratch.shape, target_shape)` and counting resizes.
- `SimulationContext` is `@dataclass`-ish — verify fresh contexts are created per `simulate.run()` call. If `ctx` were reused across calls without reset, scratch pooling could leak data between runs. (`SimulationContext` is built fresh in `simulate.py:1056` per the existing structure; verify still true.)

## K4 (gating step) — bottleneck profiling

**K4 is a gate, not a recommendation.** Without it, K1-K3 attempts are blind and may all under-deliver against the 2 % wall-time threshold. K1 / K2 / K3 cannot start until K4 results justify each item per the thresholds below.

**Approach.**
1. Run the engine under `cProfile` on eec_full 5-year (single-threaded for clarity):
   ```
   .venv/bin/python -m cProfile -o /tmp/eec_full_5yr.prof scripts/benchmark_engine.py --config eec_full --years 5 --repeats 1
   ```
   With the JIT-compiled hot path, cProfile sees the Numba kernel as one big call. To break it up, add `nopython=False` temporarily (don't ship — just for profiling) or use `numba`'s built-in line profiler.
2. Annotate hot lines in `mortality.py` with `@njit(boundscheck=True, nogil=True)`-style debug prints behind a `if PROFILE:` flag (revert before shipping).
3. The output should answer specific go/no-go thresholds (below).

**Numeric gates** (K-item proceeds iff the corresponding profiler-measured share is met):

| K-item | Profiler-measured share required to proceed |
|---|---|
| K1 (predation scratch hoisting) | ≥ 15 % of `_apply_predation_numba` self-time spent on `np.zeros`-shaped allocations or memset paths |
| K2 (cause_orders + Fisher-Yates) | ≥ 5 % of total `mortality()` self-time spent in the `np.random.permutation` + `np.random.shuffle` block |
| K3 (ctx-pool larva_deaths + inst_abd) | ≥ 2 % of total `mortality()` self-time spent in the two `.copy()` calls |

If a K-item's profiled share falls **below half** of its gate threshold, drop it without further work and write the `not-shipping.md` entry now (template below). If it's between half and the threshold, still permitted to attempt but record the marginal expectation in the PR description.

**Acceptance.** A profiling-output table committed to `docs/perf/2026-05-08-eec_full-5yr-profile.md` listing the top 20 hot functions / lines and the per-call cost, with explicit go/no-go decisions per K-item. This becomes the canonical reference for future perf work.

## Per-K-item parity contracts

Different K-items have different parity-preservation strategies. Spelled out explicitly so the per-PR "parity gate" check is unambiguous:

| K-item | Parity contract | Verification |
|---|---|---|
| **K1** | Bit-exact (allocation hoisting is a memory-management refactor; algorithmic path unchanged) | `tests/test_engine_parity.py` 12/12 pass with **no tolerance widening** |
| **K2** | **Within-Python-deterministic; not bit-exact vs master.** RNG draw order changes when the Fisher-Yates is inlined, so per-step state diverges from pre-K2 master. | `tests/test_engine_parity.py` 12/12 pass within **existing** tolerances (no widening); `tests/test_jit_determinism::test_single_thread_is_deterministic` passes byte-equal post-merge (same seed → same output across reruns); explicitly call out the cross-version drift in the K2 PR description. |
| **K3** | Bit-exact (`np.copyto` + scratch reuse produce identical bytes to the original `.copy()` call) | `tests/test_engine_parity.py` 12/12 pass with no tolerance widening; new `test_mortality_ctx_scratch_pooling.py` asserts ref-impl bit-equality |

## Sequencing

K1, K2, K3 are independent in terms of code sites (they touch different files / functions) but **K4 is a hard prerequisite** — none of K1-K3 can start until K4's profile is published and the per-K numeric gates clear.

Recommended order:

1. **K4** (profiling) — gating step. **Required before any of K1-K3.**
2. **K3** (smallest blast radius: only `larva_deaths` + `inst_abd`). Likely under 2 % alone — drop early if K4 measures it under the 1 % half-gate.
3. **K1** (predation scratch hoisting) — touches the hot kernel; careful Numba `prange` ownership.
4. **K2** (cause_orders + Fisher-Yates) — highest impact-per-line but introduces RNG-stream change; must bump `osmose/__version__.py` for cache invalidation; explicit cross-version-reproducibility breaking change.

Each lands as its own PR with the measurement protocol's before/after JSON attached to the PR body. **If K1 or K3 lands first, K2's line numbers will shift** — anchor by function symbol when re-applying.

### `not-shipping.md` template

For each K-item that fails its 2 % gate after honest implementation, write a short post-mortem at `docs/perf/2026-05-{NN}-K{n}-not-shipping.md`. Required sections:

```markdown
# K{n} — not shipping

## Measured delta
- Baseline: <X.XXX>s (eec_full 5yr, 7-repeat median)
- After K{n}: <Y.YYY>s
- Delta: <±Z.Z%>

## Why it didn't hit the gate
<one paragraph: where the time actually went, vs the K-item's premise>

## Profile excerpt
<top-5 hot lines from K4's profile relevant to this K-item>

## Do not retry without
<bullet list: a different fixture, a profiling-driven hypothesis, a Numba upgrade, etc>
```

The point is to prevent a future engineer from re-attempting the same approach blind. Cite the file in CHANGELOG under a "Perf attempts not shipped" subsection.

## Acceptance / Done definition

The K-plan is complete when:

1. **K4** profiling artifact landed in `docs/perf/`.
2. **At least one** of K1-K3 has shipped a measured ≥ 2 % wall-time improvement on eec_full 5-year, with bit-exact (or within-1-OoM tolerated) Java parity preserved.
3. **For each K-item that fails the 2 % gate**: a `docs/perf/{date}-K{n}-not-shipping.md` write-up explaining the measured delta and why the architectural sketch didn't hit the target. Future engineers shouldn't re-attempt the same approach blind.
4. **Cumulative target** (stretch): ≥ 5 % combined wall-time improvement across K1+K2+K3.

## Out of scope

- **Reimplementing the mortality kernel in C/Rust.** Very-much-out-of-scope. Numba is the agreed runtime; pre-K-item profiling will determine whether kernel rewriting is even theoretically worth it.
- **Replacing PCG64 with MT19937 to gain Java bit-exact parity.** Per the existing RNG documentation in `osmose/engine/rng.py`, this is explicitly out of scope.
- **Calibration-side parallelism (ProcessPoolExecutor).** Tracked separately as a v0.10.0+ follow-up in `CHANGELOG.md`; not on the K-list.
- **Phase 4 M11 (memoise PythonEngine construction).** Verified non-meaningful in the predecessor plan's wrap-up: `PythonEngine.__init__` is two lines and Numba JIT is module-level cached.

## Risk register

| Risk | Mitigation |
|---|---|
| K1 hoisted buffer in `prange` triggers per-thread heap traffic, negating the win | Allocate one set per `prange` iteration (each thread owns its buffer for the cell it processes); do **not** share a single buffer across threads. K1's sketch above carries the same caveat. If complexity blows up, ship sequential-path only and keep `prange` per-iter alloc. |
| K2 RNG-stream change invalidates calibration cache | The cache key at `osmose/calibration/problem.py:341-354` is `f"python-{__version__}"`. Bumping `osmose/__version__.py` in the K2 PR auto-invalidates pre-K2 cached evaluations. Document in CHANGELOG as a breaking change to between-version reproducibility. |
| K2 ships sequential-path optimization but skips the parallel kernel — silent bit-flip when users toggle `parallel=True` | K2 must touch both `_mortality_all_cells_numba` and `_mortality_all_cells_parallel`, OR ship sequential-only behind an explicit `os.environ["OSMOSE_K2_PARALLEL"]` opt-in. Verify both `parallel=True` and `parallel=False` runs produce within-Python-deterministic, parity-test-passing output. |
| K3 scratch resize path is rare but breaks under cohort-add (egg hatch) | Add a defensive `assert scratch.shape >= target_shape or new_alloc` at the resize site; new test exercises the resize path explicitly. |
| Numba JIT cache invalidation: changing `_apply_predation_numba`'s signature (K1) forces full recompilation on first call after merge | Document expected recompile cost (one-time, ~few seconds); the first run after merge will be slower than steady-state. |
| **Mid-run trajectory drift undetected by terminal-state parity tests.** The 12 Java parity tests check final-step biomass/abundance — a kernel change could silently drift mid-run state if the final converges back. | Add a per-step checksum test under one fixture: serialise `state.abundance.sum()` + `state.biomass.sum()` + `state.n_dead.sum()` after every step of an eec_full 1-year run, compare byte-equal vs a saved master snapshot (committed to `tests/_fixtures/eec_full_1yr_step_checksums.npz`). Pre-K-merge, regenerate the snapshot. Post-merge, the test pins per-step trajectory parity for K1 and K3 (K2 explicitly excluded — K2's RNG-stream change will produce a different snapshot, requiring a regenerate inside the K2 PR). |
| Cumulative changes still don't hit the 2 % gate on eec_full | Document the negative result. The gate exists to prevent shipping Pyrrhic perf claims. |
| K-items ship in different orders than recommended → line-number references in this plan are stale by the time the next item is implemented | Each K-item section ends with "Anchor by symbol when re-applying after a sibling K-item merges." The K1/K2/K3 entries name the relevant function symbols, not just line numbers. |
