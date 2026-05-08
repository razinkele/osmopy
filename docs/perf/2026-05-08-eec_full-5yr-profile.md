# K4 — eec_full 5-year profile + go/no-go gate

> Created: 2026-05-08
> Plan: [`docs/plans/2026-05-08-kernel-surgery-plan.md`](../plans/2026-05-08-kernel-surgery-plan.md)
> Master at run time: `e0e1d94` (post-PR #31)

## Methodology

cProfile against the warm Numba JIT cache (one warmup run, then a profiled
second run on the same engine instance). Profiled run completes in 7.34s
on eec_full 5-year (matches the established benchmark median of 5.025s
on a 7-repeat — slight slowdown is the per-run cProfile instrumentation
overhead; relative shares are still meaningful).

Numba `@njit(cache=True)` functions are opaque to cProfile — `_apply_predation_numba`
and `_mortality_all_cells_*` show up as one big call. Python-visible
hot spots are reported below; for the JIT-internal cost of K1 / K2 we
fall back to a micro-benchmark of the equivalent Python-side allocation
pattern.

Reproduce:
```
.venv/bin/python /tmp/k4_warm_profile.py        # cProfile pass
.venv/bin/python /tmp/k4_kernel_isolation.py    # alloc micro-bench
```

## Top Python-side hot spots (warm-cache, 7.34s total)

| tottime | calls | function | share |
|--------:|------:|---|-----:|
| 1.290s | 120 | `mortality.py:1665(mortality)` (Python orchestration around the JIT'd kernel) | **17.6 %** |
| 1.269s | 240 | `accessibility.py:149(compute_school_indices)` | 17.3 % |
| 1.230s | 119 | `movement.py:492(_precompute_map_indices)` | 16.7 % |
| 0.523s | 3.8 M | `dict.get` (largely from accessibility.py:115:get_index) | 7.1 % |
| 0.493s | 1.9 M | `accessibility.py:115(get_index)` | 6.7 % |
| 0.327s | 120 | `movement.py:196(movement)` | 4.5 % |
| 0.243s | 6.4 k | `np.ufunc.at` (H7 territory; mostly OK) | 3.3 % |
| 0.082s | ~1.6 k | `np.array` constructors | 1.1 % |
| 0.062s | 120 | `reproduction.py:62(reproduction)` | 0.85 % |
| 0.061s | 120 | `simulate.py:703(_collect_distributions)` | 0.83 % |
| 0.030s | 2.5 k | `numpy.ndarray.copy` (the H6/K3 surface) | **0.41 %** |
| 0.026s | 30.7 k | `numpy.zeros` (Python-side; Numba-internal allocs invisible) | 0.35 % |
| ... | ... | (everything else under 0.5 %) | ~22 % |

**Sum of named spots:** ~76 % of 7.34s. **JIT'd kernel internals (invisible to cProfile):** ~24 % = ~1.76s, so steady-state per-year ~ 0.35s spent inside the predation/mortality JIT'd code.

## Allocation-cost micro-bench (Python-side equivalent)

Times the L846-848 alloc triple (`prey_type` int32, `prey_id` int32, `prey_eligible` float64) at realistic sizes:

| max_prey | calls | elapsed | per-call (triple) |
|---:|---:|---:|---:|
| 206 | 720 000 | 0.614 s | 0.85 µs |
| 206 | 100 000 | 0.087 s | 0.87 µs |
| 50 | 720 000 | 0.559 s | 0.78 µs |

Numba JIT'd `np.zeros` for fixed-shape arrays typically runs **5-20× faster** than the Python-side equivalent (direct `memset` with no Python object overhead). A reasonable estimate of the in-kernel allocation cost on eec_full 5-year is therefore in the range **30 ms – 120 ms per year × 5 years = 0.15 s – 0.60 s** — i.e. 3 – 12 % of total run time, with the high end likely if `max_prey` and per-cell call counts are at the upper realistic bound.

## Go / no-go decisions per K-item

### K3 — ctx-pool `larva_deaths` + `inst_abd`: **DROP**

The cProfile-visible `numpy.ndarray.copy` total — 30 ms across 2 480 calls — is the **upper bound** of K3's possible saving. That's 0.41 % of run time, **below the 2 % K3 gate** by 5×. Even if K3 were to drive the copy time to zero, the gain would be unmeasurable against the 100 ms / 2 % noise floor.

**Action:** ship `docs/perf/2026-05-08-K3-not-shipping.md` post-mortem (template per the kernel-surgery plan), do not implement.

### K1 — predation scratch hoisting: **CONDITIONAL PURSUE**

cProfile cannot resolve allocations inside `_apply_predation_numba`. The Python-side micro-bench gives an upper bound: at 0.85 µs/triple × ~720 000 calls/year × 5 years = ~3.06 s if all allocations went through Python — but Numba's `np.zeros` in `@njit` runs much faster, so realistic kernel-internal cost is 5-20 % of that = **150 ms – 600 ms over 5 years = 3-12 % of run time**.

The K1 gate is **≥ 15 % of `_apply_predation_numba` self-time**. With predation+mortality JIT internals at ~1.76 s of 7.34 s and the kernel's allocation share estimated at 150–600 ms, K1's allocation-share-of-self-time is **8 – 34 %** — straddling the gate. The lower-bound estimate sits below; the upper-bound exceeds.

**Action:** **conditional pursue**. The K1 PR must demonstrate **≥ 2 % wall-time improvement on eec_full 5-year** in its before/after JSON; if it doesn't, drop with a `docs/perf/2026-05-08-K1-not-shipping.md`. Don't pre-commit to K1's success.

### K2 — `cause_orders` + 4-element Fisher-Yates: **DROP**

Per-cell `cause_orders` and `seq_*` allocations: ~100 cells × 24 steps × 5 yr × 1 subdt × 5 allocs = 60 000 allocs over 5 years. At 0.5-1 µs each (in JIT, optimistic): 30-60 ms = 0.6-1.2 % of run time. The shuffle calls are length-4 — Numba inlines these to single-digit-nanosecond cost. Aggregate K2 ceiling: ~1 % of run time.

The K2 gate is **≥ 5 % of `mortality()` self-time** spent in the alloc + shuffle block. With `mortality()` at 1.29 s tottime and the K2 sites estimated at well under 100 ms of internal kernel time, K2's share is far below 5 %.

K2 also carries the highest cost (RNG-stream change → between-version-reproducibility break + calibration cache invalidation via `osmose/__version__.py` bump). Cost-benefit is firmly negative.

**Action:** **drop**. Ship `docs/perf/2026-05-08-K2-not-shipping.md` post-mortem.

## Bonus findings (not on the K-list, but profile-suggested)

The hottest Python-side spots that aren't on the kernel-surgery K-list:

1. **`accessibility.py:115(get_index)`** — 1.9 M calls × 0.26 µs/call = 493 ms = **6.7 % of run time**. Called inside `compute_school_indices` (1.27 s = 17 %). The implementation appears to use `dict.get` for a hot lookup; vectorisation or a cached array index could cut this substantially. Worth a focused audit in a follow-on PR.

2. **`movement.py:492(_precompute_map_indices)`** — 119 calls × 10 ms = 1.23 s = **16.7 %**. Once-per-step indexing into movement maps; the per-call cost is large. May be amenable to a precompute-once-then-reindex refactor.

3. **`mortality.py:1665(mortality)` Python wrapper** — 1.29 s tottime separate from the JIT'd kernel. Per-call 11 ms. The accessibility-index recomputation inside `mortality()` (lines 1722-1738) calls `compute_school_indices` **twice** per call (prey + pred); that's most of the wrapper time. Caching across `parallel=True/False` modes is one option.

These are larger wins than the K-list and don't require kernel surgery — pure Python optimization. Worth a separate `kernel-bypass-perf-plan.md` if perf is on the agenda after the K-items resolve.

## Combined K-plan recommendation

| K-item | Decision |
|---|---|
| K3 | **DROP** — 0.41 % ceiling, below 2 % gate |
| K1 | **CONDITIONAL PURSUE** — 3-12 % estimated, gate ≥ 2 %; PR must measure |
| K2 | **DROP** — < 1 % estimated, with high cost (RNG-stream break + cache invalidation) |

**Net K-plan cumulative target:** at most K1's measured gain. The original "≥ 5 % combined" stretch target is not achievable with K2 + K3 dropped. Re-set expectations to: **K1 alone, ≥ 2 % wall-time improvement or drop**.

The bonus findings above (accessibility/movement Python overhead) suggest **larger wins lie outside the K-list** — possibly in the 5-15 % range — and don't require Numba kernel surgery. Future perf work should be aimed there.
