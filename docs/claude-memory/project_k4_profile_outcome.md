---
name: K4 kernel-surgery profile outcome
description: 2026-05-08 — K4 profile gate decisions for the kernel-surgery plan: K3/K2 dropped, K1 conditional, plus bonus non-K-list perf findings
type: project
originSessionId: d2b7f4a5-d107-4042-a473-f491e81f4df1
---
K4 (eec_full 5yr cProfile + alloc micro-bench) shipped 2026-05-08 as PR #33.
Master at run time was post-#31 (`e0e1d94`); profiled run 7.34 s warm-cache.

**Per-K decisions (from `docs/perf/2026-05-08-eec_full-5yr-profile.md`):**
- **K3 = DROP.** `numpy.ndarray.copy` ceiling 0.41 % of run time, below half the 2 % gate. Post-mortem at `docs/perf/2026-05-08-K3-not-shipping.md`.
- **K2 = DROP.** Estimated ≤ 1 % saving + high cost (RNG-stream divergence, `__version__` bump invalidating calibration cache, reproducibility-test churn). Post-mortem at `docs/perf/2026-05-08-K2-not-shipping.md`.
- **K1 = CONDITIONAL PURSUE.** Estimated 3-12 % kernel-internal allocation cost — straddles the 2 % gate. PR must measure ≥ 2 % wall-time on eec_full 5 yr or drop with a not-shipping post-mortem.

**Bonus findings (NOT on the K-list, larger wins):**
1. `accessibility.py:115:get_index` — 1.9 M calls × 0.26 µs = 493 ms = **6.7 %** of total run time. Called inside `compute_school_indices` (1.27 s = 17 %). `dict.get` on a hot path; cached array index could cut substantially.
2. `movement.py:492:_precompute_map_indices` — 119 calls × 10 ms = 1.23 s = **16.7 %** of total run time. Once-per-step indexing; per-call cost large; possible precompute-once-then-reindex refactor.
3. `mortality.py:1665:mortality` Python wrapper — 1.29 s tottime (separate from JIT'd kernel). Per-call 11 ms. The accessibility-index recomputation inside `mortality()` (lines 1722-1738) calls `compute_school_indices` **twice** per call (prey + pred); caching across `parallel=True/False` modes is one option.

**Why:** the K-list assumed Numba-internal allocations were the largest perf surface; cProfile shows the largest wins are in the Python orchestration layer above the JIT'd kernel.

**How to apply:** if pursuing further perf work after K1 resolves, open a *new* plan targeting these three bonus findings — they're likely to deliver 5-15 % cumulatively without kernel surgery. Don't stretch the K-plan to cover them.
