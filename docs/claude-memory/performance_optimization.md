---
name: performance_optimization
description: Python engine performance optimization — Tiers 1-3 + Phase A/B + Scaling Parity complete, Python FASTER than Java on all benchmarks
type: project
---

Python engine performance optimization — Tiers 1-3 + Phase A/B all completed 2026-03-22.

**Why:** Python engine was ~180x slower than Java (BoB 5yr: 406s vs 2.3s). Root cause was per-operation interpreter overhead in the interleaved mortality loop — 7.67M scalar function calls paying 2-5μs Python dispatch tax each.

**Final Results:**

| Tier | BoB 1yr | BoB 5yr | Cumulative Speedup (5yr) |
|------|---------|---------|--------------------------|
| Pre-optimization | 20.87s | 406s | 1x |
| Tier 1 (cached inst_abd) | 8.84s | — | — |
| Tier 2 (Numba predation) | 2.51s | — | — |
| Tier 3 (full Numba cell loop) | 0.79s | 11.3s | 36x |
| Phase A (batch cell + inline RNG) | ~0.5s | 4.9s | 83x |
| Phase B (prange parallel) | 0.26s | 2.45s | 166x |
| **Scaling Parity (movement + rates)** | **0.24s** | **1.99s** | **204x** |
| Java reference | 0.80s | 2.3s | — |

**EEC (English Channel, 14 species) Results:**

| Config | Python | Java | Result |
|--------|--------|------|--------|
| EEC 1yr | 0.44s | 2.5s | Python 5.7x faster |
| EEC 5yr | 5.2s | 7.2s | Python 1.4x faster |

**Biomass Parity:** EEC 14/14 species within 1 order of magnitude (ratios 0.79-1.18).

**How to apply:**
- Tiers 1-3: See previous entries (cached inst_abd, Numba predation, full Numba cell loop)
- Phase A: `_mortality_all_cells_numba()` — batch Numba function with inline RNG via `np.random.seed/permutation/shuffle`; replaces 250+ per-cell Python→Numba dispatches with one call. Key insight: Python pre-generation loop was itself a bottleneck (6s/11s); moving RNG into Numba eliminated it.
- Phase B: `_mortality_all_cells_parallel()` — `prange(n_cells)` with per-cell deterministic seeding (`rng_seed + cell * 7919`). Inline RNG per cell avoids sequential pre-gen bottleneck.
- `mortality()` has `parallel=True` kwarg to select batch function
- All in `osmose/engine/processes/mortality.py`; pure Python fallback when Numba unavailable
- Statistical parity test: `TestStatisticalParity` in `tests/test_engine_parity.py` (10 seeds, 5% rtol, atol=1.0 for near-zero species)
- Statistical baseline: `tests/baselines/statistical_baseline_bob_1yr_10seeds.npz`
- Baseline script: `scripts/save_parity_baseline.py --statistical --seeds 10`

**Key design decisions:**
- Numba's internal MT19937 RNG used instead of NumPy's PCG64 (statistical equivalence accepted)
- Per-cell seeding in prange: `np.random.seed(rng_seed + cell * 7919)` — deterministic, independent per cell
- Sequential Numba pre-gen loop was a dead-end (14.9s — slower than original due to parallel compilation overhead + buffer allocation)
- `_pre_generate_cell_rng()` still exists but unused by batch path — kept for Python fallback
