# RNG Reproduction Feasibility Spike — 2026-06-25

**Verdict: PASS**

---

## 0. Premise (spec §0a)

The cell-loop kernel in `osmose/engine/processes/mortality.py` (production Numba path,
`_mortality_all_cells_numba`) drives per-cell randomness via `np.random.seed(seed)` and
`np.random.permutation` / `np.random.shuffle` inside `@njit`-compiled code.  Numba's
`np.random` in JIT context is the **NumPy legacy MT19937 (Mersenne Twister)**, seeded by a
scalar `uint32` seed via the `init_genrand` path — identical to `numpy.random.RandomState`
seeded the same way.  The spike confirms this by checking that the `@njit` oracle and the
CPython `RandomState` reference produce bit-identical output across the full parity grid.

The C port (`mt19937.c`) targets the same algorithm: `init_genrand(seed & 0xFFFFFFFF)`
(scalar seed, NOT `init_by_array`) followed by the same draw sequence (4× `genrand_perm(n)` +
`n`× `genrand_shuffle(causes)`).

---

## 1. Parity grid

Portable build (`-O3`, no `-march=native`).  Every `(seed, n)` pair must be bit-identical
across all 5 output arrays (`seq_pred`, `seq_starv`, `seq_fish`, `seq_nat`, `cause_orders`).

Grid: `GRID_N = [1, 2, 3, 4, 5, 8, 12, 24, 33, 50, 100]`
Seeds (count 8): `[0, 1, 7919, 12345, 0, 7919, 987693916, 4611686018435240059]`

| seed | n | result |
|---|---|---|
|                      0 |    1 | match |
|                      0 |    2 | match |
|                      0 |    3 | match |
|                      0 |    4 | match |
|                      0 |    5 | match |
|                      0 |    8 | match |
|                      0 |   12 | match |
|                      0 |   24 | match |
|                      0 |   33 | match |
|                      0 |   50 | match |
|                      0 |  100 | match |
|                      1 |    1 | match |
|                      1 |    2 | match |
|                      1 |    3 | match |
|                      1 |    4 | match |
|                      1 |    5 | match |
|                      1 |    8 | match |
|                      1 |   12 | match |
|                      1 |   24 | match |
|                      1 |   33 | match |
|                      1 |   50 | match |
|                      1 |  100 | match |
|                   7919 |    1 | match |
|                   7919 |    2 | match |
|                   7919 |    3 | match |
|                   7919 |    4 | match |
|                   7919 |    5 | match |
|                   7919 |    8 | match |
|                   7919 |   12 | match |
|                   7919 |   24 | match |
|                   7919 |   33 | match |
|                   7919 |   50 | match |
|                   7919 |  100 | match |
|                  12345 |    1 | match |
|                  12345 |    2 | match |
|                  12345 |    3 | match |
|                  12345 |    4 | match |
|                  12345 |    5 | match |
|                  12345 |    8 | match |
|                  12345 |   12 | match |
|                  12345 |   24 | match |
|                  12345 |   33 | match |
|                  12345 |   50 | match |
|                  12345 |  100 | match |
|                      0 |    1 | match |
|                      0 |    2 | match |
|                      0 |    3 | match |
|                      0 |    4 | match |
|                      0 |    5 | match |
|                      0 |    8 | match |
|                      0 |   12 | match |
|                      0 |   24 | match |
|                      0 |   33 | match |
|                      0 |   50 | match |
|                      0 |  100 | match |
|                   7919 |    1 | match |
|                   7919 |    2 | match |
|                   7919 |    3 | match |
|                   7919 |    4 | match |
|                   7919 |    5 | match |
|                   7919 |    8 | match |
|                   7919 |   12 | match |
|                   7919 |   24 | match |
|                   7919 |   33 | match |
|                   7919 |   50 | match |
|                   7919 |  100 | match |
|              987693916 |    1 | match |
|              987693916 |    2 | match |
|              987693916 |    3 | match |
|              987693916 |    4 | match |
|              987693916 |    5 | match |
|              987693916 |    8 | match |
|              987693916 |   12 | match |
|              987693916 |   24 | match |
|              987693916 |   33 | match |
|              987693916 |   50 | match |
|              987693916 |  100 | match |
|    4611686018435240059 |    1 | match |
|    4611686018435240059 |    2 | match |
|    4611686018435240059 |    3 | match |
|    4611686018435240059 |    4 | match |
|    4611686018435240059 |    5 | match |
|    4611686018435240059 |    8 | match |
|    4611686018435240059 |   12 | match |
|    4611686018435240059 |   24 | match |
|    4611686018435240059 |   33 | match |
|    4611686018435240059 |   50 | match |
|    4611686018435240059 |  100 | match |

**Summary:** 88/88 combos bit-identical.

---

## 2. C-vs-Numba RNG-gen speed (portable build)

`n_iter=200`, `n_samples=30`, interleaved A/B to cancel drift.
Metric: median ± IQR (ns per iteration).

| n  | Numba median (ns) | Numba IQR (ns) | C median (ns) | C IQR (ns) | ratio Numba/C |
|----|-------------------|----------------|---------------|------------|---------------|
| 12 | 2163 | 51 | 1584 | 48 | 1.37x |
| 33 | 3850 | 79 | 2335 | 83 | 1.65x |

Note: `-march=native` is NOT the feasibility gate.  The portable build ratio already answers
whether a C RNG-gen call is a meaningful fraction of per-cell cost.

---

## 3. Verdict

**PASS**

Bit-exact RNG reproduction is FEASIBLE.
The C implementation (`mt19937.c`) reproduces the Numba MT19937 stream bit-for-bit across the full GRID_N × GRID_SEEDS grid (portable build).

### What a PASS authorises

A PASS authorises **designing** Stage 1 — the cell-loop port: porting `_mortality_all_cells_parallel`, `_apply_single_cause`, and the leaf integration, adding `prange`/OpenMP parallelism, and then **measuring** end-to-end `eec_full` wall-time before deciding whether to ship.

A PASS does **not** authorise building Stage 1 blind.  The full design review must address:

- **Maintenance caveat (permanent):** a compiled OpenMP C extension is a second implementation
  of the mortality kernel that must be kept in sync with every future change to the Numba
  production path.  It adds CI/wheel/packaging complexity for what is currently ~40% of a 2.5 s
  benchmark with no workload actively blocked on it.
- **RNG-gen cost share:** if the C RNG-gen speed ratio (Numba/C ≈ 1.4x at n=12,
  1.6x at n=33) is a large fraction of per-cell cost, it would erode the
  integration win even if the loop + predation kernel itself is fast.  This must be quantified
  in the Stage 1 design before committing to the implementation.
- **Stage 1 scope:** port `_apply_single_cause` + leaf integration, wire `prange`/OpenMP,
  gate behind an env-var or config flag, and measure end-to-end on `eec_full` before any
  merge decision.

---

*Spike committed: `scripts/spikes/rng_repro/`*
*Harness retained for reproducibility.*
*Build artifacts (`.so`, `.o`, `_rng_*.c`) not tracked in git.*
