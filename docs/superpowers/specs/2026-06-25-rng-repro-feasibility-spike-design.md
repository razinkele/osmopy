# Stage-0 RNG-reproduction feasibility spike — design

> Status: design (awaiting review) · 2026-06-25
> Parent: the native predation-kernel integration spike (Stage 1). This is Stage 0 —
> the cheapest disqualifier for the bit-exact path, run BEFORE committing to the port.
> Predecessor: `docs/superpowers/specs/2026-06-24-native-predation-kernel-spike-design.md`
> (leaf spike: C predation leaf is bit-exact + ~10–17× faster, but boundary-bound →
> the win needs the whole parallel cell-loop ported, which requires reproducing the RNG).

## 1. Why this spike exists

The leaf spike proved a C port of `_apply_predation_numba` is bit-exact with Numba and
faster on leaf math, but the per-call Python→C boundary (~2 µs) dwarfs the ~184 ns leaf, so
the win only materialises if the **entire parallel mortality cell-loop**
(`_mortality_all_cells_parallel`, mortality.py:1411) is ported to C — amortising one
boundary per timestep. That cell-loop does three things the leaf does not:

1. draws per-cell randomness — `np.random.seed(rng_seed + cell*7919)`, then
   `seq_pred/starv/fish/nat = np.random.permutation(n_local)` (×4) and a per-school
   `np.random.shuffle(causes)` building `cause_orders` (mortality.py:1479–1497);
2. interleaves four mortality causes per school — predation calls the (already-ported,
   bit-exact) leaf; starvation/natural/fishing call `_apply_single_cause` (a small njit fn);
3. runs as a `prange` parallel loop.

For a **bit-exact** integration (one that keeps the 14/14 EEC `atol=0` + 8/8 BoB
Java-parity tests passing unchanged), the C cell-loop must reproduce that per-cell RNG
sequence **bit-for-bit**. That is the single disqualifying unknown for the whole bit-exact
path — and it is cheap to settle. This Stage-0 spike settles it before any cell-loop work.

## 2. Premise already established (Component 0a — done during design exploration)

A pure-Python probe compared the kernel's exact RNG usage under `@njit` against CPython's
legacy `numpy.random.RandomState` (MT19937), across seeds `{0, 7919, 12345, 2^40+7919*3}`
and `n ∈ {1,2,4,12,24,33}`:

> **Result: Numba `np.random` ≡ CPython legacy `RandomState` (MT19937), bit-identical on
> every tested (seed, n)** — including the large-int64 per-cell seed form, which masks to
> uint32 (matched `RandomState(seed % 2**32)`).

Therefore the C reproduction target is the **fully-documented NumPy-legacy MT19937**
algorithm — not a Numba-specific one. This removes the scary unknown; what remains is an
implementation + speed question, which Component 0b answers in C.

## 3. Scope (locked)

- **Language/toolchain:** C via `cffi` + `gcc -O3`, reusing the leaf-spike infra under
  `scripts/spikes/`. No Rust (not installed; would burden CI/prod). No new dependency.
- **Target: the per-cell RNG only.** Reproduce exactly what the cell-loop draws per cell:
  `seed` → 4× `permutation(n_local)` → per-school `shuffle` of `[0,1,2,3]`. NOT in scope:
  `_apply_single_cause`, the leaf integration, the prange/OpenMP loop, any cell-loop port,
  any engine wiring. Those belong to Stage 1 and are only reached if this passes.
- **Parity bar (this spike): bit-exact.** The whole point is to test whether bit-exact is
  achievable; there is no "close enough" — sequences match to the integer or they do not.

## 4. Method

Throwaway harness under `scripts/spikes/rng_repro/` (not imported by `osmose/`, not in CI;
only its own manual tests).

### 4.0 Provenance guard (run first)
Assert `osmose.engine.processes.mortality.__file__` resolves under this worktree and
`_HAS_NUMBA is True` (same trap-guards as the leaf spike; run with `PYTHONPATH=.`).

### 4.1 Oracle — capture what production actually draws
An `@njit` function mirroring mortality.py:1479–1497 verbatim:
`np.random.seed(s)`; `a=permutation(n); b=permutation(n); c=permutation(n); d=permutation(n)`;
then per `i in range(n)`: `np.random.shuffle(causes)` over `causes=[0,1,2,3]`, recording
`cause_orders[i]`. Returns `(a,b,c,d, cause_orders)` as int32 arrays for a given `(s, n)`.
This is ground truth — the exact draws the parity baselines encode.

### 4.2 C reproduction (the feasibility artifact)
Implement NumPy-legacy MT19937 in C (`mt19937.c`, compiled via cffi), faithfully:
- **Seeding:** NumPy-legacy **scalar** seeding `init_genrand(seed)` — the single-integer path
  that `RandomState(scalar)` and Numba `np.random.seed(scalar)` use (NOT `init_by_array`;
  verified empirically during plan review — `init_by_array` produces a different stream). The
  int64→uint32 reduction confirmed in 0a applies first: the per-cell seed `rng_seed + cell*7919`
  masked to `seed & 0xFFFFFFFF`. Match `RandomState(seed)` exactly.
- **Bounded integer:** NumPy-legacy `rk_interval(max)` — smallest mask `2^k − 1 ≥ max`,
  draw `genrand_uint32() & mask` rejecting until `≤ max`. (NOT Lemire — legacy uses masked
  rejection.)
- **`permutation(n)` / `shuffle`:** NumPy-legacy Fisher-Yates in NumPy's exact loop
  direction (`for i = n−1 down to 1: j = rk_interval(i); swap(arr[i], arr[j])`), operating
  on `arange(n)` for permutation and on the live array for shuffle.
Expose a C function that, given `(s, n)`, produces the same `(a,b,c,d, cause_orders)` the
oracle does, marshalled via the leaf-spike's cffi pattern (contiguous int32 out-buffers).

### 4.3 Parity gate (hard, pre-registered)
C output must be **bit-identical** to the oracle (`np.array_equal`, every array) across:
- `n_local ∈ {1, 2, 4, p10, p50, p95, max}` where p10/p50/p95/max come from the real
  `eec_full` cell `n_local` histogram (reuse the leaf-spike capture or recompute the
  boundaries histogram);
- seeds `∈ {0, 1, 7919, 12345}` **and** the real per-cell form `rng_seed + cell*7919`
  (`np.int64`) for several `(rng_seed, cell)` pairs spanning small **and a near-2^63
  `rng_seed`** — production draws `rng_seed = int(rng.integers(0, 2**63))`, so
  `rng_seed + cell*7919` can wrap int64; the C must reproduce whatever Numba's
  `np.random.seed` does with that wrapped/large value (0a showed plain large seeds reduce as
  `seed & 0xFFFFFFFF`; the wrap edge is explicitly tested here, not assumed).
**Any** mismatch fails. On failure, root-cause which draw first diverges (permutation #k at
element #m, or shuffle at school #i) and classify: implementation bug (mask, loop direction,
seeding word order) → iterate; or a fundamental Numba-vs-NumPy gap → STOP/β-signal.

### 4.4 Speed probe
Boundary-free per-cell RNG-generation time, C vs the Numba njit oracle, at representative
`n_local` (p50, p95): loop the full per-cell draw (`seed`+4×perm+shuffle) N times inside
native code on each side, report median ns/cell each, and the **C-RNG fraction of the
per-cell mortality cost** (using the leaf spike's measured per-leaf time × predator count as
the cell-mortality proxy). This tells Stage 1 whether re-implementing the RNG in C erodes
the integration win or is negligible.

## 5. Gate / verdict (pre-registered)

- **PASS** — bit-identical across the entire grid AND C-RNG-gen is not a dominant fraction
  of per-cell cost → **bit-exact (α) is FEASIBLE**; Stage 1 (cell-loop port) is justified at
  the `atol=0`-preserving bar. A PASS authorises designing Stage 1, not building it blind.
- **STOP / β-signal** — any irreducible divergence (a real Numba-vs-NumPy-legacy gap) → the
  bit-exact path is blocked; Stage 1 would require the within-1-OoM + re-baseline bar (a
  separate decision), or the integration is dropped.
- Either outcome is a real, cheap answer — the spike is designed to fail cheaply.

## 6. Deliverable

`docs/perf/2026-06-25-rng-repro-feasibility-spike.md`, artifact-of-record in the leaf-spike
style: the 0a premise; the parity-grid result (every (n, seed) → match/diverge); on any
diverge, the first-diverging draw + classification; the C-vs-Numba RNG-gen speed + the
C-RNG fraction-of-cell-cost; and a go/no-go for Stage 1 stating explicitly that PASS
authorises only *designing* Stage 1.

The harness stays under `scripts/spikes/rng_repro/` (reproducible). Not wired into `osmose/`,
the engine, the suite, or CI.

## 7. Risks & honesty notes

- **0a already de-risked the target** — the C target is documented NumPy-legacy MT19937, so a
  PASS is the likely outcome; the residual risk is an implementation off-by-one (mask, loop
  direction, seed-word order), which the parity gate catches and localises.
- **Necessary-not-sufficient (again).** A PASS proves only that the RNG is reproducible — it
  does NOT prove the full integration wins end-to-end (that needs Stage 1: `_apply_single_cause`
  + the leaf + the prange/OpenMP loop + an end-to-end `eec_full` measurement at the parity bar).
  The standing caveat from the leaf spike holds: the end state is a permanent second parallel-C
  mortality implementation + a compiled OpenMP extension in CI/wheels/prod, for ~40% of a 2.5 s
  benchmark no current workload is waiting on. Stage 1 remains its own go/no-go.
- **`-march=native`** is fine for a local feasibility measurement but a portable build cannot
  assume it; report the speed with plain `-O3`.

## 8. Out of scope / explicitly deferred (Stage 1)

- `_apply_single_cause` port; the predation-leaf integration; the prange→OpenMP cell-loop;
  engine wiring behind a flag; end-to-end `eec_full` wall-time; the within-1-OoM/re-baseline
  parity-bar decision; build/CI/wheel/prod-deploy of a parallel compiled extension.
