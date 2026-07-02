---
name: project-native-predation-kernel-spike
description: 2026-06-24 native C predation-kernel feasibility spike — verdict + the precise trigger for the follow-on integration spike
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

Throwaway C-vs-Numba microbench feasibility spike for the predation leaf
(`_apply_predation_numba`), merged to master 2026-06-24 (`610f517`, --no-ff). Lives at
`scripts/spikes/native_predation/` (NOT wired into engine/suite/CI; build artifacts +
fixture gitignored). Artifact of record: `docs/perf/2026-06-24-native-predation-kernel-spike.md`.
Spec/plan: `docs/superpowers/{specs,plans}/2026-06-24-native-predation-kernel-spike*`.
This RESOLVES the "C/Rust predation kernel" backlog item — do not re-litigate the leaf
question; the open question is now the *integration* spike (below).

## Why it happened
User picked Engine/perf structural levers off the backlog. Re-profile (per the standing
"re-profile before any perf plan" rule) confirmed: eec_full still 2.856s (== v0.12.0 record);
`__slots__` / mutable-SchoolState are CONFIRMED DEAD (P4b/PR#44 already measured the
`__post_init__`-bypass at 1.8%/0.9%, sub-gate — nothing changed). Only real headroom is the
`mortality()` Numba kernel (~35% wall) → spike a native C port of the leaf.

## Verdict: PASS (at the leaf) — but boundary-bound
- **Parity: BIT-EXACT.** Hand-written C (`kernel.c`, cffi, gcc -O3) reproduces the Numba leaf
  to 0.0 rel diff on all 4 cells × 7 mutated arrays. The leaf draws NO RNG (no MT19937 to
  reproduce at this level), so bit-exact parity is free here.
- **Speed: ~10–17× faster on leaf MATH** (robust; the 47× call-weighted headline is
  noise-inflated by a p50 noise-floor artifact — the artifact says so).
- **CRITICAL CAVEAT (necessary-not-sufficient):** the leaf is only ~184 ns, while the per-call
  Python→C ABI boundary is ~2000 ns and Numba's Python-dispatch ~28000 ns. Production calls the
  leaf njit→njit with ZERO boundary. So a per-leaf C call from Python would LOSE. The win
  materializes ONLY if the WHOLE parallel cell-loop is ported to C (amortizing one boundary per
  timestep).

## ▶▶ STAGE 0 (RNG reproduction) — DONE 2026-06-25, PASS. Merged master `dbca0e1`.
Throwaway C reproduction of NumPy-legacy MT19937 (`scripts/spikes/rng_repro/`, artifact
`docs/perf/2026-06-25-rng-repro-feasibility-spike.md`). **C reproduces the cell-loop's per-cell
RNG (seed → 4× permutation → per-school shuffle) BIT-FOR-BIT vs the Numba oracle across the full
grid** (88/88, n 1..100, incl ~2^62 seed), and **C RNG-gen is 1.4–1.6× FASTER than Numba** (so
re-implementing the RNG won't erode a Stage-1 win). **→ bit-exact native cell-loop port is
FEASIBLE; Stage 1 may target the `atol=0`-preserving bar (no re-baseline needed).** KEY facts
proven: Numba `np.random` ≡ CPython legacy `RandomState`; **seed via `init_genrand(seed)` SCALAR
path, NOT `init_by_array`** (in-loop workflow review caught this empirically — init_by_array
diverged 391/440); `causes=[0,1,2,3]` carries over (shuffled in place, not reset); seed masks to
`seed & 0xFFFFFFFF`. Reuse the harness's MT19937 C + parity/bench methodology for Stage 1.

## ▶▶ RESUME TRIGGER (Stage 1 — the real go/no-go; still NOT authorized to build blind)
Stage 0 PASS authorizes only DESIGNING this, NOT building it:
1. Port `_mortality_all_cells_parallel` (the **parallel** prange cell-loop — production default,
   NOT the serial `_mortality_all_cells_numba`) to C — including `_apply_single_cause` (the
   starvation/natural/fishing leaf) + the already-ported predation leaf + the 4-cause interleave.
2. RNG is SOLVED (Stage 0) — drop in the `rng_repro` MT19937 C.
3. Add `prange`→OpenMP parallelism in C.
4. Measure END-TO-END eec_full wall-time vs the Numba baseline. Only a positive result greenlights
   a full port. **Permanent maintenance caveat stands** (a compiled OpenMP extension = a 2nd
   mortality impl tracking the Numba path forever, in CI/wheels/prod, for ~40% of a 2.5s benchmark
   no workload is waiting on). Still gated on a concrete need.

## Reusable facts / gotchas (verified against source this session)
- Production mortality runs `_mortality_all_cells_parallel` (`parallel=True` default,
  mortality.py:1806/1985), a `prange` loop — NOT the serial kernel. Integration target is parallel.
- The leaf is `@njit`, called from inside the njit cell-loop → can't surgically replace it with
  C (objmode kills perf); the native boundary must be the whole cell-loop.
- Capture trick: the leaf's args are ~all the cell-loop kernel's args; monkeypatch the
  module-global `_mortality_all_cells_parallel` (dispatched by name in plain-Python `mortality()`)
  to snapshot pre-state, then reconstruct per-leaf args deterministically.
- Bench methodology that matters: boundary-FREE throughput (njit driver vs C loop, both
  zero-boundary) with identical reset-subtraction, because timing the Numba leaf FROM PYTHON
  pays a ~28µs dispatch tax production never pays. Wide C IQR = full-array reset dwarfs the leaf;
  cell-scoped reset would tighten it (deferred — doesn't change the order-of-magnitude verdict).
- Toolchain: C/cffi + gcc only (Rust NOT installed; would burden CI/prod). cffi dotted module
  name + tmpdir double-nests the .so (compiles but unimportable) → use a non-dotted name.

Related: [[reference-engine-mortality-dispatch]] (two dispatch paths), the closed perf arc
(`docs/perf/2026-05-08-perf-arc-overview.md`).
