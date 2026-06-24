# Native predation-kernel feasibility spike — design

> Status: design (awaiting review) · 2026-06-24 · author: perf arc continuation
> Decision context: MEMORY perf surface "effectively closed" since v0.12.0; this
> spike re-opens it deliberately to test the one structural lever with real
> headroom — a native (C) rewrite of the Numba predation kernel.

## 1. Why this spike exists

A fresh profile on current master (`eec_full` 5yr, **2.856s** median wall — statistically
identical to the v0.12.0 record of 2.881s) ranks the engine's hot surfaces:

| Surface | ~% wall | Lever | Notes |
|---|---:|---|---|
| `mortality()` Numba kernel | ~35% | native rewrite / GPU | the only large headroom |
| `movement()` Python body | ~11% | vectorise | not pursued here |
| `aggregate_diet_by_species` (`np.add.at`) | ~6% | `bincount` swap | **output-gated**, skipped in calibration |
| `SchoolState` `__slots__` / mutable | ~1.8% (true) | — | **confirmed dead**: PR #44 (P4b) measured the `__post_init__`-bypass at 1.8% eec / 0.9% baltic, below the 2% gate |

The two structural levers originally on the backlog (`__slots__`, mutable `SchoolState`)
are confirmed sub-gate and are **not** pursued. The only lever with meaningful
headroom — and the only one that speeds **calibration** (where per-run cost compounds
over thousands of sims) — is the predation/mortality kernel.

**But the kernel is already native.** `_apply_predation_numba` and
`_mortality_all_cells_numba` are `@njit` — Numba hands them to LLVM at roughly `-O3`.
A hand-written C kernel only wins if it does something Numba's codegen cannot:
better SIMD, better memory layout, fewer bounds checks, or absorbing orchestration
glue. "C beats Python" does **not** apply — Python is not running the hot loop.
So the speedup is **not guaranteed and may be small.**

This spike answers the single **disqualifying** question as cheaply as possible —
*can hand-written C beat Numba's codegen on this kernel's math at all?* — before
any multi-week commitment to a full port.

## 2. Scope decisions (locked)

- **Language/toolchain: C compiled via `cffi`.** `cffi` 2.0 and `gcc` 13.3 are already
  present in the environment; **Rust is not installed** and would add a heavyweight
  toolchain to dev + CI + the prod clone. C/cffi adds zero new system dependencies.
- **Target function: `_apply_predation_numba` (the leaf), microbench only.** It is the
  named backlog target and the innermost predation compute. It **draws no RNG** (the
  caller `_mortality_all_cells_numba` owns the school-order permutations), so at this
  boundary parity is purely float-op-order — no MT19937 to reproduce. That makes it the
  cleanest possible parity test and the cheapest disqualifier.
- **NOT in scope:** integration, the cell-loop orchestration, the build wiring, CI,
  wheels, prod deployment, RNG-stream reproduction. Those belong to later sub-projects
  and are only reached if this spike passes.

### Parity bar
Deferred to the spike per the brainstorming decision ("let the spike decide"). The spike
**measures** the C-vs-Numba output difference on identical inputs and reports it; it does
not commit to bit-exact vs within-1-OoM. Because there is no RNG at this boundary, we
expect the difference to be at or near float-rounding (op-order) level, which also tells
us how nearly "free" bit-parity would be for the eventual port.

## 3. Architectural constraint discovered (informs the verdict, not the spike build)

`_apply_predation_numba` is called from **inside** the `@njit` function
`_mortality_all_cells_numba`. A C extension cannot be called from inside `njit` without
`objmode`, which destroys performance. Therefore a *real* port's native boundary must be
the **entire cell-loop**, not this leaf function. The leaf microbench cannot be wired
into the engine as-is — and is not meant to be. This is why the spike is explicitly a
throwaway head-to-head, and why a passing result unlocks a **second integration spike**
(porting the cell-loop) rather than greenlighting the full port directly.

## 4. Method

The spike is a standalone harness under `scripts/spikes/` (throwaway; not imported by the
engine, not covered by the suite). Steps:

1. **Capture** — temporarily instrument `_apply_predation_numba`'s caller during one real
   `eec_full` run to serialise the full argument set for a representative, non-trivial
   cell (many predators, schools + resources present) to a `.npz` fixture. Pick a cell
   with `n_local` in the busy range so the microbench reflects realistic work, not an
   empty/early-return path. Capture the resulting output arrays too, as the parity oracle.
2. **Numba baseline** — load the fixture; warm the JIT cache (one untimed call); time
   `_apply_predation_numba` over many iterations (enough for a stable median; reset the
   mutated output buffers — `n_dead`, `preyed_biomass`, `diet_matrix`, `tl_weighted_sum`,
   `egg_retained` — to their captured pre-call state before each timed iteration).
3. **C port** — reimplement the identical algorithm in C, compiled via `cffi`
   (`ffi.set_source` + `ffi.cdef`, `gcc -O3 -march=native`). Arrays passed as typed
   pointers + shapes. Two-phase prey scan preserved (the K1 scratch-buffer pattern). Same
   reset-buffers-per-iteration timing protocol.
4. **Parity** — run C once on the fixture; compare every mutated output against the Numba
   oracle. Report max abs and max rel difference per output array.
5. **Verdict** — compute the C-vs-Numba median-time ratio.

### Disqualifying gate
- **If C is not ≥ ~1.3× faster** on the leaf math → **STOP.** The integration cost (native
  cell-loop), the second code path to maintain, and the build/CI/prod-deploy burden of the
  full port cannot be justified by a smaller leaf-level margin, especially with **no
  workload currently demanding the speed** (no calibration-time driver exists; this was a
  "re-profile then decide" exploration, not a complaint about a slow run).
- **If C ≥ ~1.3×** → recommend a follow-on **integration spike** that ports the whole
  cell-loop and measures end-to-end `eec_full` wall-time, since the leaf microbench is
  necessary but not sufficient for an end-to-end win.

## 5. Deliverable

A `docs/perf/2026-06-24-native-predation-kernel-spike.md` artifact-of-record, in the style
of the K-arc not-shipping write-ups, containing:
- the captured-cell characteristics (`n_local`, predator count, resources present),
- Numba vs C median times + ratio,
- per-output max abs/rel parity differences,
- a go / no-go recommendation with reasoning.

The spike code (`scripts/spikes/native_predation/`) is throwaway and may be deleted after
the write-up, or kept under `scripts/spikes/` as a reproducible harness. It is **not**
wired into the engine and **not** added to the test suite or CI.

## 6. Risks & honesty notes

- **Numba is already LLVM-optimised** — a negative result is a real and acceptable outcome;
  the spike is designed to fail cheaply.
- **Microbench ≠ end-to-end** — even a positive leaf result does not prove a production win;
  the integration spike is the real gate before committing.
- **`-march=native` caveat** — fine for a local feasibility measurement, but a production
  build could not assume it; note this so the spike ratio isn't over-read as a portable
  number.
- **No new dependency** — `cffi`/`gcc` only; if the eventual port chose a setuptools C
  extension or Cython instead, that is a later sub-project decision and does not affect the
  spike's validity.

## 7. Out of scope / explicitly deferred

- Full kernel port, native cell-loop, Numba fallback path.
- RNG-stream (MT19937 + NumPy Fisher-Yates) reproduction.
- Build wiring, CI compilation, wheels, prod-clone deployment.
- `movement()` and `aggregate_diet_by_species` wins (separate, lower-effort levers).
- GPU/CUDA.
