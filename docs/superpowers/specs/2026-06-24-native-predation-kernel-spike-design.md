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
  named backlog target and the innermost predation compute. It **draws no RNG** — verified:
  no `np.random`/`permutation`/`shuffle` in its body (mortality.py:881-1053) and no
  rng/seed argument in its signature (mortality.py:828-870); the school-order permutations
  live in the *callers* (mortality.py:1292, 1304-1315). So at this boundary parity is
  purely float-op-order — no MT19937 to reproduce. That makes it the cleanest possible
  parity test and the cheapest disqualifier.
- **NOT in scope:** integration, the cell-loop orchestration, the build wiring, CI,
  wheels, prod deployment, RNG-stream reproduction. Those belong to later sub-projects
  and are only reached if this spike passes.

### Output arrays this function mutates (the parity oracle set)
Verified against source — `_apply_predation_numba` writes into **all** of:
`inst_abd` (mortality.py:1021, the prey entry `inst_abd[q_idx] -= n_dead_prey`),
`rsc_biomass` (1035), `n_dead` (1020),
`preyed_biomass` (1053), `pred_success_rate` (1052), `diet_matrix` (1032/1048, gated on
`diet_enabled`), `tl_weighted_sum` (1027/1043, gated on `tl_tracking`). Note
`egg_retained` is **read-only** here (1913 is a read) and is therefore **not** an output —
an earlier draft wrongly listed it. The parity oracle must diff every array in the
mutated set above, not a subset.

### Parity bar
Deferred to the spike per the brainstorming decision ("let the spike decide") **for the
eventual port's policy**, but the spike itself uses a **pre-declared correctness bar** so a
"pass" is falsifiable: because there is no RNG at this boundary, the C and Numba results
should agree to f64 op-order rounding — **≤ 1e-12 max relative difference** on every
mutated array, *provided reduction order matches* (the prey-scan sums at mortality.py:905
must be summed in the same order in C). Anything materially larger is a **correctness
failure requiring root-cause**, not a tunable — it means the algorithm diverged, not that
"C is close enough." The measured difference also tells us how nearly free bit-parity
would be for the eventual port.

## 3. Architectural constraint (informs the verdict, not the spike build)

`_apply_predation_numba` is called from **inside** `@njit` cell-loop kernels — and
**production runs the *parallel* one.** Verified: `mortality()` defaults `parallel=True`
(mortality.py:1806) and the production call site (`_mortality` → `mortality(...)`,
simulate.py:203-205) does not override it, so dispatch (mortality.py:1985) selects
`_mortality_all_cells_parallel` (`@njit(cache=True, parallel=True)`, a `prange` loop,
mortality.py:1411), **not** the serial `_mortality_all_cells_numba`. The leaf is invoked
inside that njit `prange` body (call site mortality.py:1511; the serial kernel calls it at
1330 too). A C extension cannot be called from inside `njit` without `objmode` (which
destroys performance). Therefore a *real* port's native boundary must be the **entire
cell-loop**, and that loop is **parallel** — so the integration spike must also reproduce
the prange parallelism (OpenMP or equivalent), which is harder than a serial port. The
leaf microbench cannot be wired into the engine as-is — and is not meant to be. This is
why the spike is a throwaway head-to-head, and why a passing result unlocks a **second
integration spike** (porting the parallel cell-loop) rather than greenlighting the full
port.

## 4. Method

The spike is a standalone harness under `scripts/spikes/native_predation/` (throwaway; not
imported by the engine, not covered by the suite).

### 4.0 Provenance & call-path guards (run FIRST, fail loudly)
This repo has two documented benchmark traps (perf-arc-overview.md:103-104). The harness
**must** guard both before any timing:
- **Worktree-import guard:** print and `assert` that
  `osmose.engine.processes.mortality.__file__` resolves under *this worktree* path, not an
  installed/site-packages or master checkout. Run with explicit `PYTHONPATH=<worktree>`.
- **Numba-path guard:** `assert mortality._HAS_NUMBA is True` — the per-cell Python path is
  `_HAS_NUMBA=False` dead code in production; timing it would measure the wrong code.
- **Flag-config guard:** record the boolean gate values present at capture
  (`diet_enabled`, `tl_tracking`, `use_stage_access`, `has_access`) into the fixture and
  re-apply them verbatim in the bench. Capture under the **calibration workload**
  (`diet`/`tl` *off*) — calibration is the stated motivation and `aggregate_diet` is
  output-gated/skipped there — and, if cheap, *also* under a default full-output run, so
  the verdict isn't an artifact of a flag configuration production calibration never pays.

### 4.1 Capture (at the call site, across a distribution of cells)
Temporarily instrument the leaf **call site** during one real `eec_full` run. For each of
a chosen set of invocations, deep-copy (`np.copy`) **all ~40 array/scalar arguments
immediately before the call** and deep-copy the mutated buffers **immediately after** — a
per-call before/after snapshot. This isolates exactly one leaf call: the leaf accumulates
into buffers shared across predators and **interleaved with other mortality causes**
(starvation/fishing/natural in the 4-cause order loop), so a cell-*entry* snapshot would
capture the wrong pre-state. Snapshot a **distribution** keyed on `n_local`
(`len(cell_indices)`): record the run's `n_local` histogram and capture cells at **p10,
p50, p95**, plus one **empty/early-return** cell. The per-call win is `n_local`-dependent
(prey-scan loop length), so one cell cannot calibrate the gate.

### 4.2 Boundary-free math benchmark (the gate metric)
The production Numba path calls the leaf **njit→njit with zero Python boundary**; timing
the Numba leaf *from Python* would charge it Numba's per-call dispatch cost that production
never pays, flattering C. So measure **math throughput with the boundary amortized on both
sides**:
- **Numba side:** a thin `@njit` driver that loops the leaf N times over the snapshotted
  inputs (one dispatch, N in-njit calls — the production reality).
- **C side:** the C reimplementation looped N times **inside C** behind a single cffi call
  (one FFI boundary, N native calls).
**Buffer re-init is mandatory and must be identical on both sides** — the leaf *mutates*
shared buffers (it reduces prey `inst_abd[q_idx]` and `rsc_biomass`, and early-returns once
a predator's own abundance is depleted, mortality.py:885). So each iteration must
re-initialise every mutated array (§2 set) to the snapshot **pre-state** *before* the leaf
call, and the **reset must sit outside the timed span** (time only the leaf call itself).
Both the Numba `@njit` driver and the C loop perform the *same* reset, so the gate ratio
reflects leaf math, not a memset asymmetry, and every timed call sees the production
pre-state rather than an artificially depleted one. Interleave A/B samples (not
all-Numba-then-all-C) to cancel thermal/governor drift; report **median + IQR** per cell,
and a Mann-Whitney on paired samples. The **gate ratio** is the call-count-weighted
C-vs-Numba math-throughput ratio across the p10/p50/p95 distribution.

(The `@njit` driver compiles cleanly despite the ~40 args and early-returns — the leaf is
already `@njit` and already called from `@njit` bodies at 1330/1511, so a thin `@njit`
driver calling it in a `range` loop type-infers identically, no `objmode`.)

### 4.3 Boundary-cost probes (quantify the integration risk)
Separately measure the per-call boundary cost each side pays when called *from Python*: a
no-op C function with the **same ~40-arg signature** via cffi, and an empty-cell Numba leaf
call. Report fixed per-call ns for each. This makes the §6 integration risk (FFI boundary
re-paid per cell on integration) a number, not a hand-wave.

### 4.4 Build variants
Compile the C with **portable `-O3` (no `-march=native`)** *and* with `-O3 -march=native`.
Report both ratios. **Gate on the portable number** — a production build cannot assume
`-march=native`.

### 4.5 Parity
Feed C the snapshotted **pre-state** of each captured call, run **once**, diff every
mutated array (§2 set) against the snapshotted **post-state**. Report max abs and max rel
difference per array. Apply the §2 bar (≤1e-12 rel = pass; larger = correctness fail).

### Disqualifying gate (pre-registered, hard numbers)
- **If the boundary-free, portable-build, call-weighted math ratio is < 1.3×** → **STOP.**
  The integration cost (a *parallel* native cell-loop with re-implemented RNG), the second
  code path to maintain, and the build/CI/prod-deploy burden cannot be justified by a
  smaller margin — especially with **no workload currently demanding the speed** (this was
  a "re-profile then decide" exploration, not a slow-run complaint).
- **If ≥ 1.3×** → this PASS authorizes **only** a follow-on **integration spike** (port the
  parallel cell-loop, reproduce the RNG + 4-cause interleave, measure end-to-end `eec_full`
  wall-time). It does **not** greenlight the full port. The leaf math ratio is *necessary,
  not sufficient*: integration re-pays an FFI/boundary tax (quantified in §4.3) that
  njit→njit avoids, so a bare-1.3× leaf win can still be a net loss once wrapped. The
  integration spike — not this one — owns the real go/no-go.

## 5. Deliverable

A `docs/perf/2026-06-24-native-predation-kernel-spike.md` artifact-of-record, in the style
of the K-arc not-shipping write-ups, containing:
- the provenance assertions (resolved `mortality.__file__`, `_HAS_NUMBA`, captured flag
  config) so the measurement is auditable;
- the `n_local` histogram and the p10/p50/p95 (+empty) cell characteristics;
- per-cell **and** call-weighted Numba-vs-C math-throughput median+IQR and ratio, for
  **both** portable `-O3` and `-march=native` builds;
- the per-side boundary-cost probe numbers (§4.3);
- per-output max abs/rel parity differences against the §2 bar;
- a go / no-go recommendation with reasoning, stating explicitly that a PASS authorizes
  only the integration spike.

The spike code stays under `scripts/spikes/native_predation/` as a reproducible harness
(kept, not deleted, so the artifact's numbers can be re-derived). It is **not** wired into
the engine and **not** added to the test suite or CI.

## 6. Risks & honesty notes

- **Numba is already LLVM-optimised** — a negative result is real and acceptable; the spike
  is designed to fail cheaply.
- **Microbench ≠ end-to-end** — even a positive boundary-free leaf result does not prove a
  production win; the *parallel* integration spike is the real gate before committing.
- **Boundary tax** — njit→njit pays no Python/FFI boundary; the C path will, re-paid per
  cell on integration. §4.3 quantifies it so a 1.3× leaf win isn't over-read.
- **Production kernel is parallel** — the eventual port must reproduce `prange` parallelism
  (OpenMP), not just the serial math. The spike does not, by design, but the verdict text
  must remind the reader the real target is harder.
- **`-march=native`** — used for one of two reported builds; the gate uses the portable
  build so the ratio isn't over-read.
- **No new dependency** — `cffi`/`gcc` only; an eventual setuptools-C-extension or Cython
  choice is a later sub-project decision and does not affect the spike's validity.

## 7. Out of scope / explicitly deferred

- Full kernel port, native cell-loop, Numba fallback path.
- RNG-stream (MT19937 + NumPy Fisher-Yates) reproduction.
- Build wiring, CI compilation, wheels, prod-clone deployment.
- `movement()` and `aggregate_diet_by_species` wins (separate, lower-effort levers).
- GPU/CUDA.
