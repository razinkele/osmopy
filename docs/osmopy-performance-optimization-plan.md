# Osmopy Performance Optimization Plan

Revised 2026-11-14 after review (docs/perf/2026-11-14-performance-plan-review.md); original draft in git history (0119c53e).

Every number below is labelled **measured** (with its source) or **inferred**. File:line
citations are against master on 2026-11-14 and will drift; re-check them with `grep -n`
before relying on one.

---

## 1. Objective

Improve Osmopy performance without compromising:

- **Python fixed-seed reproducibility.** The same config and seed give bit-equal output
  across runs of the Python engine. The committed baselines in `tests/test_engine_parity.py`
  are compared at `atol=0`.
- **Java agreement "within 1 OoM" per species.** This is a statistical contract, not a
  bit-exact one (NumPy PCG64 vs Java MT19937). It holds for 14/14 EEC species and 7/8 Bay
  of Biscay. A performance change must not move it, and cannot be validated by it alone.
- ecological validity of the outputs;
- existing calibration, sensitivity and UQ workflows.

The original draft listed "compatibility with the Java-compatible reference engine" as one
objective. That conflated the two contracts above; they are checked by different tests and
have different tolerances.

---

## 2. Guiding principles

### 2.1 Optimize algorithms before changing technology

Reduce work first, then parallelise independent work, then reduce the number of simulations
a calibration needs. Do not move the current algorithm to another technology (native
extension, GPU) before the first three are exhausted and measured.

### 2.2 Classify every change before starting it

| Class | Meaning | Gate |
|---|---|---|
| Performance-preserving | Reproduces the existing algorithm and the fixed-seed stream | `tests/test_engine_parity.py` at `atol=0`, unchanged baselines |
| Statistically equivalent | Changes individual trajectories, keeps output distributions | Regenerated baselines plus a multi-seed comparison with tolerances fixed in advance |
| Model-changing | Changes the numerical model | Separate scientific validation; never part of a performance release |

Only performance-preserving work belongs in a performance release. Section 6 lists changes
that look performance-preserving in a generic engine and are not in this one.

### 2.3 Measure before and after, on this box, interleaved

- Profile with a warm Numba cache before proposing a change. A cold first run cost 45.96 s
  against about 5.6 s warm on EEC 5 yr (measured 2026-09-12).
- Compare control and treatment **interleaved** (A, B, A, B, ...), and report the **median
  of N**, never the best of N. Master's own EEC runs spread 15% in September, and a
  best-of-5 comparison produced a phantom 11.9% win in July
  (`docs/diagnostics/2026-07-14-schoolstate-perf-negative.md`).
- Confirm the changed code runs on the benchmark config. Two September items were inert
  because the code path was gated off on the fixtures.
- Run one engine job at a time (section 6).

---

## 3. Targets

| Original target | Status | Replacement |
|---|---|---|
| Single simulation 2–5× faster | **Unsupported.** No evidence in the repo's history points to it. | No headline target. Each item in section 5 carries its own bound. |
| Ensemble 8–12× on 16 physical cores | **Wrong hardware, undefined baseline.** The box has 14 physical cores (28 logical, 31 GB). | Measure a worker-scaling curve (item 1). Expect at most about 6× against threaded serial runs (inferred, below). |
| 5–20× fewer calibration evaluations | **Cannot be judged.** The only optimizer comparison is on synthetic functions. | Produce one matched-budget comparison on a real objective (item 5), then set a target. |
| Conservation residual below 1e-10 per process | **Undefined for this model.** | Death-accounting identity (below). |
| Fixed-seed regression error zero where event order is unchanged | Kept. | `tests/test_engine_parity.py` at `atol=0`. |

**Why 2–5× is unsupported.** Shipped history (all measured): −41% wall time on EEC from
Python-side vectorisation (May, `docs/perf/2026-05-08-perf-arc-overview.md`); 1.5–2× from
capping single runs at physical cores (July,
`docs/perf/2026-07-15-mortality-thread-oversubscription.md`); about 1.7× on Baltic from the
sub-one-fish floor (November). The September pass concluded there was no cheap win left in
the mortality kernel (`docs/plans/2026-05-05-deep-review-remediation-plan.md`, "Exploring
the algorithmic improvement"). On the 2026-11-14 Baltic profile (section 4.1), an infinitely
fast mortality kernel gives about 2× at 14 threads; at one thread, 2× needs the kernel about
2.8× faster (inferred from the shares by Amdahl).

**Why the ensemble target needs a baseline.** A one-thread Baltic run took 2.2× as long as
a 14-thread run (measured 2026-11-14, under load). So 14 workers pinned to one thread each
are at best about 14 / 2.2 ≈ 6× faster than running the same simulations serially with
threads (inferred). 8–12× is only reachable against a one-thread serial baseline. April
data (older engine, thread count uncontrolled): 175 evals/h at 8 DE workers; the per-worker
rate fell from 0.4 to about 0.35 evals/min at 16 workers and 0.13 at 24, attributed to
memory bandwidth (`docs/claude-memory/feedback_de_workers_default.md`).

**Why the conservation target is undefined.** With bioenergetics off, growth is
length-based and gated by predation success (`osmose/engine/processes/growth.py:48-90`);
eaten biomass does not become predator biomass. The sub-one-fish floor
(`osmose/engine/simulate.py:1016`) and aging are deliberate sinks. There is no per-process
biomass balance to hold to 1e-10.

**Replacement: death accounting.** For every school and step, the abundance removed equals
the sum of `n_dead` over causes, plus the abundance zeroed by the floor. This is checkable
to floating-point accuracy and is what an optimisation of the mortality path can break.
No such ledger exists today; add it as a test when item 6 is ever started, not before.

---

## 4. Current state

### 4.1 Where the time goes (measured 2026-11-14)

`scripts/profile_engine.py --config baltic --years 10`, current master, Intel i9-10940X.
Three sequential runs: warm-up, warm at the automatic thread count, then
`NUMBA_NUM_THREADS=1`. **Another project's test suite was running (load 1.6 → 3.0), so read
the shares, not the seconds.** The automatic thread count is 14 (`physical_budget()`,
confirmed by a separate call; the profiler does not print it).

| Phase | 14 threads (15.1 s total) | 1 thread (33.4 s total) |
|---|---|---|
| `_mortality` | 7.65 s, 50.8% | 26.16 s, 78.4% |
| `_movement` | 3.82 s, 25.4% | 3.74 s, 11.2% |
| `_collect_by_life_stage` | 0.60 s, 4.0% | 0.58 s, 1.8% |
| `_growth` | 0.39 s, 2.6% | 0.37 s, 1.1% |
| `_collect_mortality` | 0.22 s, 1.4% | 0.21 s, 0.6% |
| `_reproduction` | 0.18 s, 1.2% | 0.18 s, 0.5% |
| state init + loop overhead | 1.58 s, 10.5% | 1.50 s, 4.5% |
| engine setup (`_prepare_run`) | 0.25 s, 1.7% | 0.25 s, 0.8% |
| `write_outputs()` | 0.07 s, 0.5% | 0.07 s, 0.2% |

What this shows and does not show:

- Movement is serial and is a quarter of a threaded Baltic run. Earlier profiles did not
  show this: they were on EEC (May) or on a 4-core container (September).
- A calibration worker pinned to one thread sees the right-hand column: mortality is 78%.
- The profiler cannot see inside the Numba kernels or count predator–prey pairs. Baltic
  cell occupancy and size-window selectivity have never been measured; the September table
  covered EEC and Bay of Biscay only. Baltic runs `mortality.subdt = 10`.
- Do not compare these seconds with the July or September figures. The Baltic config and
  the machine load differ.

### 4.2 Status of every original phase and work package

Status values: **done** / **rejected** (tried or argued, with the record) / **conflict**
(breaks an invariant in section 6) / **wrong assumption** / **open**.

| Original item | Status | Evidence |
|---|---|---|
| §4 0.1–0.3, 0.6 benchmark and profiling tools | done | `scripts/benchmark_engine.py`, `scripts/profile_engine.py`, `tests/baselines/` |
| §4 0.2 `osmose benchmark` CLI | open, not needed | `osmose/cli.py:123-135` has `validate`, `run`, `report` only |
| §4 0.1 minimal 2-species CI benchmark | wrong assumption | `data/minimal` never produces a predation event |
| §4 0.1 candidate-pair counters | open | never measured for Baltic; item 3 |
| §4 0.5 per-process conservation diagnostics | wrong assumption | section 3 |
| §5 WP1.2 cell sort and offsets | done | `osmose/engine/processes/mortality.py:2576-2581` |
| §5 WP1.2 re-key the sort by cell × species | conflict | changes within-cell order; `mortality.py:1990-2007`, `:1195` |
| §5 WP1.3 species-pair accessibility graph | open, low value, has a trap | cells hold a median of 5–15 schools (EEC, measured 2026-09-12); see section 6 |
| §5 WP1.4 size filter before expensive work | done | `mortality.py:1167` precedes accessibility and biomass |
| §5 sort by length + binary search | rejected (argued, not built) | remediation plan, "Exploring the algorithmic improvement" |
| §5 contiguous per-cell gather | rejected (built, bit-exact, measured slightly slower) | same section: 5.342 vs 5.376 s, 5.286 vs 5.427 s |
| §6 WP2.1 structure-of-arrays state | done | `osmose/engine/state.py:31` |
| §6 WP2.1 mixed precision | conflict | `atol=0` baselines |
| §6 WP2.2 no Python objects in kernels | done | kernels are `@njit` on flat arrays |
| §6 WP2.3 fused natural + fishing mortality | conflict | causes are interleaved per school in shuffled order, `mortality.py:2024` |
| §6 WP2.4 allocation control | rejected | K1 shipped (6.4%); K2, K3, P4, P4b dropped at 0.2–1.8% (`docs/perf/2026-05-08-*-not-shipping.md`); removing all `SchoolState` validation measured 6.675 vs 6.674 s |
| §6 WP2.5 compaction policy | done + conflict | floor at `simulate.py:2129`, compaction every step at `:2247`; deferring it changes `n_local` |
| §7 WP3.1 objective-only run mode | done | `PythonEngine.run_in_memory`, `osmose/engine/__init__.py:123` |
| §7 WP3.2 process pool | done for calibration and UQ | forkserver `osmose/calibration/problem.py:354-357`; spawn `osmose/calibration/uq/design.py:194-196` |
| §7 WP3.2 ensemble runs | open | `PythonEngine.run_ensemble` (`__init__.py:157`) is a serial loop with one `output_dir`; no callers found |
| §7 WP3.3 one thread per worker | **done 2026-11-14** (split rule adopted, 3.96×) | item 1; `docs/perf/2026-11-14-worker-thread-sweep.md` |
| §7 WP3.4 shared worker initialisation | partly done; not a lever | built once per worker; config re-read per evaluation at `problem.py:445`; setup is 0.25 s of a 15–33 s run |
| §7 WP3.5 seed hierarchy | open at task level; conflict inside the engine | item 5 and section 7 |
| §8 Phase 4 hybrid stochastic sampling | wrong assumption | no binomial death sampling exists; appendix |
| §9 WP5.1 Morris screening | done | `osmose/calibration/preflight.py:137` |
| §9 WP5.2 common random numbers | done for DE; open for NSGA-II | DE uses one fixed seed; NSGA-II uses `seed=run_id`, `problem.py:448` |
| §9 WP5.3 staged seeds, finalist re-ranking | partly done | `osmose/calibration/multiseed.py`; `scripts/calibrate_baltic.py:1566` |
| §9 WP5.4 active surrogate workflow | done for UQ; untested for calibration | `osmose/calibration/uq/run.py` (GP emulator, calibration gate, sampler); `surrogate_de.py` |
| §9 WP5.5 shorter simulations as low fidelity | conflict for Baltic | short horizons measure the seeding transient |
| §10 Phase 6 JAX/GPU | dropped | appendix |
| §11 research branch | unchanged, out of scope | model-changing by the plan's own classification |
| §13 repository restructure | dropped | appendix |

---

## 5. Work plan

Six items, in order. Items 1–3 are cheap and decide whether 4–6 are worth starting.
Standard A/B protocol for every engine change (referred to below as **the A/B protocol**):

```bash
# control and treatment alternate; warm cache; one job at a time
.venv/bin/python scripts/benchmark_engine.py --config baltic --years 5 --repeats 7 --output /tmp/a1.json
# switch to treatment, repeat, switch back, repeat -- at least two pairs
.venv/bin/python scripts/benchmark_engine.py --compare /tmp/a1.json /tmp/b1.json
```

Report medians. Keep a result only if every interleaved pair agrees in sign. When the
treatment lives in a worktree, run with `PYTHONPATH=.` from the worktree (CLAUDE.md,
worktree import note) and confirm the imported `osmose.__file__`.

### Item 1 — Worker × thread sweep for the process pools

**Status: DONE 2026-11-14 — pinned split rule adopted** (`docs/perf/2026-11-14-worker-thread-sweep.md`): 14 workers × 1 thread give 492 evals/h against 124/h for 8 unrestricted workers (3.96×); the calibration and UQ pools and `calibrate_baltic.py`'s DE path now apply `max(1, physical // W)` threads per worker.

**Goal.** Find the worker count and threads-per-worker that maximise evaluations per hour
for calibration and UQ on this box, then make the pools apply it themselves.

**This reverses a documented decision.** The 2026-07-15 thread-policy spec
(`docs/superpowers/specs/2026-07-15-single-run-thread-policy-design.md`) keeps calibration
workers at Numba's unrestricted default, and two tests assert it:
`tests/test_thread_policy.py:237` (`test_cap_does_not_leak_into_forkserver_worker`) and
`:286` (`test_worker_init_does_not_reset_numba_threads`). Both must be rewritten to assert
the new policy if the sweep supports a cap. The module docstring of
`osmose/engine/thread_policy.py` must change with them.

**Evidence.**
- For the existing decision: capping Numba threads slowed a 4-worker run (measured April,
  **thread** backend, GIL-bound; `docs/claude-memory/project_v010_calibration_python_engine.md`).
- Against it, for **process** pools: an uncapped 8-worker UQ run took 4.4 h against about
  50 min pinned (observed 2026-07-24, not a controlled A/B); a surrogate-DE sweep recorded
  thread contention across 8 workers. Today the cap is a launch-environment convention
  (`NUMBA_NUM_THREADS=1` etc.), not code.

**Expected gain.** Unmeasured. Large for calibration and UQ throughput if the July
observation holds; possibly nil for small worker counts.

**Invariant risk.** None to results: mortality output is bit-identical at every thread
count (`tests/test_thread_policy.py:192`). Risk is operational: memory (about 400 MB per
50-yr Baltic run, `scripts/calibrate_baltic.py:1472-1477`) and memory bandwidth above 16
workers.

**Verification.**
- `scripts/benchmark_calibration.py --backend process`, workers ∈ {1, 4, 8, 12, 14, 16},
  threads per worker ∈ {1, 2, unrestricted}. Record evals/h and peak resident memory.
- Run the grid cells sequentially. Repeat the best two cells and the current default
  interleaved.
- Check workers show about 100% CPU each when pinned to one thread.

**Exit criterion.** A table of evals/h per cell. If a pinned setting beats the unrestricted
default by more than the spread between repeats, set it in `_worker_init`
(`problem.py:77`) and the UQ `_worker_init` (`uq/design.py:157`), update the two tests and
the docstring. If not, record the negative in `docs/perf/` and leave the code alone.

### Item 2 — Diagnose Baltic movement

**Status: DONE 2026-11-14 — no code change** (`docs/perf/2026-11-14-baltic-movement-diagnosis.md`): 92% of movement time is the serial seeded Numba kernel (0.94 s of 1.02 s in a warm 5-yr Baltic run); the Python side is 1.8% of the run, below the 2% bar. Parallelising the kernel is a statistically-equivalent change (it needs independent per-school streams).

**Goal.** Explain the 3.8 s that movement costs in a 10-yr Baltic run, and remove whatever
part of it is Python-side.

**Expected gain.** Bounded at 25% of a threaded Baltic run and 11% of a one-thread run
(measured shares, section 4.1). The split between the Python wrapper
(`_precompute_map_indices`, array copies) and the Numba kernel `_map_move_batch_numba` is
unmeasured.

**Invariant risk.** The map kernel draws from one serial seeded stream
(`osmose/engine/processes/movement.py:470`). Parallelising or reordering it changes
fixed-seed output: that is a statistically-equivalent change, not a performance-preserving
one. Python-side wrapper work is safe if it leaves draw order alone.

**Verification.**
- Diagnose first: warm cProfile of a Baltic 5-yr run, sorted by `tottime`, restricted to
  `movement.py`. Do not write code before this.
- Gate: `.venv/bin/python -m pytest tests/test_engine_parity.py tests/test_perf_movement_diet_vectorization.py`.
- Measure: the A/B protocol on `--config baltic`, and once on `--config eec_full` to check
  for a regression.

**Exit criterion.** Either a performance-preserving change with a median gain of at least
2% on Baltic in every interleaved pair, or a `docs/perf/` note recording that the cost is
inside the serial kernel and what a statistically-equivalent change would require.

### Item 3 — Baltic pair and occupancy counters

**Status: DONE 2026-11-14 — criterion fails, Baltic cells are full**
(`docs/perf/2026-11-14-baltic-cell-occupancy.md`). Over years 6–10 the median cell holds 26 schools
(mean 29.1, max 390), against the ≤15 that would have carried September's conclusion over.
- 57.5% of scanned pairs sit in cells over 50 schools, and 29% in the roughly 9 cells per step over
  100.
- 11.8% of scanned pairs pass the size window.
- With `mortality.subdt = 10` that is about 1.08 billion pair tests over the five years, by an
  end-of-step count that approximates the kernel's `n_local`.

This weakens only the "cells are tiny" objection. The visit-order objection stands.

**Goal.** Measure, for the Baltic config, schools per cell (mean, median, maximum), pairs
scanned and the fraction passing the size window, as was done for EEC and Bay of Biscay on
2026-09-12.

**Expected gain.** None directly. It decides whether any work on the mortality kernel
(item 6, or the WP1.3 graph) has a plausible payoff. If the median cell holds about 15
schools or fewer, the September conclusion carries over and kernel search work stays
closed.

**Invariant risk.** None. Use `step_observer` on `PythonEngine.run`
(`osmose/engine/__init__.py:91-99`); the engine is not modified.

**Verification.** One Baltic 10-yr run with the observer. Cross-check the school count
against the roughly 18.6k live schools per step recorded after the floor.

**Exit criterion.** A table in `docs/perf/` beside the September one, with the note that
Baltic runs `mortality.subdt = 10`.

### Item 4 — The serial tail

**Goal.** Reduce state initialisation, loop overhead and output collection, which together
are about 14% of a threaded Baltic run (10.5% loop overhead plus 4.0%
`_collect_by_life_stage`, measured).

**Expected gain.** Unmeasured; modest. The May arc found most single edits here below 2%.
One candidate specific to calibration: collectors whose output no objective reads.

**Invariant risk.** Low if outputs are unchanged. Skipping a collector changes what
`OsmoseResults` serves, so make it explicit and opt-in.

**Verification.** Warm cProfile to name the functions inside "loop overhead"; the A/B
protocol; `tests/test_engine_parity.py`.

**Exit criterion.** Ship only edits with a median gain of at least 2% in every interleaved
pair. Record the rest as not-shipping notes in the existing `docs/perf/` format.

### Item 5 — Calibration: seeding policy and a real-objective comparison

**Goal.** (a) Give the NSGA-II path an opt-in common-random-number mode. (b) Compare DE,
CMA-ES and surrogate-DE at a matched evaluation budget on one real OSMOSE objective.

**Evidence.** `OsmoseCalibrationProblem` passes `seed=run_id`, the candidate's index in the
population (`problem.py:448`), so seed noise is tied to population position, and
`_cache_key` (`problem.py:576`) omits the seed. The DE path already runs every candidate on
one fixed seed and re-ranks the best on five (`scripts/calibrate_baltic.py:1566`). The
optimizer comparison in `data/benchmarks/optimizer_comparison.json` is synthetic only.

**Expected gain.** Unmeasured. No evaluation-count target is set until (b) exists.

**Invariant risk.** Changing the default seed policy would make past NSGA-II runs
unreproducible and silently mix cached results. Keep the current behaviour as the default;
add the seed to the cache key only together with the opt-in mode.

**Verification.** Unit test that the opt-in mode gives every candidate in a generation the
same seed set and that thread and process backends still agree bit-for-bit. For (b): same
budget, same seeds, final candidates re-ranked on five seeds, run after item 1 has fixed
the pool settings. Use full-length runs; a short Baltic horizon measures the seeding
transient.

**Exit criterion.** The opt-in mode merged with its test, and a results table for (b).
Only then decide whether a staged workflow (screen → surrogate → re-rank) is worth wiring
as a default.

### Item 6 — Single-thread mortality kernel (open, not scheduled)

**Goal.** Decide whether the mortality kernel can be made faster for the one-thread worker
regime, where it is 78% of the run (measured).

**Status.** Open and unmeasured. The July decision not to pursue a native port was argued
for multi-thread single runs, where fork/join and memory bandwidth bind. It was not
measured at one thread. The June leaf-kernel spike
(`docs/perf/2026-06-24-native-predation-kernel-spike.md`) passed its gate, but its Numba
leaf times were flat from 1 to 24 schools per cell, which suggests it measured call
overhead; do not plan against its ratio.

**Invariant risk.** High. This is the engine's most parity-sensitive code: within-cell
visit order, `n_local`, the per-cell MT19937 stream and the `exp` implementation all feed
the `atol=0` baselines.

**Verification, if ever started.** The death-accounting test from section 3 first; the
kernel equivalence tests against `_mortality_in_cell`; `tests/test_engine_parity.py`; the
A/B protocol at `NUMBA_NUM_THREADS=1` and at the automatic count.

**Entry criterion.** Items 1 and 3 are done, item 1 shows that pinned one-thread workers
are the production regime, and item 3 shows enough pairs per cell on Baltic to give a
kernel change something to save.

**Entry status, 2026-11-14:**
- **Item 1:** met. The split rule gives one thread per worker at 8 or more workers on this
  14-core box, including the DE default of 8 and the sweep's best of 14.
- **Item 3's occupancy part:** met (median 26 schools per cell, 57.5% of pairs in cells over 50).
- **Payoff:** still unmeasured. The first number to take is the share of one-thread kernel time
  spent in the Phase 1a scan on Baltic.

**One candidate, a hypothesis only: an order-preserving prefilter.** It would find the eligible
prey faster and visit them in their original cell order. To count as performance-preserving it
must keep the eligible set, the visit order (so `total_available`), `n_local` and the per-cell
random stream all unchanged. See the item 3 note.

---

## 6. Guardrails

Any change must respect these. Each has cost at least one retraction in this repo.

1. **Fixed-seed baselines.** `tests/test_engine_parity.py` compares committed baselines at
   `atol=0`. A change that moves them is not performance-preserving, whatever it looks
   like.
2. **The bioen-OFF random stream.** It includes:
   - the per-cell seed and the five unconditional permutations (four used without bioen);
   - **within-cell school order**, set by the stable sort at `mortality.py:2576`, because
     permutations index positions in the cell slice;
   - **`n_local`**, the number of schools in the cell, including zero-abundance ones not yet
     compacted. Deferring compaction changes it;
   - the running sum `total_available` (`mortality.py:1195`), whose value depends on visit
     order.
3. **Numba `exp` is not NumPy `exp`.** They differ in the last bit on 5–9% of arguments,
   amplified by `1 - exp(-D)`. Do not chase this as a kernel bug; choose fixtures that
   avoid it. `expm1` is a different function again and moves every baseline.
4. **Mortality causes are interleaved per school in shuffled order over sub-steps.** That
   is Java's algorithm. Combining causes into one hazard is model-changing.
5. **Process pools use forkserver or spawn.** Fork deadlocks with Numba's thread pool;
   loky broke worker un-pickling once. Scripts that start a spawn pool need a
   `__main__` guard.
6. **One engine job at a time.** Two engine workloads on this box contend for memory
   bandwidth and have been OOM-killed. Never benchmark while the test suite or a
   calibration runs.
7. **`data/minimal` never predates.** It cannot detect a predation or mortality-kernel
   regression. Use the tutorial config (`tests/_tutorial_config.build_config`) or Baltic.
8. **`@pytest.mark.slow` does not gate CI.** A benchmark test that must not run in CI needs
   an environment-variable gate as well.
9. **Bay of Biscay past year 10 is a dying ecosystem.** Use it for short runs only; use EEC
   or Baltic for anything longer.
10. **A missing accessibility entry means full access, not none.** A precomputed
    "positive accessibility" species graph would drop those pairs. The matrix is also
    stage-indexed and rescaled per call by the prey-density scale.
11. **Java cross-checks are "within 1 OoM".** They cannot confirm a bit-level change is
    harmless, and a green one-year Java parity test says nothing about a ten-year run.

---

## 7. Open item to measure: adjacent seeds may share a stream

**Inferred from code; effect unmeasured.** `PythonEngine._prepare_run`
(`osmose/engine/__init__.py:72-91`) builds the movement generators from `seed` and the
mortality generators from `seed + 1`, both through `build_rng`
(`osmose/engine/rng.py:35-50`). Replicates and UQ design tasks use consecutive seeds. So
the movement stream of replicate `s + 1` is built from the same seed as the mortality
stream of replicate `s`.

What is not known: whether the batched Numba mortality path consumes those generators at
all (it seeds MT19937 per cell from a separate value), and whether any shared draws produce
a measurable correlation between adjacent replicates.

**To settle it.** Trace which generator each production path draws from. If a shared
stream is consumed, compare the between-replicate variance of final biomass for seeds
`{s, s+1, ...}` against seeds spaced far apart, on one config, at least 16 replicates each.

**If it is real.** Fix it at the task level by spacing or spawning replicate seeds in the
caller. Changing seeding inside the engine moves every fixed-seed baseline and invalidates
calibration caches.

---

## Appendix: dropped from the original

| Original item | Reason |
|---|---|
| Phase 4, hybrid stochastic sampling (§8) | The engine has no binomial death sampling. Deaths are deterministic `abd * (1 - exp(-D))` on float abundance; random numbers go to ordering and movement. There is nothing to approximate, and a hybrid sampler would add noise. |
| Phase 6, JAX/GPU prototype (§10) | Within-cell mortality mutates instantaneous abundance sequentially in permuted order, so it cannot be batched without changing the algorithm. The box's GPU is a GT 1030 with 2 GB. |
| §13 repository restructure | No performance value. It would invalidate the Numba cache layout and the many file:line citations in CLAUDE.md and the docs. |
| Mixed precision (WP2.1, Epic C) | Breaks the `atol=0` baselines; no measured memory-bound kernel to justify it. |
| Fused natural + fishing mortality (WP2.3), competing-risk mortality (§11.1) | Causes are interleaved per school in shuffled order; fusing them is model-changing and moves the random stream. |
| `PythonEngine(numerics_mode=..., performance_profile=...)`, `engine.run_objective(...)` | These APIs do not exist. `PythonEngine.__init__` takes only `backend` (`osmose/engine/__init__.py:35`); the in-memory path is `run_in_memory`. A parallel "legacy/optimized" engine is not needed for performance-preserving work, which the parity tests already gate. |
| Sparse, cell-sorted predation as the first milestone (§5, §16) | The cell sort and the size-first filter exist. The two remaining ideas were rejected in September, one on measurement. |
| Deferred school compaction (WP2.5) | The engine floors and compacts every step; deferring changes `n_local` and the stream. |
| CI gates on 10–15% runtime regressions (§12) | Run-to-run spread on master has reached 15%. Use interleaved medians locally. |
| 100-seed statistical tests per config (§12) | Cost is out of proportion; the existing guard is 10 seeds at `rtol=0.05` (`tests/test_engine_parity.py:358-363`). Set the seed count per change. |
