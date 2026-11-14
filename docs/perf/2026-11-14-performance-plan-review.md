# Review of `docs/osmopy-performance-optimization-plan.md`, 2026-11-14

- **Reviewer:** a Claude Fable 5.2 subagent, at the maintainer's request. It read the draft plan
  (1,164 lines, original in git as `0119c53e`), the engine code, `CLAUDE.md`, and the prior
  performance record in `docs/perf/` and `docs/superpowers/specs/2026-03-22-performance-*`.
- **Profile:** one fresh profile of current master on this machine.
- **Spot-checked by the main session** (all confirmed):
  - `PythonEngine.__init__` takes only `backend` (`osmose/engine/__init__.py:35`);
  - movement is seeded with `seed` and mortality with `seed + 1` (`__init__.py:87-90`);
  - `osmose/engine` has no binomial or Poisson sampling;
  - the box has 14 physical cores and 28 logical (`lscpu`).
- **The rest** is the reviewer's own reading, with the citations below.
- **Revised plan:** `docs/osmopy-performance-optimization-plan.md`.

## 1. Verdict

**Partly sound in its principles; not worth executing as written.** The plan is generic: it describes an
engine with an all-pairs predation search, object-heavy state, binomial death sampling and no parallel
pools, and osmopy is none of those.
- **Much of it already exists or was rejected.** Roughly half of Phases 0–3 already exists, and the Phase 1
  centrepiece was explored and rejected on measurement on 2026-09-12.
- **Phase 4 targets a sampler the engine does not have.**
- **Several "performance-preserving" items would break the fixed-seed baselines.**
- **Nothing in the repo's history supports the 2–5× single-run target.**

What is worth doing is small and specific (section 5).

## 2. Per phase / work package

Classes: (a) already implemented, (b) tried and rejected, (c) feasible and likely worthwhile, (d)
conflicts with an invariant, (e) wrong assumption.

**Phase 0 (§4) — mostly (a).**
- §0.1–0.3 and 0.6 already exist: `scripts/benchmark_engine.py` (fixtures, repeats, compare) and
  `scripts/profile_engine.py` (stage and phase timers). `tests/baselines/` holds the parity and benchmark
  baselines.
- The §0.2 `osmose benchmark` CLI is absent: `osmose/cli.py:123-135` has only `validate`, `run` and
  `report`. It is not needed.
- The §0.5 per-process conservation ledger is (e); see section 3.
- §0.1's "Baltic, 8 focal species" is 9 focal (`baltic_param-simulation.csv:4`), plus background predators
  and 6 resources.
- §0.1's candidate-pair counters are (c): cheap through `step_observer`, and never measured for Baltic.

**Phase 1 (§5) — mostly (a)/(b).**
- **WP1.2 cell sort: (a).** `mortality.py:2576-2581` does a stable argsort by cell, plus `searchsorted`
  boundaries, on every call.
- **WP1.2 re-keying by cell × species: (d).** Within-cell order feeds `np.random.permutation(n_local)`
  (`mortality.py:1990-2007`) and the running sum `total_available` (`:1195`). Reordering changes both the
  fixed-seed stream and the floating-point result.
- **WP1.4 size filter first: (a).** The ratio test at `mortality.py:1167` precedes the accessibility and
  biomass work.
- **Sort by length plus binary search: (b), rejected on argument, not built.** Cells hold a median of 5–15
  schools, and the change forfeits bit-exactness (`docs/plans/2026-05-05-deep-review-remediation-plan.md`,
  §"Exploring the algorithmic improvement").
- **Contiguous per-cell gather: (b), built.** It was bit-exact at atol = 0 and measured slightly slower
  (5.342 vs 5.376 s and 5.286 vs 5.427 s; same section).
- **WP1.3 species-pair graph:** never tried. Its value is low at these cell sizes, and it carries a trap
  (section 6).
- **The "3–20× predation kernel" estimate assumes an all-pairs search.** The search is already per cell.

**Phase 2 (§6) — (a)/(b)/(d).**
- **WP2.1 structure-of-arrays: (a).** `SchoolState` is at `osmose/engine/state.py:31`.
- **WP2.2 no Python objects in kernels: (a).** The kernels are `@njit` on flat arrays.
- **WP2.4 allocation control: (b).**
  - K1 shipped at 6.4%. K2, K3, P4 and P4b were dropped at 0.2–1.8%
    (`docs/perf/2026-05-08-*-not-shipping.md`).
  - Removing all `SchoolState` validation measured 6.675 vs 6.674 s
    (`docs/diagnostics/2026-07-14-schoolstate-perf-negative.md`).
  - The September pass ruled out hoisting scratch arrays in the parallel kernel.
- **WP2.5 compaction policy: (a) + (d).** The engine floors sub-one-fish schools (`simulate.py:2129`) and
  compacts every step (`:2247`). Deferring compaction leaves dead schools in cell slices, which changes
  `n_local` and therefore the permutation stream.
- **WP2.3 fused natural and fishing mortality: (d).** The engine interleaves causes per school, in shuffled
  order, over sub-steps (`cause_orders`, `mortality.py:2024`), which is Java's algorithm. Fusing changes
  both the model and the stream.
- **Mixed precision (Epic C): (d)**, against the atol = 0 baselines in `tests/test_engine_parity.py`.

**Phase 3 (§7) — mostly (a).**
- **WP3.1 objective-only mode: (a).** This is `PythonEngine.run_in_memory` (`osmose/engine/__init__.py:123`).
- **WP3.2 process pool: (a)** for calibration (forkserver, `problem.py:354-357`) and UQ (spawn,
  `uq/design.py:194-196`). `PythonEngine.run_ensemble` (`__init__.py:157`) is a serial loop that writes
  every replicate to one `output_dir`; the reviewer found no callers.
- **WP3.3 one thread per worker: (c), but it reverses a documented decision.**
  - The 07-15 thread-policy spec keeps calibration workers unrestricted, and
    `tests/test_thread_policy.py:237,286` assert it.
  - That decision rests on a 4-worker thread-backend measurement from April.
  - The process-pool evidence points the other way: an uncapped UQ run took 4.4 h against about 50 min
    pinned, and a surrogate-DE sweep noted thread contention across 8 workers. Neither was a controlled A/B.
  - Today the cap is only a launch-environment convention.
- **WP3.4 per-worker initialisation: partly (a).** The problem and evaluator are built once per worker, but
  `_run_python_engine` re-reads the config on every evaluation (`problem.py:445`). Setup is 0.25 s of a
  15–33 s run, so this is not a lever.
- **WP3.5 seed hierarchy:** (c) at the task level, (d) inside the engine. Section 6 describes a
  stream-reuse hazard.

**Phase 4 (§8) — (e).** The engine has no binomial death sampling.
- **Deaths are deterministic:** `abd * (1 - exp(-D))` on float abundance.
- **The random numbers drive ordering and movement only.** A grep for binomial or Poisson in
  `osmose/engine` finds only genetics noise.
- **So there is nothing to approximate,** and a "hybrid" would add noise.

**Phase 5 (§9) — the pieces are (a); the workflow is untested.**
- **Already present:** Morris screening (`preflight.py:137`), Sobol (`sensitivity.py`), CMA-ES,
  surrogate-DE, and multi-seed re-ranking (`multiseed.py`, `calibrate_baltic.py:1566`).
- **WP5.4 active surrogate: largely (a) for UQ.** `uq/run.py` grows a design until a GP emulator passes a
  calibration gate, then samples.
- **WP5.2 common random numbers: (c) as an opt-in.** The DE path already uses one fixed seed. The NSGA-II
  path uses `seed=run_id`, the candidate's index (`problem.py:448`), so its noise is tied to population
  position.
- **WP5.5 shorter simulations as low fidelity: (d)-like for Baltic.** Short horizons measure the seeding
  transient (see the `persists` note in `CLAUDE.md`).

**Phase 6 (§10) — not justified.**
- **The GPU is weak:** the box has a GT 1030 with 2 GB.
- **The kernel cannot be vectorised as is.** Within-cell mortality mutates `inst_abd` sequentially, in
  permuted order, so it cannot be vmapped without changing the algorithm.

**§11 research branch.** It is correctly labelled model-changing. §11.1's `expm1` alone would move every
fixed-seed baseline.

**§13 repository restructure.** Churn with no performance value.

## 3. Targets

- **2–5× single simulation: not supported.**
  - The shipped history: −41% on EEC (May, Python-side), 1.5–2× from the physical-core thread cap (July),
    and about 1.7× on Baltic from the school floor (November).
  - The September pass concluded "no cheap win left".
  - On today's Baltic profile, an infinitely fast mortality kernel gives about 2× at 14 threads. At one
    thread, 2× needs the kernel to be about 2.8× faster.
- **8–12× on 16 physical cores: the box has 14 physical cores** (28 logical, 31 GB).
  - No worker-scaling curve with a 1-worker baseline exists.
  - April data: 175 evaluations/h at 8 workers. The per-worker rate fell from 0.4 to about 0.35/min at 16
    workers and 0.13/min at 24 (memory bandwidth).
  - *Inference:* a 1-thread run is 2.2× slower than a 14-thread run, so 14 pinned workers are at best
    about 6× faster than threaded serial runs. 8–12× is plausible only against a 1-thread serial baseline.
- **5–20× fewer calibration evaluations: cannot be judged.** The only optimizer comparison is synthetic
  (`data/benchmarks/optimizer_comparison.json`), and there is no matched-budget comparison on a real OSMOSE
  objective.
- **A 1e-10 conservation residual per process: not defined for this model.**
  - With bioenergetics off, growth is length-based and gated by predation success (`growth.py:48-90`), so
    eaten mass does not become predator mass.
  - The floor and aging are deliberate sinks.
  - The checkable identity is death accounting: the change in abundance equals summed `n_dead`.

## 4. Where the time goes today

**Fresh measurement, 2026-11-14, this box, current master:** `scripts/profile_engine.py --config baltic
--years 10`, run three times in sequence (warm-up, warm, then `NUMBA_NUM_THREADS=1`).
- **Read shares, not absolute seconds.** Another project's pytest was running at the time (load
  1.6 → 3.0).
- **The auto thread count is 14** (`physical_budget()`, confirmed by a separate call).

| Phase | 14 threads (15.1 s) | 1 thread (33.4 s) |
|---|---|---|
| `_mortality` | 7.65 s, 50.8% | 26.16 s, 78.4% |
| `_movement` | 3.82 s, 25.4% | 3.74 s, 11.2% |
| `_collect_by_life_stage` | 0.60 s, 4.0% | 0.58 s, 1.8% |
| `_growth` | 0.39 s, 2.6% | 0.37 s, 1.1% |
| state init + loop overhead | 1.58 s, 10.5% | 1.50 s, 4.5% |
| `write_outputs()` | 0.07 s, 0.5% | 0.07 s, 0.2% |

- **Movement is serial and now a quarter of a threaded Baltic run.** Older profiles do not show this: May
  put EEC movement at 16.7% before A2, and September had 8.2% on EEC on a 4-core container.
- **The profile cannot see inside the kernels or count pairs per cell.** Baltic occupancy and selectivity
  are unmeasured; the September table covered EEC and Bay of Biscay only. Baltic runs `mortality.subdt = 10`.
- **From the old docs, not re-measured:** output writing is not a lever (1.5–6.4% in September), and the
  parallel kernel scales to about the physical-core count.

## 5. Corrected, prioritized plan

1. **Sweep workers × threads per worker for the process pools, then encode the winner in the worker
   initialisers.**
   - Expected gain: large for calibration and UQ throughput if the 4.4 h vs 50 min observation holds;
     unproven until swept.
   - Risk: none to results; mortality is bit-identical across thread counts
     (`tests/test_thread_policy.py:192`). The tests at `:237` and `:286` must change with the decision.
   - Verify: `scripts/benchmark_calibration.py --backend process`, evaluations/h at 1, 4, 8, 12, 14 and 16
     workers × 1, 2 and auto threads. One job at a time.
2. **Diagnose Baltic movement.**
   - Expected gain: bounded at 25% of a threaded run and 11% of a 1-thread run.
   - Risk: the map kernel draws from one seeded serial stream (`movement.py:470`), so parallelising it
     changes fixed-seed output. Python-side wrapper work is safe.
   - Verify: a warm cProfile first; gate with `tests/test_engine_parity.py` and
     `tests/test_perf_movement_diet_vectorization.py`.
3. **Add Baltic pair and occupancy counters through `step_observer`.** No engine change. This decides
   whether any mortality-kernel work is worth starting.
4. **Look at the serial tail.** Loop overhead plus `_collect_by_life_stage` is about 14% of a threaded
   Baltic run. The gain is modest and unmeasured; verify with an interleaved median-of-N A/B.
5. **Calibration: opt-in common-random-number seeding for NSGA-II, plus one matched-budget optimizer
   comparison on a real objective.** Changing the seed policy alters the reproducibility of past runs, so
   keep it opt-in.
6. **Single-thread mortality kernel for the worker regime: open and unmeasured.** The July closure of the
   native port was argued for multi-thread single runs. Do not start before items 1–3.

## 6. Errors and risky advice in the plan

- **The API examples are fictional.** `PythonEngine.__init__` takes only `backend`
  (`osmose/engine/__init__.py:35`); there is no `numerics_mode`, `performance_profile` or `run_objective`.
- **The 2-species minimal CI benchmark never predates** (`CLAUDE.md`), so it cannot detect a predation
  regression.
- **WP1.3's "positive accessibility" graph drops species missing from the matrix,** which the kernel
  treats as full access. The matrix is also stage-indexed and rescaled on every call.
- **"Five runs within 5%" and 10–15% CI regression gates are unrealistic.** Master's own EEC runs spread
  15% in September, and three measurement artifacts were caught in that pass. Use an interleaved A/B with
  medians. Note that `@pytest.mark.slow` does not gate CI.
- **WP3.2 suggests joblib/loky, and WP3.3 suggests `physical_cores − 2`.** The repo needed forkserver or
  spawn to avoid Numba deadlocks, and loky broke worker un-pickling once.
- **Adjacent seeds may share a stream** (inferred; effect unmeasured). `_prepare_run` seeds movement with
  `seed` and mortality with `seed + 1` (`__init__.py:87-90`), and replicates use consecutive seeds.
- **§12's 100-seed statistical tests are expensive.** The existing guard is 10 seeds on Bay of Biscay at
  rtol 0.05 (`tests/test_engine_parity.py:358-363`), and Bay of Biscay is a dying ecosystem past year 10.
- **The objective "compatibility with the Java reference" conflates two contracts.** Python-side fixed-seed
  runs are bit-exact; Java agreement is "within 1 OoM".
