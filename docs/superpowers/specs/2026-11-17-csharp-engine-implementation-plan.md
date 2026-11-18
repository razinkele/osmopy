# C# simulation engine: implementation plan

> Date: 2026-11-17. Companion to `2026-11-17-csharp-engine-design.md` ("the spec"); section
> numbers below refer to it. Nothing here has been started.

Each task is one reviewable PR. Sizes are relative: **S** (a day or two of focused work),
**M** (several days), **L** (a week or more). They are estimates, not measurements.

## Rules for every task

1. **Tests first, from Python.** A C# unit is accepted against a fixture the Python engine
   generated (spec 10.1). Fixtures are regenerated in CI, never committed as frozen numbers.
2. **Mutation-check each new test once** before trusting it: break the code the test guards and
   see it fail. Record the mutation in the PR description.
3. **Never use `data/minimal`** for anything that needs feeding. Use
   `tests/_tutorial_config.build_config`, EEC or Baltic.
4. **Fail closed.** A task that adds a feature also adds its capability bit; until then the
   config is refused.
5. **No benchmark while anything else runs on the machine.** Interleaved pairs, medians, at
   least seven repeats (the performance plan's A/B protocol).
6. **Python-side tests of the C# engine are gated on `OSMOSE_REQUIRE_CSHARP`**: skip with a
   reason when unset and the engine is unavailable, fail when set.
7. **A PR that changes Python engine semantics** must either update the C# engine or extend
   `csharp_engine_block_reason`. (To be added to CLAUDE.md in task 1.13.)

---

## Phase 0. Spikes and foundations (go/no-go)

Nothing in Phase 1 starts until gate G0 is decided by the owner.

| # | Task | Size | Depends on |
|---|---|---|---|
| 0.1 | **Toolchain and skeleton.** `csharp/` solution with the engine, test and benchmark projects; `RunSimulation` throws `NotImplementedException`; `scripts/build_csharp_engine.sh`; `.gitignore` for `osmose/engine_cs/bin`; the `csharp-engine` CI job running format, build and an empty `dotnet test`; developer-setup section in the docs. | S | owner's answer to Q1 |
| 0.2 | **Bridge spike.** A throwaway method in the skeleton plus `tests/test_csharp_bridge_spike.py`. | S | 0.1 |
| 0.3 | **Bundle container.** `SimulationConfig` (set, get, validate names and shapes), the flat file format with `Save`/`Load`, and the Python `Bundle` writer and reader. No model content. | M | 0.1 |
| 0.4 | **Fixture capture.** `scripts/csharp/make_fixtures.py`: run the Python engine to a chosen step on EEC and Baltic, capture the inputs of one mortality step and, with injected permutations and cause orders, the outputs of `_mortality_in_cell_numba` for every cell. Written as bundle files. | M | 0.3 |
| 0.5 | **Philox and draws.** `Philox.Block`, bounded integers, uniform doubles, Fisher-Yates, cause-order decoding, the `IDrawSource` implementations (Philox and injected). | S | 0.1 |
| 0.6 | **Kernel spike.** `SchoolStore` with the split layout and `Regroup`; `MortalityKernel.RunCell` for bioen off, stage accessibility, type I response; the mask prefilter; a BenchmarkDotNet harness; a Python timing script for the Numba kernel on the same captured step. | L | 0.4, 0.5 |

**Acceptance, Phase 0.**

- 0.2, all on the dev box under `OSMOSE_REQUIRE_CSHARP=1`:
  1. pythonnet 3.2.1 loads the .NET 10 runtime through `clr_loader` and calls the method.
  2. A NumPy array's address passed as `Int64` is read and modified in place; the change is
     visible in Python with no copy.
  3. The GIL is released during the call: a Python thread increments a counter at least 100
     times while a 2-second C# call runs.
  4. Writing the control block from Python stops a running C# loop within one iteration.
  5. A .NET exception arrives as a Python exception carrying its message.
  6. Two Python threads call the method at once and both return correct results.
  7. Importing `osmose.engine_cs` does not load `clr`.
- 0.3: a bundle written by Python is read by C# and written back byte-identical; a bundle with
  a wrong dtype, shape or version is rejected with every problem listed.
- 0.5: Random123 known-answer vectors pass; the NumPy call that reproduces a block is
  documented, or replaced by a pure-Python Philox in the tests; every permutation of four
  elements and every cause order appears with the expected frequency (chi-square, fixed
  counters, so the test is deterministic).
- 0.6:
  1. `Math.Exp` is compared with Numba's `exp` on a few million arguments in the engine's range
     (the fixture carries Numba's values). The result, identical or not, is recorded and sets
     the tolerance of check 3.
  2. `RunCell` with injected draws and **zero** starvation, additional and fishing rates equals
     the Python oracle bit for bit on every captured cell (tier E: no `exp` is evaluated).
  3. `RunCell` with injected draws and the real rates agrees with the oracle as a whole: at
     `atol=0` if check 1 found the two `exp` identical, otherwise at tier L with the worst
     error recorded. The comparison is not split into "predation" and "`exp`" parts; they are
     coupled through the abundance (spec 6.1).
  4. Results are identical with the mask prefilter on and off.
  5. One-thread throughput on the captured Baltic step, interleaved against Numba: reported
     for four variants (regrouped or indexed layout, mask prefilter or candidate lists).
- **Deliverable:** `docs/perf/<date>-csharp-engine-spikes.md` with the bridge checklist, the
  four kernel numbers and a recommendation.

**Gate G0: decided GO by the owner on 2026-11-18.** The kernel ran 2.15–2.37x production Numba and was bit-exact against the oracle (`docs/perf/2026-11-18-csharp-engine-spikes.md`).

**Gate G0 (owner).** Go if the bridge passes and the best kernel variant is at least 1.0x
Numba. Stop if the bridge cannot release the GIL or share buffers and the owner does not want
the NativeAOT fallback, or if the kernel is below 0.8x after one tuning pass. Between 0.8x and
1.0x: owner's call (spec 7.2, Q2).

---

## Phase 1. MVP: EEC full and Bay of Biscay, bioen off, minimal outputs

| # | Task | Size | Depends on |
|---|---|---|---|
| 1.1 | **Parameters and capabilities.** Typed `EngineParameters` built and validated once from a `SimulationConfig` (header groups); the feature catalogue `osmose/engine_cs/features.json` (one source, embedded in C#); `UnsupportedFeatureException`; the build's `capabilities.json` copy with a stale-build check; Python `csharp_engine_block_reason` with per-feature detectors **and key coverage**, so any engine-read key no rule classifies blocks the run (spec 8.6). | M | 0.3 |
| 1.2 | **Bundle builder, MVP scope.** `osmose/engine_cs/bundle.py::build_bundle`: grid, species, growth, predation and stage tables, stage accessibility with -1 preserved, per-step rate tables, fishing (rate, seasonality, age selectivity, spatial maps, discards), reproduction, resources (regridded frames, `frame_by_step`), movement maps and index tables, random-movement domains, output settings, result buffers, control block. Pure Python; tested in the main CI job. | L | 1.1 |
| 1.3 | **School store.** Append, id assignment, capacity growth, `Regroup` (compaction, cell sort, tail segment, length lane), pooled buffers. | M | 0.6 |
| 1.4 | **Per-step preparation.** Resource update (non-depletable), per-school rates, feeding stages, accessibility indices, egg retention. | M | 1.2, 1.3 |
| 1.5 | **Mortality step.** Larva pre-pass, the cell loop on one thread, commit, next starvation rate, trophic level; the death-accounting check behind `run.checks`. | M | 1.4 |
| 1.6 | **Movement.** Maps (placement by rejection, same-map walk, null map), random (domain placement, walk). One thread. | M | 1.3, 0.5 |
| 1.7 | **Life cycle.** Floor, von Bertalanffy growth, aging, reproduction (season, seeding, egg schools, age increment). | M | 1.3 |
| 1.8 | **End to end.** The step pipeline, minimal outputs, record window, initial record, control block, deadline, `RunSummary`; Python `bridge.py` and `CSharpEngine.run` / `run_in_memory`; `StepOutput` construction from the result buffers. | L | 1.5, 1.6, 1.7 |
| 1.9 | **Parallelism.** Cost-ordered cell scheduler, parallel movement, chunked reductions, per-worker scratch. | M | 1.8 |
| 1.10 | **Ensemble harness.** A `csharp` arm and a Python-versus-Python control arm in `scripts/cross_engine_parity_440.py`; the acceptance rule of spec 10.2 as code with unit tests; the EEC and Bay of Biscay runs. | M | 1.9 |
| 1.11 | **Performance, EEC.** `--engine` in `scripts/benchmark_engine.py` and `scripts/profile_engine.py`; the measurements of spec 10.4 for EEC; phase timings from `RunSummary`. | S | 1.9 |
| 1.12 | **Selection.** `osmose/engine/select.py`; `OsmoseCalibrationProblem(engine=...)` with `use_java_engine` as an alias; the engine in `_cache_key`; `--engine` in `calibrate_baltic.py` and `osmose run`. | M | 1.8 |
| 1.13 | **Packaging and deployment.** The `csharp` extra; `deploy.sh`, `Dockerfile`, `apptainer/osmose.def`; the post-deploy self-test; DEPLOY.md and CLAUDE.md entries. | M | 1.8 |

**Acceptance per task (the test that must exist and pass).**

- 1.1: each of the three demo configs is classified correctly for the current phase; switching
  on each out-of-scope feature in the EEC config yields a block reason naming it; the C# and
  Python feature lists are generated from one source and a test fails if they differ.
- 1.2: for EEC and Bay of Biscay, every bundle table equals what the Python engine itself would
  read at each step, for example `fishing.rate_by_step` and `season_by_step` reproduce
  `_precompute_effective_rates`' inputs, and the map index tables reproduce
  `_precompute_map_indices`. A species absent from the accessibility CSV produces an empty
  range, not a zero row.
- 1.3: a regroup of a random store preserves the multiset of live schools, drops exactly the
  dead ones, leaves every cell contiguous and keeps the lane equal to `Core.Length`; ten
  thousand append/regroup cycles allocate nothing after warm-up.
- 1.4: rates, stages and indices equal the Python fixtures at tier E on captured EEC states.
- 1.5: one whole mortality step with injected draws equals Python's on EEC and on the tutorial
  config, as one comparison at the tolerance Phase 0 established (spec 6.1); the larva pre-pass
  and the rate tables come from NumPy and are tier L; death accounting holds for every school.
- 1.6: map index, same-map flag, null-map handling and the accessible set equal the fixtures at
  tier E; placement frequencies match the map (chi-square); no school lands on land.
- 1.7: growth and reproduction equal `growth` and `reproduction` at tier L; egg-school counts,
  the seeding rule (`SSB == 0.0` exactly) and the aging threshold are tier E.
- 1.8: with `output.minimal.enabled = true`, `CSharpEngine().run_in_memory(cfg)` on Bay of
  Biscay (1 year) and EEC (2 years) returns
  an `OsmoseResults` whose `biomass()`, `abundance()`, `yield_biomass()` and `mortality()`
  frames match the Python engine's in shape, columns and dtypes; `run()` writes the same file
  set; cancel and deadline stop within one step and leave the written records valid; the stock
  configs **without** `output.minimal.enabled` are refused with a message naming the full
  output families; a `KeyboardInterrupt` or `_SimTimeout` raised in the waiting thread is
  re-raised only after the worker thread has joined (the buffer-lifetime invariant, spec 8.5).
- 1.9: result buffers byte-identical at 1, 2 and 14 threads on both configs; repeat runs
  identical; different seeds differ; steady-state allocation zero at one thread.
- 1.10: the acceptance rule's unit tests pass on synthetic arms (identical, shifted, noisy);
  the rule's false-fail rate is estimated by re-splitting the pooled Python replicates and is
  at most 5% before the C# arm is compared (spec 10.2); the two ensembles are run and written
  up.
- 1.12: the calibration problem runs one generation with `engine="csharp"` on the forkserver
  backend with one thread per worker; a cached Python result is not served to a C# request.
- 1.13: the self-test passes in the Docker image and the Apptainer image in CI.

**Acceptance, Phase 1 as a whole** (spec 7.2): process tests green; determinism; death
accounting; ensemble equivalence on EEC and Bay of Biscay by the rule fixed in advance;
out-of-scope configs refused; the EEC performance gate.
**Deliverables:** `docs/csharp_cross_check_<date>.md` and `docs/perf/<date>-csharp-eec.md`.

**Gate G1 (owner).** Proceed to Baltic if equivalence holds and the EEC gate (no regression at
1 and 14 threads) is met.

---

## Phase 2. Baltic

| # | Task | Size | Depends on |
|---|---|---|---|
| 2.1 | **Background species.** Bundle entries (classes, per-step proportions, uniform or forced biomass); injection in `Regroup`; background outputs recorded before removal. | M | G1 |
| 2.2 | **Out-of-domain mortality and `lmax`.** Null maps set `IsOut`; out mortality; the length cap. | S | G1 |
| 2.3 | **Recruitment.** The four stock-recruitment curves, `seeding_mode`, the RV gate table. | S | G1 |
| 2.4 | **Salinity movement gate.** Weighted placement and walk; the per-(map, frame) maximum computed once instead of once per school. | M | G1 |
| 2.5 | **Oxygen to benthos.** The pre-evaluated factor table applied to the named resource's carrying capacity. | S | G1 |
| 2.6a | **Outputs: life-stage split and per-step rates**, including the `inf` on an emptied stage. | M | G1 |
| 2.6b | **Outputs: by-age and by-size distributions** (no cutoff), SSB, mean size, mean TL, yield in numbers. | M | G1 |
| 2.6c | **Outputs: diet and predator pressure**, including background predators' rows. | M | 2.1 |
| 2.7 | **UI.** Third `engine_mode` value; every `!= "python"` test replaced by an explicit predicate; the run-page branch; availability and block reason shown; live pane disabled for the C# engine. It waits for 2.6 because Phase 1 produces minimal outputs only and the Results pages would be nearly empty. | M | 2.6a-c, 1.12 |
| 2.8 | **Baltic validation.** The ensemble (16 replicates, 15 years); `scripts/baltic_stability_certify.py` with an engine option, compared verdict by verdict with Python on the same seeds; the Baltic performance rows. | M | 2.1-2.6 |

**Acceptance per task.** Each feature has a process fixture against its Python function
(`BackgroundState.get_schools`, `out_mortality`, `apply_stock_recruitment` and
`regulate_recruitment`, the gated branch of `_map_move_batch_numba`, `ResourceState.update`,
`_collect_by_life_stage` and `mortality_rates_from_counts`, the distribution collectors,
`aggregate_diet_by_species`), and an A/B: with the feature absent from the config, output is
byte-identical to the build before the PR. For 2.7: a test per page that each predicate gives
the intended answer for all three modes.

Specific traps to test, all from CLAUDE.md:

- 2.1: background biomass is added to the output **without** the cutoff; background schools take
  no starvation, additional or fishing mortality; they feed from age "-1".
- 2.3: gates are skipped on a seeded step; the order is curve, RV, ceiling, thermal,
  depensation.
- 2.5: the forcing must carry exactly `n_dt_per_year` frames; the bundle builder raises
  otherwise, as `_load_oxygen_data` does.
- 2.6a: an emptied stage reads `inf`, and the window value is a sum of per-step rates.
- 2.6b: the distributions include eggs and so legitimately disagree with `abundance()`.
- 2.6c: resources come after focal species on the prey axis; background predators have rows.

**Acceptance, Phase 2 as a whole** (spec 7.2): process tests; determinism at 1, 2 and 14
threads; ensemble equivalence on Baltic; certification verdicts equal to Python's or every
difference explained; the Baltic performance gates (1.15x at one thread, 1.5x at 14).
**Deliverables:** `docs/baltic_csharp_crosscheck_<date>.md`, `docs/perf/<date>-csharp-baltic.md`.

**Gate G2 (owner).** Decide whether calibration may use the C# engine (Q8) and whether
bioenergetics is worth porting (Q5).

---

## Phase 3. Optional features, one PR each, in any order

Each: bundle entries, capability bit, process fixture, the off-is-identical A/B, and a small
ensemble on the demo variant that uses it (`osmose/demo.py`).

| # | Feature | Size |
|---|---|---|
| 3.1 | Depletable LTL with logistic regrowth | S |
| 3.2 | Incoming flux | S |
| 3.3 | MPAs | S |
| 3.4 | By-year fishing rate, by-step additional and larva rates, additional-mortality maps | S |
| 3.5 | Logistic selectivity | S |
| 3.6 | Ceiling, thermal and depensation gates; spatial RV egg survival | M |
| 3.7 | Type II and III functional response | S |
| 3.8 | Species-level accessibility matrix; dynamic accessibility | M |
| 3.9 | Gompertz growth; feeding stages by age, weight and trophic level | S |
| 3.10 | Warm-start initial population passed as `state.*` | S |
| 3.11 | Spatial outputs; carbon flux | M |
| 3.12 | Snapshot buffer for a live map (only if Q10 is answered yes) | M |

---

## Phase 4. Bioenergetics

Not planned in detail. First task: a design note that reads `_bioen_step`,
`_bioen_reproduction` and the `bioen_*` process modules (none of which the spec author read)
against CLAUDE.md's bioen notes, and lists the semantics with tiers as spec section 6 does.
Only then size the work. Entry criterion: gate G2 and a "yes" to Q5.

Genetics and economics are not planned (spec 7.1).

---

## Order at a glance

- Phase 0: 0.1 first. Then 0.2, 0.3 and 0.5 in parallel; 0.4 after 0.3; 0.6 after 0.4 and 0.5. Gate G0.
- Phase 1: 1.1, then 1.2. 1.3 can start with 1.1. 1.4 needs 1.2 and 1.3; 1.5 needs 1.4; 1.6 and
  1.7 need 1.3. 1.8 joins 1.5, 1.6 and 1.7. Then 1.9, then 1.10 and 1.11. 1.12 and 1.13 can
  follow 1.8 at any time. Gate G1.
- Phase 2: 2.1 to 2.6b in parallel; 2.6c after 2.1; 2.7 after the 2.6 tasks; 2.8 last. Gate G2.
- Phase 3 in any order. Phase 4 only if chosen.

## Questions that block a task

| Task | Needs |
|---|---|
| 0.1 | Q1, the .NET version |
| G0 | Q2, whether a failed kernel gate ends the project |
| 1.1 | Q4, refuse or imitate Python's ignored features |
| 1.10 | Q11, the equivalence margins, fixed before the first ensemble run |
| 1.13 | Q6, build on the server or ship an artifact |
| G2 | Q5 and Q8 |
