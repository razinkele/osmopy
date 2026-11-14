# Osmopy Performance Optimization Implementation Plan

## 1. Objective

Improve Osmopy performance without compromising:

- compatibility with the Java-compatible reference engine;
- reproducibility of stochastic simulations;
- biomass and abundance conservation;
- ecological validity;
- existing calibration and sensitivity workflows.

### Target outcomes

| Outcome | Target |
|---|---:|
| Single-simulation speedup | **2–5×** |
| Ensemble speedup on 16 physical cores | **8–12× wall-clock** |
| Reduction in full calibration evaluations | **5–20×** |
| Deterministic biomass residual | Below **$10^{-10}$** relative per process where conservation is expected |
| Fixed-seed regression error | Zero where event order is unchanged; otherwise within documented tolerance |
| Statistical parity | No material change in ensemble means, variances, quantiles or collapse probabilities |

The work should preserve the current implementation as a reference path:

```python
engine = PythonEngine(
    numerics_mode="legacy",
    performance_profile="reference",
)
```

Optimizations should initially be opt-in:

```python
engine = PythonEngine(
    numerics_mode="legacy",
    performance_profile="optimized",
)
```

---

# 2. Guiding principles

## 2.1 Optimize algorithms before changing technology

Priority order:

1. Reduce unnecessary predator–prey comparisons.
2. Improve state layout and memory access.
3. Fuse repeated loops.
4. Parallelize independent simulations.
5. Reduce the number of simulations needed for calibration.
6. Only then consider a JAX/GPU backend.

Moving an inefficient all-pairs algorithm to a GPU should not be the first intervention.

## 2.2 Separate performance from numerical reformulation

Changes should be classified as:

### Performance-preserving

Expected to reproduce the existing algorithm:

- cell/species sorting;
- sparse candidate generation;
- structure-of-arrays storage;
- fused Numba kernels;
- parallel ensemble execution;
- output suppression.

### Statistically equivalent

May alter individual stochastic trajectories but should preserve output distributions:

- normal or Poisson approximations;
- reordered vectorized random draws;
- batch processing;
- alternative parallel random streams.

### Model-changing

Require separate scientific validation:

- balanced predation allocation;
- Strang splitting;
- deterministic relaxation;
- adaptive process scheduling;
- differentiable approximations.

Only performance-preserving work belongs in the first production release.

---

# 3. Delivery roadmap

| Phase | Duration | Principal result |
|---|---:|---|
| 0. Baseline and instrumentation | 1–2 weeks | Reproducible benchmark suite |
| 1. Sparse predation search | 2–4 weeks | Fewer predator–prey candidate pairs |
| 2. State and kernel optimization | 3–5 weeks | Contiguous arrays and fused kernels |
| 3. Ensemble parallelism | 1–3 weeks | Multi-core calibration and ensembles |
| 4. Hybrid stochastic sampling | 2–3 weeks | Cheaper mortality and recruitment draws |
| 5. Calibration efficiency | 3–5 weeks | Fewer complete model evaluations |
| 6. Optional JAX/GPU prototype | 6–12 weeks | Batched accelerator backend |
| 7. Modern numerical methods | research branch | Improved splitting and predation mathematics |

Phases 1–5 can partly overlap after Phase 0 establishes stable benchmarks.

---

# 4. Phase 0: baseline and instrumentation

## Goal

Determine where simulation time is actually spent and establish non-negotiable numerical baselines.

## Tasks

### 0.1 Create representative benchmark cases

Use four configurations:

1. **Minimal smoke model**
   - 2 species;
   - short run;
   - intended for continuous integration.

2. **Baltic configuration**
   - 8 focal species plus lower-trophic-level forcing;
   - representative scientific benchmark.

3. **Eastern English Channel**
   - 14 species;
   - larger interaction network.

4. **Calibration benchmark**
   - 20 parameter candidates;
   - 5 seeds each;
   - no complete simulation output files;
   - objective values returned in memory.

Osmopy already supports a pure Python NumPy/Numba engine and an original Java-engine execution route. These provide the two main parity references.

### 0.2 Add benchmark commands

Proposed interface:

```bash
osmose benchmark \
    data/baltic/baltic_all-parameters.csv \
    --engine python \
    --years 20 \
    --warmup 1 \
    --repetitions 5 \
    --seed 42 \
    --output benchmark-baseline.json
```

Separate compilation from execution time. The first Numba run should be reported as warm-up and excluded from steady-state timing.

### 0.3 Profile the engine

Use:

- `py-spy record` for low-overhead call-stack sampling;
- `Scalene` for CPU, memory and copying attribution;
- `line_profiler` for selected Python functions;
- `time.perf_counter_ns()` for stable internal timing;
- Numba diagnostics for object-mode fallbacks and inferred signatures.

Example:

```bash
py-spy record \
    -o osmopy-baltic-profile.svg \
    -- python scripts/run_baltic.py
```

### 0.4 Add process timers

Collect time for:

```text
configuration and initialization
movement
growth
predation candidate generation
predation allocation
natural mortality
fishing
starvation
reproduction
school creation/deletion/compaction
aggregation
output writing
```

### 0.5 Add numerical diagnostics

For each process and timestep, optionally collect:

```python
@dataclass
class ProcessDiagnostics:
    biomass_before: float
    biomass_after: float
    external_input: float
    external_output: float
    residual: float
    schools_processed: int
    candidate_pairs: int
    accepted_pairs: int
    random_draws: int
```

Conservation residual:

$$
R=B_{\mathrm{after}}-B_{\mathrm{before}}-I+O.
$$

Normalized residual:

$$
R_{\mathrm{rel}}
=
\frac{|R|}
{\max(|B_{\mathrm{before}}|,10^{-15})}.
$$

### 0.6 Record reproducibility metadata

Every benchmark should store:

- commit hash;
- Python version;
- NumPy version;
- Numba version;
- operating system;
- CPU model;
- thread counts;
- seed;
- configuration checksum;
- compilation and execution times separately.

## Exit criteria

- Five repeated steady-state runs vary by less than **5%**.
- At least **90%** of runtime is attributed to named processes.
- Java and Python reference outputs are archived.
- Fixed-seed Python output is reproducible.
- Baseline candidate-pair and school-count statistics are available.

## Deliverables

```text
benchmarks/
  configs/
  baseline/
  benchmark_engine.py
  compare_benchmarks.py
docs/performance/
  baseline.md
  methodology.md
```

---

# 5. Phase 1: sparse, cell-sorted predation

## Goal

Reduce predation complexity without changing the ecological interaction rules.

## Expected gain

- **3–20×** faster predation kernel;
- approximately **2–4×** faster complete simulation if predation consumes 60–85% of runtime.

## Work package 1.1: quantify current search efficiency

Add counters:

```python
n_possible_pairs
n_cell_matched_pairs
n_species_accessible_pairs
n_size_accessible_pairs
n_accepted_pairs
```

Calculate:

$$
\eta
=
\frac{N_{\mathrm{accepted}}}
{N_{\mathrm{considered}}}.
$$

If $\eta<0.1$, sparse filtering has high potential.

## Work package 1.2: create compact interaction keys

Construct:

$$
k_i=
\mathrm{cell}_iN_{\mathrm{species}}
+
\mathrm{species}_i.
$$

Stable-sort schools using `np.argsort(..., kind="stable")`.

Create:

```python
sorted_indices
cell_starts
cell_ends
cell_species_starts
cell_species_ends
```

The sort should happen no more often than necessary. If movement occurs once per global timestep, sort schools after movement and reuse the ordering through subsequent local processes.

## Work package 1.3: precompute admissible species pairs

Represent the species interaction graph as compact arrays:

```python
prey_offsets[predator_species]
prey_species_flat[:]
```

For each predator species, store only prey species having positive accessibility.

Also store:

```python
min_ratio[predator_species, prey_species]
max_ratio[predator_species, prey_species]
accessibility[predator_species, prey_species]
```

## Work package 1.4: filter by size before expensive calculations

For each candidate predator–prey block:

1. select same-cell schools;
2. select admissible species;
3. filter by size ratio;
4. only then calculate ingestion, preference and energetic constraints.

Avoid computing detailed energetics for pairs already excluded by species, space or size.

## Work package 1.5: implement reference and optimized paths

```python
def run_predation(state, config, mode):
    if mode == "legacy":
        return predation_legacy(state, config)

    if mode == "sparse":
        return predation_sparse(state, config)

    raise ValueError(f"Unknown predation mode: {mode}")
```

## Tests

### Exact tests

- candidate pairs match the legacy implementation;
- accepted predator–prey relationships match;
- school-level biomass transfers match;
- no prey school loses more biomass than available;
- no predator consumes more than its demand.

### Edge cases

- empty cell;
- predator without accessible prey;
- prey with zero abundance;
- schools exactly at size-ratio thresholds;
- one predator and many prey;
- many predators and one prey;
- all schools in one cell;
- maximum configured school count.

## Exit criteria

- Candidate-pair evaluations reduced by **at least 70%** in one realistic model.
- Predation kernel at least **2× faster**.
- Complete simulation at least **1.5× faster**.
- Fixed-seed outputs unchanged, or every difference explained by an intentionally changed iteration order.
- Conservation residual not worse than baseline.

---

# 6. Phase 2: structure-of-arrays and fused kernels

## Goal

Reduce cache misses, temporary allocations and repeated state traversal.

## Expected gain

- **1.5–3×** for affected kernels;
- **1.2–2×** additional complete-model speedup.

## Work package 2.1: define numerical state arrays

Use a structure-of-arrays representation:

```python
@dataclass
class SchoolState:
    species: np.ndarray       # int16 or int32
    cell: np.ndarray          # int32
    age: np.ndarray           # float32 or float64
    length: np.ndarray        # float32 or float64
    weight: np.ndarray        # float32 or float64
    abundance: np.ndarray     # float64
    biomass: np.ndarray       # float64
    energy: np.ndarray        # float32 or float64
    alive: np.ndarray         # bool
```

Use `float64` initially for all ecological quantities. Introduce mixed precision only after correctness is established.

## Work package 2.2: eliminate hot-loop Python objects

The compiled numerical kernel must not:

- access dictionaries;
- construct dataclasses;
- allocate lists;
- catch exceptions;
- create temporary Python objects;
- perform file output.

Configuration data required by kernels should be converted into typed numeric arrays before simulation begins.

## Work package 2.3: fuse compatible operations

Candidate fused kernel:

```text
calculate natural mortality hazard
calculate fishing mortality hazard
combine competing risks
sample removals
update abundance
update biomass
accumulate catch
accumulate natural deaths
write process diagnostics
```

This replaces multiple full-array passes.

## Work package 2.4: control allocations

Preallocate reusable work arrays:

```python
workspace.predator_demand
workspace.prey_supply
workspace.mortality_probability
workspace.removals
workspace.random_buffer
workspace.candidate_indices
```

No array proportional to the number of schools should be allocated inside the main timestep loop unless its size genuinely changes.

## Work package 2.5: school compaction policy

Deleting schools on every process can be expensive. Use an active mask and compact at controlled intervals:

```python
if dead_fraction > 0.10 or steps_since_compaction >= 12:
    compact_state()
```

Benchmark thresholds of 5%, 10% and 20%.

## Tests

- state conversion round trip;
- no state field lost during compaction;
- stable school identifiers where required;
- no stale indices after sorting or compaction;
- biomass totals preserved;
- fixed-seed behavior preserved where event order is unchanged.

## Exit criteria

- No Numba object-mode execution in production kernels.
- Temporary allocated bytes per timestep reduced by **at least 70%**.
- Fused mortality/fishing kernel at least **1.5× faster**.
- Combined Phases 1–2 provide at least **2× complete-model speedup** on the Baltic benchmark.

---

# 7. Phase 3: ensemble and calibration parallelism

## Goal

Exploit independent candidates and random seeds across physical CPU cores.

## Expected gain

- **6–7×** on 8 physical cores;
- **10–14×** on 16 physical cores;
- limited primarily by initialization, memory bandwidth and I/O.

## Work package 3.1: calibration execution mode

Create an in-memory mode that does not write detailed outputs:

```python
result = engine.run_objective(
    config=config,
    parameters=candidate,
    seed=seed,
    targets=targets,
)
```

Return only:

```python
{
    "objective": ...,
    "component_losses": ...,
    "diagnostics": ...,
    "status": ...,
}
```

## Work package 3.2: process-based worker pool

Use one task per candidate–seed combination:

```python
tasks = [
    EvaluationTask(candidate_id=i, seed=seed)
    for i in range(n_candidates)
    for seed in seeds
]
```

Use `concurrent.futures.ProcessPoolExecutor`, `joblib` with the `loky` backend, or the parallel interface already supported by the calibration framework.

## Work package 3.3: prevent thread oversubscription

Worker environment:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
NUMBA_NUM_THREADS=1
```

Default worker count:

```python
workers = max(1, physical_cores - 2)
```

Do not default to logical CPU count for memory-intensive simulations.

## Work package 3.4: share immutable initialization

Avoid reparsing and rebuilding all configuration objects for every task.

Possible approaches:

- initialize model configuration once per worker;
- pass only parameter overrides and seed;
- memory-map large read-only forcing arrays;
- cache compiled Numba functions before launching the main workload.

## Work package 3.5: deterministic seed hierarchy

Use `numpy.random.SeedSequence`:

```python
root = np.random.SeedSequence(base_seed)
candidate_sequences = root.spawn(n_candidates)
seed_sequences = candidate_sequences[i].spawn(n_replicates)
```

Store enough metadata to recreate each random stream independently of scheduling order.

## Scaling benchmark

Test:

```text
1 worker
2 workers
4 workers
8 workers
12 workers
16 workers
```

Report:

$$
S_p=\frac{T_1}{T_p},
\qquad
E_p=\frac{S_p}{p}.
$$

## Exit criteria

- At least **70% parallel efficiency** at 8 workers.
- At least **60% parallel efficiency** at 16 workers.
- Results independent of task completion order.
- Seed-to-result mapping is reproducible.
- Calibration mode writes no large per-run output unless requested.

---

# 8. Phase 4: hybrid stochastic sampling

## Goal

Reduce stochastic-event costs for very large schools while maintaining appropriate means and variances.

## Expected gain

- **1.2–3×** in mortality/recruitment kernels;
- overall gain depends on the fraction of runtime spent sampling.

## Work package 4.1: preserve exact binomial reference

```python
deaths = rng.binomial(abundance, probability)
```

This remains the reference for integer abundance.

## Work package 4.2: implement selectable sampling modes

```python
sampling_mode = "exact"  # reference
sampling_mode = "hybrid"
sampling_mode = "deterministic"
```

Initial hybrid policy:

| Condition | Sampler |
|---|---|
| $N<1{,}000$ | Binomial |
| $N\ge1{,}000$ and $Np<30$ | Poisson |
| $N\ge1{,}000$ and $Np\ge30$ | Normal approximation |
| Very large $N$, optional low-noise mode | Deterministic expectation |

All approximations must clamp:

$$
0\le D\le N.
$$

## Work package 4.3: protect sensitive populations

Always use exact sampling for:

- depleted populations;
- threatened species;
- recruitment bottlenecks;
- populations close to extinction;
- groups below a configurable abundance threshold.

## Validation

For each $N,p$ combination, run at least 100,000 draws and compare:

$$
E[D]=Np,
\qquad
\operatorname{Var}(D)=Np(1-p).
$$

Acceptance thresholds:

- mean relative error below **0.5%**;
- variance relative error below **2%**;
- tail-probability errors evaluated separately for extinction-sensitive cases.

## Exit criteria

- Kernel speedup at least **1.5×** for abundant-school benchmarks.
- Ensemble means and standard deviations remain within predefined statistical tolerance.
- Exact mode remains available for reference and publication runs.

---

# 9. Phase 5: calibration efficiency

## Goal

Reduce the number of complete Osmopy runs needed for calibration.

Osmopy already contains modules for CMA-ES, `pymoo`, Gaussian-process surrogates, surrogate-assisted differential evolution, multiseed evaluation, Sobol analysis, stability penalties and Bayesian uncertainty quantification. The implementation priority is to connect these components into a default staged workflow.

## Work package 5.1: screening before calibration

Use Morris screening for broad parameter sets.

Suggested design:

- 10–20 trajectories;
- 4–8 levels;
- common seed set;
- retain parameters with material $\mu^*$ or $\sigma$.

SALib supports Morris, Sobol, FAST, PAWN, DGSM and other global sensitivity methods.

## Work package 5.2: common random numbers

Evaluate every candidate in a calibration generation on the same seeds:

```python
calibration_seeds = [11, 37, 58, 91, 133]
```

Change the common seed set between independent calibration experiments, not between candidates within one experiment.

## Work package 5.3: staged seed allocation

Use:

| Calibration stage | Seeds per candidate |
|---|---:|
| Initial screening | 1–3 |
| Surrogate search | 3–5 |
| Candidate re-ranking | 10–20 |
| Final validation | 30–100 |

Do not evaluate poor candidates with 30 seeds.

## Work package 5.4: active surrogate workflow

For $D$ retained parameters:

1. generate $10D$–$20D$ Sobol or Latin-hypercube points;
2. evaluate with 2–3 common seeds;
3. fit a Gaussian-process or Extra Trees surrogate;
4. propose batches of 8–32 candidates;
5. evaluate the real model;
6. retrain;
7. stop when improvement or uncertainty reduction becomes negligible.

## Work package 5.5: multi-fidelity objectives

During search, use cheaper evaluations:

- shorter simulation;
- fewer seeds;
- reduced spatial output;
- fewer objective components;
- lower reporting frequency.

Finalists must be re-evaluated with the full configuration.

## Exit criteria

- At least **5× fewer full-resolution simulations** than baseline evolutionary calibration.
- Best candidates remain competitive after 30-seed validation.
- Calibration history records surrogate predictions and actual results.
- Failed or unstable runs are handled consistently and penalized explicitly.

---

# 10. Phase 6: optional JAX/GPU prototype

## Goal

Assess whether batched ensembles benefit from accelerator execution.

## Entry conditions

Do not start until:

- state is already stored as arrays;
- school capacity can be bounded or preallocated;
- sparse interaction generation is stable;
- the CPU implementation has reached at least 2× baseline speed;
- a real workload includes at least dozens of simulations per batch.

A JAX-based agent-model architecture can provide JIT compilation, vectorization and accelerator execution, but dynamic creation and deletion of schools must be reformulated using fixed arrays and masks.

## Prototype scope

Implement only:

1. growth;
2. natural mortality;
3. fishing;
4. simplified movement;
5. one predation kernel;
6. batching across seeds with `vmap`;
7. timesteps with `jax.lax.scan`.

## State design

```python
state = {
    "species": int_array,
    "cell": int_array,
    "age": float_array,
    "length": float_array,
    "abundance": float_array,
    "biomass": float_array,
    "active": bool_array,
}
```

Use a fixed-capacity school pool with inactive slots.

## Go/no-go criteria

Proceed only if:

- GPU batch of 64 simulations is at least **3× faster** than optimized multi-core CPU;
- compilation cost is amortized in a realistic workflow;
- memory usage remains manageable;
- ecological outputs pass parity tests;
- implementation complexity does not compromise the reference engine.

Otherwise, retain Numba CPU as the primary backend.

---

# 11. Research branch: modern numerical reformulation

These changes should not block performance work.

## 11.1 Competing-risk mortality

Combine natural and fishing mortality:

$$
Z=M+F,
$$

$$
D=N\left(1-e^{-Z\Delta t}\right),
$$

$$
D_M=D\frac{M}{Z},
\qquad
D_F=D\frac{F}{Z}.
$$

Use `numpy.expm1`:

```python
removed = abundance * (-np.expm1(-z * dt))
```

This can eliminate order bias and combine two update passes.

## 11.2 Strang splitting

Evaluate smooth process groups symmetrically:

$$
X^{n+1}
=
A_{\Delta t/2}
B_{\Delta t}
A_{\Delta t/2}
X^n.
$$

This is an accuracy feature and may increase runtime.

## 11.3 Balanced predation

Formulate predation as a constrained allocation:

$$
0\le x_{ij},
$$

$$
\sum_jx_{ij}\le D_i,
$$

$$
\sum_ix_{ij}\le S_j.
$$

This removes predator-order dependence but changes the numerical model. It therefore requires an independent configuration mode and ecological validation.

---

# 12. Test strategy

## Unit tests

Test:

- key construction;
- stable sorting;
- cell/species offset generation;
- candidate-pair filtering;
- size-ratio boundaries;
- fused mortality calculations;
- seed hierarchy;
- school compaction;
- hybrid sampler moments;
- diagnostic residual calculations.

## Regression tests

Maintain reference outputs for:

```text
minimal deterministic
minimal stochastic fixed seed
Baltic fixed seed
Eastern English Channel fixed seed
Java-versus-Python aggregate comparison
```

## Statistical tests

For each benchmark configuration, run at least 100 seeds and compare baseline with optimized mode.

Report:

- mean;
- standard deviation;
- median;
- 5th and 95th percentiles;
- collapse/extinction frequency;
- catch distribution;
- terminal biomass;
- spatial occupancy.

Use:

- confidence intervals for mean differences;
- Kolmogorov–Smirnov distance;
- Wasserstein distance;
- energy distance where appropriate.

Do not rely on a non-significant $p$-value alone. Define ecological equivalence tolerances in advance.

## Performance tests

Add non-blocking continuous-integration smoke benchmarks and scheduled full benchmarks.

CI should detect, for example:

- more than **10%** regression in minimal benchmark runtime;
- more than **15%** regression in Baltic kernel time;
- increased candidate-pair counts;
- unexpected allocation growth.

---

# 13. Proposed repository structure

```text
osmose/
  engine/
    python_engine.py
    state.py
    workspace.py
    diagnostics.py

    kernels/
      growth.py
      movement.py
      predation_legacy.py
      predation_sparse.py
      mortality.py
      fishing.py
      reproduction.py
      compaction.py

    indexing/
      cell_sort.py
      species_graph.py
      candidate_pairs.py

    random/
      seed_manager.py
      samplers.py

  calibration/
    parallel.py
    staged.py
    active_learning.py
    multiseed.py

benchmarks/
  benchmark_engine.py
  benchmark_predation.py
  benchmark_sampling.py
  configs/
  baselines/

tests/
  unit/
  regression/
  statistical/
  performance/

docs/
  performance/
    baseline.md
    optimized-engine.md
    reproducibility.md
    calibration-performance.md
```

---

# 14. Work backlog

## Epic A: performance baseline

- [ ] Add benchmark CLI.
- [ ] Add per-process timers.
- [ ] Add school and candidate-pair counters.
- [ ] Separate Numba compilation from runtime.
- [ ] Store hardware and package metadata.
- [ ] Commit fixed benchmark configurations.
- [ ] Publish baseline report.

## Epic B: sparse predation

- [ ] Build stable cell/species sort.
- [ ] Build offsets and compact blocks.
- [ ] Precompute trophic accessibility graph.
- [ ] Add size-window filtering.
- [ ] Implement sparse predation kernel.
- [ ] Add reference-versus-sparse comparison.
- [ ] Benchmark pair-count reduction.
- [ ] Document any iteration-order differences.

## Epic C: state and kernels

- [ ] Introduce `SchoolState`.
- [ ] Introduce reusable `EngineWorkspace`.
- [ ] Remove Python objects from hot loops.
- [ ] Fuse natural and fishing mortality loops.
- [ ] Add controlled school compaction.
- [ ] Audit array allocations.
- [ ] Test mixed precision experimentally.

## Epic D: parallel execution

- [ ] Create objective-only execution method.
- [ ] Add process pool.
- [ ] Add `SeedSequence` hierarchy.
- [ ] Set one thread per worker.
- [ ] Cache worker initialization.
- [ ] Add worker-scaling benchmark.
- [ ] Implement restartable evaluation batches.

## Epic E: stochastic sampling

- [ ] Add exact/hybrid/deterministic modes.
- [ ] Implement binomial, Poisson and normal branches.
- [ ] Protect extinction-sensitive groups.
- [ ] Add distributional sampler tests.
- [ ] Benchmark mortality and recruitment kernels.

## Epic F: calibration efficiency

- [ ] Integrate Morris screening.
- [ ] Add common-random-number evaluation.
- [ ] Add staged seed allocation.
- [ ] Connect surrogate search to multiseed re-ranking.
- [ ] Add active-learning batch acquisition.
- [ ] Add full-resolution finalist validation.
- [ ] Compare against baseline CMA-ES and NSGA-II budgets.

---

# 15. Release strategy

## Release 1: optimized CPU engine

Include:

- benchmark framework;
- sparse predation;
- structure-of-arrays;
- fused kernels;
- objective-only execution;
- parallel candidate–seed evaluation.

Target version:

```text
1.x performance release
```

## Release 2: calibration acceleration

Include:

- staged seed allocation;
- common random numbers;
- active surrogate calibration;
- automated finalist re-ranking;
- calibration budget reporting.

## Experimental release

Include behind explicit flags:

- hybrid stochastic approximations;
- JAX backend;
- competing-risk mortality;
- balanced predation;
- Strang splitting.

---

# 16. Final acceptance gate

The optimized engine should be merged as the default only when all conditions are met:

1. Baltic and Eastern English Channel simulations are at least **2× faster**.
2. An 8-worker ensemble achieves at least **5.5× wall-clock speedup**.
3. Fixed-seed differences are either zero or explicitly attributable to a documented ordering change.
4. Statistical outputs remain within predefined ecological tolerances.
5. No biomass-conservation regression is detected.
6. Reference `legacy` mode remains available.
7. Calibration reproduces or improves the baseline objective with at least **5× fewer full-resolution runs**.
8. Benchmarks and reproducibility metadata are generated automatically.

## Recommended first development milestone

The first milestone should contain only:

- benchmark instrumentation;
- candidate-pair counters;
- stable cell/species sorting;
- sparse predation candidate generation;
- reference-versus-optimized regression tests.

That milestone will establish whether predation is truly the dominant bottleneck and whether the anticipated **2–4× whole-engine speedup** is realistic before committing effort to JAX, GPU execution or more substantial mathematical reformulation.
