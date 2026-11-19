# C# engine, Phase 2: acceptance cross-check (Baltic)

Date: 2026-11-19. Plan: `docs/superpowers/specs/2026-11-17-csharp-engine-implementation-plan.md`.
Spec: `docs/superpowers/specs/2026-11-17-csharp-engine-design.md`. This is the Phase 2 deliverable
the plan names. It maps each acceptance criterion (spec 7.2) to its evidence, as the record for the
owner's gate G2. The Phase 1 record is `docs/csharp_cross_check_2026-11-18.md`.

## Gate G2

> **G2 (owner).** Decide whether calibration may use the C# engine (Q8) and whether
> bioenergetics is worth porting (Q5).

G2 is a decision, not a threshold: the plan gives it no pass mark. What it rests on is the
Phase 2 acceptance below, which holds in full.

**The owner decided on 2026-11-19:**

- **Calibration (Q8): opt-in.** The Python engine stays the default for calibration. The C#
  engine is available where it is asked for, and `scripts/calibrate_baltic.py --engine csharp`
  was made to work (it refused until then; see "Scope and limits"). No recalibration was run.
- **Scope (Q5): bioenergetics is not ported for now.** It stays on the Python engine, with
  genetics and economics. To revisit if a bioenergetics config heads for production. C# work
  continues with Phase 3's optional features only as they are needed.
- **The Python-engine diet defect** found on the way is recorded as issue #322.

The two questions as they were put, with the spec's own recommendations:

| Question | Spec's recommendation | What Phase 2 adds |
|---|---|---|
| Q8. Should calibration switch to the C# engine by default? | Opt-in until one full Baltic certification reproduces on the C# engine. | That certification now reproduces: all 18 verdicts agree. Calibration generations on Baltic went through the C# engine and ranked their candidates as the Python engine did. In a loop of many runs in one process (the engine called directly, not a calibration worker), the C# engine's resident memory stays about 95 MB under the Python engine's (all three below). |
| Q5. Where does the scope end; is bioenergetics worth porting? | Decide after the Phase 2 measurements. Genetics and economics stay on the Python engine. | The measurements: 2.47x at one thread and 6.81x at 14 on Baltic. |

## Phase 2 acceptance (spec 7.2)

| Criterion | Status | Evidence |
|---|---|---|
| A process fixture per feature against its Python function, and an A/B: feature off is byte-identical to before | **Met** for every task | The PRs in "Phase 2 tasks" below. Each carries its fixtures, its mutation list, and the result digests of EEC and Bay of Biscay, equal to master's in all 8 cases. |
| Determinism at 1, 2 and 14 threads | **Met** | Byte-identical on EEC and on the Baltic-derived config, for the minimal outputs and for every output family (`tests/test_csharp_parallel.py`, `tests/test_csharp_outputs_*.py`); and on the Baltic config as shipped, every family on, two seeds (`tests/test_csharp_engine_e2e.py::test_the_stock_baltic_run_is_the_same_at_1_2_and_14_threads`, added with this note). |
| Ensemble equivalence on Baltic | **Met**: PASS | "Ensemble equivalence" below. |
| The full default output families agree with Python's frames in shape, and statistically in value | **Met** | Shape: stock EEC, Baltic and Bay of Biscay give the Python engine's frame keys, shapes, labels and dtypes, and the same files (`tests/test_csharp_engine_e2e.py`). Value: "The full output families" below. |
| `baltic_stability_certify.py` on the C# engine gives the Python engine's verdicts on the same seeds | **Met**: all 18 agree | "Certification" below. |
| The Baltic performance gates: 1.15x at one thread, 1.5x at 14 | **Met**: 2.47x and 6.81x | `docs/perf/2026-11-19-csharp-baltic.md`. |

## Ensemble equivalence (spec 10.2)

`scripts/cross_engine_parity_440.py --engines python,python-control,csharp`, on
`data/baltic/baltic_all-parameters.csv`: 15 years, the first 5 dropped, `output.minimal.enabled`
on every arm. Two Python arms on disjoint seeds (the control is A against B) and one C# arm (the
treatment is A against C#). Metrics: biomass, abundance, yield and mean weight, per species.
Margins: 1.25x, and 1.05x for mean weight. The rule and its calibration are those of Phase 1
(`osmose/validation/engine_equivalence.py`, Bonferroni criterion 1 by the owner's decision).

| | 32 replicates per arm | The first 16 (the spec's number) |
|---|---|---|
| Verdict | **PASS** | **PASS** |
| Equivalent cells, C# / control, of 42 | 41 / 41 | 41 / 41 |
| Largest difference, Python vs C# | 0.0101 log10 (1.024x) | 0.0184 (1.043x) |
| Largest difference, Python vs Python | 0.0336 (1.080x) | 0.0341 (1.082x) |
| Median difference, Python vs C# / Python vs Python | 0.0008 / 0.0011 | 0.0015 / 0.0010 |
| Cells beyond the margin, Python vs C# / Python vs Python | 0 / 1 | 0 / 1 |
| Frozen rung (slack, ratio, variance) and its false-fail rate for identical engines | 5, 4.0, 8: 2.9% | 5, 4.0, 8: 3.5% |

The owner's protocol is 32 replicates; the spec's Phase 2 line says 16. The 32-replicate run is
the measurement, and the 16-replicate column is the same run re-judged on its first 16
replicates per arm.

**The one cell that is not equivalent is not equivalent on the control arm either:** the mean
weight of pikeperch. Against the C# engine the difference is small (0.0037), but its 90%
interval (0.029) is wider than the 1.05x margin (0.021). Between the two Python arms the
difference itself (0.034) is beyond the margin. So 32 replicates cannot resolve that cell for
any pair of arms: it is a power limit of the cell, not an engine difference.

**Six of the 42 cells carry no information.** The biomass, abundance and mean weight of the
two background species (grey seal, cormorant) are set by the forcing and are one same number in
every replicate of every arm: nothing preys on them, so they depend on no draw. The rule counts
them as equivalent, so of the 36 cells that vary, 35 are equivalent on each arm.

**This run predates the collection added in this PR**, as does the certification below. The
collection changes no result (see the end of the next section).

## The full output families

The ensemble above runs minimal outputs. For the families the C# engine collects since task 2.6,
`scripts/csharp/families_equivalence.py` applies the same design and the same rule to runs of
the config as shipped: 15 years, the first 5 dropped. Each run is reduced per species to
post-spin-up means. As above, the 32-replicate run is the measurement and the 16-replicate
column is its first 16 replicates per arm.

| | 32 replicates per arm | The first 16 |
|---|---|---|
| Verdict | **PASS** | **PASS** |
| Equivalent cells, C# / control, of 103 | 103 / 102 | 102 / 102 |
| Largest difference, Python vs C# | 0.0098 log10 (1.023x) | 0.0173 (1.041x) |
| Largest difference, Python vs Python | 0.0154 (1.036x) | 0.0155 (1.036x) |
| Frozen rung (slack, ratio, variance) and its false-fail rate for identical engines | 2, 2.0, 4: 1.5% | 4, 3.0, 6: 3.6% |

Largest difference by family, at 32 replicates, in log10 units:

| Family | Metric | Margin | Python vs C# | Python vs Python |
|---|---|---|---|---|
| SSB | `ssb` | 1.25x | 0.0075 | 0.0122 |
| Yield in numbers | `yield_n` | 1.25x | 0.0097 | 0.0151 |
| Life stages | eggs | 1.25x | 0.0028 | 0.0082 |
| | juveniles | 1.25x | 0.0055 | 0.0154 |
| | adults | 1.25x | 0.0098 | 0.0124 |
| By age | biomass in age class 0 | 1.25x | 0.0089 | 0.0089 |
| | biomass in older classes | 1.25x | 0.0070 | 0.0120 |
| Diet | fish eaten, by predator | 1.25x | 0.0065 | 0.0122 |
| | resources eaten, by predator | 1.25x | 0.0065 | 0.0113 |
| Mean size | `mean_size` | 1.05x | 0.0036 | 0.0132 |
| Mean trophic level | `mean_tl` | 1.05x | 0.0002 | 0.0004 |

In every family the C# engine is at least as close to the Python engine as a second Python arm
is. The diet rows include the two background predators. The one cell that is not equivalent,
on the control arm only at 32 replicates and on both arms at 16, is the mean size of pikeperch:
the same species and the same reason as above.

**Four of the 103 cells carry no information.** They are one same number in every replicate of
every arm: stickleback eats no fish, the two background predators eat no resources, and
stickleback's trophic level is exactly 2. The rule counts them as equivalent; 99 cells vary.

**What the script checks beyond the rule.** The rule compares the cells it is given; four
things would let a missing output through it, and each is a failure of its own:

- a family that only some arms produced;
- a cell only one engine has;
- a cell that is zero throughout on one engine and not on the other (an output the engine never
  filled comes back as zeros, and the rule's log scale reads every value under 1 as 1, so zeros
  against small values would read as no difference);
- a missing or non-finite value, in any arm, in an output that exists for every species (all
  of them but the two weighted means).

None of the four occurs in this run. No cell of it has a value under that floor, other than
the three that are zero on every arm.

**This run also checks the collection added in this PR** (Budgets, below). The 16-replicate
run was made before it and the 32-replicate run after it, on the same seeds. The first 16
replicates of the later run equal the earlier run bit for bit, in all 103 cells of all three
arms. So 15-year runs with every output on are unchanged by it.

**What this does not cover:**
- **By size** is covered through the mean size only, not class by class.
- **The per-stage mortality rates** are not in it. They are sums of per-step rates and can be
  `inf` (CLAUDE.md), which the rule's log scale cannot take. Their counts are covered through
  the life-stage abundances; the rates themselves are checked bit for bit on captured states
  (`tests/test_csharp_outputs_stage.py`).
- **Abundance by age** is not produced by the Baltic config as shipped (it asks for biomass by
  age), so it is absent here. It is checked bit for bit on captured states.

## Certification

`scripts/baltic_stability_certify.py --csharp`: the production Baltic config, 50 years, the
five certification seeds (42, 123, 7, 999, 2024), the Python engine and then the C# engine on
the same seeds, the same criterion.

**All 18 verdicts agree** (9 species, persists and in-envelope).

| Species | Tier | Python | C# | Python final-decade mean (t) | C# final-decade mean (t) |
|---|---|---|---|---|---|
| cod_west | assessed | pass | pass | 18,947 to 19,398 | 18,697 to 19,525 |
| cod_east | assessed | pass | pass | 72,461 to 74,220 | 72,544 to 73,233 |
| herring | assessed | pass | pass | 1,520,219 to 1,547,826 | 1,490,919 to 1,554,383 |
| sprat | assessed | pass | pass | 996,320 to 1,009,241 | 992,619 to 1,008,546 |
| flounder | assessed | pass | pass | 46,946 to 48,339 | 46,729 to 48,151 |
| perch | indicative | pass | pass | 34,052 to 35,842 | 34,337 to 36,703 |
| pikeperch | indicative | persists, above its envelope | the same | 515,161 to 531,839 | 519,475 to 530,278 |
| smelt | indicative | persists, above its envelope | the same | 650,930 to 665,946 | 642,715 to 664,890 |
| stickleback | indicative | pass | pass | 86,330 to 94,748 | 87,011 to 91,871 |

Both engines: 5 of 5 assessed species persistent and in-envelope; 2 of 4 indicative. The ranges
are the smallest and largest of the five seeds' final-decade means.

## Budgets (spec 10.4)

| Budget | Result | Status |
|---|---|---|
| First call in a fresh process, at most 2 s | 0.51 s (1 thread), 0.35 s (14) | Met |
| Bundle build plus marshalling, at most 5% of a run | 0.5% (1 thread), 3.0% (14) | Met |
| Peak memory, no more than the Python engine's | 232 vs 308 MB (1 thread); 238 vs 310 MB (14) | Met, by about 70 MB. **It was not met before this PR** for a process that runs many simulations: see below. |

**Memory over many runs.** The harness's five readings at 14 threads had ranged from 290 to
336 MB against the Python engine's 309. Run many times in one process, the C# engine at one
thread drifted up to a 324 MB peak over 60 runs, above the Python engine's. The cause was the
.NET collector leaving each run's dead arrays (about 30 MB) for two or three more runs; nothing
leaked outside the managed heap. The engine now collects once when a run returns (about 4 ms).
After that, resident memory after a run is flat at 203 to 213 MB at either thread count, with
the full outputs too, against the Python engine's 299 MB; peaks are 232 to 246 MB against 309.
The results are unchanged, byte for byte. The measurements are in the performance report.

## Calibration through the C# engine

Generations of four candidates on the Baltic config as shipped (5 years; herring's additional
mortality at 0.1, 0.2, 0.4 and 0.8; objective: biomass RMSE against a Python baseline run on
seed 0), through `OsmoseCalibrationProblem`. A candidate runs on the seed of its position in
the population, so the same candidates in another order run on other seeds: three orders on
the Python engine and two on the C# engine. The spread across seeds within the Python engine
is the control for the difference between the engines.

| Herring additional mortality | Python objective, by seed | C# objective, by seed |
|---|---|---|
| 0.1 | 84,597 (seed 0); 62,392 (1); 74,618 (3) | 59,384 (0); 64,259 (3) |
| 0.2 (as shipped) | 0 (seed 0: the baseline itself); 18,879 (1); 42,176 (2) | 25,681 (1); 40,645 (2) |
| 0.4 | 121,233 (1); 133,347 (2); 132,386 (3) | 121,333 (1); 134,394 (2) |
| 0.8 | 265,280 (0); 267,917 (2); 266,342 (3) | 272,012 (0); 266,958 (3) |

- **Both engines rank the candidates the same way in every generation** (0.2, 0.1, 0.4, 0.8):
  the ranges of neighbouring candidates do not overlap on either engine.
- **The differences between the engines are the size of the Python engine's own seed noise.**
  Within the Python engine one candidate varies between seeds by 1.36x at 0.1 and 2.2x at 0.2
  (1.10x at 0.4, 1.01x at 0.8). Every C# value is inside the Python engine's range for its
  candidate or within 5% of it.
- **At 0.1 both C# values sit below the three Python values** (59 and 64 thousand against 62,
  75 and 85 thousand). With two runs against three that is not a distinguishable difference;
  it is not evidence of equivalence either. The ensembles above are that evidence, at the
  shipped parameters.
- **The path is the one calibration uses**: the C# generations ran on a forkserver pool of two
  workers, one thread each, and the .NET runtime was never loaded in the parent process.

This shows that calibration runs on Baltic through the C# engine and orders candidates as the
Python engine does. It is five generations of one parameter, not a calibration.

## Phase 2 tasks

| Task | PR |
|---|---|
| 2.5 Oxygen to benthos | #312 |
| 2.3 Recruitment: the four curves, the seeding mode, the RV gate | #313 |
| 2.4 Salinity movement gate | #314 |
| 2.1 Background species | #315 |
| The A/B harness for any config; the first Baltic reading | #316 |
| 2.6a Outputs: life-stage split and per-step rates | #317 |
| 2.6b Outputs: by age, by size, SSB, yield in numbers, mean size, mean trophic level | #318 |
| 2.6c Outputs: the diet; the full outputs switched on | #319 |
| 2.7 UI: three engines | #320 |
| 2.8 Baltic validation (this note, the performance report, the harness changes, the collection at the end of a run) | this PR |

**Task 2.2 was not needed and is not done.** Out-of-domain mortality was already in the Phase 1
core. Its other two features, the growth cap (`species.lmax`) and immobile focal species
(`movement_none`), are used by none of the three shipped demos. A config that sets either is
refused, by name.

## Scope and limits (what Phase 2 does not do)

- **Features still refused, by name:** the growth cap and immobile species (above), and Phase
  3's optional features (depletable resources, incoming flux, marine protected areas,
  time-varying rates, the other recruitment gates, spatial and carbon outputs, and others).
  Benguela is refused for `incoming_flux`. Bioenergetics, genetics and economics run on the
  Python engine only.
- **A Python-engine defect was reproduced, not fixed.** When a background school is eaten, the
  Python engine books it in the diet column of the resource with the same index
  (`mortality.py`, `diet_matrix[p, species_id[q]]`). It is inert on Baltic as shipped, because
  nothing eats a seal or a cormorant. The C# engine copies it for parity, and a test forces the
  case. Whether to fix it in the Python engine is the owner's call.
- **The UI gives a C# run no step progress and no live view.** The engine does not expose its
  schools. Progress could come from its control block later.
- **`scripts/calibrate_baltic.py --engine csharp` works since the G2 follow-up**, with
  `--optimizer de` only (the other two optimizers evaluate through joblib, which was not tried
  with the C# engine). When this note was first written the script refused, on the belief that
  its forking DE pool ruled the C# engine out. Measured since: a child forked from a process
  that has NOT loaded the .NET runtime runs the engine normally; only a fork made AFTER the
  runtime is loaded kills the child. So the script forks its workers first and runs the engine
  in its own process only after the last fork, and a guard turns the wrong order into an error.
  A C# run writes `phase<N>_csharp_*.json`, so it never replaces a Python result.
- **Production is unchanged until a deploy opts in** (`OSMOSE_DEPLOY_CSHARP=1`, DEPLOY.md). The
  UI shows the C# button only where the engine is installed.
- **Two implementations of one model** (spec risk R3). A PR that changes Python engine semantics
  must update the C# engine or extend `csharp_engine_block_reason` (CLAUDE.md). Since task 2.6c
  an output key the Python engine starts reading is refused by the C# engine until someone
  classifies it.

## Reproduce

```bash
bash scripts/build_csharp_engine.sh
# the ensemble (about 50 minutes; the raw replicates are kept in the JSON)
PYTHONPATH=. python scripts/cross_engine_parity_440.py --engines python,python-control,csharp \
  --config data/baltic/baltic_all-parameters.csv --n 32 --years 15 --spinup-years 5 \
  --persist-results equiv_baltic.json
# the full output families (about 50 minutes)
PYTHONPATH=. python scripts/csharp/families_equivalence.py --config baltic --n 32 --years 15 \
  --spinup-years 5 --json families_baltic.json
# the certification on both engines (about 20 minutes)
PYTHONPATH=. python scripts/baltic_stability_certify.py --csharp --out certification.md
```

Re-judge a saved ensemble without simulating: add `--rejudge <json>` to either harness (the
main one also needs its `--engines` argument).
