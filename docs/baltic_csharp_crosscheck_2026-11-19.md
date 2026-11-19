# C# engine, Phase 2: acceptance cross-check (Baltic)

Date: 2026-11-19. Plan: `docs/superpowers/specs/2026-11-17-csharp-engine-implementation-plan.md`.
Spec: `docs/superpowers/specs/2026-11-17-csharp-engine-design.md`. This is the Phase 2 deliverable
the plan names. It maps each acceptance criterion (spec 7.2) to its evidence, as the record for the
owner's gate G2. The Phase 1 record is `docs/csharp_cross_check_2026-11-18.md`.

## Gate G2

> **G2 (owner).** Decide whether calibration may use the C# engine (Q8) and whether
> bioenergetics is worth porting (Q5).

G2 is a decision, not a threshold: the plan gives it no pass mark. What it rests on is the
Phase 2 acceptance below, which holds in full. **The decision has not been taken.** The two
questions, with the spec's own recommendations:

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

The background species' columns (grey seal, cormorant) are identical on every arm: nothing
preys on them, so they depend on no draw.

## The full output families

The ensemble above runs minimal outputs. For the families the C# engine collects since task 2.6,
`scripts/csharp/families_equivalence.py` applies the same design and the same rule to runs of
the config as shipped: 16 replicates per arm, 15 years, the first 5 dropped. Each run is reduced
per species to post-spin-up means.

**Verdict: PASS.** 102 of 103 cells are equivalent, on the C# arm and on the control arm alike.
Frozen rung: slack 4, ratio 3.0, variance 6; false-fail rate for identical engines 3.6%.

| Family | Metric | Margin | Largest difference, Python vs C# | Python vs Python |
|---|---|---|---|---|
| SSB | `ssb` | 1.25x | 0.0124 | 0.0118 |
| Yield in numbers | `yield_n` | 1.25x | 0.0135 | 0.0118 |
| Life stages | eggs | 1.25x | 0.0045 | 0.0083 |
| | juveniles | 1.25x | 0.0055 | 0.0091 |
| | adults | 1.25x | 0.0161 | 0.0096 |
| By age | biomass in age class 0 | 1.25x | 0.0173 | 0.0091 |
| | biomass in older classes | 1.25x | 0.0109 | 0.0116 |
| Diet | fish eaten, by predator | 1.25x | 0.0099 | 0.0155 |
| | resources eaten, by predator | 1.25x | 0.0121 | 0.0106 |
| Mean size | `mean_size` | 1.05x | 0.0054 | 0.0103 |
| Mean trophic level | `mean_tl` | 1.05x | 0.0004 | 0.0012 |

Differences are in log10 units; the largest, 0.0173, is 1.041x. The diet rows include the two
background predators. The one cell that is not equivalent, on both arms, is the mean size of
pikeperch: the same species and the same reason as above.

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

One generation of four candidates on the Baltic config as shipped (5 years; herring's
additional mortality at 0.1, 0.2, 0.4 and 0.8; objective: biomass RMSE against a Python
baseline run), through `OsmoseCalibrationProblem`, on each engine:

| Herring additional mortality | Python objective | C# objective | Ratio |
|---|---|---|---|
| 0.1 | 84,597 | 59,384 | 0.70 |
| 0.2 (as shipped) | 18,879 | 25,681 | 1.36 |
| 0.4 | 133,347 | 134,394 | 1.008 |
| 0.8 | 266,342 | 266,958 | 1.002 |

Both engines rank the candidates the same way (0.2, 0.1, 0.4, 0.8). The C# generation ran on a
forkserver pool of two workers, one thread each, and the .NET runtime was never loaded in the
parent process. The two candidates far from the baseline agree within 1%. The two near it
differ more because each is a single 5-year run on its own seed, and at 0.2 the objective is
nothing but seed noise (the candidate is the baseline's own config). This shows the path works
on Baltic and orders candidates sensibly; it is one generation, not a calibration.

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
- **`scripts/calibrate_baltic.py --engine csharp` still refuses.** The engine can run the config;
  the script's DE pool forks, and the .NET runtime must not be hosted in a process that forks.
  `OsmoseCalibrationProblem(engine="csharp")` (forkserver workers) does work, since task 1.12.
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
# the full output families (about 25 minutes)
PYTHONPATH=. python scripts/csharp/families_equivalence.py --config baltic --n 16 --years 15 \
  --spinup-years 5 --json families_baltic.json
# the certification on both engines (about 20 minutes)
PYTHONPATH=. python scripts/baltic_stability_certify.py --csharp --out certification.md
```

Re-judge a saved ensemble without simulating: add `--rejudge <json>` to either harness (the
main one also needs its `--engines` argument).
