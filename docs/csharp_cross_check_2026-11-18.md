# C# engine, Phase 1: acceptance cross-check

Date: 2026-11-18. Plan: `docs/superpowers/specs/2026-11-17-csharp-engine-implementation-plan.md`.
Spec: `docs/superpowers/specs/2026-11-17-csharp-engine-design.md`. This is the Phase 1 deliverable
the plan names. It maps each acceptance criterion (spec 7.2) to its evidence, as the record for the
owner's gate G1.

## Gate G1

> **G1 (owner).** Proceed to Baltic if equivalence holds and the EEC gate (no regression at 1 and
> 14 threads) is met.

| G1 criterion | Status | Evidence |
|---|---|---|
| Ensemble equivalence holds | **Met** (EEC and Bay of Biscay) | `docs/perf/2026-11-18-csharp-ensemble-equivalence.md` |
| EEC gate: no regression at 1 and 14 threads | **Met**: 3.77x and 7.57x (gate 1.0x) | `docs/perf/2026-11-18-csharp-eec.md` |

Both criteria hold. **The owner decided GO on 2026-11-18: Phase 2 starts.**

## Phase 1 acceptance (spec 7.2)

| Criterion | Status | Evidence |
|---|---|---|
| Process tests green | **Met** | CI job `csharp-engine`: xUnit (76) plus `tests/test_csharp_*.py` with `OSMOSE_REQUIRE_CSHARP=1` (257 at master `e876e836`, none skipped). Tier E/L oracles: mortality step (`test_csharp_mortality_step.py`), movement (`test_csharp_movement.py`), life cycle (`test_csharp_life_cycle.py`), step pipeline and record windows (`test_csharp_pipeline.py`), per-step preparation (`test_csharp_prep.py`). |
| Determinism | **Met** | Result buffers byte-identical at 1, 2 and 14 threads on EEC and Bay of Biscay; repeat runs identical; different seeds differ; zero allocation per steady-state step at one thread (`test_csharp_parallel.py`, task 1.9). |
| Death accounting | **Met** | `MortalityStep.CheckDeathAccounting` (relative tolerance 1e-9) under `run.checks`; tested for every school on both configs (`test_csharp_mortality_step.py::test_death_accounting_holds_for_every_school`) and in xUnit, including over-kill the clamp would hide. |
| Ensemble equivalence, by the rule fixed in advance | **Met**, under the owner's amendment | Spec 10.2, 32 replicates per arm, Python-vs-Python control. The rule as first written rejected identical engines 78.5% of the time on EEC (criterion 1). The owner's amendment (2026-11-18): criterion 1 uses a Bonferroni interval. Under it, EEC passes (identical-engine false-fail 4.8%) and Bay of Biscay passes (2.0%). On EEC, C# is closer to Python arm B (max \|d\| 0.094) than Python arm A is (0.382). |
| Out-of-scope configs refused | **Met** | Fail-closed feature catalogue: every config key the Python engine reads is classified, and an unclassified key blocks and is named (`test_csharp_capabilities.py`). Baltic is refused (Phase 2 features), so are stock configs without `output.minimal`, the live map observer, and Q4's three cases. |
| EEC performance gate | **Met** | 3.77x at one thread, 7.57x at 14; every pair agrees (`docs/perf/2026-11-18-csharp-eec.md`). |

## Budgets (spec 10.4)

| Budget | Status | Note |
|---|---|---|
| First call in a fresh process, at most 2 s | **Met**: 0.57 s (1 thread), 0.40 s (14) | |
| Peak memory, no more than the Python engine's | **Met**: 217 vs 286 MB, 218 vs 287 MB | Since #310: the C# path no longer imports Numba (it missed by 1.9 MB before). |
| Bundle build plus marshalling, at most 5% of a 10-yr EEC run | **Met**, on the owner's reading (2026-11-18): the budget counts C#-specific overhead | The C#-only part is 6.5 ms: **1.8% at 14 threads**, 0.4% at one. With the input resolution both engines perform included (29.5 ms, which the Python engine does inside its own timed run), it would be 10.6% at 14 threads. |

**For the record: what the strict reading would have taken (measured, not done):**
- Exact optimisations helping both engines bring the build to ~27 ms, 7.4% at 14 threads:
  - a lazy data-directory search: ~3 ms of 3.8;
  - one gather in place of the masked remap of the map tables: 6.4 → 2.2 ms;
  - one config scan instead of one per species: ~2 ms.
- Adding per-process caches of the parsed map grids (3.8 ms) and the LTL forcing (5.6 ms) brings it to ~17.5 ms (4.8%). Those caches only help long-lived processes, such as calibration workers and the A/B harness, and are a shared-code change keyed by file identity.

## Phase 1 tasks

| Task | PR |
|---|---|
| 1.1 Feature catalogue, fail-closed key coverage | #298 |
| 1.2–1.4 Bundle builder, school store, per-step preparation (bit-exact) | #299 |
| 1.5 Mortality step (libm-exact) and death accounting | #300 |
| 1.6 Movement | #301 |
| 1.7–1.8a Life cycle, step pipeline, outputs, record windows | #302 |
| 1.8b–1.9 `CSharpEngine`; parallel and deterministic | #303 |
| 1.10 Ensemble equivalence | #305 |
| 1.11 EEC performance | #307, then #310 (memory) |
| 1.12 Engine selection; calibration acceptance | #304, #308 |
| 1.13 Packaging, deployment self-test | #306, then #309 (deploy privileges) |

## Scope and limits (what Phase 1 does not do)

- The C# engine runs **EEC and Bay of Biscay with `output.minimal`**. Baltic needs Phase 2's features (background species, out-of-domain mortality, recruitment curves, the salinity gate, oxygen to benthos, the full outputs) and is refused until each lands.
- **The UI does not offer the engine** (Phase 2, decision D12). Calibration and the CLI do (`engine="csharp"`, `osmose run --engine csharp`). `calibrate_baltic.py --engine csharp` refuses, because its pool forks.
- **Production:** the engine is off unless a deploy sets `OSMOSE_DEPLOY_CSHARP=1` (DEPLOY.md). The Docker and Apptainer images build it, and CI self-tests it in both.
- **Two implementations of one model** (spec risk R3). A PR that changes Python engine semantics must update the C# engine or extend `csharp_engine_block_reason` (CLAUDE.md).
