# Activate Ev-OSMOSE Genetics — FIE-on-Cod Demonstration

**Date:** 2026-05-18
**Author:** brainstormed with Claude (Opus 4.7)
**Status:** Spec — pending user review, then writing-plans

## Context

The Ev-OSMOSE genetics module landed in 2026-04-06 (`docs/superpowers/plans/2026-04-06-ev-osmose-genetics-core-plan.md`) and is structurally complete: `ctx.genetic_state` is created when `simulation.genetic.enabled=true`, `express_traits` runs per step and pushes phenotypes into `trait_overrides`, the bioen step + bioen reproduction consume overrides, `create_offspring_genotypes` inherits alleles, and `compact_genetic_state` syncs with the dead-school mask. Four unit-level test files exist (neutral, trait, inheritance, expression).

**What is missing:** no fixture has `simulation.genetic.enabled=true`; the module has never been exercised end-to-end against a real configuration; there is no per-step output exposing trait statistics; there is no scientific demonstration of the module's intended scientific use case.

This spec activates the module via a single, focused scientific demonstration: fishery-induced evolution (FIE) on Baltic cod growth rate (`imax`). Cod *maturation* FIE is well-established in wild stocks (Olsen et al. 2004); cod *growth-rate* FIE under wild fishing is rarer and is typically confounded with concurrent maturation evolution (Heino, Pauli & Dieckmann 2015). Growth-rate FIE has only been cleanly isolated in lab common-garden experiments on silversides (Conover & Munch 2002; Walsh et al. 2006). This demo targets the growth-rate pathway via the bioenergetic `imax` trait — a different surface than Olsen 2004's PMRN-maturation — by holding the maturation gate constant (m0 fixed-threshold, no `bioen_m0` trait).

This work parallels backlog top-5 #2 (DSVM bioeconomic activation, shipped 2026-05-08 in PR #47), which followed the same wired-but-inactive → activated-via-demo-script pattern.

## Goal and non-goals

**Goal.** Ship a reproducible scientific demonstration of fishery-induced evolution on Baltic cod growth rate. Deliver paired high-F vs low-F scenarios over ~30 cod generations (200 model-years) showing measurable, multi-seed-validated shift in mean `imax` trait under selective fishing.

**Non-goals.**
- No Shiny UI surface for trait configuration or evolution plots — explicitly out of scope per the chosen "scientific demonstration" framing.
- No multi-trait demonstration — single trait (`imax`) only.
- No modification of the existing calibrated `baltic/` fixture — strictly isolated via a new `baltic_ev/` clone.
- No bringing the dead `state.imax_trait` bridge live — documented as vestigial, separate follow-up.
- No bioen calibration of `baltic_ev` against ICES assessments — absolute biomass is acknowledged as out-of-spec; only the *directional trait response* is the deliverable.

## Architecture

Three units of work, each independently testable:

```
data/baltic_ev/               [new fixture]
  ├── config files cloned from baltic/
  ├── simulation.bioen.enabled=true
  ├── species.bioen.* params per-species (cod focus)
  ├── simulation.genetic.enabled=true
  └── evolution.trait.imax.* keys for cod (sp0) only

osmose/engine/                [minimal new engine code]
  ├── simulate.py             collect genetic_state phenotype stats per step
  ├── outputs.py              new StepOutput field: trait_stats per species
  └── results.py              reader for genetic_trait_means.csv

scripts/                      [demonstration glue]
  └── run_fie_demo.py         paired high-F / low-F runs, multi-seed, chart

docs/tutorials/               [scientific story]
  └── fie-on-baltic-cod.md    interpretation + chart + run command
```

## Components

### 1. `data/baltic_ev/` fixture

Clone `data/baltic/`. Add:

- `simulation.bioen.enabled=true`
- Per-species bioen params:
  - `species.beta.sp{i}` — allometric exponent
  - `species.bioen.maturity.r.sp{i}` — reproductive allocation
  - `species.bioen.intake.imax.sp{i}` — max ingestion rate
  - `species.bioen.forage.{k1_for,k2_for,I_max}.sp{i}` (only when genetic foraging is on; out-of-scope here)
- `simulation.genetic.enabled=true`
- For cod (sp0) only:
  - `evolution.trait.imax.target=bioen_i_max`
  - `evolution.trait.imax.mean.sp0` — equal to `predation.ingestion.rate.max.bioen.sp0` (the per-species mean baseline; engine reader key verified at `config.py:1796`; exact literature value sourced during plan execution)
  - `evolution.trait.imax.var.sp0` — additive genetic variance
  - `evolution.trait.imax.envvar.sp0` — environmental noise variance (chosen alongside var so h² ≈ 0.25 per Nielsen et al. 2014; see plan for exact values)
  - `evolution.trait.imax.nlocus.sp0` = 10
  - `evolution.trait.imax.nval.sp0` = 20

(The earlier draft of this spec referenced `species.bioen.intake.imax.sp0` as the engine's max-ingestion-rate key. That key is silently ignored by the reader — it doesn't exist. Correct key is `predation.ingestion.rate.max.bioen.sp{i}`.)
- `population.genotype.transmission.year.start=10` — seed-phase years before inheritance kicks in (avoids cold-start drift dominating early signal).

**Provenance.** Bioen params for Baltic cod sourced from:
- Brander (1995) for growth + intake rates on Baltic cod
- Mehner & Wieser (1994) for metabolic exponents in coldwater gadoids
- Each value documented inline in fixture README with citation

Where Baltic-specific values are unavailable, fall back to EEC cod values from `data/eec_full/` and flag in README. Acceptable because §7 absolute biomass is out-of-spec — directional trait response is the deliverable.

### 2. Per-step genetic trait statistics output

New CSV `genetic_trait_means.csv` written by `_collect_outputs` in `osmose/engine/simulate.py` and recorded in `osmose/engine/outputs.py`. Columns:

| time | species_id | trait_name | mean | variance | n_individuals |
|---|---|---|---|---|---|

Populated only when `ctx.genetic_state is not None`. When None, the CSV is not created (consistent with how spatial outputs are skipped when disabled).

Implementation:
- Add `trait_stats: dict[str, dict[int, TraitStats]] | None` to `StepOutput` dataclass.
- Compute in `_collect_outputs`: iterate `ctx.genetic_state.registry.traits`, group expressed phenotypes by `state.species_id`, compute mean/var/count. `express_traits` is already called per step at `simulate.py:1341`; pass the result through to `_collect_outputs` rather than recomputing.
- Writer: extend the existing `write_step_outputs` to flush `trait_stats` if present.
- Reader: add `read_genetic_trait_means(output_dir) -> xarray.Dataset` in `osmose/results.py`.

### 3. FIE demo script

`scripts/run_fie_demo.py`. Two paired scenarios:

| Scenario | Description | Cod F |
|---|---|---|
| `baltic_ev_high_f` | Modern Baltic cod fishing pressure | ≈ 0.6/yr |
| `baltic_ev_low_f` | Near-unfished reference | ≈ 0.1/yr |

200 model-years each → ~30 cod generations (cod gen time ≈ 7y). Three seeds per scenario → 6 total runs (~20 minutes total wall-clock at current engine perf).

Script:
1. Apply each scenario's cod-F override to a fresh `baltic_ev` config.
2. Run via `PythonEngine`, write outputs under `outputs/fie_demo/<scenario>/seed<n>/`.
3. Read `genetic_trait_means.csv` from each run.
4. Plot mean `imax` for cod over time, with ribbon = ±1 SD across seeds, one line per scenario.
5. Save PNG to `outputs/fie_demo/fie_imax_trajectory.png`.
6. Print summary to stdout: end-of-run cod mean `imax` per scenario, delta, two-sample t-test p-value across seeds.

### 4. Tutorial

`docs/tutorials/fie-on-baltic-cod.md`, ~300 lines. Layout follows the 30-minute tutorial pattern:

1. Introduction — what FIE is, why it matters, classical literature
2. The Baltic cod story — historical evidence, why this is a good test case
3. Run the demo — single command (`python scripts/run_fie_demo.py`)
4. Interpretation — show the PNG inline; explain the high-F trajectory dropping; relate to literature predictions
5. Exercise — invite the reader to change `nlocus.sp0` or `var.sp0` and observe how genetic architecture changes the response speed
6. Caveats — absolute biomass is uncalibrated; demo isolates trait direction
7. Citations — Olsen 2004, Heino 2015, Conover & Munch 2002, Walsh 2006, Brander 1995

## Data flow

```
config: simulation.genetic.enabled=true
         + simulation.bioen.enabled=true
         + evolution.trait.imax.target=bioen_i_max
                 │
                 ▼
simulate.py:1093  create_initial_genotypes() → ctx.genetic_state
                 │
              per step
                 ▼
simulate.py:1338  express_traits(genetic_state, species_id) → phenotypes['imax']
simulate.py:1342  apply_trait_overrides(trait_overrides, phenotypes, registry)
                  → trait_overrides['bioen_i_max'] = per-school array
                 │
                 ▼
simulate.py:1352  _bioen_step(state, config, ..., trait_overrides) uses bioen_i_max in intake calc
                 │
                 ▼
processes/predation.py  selective fishing kills high-growth schools preferentially
                 │
                 ▼
simulate.py:1370  _bioen_reproduction(state, ..., trait_overrides)
                  high-imax parents have lower spawning weight under selection
                 │
                 ▼
simulate.py:1387  create_offspring_genotypes(parent_gs, gonad_weight, ...)
                  imax alleles inherited weighted by parental reproductive success
                  → mean imax drifts downward over generations
                 │
                 ▼
simulate.py:_collect_outputs  emits per-step trait_stats → genetic_trait_means.csv
                 │
                 ▼
scripts/run_fie_demo.py  reads, charts, saves PNG
```

## Error handling

| Condition | Behavior |
|---|---|
| Genetics off + `state.imax_trait` populated | Existing `is not None` guards handle. Unchanged. |
| Genetics on + empty trait registry | Warn, run as if genetics off, do not crash. |
| `genetic_trait_means.csv` write failure | Log and continue. Biomass output is critical-path; trait output is observational. |
| FIE demo with bioen disabled | Hard fail with explicit error: "Ev-OSMOSE traits require simulation.bioen.enabled=true". |
| Trait variance == 0 + nlocus > 0 | Existing `TraitRegistry.from_config` already handles (skips allele pool). Verify via test. |
| Missing per-species mean | Existing default of 0.0 is wrong for cod imax. Validator should fail at config-load if a trait is declared but per-species mean is unset on a non-zero-variance species. **Add validator.** |

## Testing

| Test | Effort | Wall-clock |
|---|---|---|
| `tests/test_ev_osmose_activation.py` — smoke: baltic_ev runs 10y with genetics on, ctx.genetic_state non-None at end, CSV produced | 0.3d | <30s |
| `tests/test_genetics_trait_expression.py` (extended) — under zero genetic variance, expressed `imax` equals species_mean + env_noise within tolerance | 0.2d | <5s |
| `tests/test_fie_demo_direction.py` — paired 50y run (1 seed each scenario), assert end-of-run mean cod `imax` in high-F is **lower** than low-F by ≥ 1% | 0.3d | ~3 min |
| `tests/test_baltic_ev_validator.py` — declared trait without per-species mean fails config-load | 0.1d | <1s |
| `tests/test_baltic_parity.py` (existing) | unchanged | unchanged — `baltic_ev` is a separate dir |

Total new test wall-clock: ~3.5 min. The 50y FIE-direction test is the slowest; place it behind a `pytest -m slow` marker to keep default `pytest` runs under 1 minute.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Baltic cod bioen params not in literature → guesswork | Med | Document each value's provenance; fall back to EEC cod where needed. Out-of-spec absolute biomass is acknowledged. |
| 200-year demo too slow for tutorial UX | Low | Engine perf is ~3s per 5y on baltic = ~2min per scenario at single-seed; 6-run demo ≈ 12-20 min on commodity hardware. Acceptable. |
| FIE signal too weak in 30 generations | Med | If absent in 30 gens, escalate to 100 gens (~600 model-years, ~70 min). Choose window after first run. |
| Genetic drift dominates response → no FIE signal | Med | Multi-seed (3 seeds × 2 scenarios = 6 runs); report mean ± SD. Two-sample t-test on end-state mean across seeds. |
| Bioen-on-Baltic-cod produces unrealistic biomass | High | **Accepted.** Tutorial explicitly frames this — demo is about trait direction, not absolute calibration. |
| `state.imax_trait` dead bridge surfaces as a confusing relic | Low | Add 1-line code comment at `state.py:81` referencing this spec and noting the path is vestigial. Open follow-up ticket. |

## Effort estimate

| Component | Effort |
|---|---|
| `data/baltic_ev/` fixture + bioen params + README provenance | 1.0d |
| `genetic_trait_means.csv` output (engine + writer + reader) | 0.5d |
| `scripts/run_fie_demo.py` + multi-seed wrapper + chart | 1.0d |
| Tutorial doc | 0.5d |
| Tests (activation + direction + validator + expression-extension) | 1.0d |
| Iteration + multi-seed validation + literature cross-check | 1.0d |
| **Total** | **~5 working days** |

## Out-of-scope follow-ups

These are explicitly excluded from this spec but worth tracking:

1. **Delete dead `state.imax_trait` field + branches** (`state.py:81`, `mortality.py:296-316`, `foraging_mortality.py:36`, `reproduction.py:188`). Cleanup-as-cleanup, separate PR.
2. **Calibrate `baltic_ev` against ICES assessments** under bioen — would extend baltic_ev from "demo fixture" to "production fixture."
3. **Multi-trait extensions** — e.g., `gsi → bioen_r` for evolution of reproductive effort; `m0` for larval mortality.
4. **Shiny UI surface** — toggle in the simulation tab, evolution-trajectory chart in Results.
5. **ICES validator extension** for trait-evolution metrics — compare evolved `Linf` / age-at-maturation to published FIE estimates.

## References

- Olsen, E. M., Heino, M., Lilly, G. R., Morgan, M. J., Brattey, J., Ernande, B., & Dieckmann, U. (2004). Maturation trends indicative of rapid evolution preceded the collapse of northern cod. *Nature*, 428(6986), 932-935. https://doi.org/10.1038/nature02430
- Heino, M., Pauli, B. D., & Dieckmann, U. (2015). Fisheries-induced evolution. *Annual Review of Ecology, Evolution, and Systematics*, 46, 461-480. https://doi.org/10.1146/annurev-ecolsys-112414-054339
- Conover, D. O., & Munch, S. B. (2002). Sustaining fisheries yields over evolutionary time scales. *Science*, 297(5578), 94-96. https://doi.org/10.1126/science.1074085
- Walsh, M. R., Munch, S. B., Chiba, S., & Conover, D. O. (2006). Maladaptive changes in multiple traits caused by fishing: impediments to population recovery. *Ecology Letters*, 9(2), 142-148. https://doi.org/10.1111/j.1461-0248.2005.00858.x
- Brander, K. M. (1995). The effect of temperature on growth of Atlantic cod (Gadus morhua L.). *ICES Journal of Marine Science*, 52(1), 1-10. (Note: original draft cited DOI `10.1016/1054-3139(95)80010-9` but that does not resolve in scite; cross-citations point to `10.1016/1054-3139(95)80010-7`. DOI requires direct verification before publication.)
- Mehner, T., & Wieser, W. (1994). Energetics and metabolic correlates of starvation in juvenile perch (Perca fluviatilis). *Journal of Fish Biology*, 45(2), 325-333.

Citations to be verified via the `scite` MCP server before tutorial publication.

## Related plans and specs

- `docs/superpowers/plans/2026-04-06-ev-osmose-genetics-core-plan.md` — original implementation plan for the wiring; this spec activates that work.
- `docs/superpowers/specs/2026-04-05-ev-osmose-economic-design.md` — parallel pattern for DSVM bioeconomic activation, shipped 2026-05-08 (backlog top-5 #2).
