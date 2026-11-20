# OSMOSE 4.4.x parity status (issue #137)

> Status as of 2026-11-06. Sources: the upstream v4.4.0 release notes, and the v4.4.1 Java source
> for `ResourceForcing`, `BackgroundSpecies`, `BackgroundSchool`, `DietOutput` and
> `PredatorPressureOutput` (read from `osmose-model/osmose@v4.4.1`). The local tree at
> `~/osmose-reference/osmose-master` **predates 4.4.0** (`BackgroundSchool` is not a `School`
> there, and it still reads `species.file.spN`), so don't use it for 4.4.x questions.

osmopy's original engine parity claim (14/14 EEC, 8/8 Bay of Biscay) was measured against Java
**4.3.3**. Against 4.4.1, EEC still holds at 14/14 within 1 OoM, but the Bay of Biscay is 7/8:
Hake's biomass runs ~123× Java's (`docs/parity-roadmap.md`).
Since then the repo has gained 4.4.x *config and staging* support: it can write configs a 4.4.1 jar
runs, and it reads native 4.4.x configs. The 4.4.0 *process* changes are not ported. This table
tracks each item.

Status: **done** · **partial** · **TODO** · **N/A** (no Python counterpart is needed).

## Configuration

| 4.4.x change | Status | Where / evidence |
|---|---|---|
| Module-toggle, restart, output and bioen key renames (release notes §1.1) | done | `RENAMES_440` in `osmose/config/aliases.py`. `canonicalize_config` migrates old configs; `to_target_keys` writes the renamed keys for a 4.4.x jar |
| `mortality.additional.larva.rate` is per year in 4.4.0 | done | The reader divides by `ndtperyear`, and the writer multiplies back for 4.4.x (`osmose/config/reader.py`; CLAUDE.md, SP1b gotcha) |
| `species.lmax` removed; `species.beta` read only under bioen | done | `osmose/config/aliases.py` (removed when writing for 4.4.x) |
| Resource forcing `species.biomass.file.spN` (4.4.1 rename of `species.file.spN`) | done | Written for 4.4.x by `_emit_resource_biomass_forcing`, along with `mode`, `varname` and `nsteps.year`. **Read by the Python engine since 2026-11-06** (`osmose/engine/resources.py`; either key works) |
| `species.biomass.varname.spN` (NetCDF variable other than the species name) | TODO | Python looks the variable up by `species.name`. The key is allowlisted as Java-only, so a config that relies on a different name gets a warning rather than silent misreading |
| `species.biomass.mode` / `species.biomass.constant.spN` (UNIFORM forcing) | partial | Python has its own uniform path (`species.biomass.total.spN`). The 4.4.1 keys are allowlisted as Java-only |
| Background `species.multiplier.spN` | done (2026-11-06) | **Java arms ran Cormorant at 1× instead of 2× before this.** The staged inline series is the raw NetCDF, and `species.multiplier` defaulted to 1. Now carried from `species.biomass.multiplier.spN`, and the Python engine reads either key |
| Background inline `species.biomass.spN` series | done (staging) | `osmose/java_background_staging.py`; Baltic-specific tables |
| `species.multiplier.log.spN` | TODO | Not read |
| `simulation.fixed.seed.enabled` | done (2026-11-19) | 4.4.0 replaces five per-process keys by this one. The Python engine reads two of the five (`movement.randomseed.fixed`, `stochastic.mortality.randomseed.fixed`), and the new key now sets both. The old keys are still read: configs of both generations are run. Cross-engine runs are never bit-equal anyway (PCG64 vs MT19937) |
| `simulation.nschool.multiplier` | TODO | Not read |
| `simulation.kill.if.no.school.enabled` (4.4.1 debug switch) | done (2026-11-19) | After the seeding period (`population.seeding.year.max`, or the longest lifespan), a step that ends with a focal species holding no fish stops the run: `osmose.engine.SpeciesCollapsed`, where Java exits with status 1 (`SimulationStep.step`). The C# engine has no such check and refuses a config that switches it on |
| `population.genotype.transmission.year.start` (genetics) | done (2026-11-20) | Java's name for the first year in which offspring inherit (`ReproductionProcess.init`). The Python engine read only its own `evolution.seeding.year`; it now reads Java's key first and its own as the fallback. **Not ported:** Java refuses a run in which a coding trait (`m0`, `m1`, `imax`, `r`) evolves and this year precedes `population.seeding.year.max`. The shipped `baltic_ev` has 10 against 60 and would not load under that rule |
| `species.is.enabled.spX` (4.4.1 debug switch) | done (2026-11-20) | `drop_disabled_species` in `osmose/engine/config.py`, applied in `EngineConfig.from_dict` right after key canonicalisation. Java drops the species and numbers the rest in order while keys keep their file index; this engine numbers focal species 0..n-1 from the keys, so it renumbers: the disabled species' `.sp<i>` keys go, later focal species move down, background and resource keys (`i >= n`) stay, and the species' movement maps go (fishery maps sharing the map numbering stay). Name-keyed matrices (accessibility, catchability) just no longer use its row. A run with the switch equals the config written without the species, bit for bit. The C# engine receives the renumbered config |

## Processes

| 4.4.x change | Status | Notes |
|---|---|---|
| Region-aware mortality (`nDead[region][cause]`) | TODO | Python books deaths per school and cause, with no output regions |
| Fishing and discards tracked in numbers | N/A | An internal Java refactor. Python already removes fish as numbers (`n_dead`) and writes both `yield` and `yieldN` |
| Genetics on the standard (non-bioenergetic) reproduction path | done (2026-11-20, #338) | Java's `ReproductionProcess.run_genet` draws the two parents of an offspring school with the weight of each mature school's egg production, and gives fresh genotypes to a species bootstrapped from its seeding biomass. The Python engine chose parents by gonad weight on both paths, which only bioenergetics sets, so genetics without bioenergetics stopped at its first reproduction outside the genetics seeding window. It now follows Java (`tests/test_genetics_standard_path.py`) |
| Stochastic maturity ogive (`species.maturity.mode`, `l50`/`l75`) | TODO | Python maturity is deterministic (length and age thresholds) |
| Post-reproduction mortality (`species.reproduction.strategy`, `postspawning.survivaltime`) | TODO | |
| Simplified bioenergetics (`species.bioenergetics.model = simple`, `species.temperature.tmin/topt/tmax`) | TODO | Python implements the full model only |
| Spherical egg size (`species.egg.density`) | TODO | |
| Gradient-based movement (`movement.random.walk.coef`, `movement.base.search.radius`) | TODO | |
| Background movement from NetCDF maps (`movement.netcdf.enabled`) | TODO | `movement.randomwalk.range` is read |
| Log-form LTL `computePercent` (`simulation.resources.computePercent.legacy`) | TODO | Python uses the legacy linear form |
| Bioeconomics (elasticities, costs, demand) | partial | Python has its own economics module (`module.bioeconomics.enabled`); the 4.4.0 parameters are not read |

## Outputs

| 4.4.x change | Status | Notes |
|---|---|---|
| Background species always included | done | Biomass and abundance tables span `all_species_names`. `predatorPressure` gained background predators in #198 and `dietMatrix` in #207. Java 4.4.1's `DietOutput` writes them; its `PredatorPressureOutput` accumulates them but prints focal predator columns only. The PREY axis of both outputs is `DietOutput`'s since #322: focal species, background species, resources (a background prey has its own column) |
| Ragged 4.4.1 mortality header | done | #141 |
| NetCDF chunking default | N/A | Writer-side Java detail |

## Java-side defects worked around in staging

| Defect | Versions | Workaround |
|---|---|---|
| `BackgroundSchool.getWeight()` returns grams where every consumer expects tonnes, so a background school holds 10⁶× too few animals and is emptied (biomass and predation) the first time anything increments its deaths; the fishery module does so every step | v4.4.0, v4.4.1, upstream `master` as of 2026-06-30 | `stage_background_for_java` scales the background `species.length2weight.condition.factor` ×10⁻⁶ in place. Remove it when a jar returns tonnes. Details: `docs/baltic_java_crosscheck_2026-11-06.md` §3 |
| A key defined twice keeps its FIRST value (`Configuration.java:921-929`) | all | Staging rewrites keys in place or passes `-P`; an appended correction is ignored |

## Phase plan (from the issue)

- **Phase 1** (config alignment, low risk): this document; the resource-forcing and background
  multiplier key alignment (done 2026-11-06); `simulation.fixed.seed.enabled` and the
  `kill.if.no.school` switch (done 2026-11-19); `species.is.enabled`, a renumbering pass over
  the config (done 2026-11-20, see its row above). Phase 1 is complete.
- **Phase 2** (process parity): region-aware mortality, stochastic maturity, post-reproduction
  mortality. Each changes engine outputs and needs its own parity run.
- **Phase 3** (extensions): simplified bioenergetics, gradient movement, log `computePercent`,
  egg density.

Any Java cross-check involving the Baltic background predators (`certify_java`, the UI Java path,
`scripts/baltic_440_smoke.py`) that ran before 2026-11-06 had Cormorant at half the Python engine's
biomass, and any that ran before 2026-11-07 had its seal and cormorant schools emptied mid-step by
the weight-units defect above. The combined correction is at most 17% on any species
(`docs/baltic_java_crosscheck_2026-11-06.md` §3); re-run before comparing seal and cormorant
effects across engines.
