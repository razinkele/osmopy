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
| `species.biomass.varname.spN` (NetCDF variable other than the species name) | done (2026-11-20) | `ResourceSpeciesInfo.forcing_var`, read by the resource update and by the C# engine's bundle. Without the key the variable is the resource's name, as before. A name missing from the file is refused (Java fails on it too) |
| `species.biomass.mode` / `species.biomass.constant.spN` (UNIFORM forcing) | partial | Python has its own uniform path (`species.biomass.total.spN`). The 4.4.1 keys are allowlisted as Java-only |
| Background `species.multiplier.spN` | done (2026-11-06) | **Java arms ran Cormorant at 1× instead of 2× before this.** The staged inline series is the raw NetCDF, and `species.multiplier` defaulted to 1. Now carried from `species.biomass.multiplier.spN`, and the Python engine reads either key |
| Background inline `species.biomass.spN` series | done (staging) | `osmose/java_background_staging.py`; Baltic-specific tables |
| `species.multiplier.log.spN` | done (2026-11-20) | A background species' multiplier given as its logarithm (`BackgroundSpecies.java`, v4.4.1 lines 223-238); refused together with `species.multiplier.spN`, as in Java |
| `simulation.fixed.seed.enabled` | done (2026-11-19) | 4.4.0 replaces five per-process keys by this one. The Python engine reads two of the five (`movement.randomseed.fixed`, `stochastic.mortality.randomseed.fixed`), and the new key now sets both. The old keys are still read: configs of both generations are run. Cross-engine runs are never bit-equal anyway (PCG64 vs MT19937) |
| `simulation.nschool.multiplier` | done (2026-11-20) | `(int) (multiplier * nschool)` per focal species, a truncation (`Configuration.java`, lines 583-591). A count that falls to 0 is refused. **Not changed:** without `simulation.nschool` the Python default is 20 schools, Java's is 10 (`nSchoolDef`); every shipped config sets the key |
| `simulation.kill.if.no.school.enabled` (4.4.1 debug switch) | done (2026-11-19) | After the seeding period (`population.seeding.year.max`, or the longest lifespan), a step that ends with a focal species holding no fish stops the run: `osmose.engine.SpeciesCollapsed`, where Java exits with status 1 (`SimulationStep.step`). The C# engine has no such check and refuses a config that switches it on |
| `population.genotype.transmission.year.start` (genetics) | done (2026-11-20) | Java's name for the first year in which offspring inherit (`ReproductionProcess.init`). The Python engine read only its own `evolution.seeding.year`; it now reads Java's key first and its own as the fallback. **Not ported:** Java refuses a run in which a coding trait (`m0`, `m1`, `imax`, `r`) evolves and this year precedes `population.seeding.year.max`. The shipped `baltic_ev` has 10 against 60 and would not load under that rule |
| `species.is.enabled.spX` (4.4.1 debug switch) | done (2026-11-20) | Read as Java's `Boolean.valueOf`: only `true` (any case) enables; an empty value is refused. `drop_disabled_species` in `osmose/engine/config.py`, applied in `EngineConfig.from_dict` right after key canonicalisation. Java drops the species and numbers the rest in order while keys keep their file index; this engine numbers focal species 0..n-1 from the keys, so it renumbers: the disabled species' `.sp<i>` keys go, later focal species move down, background and resource keys (`i >= n`) stay, and the species' movement maps go (fishery maps sharing the map numbering stay). Name-keyed matrices (accessibility, catchability) just no longer use its row. A run with the switch equals the config written without the species, bit for bit. The C# engine receives the renumbered config |

## Processes

| 4.4.x change | Status | Notes |
|---|---|---|
| Region-aware mortality (`nDead[region][cause]`) | TODO | Python books deaths per school and cause, with no output regions |
| Fishing and discards tracked in numbers | N/A | An internal Java refactor. Python already removes fish as numbers (`n_dead`) and writes both `yield` and `yieldN` |
| Genetics on the standard (non-bioenergetic) reproduction path | done (2026-11-20, #338) | Java's `ReproductionProcess.run_genet` draws the two parents of an offspring school with the weight of each mature school's egg production, and gives fresh genotypes to a species bootstrapped from its seeding biomass. The Python engine chose parents by gonad weight on both paths, which only bioenergetics sets, so genetics without bioenergetics stopped at its first reproduction outside the genetics seeding window. It now follows Java (`tests/test_genetics_standard_path.py`). **Under bioenergetics too, parents are now weighted by the eggs they release** (Java's `nEgg` in `BioenReproductionProcess.run`), not by the gonad left after release; this changed the parent draws of every bioenergetic genetics run (`baltic_ev`, the FIE demos) in #343 |
| Stochastic maturity ogive (`species.maturity.mode`, `l50`/`l75`) | done (2026-11-20, Python engine) | `osmose/engine/processes/maturity.py`. As `Species.isSexuallyMature`: a school not yet mature matures this step with probability (F(L) - F(L0)) / (1 - F(L0)), F the normal CDF with mean l50 and sd (l75 - l50) / 0.6745; once mature, always mature; drawn in reproduction, only for species that reproduce; ignored under bioenergetics, as in Java. Reproduction, the life-stage output and SSB share one maturity test. A stochastic species without l50 and l75, or with l75 <= l50, is refused. The C# engine refuses the mode (feature `maturity_stochastic`). A school that is already far past l50 when first tested (F(L0) = 1 in double precision, about 8 sd) gets a probability of 0/0 and never matures, as in Java |
| Ninth mortality cause, SPAWNING (`MortalityCause.SPAWNING`) | done (2026-11-20) | Java 4.4.x has nine causes and every mortality output loops over all of them, so its files carry a ninth stage triple whatever the config (its header names only eight; #141, where the reader files it as an unnamed cause). The engine carries the cause too: `n_dead`, the mortality CSVs and the NetCDF `cause` axis gain a Spawning column, 0 until a species dies after spawning. The C# engine carries the ninth cause too (`RunResults.MortalityCauses` = 9), never booked, since it refuses semelparous species. The regenerated parity baselines equal the old ones plus that zero column |
| Post-reproduction mortality (`species.reproduction.strategy`, `postspawning.survivaltime`) | TODO | |
| Simplified bioenergetics (`species.bioenergetics.model`, `species.temperature.tmin/topt/tmax`) | done (2026-11-20, Python engine) | Java 4.4.x (`EnergyBudget.computeEnetLite`): a species whose key is set to anything but `full` (Java tests `equals("full")`, so `Full` is lite too) takes E_net = assimilation × (ingestion × f_T × f_O₂ − Imax / steps per year × θ × critical efficiency × w[g]^β × N × 10⁻⁶), with θ the larval ingestion ratio below the larval threshold age, no φ_T and no maintenance term. f_T = (T − T1)(T − T2) / ((T − T1)(T − T2) − (T − Topt)²), the roots T1, T2 derived from tmin, tmax, topt and the critical efficiency, so that f_T is 1 at Topt and equals the critical efficiency at tmin and tmax. `energy_terms_lite` in `osmose/engine/processes/energy_budget.py`; the step computes the terms once per species and hands them to `compute_energy_budget`. Java never sets E_gross or E_maint for a lite school, so those outputs read 0 for it, as here. A lite species without its three temperatures, with tmax ≤ tmin, or with a critical efficiency outside (0, 1) is refused, where Java's arithmetic divides by 0 or returns NaN. The C# engine refuses bioenergetics as a whole |
| Spherical egg size (`species.egg.density`) | N/A (Java defect) | Read only under bioenergetics, where Java takes the egg length from `computeEggLength` (`Species.java`, v4.4.1 line 388): `Math.pow(6 * weight / (Math.PI * eggDensity), (1 / 3))`. `1 / 3` is integer division in Java, so the power is 0 and **every Java 4.4.1 bioenergetic egg is 1 cm long**, whatever the species and the density. Without bioenergetics the egg size is `species.egg.size` and the key is unused. Nothing to port; a bioenergetic comparison against the 4.4.1 jar differs at the egg for this reason |
| Gradient-based movement (`movement.random.walk.coef`, `movement.base.search.radius`) | TODO | |
| Background movement from NetCDF maps (`movement.netcdf.enabled`) | TODO | `movement.randomwalk.range` is read |
| Log-form LTL `computePercent` (`simulation.resources.computePercent.legacy`) | done (2026-11-20, Python engine) | `osmose/engine/processes/resource_percent.py`, used by both mortality paths. Absent or `true` keeps the linear form, as Java does. A resource whose smallest size is not above 0 is refused in log form (Java would silently never feed on it). The C# engine refuses the key (feature `resource_percent_log`) |
| Density-dependent larval mortality (`mortality.additional.larva.referencedensity.sp#`) | done (2026-11-20, Python engine) | Java 4.4.x (`AnnualLarvaMortality`, `ByDtLarvaMortality`): the larval rate of a step becomes M × (1 + (ln 2 / M) × SSB / reference), that is M + ln 2 × SSB / reference, so larval survival halves when the spawning stock equals the reference. Java applies larval mortality to eggs before the sub-step loop, so SSB is the total taken at the start of the step after movement: every mature focal school, out-of-domain schools included. `osmose/engine/processes/natural.py`. A reference below 0 (or absent) is off, as in Java; 0 is refused. Where M is 0 Java computes 0 × ∞, not a number; the limit is used. Java totals the `isMature()` flag set at the previous step's reproduction, before the age increment, so a school crossing a maturity age counts one step later there, and a species that does not reproduce never counts; the engine tests the current state. Refused with bioenergetics, where Java's flag follows the m0/m1 line, which the engine does not latch. The C# engine refuses the key (feature `larva_density_dependence`) |
| Additional-mortality multiplier and seasonality (`mortality.additional.rate.multiplier.sp#`, `mortality.additional.rate.seasonality(.file).sp#`, and the same under `mortality.additional.larva.rate.` and `mortality.additional.neonate.rate.`); the neonate larval scenario (`mortality.additional.neonate.rate.sp#`) | done (2026-11-20, Python engine) | Java 4.4.x (`AnnualAdditionalMortality`, `AnnualLarvaMortality`): the rate of a step is multiplier × base × season / steps per year, with multiplier and season 1 when unset (`SeasonTimeSeries`: one value per step of a year, repeated, or one per step of the run, or a file; not normalised). `mortality_rate_factor` in `osmose/engine/config.py` builds the factor per species and step; it is applied where the rate is resolved (additional mortality in both mortality paths and the per-cell path, larval mortality), and only to species without a by-step rate, as Java reads it only in the constant scenarios. A season of another length is refused, where Java would index past its array. The neonate scenario (Java's VCONSTANT) is the larval process under its own keys: its rate becomes the larval rate (per year in 4.4.x, converted like the larval one), a species setting both is refused as in Java, and its reference density is ignored, as `AnnualLarvaMortality` reads it only for the larva stage. The SP1b drift guard refuses a cod stock with a larval multiplier or seasonality, whose live rate is no longer the scalar it was solved against. The warm-start initial population (`module.population.initialisation.enabled`, `initialization.py`) still builds its age structure from the base rate, without the multiplier or seasonality. The C# engine refuses the multiplier and seasonality (feature `mortality_rate_factor`) and runs the neonate rate |
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
