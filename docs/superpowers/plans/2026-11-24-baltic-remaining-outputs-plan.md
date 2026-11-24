# Baltic remaining Java-only keys — outputs, prefix, staged diet (plan)

Date: 2026-11-24. Owner decisions (2026-11-24): implement in Python AND C#; prefix with readers
auto-detecting; staged diet as a SEPARATE output; order maps → outputs → diet. Maps + map
normalisation shipped (v1.14.0, v1.14.1). This plan covers the 19 keys Baltic still lists.

## Hard constraint
All 19 keys are set (to true / staged) in Baltic and EEC. "Python first, C# refuses" would take
Baltic off the C# engine on prod. So each family ships Python collector + C# collector +
classification (`engine_cs/capabilities.py`) + implemented-features pin in ONE PR.

## Java facts (OutputManager / DistribOutput / WeightedDistribOutput; local tree predates 4.4.0)
| key | Java output | kind |
|---|---|---|
| output.biomass.bytl.enabled | `Indicators/biomassDistribByTL` | DistribOutput by TL class |
| output.meansize.byage.enabled | `AgeIndicators/meanSizeDistribByAge` | WeightedDistribOutput |
| output.meantl.bysize.enabled | `Trophic/meanTLDistribBySize` | WeightedDistribOutput |
| output.yield.biomass.bysize.enabled | `SizeIndicators/yieldDistribBySize` | DistribOutput on deaths |
| output.size.catch.enabled | `SizeIndicators/meanSizeCatch` | WeightedSpeciesOutput (per-species MEAN) |
| output.tl.catch.enabled | `Trophic/meanTLCatch` | WeightedSpeciesOutput (MEAN) |
| output.mortality.perspecies.bysize.enabled | per-species mortality by size | (read MortalitySpeciesOutput) |
| output.fisheries.enabled | yieldByFishery (NetCDF) | per-fishery split of fishing deaths |
| output.diet.stage.structure + thresholds | stage-split diet + `BiomassDietStageOutput` | diet by predator (and prey?) stage |

- DistribOutput: ALL alive schools, NO output cutoff; summed over the record period, divided by
  the number of steps; region selectivity applies; first column = lower class threshold
  (class 0 → 0), "Class i designates interval [i,i+1[".
- WeightedDistribOutput: NO cutoff; RATIO OF SUMS over the period (Σ w·x / Σ w), NaN where Σ w = 0.
  → carry numerator and denominator per step; `_average_step_outputs` must SUM both, divide at write.
- Baltic sets PER-SPECIES bins (`output.distrib.by{age,size,tl}.{min,max,incr}.sp{i}`); the engine
  reads global size bins only and these keys are UNKNOWN today — part of these families.
- Weighted means of the existing species outputs (meanTL, meanSize) are averaged per step in
  `_average_step_outputs`, not ratio of sums: check vs Java when touching them (separate issue).

## PR sequence
A state-derived: biomass.bytl, meansize.byage, meantl.bysize (+ per-species bins).
B death-derived: size.catch, tl.catch, yield.biomass.bysize, mortality.perspecies.bysize.
C fisheries: per-fishery split of fishing deaths (each term's share of the summed rate on the
  per-fishery path; whole fishing on the per-species path), carried from rate prep in both engines.
D output.file.prefix: write with the configured prefix; readers detect deterministically
  (config prefix → single prefix present → fail loudly). Grep every `osm` literal in osmose/, ui/,
  scripts/, tests/ and the C# writer; gate: one certification row + one calibration objective
  identical before/after and nonzero (silent failure mode: an empty frame scores zeros).
E staged diet: predator stage from the per-school matrix; prey stage needs booking at predation
  time in both kernels (hot path, bit-exact parity baselines, C# fixed width) — verify Java stages
  both axes, measure cost, ask the owner if material.

## Per family checklist
- gate on its switch AND output.minimal.enabled; no RNG; no state mutation
- disk writer AND in-memory `from_outputs`; record frequency; output.regions
- key out of `_ALLOWLIST_JAVA_ONLY` → PY_HONORED; frozen snapshot; C# classification; pin
- measure Baltic step cost (switches are on by default); calibration/certification run minimal
- C#: deterministic output probe if one exists (byte-for-byte on a captured state), then
  `families_equivalence.py`; its log floor (1.0) hides values < 1 → own comparison for rates
- non-vacuity guard per family (nonzero catch, >1 populated class) — three C# checks were vacuous
- pre-push: ruff, pyright, full pytest, C# build, bridge suite, dotnet test, dotnet format

## Control (done 2026-11-24)
Unmodified EEC, 2 yr, Python vs C# total yield: seed 7 1.082, seed 8 1.008, seed 9 0.997 → the 8%
seen on the gradient-map run (seed 7) is seed noise, not a defect.
