# Per-fishery fishing (Java parity) — design

Date: 2026-11-24. Status: APPROVED 2026-11-24; G5 IMPLEMENTED 2026-11-24 (no-op on shipped configs, no certification re-run needed) (staging: PR-2a Python path + C# refuses, PR-2b C# port; G5 normalisation fixed in its own PR after, with Baltic certification re-run). Context: the Baltic run lists 28 Java-only keys
(#123 follow-up, PR #382). Owner order: per-fishery maps first, then outputs + prefix, then a
separate staged-diet output. This spec covers the first item.

## 1. What is wrong today (verified in code)

Java (`FishingGear`, `FisheryMapSet`) computes, per school and step:

    F = sum over fisheries f of  q[sp, f] * rate_f(step) * sel_f(school) * map_f(cell, step) * season_f(step)

The Python engine (and the C# engine, which mirrors it) instead reduces fisheries to ONE set of
per-species arrays in `config._parse_fisheries`:

| Gap | Python today | Java |
|---|---|---|
| G1 species in >1 fishery | only the FIRST fishery with q > 0 is used | sum over all fisheries |
| G2 catchability value | used as a yes/no flag | multiplies the rate |
| G3 fishery maps | per SPECIES (`mortality.fishing.spatial.distrib.file.sp{i}`) with `fisheries.movement.file.map0` as a shared fallback; `fisheries.movement.fishery.mapN` (which fishery a map belongs to) is never read | per FISHERY, by name |
| G4 map schedule | none | each map applies on `fisheries.movement.steps.mapN` × years (`.initialyear/.lastyear/.years.mapN`); a step with no map switches the fishery OFF |
| G5 map normalisation | raw values | every map rescaled so its mean over valid ocean cells is 1 (`normalize_map`) |
| G6 duplicate maps | n/a | twin files merged (no behavioural effect) |

Shipped configs (`baltic`, `baltic_ev`, `baltic-fine`, `eec_full`): exactly one fishery per species,
q = 1, every map of a species identical, no map schedules → G1–G4 do not bite. **G5 does not
either (corrected 2026-11-24):** Java's mean is over VALID cells only (`isValueOk`: value > 0, not
NaN, not land), so zero cells are excluded and the Baltic 0/1 mask and EEC's all-ones map both
have mean exactly 1 — normalisation is a no-op for every shipped config. (An earlier draft said
"+0.65% on Baltic"; that mean wrongly counted the zero cells.)

## 2. Design

### 2.1 Two paths, chosen at config load

`EngineConfig` gains `fisheries: FisheryModel | None`.

- **Per-species path (unchanged):** used when the config is *species-reducible* — every species has
  at most one fishery with q > 0, that q is 1, all maps of a fishery are the same file, and no map
  has a step/year schedule that leaves a step uncovered. This is exactly Java's semantics for such
  configs, so the existing arrays, kernels and C# bundle are untouched. All shipped configs take it.
- **Per-fishery path (new, Python):** any other fisheries config. `FisheryModel` holds per fishery:
  base rate (+ rate-by-period/seasonality), selectivity (type, a50/l50/slope), catchability column
  `q[:, f]`, discard rate, and a map table `maps[k]` + `map_index[step, f]` (−1 = fishery off that
  step). The mortality rate per school becomes the sum above. The `fisheries.movement.fishery.mapN`
  / `.steps.mapN` / year keys are READ (they leave the Java-only list).

### 2.2 C# engine

- Stage 1 (this PR): `csharp_engine_block_reason` gains a detector `per_fishery_fishing` that refuses
  configs taking the per-fishery path; the over-broad `fisheries\..+` → phase1 rule is narrowed so a
  per-fishery feature cannot be "supported" by accident. Shipped configs keep running on C#.
- Stage 2 (follow-up PR): port the per-fishery path — bundle fields for the fishery tables (contract
  version 2 → 3, `features.json` `contract_versions` [2, 3]), `ModelReader.Fishing`,
  `StepPreparation.Rates` summing over fisheries, test replicas in `test_csharp_build.py`; gate:
  `scripts/csharp/families_equivalence.py` on a synthetic multi-fishery config.

### 2.3 Map normalisation (G5) — OWNER DECISION

Options:
- **A. Normalise as Java (parity).** Baltic fishing in fished cells +0.65%. Moves every absolute
  Baltic figure slightly; requires re-checking the certification (5/5 gate), and the SP1b `d0` guard
  must be re-solved if cod baselines move beyond its tolerance; Python and C# both change; the
  Python/C# equivalence stays (both move together).
- **B. Keep raw values; document the divergence.** No result changes; Baltic stays 0.65% below Java
  in fished cells.

Recommendation: **A, in its own PR after the per-fishery path**, with the certification re-run, so
the parity fix and the refactor are never confounded.

## 3. Verification

1. Shipped configs bit-identical before/after (Python and C#): they take the unchanged path.
2. Synthetic per-fishery configs vs hand-computed rates: two fisheries on one species (G1),
   q = 0.5 (G2), two maps by fishery name (G3), a map active on steps 0–5 only → fishery off on
   6–11 (G4). Each test mutation-checked.
3. Equivalence of the two paths: a species-reducible config forced through the per-fishery path
   gives bit-identical rates to the per-species path.
4. C#: shipped demos still run (`test_every_demo_runs_as_shipped`), per-fishery configs refused
   with a clear reason; full suite + C# bridge suite green.

## 4. Out of scope here

Outputs (`output.fisheries.enabled` yield-by-fishery is a natural follow-on once fisheries exist in
the engine), file prefix, staged diet — later PRs per the owner's order.
