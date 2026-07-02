---
name: project-baltic-440-background-adaptation
description: Baltic now loads+runs on Java 4.4.1 (sub-project A shipped); the 4 non-obvious 4.4.1 background-species jar requirements + sub-project C deferred
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

**Baltic runs on Java OSMOSE 4.4.1** — sub-project A SHIPPED 2026-06-29, merged to master `072b388` (local; not pushed — push-on-request). Adapts the Baltic example (2 background predators: GreySeal sp14, Cormorant sp15) to load+run on the 4.4.1 jar. Validated: 3-yr run exit 0, no focal collapse, predators feed (53 non-zero diet values). NO parity claim, NO Python-engine change, 4.3.3 default untouched, `data/baltic/*` byte-identical (automated guard).

**Two pieces:**
- `osmose/config/aliases.py::_emit_background_species_keys` — emits `species.multiplier` (SCALAR "1", not "1;1" — jar reads via getFloat) + `species.beta` ("1") for `type=background` species, ONLY on the 4.4.x write path (`>=4.4.0` branch of `to_target_keys`).
- `scripts/baltic_440_smoke.py` — staged-copy harness (everything authored into the staged 4.4.x copy, source untouched).

**The 4 non-obvious 4.4.1 jar requirements for background species (bytecode-confirmed; these WILL re-bite sub-project C / any 4.4.x config with background predators):**
1. **Biomass is an INLINE numeric array** `species.biomass.spN` (per-step domain-TOTAL, length=ndt; jar splits across nclass internally via `species.size.proportion` — `BackgroundSpecies$Proportion`). NOT the resource NetCDF keys (`species.biomass.{mode,file,varname}` are the `type=resource` `ResourceForcing` path); `species.file.spN`/the `.nc` is NEVER read for background. Materialize from the NetCDF at stage time.
2. **Predation-accessibility matrix** must include the background predators as columns (authored access values) AND rows. The jar builds the prey universe from the accessibility matrix, then **requires every such row in the catchability AND discards matrices too** (add ZERO rows — registered but not fished; fishing is focal-only via `Matrix.getFocalIndex`/`isFishable`). The original 4.3.3 `java_engine_block_reason` blocker, unchanged in 4.4.1.
3. **`simulation.nschool.spN`** is mandatory for background species (`BackgroundProcess.init`).
4. **Explicit `movement.{species,file,steps,class}.mapN`** — 4.4.1 `BackgroundMapDistribution` has NO random/default mode (used uniform all-sea maps). And **`output.cutoff.enabled=true` throws ArrayIndexOutOfBounds with nbackground>0** (`OutputRegion.include` indexes a focal-sized cutoffAge array with resource file-indices) → workaround `-Poutput.cutoff.enabled=false`.

**Sub-project C — "standardise the Python setup to 4.4.x" — DEFERRED to a separate cycle** (A is its prerequisite). User-confirmed C scope: (a) flip default write-target 4.3.3→4.4.x; (b) migrate bundled config masters to native 4.4.x; (c) Python engine reads 4.4.x semantics natively (not only write path); (d) UI 4.4.1 engine + version-aware `java_engine_block_reason` + full cross-engine re-validation. C also needs the BoB 365-step-forcing fix so every config loads on 4.4.1. EEC-full already runs on 4.4.1 (resource-only, [[project-config-key-migration-440]]); Baltic now too. Java-block guard [[project-baltic-java-engine-guard]] is still version-agnostic (blocks nbackground>0 in the UI) — C makes it version-aware.
