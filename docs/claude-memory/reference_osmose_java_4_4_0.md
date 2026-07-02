---
name: reference-osmose-java-4-4-0
description: Official OSMOSE Java 4.4.0 release (2026-05-21) — new features + breaking config renames + parity-drift implications for OSMOPY
metadata: 
  node_type: memory
  type: reference
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

Official engine repo: **github.com/osmose-model/osmose** (Java + R/CRAN). Latest release **v4.4.0** (2026-05-21); feature PRs merged ~2026-04-23; `master` since is mostly notification/wiki chores. 4.4.0 fat-jar is a downloadable release asset: `osmose_4.4.0-jar-with-dependencies.jar`. Wiki has per-feature pages.

**OSMOPY bundles `osmose_4.3.3-jar-with-dependencies.jar`** (24MB, git-tracked in `osmose-java/`) → one minor version behind. The "Java engine stays unchanged" assumption in [[architecture]] is no longer true upstream; 4.4.0 diverged substantially.

**4.4.0 new capabilities (NOT in the OSMOPY Python engine — parity drift):** stochastic maturity ogive (`species.maturity.mode` stochastic/legacy, l50/l75); post-reproduction mortality (iteroparous/semelparous + postspawning survival); density-dependent mortality; region-aware mortality; gradient-based movements (random-walk coef + search radius); background-species overhaul + biomass multipliers (`species.multiplier[.log]`); new egg-size bioen (`species.egg.density`); simplified bioenergetics (`species.bioenergetics.model` full/simple) for data-poor species; LTL predation log `computePercent`; fishing/discards tracked in numbers; functional bioeconomics module; `simulation.fixed.seed.enabled`, `simulation.nschool.multiplier`; reworked restart.

**BREAKING config-key renames (mandatory for existing configs) — OSMOPY schema still emits the OLD keys:**
- `simulation.bioen.enabled`→`module.bioenergetics.enabled`; `simulation.genetic.enabled`→`module.genetics.enabled`; `fisheries.enabled`→`module.multispecies.fisheries.enabled`; `economy.enabled`→`module.bioeconomics.enabled`
- `output.restart.*`→`simulation.restart.*` (incl. `spinup`→`spinup.nyear`)
- bioen drops the `.bioen.` infix: `predation.ingestion.rate.max.bioen.spX`→`predation.ingestion.rate.max.spX`; `species.bioen.maturity.{eta,r,m0,m1}.spX`→`species.maturity.{...}.spX`
OSMOPY schema files emitting old keys: `schema/simulation.py` (bioen/genetic), `schema/fishing.py` (fisheries), `schema/economics.py` (economy), `schema/output.py` (restart), `schema/bioenergetics.py` (bioen.maturity).

**Implications for OSMOPY (see [[project_feature_improvements_backlog]] config-migration item):**
1. Java-core update to 4.4.0 is NOT a blind jar swap — config-format compat must be handled (the renamed keys). `runner.py` already wraps the Java `-update` migration (`_run_config_migration`, line ~251) — the likely bridge (Python writes old-format → `java -jar 4.4.0 -update` → run), IF 4.4.0 doesn't already auto-migrate old keys at load (UNVERIFIED — test empirically).
2. Hardcoded `4.3.3` refs to update on a swap: `ui/state.py:42` (default jar_path), `tests/test_state.py:79` (asserts it), `demo.py:228` (`target_version`); `ui/pages/run.py` globs `osmose-java/*.jar` so it auto-discovers. `osmose/engine/timeseries.py:4` docstring says "Matches Java OSMOSE 4.3.3".
3. Committed parity baselines are PYTHON-engine `.npz` (`tests/baselines/`, via `scripts/save_parity_baseline.py`) — unaffected by a JAR swap. Java cross-engine equivalence ("within 1 OoM") used 4.3.3.
