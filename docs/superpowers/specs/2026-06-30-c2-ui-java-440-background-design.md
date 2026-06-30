# C2 — Run nbackground>0 configs on Java 4.4.1 from the UI — design

> Status: design (awaiting review) · 2026-06-30
> Make the UI's Java-engine path **version-aware** so a `simulation.nbackground > 0` config (Baltic,
> baltic_ev) runs on the **Java 4.4.1 jar** from the app — by extracting sub-project A's validated staging
> (currently in `scripts/baltic_440_smoke.py`) into a reusable module and wiring it into the run path.
> Sub-project **C2** of the 4.4.x cutover. Builds on **A** (Baltic background keys, merged) + **C1** (native
> configs, jar-derived write-target, merged `ac03be0`). **Baltic-specific scope** (chosen): the hand-authored
> GreySeal/Cormorant accessibility table from A. No Python-engine change.

## 1. Background (verified against the code)

- `java_engine_block_reason(config) -> str | None` (`osmose/runner.py`) is **version-agnostic**: it blocks any
  `nbackground > 0` config. Callers: the UI gate (`ui/pages/run.py:824`) and `osmose/engine_capabilities.py:65`.
- The UI Java path `_run_java_engine` (`run.py:347`) stages via `write_temp_config(..., target_version=
  target_version_for_jar(jar_path))` (line 369), then runs `OsmoseRunner(jar_path)` (line 390). The staging
  insertion point is **between** those two.
- `scripts/baltic_440_smoke.py` already contains the validated staging recipe (A): `inline_biomass_series`
  (materialize per-step total biomass from the predator NetCDF), `augment_accessibility` (author the
  GreySeal/Cormorant accessibility matrix), `_write_background_movement_maps`, the hand-authored `BG_ACCESS`
  table, plus `simulation.nschool.spN` and `output.cutoff.enabled=false` — all on the STAGED copy. It runs the
  4.4.1 jar to exit 0 with the predators feeding.

## 2. Goal & non-goals

**Goal:** from the UI, with the 4.4.1 jar selected, a Baltic / baltic_ev run on the Java engine is no longer
blocked — it stages the background species and runs to completion. With the 4.3.3 jar (or an unrecognised
background species) it stays blocked with a clear reason.

**Non-goals:** generalizing to arbitrary background species (Baltic-specific scope, chosen); BoB (C3); any
Python-engine behaviour change; authoring accessibility into the canonical source (A kept it staged-copy-only).

## 3. Design

### 3.1 Extract A's staging — `osmose/java_background_staging.py` (new module)
Move the staging logic out of `scripts/baltic_440_smoke.py` into a library module the UI can import (the
smoke script then imports from it — DRY, and its integration run proves the extraction is behaviour-identical):
- `BG_ACCESS: dict[str, dict[str, float]]` — the hand-authored prey→accessibility per background predator
  (GreySeal, Cormorant), verbatim from A.
- `inline_biomass_series(nc_path, varname) -> list[float]` — per-step domain-total biomass (moved verbatim).
- `augment_accessibility(csv_path, predators) -> None` — staged-copy accessibility authoring (moved verbatim).
- `_write_background_movement_maps(...)` — uniform all-sea movement maps (moved verbatim).
- `background_staging_supported(config) -> bool` — true iff **every** `species.type.spN == "background"`
  species name (`species.name.spN`) is a key in `BG_ACCESS`. Baltic/baltic_ev → true; an unknown background
  species → false.
- `stage_background_for_java(stage_dir: Path, raw_config: dict) -> None` — the orchestrator (A's
  `stage_and_run` minus the jar launch + validation): for each `type=background` species, append inline
  `species.biomass.spN` + `species.biomass.nsteps.year.spN` (from the staged predator NetCDF), `nschool`,
  movement-map keys to the staged master; author the accessibility + catchability/discards rows on the staged
  matrices; and write `output.cutoff.enabled=false` **into the staged config** (so the UI runner needs no
  special CLI flag — the 4.4.1 `OutputRegion.include` OOB with nbackground>0 is avoided). Idempotent /
  staged-copy only; never touches `data/`.

`scripts/baltic_440_smoke.py` keeps its CLI + validation (`assert_predators_feed`, `_read_biomass_means`) and
imports the staging helpers + `stage_background_for_java` from the new module.

### 3.2 Version-aware block — `java_engine_block_reason(config, jar_version: str | None = None)`
Add an optional `jar_version` parameter (back-compat default `None`). Logic:

| condition | result |
|---|---|
| `nbackground == 0` | allow (`None`) — unchanged |
| `nbackground > 0`, `jar_version` `< 4.4.0` or `None` | **block** — the existing reason |
| `nbackground > 0`, `jar_version` `>= 4.4.0`, `background_staging_supported(config)` | allow (`None`) |
| `nbackground > 0`, `jar_version` `>= 4.4.0`, NOT supported | **block** — "background species *X* is not staging-supported on the Java engine; use the Python engine" |

Version compare via `aliases._numeric_version`. Importing `background_staging_supported` into `runner.py` must
not create an import cycle (the staging module imports only stdlib + numpy/xarray + the reader, not `runner`).

**Callers** pass the jar version (derived from the selected jar via `target_version_for_jar`):
- `ui/pages/run.py:824` gate: `java_engine_block_reason(config, target_version_for_jar(Path(state.jar_path.get())))`.
- `osmose/engine_capabilities.py:65`: thread the jar version through (its caller supplies the selected jar; if
  none is available there, pass `None` → conservative block, preserving today's behaviour).

### 3.3 UI wiring — `_run_java_engine` (`run.py`)
After `write_temp_config` (line 369) produces `config_path` and before `OsmoseRunner` runs (line 390): if
`int(nbackground) > 0` **and** `_numeric_version(target_version_for_jar(jar_path)) >= (4,4,0)`, call
`stage_background_for_java(config_path.parent, raw_config)`. (The gate at §3.2 has already guaranteed the
config is staging-supported, so this only runs for supported configs.) The runner then runs the staged config.

## 4. Validation
- **Unit (`tests/test_java_background_staging.py`):** `background_staging_supported` (Baltic source → true; a
  synthetic config with an unknown `type=background` species → false); `augment_accessibility` /
  `inline_biomass_series` (moved from `tests/test_baltic_440_staging.py`, re-pointed at the new module, incl.
  the source-untouched guard).
- **Unit (`tests/test_runner.py` or `test_engine_capabilities.py`):** the §3.2 block matrix — all 4 rows
  (nbg=0 → allow; nbg>0 + 4.3.3 → block; nbg>0 + 4.4.1 + Baltic → allow; nbg>0 + 4.4.1 + unknown species →
  block).
- **Integration:** `scripts/baltic_440_smoke.py` (now importing the module) still runs the 4.4.1 jar → exit 0,
  no focal collapse, predators feed — proving the extraction is behaviour-identical to A.
- **UI:** a `run.py` handler/render test that the gate does NOT block Baltic with the 4.4.1 jar and DOES block
  it with the 4.3.3 jar (assert `java_engine_block_reason` return through the gate's jar-version wiring).
- Full suite stays green (the moved tests + the new signature's `jar_version=None` default keep existing
  callers working).

## 5. Risks
- **Allow/block correctness:** allowing a config the staging can't handle launches a doomed Java run. The
  `background_staging_supported` predicate is the guard; the integration test confirms the supported path runs.
- **Import cycle:** `runner.py` importing `background_staging_supported` from the staging module — the staging
  module must NOT import `runner`/`ui`. Keep its deps to the reader + numpy/xarray.
- **`engine_capabilities` caller:** if it can't supply the selected jar, default `jar_version=None` →
  conservative block (today's behaviour) — no regression, just no new capability surfaced there.

## 6. Out of scope
Non-Baltic background species; BoB (C3); Python-engine changes; source-matrix authoring; the predator-NetCDF
data itself (Baltic ships `baltic_predator_biomass.nc`).
