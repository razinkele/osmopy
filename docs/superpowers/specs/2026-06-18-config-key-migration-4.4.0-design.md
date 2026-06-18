# Config-key migration to OSMOSE 4.4.0 — Design (v3)

**Date:** 2026-06-18
**Status:** Approved approach; revised after 3 in-loop review rounds (incl. scite + authoritative-source verification).

## Goal

Make OSMOSE 4.4.0 config-key names the **canonical** internal form in OSMOPY by **faithfully porting
OSMOSE's own migration step** (`Releases.java` release `$15` = v4.4.0) to Python, since the engine's
built-in `-update` is broken on our configs (crashes in `Release.commentParameter`, verified twice).
This lets OSMOPY emit 4.4.0-format configs natively — the prerequisite for bundling the 4.4.0 Java
engine (a separate, sequenced follow-up). See [[reference-osmose-java-4-4-0]].

**Authoritative source:** `osmose-model/osmose` →
`java/src/main/java/fr/ird/osmose/util/version/Releases.java`, the `$15` block (≈ lines 638-720+).
The Python migration must mirror its `updateKey(old, new)` calls and semantics — NOT the release-notes
table (which is an imperfect summary). Pull the live `$15` block at implementation time and port it.

## Split into two sequenced PRs

This touches ~15 files across core/engine/schema/UI/calibration, with two correctness-risky items (the
ingestion merge, the `-P` override mapping). Implement as two plans:

- **PR1 — core migration + tests (pure `osmose/`, parity-safe in isolation).** The chain entry +
  applier, `canonicalize_config`/`to_target_keys`, `key_case_map` rebuild, schema moves, engine
  literal-read updates, validators. Default write `target_version="4.3.3"` ⇒ provably no-op on disk.
- **PR2 — load/write wiring (UI + calibration).** `AppState.load_config`, reader hook, writer/`write_temp_config`
  `target_version`, the 3 calibration writes, `problem.py` `-P` override-key reverse-map, scenario
  load/save, deprecation UI.

## The 4.4.0 rename set (port verbatim from Releases.java `$15`)

`updateKey(OLD, NEW)` renames OLD→NEW **and skips if NEW already exists** (keep-existing-target). Reader
lowercases all keys, so match case-insensitively. Confirmed entries (verify the full block at impl time):

| OLD | NEW |
|---|---|
| `simulation.bioen.enabled` | `module.bioenergetics.enabled` |
| `simulation.genetic.enabled` | `module.genetics.enabled` |
| `fisheries.enabled` | `module.multispecies.fisheries.enabled` |
| `economy.enabled` | `module.bioeconomics.enabled` |
| `output.restart.enabled` | `simulation.restart.enabled` |
| `output.restart.recordfrequency.ndt` | `simulation.restart.recordfrequency.ndt` |
| `output.restart.spinup` | `simulation.restart.spinup.nyear` |
| `output.fishery.enabled` | `output.fisheries.enabled` |
| `output.fishery.byage.enabled` | `output.fisheries.byage.enabled` |
| `output.fishery.bysize.enabled` | `output.fisheries.bysize.enabled` |
| `output.spatial.fishery.enabled` | `output.spatial.fisheries.enabled` |
| `species.bioen.maturity.{eta,r,m0,m1}.sp{idx}` | `species.maturity.{eta,r,m0,m1}.sp{idx}` |
| `predation.ingestion.rate.max.bioen.sp{idx}` | `predation.ingestion.rate.max.sp{idx}` |
| `predation.coef.ingestion.rate.max.larvae.bioen.sp{idx}` | `predation.larval.ingestion.rate.increase.ratio.sp{idx}` |

These are **per-key (not blanket-prefix) renames**, exactly as `Releases.java` does them (each guarded by
`cfg.canFind(old)`), which avoids over-matching pre-existing keys.

## Merge / collision rule — `updateKey` semantics (keep-existing-target)

A rename is **skipped if the NEW key already exists** (Java's "already defined" behavior, verified). This
single rule resolves both the conflict question and the ingestion "merge":

- **Ingestion merge:** for a focal species that has BOTH `predation.ingestion.rate.max.sp{idx}` (legacy)
  and `predation.ingestion.rate.max.bioen.sp{idx}` (e.g. `baltic_ev` sp0: 3.5 and 3.0), the rename
  `bioen→base` is SKIPPED because base exists ⇒ **base value (3.5) is kept, the `.bioen.` value is
  dropped.** This matches 4.4.0, where ALL ingestion reads (`PredationMortality`, `ForagingMortality`,
  `EnergyBudget`, `BioenPredationMortality`) use the single `predation.ingestion.rate.max.sp{idx}`.
- Same skip-if-exists rule for any other old+new collision.

**This is a real, intended behavior change for bioen / Ev-OSMOSE configs** (the separate bioen ingestion
value no longer exists in 4.4.0). It is NOT a result-preserving rename. Consequences:
- **Engine reconciliation (PR1):** `osmose/engine/config.py` currently reads `predation.ingestion.rate.max.sp{idx}`
  (legacy, `:606`) AND `predation.ingestion.rate.max.bioen.sp{idx}` (bioen, `:1890/1912`) as two distinct
  values. After canonicalization there is ONE key; both reads must point at it. Update both reads to the
  unified key, matching 4.4.0's `EnergyBudget`/`ForagingMortality`/`PredationMortality`.
- **Bioen parity gate (PR1, blocking):** add an end-to-end bioen test asserting OSMOPY's unified-ingestion
  behavior matches the 4.4.0 intent (one ingestion value drives both predation and the energy budget),
  and document the change for Ev-OSMOSE users (changelog) — existing bioen configs will produce different
  numbers than under 4.3.x. This is the "verify + re-validate parity now" the approach requires.

## Architecture

**Forward (ingestion → canonical 4.4.0):** extend `osmose/demo.py::_MIGRATION_CHAIN` with the `$15`
("4.4.0") entry and extend the applier to support per-key renames + the skip-if-target-exists semantics
(today's applier does blanket-prefix + blind `result[new]=result.pop(old)` — it must (a) match per-key,
(b) skip when NEW exists, (c) comment/drop the OLD, (d) rebuild `key_case_map`). Bump `migrate_config`
default `target_version` to `"4.4.0"`. Expose `osmose.config.canonicalize_config(cfg) -> (dict, deprecated)`
(thin wrapper; logs each distinct deprecated old key once) so callers don't import from `demo`.

**Forward choke points (PR1 core readers; PR2 UI):** `reader.read_file()` (covers `read()` and the
`advanced.py` import path); `EngineConfig.from_dict` + `config_validation.py` + `validator.py`
(idempotent guard); `ui/state.py` new `AppState.load_config(cfg)` (PR2) routing demo (`grid.py:841`),
scenario (`scenarios.py:181`), and fishbase config-sets; also canonicalize in `ScenarioManager.load`
(`osmose/scenarios.py`) so the library API (compare/fork) is covered, not just the UI.

**Version hardening:** `migrate_config` keys off `osmose.version` via `_version_tuple`, which returns
`(0,)` on a non-numeric string (e.g. `4.4.0-SNAPSHOT`, empty) → would force-apply ALL chain steps. Harden:
treat an unparseable/missing version as "apply from the start ONLY where `cfg.canFind(old)`" (the per-key
`canFind` guard already prevents corrupting keys that aren't present), and never let a malformed version
cause a NEW-key config to be reverse-processed. Add tests for missing / `4.4.0` / `4.4.0-SNAPSHOT` / `4.2.1`.

**Inverse (canonical → target engine version), generalized:** `osmose.config.to_target_keys(cfg, target_version="4.3.3")`
derives its map by **inverting each `_MIGRATION_CHAIN` entry between the canonical version and the target**
(so a future `4.5.0` entry auto-extends the inverse — not a hand-coded 4.4.0 mirror). For `4.3.x`: reverse
the renames (per-key, NOT blanket prefix — e.g. `species.maturity.{eta,r,m0,m1}` reverse only those four
leaves, never a `species.maturity.` prefix that would corrupt the pre-existing growth keys
`species.maturity.size`/`age` at `schema/species.py:195/206`) and set `osmose.version=4.3.3`; for `4.4.0`:
identity + stamp. The ingestion merge is **lossy/irreversible** — the inverse emits the unified value under
the legacy `predation.ingestion.rate.max.sp{idx}` (a 4.3.3 jar reads it as the legacy key); register it as
an explicitly non-invertible step.

**Write sites (PR2) — route every Java-bound config through the inverse (default `4.3.3`):**
`OsmoseConfigWriter.write` (gains `target_version`, reverse-maps BEFORE its prefix-based `ROUTING` runs —
note: `module.*`/`output.fisheries.*` have no `ROUTING` prefix, so on a `4.4.0` target the routing table
must also gain those prefixes; for `4.3.3` the reverse-map restores old prefixes first, so routing is
unaffected — PR1 default keeps this a no-op); `ui/pages/run.py::write_temp_config` (the real Java run
path); `ui/pages/calibration_handlers.py` ×3; `ui/pages/advanced.py:201`; and
`osmose/calibration/problem.py::_run_java_subprocess` (≈451-454) — its `-P<key>=<value>` CLI overrides
carry canonical 4.4.0 keys (from `field.resolve_key`), so map the override KEYS to the target version too.

**Schema (correct scope):** move `key_pattern` to NEW names in `schema/{simulation,fishing,economics,output,bioenergetics}.py`.
`schema/species.py` is **read-only-verified** — its `species.maturity.size/age` keys do NOT move; they are
the collision targets the inverse must dodge (and the new `species.maturity.{eta,r,m0,m1}` share that
namespace — add a schema comment distinguishing the bioenergetic MRN params from the growth maturity
size/age). `output.fishery.*` lives in `schema/output.py`'s `_OUTPUT_ENABLE_FLAGS` list — rename those
**list entries**, not a `key_pattern=`.

**Engine literal reads:** update `osmose/engine/config.py` module toggles + the two ingestion reads
(`:606`, `:1890/1912`) to the unified `predation.ingestion.rate.max.sp{idx}` + maturity reads to new names.

**`key_case_map` rebuild:** the migration must synthesize a case entry for each new key and drop the
orphaned old entry (incl. dropping the `.bioen.` entry on the ingestion skip). Today's `migrate_config`
does not touch the case map — net-new requirement. All renamed keys are lowercase ⇒ identity casing safe.

## Data flow

`old-or-new config` → (reader `read_file` | `AppState.load_config` | `ScenarioManager.load` | `from_dict`/validators)
→ `canonicalize_config` (skip-if-exists, version-hardened) → **canonical 4.4.0 dict** (+ deprecation log,
+ one-time UI notification of the migrated keys) → any Java-bound write → `to_target_keys(cfg, target)`
(default `4.3.3` → old keys + version; `4.4.0` enabled by the jar-swap follow-up) → file or `-P` args.

## Reproducibility / user impact (from the review)

- **Non-bioen Python-engine results: unchanged** (purely nominal renames). Committed parity `.npz`
  baselines (BoB, non-bioen) are unaffected — no regeneration.
- **Bioen Python-engine results: intentionally change** (ingestion unification) — gated by the bioen
  parity test + changelog note. Java cross-engine parity refreshes at the jar-swap step.
- **Saved scenarios:** load canonicalizes; save persists canonical 4.4.0 keys — note the format change in
  scenario metadata; flag interop with older OSMOPY in the changelog.
- **Calibration:** `-P` override keys reverse-mapped (PR2); also map stored free-param/baseline/checkpoint
  keys through `canonicalize_config` on resume so a resumed calibration binds (add a checkpoint-resume test).

## Scope (YAGNI)

In: faithful port of the `Releases$15` rename set (incl. `output.spatial.fishery`, the larvae rename, the
ingestion merge) + skip-if-exists semantics; extend `migrate_config` (not a parallel system); forward
canonicalize at all choke points; generalized inverse at all write sites (default 4.3.3 = no-op);
schema moves (+ `species.py` reconciliation, `output.fishery` flag-list); engine read reconciliation;
`key_case_map` rebuild; version hardening; bioen parity gate. Out: new 4.4.0 features; jar swap; on-disk
rewrite of bundled example configs; a standalone user "migrate file" command.

## Testing

**PR1 (pure `osmose/`):**
- New `tests/test_config_migration_440.py`: port-fidelity of the `$15` rename set (each entry incl.
  `output.spatial.fishery`, indexed maturity, the larvae rename); **skip-if-target-exists** (ingestion:
  base present → kept, `.bioen.` dropped; base absent → renamed); idempotency on already-4.4.0;
  version-hardening (missing / `4.4.0` / `4.4.0-SNAPSHOT` / intermediate `4.2.1`); deprecation list +
  **log-once** (caplog); `key_case_map` rebuild incl. the merge-drop.
- Inverse `to_target_keys`: chain-inversion to `4.3.3` (per-key reverse, maturity leaf-scoped — assert it
  does NOT touch `species.maturity.size/age`; `spinup.nyear→spinup`); `4.4.0` identity + stamp; merge
  declared non-invertible.
- Round-trip fixpoint: old→canonical→`to_target_keys(4.3.3)`→old (for non-merge keys); confirm the merge
  is a stable fixpoint after the first pass (no `.bioen.` to re-merge).
- Engine: `EngineConfig.from_dict(old-key dict)` builds identically to a new-key dict; the unified
  ingestion read returns the kept base value; **bioen end-to-end parity gate** (the unified-ingestion run
  is internally consistent and matches the 4.4.0 intent).
- Validator: example configs stay warning-free post-canonicalize; `test_from_dict_warn_mode_clean_on_example_configs`
  green; verify each renamed key's allowlist source (schema vs `_SUPPLEMENTARY_ALLOWLIST`).
- Drift guard: every NEW key produced by the `$15` entry resolves to a real schema field (or a known
  flag-list/allowlist entry).
- Regression: `test_bioen_*`, `test_genetics_*`, `test_engine_fisheries`, `test_roundtrip`,
  `test_config_writer`, `test_state`, `test_demo`, `test_ui_load_scenarios` — re-check assertions that pin
  an emitted key name or call `migrate_config` with the (now-4.4.0) default.

**PR2 (UI/calibration):** every write site → on-disk config is OLD keys + `osmose.version=4.3.3` by default
(`write_temp_config` round-trip; a calibration `OsmoseConfigWriter.write`); `-P` override keys come out old
for a 4.3.3 target (new test); all load choke points (reader `read_file`, `AppState.load_config` for
demo/scenario/fishbase, `ScenarioManager.load`) yield canonical 4.4.0; scenario save→load round-trip;
checkpoint-resume across the rename.

**CI gates (both PRs):** pyright (clean `[dev]` venv `--pythonpath`) on touched modules incl. `demo.py`,
`config/*`, `engine/config.py`, the schema files, `ui/state.py`, `ui/pages/{run,advanced,calibration_handlers,grid,scenarios}.py`,
`calibration/problem.py`; ruff check + format; full `-m "not e2e"` (modulo known xdist flakes).

## Blocking pre-implementation gate (PR1)

Before writing the chain entry, pull the live `Releases.java` `$15` block and confirm: (a) the COMPLETE
rename list (this table may be missing entries below the lines inspected, e.g. genetics/economy/other
fishery/spatial); (b) the `updateKey` skip-if-target-exists semantics (the merge direction = base-wins);
(c) the fisheries toggle is `module.multispecies.fisheries.enabled`. The port must match the source, not
this table. (The crashing `-update` cannot be used to verify at runtime — read the source.)

## Follow-up (out of scope, enabled by this)

Bundle `osmose_4.4.0-jar-with-dependencies.jar`; flip the write-default `target_version` to `4.4.0` (+ add
`module.*`/`output.fisheries.*` to `OsmoseConfigWriter.ROUTING`); update the hardcoded `4.3.3` refs
(`ui/state.py:42`, `tests/test_state.py:79`, `demo.py` default); refresh Java cross-engine parity.
