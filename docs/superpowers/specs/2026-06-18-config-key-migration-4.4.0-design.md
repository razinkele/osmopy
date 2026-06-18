# Config-key migration to OSMOSE 4.4.0 — Design (v2)

**Date:** 2026-06-18
**Status:** Approved approach (canonical = 4.4.0 keys); revised after a 4-angle in-loop spec review.

## Goal

Make OSMOSE 4.4.0 config-key names the **canonical** internal form throughout OSMOPY, reading OLD
(4.3.x) keys via the migration chain and writing keys for a chosen **target engine version**. This lets
OSMOPY emit 4.4.0-format configs natively — the prerequisite for bundling the 4.4.0 Java engine (a
separate, sequenced follow-up), since 4.4.0 refuses 4.3.x configs and its own `-update` migration
crashes on ours (see [[reference-osmose-java-4-4-0]]).

**This feature is the rename layer only** — not the new 4.4.0 biological features, not the jar swap.

## Why v2 (review findings that reshaped the design)

1. **Reconcile with the EXISTING migration system.** `osmose/demo.py::migrate_config` + `_MIGRATION_CHAIN`
   already does sequential, version-stamped, prefix-based old→new renames (up to 4.3.0), invoked at
   `ui/pages/grid.py:841` on demo load. We **extend this chain** with a `"4.4.0"` step rather than add a
   parallel `aliases.py` canonicalizer. `migrate_config(cfg, target_version="4.4.0")` IS the forward
   canonicalizer; a new inverse `to_target_keys` handles writing for older engines.
2. **The Java run path is `write_temp_config` (`ui/pages/run.py:131`), NOT `OsmoseConfigWriter`.** Plus
   un-targeted `OsmoseConfigWriter.write` calls in `ui/pages/calibration_handlers.py` (×3) and
   `ui/pages/advanced.py:201`. EVERY write-to-disk site must reverse-map to the target engine version.
3. **`state.config` is fed from many load sites** (demo load, `advanced.py` `read_file` import,
   scenario load, fishbase bootstrap, per-field edits). Canonicalization must cover all of them, via a
   single chokepoint.
4. **Key collision (bioen-ingestion merge):** `predation.ingestion.rate.max.bioen.sp{idx}`'s new name
   `predation.ingestion.rate.max.sp{idx}` is **already a distinct live key**. Needs a merge rule, not a
   blind rename.
5. **Missed rename:** `output.fishery.* → output.fisheries.*` (3 keys in our configs/schema).

## The 4.4.0 rename set (complete)

Added as a new `_MIGRATION_CHAIN` entry `("4.4.0", {...})`. Clean **prefix** renames (the chain's
existing mechanism handles these incl. indexed `spN` automatically):

| Old prefix | New prefix |
|---|---|
| `simulation.bioen.enabled` | `module.bioenergetics.enabled` |
| `simulation.genetic.enabled` | `module.genetics.enabled` |
| `fisheries.enabled` | `module.multispecies.fisheries.enabled` |
| `economy.enabled` | `module.bioeconomics.enabled` |
| `output.restart.enabled` | `simulation.restart.enabled` |
| `output.restart.recordfrequency.ndt` | `simulation.restart.recordfrequency.ndt` |
| `output.fishery.` | `output.fisheries.` |
| `species.bioen.maturity.` | `species.maturity.` |

Two renames are **NOT clean prefix swaps** and need explicit handling (extend the chain applier to
support whole-key renames + a merge hook):
- **Leaf rename:** `output.restart.spinup` → `simulation.restart.spinup.nyear` (the leaf changes, not
  just the prefix). Apply as a whole-key rename. NOTE: the restart renames are **full-key** entries
  (`output.restart.enabled`, `output.restart.recordfrequency.ndt`) — there is **no** blanket
  `output.restart.` prefix rule, so `spinup` cannot be half-rewritten by another rule. Do NOT introduce
  a blanket `output.restart.` prefix (that WOULD then collide with `spinup`); keep the three restart
  renames as distinct full-key entries.
- **Merge/collision:** `predation.ingestion.rate.max.bioen.sp{idx}` → `predation.ingestion.rate.max.sp{idx}`
  where the target already exists as the legacy (non-bioen) ingestion rate. **Merge rule:** when the
  `.bioen.` key is present (bioenergetics mode), its value becomes the canonical
  `predation.ingestion.rate.max.sp{idx}` (overwriting the legacy) and the `.bioen.` key is dropped;
  when only the legacy key is present, keep it unchanged. Also
  `predation.coef.ingestion.rate.max.larvae.bioen.sp{idx}` → `predation.larval.ingestion.rate.increase.ratio.sp{idx}`
  (whole-key rename, no collision). **VERIFY at implementation against the 4.4.0 jar** that this is the
  correct merge direction (the engine's accepted parameter set is the arbiter).

**Watch-item:** confirm the fisheries toggle is `module.multispecies.fisheries.enabled` (one upstream
note says `process.`); the structured release-notes table says `module.` — verify against the 4.4.0 jar
before flipping the write default to 4.4.0.

## Architecture

**Canonical = 4.4.0 keys.** One forward canonicalizer (extended `migrate_config`), one inverse
(`to_target_keys`), routed through single choke points on both the load and write sides.

### Forward (ingestion → canonical 4.4.0)

- **`osmose/demo.py`:** add the `"4.4.0"` chain entry + extend the applier for whole-key renames and the
  ingestion merge hook. Bump `migrate_config`'s default `target_version` to `"4.4.0"`. Expose a thin
  `osmose.config.canonicalize_config(cfg) -> (dict, deprecated: list[str])` wrapper (logs each distinct
  deprecated old key once) so callers don't import from `demo`.
- **Load choke points (all must canonicalize):**
  - `osmose/config/reader.py`: canonicalize at the end of `read_file()` (which `read()` calls), so both
    the normal and `advanced.py` import paths are covered.
  - `EngineConfig.from_dict` + `osmose/engine/config_validation.py` + `osmose/config/validator.py`:
    canonicalize at entry (idempotent) so old-key test hand-dicts and any old config still build/validate.
  - **`ui/state.py`: add `AppState.load_config(cfg)`** that canonicalizes before `self.config.set(...)`;
    route demo load (`grid.py:841`, replacing the bare `migrate_config` call), scenario load
    (`scenarios.py:181`), and fishbase bootstrap through it. Per-field `update_config` edits already use
    canonical keys (schema moved), so they need no change.
- **Schema (`schema/{simulation,fishing,economics,output,bioenergetics}.py`):** move the renamed fields'
  `key_pattern` to the NEW names. Old names live only in the migration chain.
- **Engine literal reads:** update `osmose/engine/config.py` (the module toggles, bioen ingestion +
  maturity reads) to the NEW names — valid because the dict is canonicalized before `from_dict` reads it.

### Inverse (canonical 4.4.0 → target engine version, for writing)

- **`osmose.config.to_target_keys(cfg, target_version="4.3.3") -> dict`** (new; lives with the alias
  data, inverse of the 4.4.0 chain step): for a `4.3.x` target, reverse the 4.4.0 renames (new→old) and
  set `osmose.version=4.3.3`; for a `4.4.0` target, identity + `osmose.version=4.4.0`.
  **The inverse must reverse ONLY the exact keys the forward step created — never a blanket prefix that
  could catch a pre-existing canonical key.** Two concrete hazards:
  - `species.maturity.{eta,r,m0,m1}.sp{idx}` reverses to `species.bioen.maturity.…` **leaf-scoped (only
    those four leaves)** — a bare `species.maturity.`→`species.bioen.maturity.` prefix reverse would
    corrupt the **pre-existing growth keys** `species.maturity.size.sp{idx}` / `species.maturity.age.sp{idx}`
    (schema `species.py:195/206`, never bioen). So the maturity entry needs an explicit leaf set.
  - The ingestion **merge is intentionally lossy** — `to_target_keys` does NOT reconstruct the
    legacy/bioen split; it emits the unified value under the legacy `predation.ingestion.rate.max.sp{idx}`,
    which the 4.3.3 jar reads correctly as the legacy key.
- **Route EVERY config that reaches the Java engine through the target-version inverse (default `4.3.3`):**
  - `ui/pages/run.py::write_temp_config` (the actual Java run path) — the load-bearing one.
  - `ui/pages/calibration_handlers.py` ×3 `OsmoseConfigWriter.write(...)`.
  - `ui/pages/advanced.py:201` `OsmoseConfigWriter.write(...)`.
  - **`osmose/calibration/problem.py` `_run_java_subprocess` (≈451-454)** — the Java calibration path
    ALSO passes parameter overrides as CLI `-P<key>=<value>` args (keys come from `FreeParameter.key`
    → `field.resolve_key(i)`, i.e. canonical 4.4.0 after the schema move). These override KEYS must be
    reverse-mapped (new→old) for the 4.3.3 jar too — not just the base-config file. Apply a key-only
    `to_target_keys`-style map to the `overrides` dict before building the `-P` args. (Only the
    `use_java_engine=True` path; the default Python calibration path reads canonical keys directly.)
  - `OsmoseConfigWriter.write` gains a `target_version` param (default `"4.3.3"`) so it reverse-maps
    centrally BEFORE its prefix-based `ROUTING` runs (routing keys on the old prefixes); `write_temp_config`
    either gains the same param or calls `to_target_keys` before serializing. Default `4.3.3` ⇒ this
    feature is a **no-op on disk for the current 4.3.3 Java path**.

### key_case_map

`migrate_config`/`canonicalize_config` and `to_target_keys` must rebuild `key_case_map` in lockstep:
synthesize an entry for each renamed key and drop the orphaned old-key entry (including, for the
ingestion **merge**, dropping the `.bioen.` key's case entry while keeping the surviving
`predation.ingestion.rate.max.sp{idx}` entry). All renamed keys are lowercase, so identity casing is
correct; the spec requires the rebuild explicitly so the writer never restores stale casing onto an
unrelated key. (Today's `migrate_config` does not touch `key_case_map` at all — this is a new requirement.)

## Data flow

`old-or-new config` → (reader `read_file` | `AppState.load_config` | `from_dict`/validators)
→ `canonicalize_config` → **canonical 4.4.0 dict** in `state.config` / `EngineConfig` (+ deprecation log)
→ any write-to-disk site → `to_target_keys(cfg, target_version)` (default `4.3.3` → old keys + version,
so the bundled 4.3.3 jar runs; `4.4.0` enabled by the jar-swap follow-up) → file.

## Error handling / edge cases

- Already-canonical (4.4.0) config → forward chain is a no-op (idempotent: a key already new isn't in any
  old→new entry).
- Double canonicalize (reader then from_dict) → idempotent, safe.
- Bioen-ingestion collision → merge rule above (bioen value wins when present; legacy kept otherwise).
- `output.restart.spinup`/larvae renames → whole-key, longest-match-first, applied after prefix rules.
- Conflict (both old+new present for a non-merge key) → keep NEW, drop old + its case-map entry, warn.
- Unmapped keys → untouched.
- `osmose.version`: the chain stamps it (4.4.0 forward / 4.3.3 on inverse). `migrate_config` already owns
  version stamping — single owner preserved.

## Scope (YAGNI)

- **In:** the 4.4.0 rename set (incl. `output.fishery`, the leaf rename, the ingestion merge); extend the
  existing `migrate_config` chain (NOT a parallel system); forward canonicalize at all load choke points;
  inverse `to_target_keys` at all write sites (default 4.3.3 = no-op for current Java path); schema
  key_pattern moves; engine literal-read updates; key_case_map rebuild.
- **Out:** new 4.4.0 features; the jar swap (next step); rewriting bundled example configs on disk (left
  old, canonicalized on read); a standalone user "migrate file" command.

## Testing

- **`tests/test_config_migration_440.py` (new):** the forward `"4.4.0"` chain — each clean prefix rename
  (incl. indexed `species.maturity.*`), the `output.fishery.→output.fisheries.` prefix, the
  `spinup→spinup.nyear` leaf rename, the ingestion **merge** (bioen present → bioen value wins + legacy
  overwritten + `.bioen` dropped; legacy-only → unchanged), idempotency on already-4.4.0 input, conflict
  rule, deprecation list, and **log-once** (caplog: one message per distinct old key).
- **Inverse `to_target_keys`:** 4.3.3 target restores old keys (whole-key longest-match; spinup.nyear→spinup;
  module.*→old) + `osmose.version=4.3.3`; 4.4.0 target = identity + stamp.
- **Round-trip THROUGH the real write paths:** (a) `write_temp_config` default → on-disk config is OLD
  keys + `osmose.version=4.3.3` (current 4.3.3 Java path unaffected); (b) read old config → state.config
  is canonical 4.4.0 → `write_temp_config` → old again (fixpoint). Same assertion for an
  `OsmoseConfigWriter.write` calibration call (default target).
- **Ingestion choke points:** `reader.read_file` (advanced import path), `AppState.load_config` (demo +
  scenario + fishbase), and `EngineConfig.from_dict` each yield canonical 4.4.0 keys from an old-key input.
- **Scenario round-trip:** save (from canonical state) → load via `AppState.load_config` → canonical.
- **Validator:** bundled example configs (old keys) stay warning-free after canonicalization; new keys in
  the schema/allowlist; `test_from_dict_warn_mode_clean_on_example_configs` green. Verify each renamed
  key's allowlist source (schema vs `_SUPPLEMENTARY_ALLOWLIST`) so moving the `key_pattern` doesn't orphan it.
- **Alias↔schema drift guard:** assert every NEW key produced by the `"4.4.0"` chain entry resolves to a
  real schema field (catches a typo'd/never-added rename target).
- **Regression:** existing `test_bioen_*`, `test_genetics_*`, `test_engine_fisheries`, `test_roundtrip`,
  `test_config_writer`, `test_state`, scenario/calibration suites pass (old-key fixtures still work via
  canonicalize; update only assertions that pin a specific emitted key name). **`tests/test_demo.py` and
  `tests/test_ui_load_scenarios.py` call `migrate_config(...)` with the DEFAULT target** — bumping the
  default to `4.4.0` shifts their output to new keys; re-check those assertions explicitly.
- **CI gates:** pyright (clean `[dev]` venv `--pythonpath`) on touched modules incl. `demo.py`,
  `config/*`, `ui/state.py`, `ui/pages/{run,advanced,calibration_handlers,grid,scenarios}.py`; ruff
  check + format; full `-m "not e2e"` (modulo known `test_runner`/`test_study_fullmodel` xdist flakes).

## Follow-up (out of scope, enabled by this)

Bundle `osmose_4.4.0-jar-with-dependencies.jar`; flip the write-default `target_version` to `4.4.0`;
update the hardcoded `4.3.3` refs (`ui/state.py:42`, `tests/test_state.py:79`, `demo.py` default);
**verify** the fisheries-toggle name + the ingestion-merge direction against the jar; refresh Java
cross-engine parity (4.4.0 numerics differ from 4.3.3).
