# Config-key migration to OSMOSE 4.4.0 — Design

**Date:** 2026-06-18
**Status:** Approved (brainstorming), pending implementation plan

## Goal

Make OSMOSE 4.4.0 config-key names the **canonical** internal form throughout OSMOPY, while still
reading the OLD (4.3.x) keys via aliases. This lets OSMOPY emit 4.4.0-format configs natively — the
prerequisite for bundling the 4.4.0 Java engine (a separate, sequenced follow-up), since 4.4.0 refuses
to run 4.3.x configs and its own `-update` migration crashes on our configs
(`IndexOutOfBoundsException` in `Release.commentParameter`; see [[reference-osmose-java-4-4-0]]).

**This feature is the rename layer only** — not the new 4.4.0 biological features, not the jar swap.

## Background — current state

- Config flow: file → `OsmoseConfigReader.read()` (lowercases keys → `dict[str,str]`, tracks
  `key_case_map` for round-trip casing) → `state.config` / `EngineConfig.from_dict` / schema-driven UI
  → `OsmoseConfigWriter.write(config, dir, key_case_map)` → file.
- The 4.4.0 renames are mandatory (4.4.0 hard-stops on a 4.3.3-versioned config). The renamed keys are
  read by **literal string** in: `osmose/engine/config.py` (`fisheries.enabled` ~1527,
  `simulation.bioen.enabled` ~1859, `simulation.genetic.enabled` ~1916, the bioen ingestion/maturity
  keys), `osmose/engine/processes/foraging_mortality.py` (ingestion-rate), `osmose/engine/config_validation.py`
  (`economy.enabled`, `output.restart.*`), and the schema (`schema/{simulation,fishing,economics,output,bioenergetics}.py`).
- Many test files build configs with OLD keys directly (`EngineConfig.from_dict({...old...})`,
  bypassing the reader): `test_bioen_*`, `test_genetics_*`, `test_engine_fisheries`, `test_roundtrip`,
  `test_config_writer`, etc.
- UI input IDs derive from `key_pattern.replace(".", "_")`. The renamed keys have **no hardcoded UI
  input-ID references** (verified) — they are schema-auto-rendered, so changing `key_pattern` changes
  the generated ID harmlessly.

## The rename map (the complete 4.4.0 table)

`KEY_ALIASES` (old → new). Indexed entries carry the `sp{idx}` (or `rsc{idx}`) suffix verbatim:

| Old (4.3.x) | New (4.4.0) |
|---|---|
| `simulation.bioen.enabled` | `module.bioenergetics.enabled` |
| `simulation.genetic.enabled` | `module.genetics.enabled` |
| `fisheries.enabled` | `module.multispecies.fisheries.enabled` |
| `economy.enabled` | `module.bioeconomics.enabled` |
| `output.restart.enabled` | `simulation.restart.enabled` |
| `output.restart.recordfrequency.ndt` | `simulation.restart.recordfrequency.ndt` |
| `output.restart.spinup` | `simulation.restart.spinup.nyear` |
| `predation.ingestion.rate.max.bioen.sp{idx}` | `predation.ingestion.rate.max.sp{idx}` |
| `predation.coef.ingestion.rate.max.larvae.bioen.sp{idx}` | `predation.larval.ingestion.rate.increase.ratio.sp{idx}` |
| `species.bioen.maturity.eta.sp{idx}` | `species.maturity.eta.sp{idx}` |
| `species.bioen.maturity.r.sp{idx}` | `species.maturity.r.sp{idx}` |
| `species.bioen.maturity.m0.sp{idx}` | `species.maturity.m0.sp{idx}` |
| `species.bioen.maturity.m1.sp{idx}` | `species.maturity.m1.sp{idx}` |

(Keys are matched case-insensitively — the reader already lowercases. The map stores lowercase forms.)

## Architecture

**Approach:** a central alias map + **canonicalize-on-ingestion** (not translate-only-at-write).
Canonical = NEW keys, so every config-ingestion boundary canonicalizes old→new; old-key inputs
(example configs AND old-key test fixtures) keep working untouched.

### Components

1. **`osmose/config/aliases.py` (new) — single source of truth.**
   - `KEY_ALIASES: dict[str, str]` — the table above, lowercase, with `sp{idx}` placeholders.
   - `canonicalize_keys(cfg: dict[str, str]) -> tuple[dict[str, str], list[str]]` — returns a new dict
     with every old key renamed to its new form (indexed keys: match the alias pattern with the index
     wildcard, substitute the prefix, preserve the index), plus the sorted list of deprecated old keys
     seen (for logging/UX). **Idempotent** on already-new keys. If both an old key and its new
     equivalent are present, keep the NEW value and drop the old with a recorded warning. Unmapped keys
     pass through unchanged.
   - Implementation note: build a regex-or-prefix matcher from `KEY_ALIASES` (mirroring how
     `schema/registry.py` turns `{idx}` into `\d+`) so indexed renames are a prefix substitution.

2. **Reader (`osmose/config/reader.py`):** after parsing, call `canonicalize_keys`; `_log` a one-time
   deprecation INFO per distinct old key seen ("config key X is deprecated; renamed to Y for OSMOSE
   4.4.0"); set `key_case_map` for the new canonical keys.

3. **`EngineConfig.from_dict` + `osmose/engine/config_validation.py` + `osmose/config/validator.py`:**
   canonicalize the incoming dict at the top of each entry point. This keeps the many old-key test
   hand-dicts and any old config valid without per-test edits.

4. **Schema (`schema/{simulation,fishing,economics,output,bioenergetics}.py`):** move the renamed
   fields' `key_pattern` to the NEW names. Old names live ONLY in `KEY_ALIASES`. (UI auto-renders the
   new IDs; no hardcoded ID refs exist for these.)

5. **Engine literal reads:** update `osmose/engine/config.py` (`fisheries.enabled`→
   `module.multispecies.fisheries.enabled`, `simulation.bioen.enabled`→`module.bioenergetics.enabled`,
   `simulation.genetic.enabled`→`module.genetics.enabled`, the bioen ingestion/maturity keys) and
   `osmose/engine/processes/foraging_mortality.py` (ingestion-rate) to read the NEW keys — valid because
   `from_dict` canonicalizes first, so the dict holds new keys.

6. **Writer (`osmose/config/writer.py`) — emits for the TARGET engine version (critical for safe
   sequencing).** The internal dict is canonical (new keys), but the **bundled Java engine is still
   4.3.3 until the later jar-swap step** — a 4.3.3 jar would REJECT a config with `module.*` keys /
   `osmose.version=4.4.0`. So the writer takes a `target_version` (default **`"4.3.3"`**, matching the
   bundled jar) and:
   - for a `4.3.x` target → **reverse-canonicalize** (new→old via the same `KEY_ALIASES`, applied
     backwards) so the emitted config matches what the 4.3.3 engine expects (unchanged on-disk output —
     this feature is a no-op for the current Java path);
   - for a `4.4.0` target → emit new keys natively + stamp `osmose.version=4.4.0`.
   The `KEY_ALIASES` map is therefore **bidirectional**; add `to_target_keys(cfg, version)` alongside
   `canonicalize_keys` in `aliases.py`. The jar-swap follow-up flips the default `target_version` to
   `4.4.0`; until then the existing 4.3.3 Java path is unaffected.

### Data flow

`old-or-new config file` → reader → `canonicalize_keys` → **new-key dict** (+ deprecation log)
→ `EngineConfig.from_dict` (idempotent canonicalize guard) reads new literals / schema UI uses new
`key_pattern`s → writer emits keys for `target_version` (default `4.3.3`→old keys, so the current Java
path is unchanged; `4.4.0`→new keys + version, enabled by the jar-swap follow-up).

## Error handling / edge cases

- Already-new config → `canonicalize_keys` is a no-op (idempotent).
- Both old and new present for one logical key → keep NEW, drop old, record a warning (mirrors the Java
  `-update` "already defined" message).
- Indexed keys → rename the prefix, preserve the `spN` index exactly.
- Unmapped keys → untouched.
- `config_validation`'s example-config warning-free test must stay clean: the NEW keys must be in the
  schema/allowlist (they will be, via the moved `key_pattern`s); old keys arriving in a raw dict are
  canonicalized before validation so they never reach the unknown-key check.

## Scope (YAGNI)

- **In:** the 4.4.0 rename table only; canonicalize at ingestion; schema key_pattern moves; engine
  literal-read updates; writer version stamp.
- **Out:** new 4.4.0 features (stochastic maturity, post-repro/density-dependent mortality, gradient
  movements, simplified bioen — separate parity work); the 4.4.0 jar swap (next sequenced step);
  rewriting the bundled example configs on disk (left old, read-aliased); a standalone user-facing
  "migrate file" command (the writer already produces new-key configs).

## Testing

- **`tests/test_config_aliases.py` (new):** `canonicalize_keys` per rename (incl. each indexed
  `species.bioen.maturity.*`→`species.maturity.*` and the two ingestion renames); idempotency on
  already-new input; old+new conflict (keep new + report); unmapped keys untouched; the returned
  deprecated-key list.
- **Reader:** a fixture config with old keys → `read()` yields new keys and logs the deprecation.
- **`from_dict`:** an old-key hand-dict still builds an `EngineConfig` (canonicalize guard); a new-key
  dict builds identically.
- **Writer target version:** `to_target_keys` reverse-maps new→old for a `4.3.x` target and emits
  new + `osmose.version=4.4.0` for a `4.4.0` target; round-trip with DEFAULT target (`4.3.3`): read an
  old-key config → canonical(new) → write → output is OLD keys again (current Java path unchanged), and
  re-reading yields the identical canonical dict. Round-trip with `target_version="4.4.0"`: output uses
  new keys + the version stamp.
- **Validator:** the bundled example configs (old keys) stay warning-free after canonicalization; new
  keys are recognized; `test_from_dict_warn_mode_clean_on_example_configs` stays green.
- **Regression:** the existing `test_bioen_*`, `test_genetics_*`, `test_engine_fisheries`,
  `test_roundtrip`, `test_config_writer` suites must pass unchanged — proving old-key fixtures still work
  via canonicalize. Any that assert on a specific emitted key name update to the new name.
- Full `-m "not e2e"` suite green (modulo the known `test_runner`/`test_study_fullmodel` xdist flakes).

## Follow-up (out of this spec, enabled by it)

Once OSMOPY emits 4.4.0-format configs: bundle `osmose_4.4.0-jar-with-dependencies.jar`, update the
hardcoded `4.3.3` refs (`ui/state.py:42`, `tests/test_state.py:79`, `demo.py:228`), wire the Java-engine
run path to the now-4.4.0-native written config, and refresh Java cross-engine parity expectations
(4.4.0 adds engine processes → numerics shift from 4.3.3).
