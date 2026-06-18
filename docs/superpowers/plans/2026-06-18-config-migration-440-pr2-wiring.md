# Config Migration to 4.4.0 — PR2 (UI/Writer/Calibration Wiring) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the 4.4.0 config-key migration: move the schema to the new key names, canonicalize at every load boundary so `state.config` holds 4.4.0 keys, and reverse-map at every Java-bound write (default target 4.3.3) so the bundled 4.3.3 jar still runs.

**Architecture:** PR1 shipped the core (`osmose/config/aliases.py` `canonicalize_config`/`to_target_keys`/`RENAMES_440`, the `migrate_config` 4.4.0 chain entry, `from_dict`/validators canonicalize). PR2 flips the canonical form to NEW keys across the I/O surface: schema `key_pattern`s, the reader, `AppState`/scenario/fishbase load, and the writer/`write_temp_config`/calibration write sites — all atomic in one PR. Default write `target_version="4.3.3"` keeps on-disk output old-format (no behavior change for the bundled jar).

**Tech Stack:** Python 3.12, Shiny for Python, pytest. Run with `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-key-migration-4.4.0-design.md` (see `docs/superpowers/specs/2026-06-18-config-key-migration-4.4.0-design.md`). **PR1 plan/merged:** `docs/superpowers/plans/2026-06-18-config-migration-440-pr1-core.md` (master `001c530`).

## Coupling note (task ordering)

The schema move + reader canonicalize together flip the in-memory key form to NEW; until BOTH land the UI's schema↔`state.config` binding for the ~16 renamed keys is transiently inconsistent. No UNIT test exercises that binding (the e2e suite is excluded from default CI), so per-task commits stay green; **Task 7 verifies end-to-end coherence** (load→write round-trip + `import ui.pages.*`). Land PR2 as one PR. The renamed keys have NO hardcoded UI input-ID references (verified in PR1), so the input-ID change (`key_pattern.replace(".","_")`) is internal-only.

## Key facts (verified)

- `osmose/config/aliases.py` (PR1): `canonicalize_config(cfg)->(dict, deprecated)`, `to_target_keys(cfg, target_version="4.3.3")->dict`, `RENAMES_440` (16), `_INVERSE_440`.
- `OsmoseConfigReader.read(master)` builds the flat dict via `_read_recursive`→`read_file(filepath)` per file; `self.key_case_map` is instance state built in `read_file`. `read_file` is ALSO called directly by `ui/pages/advanced.py` import.
- `OsmoseConfigWriter.write(config, output_dir, key_case_map=None, ...)` routes by OLD prefixes (`ROUTING`); `write_temp_config(config, output_dir, source_dir=None, key_case_map=None)` (ui/pages/run.py) writes a flat master via `sorted(config.items())` with `case_map.get(key,key)`.
- `AppState` (ui/state.py): `self.config: reactive.Value[dict]` (:36), `self.key_case_map: reactive.Value[dict]` (:63), `update_config` (:75), `reset_to_defaults` (:91). NO `load_config` yet.
- `state.config.set(...)` load sites: demo `ui/pages/grid.py:841` (`migrate_config(reader.read(...))`), scenario `ui/pages/scenarios.py:181` (`state.config.set(loaded.config)`), fishbase `ui/components/fishbase_bootstrap.py:232` (`state.config.set(new_cfg)`).
- Calibration Java `-P` overrides: `osmose/calibration/problem.py` ≈453 `for key, value in overrides.items(): cmd.append(f"-P{key}={value}")`; 3 `OsmoseConfigWriter.write` calls in `ui/pages/calibration_handlers.py` (1392/1598/1745).
- Schema OLD `key_pattern`s to move: `schema/simulation.py:87` (`simulation.bioen.enabled`), `:95` (`simulation.genetic.enabled`); `schema/fishing.py:8` (`fisheries.enabled`); `schema/economics.py:7` (`economy.enabled`); `schema/output.py:52` (`output.restart.enabled`) + `output.fishery.*` in its `_OUTPUT_ENABLE_FLAGS` list; `schema/bioenergetics.py:148/156/165/174` (`species.bioen.maturity.{eta,r,m0,m1}`), `:230` (`predation.ingestion.rate.max.bioen`), `:237` (larvae).
- PR1-bridge OLD-key allowlist entries to REMOVE once the schema moves: `osmose/engine/config_validation.py:139-149` (`fisheries.enabled`, `simulation.bioen.enabled`, `simulation.genetic.enabled`, the 4 `species.bioen.maturity.*`, the 2 ingestion `.bioen` keys).

---

## Task 1: Schema `key_pattern` moves + drop the PR1-bridge allowlist

**Files:** Modify `osmose/schema/{simulation,fishing,economics,output,bioenergetics}.py`, `osmose/engine/config_validation.py`; Test: `tests/test_config_migration_440.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_schema_uses_new_440_key_patterns():
    from osmose.schema import build_registry  # exported from osmose/schema/__init__.py:29

    reg = build_registry()
    patterns = {f.key_pattern for f in reg.all_fields()}
    assert "module.bioenergetics.enabled" in patterns
    assert "module.multispecies.fisheries.enabled" in patterns
    assert "module.genetics.enabled" in patterns
    assert "module.bioeconomics.enabled" in patterns
    assert "simulation.restart.enabled" in patterns
    assert "species.maturity.eta.sp{idx}" in patterns
    assert "predation.ingestion.rate.max.bioen.sp{idx}" not in patterns  # old gone
    assert "simulation.bioen.enabled" not in patterns
```
NOTE: `build_registry`/`reg.all_fields()`/`f.key_pattern` and the `sp{idx}` placeholder are the verified real API (CLAUDE.md: "Species-indexed params use `sp{idx}`").

- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_config_migration_440.py -k schema_uses_new -q`

- [ ] **Step 3: Move each `key_pattern` to its NEW name** (per the "Key facts" list):
- `schema/simulation.py`: `simulation.bioen.enabled`→`module.bioenergetics.enabled`; `simulation.genetic.enabled`→`module.genetics.enabled`
- `schema/fishing.py`: `fisheries.enabled`→`module.multispecies.fisheries.enabled`
- `schema/economics.py`: `economy.enabled`→`module.bioeconomics.enabled`
- `schema/output.py`: `output.restart.enabled`→`simulation.restart.enabled`; and in `_OUTPUT_ENABLE_FLAGS`, rename the LIST ENTRIES `output.fishery.enabled`/`output.fishery.byage.enabled`/`output.fishery.bysize.enabled`→`output.fisheries.*` (these are string-list entries, NOT `key_pattern=`)
- `schema/bioenergetics.py`: `species.bioen.maturity.{eta,r,m0,m1}.sp{idx}`→`species.maturity.{...}.sp{idx}`; `predation.ingestion.rate.max.bioen.sp{idx}`→`predation.ingestion.rate.max.sp{idx}`; `predation.coef.ingestion.rate.max.larvae.bioen.sp{idx}`→`predation.larval.ingestion.rate.increase.ratio.sp{idx}`
Add a one-line comment in `schema/bioenergetics.py` distinguishing `species.maturity.{eta,r,m0,m1}` (bioenergetic maturation reaction norm) from the growth `species.maturity.size/age` (in `schema/species.py`).

  Then in `osmose/engine/config_validation.py` REMOVE the PR1-bridge OLD-key allowlist entries (lines 139-149: `fisheries.enabled`, `simulation.bioen.enabled`, `simulation.genetic.enabled`, the 4 `species.bioen.maturity.*`, the 2 `.bioen` ingestion keys) — the schema now defines the NEW names, and old keys never reach the unknown-key check (canonicalized first).

- [ ] **Step 4: Verify**
- `.venv/bin/python -m pytest tests/test_config_migration_440.py -k schema_uses_new -q` → pass.
- `.venv/bin/python -m pytest tests/test_engine_config_validation.py tests/test_schema_engine_key_parity.py -q` → green (schema now has new keys, engine reads new keys → parity holds WITHOUT the bridge entries; `test_from_dict_warn_mode_clean_on_example_configs` still green because the example configs canonicalize to the now-schema-known new keys).
- `.venv/bin/python -m pytest tests/test_schema.py -q` (or the schema test module) → green.

- [ ] **Step 5: Commit**
```bash
git add osmose/schema/ osmose/engine/config_validation.py tests/test_config_migration_440.py
git commit -m "feat(schema): move renamed key_patterns to 4.4.0 names; drop PR1 bridge allowlist"
```

---

## Task 2: Reader canonicalizes + rebuilds key_case_map

**Files:** Modify `osmose/config/reader.py`; Test: `tests/test_config_migration_440.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_reader_canonicalizes_and_rebuilds_case_map(tmp_path):
    from osmose.config.reader import OsmoseConfigReader

    f = tmp_path / "osm_param-simulation.csv"
    f.write_text("simulation.bioen.enabled ; true\noutput.restart.spinup ; 5\n")
    reader = OsmoseConfigReader()
    cfg = reader.read_file(f)
    assert cfg["module.bioenergetics.enabled"] == "true"   # canonicalized on read
    assert cfg["simulation.restart.spinup.nyear"] == "5"
    assert "simulation.bioen.enabled" not in cfg


def test_reader_case_map_preserves_renamed_key_source_casing(tmp_path):
    # REGRESSION: a renamed key that is camelCase in the source must survive a
    # read -> canonicalize(new) -> to_target_keys(old) -> write round-trip with its
    # ORIGINAL casing intact, not lowercased. (Java key casing gotcha — feedback_config_case.)
    from osmose.config.reader import OsmoseConfigReader
    from osmose.config.aliases import to_target_keys

    f = tmp_path / "osm_param-output.csv"
    f.write_text("output.fishery.byAge.enabled ; true\n")
    reader = OsmoseConfigReader()
    cfg = reader.read_file(f)                       # cfg holds the NEW (4.4.0) key
    old_cfg = to_target_keys(cfg, target_version="4.3.3")  # back to the OLD key for the jar
    (old_key,) = [k for k in old_cfg if k.endswith("byage.enabled")]
    # the reverse-mapped OLD key must resolve to the source camelCase via the kept case_map entry
    assert reader.key_case_map.get(old_key) == "output.fishery.byAge.enabled"
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement** in `osmose/config/reader.py` `read_file` — at the end, before `return result`, canonicalize the per-file dict and register the new-key case entries WITHOUT discarding the old ones:
```python
        from osmose.config.aliases import canonicalize_config

        canon, deprecated = canonicalize_config(result)
        if deprecated:
            # KEEP the renamed-old case_map entries: the writer reverse-maps NEW->OLD
            # before serializing, so it looks the case_map up by the OLD key and must
            # still find the source casing (e.g. output.fishery.byAge.enabled). Only ADD
            # entries for the new keys; do NOT pop the old ones.
            for new_key in canon:
                self.key_case_map.setdefault(new_key, new_key)
        result = canon
        return result
```
**Why keep the old entries:** popping them (the naive rebuild) silently lowercases any renamed key that was camelCase in the source — the reverse-mapped OLD key would miss the now-NEW-keyed case_map and fall back to the lowercase key. Shipped configs are all-lowercase (so CI/e2e wouldn't catch it), but a user config using the real Java camelCase spelling would be corrupted. The kept old entry is harmless (the writer only looks up keys present in the config it's writing). `read()` aggregates canonicalized per-file dicts; `canonicalize_config` is idempotent (verified: re-running renames nothing — NEW keys aren't in `RENAMES_440`, and the version stamp early-returns). `_read_recursive`'s `osmose.configuration.*` sub-file detection still works because those keys aren't renamed.

- [ ] **Step 4: Verify**
- `.venv/bin/python -m pytest tests/test_config_migration_440.py -k "reader_canonicalizes or case_map_preserves" -q` → pass.
- `.venv/bin/python -m pytest tests/test_config_reader.py tests/test_ui_load_scenarios.py tests/test_demo.py -q` → green. Update any assertion that checked OLD keys present after `reader.read(...)` to the NEW canonical keys — name each you change.
- **Known breaks to fix in THIS task** (the reader now canonicalizes; these assert old keys survive read→write):
  - `tests/test_roundtrip.py::test_full_roundtrip` (asserts every input key incl. `fisheries.enabled`/`economy.enabled` survives) → assert the NEW canonical keys instead (the reader always re-canonicalizes the read-back, so this assertion is stable across the rest of PR2).
  - `tests/test_roundtrip.py::test_roundtrip_preserves_boolean_values` (`result["fisheries.enabled"]=="true"`, `result["economy.enabled"]=="false"`) → switch to `result["module.multispecies.fisheries.enabled"]` / `result["module.bioeconomics.enabled"]`.
  - `tests/test_baltic_ev_fixture_bioen.py::test_baltic_ev_runs_5_years_without_genetics` (≈:41) and `::test_baltic_ev_baseline_viable_for_fie` (≈:88) — these are `@pytest.mark.integration` (run in CI) and do `cfg = reader.read(...); cfg["simulation.genetic.enabled"] = "false"`. After canonicalize the fixture's `simulation.genetic.enabled;true` becomes `module.genetics.enabled=true`; setting the OLD key is then a no-op override that `from_dict`'s skip-if-exists merge DROPS → genetics silently stays ON (a behavioral break, not an assertion-text break). Fix: set `cfg["module.genetics.enabled"] = "false"`. Grep the whole suite for the idiom `reader.read(...); cfg["<old key>"] = ...` and fix every occurrence (`test_trait_registry_validator.py:21` is the same idiom but on `data/examples`, which has no genetics key → no collision → currently safe; convert it too for robustness).

- [ ] **Step 5: Commit**
```bash
git add osmose/config/reader.py tests/test_config_migration_440.py tests/test_roundtrip.py tests/test_baltic_ev_fixture_bioen.py tests/test_trait_registry_validator.py
git commit -m "feat(config): reader canonicalizes to 4.4.0 keys + rebuilds key_case_map"
```
(Include the three test files whose assertions Step 4 updates, so no intermediate commit leaves a dirty tree.)

---

## Task 3: AppState.load_config + route demo/scenario/fishbase; ScenarioManager.load

**Files:** Modify `ui/state.py`, `ui/pages/grid.py`, `ui/pages/scenarios.py`, `ui/components/fishbase_bootstrap.py`, `osmose/scenarios.py`; Test: `tests/test_config_migration_440.py` (+ `tests/test_state.py`)

- [ ] **Step 1: Write the failing test** (append)

```python
def test_appstate_load_config_canonicalizes():
    from ui.state import AppState

    st = AppState()
    st.load_config({"osmose.version": "4.3.3", "simulation.bioen.enabled": "true"})
    assert st.config.get()["module.bioenergetics.enabled"] == "true"
    assert "simulation.bioen.enabled" not in st.config.get()
```

- [ ] **Step 2: Run, verify FAIL** (`AppState` has no `load_config`).

- [ ] **Step 3: Implement**

3a. `ui/state.py` — add `load_config` (near `reset_to_defaults`):
```python
    def load_config(self, cfg: dict[str, str], case_map: dict[str, str] | None = None) -> list[str]:
        """Canonicalize a freshly-loaded config to 4.4.0 keys and set it as the active config.

        Returns the list of deprecated (old) keys seen, for one-time UI notification.
        """
        from osmose.config.aliases import canonicalize_config

        canon, deprecated = canonicalize_config(cfg)
        self.config.set(canon)
        if case_map is not None:
            self.key_case_map.set(case_map)
        return deprecated
```

3b. Route the three load sites through it:
- `ui/pages/grid.py:841-843` — this is a THREE-line block, not a single call:
  ```python
  cfg = migrate_config(reader.read(master))            # :841
  state.key_case_map.set(dict(reader.key_case_map))    # :842
  state.config.set(cfg)                                # :843
  ```
  Collapse all three into `load_config` (which now does the canonicalize AND the case_map set), and surface the deprecation notification in the same edit (folded in from what was a separate task — avoids a second edit of this handler + a dead `deprecated` local):
  ```python
  deprecated = state.load_config(reader.read(master), reader.key_case_map)
  if deprecated:
      ui.notification_show(
          f"Config migrated to OSMOSE 4.4.0 keys ({len(deprecated)} renamed). "
          "Files written for the engine stay 4.3.x-compatible.",
          type="message",
          duration=8,
      )
  ```
  Drop the now-redundant `migrate_config` import if unused elsewhere in the file. (`grid.py:8` imports `ui`; `ui.notification_show` is already used at `grid.py:564/681/801/824/833/859`, so no new import.)
- `ui/pages/scenarios.py:181` — replace `state.config.set(loaded.config)` with `state.load_config(loaded.config)`.
- `ui/components/fishbase_bootstrap.py:232` — replace `state.config.set(new_cfg)` with `state.load_config(new_cfg)`.

3c. `osmose/scenarios.py` — canonicalize in `ScenarioManager.load` (`scenarios.py:99`, returns a `Scenario` with `.config`). `compare`/`fork` both call `self.load` (`scenarios.py:139,154`), so canonicalizing once in `load` covers them:
```python
        from osmose.config.aliases import canonicalize_config
        # after building `config` from the loaded scenario dict, before returning Scenario(...):
        config, _ = canonicalize_config(config)
```
**Scenario SAVE is intentionally left writing NEW keys.** `handle_save` (`scenarios.py:114`) dumps `dict(state.config.get())`, which now holds 4.4.0 keys → saved scenario JSON is 4.4.0-keyed. This is the correct forward direction: a newer OSMOPY persists canonical keys, and `load` (3c) canonicalizes any OLD saved scenario on read. The only break is a *newer* save read by an *older* OSMOPY — inherent, documented in the Task 6 changelog. No code change for save.

  **Other `state.config.set(...)` / config-write sites — verified coherent, no change needed** (note here so they don't derail the executor when grep surfaces them):
  - `ui/pages/setup.py:178/199`, `ui/pages/forcing.py:132/148` edit individual fields by `field.resolve_key(i)` (schema-derived, NEW after Task 1) on `dict(state.config.get())` (NEW after Task 2/3) → coherent.
  - `ui/pages/advanced.py:175` `confirm_import` merges params parsed via `read_file` (`advanced.py:93`) — Task 2 makes that path canonicalize, so the merged dict is already NEW-keyed. Confirm the importer uses `read_file` (it does) — no extra change.

Add a wiring test asserting the notification is surfaced:
```python
def test_grid_load_surfaces_deprecation_notification():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "ui" / "pages" / "grid.py").read_text()
    assert "load_config" in src and "notification_show" in src
```

- [ ] **Step 4: Verify**
- `.venv/bin/python -m pytest tests/test_config_migration_440.py -k "appstate_load or deprecation_notification" -q` → pass.
- `.venv/bin/python -m pytest tests/test_state.py tests/test_ui_load_scenarios.py tests/test_scenarios.py -q` → green. `test_ui_load_scenarios.py` asserts only `simulation.nspecies`/`species.name`/dir/dirty (none renamed) and `test_demo.py` tests pre-4.4.0 migrations (`nplankton→nresource`) — both verified SAFE in review, so expect NO edits there; update only an assertion that genuinely checks a renamed old key (name each).
- `.venv/bin/python -c "import ui.pages.grid, ui.pages.scenarios, ui.pages.advanced, ui.components.fishbase_bootstrap"` → clean import.

- [ ] **Step 5: Commit**
```bash
git add ui/state.py ui/pages/grid.py ui/pages/scenarios.py ui/components/fishbase_bootstrap.py osmose/scenarios.py tests/test_config_migration_440.py
git commit -m "feat(ui): AppState.load_config canonicalizes + deprecation notice; route demo/scenario/fishbase + ScenarioManager.load"
```

---

## Task 4: Writer reverse-maps for the target engine version

**Files:** Modify `osmose/config/aliases.py`, `osmose/config/writer.py`, `ui/pages/run.py`; Test: `tests/test_config_migration_440.py`

- [ ] **Step 0: Harden `to_target_keys` against a mixed old+new dict** (defensive — PR2 makes this function the single reverse-map for every engine-bound write). Failing test (append):
```python
def test_to_target_keys_collapses_mixed_old_and_new(tmp_path):
    from osmose.config.aliases import to_target_keys

    # A dict carrying BOTH forms of one flag must collapse to the OLD key only — never
    # emit two lines for the same logical key (the 4.3.3 jar would read the OLD and
    # silently ignore the orphan NEW line, risking a stale value).
    mixed = {"module.bioenergetics.enabled": "true", "simulation.bioen.enabled": "false"}
    out = to_target_keys(mixed, target_version="4.3.3")
    assert "module.bioenergetics.enabled" not in out   # redundant NEW form dropped
    assert out["simulation.bioen.enabled"] == "false"  # existing OLD value wins (base-wins)
```
Run → FAIL (current code leaves the NEW key in place). Then in `osmose/config/aliases.py::to_target_keys` (lines 108-111), always pop the new-prefixed key, assigning the reversed key only if absent (so an existing OLD value wins, matching the canonicalize base-wins rule):
```python
        for key in [k for k in result if k == new_prefix or k.startswith(new_prefix + ".")]:
            reversed_key = old_prefix + key[len(new_prefix) :]
            value = result.pop(key)              # always drop the NEW-named key
            if reversed_key not in result:        # keep an existing OLD value (base wins)
                result[reversed_key] = value
```
Run → PASS. (Normal PR2 flow canonicalizes at every load boundary so mixed dicts shouldn't reach here, but `to_target_keys` must not silently double-key a file for any future/partial caller. The drift-guard `_INVERSE_440`↔`RENAMES_440` test from PR1 still holds.)

- [ ] **Step 1: Write the failing test** (append)

```python
def test_writer_default_target_emits_old_keys(tmp_path):
    from osmose.config.writer import OsmoseConfigWriter
    from osmose.config.reader import OsmoseConfigReader

    cfg = {"osmose.version": "4.4.0", "module.bioenergetics.enabled": "true",
           "simulation.restart.spinup.nyear": "5"}
    OsmoseConfigWriter().write(cfg, tmp_path)  # default target_version="4.3.3"
    flat = OsmoseConfigReader().read(tmp_path / "osm_all-parameters.csv")
    # on disk (pre-reader-canonicalize) it must be the OLD keys for the 4.3.3 jar; but the
    # reader re-canonicalizes, so assert by reading the raw master text:
    raw = (tmp_path / "osm_all-parameters.csv").read_text()
    assert "simulation.bioen.enabled" in raw          # reverse-mapped to old
    assert "output.restart.spinup" in raw
    assert "module.bioenergetics.enabled" not in raw


def test_write_temp_config_default_target_emits_old_keys(tmp_path):
    from ui.pages.run import write_temp_config

    cfg = {"module.bioenergetics.enabled": "true", "fisheries.enabled": "false"}
    # fisheries.enabled is already old here; module.* is new -> reverse to old:
    master = write_temp_config({"module.multispecies.fisheries.enabled": "false",
                                "module.bioenergetics.enabled": "true"}, tmp_path)
    raw = master.read_text()
    assert "fisheries.enabled" in raw
    assert "module.multispecies.fisheries.enabled" not in raw
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**

3a. `osmose/config/writer.py` — add `target_version: str = "4.3.3"` to `write(...)`; at the TOP of `write`, before routing, reverse-map:
```python
        from osmose.config.aliases import to_target_keys

        config = to_target_keys(config, target_version=target_version)
```
(So `ROUTING` — which keys on OLD prefixes — operates on the reverse-mapped old keys. No ROUTING change needed for the 4.3.3 default; the `module.*`/`output.fisheries.*` ROUTING entries are deferred to the jar-swap step that flips the default to 4.4.0.)

3b. `ui/pages/run.py::write_temp_config` — add `target_version: str = "4.3.3"`; reverse-map before serializing:
```python
    from osmose.config.aliases import to_target_keys

    config = to_target_keys(config, target_version=target_version)
```
(Place it after `_inject_random_movement_ncell`/before the `sorted(config.items())` loop. `case_map.get(key, key)` then looks the reverse-mapped OLD key up in the case_map — which Task 2 deliberately keeps the OLD entries in, so a renamed key that was camelCase in the source recovers its source casing. For a 4.4.0-native config with no source OLD entry, the fallback is the lowercase canonical OLD key, which is the correct on-disk spelling.)

- [ ] **Step 4: Verify**
- `.venv/bin/python -m pytest tests/test_config_migration_440.py -k "writer_default or write_temp" -q` → pass.
- `.venv/bin/python -m pytest tests/test_config_writer.py -q` → green. `test_config_writer.py` reads raw sub-file text and the writer reverse-maps OLD→OLD as identity at target 4.3.3, so its routing assertions are unaffected — confirm, don't pre-edit. (`test_roundtrip.py` was already updated in Task 2; the read-back is always canonicalized so its NEW-key assertions hold here too.)

- [ ] **Step 5: `export_config` download path** (spec-named write site, `ui/pages/advanced.py:200`): it already calls `OsmoseConfigWriter().write(state.config.get(), ...)`, so it inherits the `target_version="4.3.3"` default automatically — the exported `osm_all-parameters.csv` is emitted in 4.3.x format (consistent with every other engine-bound write; intentional, so the export runs on the bundled jar). Add a guard test:
```python
def test_export_writes_target_format(tmp_path):
    from osmose.config.writer import OsmoseConfigWriter

    OsmoseConfigWriter().write({"module.bioenergetics.enabled": "true"}, tmp_path)
    raw = (tmp_path / "osm_all-parameters.csv").read_text()
    assert "simulation.bioen.enabled" in raw  # export inherits the 4.3.3 reverse-map
```

- [ ] **Step 6: Commit**
```bash
git add osmose/config/aliases.py osmose/config/writer.py ui/pages/run.py tests/test_config_migration_440.py
git commit -m "feat(config): writer + write_temp_config reverse-map to target version (default 4.3.3); harden to_target_keys"
```

---

## Task 5: Calibration `-P` override reverse-map + checkpoint-key migration

**Files:** Modify `osmose/calibration/problem.py`, `osmose/calibration/checkpoint.py`; Test: `tests/test_config_migration_440.py`, `tests/test_calibration_checkpoint.py`

- [ ] **Step 1: Write the failing test** — use the existing `@patch("subprocess.run")` + `_run_single` + `call_args` seam (mirror `tests/test_calibration_problem.py:36-75`'s `_make_problem` helper + `FreeParameter`), NOT a raise-based monkeypatch:
```python
def test_calibration_java_cmd_reverse_maps_override_keys(tmp_path):
    from unittest.mock import MagicMock, patch
    from osmose.calibration.problem import FreeParameter
    # _make_problem is the existing helper in tests/test_calibration_problem.py — reuse it
    # (or construct OsmoseCalibrationProblem(use_java_engine=True, ...) the same way).
    problem = _make_problem(
        tmp_path, free_params=[FreeParameter("species.maturity.eta.sp0", 0.1, 0.5)]
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)  # short-circuit after cmd build
        try:
            problem._run_single({"species.maturity.eta.sp0": 0.3}, run_id=0)
        except Exception:
            pass  # results parsing fails on the mocked run; we only need the cmd
    cmd = mock_run.call_args[0][0]
    p_args = [s for s in cmd if s.startswith("-P")]
    assert any(s.startswith("-Pspecies.bioen.maturity.eta.sp0=") for s in p_args)  # reverse-mapped
    assert not any("species.maturity.eta.sp0" in s for s in p_args)                # NEW key gone
    assert not any(s.startswith("-Posmose.version=") for s in p_args)              # stamp skipped
```
NOTE: copy `_make_problem`/`FreeParameter` usage verbatim from `tests/test_calibration_problem.py:36-75`; confirm `_run_single` is the entry that reaches `_run_java_subprocess`. The test MUST fail before Step 3 (current code emits `-Pspecies.maturity.eta.sp0=...`, the NEW key, which the 4.3.3 jar rejects).

- [ ] **Step 2: Run, verify FAIL** (current `-P` loop emits the NEW key verbatim).

- [ ] **Step 3a: Reverse-map the `-P` keys** in `osmose/calibration/problem.py` `_run_java_subprocess` (line 434) — before the `-P` loop (line 453):
```python
        from osmose.config.aliases import to_target_keys

        java_overrides = to_target_keys(dict(overrides), target_version="4.3.3")
        for key, value in java_overrides.items():
            if key == "osmose.version":
                continue
            cmd.append(f"-P{key}={value}")
```
(`to_target_keys` adds an `osmose.version` stamp — skip it in the `-P` loop since it's a config key, not a calibration override. `_validate_overrides` (problem.py:362) runs on the NEW-keyed overrides BEFORE this reverse-map, which is correct since Task 1 gives the registry NEW keys. The base config written for the Java run already reverse-maps via **Task 4's** writer — `calibration_handlers.py:1392/1598/1745` call `OsmoseConfigWriter().write(...)` and inherit its `target_version="4.3.3"` default with no edit needed there. The Python-engine path needs NO reverse-map: it reads the canonicalized config + applies NEW-keyed overrides, and `from_dict` already reads NEW keys (PR1).)

- [ ] **Step 3b: Migrate checkpoint keys on read (MANDATORY).** Verified: `CalibrationCheckpoint` (`checkpoint.py:75-77`) stores `best_parameters: dict[str, float]`, `param_keys: tuple[str, ...]`, and `bounds_log10: dict[str, tuple]` — all keyed by config-parameter keys. A checkpoint written by an older OSMOPY carries OLD keys (e.g. `species.bioen.maturity.eta.sp0`) that fail the length-coupling invariants (Inv 5/6/7) against the NEW-keyed registry on resume. In `read_checkpoint`, AFTER `data = json.loads(text)` (`checkpoint.py:~316`) and BEFORE constructing `CalibrationCheckpoint`, forward-map the three key collections. **Map keys DIRECTLY via the `RENAMES_440` prefix match — do NOT use `canonicalize_config` here:** `canonicalize_config` stamps an `osmose.version` key into its output, which would leak a spurious entry into `param_keys` and then trip Inv 5/6/7. Use a tiny pure helper instead:
```python
        from osmose.config.aliases import RENAMES_440

        def _migrate_param_key(k: str) -> str:
            for old, new in RENAMES_440.items():
                if k == old or k.startswith(old + "."):
                    return new + k[len(old):]
            return k

        data["param_keys"] = [_migrate_param_key(k) for k in data["param_keys"]]
        data["best_parameters"] = {_migrate_param_key(k): v for k, v in data["best_parameters"].items()}
        data["bounds_log10"] = {_migrate_param_key(k): v for k, v in data["bounds_log10"].items()}
```
Failing test in `tests/test_calibration_checkpoint.py` (write a real file via `write_checkpoint`, then read it back and unwrap `CheckpointReadResult.checkpoint`):
```python
def test_read_checkpoint_migrates_old_param_keys(tmp_path):
    from osmose.calibration.checkpoint import write_checkpoint, read_checkpoint, CalibrationCheckpoint
    # Mirror the minimal-checkpoint construction in this file's existing fixtures
    # (see the write_checkpoint(...) usage near the top of tests/test_calibration_checkpoint.py):
    # build a CalibrationCheckpoint with param_keys=("species.bioen.maturity.eta.sp0",) and
    # matching 1-element best_x_log10 / best_parameters / bounds_log10, proxy_source="objective_disabled",
    # band/residual fields None — all 14 invariants pass for a single param.
    cp_path = tmp_path / "phase12_checkpoint.json"
    write_checkpoint(cp_path, _old_keyed_checkpoint())   # NOTE: signature is (path, ckpt) — path FIRST
    result = read_checkpoint(cp_path)
    cp = result.checkpoint                                # read_checkpoint returns CheckpointReadResult
    assert cp is not None and result.kind == "ok"
    assert "species.maturity.eta.sp0" in cp.param_keys
    assert "species.bioen.maturity.eta.sp0" not in cp.param_keys
    assert "osmose.version" not in cp.param_keys          # guard against the shim-leak regression
```
(The "checkpoints store no config keys → skip" escape hatch is removed: they verifiably DO store config keys, so this step is required.)

- [ ] **Step 4: Verify**
- `.venv/bin/python -m pytest tests/test_config_migration_440.py -k calibration_java_cmd -q` → pass.
- `.venv/bin/python -m pytest tests/test_calibration_problem.py tests/test_calibration_problem_python_engine.py tests/test_calibration_checkpoint.py -q` → green.

- [ ] **Step 5: Commit**
```bash
git add osmose/calibration/problem.py osmose/calibration/checkpoint.py tests/test_config_migration_440.py tests/test_calibration_checkpoint.py
git commit -m "feat(calibration): reverse-map Java -P override keys + migrate checkpoint keys to 4.4.0"
```

---

## Task 6: Changelog (bioen ingestion change + scenario key-format)

> (The one-time deprecation-notification wiring was folded into Task 3 — same grid.py demo-load handler — to avoid a second edit of that region and a transient unused-variable.)

**Files:** Modify `CHANGELOG.md`

- [ ] **Step 1: Add an `[Unreleased]` entry** under the appropriate sections of `CHANGELOG.md` (match the existing house format — Added/Changed):

```markdown
### Changed

- **config (OSMOSE 4.4.0 keys):** OSMOPY now uses OSMOSE 4.4.0 config-key names internally
  (e.g. `module.bioenergetics.enabled`, `simulation.restart.*`, `species.maturity.*`,
  `output.fisheries.*`), reading 4.3.x configs transparently. Engine-bound writes — the run config
  AND the **Export config** download — are emitted in the bundled engine's 4.3.x format, so they run
  on the shipped jar. **Saved scenarios** (JSON) are stored with 4.4.0 keys; a scenario saved by this
  version may need manual key edits to load in an older OSMOPY.
- **bioenergetics (ingestion):** following OSMOSE 4.4.0, the bioenergetic and base maximum-ingestion-rate
  parameters are unified into a single `predation.ingestion.rate.max.spN`. A bioenergetics/Ev-OSMOSE
  config that previously set both now uses the base value for both predation and the energy budget,
  which changes bioenergetic results relative to 4.3.x. Re-check calibrated `baltic_ev`/Ev-OSMOSE setups.
```

- [ ] **Step 2: Verify** `.venv/bin/python -m pytest tests/test_docs_content.py -q` → green (the in-app docs loader parses `CHANGELOG.md`, so this catches a format break).

- [ ] **Step 3: Commit**
```bash
git add CHANGELOG.md
git commit -m "docs(changelog): OSMOSE 4.4.0 key migration + bioen ingestion-unification note"
```

---

## Task 7: Coherence verification + final gates

**Files:** Test: `tests/test_config_migration_440.py`

- [ ] **Step 1: End-to-end coherence test** (append) — load an old-key config → state.config is new → write for the 4.3.3 jar → old on disk again (the round-trip the coupled PR2 changes must satisfy):

```python
def test_pr2_load_write_roundtrip_coherent(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import write_temp_config

    src = tmp_path / "src"
    src.mkdir()
    (src / "osm_all-parameters.csv").write_text(
        "osmose.version ; 4.3.3\nsimulation.bioen.enabled ; true\nfisheries.enabled ; false\n"
    )
    cfg = OsmoseConfigReader().read(src / "osm_all-parameters.csv")
    assert cfg["module.bioenergetics.enabled"] == "true"   # reader canonicalized
    out = tmp_path / "out"
    master = write_temp_config(cfg, out)                    # default target 4.3.3
    raw = master.read_text()
    assert "simulation.bioen.enabled" in raw                # reverse-mapped to old for the jar
    assert "module.bioenergetics.enabled" not in raw
```

- [ ] **Step 2: Run it + import all touched UI pages** (coherence):
- `.venv/bin/python -m pytest tests/test_config_migration_440.py -q` → all pass.
- `.venv/bin/python -c "import ui.pages.grid, ui.pages.scenarios, ui.pages.run, ui.pages.advanced, ui.components.fishbase_bootstrap, ui.pages.calibration_handlers"` → clean.

- [ ] **Step 3: Lint + format + pyright** (CI parity — CI lint runs BOTH `ruff check` AND `ruff format --check` on `osmose/ ui/ tests/`)
- `.venv/bin/ruff check osmose/ ui/ tests/` and `.venv/bin/ruff format --check osmose/ ui/ tests/` → clean (run `.venv/bin/ruff format osmose/ ui/ tests/` to fix). NOTE: include `tests/` in BOTH commands — PR2 edits `tests/test_config_migration_440.py`, and omitting `tests/` from format-check is a known CI-red trap.
- `.venv/bin/pyright --pythonpath .venv/bin/python osmose/config/reader.py osmose/config/writer.py osmose/scenarios.py ui/state.py ui/pages/run.py osmose/calibration/problem.py osmose/schema/simulation.py osmose/schema/fishing.py osmose/schema/economics.py osmose/schema/output.py osmose/schema/bioenergetics.py` → 0 NEW errors.

- [ ] **Step 4: Full suite** (match CI: `integration`-marked tests RUN, only `e2e`/`visual` are excluded, and CI enforces `--cov-fail-under=90`)
- `.venv/bin/python -m pytest -q -m "not e2e and not visual"` → report counts. The `integration`-marked `test_baltic_ev_fixture_bioen.py` tests run here — they were fixed in Task 2 (set `module.genetics.enabled`); if either still fails, the genetics-disable override is being silently dropped (re-check the NEW-key fix). Spot-check `tests/test_calibration_checkpoint.py` too (Task 5 added checkpoint-key migration). The only acceptable failures are the known `test_runner`/`test_study_fullmodel` xdist flakes (confirm `tests/test_runner.py tests/test_study_fullmodel.py -p no:cacheprovider` pass in isolation). Update any remaining old-key assertion to the new canonical key (name each; non-bioen value results must be byte-unchanged).
- **Optional but recommended:** run the e2e baltic suite once (`.venv/bin/python -m pytest tests/test_e2e_baltic.py -m e2e -o addopts="" -p no:cacheprovider`) since PR2 changes the UI load/write path — confirm the app loads Baltic + runs (chromium required).

- [ ] **Step 5: Commit (gate fixes, if any)**
```bash
git add -A
git commit -m "chore(config): PR2 final gate fixes"
```

---

## Notes

- **Default `target_version="4.3.3"` everywhere** = on-disk output stays old-format; the bundled 4.3.3 jar runs unchanged. The internal model is fully 4.4.0.
- **DRY:** all canonicalize/reverse goes through PR1's `aliases.py` (`canonicalize_config`/`to_target_keys`); no new rename logic.
- **The jar swap is the NEXT effort (out of PR2):** bundle `osmose_4.4.0-jar-with-dependencies.jar`, flip the write-default `target_version` to `4.4.0`, ADD `module.*`/`output.fisheries.*` to `OsmoseConfigWriter.ROUTING`, update the hardcoded `4.3.3` refs (`ui/state.py:42`, `tests/test_state.py:79`, `demo.py` default), refresh Java cross-engine parity. Fold the cosmetic stale docstring (`osmose/engine/processes/foraging_mortality.py:11`) in there too.
- **Coupling:** Tasks 1+2 (schema + reader) flip the canonical form; the UI binding is only fully coherent after both — Task 7's round-trip + import checks are the gate. The renamed keys have no hardcoded UI input-IDs (PR1-verified), so the auto-generated ID change is internal.
- **Task order is safe without reordering:** the file round-trip suites (`test_roundtrip.py`) assert on the read-back, which the reader ALWAYS re-canonicalizes — so updating those assertions to NEW keys in Task 2 makes them stable whether or not the writer (Task 4) has yet started reverse-mapping. Only Task 4's *raw on-disk text* assertions depend on the writer, and they live in Task 4. So no intermediate commit is CI-red on the round-trip path.
- **Case preservation:** Task 2 keeps (does not pop) the renamed-old `key_case_map` entries precisely so the Task 4 writer's reverse-mapped OLD key recovers its source casing — closing the camelCase-key corruption the naive rebuild would introduce (invisible to shipped all-lowercase configs, but real for user configs using Java's camelCase spelling).
