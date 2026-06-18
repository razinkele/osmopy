# Config Migration to 4.4.0 — PR1 (Core Migration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port OSMOSE's `Releases.java $15` (v4.4.0) config-key migration to Python as the canonical key form — the engine-and-machinery core — so OSMOPY can read/normalize 4.4.0 keys, leaving schema/UI/write wiring to PR2.

**Architecture:** Extend the existing `osmose/demo.py` `_MIGRATION_CHAIN` with a `"4.4.0"` entry and add `updateKey` skip-if-target-exists semantics to its applier (Java's "already defined" behavior). Expose `canonicalize_config` (forward → 4.4.0) and `to_target_keys` (inverse → older engines) in `osmose/config/`. Canonicalize at `EngineConfig.from_dict` + validators so old-key inputs (incl. all existing test fixtures) keep working, and reconcile the engine's literal key reads (notably the unified ingestion rate).

**Tech Stack:** Python 3.12, pytest. Run with `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-18-config-key-migration-4.4.0-design.md`

## Boundary note (refinement of the spec's PR1/PR2 split)

The spec listed "schema `key_pattern` moves" under PR1, but moving them WITHOUT the reader-canonicalize hook + write reverse-maps (both PR2) would break the UI's schema↔`state.config` binding for the renamed keys. **Schema moves, the reader hook, `OsmoseConfigWriter`/`write_temp_config` `target_version`, calibration `-P` mapping, scenario/fishbase load, and the deprecation UI are therefore ALL PR2** (they land atomically). PR1 is purely `osmose/` engine+machinery: it changes NO on-disk output and NO UI, so it is parity-safe in isolation. PR1 DOES intentionally change bioenergetics simulation results (the ingestion unification) — gated by a bioen test (Task 7).

## Key facts (read before starting)

- `osmose/demo.py`: `_MIGRATION_CHAIN: list[tuple[str, dict[str,str]]]` (each `(version, {old_prefix: new_prefix})`), and `migrate_config(config, target_version="4.3.3")`. The applier matches `k == old_prefix or k.startswith(old_prefix + ".")`, renames `result[new_key] = result.pop(key)` (BLIND overwrite — must become skip-if-exists), then stamps `osmose.version`. Version gate: skip a step if `current and current_tuple >= step_tuple`; break if `step_tuple > target_tuple`. `_version_tuple(v)` returns `(0,)` on a non-numeric string (hardening target).
- Reader lowercases every key (`osmose/config/reader.py`), so all map keys are lowercase.
- Engine literal reads in `osmose/engine/config.py`: `focal_ingestion_rate = _species_float(cfg, "predation.ingestion.rate.max.sp{i}", n_sp)` (≈606, ALREADY the unified key); `bioen_i_max = _species_float_optional(cfg, "predation.ingestion.rate.max.bioen.sp{i}", n_sp, 0.0)` (≈1889); `bioen_theta = _species_float_optional(cfg, "predation.coef.ingestion.rate.max.larvae.bioen.sp{i}", n_sp, 1.0)` (≈1892); `foraging_I_max = _species_float_optional(cfg, "predation.ingestion.rate.max.bioen.sp{i}", n_sp, 0.0)` (≈1911). Module toggles read at `fisheries.enabled` (≈1527), `simulation.bioen.enabled` (≈1859), `simulation.genetic.enabled` (≈1916); maturity reads `species.bioen.maturity.{eta,r,m0,m1}` in the bioen block.
- `EngineConfig.from_dict(cfg)` is the engine ingestion boundary (≈1513). Tests build old-key hand-dicts and call it directly.
- **BLOCKING GATE (Task 1):** the authoritative rename list is `Releases.java` `$15` — this plan's map was read from it (≈ lines 638-720) but may omit entries below what was inspected; Task 1 re-pulls it to lock the complete set.

## File structure

- **Create** `osmose/config/aliases.py` — the 4.4.0 rename data (`RENAMES_440`), `canonicalize_config`, `to_target_keys`. (Pure data + functions; importable without `demo`/UI.)
- **Modify** `osmose/demo.py` — add the `"4.4.0"` `_MIGRATION_CHAIN` entry; add skip-if-target-exists to the applier; harden `_version_tuple` usage.
- **Modify** `osmose/engine/config.py` — `from_dict` canonicalize guard; reconcile literal reads to 4.4.0 names (unified ingestion + module toggles + maturity).
- **Modify** `osmose/engine/config_validation.py`, `osmose/config/validator.py` — canonicalize at entry.
- **Create** `tests/test_config_migration_440.py`.

---

## Task 1: Lock the authoritative rename set (blocking gate)

**Files:** Create `osmose/config/aliases.py`

- [ ] **Step 1: Pull the live `Releases.java $15` block and confirm the rename set + semantics**

Run:
```bash
gh api repos/osmose-model/osmose/contents/java/src/main/java/fr/ird/osmose/util/version/Releases.java --jq '.content' | base64 -d | sed -n '630,740p'
```
Confirm every `updateKey(old, new)` in the v4.4.0 (`$15`) block, the skip-if-target-exists behavior (`updateKey` does not overwrite an existing target — the engine logs "already defined"), and that the toggle target is `module.multispecies.fisheries.enabled`. If the block contains renames beyond the table below (e.g. additional genetics/economy/spatial/fishery keys), ADD them to `RENAMES_440` before proceeding. Do NOT rely on the release notes or the crashing `-update`.

- [ ] **Step 2: Write `osmose/config/aliases.py` with the locked map**

```python
"""OSMOSE 4.4.0 config-key migration data + canonicalize/inverse helpers.

Faithful Python port of the OSMOSE Java `Releases.java` v4.4.0 ($15) `updateKey`
renames. The engine's own `-update` is broken (crashes in commentParameter), so
OSMOPY ports the logic. `updateKey(old, new)` renames old->new but SKIPS if the
new key already exists ("already defined") — see osmose/demo.py applier.

Keys are lowercase (the reader lowercases everything). RENAMES_440 maps an old
prefix to a new prefix; the applier matches `k == old or k.startswith(old + ".")`
so indexed `...spN` keys are caught via the `.` separator (the prefixes here
deliberately stop before `.sp`).
"""

from __future__ import annotations

# old_prefix -> new_prefix. Ported from Releases.java $15 (verified Task 1).
RENAMES_440: dict[str, str] = {
    "simulation.bioen.enabled": "module.bioenergetics.enabled",
    "simulation.genetic.enabled": "module.genetics.enabled",
    "fisheries.enabled": "module.multispecies.fisheries.enabled",
    "economy.enabled": "module.bioeconomics.enabled",
    "output.restart.enabled": "simulation.restart.enabled",
    "output.restart.recordfrequency.ndt": "simulation.restart.recordfrequency.ndt",
    "output.restart.spinup": "simulation.restart.spinup.nyear",
    "output.fishery.enabled": "output.fisheries.enabled",
    "output.fishery.byage.enabled": "output.fisheries.byage.enabled",
    "output.fishery.bysize.enabled": "output.fisheries.bysize.enabled",
    "output.spatial.fishery.enabled": "output.spatial.fisheries.enabled",
    # prefix renames (catch the indexed .spN suffix via startswith(old + ".")):
    "species.bioen.maturity": "species.maturity",
    "predation.ingestion.rate.max.bioen": "predation.ingestion.rate.max",
    "predation.coef.ingestion.rate.max.larvae.bioen": "predation.larval.ingestion.rate.increase.ratio",
}
```

- [ ] **Step 3: Commit**

```bash
git add osmose/config/aliases.py
git commit -m "feat(config): lock OSMOSE 4.4.0 rename set (ported from Releases.java \$15)"
```

NOTE: Task 1 produces no tests (it's a data-capture + verification gate). If the source block reveals the map is wrong/incomplete, STOP and report before continuing — the whole feature depends on this set being faithful.

---

## Task 2: Applier — skip-if-target-exists + the "4.4.0" chain entry

**Files:** Modify `osmose/demo.py`; Create `tests/test_config_migration_440.py`

- [ ] **Step 1: Write the failing test** (create `tests/test_config_migration_440.py`)

```python
from osmose.demo import migrate_config


def test_440_clean_renames():
    cfg = {
        "osmose.version": "4.3.3",
        "simulation.bioen.enabled": "true",
        "fisheries.enabled": "false",
        "output.restart.spinup": "5",
        "output.fishery.enabled": "true",
        "output.spatial.fishery.enabled": "true",
        "species.bioen.maturity.eta.sp0": "1.2",
        "species.bioen.maturity.m0.sp3": "0.5",
    }
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["module.multispecies.fisheries.enabled"] == "false"
    assert out["simulation.restart.spinup.nyear"] == "5"
    assert out["output.fisheries.enabled"] == "true"
    assert out["output.spatial.fisheries.enabled"] == "true"
    assert out["species.maturity.eta.sp0"] == "1.2"
    assert out["species.maturity.m0.sp3"] == "0.5"
    # old keys gone
    assert "simulation.bioen.enabled" not in out
    assert "species.bioen.maturity.eta.sp0" not in out
    assert out["osmose.version"] == "4.4.0"


def test_440_ingestion_merge_skip_if_target_exists():
    # base (3.5) AND bioen (3.0) both present for sp0 -> base wins, bioen dropped
    # (Java updateKey "already defined"); bioen-only sp1 -> renamed to base.
    cfg = {
        "osmose.version": "4.3.3",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.bioen.sp0": "3.0",
        "predation.ingestion.rate.max.bioen.sp1": "4.0",
    }
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"  # base kept
    assert "predation.ingestion.rate.max.bioen.sp0" not in out  # bioen dropped
    assert out["predation.ingestion.rate.max.sp1"] == "4.0"  # bioen-only -> renamed


def test_440_idempotent_on_new_keys():
    cfg = {"osmose.version": "4.4.0", "module.bioenergetics.enabled": "true"}
    out = migrate_config(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert "simulation.bioen.enabled" not in out
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py -q`
Expected: FAIL (the `"4.4.0"` step doesn't exist; `module.*` keys absent).

- [ ] **Step 3: Implement in `osmose/demo.py`**

3a. Append a `"4.4.0"` entry to `_MIGRATION_CHAIN` (after the `"4.3.0"` sentinel), importing the map:
```python
from osmose.config.aliases import RENAMES_440
# ... at the end of _MIGRATION_CHAIN:
    ("4.4.0", RENAMES_440),
```
(If a circular import arises — `demo` ↔ `config.aliases` — inline the dict literal in `_MIGRATION_CHAIN` instead and have `aliases.py` import it from `demo`, OR keep `RENAMES_440` in `aliases.py` and import it in `demo.py` at module top; `aliases.py` must NOT import `demo`.)

3b. Change the applier's rename inner loop to skip-if-target-exists:
```python
            for key in keys_to_rename:
                new_key = new_prefix + key[len(old_prefix) :]
                if new_key in result and new_key != key:
                    # Java updateKey: target already defined -> keep it, drop the old key.
                    result.pop(key)
                else:
                    result[new_key] = result.pop(key)
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py -q` → 3 pass.
Run: `.venv/bin/python -m pytest tests/test_demo.py -q` → still passes (older renames unaffected; skip-if-exists only changes collision behavior).

- [ ] **Step 5: Commit**

```bash
git add osmose/demo.py tests/test_config_migration_440.py
git commit -m "feat(config): add 4.4.0 migration chain entry + skip-if-target-exists applier"
```

---

## Task 3: `canonicalize_config` + version hardening

**Files:** Modify `osmose/config/aliases.py`, `osmose/demo.py`; Modify `tests/test_config_migration_440.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.config.aliases import canonicalize_config


def test_canonicalize_reports_deprecated_and_canonicalizes():
    cfg = {"osmose.version": "4.3.3", "simulation.bioen.enabled": "true"}
    out, deprecated = canonicalize_config(cfg)
    assert out["module.bioenergetics.enabled"] == "true"
    assert "simulation.bioen.enabled" in deprecated


def test_canonicalize_missing_version_does_not_corrupt_new_keys():
    # No osmose.version, but config already uses NEW keys -> must stay new, not be reverse-touched.
    cfg = {"module.bioenergetics.enabled": "true", "predation.ingestion.rate.max.sp0": "3.5"}
    out, _ = canonicalize_config(cfg)
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"


def test_canonicalize_snapshot_version_handled():
    cfg = {"osmose.version": "4.4.0-SNAPSHOT", "simulation.bioen.enabled": "true"}
    out, _ = canonicalize_config(cfg)
    # malformed/suffixed version must not crash and must still produce canonical keys
    assert out["module.bioenergetics.enabled"] == "true"
```

- [ ] **Step 2: Run, verify FAIL** (`canonicalize_config` undefined).

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py -k canonicalize -q`

- [ ] **Step 3: Implement**

3a. In `osmose/demo.py`, harden the malformed-version path: where the version is compared, treat an unparseable version (`_version_tuple` fell back to `(0,)` because the string had non-numeric parts) as "apply renames where the OLD key is present" — the existing per-key `canFind`-style guard (only renames keys that exist) already prevents corrupting absent keys, so the only fix needed is to ensure a config already in NEW form is not harmed. Add at the top of `migrate_config`, after computing `current`:
```python
    # A config that already carries 4.4.0 keys but lacks/has a malformed version is
    # treated as canonical (no old keys to rename anyway); guard the version compare.
    if current == target_version:
        return dict(config)
```
(The existing `if current == target_version: return dict(config)` already covers the exact-match case; the per-key existence check in the applier covers the rest, since NEW keys are never in `RENAMES_440`'s OLD set. No further change needed beyond confirming the applier only renames present old keys — it does.)

3b. In `osmose/config/aliases.py`, add:
```python
def canonicalize_config(cfg: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Migrate a config dict to canonical 4.4.0 keys; return (new_cfg, deprecated_old_keys).

    `deprecated_old_keys` = the OLD keys from RENAMES_440 that were present in the input
    (for one-time deprecation logging by callers). Idempotent on already-4.4.0 configs.
    """
    from osmose.demo import migrate_config

    deprecated = sorted(
        k
        for k in cfg
        if any(k == old or k.startswith(old + ".") for old in RENAMES_440)
    )
    return migrate_config(cfg, target_version="4.4.0"), deprecated
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add osmose/config/aliases.py osmose/demo.py tests/test_config_migration_440.py
git commit -m "feat(config): canonicalize_config wrapper + malformed-version hardening"
```

---

## Task 4: `to_target_keys` inverse (chain-inversion, leaf-scoped, merge non-invertible)

**Files:** Modify `osmose/config/aliases.py`; Modify `tests/test_config_migration_440.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.config.aliases import to_target_keys


def test_to_target_keys_reverses_to_4_3_3():
    cfg = {
        "osmose.version": "4.4.0",
        "module.bioenergetics.enabled": "true",
        "simulation.restart.spinup.nyear": "5",
        "species.maturity.eta.sp0": "1.2",
        "predation.ingestion.rate.max.sp0": "3.5",
        # pre-existing GROWTH maturity keys that must NOT be touched by the inverse:
        "species.maturity.size.sp0": "20.0",
        "species.maturity.age.sp0": "2.0",
    }
    out = to_target_keys(cfg, target_version="4.3.3")
    assert out["simulation.bioen.enabled"] == "true"
    assert out["output.restart.spinup"] == "5"
    assert out["species.bioen.maturity.eta.sp0"] == "1.2"
    # growth keys untouched (leaf-scoped reverse must not catch species.maturity.size/age):
    assert out["species.maturity.size.sp0"] == "20.0"
    assert out["species.maturity.age.sp0"] == "2.0"
    assert out["osmose.version"] == "4.3.3"


def test_to_target_keys_4_4_0_is_identity_plus_stamp():
    cfg = {"osmose.version": "4.3.3", "module.bioenergetics.enabled": "true"}
    out = to_target_keys(cfg, target_version="4.4.0")
    assert out["module.bioenergetics.enabled"] == "true"
    assert out["osmose.version"] == "4.4.0"


def test_to_target_keys_ingestion_merge_is_lossy_to_legacy_key():
    # The merge is non-invertible: the unified value reverses to the LEGACY base key,
    # NOT a reconstructed .bioen split. A 4.3.3 jar reads it as the legacy ingestion rate.
    cfg = {"osmose.version": "4.4.0", "predation.ingestion.rate.max.sp0": "3.5"}
    out = to_target_keys(cfg, target_version="4.3.3")
    assert out["predation.ingestion.rate.max.sp0"] == "3.5"  # stays the legacy key
    assert not any(".bioen." in k for k in out)              # no .bioen fabricated
```

- [ ] **Step 2: Run, verify FAIL** (`to_target_keys` undefined).

- [ ] **Step 3: Implement in `osmose/config/aliases.py`**

```python
# Inverse of RENAMES_440 (new_prefix -> old_prefix). LEAF-SCOPED for the maturity
# rename: only the bioenergetic leaves reverse, so the pre-existing growth keys
# species.maturity.size/age are NEVER touched. The ingestion MERGE is non-invertible
# (the .bioen split is not reconstructed) — the unified value reverses to the legacy key.
_INVERSE_440: dict[str, str] = {
    "module.bioenergetics.enabled": "simulation.bioen.enabled",
    "module.genetics.enabled": "simulation.genetic.enabled",
    "module.multispecies.fisheries.enabled": "fisheries.enabled",
    "module.bioeconomics.enabled": "economy.enabled",
    "simulation.restart.enabled": "output.restart.enabled",
    "simulation.restart.recordfrequency.ndt": "output.restart.recordfrequency.ndt",
    "simulation.restart.spinup.nyear": "output.restart.spinup",
    "output.fisheries.enabled": "output.fishery.enabled",
    "output.fisheries.byage.enabled": "output.fishery.byage.enabled",
    "output.fisheries.bysize.enabled": "output.fishery.bysize.enabled",
    "output.spatial.fisheries.enabled": "output.spatial.fishery.enabled",
    # leaf-scoped maturity reverse (NOT a bare species.maturity. prefix):
    "species.maturity.eta": "species.bioen.maturity.eta",
    "species.maturity.r": "species.bioen.maturity.r",
    "species.maturity.m0": "species.bioen.maturity.m0",
    "species.maturity.m1": "species.bioen.maturity.m1",
    "predation.larval.ingestion.rate.increase.ratio": "predation.coef.ingestion.rate.max.larvae.bioen",
    # ingestion merge is lossy: unified value -> legacy key (NOT the .bioen variant).
    # (Handled by the generic prefix reverse below; no .bioen reconstruction.)
}


def to_target_keys(cfg: dict[str, str], target_version: str = "4.3.3") -> dict[str, str]:
    """Emit config keys for a target engine version (inverse of canonicalize).

    target 4.4.0 -> identity + version stamp. target 4.3.x -> reverse the 4.4.0
    renames (longest new-prefix first so e.g. species.maturity.eta wins over a
    shorter prefix), set osmose.version. Reverse is per-key/prefix and leaf-scoped;
    keys not in _INVERSE_440 (incl. species.maturity.size/age) pass through.
    """
    result = dict(cfg)
    if target_version == "4.4.0":
        result["osmose.version"] = "4.4.0"
        return result
    for new_prefix in sorted(_INVERSE_440, key=len, reverse=True):
        old_prefix = _INVERSE_440[new_prefix]
        for key in [k for k in result if k == new_prefix or k.startswith(new_prefix + ".")]:
            reversed_key = old_prefix + key[len(new_prefix) :]
            if reversed_key not in result:
                result[reversed_key] = result.pop(key)
    result["osmose.version"] = target_version
    return result
```

- [ ] **Step 4: Run, verify PASS** + a round-trip check

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py -k "to_target or canonicalize or 440" -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add osmose/config/aliases.py tests/test_config_migration_440.py
git commit -m "feat(config): to_target_keys inverse (leaf-scoped, merge non-invertible)"
```

---

## Task 5: Engine `from_dict` canonicalize guard + literal-read reconciliation

**Files:** Modify `osmose/engine/config.py`; Modify `tests/test_config_migration_440.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.engine.config import EngineConfig


def _min_bioen_cfg(extra: dict) -> dict[str, str]:
    base = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "19.5",
        "species.k.sp0": "0.364",
        "species.t0.sp0": "-0.70",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.06",
        "species.lifespan.sp0": "4",
        "species.vonbertalanffy.threshold.age.sp0": "0",
        "mortality.subdt": "10",
        "predation.efficiency.critical.sp0": "0.57",
    }
    base.update(extra)
    return base


def test_from_dict_accepts_old_module_toggle_key():
    # OLD key on input -> engine still recognizes bioenergetics enabled (canonicalize guard).
    cfg_old = _min_bioen_cfg({"simulation.bioen.enabled": "false",
                              "predation.ingestion.rate.max.sp0": "3.5"})
    config = EngineConfig.from_dict(cfg_old)
    assert config.bioen_enabled is False


def test_from_dict_unified_ingestion_read():
    # OLD bioen ingestion key only -> after canonicalize it's the unified key; the engine's
    # bioen ingestion read must see the value (not the 0.0 default).
    cfg = _min_bioen_cfg({
        "simulation.bioen.enabled": "false",
        "predation.ingestion.rate.max.bioen.sp0": "4.2",
    })
    config = EngineConfig.from_dict(cfg)
    # focal_ingestion_rate reads predation.ingestion.rate.max.sp0 (unified) -> 4.2 after merge
    assert config.ingestion_rate[0] == 4.2
```

CONFIRMED symbols: `EngineConfig.bioen_enabled: bool` (config.py:1382) and `EngineConfig.ingestion_rate: NDArray` (config.py:1234). For a 1-species, no-background config, `ingestion_rate` = `[focal_ingestion_rate]` (config.py:848), so `config.ingestion_rate[0]` is sp0's `predation.ingestion.rate.max`. Asserts above are valid as written.

- [ ] **Step 2: Run, verify FAIL** (old key not recognized / ingestion default).

- [ ] **Step 3: Implement in `osmose/engine/config.py`**

3a. At the very top of `from_dict` (before any `cfg.get`), canonicalize:
```python
        from osmose.config.aliases import canonicalize_config
        cfg, _deprecated = canonicalize_config(cfg)
```

3b. Reconcile the literal reads to the unified/new names (the dict is now canonical):
- `fisheries.enabled` → `module.multispecies.fisheries.enabled`
- `simulation.bioen.enabled` → `module.bioenergetics.enabled`
- `simulation.genetic.enabled` → `module.genetics.enabled`
- `predation.ingestion.rate.max.bioen.sp{i}` (the `bioen_i_max` ≈1889 and `foraging_I_max` ≈1911 reads) → `predation.ingestion.rate.max.sp{i}` (the SAME unified key `focal_ingestion_rate` reads at ≈606)
- `predation.coef.ingestion.rate.max.larvae.bioen.sp{i}` (`bioen_theta` ≈1892) → `predation.larval.ingestion.rate.increase.ratio.sp{i}`
- `species.bioen.maturity.{eta,r,m0,m1}.sp{i}` → `species.maturity.{eta,r,m0,m1}.sp{i}`

- [ ] **Step 4: Run, verify PASS** + full engine regression

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py -k from_dict -q` → pass.
Run: `.venv/bin/python -m pytest tests/test_engine_bioen_integration.py tests/test_bioen_orchestration.py tests/test_genetics_integration.py tests/test_engine_fisheries.py -q` → pass (old-key fixtures still work via canonicalize; the bioen ingestion now reads the unified value — see Task 7 for the intended bioen change).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py tests/test_config_migration_440.py
git commit -m "feat(engine): canonicalize config at from_dict + unified-ingestion/new-key reads"
```

---

## Task 6: Validators canonicalize at entry

**Files:** Modify `osmose/engine/config_validation.py`, `osmose/config/validator.py`; Modify `tests/test_config_migration_440.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_config_validation_clean_on_old_keys():
    # An old-key config must validate warning-free (old keys canonicalized before the
    # unknown-key check; new keys are recognized via the allowlist).
    from osmose.engine.config_validation import validate

    cfg = {"osmose.version": "4.3.3", "simulation.bioen.enabled": "true",
           "output.restart.enabled": "false", "economy.enabled": "false"}
    unknowns = validate(cfg, mode="warn")  # returns list[UnknownKey] (config_validation.py:489)
    flagged = {u.key for u in unknowns}
    assert "simulation.bioen.enabled" not in flagged
    assert "economy.enabled" not in flagged
    assert "output.restart.enabled" not in flagged
```

NOTE: the entry point is `validate(cfg, mode)` (config_validation.py:489), returning `list[UnknownKey]` (each with a `.key`). Confirm the `UnknownKey` field name (`.key`) and the accepted `mode` values when writing the test.

- [ ] **Step 2: Run, verify FAIL** (old keys flagged unknown, or new restart/economy names not matched).

- [ ] **Step 3: Implement**

At the top of `validate(cfg, mode)` (config_validation.py:489) AND `osmose/config/validator.py::validate_config(config, registry)` (validator.py:11), canonicalize first:
```python
    from osmose.config.aliases import canonicalize_config
    cfg, _ = canonicalize_config(cfg)
```
**MANDATORY — not optional:** in PR1 the schema `key_pattern`s are still OLD (the move is PR2), so the AST-walked known-keys set does NOT contain the new names. The bundled example configs (eec, baltic, eec_full) carry OLD keys that canonicalize into NEW names, so without allowlist entries `test_from_dict_warn_mode_clean_on_example_configs` WILL fail. Add these to `_SUPPLEMENTARY_ALLOWLIST` (config_validation.py:45) — the four genuinely-needed-by-example-configs are `module.multispecies.fisheries.enabled`, `simulation.restart.spinup.nyear`, `simulation.restart.recordfrequency.ndt`, `output.fisheries.enabled` (`simulation.restart.enabled` is already allowlisted at :113); add the full new set for completeness: also `module.bioenergetics.enabled`, `module.genetics.enabled`, `module.bioeconomics.enabled`, `output.fisheries.byage.enabled`, `output.fisheries.bysize.enabled`, `output.spatial.fisheries.enabled`, `species.maturity.{eta,r,m0,m1}.sp{idx}`, `predation.larval.ingestion.rate.increase.ratio.sp{idx}`. (`predation.ingestion.rate.max.sp{idx}` is already schema-known at species.py:320.)
NOTE: canonicalizing in `validate_config` means renamed keys whose schema `key_pattern` hasn't moved yet (PR1) silently skip value-bounds validation until PR2 — acceptable (no false errors; bounds re-enabled when PR2 moves the patterns).

- [ ] **Step 4: Run, verify PASS** + the existing config-validation suite

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py -k validation -q` → pass.
Run: `.venv/bin/python -m pytest tests/test_engine_config_validation.py -q` → green, incl. `test_from_dict_warn_mode_clean_on_example_configs`.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config_validation.py osmose/config/validator.py tests/test_config_migration_440.py
git commit -m "feat(config): canonicalize at validators; allowlist 4.4.0 keys"
```

---

## Task 7: Bioenergetics parity gate (document the intended ingestion change)

**Files:** Modify `tests/test_config_migration_440.py`

- [ ] **Step 1: Write the bioen end-to-end consistency test** (append)

```python
def test_bioen_ingestion_unification_is_consistent(tmp_path):
    """After canonicalization, a bioen config with BOTH ingestion keys uses the SINGLE
    unified predation.ingestion.rate.max for both predation and the energy budget — the
    intended 4.4.0 behavior. The base value wins (skip-if-target-exists). This is an
    intended result change vs 4.3.x for bioen/Ev-OSMOSE configs (changelog note in PR2)."""
    cfg = _min_bioen_cfg({
        "simulation.bioen.enabled": "true",
        "predation.ingestion.rate.max.sp0": "3.5",       # legacy/base
        "predation.ingestion.rate.max.bioen.sp0": "3.0",  # bioen — dropped on merge
    })
    from osmose.config.aliases import canonicalize_config

    canon, _ = canonicalize_config(cfg)
    assert canon["predation.ingestion.rate.max.sp0"] == "3.5"      # base wins
    assert "predation.ingestion.rate.max.bioen.sp0" not in canon   # bioen dropped
    config = EngineConfig.from_dict(cfg)
    # the unified value (3.5) drives the engine's ingestion read used by both paths:
    assert config.ingestion_rate[0] == 3.5
```

NOTE: `from_dict` with `simulation.bioen.enabled=true` builds fine on this minimal fixture — every bioen read (config.py:1867-1913) is `_species_float_optional`/`_species_int_optional` with a default and `__post_init__` does not re-validate them, so no extra bioen params are needed (verified). The pure `canonicalize_config` assertions lock base-wins even if a future engine change makes the full build heavier.

- [ ] **Step 2–4: Run, verify it passes.**

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py -k bioen -q`

- [ ] **Step 5: Commit**

```bash
git add tests/test_config_migration_440.py
git commit -m "test(config): bioen ingestion-unification parity gate (base-wins, intended change)"
```

---

## Task 8: Final gates

**Files:** none (verification only)

- [ ] **Step 1: Lint + format**

Run; fix; re-run until clean:
- `.venv/bin/ruff check osmose/config/aliases.py osmose/demo.py osmose/engine/config.py osmose/engine/config_validation.py osmose/config/validator.py tests/test_config_migration_440.py`
- `.venv/bin/ruff format <same files>`

- [ ] **Step 2: pyright (clean `[dev]` venv, per the recurring CI gotcha)**

Run: `.venv/bin/pyright --pythonpath .venv/bin/python osmose/config/aliases.py osmose/demo.py osmose/engine/config.py` → 0 NEW errors.

- [ ] **Step 3: Targeted regression suites**

Run: `.venv/bin/python -m pytest tests/test_config_migration_440.py tests/test_demo.py tests/test_engine_config_validation.py tests/test_bioen_orchestration.py tests/test_engine_bioen_integration.py tests/test_genetics_integration.py tests/test_engine_fisheries.py tests/test_roundtrip.py -q` → all pass. (Bioen suites: if a test asserted a value that depended on the now-dropped `.bioen` ingestion key, update it to the unified base value — this is the intended change, not a regression.)

- [ ] **Step 4: Full suite**

Run: `.venv/bin/python -m pytest -q -m "not e2e"` → report counts; the only acceptable failures are the known `test_runner.py`/`test_study_fullmodel.py` xdist parallel-load flakes (confirm they pass in isolation if they fail).

- [ ] **Step 5: Commit (if any gate fixes were needed)**

```bash
git add -A
git commit -m "chore(config): PR1 final gate fixes"
```

---

## Notes

- **PR1 changes NO on-disk config output and NO UI** (schema/reader/writer/UI wiring is PR2), so it's parity-safe in isolation EXCEPT the intended bioen ingestion-unification change (Task 7), which is gated and documented.
- **DRY:** the single `RENAMES_440` map drives forward (`canonicalize_config`/chain) and is mirrored once for the inverse (`_INVERSE_440`); engine reads use the canonical (new) names.
- **The Task 1 gate is load-bearing** — if the live `Releases.java $15` differs from `RENAMES_440`, fix the map first.
- **PR2 (separate plan):** schema `key_pattern` moves + `_OUTPUT_ENABLE_FLAGS` rename; reader `read_file` canonicalize hook; `AppState.load_config` + scenario/fishbase routing; `ScenarioManager.load` canonicalize; `OsmoseConfigWriter`/`write_temp_config` `target_version` + `ROUTING` for `module.*`/`output.fisheries.*`; calibration `-P` override reverse-map + checkpoint-key migration; one-time deprecation UI notification; changelog note on the bioen change + scenario key-format.
