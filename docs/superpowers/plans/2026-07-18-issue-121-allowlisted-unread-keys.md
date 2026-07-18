# Fix #121 — allowlisted-but-unread config keys — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Python engine honor the two real upstream config keys it silently ignores (`output.tl.enabled`, `module.bioeconomics.enabled`), remove 5 verified-dead invented `output.*` keys and fix the bundled configs that set them, correct the false allowlist comments, and reframe PR #122's migration guide (whose two documented traps this fix closes).

**Architecture:** Two-line read-site fallback in `osmose/engine/config.py` (upstream name first, invented name as back-compat fallback). Allowlist edits + comment corrections in `osmose/engine/config_validation.py`. Per-file hand-edits to two bundled configs. Prose reframe in `docs/r-to-python-migration.md`. No `aliases.py`/`RENAMES_440` changes.

**Tech Stack:** Python 3.12, pytest, ruff, Sphinx 9.1 (`.venv/bin/sphinx-build`).

**Spec:** `docs/superpowers/specs/2026-07-18-issue-121-allowlisted-unread-keys-design.md`

## Global Constraints

- **METHOD RULE:** every "this key is dead / real / read" claim must be verified by running the code, not inferred from the allowlist. This issue exists because that inference failed repeatedly. Where a step asserts a key's status, the verification command is given — run it.
- **Jar checks are CASE-INSENSITIVE.** Java uses camelCase (`output.meanTL.byAge.enabled`), the reader lowercases. `grep -ci`, never `grep -cF`. (A case-sensitive check falsely cleared a key earlier.)
- **`data/examples_433_orig` is a FROZEN migration source** (`test_migrate_bob_native.py:7` — "migrate a copy of the ORIGINAL"). **Do NOT edit it.** Its dead keys stay; nothing runs strict validation on it, so de-allowlisting doesn't break it.
- **Back-compat:** the invented names (`output.meantl.enabled`, `simulation.economic.enabled`) must keep working as fallback — no existing config or test may silently break.
- **Canonical replacement in configs:** use the real upstream name where one exists (`output.tl.enabled`, not `output.meantl.enabled`). All replacement keys are verified real Java 4.4.1 jar keys (see Task 2 Step 0), so both engines produce the same output families — **eec parity (14/14) is preserved.**
- **CONFIG-FIX SCOPE (verified by repo-wide grep, do not narrow):** the 5 removed keys are set in exactly three LIVE configs — `data/examples`, `data/eec`, `data/minimal`. NOT `data/baltic`, NOT `data/eec_full` (clean), NOT `data/examples_433_orig` (frozen migration source — never edit). `data/minimal` sets only `output.frequency.ndtperyear`. An earlier draft missed `data/minimal`, which strands the parametrized test `test_from_dict_warn_mode_clean_on_example_configs[minimal]`.
- **VERIFICATION RUNS THE FULL SUITE.** The final gate is `pytest tests/ -q`, not a hand-picked subset. This is the plan's own #121 lesson: an earlier draft's "prove nothing broke" gates ran a nonexistent file (`test_config.py`) and skipped the file that goes red (`test_engine_config_validation.py`) — reasoning from a plausible filename to "this guards it" without running it.
- **`output.tl.enabled` → `output_meantl` is semantically correct, not just membership-correct** (verified by decompiling `OutputManager.class`): `output.tl.enabled` instantiates `WeightedSpeciesOutput(getTrophicLevel, getWeight)` = a weight-weighted mean trophic level per species = osmopy's `meanTL` output. The TL *distribution*/*spectrum* outputs are separate keys (`output.TL.perAge`, `output.meanTL.byAge`) with their own classes. Do not re-raise this.
- Branch `fix/issue-121-allowlisted-unread-keys` (already checked out). Commit after every task. Use `.venv/bin/python`; shell cwd resets — absolute paths.

## File Structure

| File | Change | Task |
|---|---|---|
| `osmose/engine/config.py` | 2 read-site fallbacks (`:923`, `:2431`) | 1 |
| `tests/test_r_dialect_migration_claims.py` | update the 2 trap assertions Layer A flips (`:196`, `:243`) + their docstrings/`TRAPS` framing; add Layer A tests | 1 |
| `data/examples/osm_param-output.csv` | replace/expand/delete dead output keys | 2 |
| `data/eec/osm_param-output.csv` | replace/expand/delete dead output keys | 2 |
| `data/minimal/osm_param-output.csv` | delete the one dead key (`output.frequency.ndtperyear`) | 2 |
| `osmose/engine/config_validation.py` | remove 5 dead allowlist keys + fix 2 comment blocks | 3 |
| `docs/r-to-python-migration.md` | reframe the 2 fixed traps + the "shim rescues half" arithmetic; repoint the systemic-case ref at #123 | 4 |

---

### Task 1: Layer A — canonical read-site fallback + fix the trap tests it flips

**Files:**
- Modify: `osmose/engine/config.py:923`, `osmose/engine/config.py:2431`
- Modify: `tests/test_r_dialect_migration_claims.py` (add Layer A tests; update `:196`, `:243` + framing)

**Interfaces:**
- Consumes: `_enabled(cfg: dict[str, str], key: str) -> bool` (`config.py:167`), `EngineConfig.from_dict(cfg) -> EngineConfig` with bool attrs `.output_meantl`, `.economics_enabled`.
- Produces: after this task, `output.tl.enabled` and `module.bioeconomics.enabled` flip those attributes; the invented names still do too.

- [ ] **Step 1: Write the failing Layer A tests**

Add to `tests/test_r_dialect_migration_claims.py` (imports `EngineConfig`, `OsmoseConfigReader`, `MINIMAL_CONFIG`, `_probe`, `minimal_cfg` already exist in the file):

```python
def test_layer_a_output_tl_enabled_now_honored(minimal_cfg):
    """#121 Layer A: the real upstream key output.tl.enabled now flips mean-TL output.

    Before the fix the engine read only the osmopy-invented output.meantl.enabled and silently
    ignored the genuine upstream name. Canonical realignment (config.py:923) reads the upstream
    name first, invented name as fallback.
    """
    assert _probe(minimal_cfg).output_meantl is False, "baseline"
    # upstream name — now honored (was silently ignored)
    assert _probe(minimal_cfg, **{"output.tl.enabled": "true"}).output_meantl is True
    # invented name still works (back-compat fallback)
    assert _probe(minimal_cfg, **{"output.meantl.enabled": "true"}).output_meantl is True


def test_layer_a_bioeconomics_enabled_now_honored(minimal_cfg):
    """#121 Layer A: upstream module.bioeconomics.enabled now flips economics.

    economy.enabled already renames to module.bioeconomics.enabled via RENAMES_440, so authentic
    legacy configs now reach a key the engine reads (config.py:2431).
    """
    assert _probe(minimal_cfg).economics_enabled is False, "baseline"
    assert _probe(minimal_cfg, **{"module.bioeconomics.enabled": "true"}).economics_enabled is True
    # invented name still works (back-compat fallback)
    assert _probe(minimal_cfg, **{"simulation.economic.enabled": "true"}).economics_enabled is True
```

- [ ] **Step 2: Run the new tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py::test_layer_a_output_tl_enabled_now_honored tests/test_r_dialect_migration_claims.py::test_layer_a_bioeconomics_enabled_now_honored -v`
Expected: **FAIL** — `output.tl.enabled` / `module.bioeconomics.enabled` currently leave the attribute `False` (that IS the bug).

- [ ] **Step 3: Implement the two read-site fallbacks**

In `osmose/engine/config.py`, line 923, change:
```python
        "output_meantl": _enabled(cfg, "output.meantl.enabled"),
```
to:
```python
        # #121: read the real upstream name (output.tl.enabled) first; the osmopy-invented
        # output.meantl.enabled remains a back-compat fallback.
        "output_meantl": _enabled(cfg, "output.tl.enabled") or _enabled(cfg, "output.meantl.enabled"),
```

In `osmose/engine/config.py`, line 2431, change:
```python
        economics_enabled = _enabled(cfg, "simulation.economic.enabled")
```
to:
```python
        # #121: read the real upstream 4.4.0 name (module.bioeconomics.enabled) first; the
        # osmopy-invented simulation.economic.enabled remains a back-compat fallback.
        economics_enabled = _enabled(cfg, "module.bioeconomics.enabled") or _enabled(cfg, "simulation.economic.enabled")
```

- [ ] **Step 4: Run the new tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py::test_layer_a_output_tl_enabled_now_honored tests/test_r_dialect_migration_claims.py::test_layer_a_bioeconomics_enabled_now_honored -v`
Expected: **PASS, 2 passed.**

- [ ] **Step 5: Update the two trap tests Layer A intentionally flips**

The pre-fix trap tests now assert stale behavior. Run them to confirm they break:
`.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py::test_trap_output_tl_enabled_is_silently_ignored tests/test_r_dialect_migration_claims.py::test_trap_economy_enabled_is_silently_ignored -v`
Expected: **FAIL** at the `is False` lines (`:196`, `:243`) — this is the designed tripwire firing.

Replace `test_trap_output_tl_enabled_is_silently_ignored` (rename to reflect the fix) with:
```python
def test_output_tl_enabled_now_read_after_121(minimal_cfg):
    """FORMERLY a trap (the guide's headline example); FIXED in #121.

    output.tl.enabled is the real upstream Java name. Before #121 the engine read only the
    invented output.meantl.enabled and silently ignored the upstream name. Now honored.
    """
    assert _probe(minimal_cfg).output_meantl is False, "baseline"
    assert _probe(minimal_cfg, **{"output.tl.enabled": "true"}).output_meantl is True
    assert _probe(minimal_cfg, **{"output.meantl.enabled": "true"}).output_meantl is True
```

Replace `test_trap_economy_enabled_is_silently_ignored` with:
```python
def test_bioeconomics_enabled_now_read_after_121(minimal_cfg):
    """FORMERLY a trap (the guide's latent example); FIXED in #121.

    economy.enabled -> module.bioeconomics.enabled (RENAMES_440, upstream's real 4.4.0 name).
    Before #121 the engine read only the invented simulation.economic.enabled. Now the upstream
    name is honored, so an authentic migrated config gets economics.
    """
    assert _probe(minimal_cfg).economics_enabled is False, "baseline"
    assert _probe(minimal_cfg, **{"module.bioeconomics.enabled": "true"}).economics_enabled is True
    assert _probe(minimal_cfg, **{"simulation.economic.enabled": "true"}).economics_enabled is True
```

These duplicate the Step-1 tests' intent; that's fine — delete the two Step-1 tests
(`test_layer_a_*`) to avoid duplication, keeping only these two renamed ones. Also update the
`TRAPS` table comment block (`:44-58`) and `test_traps_carry_a_provenance_citation` /
`test_a_one_sided_assertion_would_be_vacuous`: the `TRAPS` list documented the two now-fixed
traps. Keep `TRAPS` (the provenance-citation and vacuity tests still exercise it) but update its
header comment to say these are **formerly-live traps, fixed in #121, retained as regression
anchors** — not live silent gaps.

- [ ] **Step 6: Run the whole guard file — expect green**

Run: `.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py -v`
Expected: **all pass** (count = prior 11 minus 0 net, since 2 renamed replace 2 removed + 2 Step-1 deleted → 11). Confirm no test still asserts `output.tl.enabled`/`module.bioeconomics.enabled` leaves the attribute `False`:
`grep -n "output.tl.enabled\|module.bioeconomics" tests/test_r_dialect_migration_claims.py` — no `is False` assertion on a set of those keys.

- [ ] **Step 7: Lint + broader guard**

Run: `.venv/bin/python -m ruff check osmose/ tests/ && .venv/bin/python -m ruff format --check osmose/ tests/`
Expected: clean.
Run (these files EXIST — `tests/test_config.py` does NOT; an earlier draft named it and the command ran zero tests):
`.venv/bin/python -m pytest tests/test_engine_config.py tests/test_config_reader.py tests/test_config_validation.py tests/test_engine_config_validation.py -q`
Expected: pass (or pre-existing skips only). `test_engine_config_validation.py` is included deliberately — it holds the parametrized clean-config test that Task 3 must not break.

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/config.py tests/test_r_dialect_migration_claims.py
git commit -m "fix(#121): engine honors upstream output.tl/module.bioeconomics keys (Layer A)"
```

---

### Task 2: Layer B configs — fix the two live bundled configs (replace/expand/delete)

Make `data/examples`, `data/eec`, and `data/minimal` actually produce the output they request. **Do NOT touch `data/examples_433_orig`** (frozen migration source). These edits are per-file and per-key — no blanket replace.

**Files:**
- Modify: `data/examples/osm_param-output.csv`
- Modify: `data/eec/osm_param-output.csv`
- Modify: `data/minimal/osm_param-output.csv`

- [ ] **Step 0: Confirm every replacement key is a REAL Java jar key (eec parity)**

`data/eec` is parity-tested (14/14). The replacement keys must be real upstream keys so Java produces the same families. Verify (case-insensitive):
```bash
cd /home/razinka/osmopy/osmose-java
for k in output.biomass.byage.enabled output.abundance.byage.enabled output.biomass.bysize.enabled output.abundance.bysize.enabled output.size.enabled output.tl.enabled; do
  echo "$k $(unzip -p osmose-4.4.1-jar-with-dependencies.jar 'fr/ird/osmose/**' 2>/dev/null | strings | grep -ci "$k")"
done
```
Expected: every key ≥ 1. (Verified 2026-07-18: all real.) If any is 0, STOP — it would break Java parity.

- [ ] **Step 1: Write the config-correctness test**

Add a new test file `tests/test_issue_121_bundled_configs.py`:

```python
"""#121 Layer B: the fixed bundled configs actually produce the output they request."""

from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig

ROOT = Path(__file__).parent.parent


def _cfg(rel: str) -> EngineConfig:
    return EngineConfig.from_dict(OsmoseConfigReader().read(ROOT / rel))


def test_examples_requests_byage_bysize_meansize_tl():
    """data/examples is the new-user starting point; it must not silently drop output."""
    ec = _cfg("data/examples/osm_all-parameters.csv")
    assert ec.output_biomass_byage and ec.output_abundance_byage
    assert ec.output_biomass_bysize and ec.output_abundance_bysize
    assert ec.output_mean_size
    assert ec.output_meantl


def test_eec_requests_byage_bysize_meansize_tl():
    # data/eec's top-level is osm_all-parameters.csv (NOT eec_all-parameters.csv — that name
    # exists only under the unrelated data/eec_full/).
    ec = _cfg("data/eec/osm_all-parameters.csv")
    assert ec.output_biomass_byage and ec.output_abundance_byage
    assert ec.output_biomass_bysize and ec.output_abundance_bysize
    assert ec.output_mean_size
    assert ec.output_meantl


def test_no_dead_output_keys_remain_in_live_configs():
    """The 5 removed invented keys must be gone from ALL THREE live configs (not 433_orig)."""
    dead = ("output.byage.enabled", "output.bysize.enabled", "output.meansize.enabled",
            "output.trophiclevel.enabled", "output.frequency.ndtperyear")
    for rel in ("data/examples/osm_param-output.csv", "data/eec/osm_param-output.csv",
                "data/minimal/osm_param-output.csv"):
        text = (ROOT / rel).read_text()
        for k in dead:
            assert k not in text, f"{k} still in {rel}"


def test_examples_actually_produces_meantl_output(tmp_path):
    """Spec requires proving OUTPUT, not just the flag (#121's whole thesis: run it).

    A True flag under-proves — the CSV writers gate on flag AND data presence. Run a short sim
    to disk and assert the mean-TL CSV is materialized and non-empty. `mean_trophic_level()` is
    the real OsmoseResults accessor (results.py:451) — NOT `meantl()`, which does not exist.
    """
    from osmose.engine import PythonEngine
    from osmose.results import OsmoseResults

    cfg = OsmoseConfigReader().read(ROOT / "data/examples/osm_all-parameters.csv")
    cfg["simulation.time.nyear"] = "1"  # keep it fast
    PythonEngine().run(config=cfg, output_dir=tmp_path, seed=0)
    results = OsmoseResults.from_outputs(tmp_path)
    tl = results.mean_trophic_level()
    assert not tl.empty, "meanTL output empty — output.tl.enabled not honored end-to-end"
```

Note: `test_*_requests_*` load the top-level `osm_all-parameters.csv` (includes the output sub-file via `osmose.configuration.output`; confirmed at `data/examples/osm_all-parameters.csv:44`). If `run()`/`from_outputs()`/`mean_trophic_level()` signatures differ from the above, correct them against the real source before running — do NOT leave a call pointing at a method that doesn't exist (that IS the #121 failure). The accessor `mean_trophic_level()` and the `meanTL` CSV are verified real (`results.py:451,453`).

- [ ] **Step 2: Run — expect FAIL**

Run: `.venv/bin/python -m pytest tests/test_issue_121_bundled_configs.py -v`
Expected: **FAIL** — the configs set dead keys, so the working attributes are `False` and the dead keys are still present.

- [ ] **Step 3: Edit `data/examples/osm_param-output.csv`**

Current relevant lines (verified):
```
 9  output.trophiclevel.enabled ; true
10  output.meansize.enabled ; true
13  output.abundance.bysize.enabled ; true      # working key ALREADY present
16  output.frequency.ndtperyear ; 24            # dead; working recordfrequency.ndt already at line 17
17  output.recordfrequency.ndt ; 1              # working key ALREADY present
18  output.bysize.enabled ; true
19  output.byage.enabled ; true
```
Apply exactly:
- Line 9 `output.trophiclevel.enabled ; true` → `output.tl.enabled ; true` (real upstream name; now read after Layer A).
- Line 10 `output.meansize.enabled ; true` → `output.size.enabled ; true`.
- Line 16 `output.frequency.ndtperyear ; 24` → **delete the line** (working `output.recordfrequency.ndt ; 1` already present at line 17; do not duplicate/conflict).
- Line 18 `output.bysize.enabled ; true` → `output.biomass.bysize.enabled ; true` (abundance.bysize already at line 13; this completes the pair).
- Line 19 `output.byage.enabled ; true` → two lines: `output.biomass.byage.enabled ; true` and `output.abundance.byage.enabled ; true`.

- [ ] **Step 4: Edit `data/eec/osm_param-output.csv`**

Current relevant lines (verified):
```
 9  output.trophiclevel.enabled ; true
10  output.meansize.enabled ; true
14  output.frequency.ndtperyear ; 24            # dead; working recordfrequency.ndt already at line 15
15  output.recordfrequency.ndt ; 1              # working key ALREADY present
16  output.bysize.enabled ; true
17  output.byage.enabled ; true
```
Apply exactly:
- Line 9 → `output.tl.enabled ; true`.
- Line 10 → `output.size.enabled ; true`.
- Line 14 `output.frequency.ndtperyear ; 24` → **delete the line**.
- Line 16 `output.bysize.enabled ; true` → two lines: `output.biomass.bysize.enabled ; true` and `output.abundance.bysize.enabled ; true` (eec has NEITHER, so add both).
- Line 17 `output.byage.enabled ; true` → two lines: `output.biomass.byage.enabled ; true` and `output.abundance.byage.enabled ; true`.

- [ ] **Step 4b: Edit `data/minimal/osm_param-output.csv`** (missed by an earlier draft — its omission strands the `[minimal]` parametrized clean-config test)

`data/minimal` sets only ONE removed key. Current relevant lines (verified):
```
10  output.frequency.ndtperyear ; 12            # dead
11  output.recordfrequency.ndt ; 1              # working key ALREADY present
```
Apply exactly:
- Line 10 `output.frequency.ndtperyear ; 12` → **delete the line** (working `output.recordfrequency.ndt ; 1` already present at line 11). That is the only edit `data/minimal` needs.

- [ ] **Step 5: Run the config test — expect PASS, and check for duplicate keys**

Run: `.venv/bin/python -m pytest tests/test_issue_121_bundled_configs.py -v`
Expected: **PASS** (all config-correctness + non-empty-output + no-dead-keys tests).
Check no key is now set twice in ANY of the three edited files:
```bash
for f in data/examples/osm_param-output.csv data/eec/osm_param-output.csv data/minimal/osm_param-output.csv; do
  echo "$f:"; cut -d';' -f1 "$f" | grep -v '^\s*#' | sed 's/ *$//' | sort | uniq -d
done
```
Expected: no output (no duplicate keys).

- [ ] **Step 6: Handle output-snapshot fallout**

These configs now produce extra output families (byAge, bySize, meanSize). Find any test that runs these configs and snapshots/asserts their outputs:
```bash
grep -rln "data/examples\|data/eec\|eec_all-parameters\|examples/osm" tests/ | xargs grep -ln "output\|snapshot\|Simu\|results\|biomass" 2>/dev/null
```
For each hit, run it. If a test breaks **because new output appeared** (not because something is wrong), re-baseline it intentionally and note it in the commit. If a test breaks for any OTHER reason, STOP — that is a real regression, not expected fallout. If none break, say so. (eec parity is preserved because all added keys are real Java keys — Step 0 — so the Java engine produces the same families; but still run any eec parity test that exists.)

- [ ] **Step 7: Commit**

```bash
git add data/examples/osm_param-output.csv data/eec/osm_param-output.csv data/minimal/osm_param-output.csv tests/test_issue_121_bundled_configs.py
git commit -m "fix(#121): bundled examples produce the output they request (Layer B configs)"
```

---

### Task 3: Layer B allowlist removal + Layer C comment corrections

Both are in `osmose/engine/config_validation.py`.

**Files:**
- Modify: `osmose/engine/config_validation.py` (remove 5 keys `:101-105`; fix comments `:98-100`, `:134-138`)
- Modify: `tests/test_issue_121_bundled_configs.py` (add the strict-validation assertion)

- [ ] **Step 1: Re-verify the 5 keys are dead on three fronts (METHOD RULE)**

Do NOT remove on faith. Run:
```bash
cd /home/razinka/osmopy
for k in output.byage.enabled output.bysize.enabled output.meansize.enabled output.trophiclevel.enabled output.frequency.ndtperyear; do
  py=$(grep -rn "$k" osmose/ --include=*.py | grep -vE "config_validation|schema/" | wc -l)
  j441=$(cd osmose-java && unzip -p osmose-4.4.1-jar-with-dependencies.jar 'fr/ird/osmose/*.class' 2>/dev/null | strings | grep -ci "$k")
  j433=$(cd osmose-java && unzip -p osmose_4.3.3-jar-with-dependencies.jar 'fr/ird/osmose/*.class' 2>/dev/null | strings | grep -ci "$k")
  st=$(grep -c "$k" osmose/java_background_staging.py)
  echo "$k  py:$py jar4.4.1:$j441 jar4.3.3:$j433 staged:$st"
done
```
Expected: every line `py:0 jar4.4.1:0 jar4.3.3:0 staged:0`. **If any is non-zero, STOP** — that key is not dead; remove it from the removal set and report.

- [ ] **Step 2: Write the strict-validation test**

Add to `tests/test_issue_121_bundled_configs.py`:
```python
from osmose.engine.config_validation import validate


def test_dead_keys_now_flagged_unknown_by_strict_validation():
    """After de-allowlisting, the 5 invented keys are reported unknown (they should be)."""
    dead = ("output.byage.enabled", "output.bysize.enabled", "output.meansize.enabled",
            "output.trophiclevel.enabled", "output.frequency.ndtperyear")
    unknown = {u.key for u in validate({k: "true" for k in dead}, "warn")}
    assert set(dead) <= unknown, f"still allowlisted: {set(dead) - unknown}"


def test_working_replacement_keys_are_recognized():
    """The keys the engine actually reads must NOT be flagged unknown."""
    working = ("output.tl.enabled", "output.size.enabled", "output.recordfrequency.ndt",
               "output.biomass.byage.enabled", "output.abundance.byage.enabled",
               "output.biomass.bysize.enabled", "output.abundance.bysize.enabled")
    unknown = {u.key for u in validate({k: "true" for k in working}, "warn")}
    assert not (set(working) & unknown), f"wrongly flagged: {set(working) & unknown}"


def test_keys_with_real_lineage_stay_allowlisted():
    """Keys with real Java lineage must NOT be flagged (would be a false 'unknown')."""
    keep = ("output.diet.stage.threshold.sp0", "output.diet.stage.structure",
            "species.conversion2tons.sp0", "ltl.conversion2tons.rsc0")
    unknown = {u.key for u in validate({k: "true" for k in keep}, "warn")}
    assert not (set(keep) & unknown), f"wrongly de-allowlisted: {set(keep) & unknown}"
```

- [ ] **Step 3: Run — expect the first new test to FAIL**

Run: `.venv/bin/python -m pytest tests/test_issue_121_bundled_configs.py::test_dead_keys_now_flagged_unknown_by_strict_validation tests/test_issue_121_bundled_configs.py::test_keys_with_real_lineage_stay_allowlisted -v`
Expected: `test_dead_keys...` **FAILS** (still allowlisted); `test_keys_with_real_lineage...` **PASSES** (they're still allowlisted, correctly).

- [ ] **Step 4: Remove the 5 dead keys from the allowlist**

In `osmose/engine/config_validation.py`, delete these 5 lines (`:101-105`):
```python
        "output.byage.enabled",
        "output.bysize.enabled",
        "output.meansize.enabled",
        "output.trophiclevel.enabled",
        "output.frequency.ndtperyear",
```
Leave `"output.diet.stage.structure"` and everything else in the block.

- [ ] **Step 5: Fix the false comment (Layer C, target 1)**

The block comment above them (`:98-100`) currently reads:
```python
        # --- Output configuration keys (Java-side output layer) ---
        # These control the Java engine's output; the Python engine has its
        # own output system and does not parse these.
```
Replace with (it now covers only the keys that remain, e.g. `output.diet.stage.structure`, plus honesty about the removed ones):
```python
        # --- Output configuration keys (Java-side output layer) ---
        # These are real Java-engine output keys the Python engine does not parse.
        # (#121 removed 5 INVENTED coarse toggles that were here — output.byage/bysize/
        # meansize/trophiclevel.enabled, output.frequency.ndtperyear — which are in NEITHER
        # jar and nothing read. The working keys are output.biomass.byage.enabled +
        # output.abundance.byage.enabled, output.size.enabled, output.tl.enabled,
        # output.recordfrequency.ndt. They are now flagged unknown, correctly.)
```

- [ ] **Step 6: Fix the misleading conversion2tons comment (Layer C, target 2)**

The block at `:134-138` currently claims these are "Read by the Java engine." That is not accurate for these key names. Replace with:
```python
        # --- Conversion-to-tons keys (legacy 4.3.x forms) ---
        # The real 4.4.1 key is plankton.conversion2tons(.plk) -> resource.conversion2tons
        # (demo.py). These species./ltl. forms are LEGACY 4.3.x names (0 hits in either jar),
        # kept allowlisted so the preserved 4.3.3 original (data/examples_433_orig) and the
        # minimal fixtures don't surface unknown-key warnings. Aliasing them to
        # resource.conversion2tons is possible future work (out of #121 scope).
```
(Keep the two key entries `"species.conversion2tons.sp{idx}"`, `"ltl.conversion2tons.rsc{idx}"` unchanged.)

- [ ] **Step 7: Run the strict-validation tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_issue_121_bundled_configs.py -v`
Expected: **all pass** (dead keys flagged, working keys recognized, lineage keys retained).

- [ ] **Step 8: Full guard — nothing else broke on de-allowlisting**

**Run `test_engine_config_validation.py` — it holds `test_from_dict_warn_mode_clean_on_example_configs`, parametrized over `[eec, baltic, eec_full, examples, minimal]`, which runs strict validation on those configs and IS the test de-allowlisting can break.** An earlier draft ran `test_config_validation.py` (a *different* file) and would have shipped the break unobserved.
```bash
.venv/bin/python -m pytest tests/test_engine_config_validation.py tests/test_config_validation.py tests/test_migrate_bob_native.py -q
```
Expected: pass — because Task 2 already cleaned `output.frequency.ndtperyear` from `data/minimal` (the only parametrized config that set a removed key besides examples/eec, which are also cleaned). If `[minimal]` (or any) fails with an unknown-key error naming a removed key, a live config still sets it — go back to Task 2.
Run: `.venv/bin/python -m ruff check osmose/ tests/ && .venv/bin/python -m ruff format --check osmose/ tests/` — clean.

- [ ] **Step 9: Commit**

```bash
git add osmose/engine/config_validation.py tests/test_issue_121_bundled_configs.py
git commit -m "fix(#121): de-allowlist 5 dead invented keys + correct false comments (Layers B/C)"
```

---

### Task 4: Reframe the migration guide — its two documented traps are now fixed

`docs/r-to-python-migration.md` documents `output.tl.enabled` (headline) and `economy.enabled` (latent) as **live** silent traps. Layer A fixed both. Reframe without gutting: the class-level lesson and the still-live spatial-inputs trap stay.

**Three passages become false after Layer A; a grep alone will NOT route you to all of them** — the "shim rescues half" arithmetic is invisible to the obvious greps. Fix all THREE by name:
1. `### The two traps you can verify right now` (~line 232)
2. `### The shim rescues half and strands half` (~line 119) — **the one an earlier draft missed**
3. Appendix "The two verified traps (reference)" (~line 607) + the systemic-case reference (~line 620)

**Files:**
- Modify: `docs/r-to-python-migration.md`

- [ ] **Step 1: Reframe `### The two traps you can verify right now` (~line 232)**

Rewrite so the two keys are **worked examples fixed in #121**, illustrating the *class* — not reproducible traps:
- State plainly: `output.tl.enabled` and `economy.enabled` were silently ignored before #121; the engine now honors both (link #121).
- Keep the class lesson: config keys can load clean and do nothing; run `scripts/check_config.py` + `validation.strict.enabled=error` on YOUR config.
- Point "what still bites you" at the **spatial-inputs `.nc` trap** (untouched — still the guide's #1 example), missing sub-configs, cross-file precedence, and restart (#120, NOT fixed).
- Do NOT claim the reader can reproduce the two fixed traps.

- [ ] **Step 2: Fix the arithmetic in `### The shim rescues half and strands half` (~lines 119–146)**

Layer A moves `economy.enabled` from "strands" to "reaches the engine." Read the subsection and apply:
- ~line 127 "**Four** migrate to a key the engine never reads" → "**Three** migrate…" (drop `economy.enabled` from this list; keep `output.restart.enabled`, `output.restart.spinup`, `output.fishery.enabled`).
- ~line 137 "**Four** reach the engine and change its behavior" → "**Five** reach…" (add `economy.enabled → module.bioeconomics.enabled`).
- ~lines 143–144 "half arrive and half don't … three of the four dead ones are fully silent" → "**five of eight arrive, three don't** … the three dead ones are fully silent".
- ~lines 130–136: the mechanism claim that `economy.enabled`'s migrated key "does nothing, because the simulation's actual economics switch is a different key" is now **false** — rewrite to say the migrated key `module.bioeconomics.enabled` is now read directly (fixed in #121).
- Also check ~line 256 (a back-reference to this subsection's "rescues and strands" framing) and adjust if it states the same arithmetic.

- [ ] **Step 3: Update ONLY the two appendix trap rows (~lines 614–615) to "fixed in #121" — LEAVE the systemic reference (~line 620) as "tracked", repointed at #123**

- Rows 614–615 (`output.tl.enabled`, `economy.enabled`): change "silently ignored" → "fixed in #121". Also update their now-stale mechanism text: row 614 says the engine reads `output.meantl.enabled` (now it reads `output.tl.enabled` first); row 615 says "the engine's real switch is `simulation.economic.enabled`" (now `module.bioeconomics.enabled` works too). State the post-fix reality.
- **Line ~620 is the GENERAL/systemic case** ("The real fix is tooling that names the correct key… tracked in #121"). This PR does NOT ship the systemic fix (Layer D) — it is deferred to **#123**. So do **NOT** change this to "fixed"; change `[#121]` → `[#123]` and keep it "tracked". A blanket "tracked→fixed" here would assert the systemic fix shipped, which is false.

- [ ] **Step 4: Scan for any OTHER now-false claim**

```bash
grep -niE "output\.tl\.enabled|economy\.enabled|simulation\.economic|module\.bioeconomics|silently (ignored|absent|get none)|Four migrate|Four reach|half arrive" docs/r-to-python-migration.md
```
For each hit confirm it is (a) accurate post-fix / historical framing, or (b) about a STILL-live trap (spatial inputs, restart #120, missing sub-config, cross-file precedence). **Restart (#120) is NOT fixed — leave its trap framing intact.** The spatial-inputs `.nc` trap is NOT fixed — leave it. Do not accidentally reframe a still-live trap as fixed.

- [ ] **Step 5: Clean build**

```bash
rm -rf docs/_build/html && .venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```
Expected: exit 0, no warnings. (Stale cache skips `-W` — the `rm` is required.)

- [ ] **Step 6: Commit**

```bash
git add docs/r-to-python-migration.md
git commit -m "docs(#121): reframe the two now-fixed traps in the migration guide"
```

---

### Task 5: Full verification, close #121, PR

- [ ] **Step 1: FULL suite + lint + clean docs build**

Run the **whole** suite, not a hand-picked subset. This is the plan's root fix: an earlier draft's subset ran a nonexistent file and skipped the file that goes red — the exact #121 failure, committed against the plan's own gate. The full suite catches any stranded config, any snapshot fallout, any broken guide test automatically.
```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check osmose/ ui/ tests/ && .venv/bin/python -m ruff format --check osmose/ ui/ tests/
rm -rf docs/_build/html && .venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html && echo "DOCS CLEAN"
```
Expected: all pass (modulo pre-existing skips / known-flaky CI-only tests — compare against a clean `git stash` baseline if unsure); ruff clean; `DOCS CLEAN`. **If any test fails, it is in scope — do not ship over it.**

- [ ] **Step 2: End-to-end proof — the fix actually works**

```bash
.venv/bin/python -c "
import logging; logging.disable(logging.CRITICAL)
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
# an authentic config using the upstream names now works:
cfg = OsmoseConfigReader().read(Path('data/examples/osm_all-parameters.csv'))
ec = EngineConfig.from_dict(cfg)
print('examples output_biomass_byage:', ec.output_biomass_byage, '(was silently False)')
print('examples output_meantl       :', ec.output_meantl)
"
```
Expected: both `True`.

- [ ] **Step 3: Verify the METHOD-RULE promise — no unverified dead-claim shipped**

Confirm the diff removed exactly the 5 three-front-cleared keys and nothing with lineage:
```bash
git diff 1928a90..HEAD -- osmose/engine/config_validation.py | grep '^-' | grep '"output\|"species.conversion\|"ltl.conversion\|"output.diet'
```
Expected: only the 5 `output.*` dead keys as removals; no `diet.stage`, no `conversion2tons`.

- [ ] **Step 4: Push + PR (closes #121)**

```bash
git push -u origin fix/issue-121-allowlisted-unread-keys
gh pr create --base master --head fix/issue-121-allowlisted-unread-keys \
  --title "fix(#121): honor upstream config keys the engine silently ignored" \
  --body "$(cat <<'BODY'
Closes #121.

Fixes the two verified user-facing bugs where osmopy invented its own config key name and silently ignored the genuine upstream one:
- `output.tl.enabled` (real 4.4.1 jar key) — the engine now reads it; `output.meantl.enabled` stays as a back-compat fallback.
- `module.bioeconomics.enabled` (upstream's 4.4.0 name, the `RENAMES_440` target of `economy.enabled`) — now honored; `simulation.economic.enabled` stays as fallback.

Removes 5 verified-dead invented `output.*` keys from the validation allowlist (each cleared on three fronts: Python-unread, 0 hits in BOTH jars case-insensitively, not staged for Java) and fixes `data/examples` + `data/eec` + `data/minimal` to produce the output they request (`data/examples` — the new-user starting point — was silently dropping by-age/by-size/mean-size output). All replacement keys are real Java 4.4.1 keys, so **eec parity (14/14) is preserved**. Corrects two factually-wrong allowlist comments.

Verification corrected the issue's own list twice: `output.diet.stage.threshold` is staged for Java 4.4.1 (kept), and the `conversion2tons` pair are legacy 4.3.x with real `plankton.conversion2tons` lineage (kept). `data/examples_433_orig` (a frozen migration source) is left untouched. The `output.tl.enabled` → mean-TL mapping was verified semantically (it gates Java's `WeightedSpeciesOutput(getTrophicLevel, getWeight)`), not just by jar membership.

Reframes PR #122's migration guide, whose two documented "traps" this fix closes, to present them as worked examples now fixed — the class-level lesson and the still-live spatial-inputs / restart traps remain.

**Scope:** this fixes the specific key-granularity mismatches, the dead keys, and the false comments. The *systemic* known-but-unread warning (#121's "real fix") and the `conversion2tons` aliasing are deferred to **#123** — the guide's systemic-case reference is repointed there.

Spec: `docs/superpowers/specs/2026-07-18-issue-121-allowlisted-unread-keys-design.md`
Plan: `docs/superpowers/plans/2026-07-18-issue-121-allowlisted-unread-keys.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

---

## Self-Review

> **Revised 2026-07-18 after a 23-agent adversarial workflow (GO-WITH-FIXES, 9 survivors).** The workflow's headline: the plan's own verification gates failed the exact way #121 is about — a "prove nothing broke" command ran a nonexistent file (`test_config.py`) and skipped the file that goes red (`test_engine_config_validation.py`). All 5 blocking fixes applied: (1) real config-test filenames; (2) `data/eec/osm_all-parameters.csv` not `eec_all-parameters.csv`; (3) `data/minimal` added to the config-fix set + full-suite verification; (4) the "shim rescues half" arithmetic named in T4; (5) line-620 systemic ref stays "tracked", repointed at #123. Plus the non-empty-output test (Finding 4) and #123 filed.

**Spec coverage:** Layer A → Task 1; Layer B configs (examples + eec + **minimal**) → Task 2; Layer B allowlist + Layer C comments → Task 3; guide reframe (two traps + **the "shim rescues half" arithmetic** + systemic ref → #123) → Task 4; verify (**full suite**) + close #121 + PR → Task 5. Snapshot risk → Task 2 Step 6; parity → Task 2 Step 0; non-empty output → Task 2 `test_examples_actually_produces_meantl_output`; three-front re-verify → Task 3 Step 1. **No gaps.**

**Placeholder scan:** No TBD/TODO. Every command names files that EXIST (verified: the config test files, `data/eec/osm_all-parameters.csv`, `mean_trophic_level()` accessor). Every config edit gives the exact before/after line, read verbatim. The final gate is the full suite, not a subset — so a mistake surfaces instead of shipping green.

**Type consistency:** `EngineConfig` attrs used in tests (`output_meantl`, `economics_enabled`, `output_biomass_byage`, `output_abundance_byage`, `output_biomass_bysize`, `output_abundance_bysize`, `output_mean_size`) all match `config.py:919-929`. `_enabled(cfg, key)` and `validate(cfg, mode) -> list[UnknownKey]` (`.key`) match source. `_probe` / `minimal_cfg` are reused from the existing test file (defined there). The 5 dead keys and their working replacements are consistent across Tasks 1–3 and the config edits.
