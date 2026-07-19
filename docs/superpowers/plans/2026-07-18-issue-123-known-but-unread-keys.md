# Issue #123 — Systemic Known-but-Unread Config-Key Warning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a Python-engine run, emit one deduped warning naming any config keys that are valid OSMOSE keys the Python engine silently never reads (inert on this engine), so a ported Java config's silent no-ops become loud.

**Architecture:** Split the 149-entry `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py` into two frozensets — `_ALLOWLIST_PY_HONORED` (engine reads it, or reader-injected metadata) and `_ALLOWLIST_JAVA_ONLY` (real key, provably Python-unread) — whose union is byte-identical to today's set. A `java_only_keys_set(cfg)` matcher returns the java-only keys a config sets; a `warn_unread_java_only_keys(cfg)` wrapper emits one deduped summary. The call is placed in `PythonEngine._prepare_run` (the true Python-run seam), NOT in `from_dict` (which the Fisheries UI calls engine-agnostically to analyze Java runs). Correctness rests on two executable test guards — read-clearance (no java-only key is read by any engine module, incl. `config.py`) and metadata-clearance (all `osmose.*` ∈ PY_HONORED) — plus a frozen-snapshot partition test.

**Tech Stack:** Python 3.12/3.13, pytest, stdlib `ast`/`re`/`logging`. No new dependencies.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-07-18-issue-123-known-but-unread-keys-design.md`. Every task's requirements implicitly include these.

- **Warn-only, unconditional.** No error mode, not gated by `validation.strict.enabled`. One deduped summary line per run.
- **Union byte-identical:** `_SUPPLEMENTARY_ALLOWLIST = _ALLOWLIST_PY_HONORED | _ALLOWLIST_JAVA_ONLY` — the name `_SUPPLEMENTARY_ALLOWLIST` must survive so `build_known_keys()` and all existing unknown-key validation are unchanged.
- **Placement = `PythonEngine._prepare_run` (`osmose/engine/__init__.py`), NOT `from_dict`.** The analysis paths (Fisheries UI `ui/pages/fisheries.py:181,211`; `fmsy_sweep.py:321,404`) call `from_dict` directly and MUST NOT warn.
- **Classification by executable guard, never by allowlist comment.** Every `_ALLOWLIST_JAVA_ONLY` member is cleared Python-unread by the read-clearance guard; the guard scans every `.py` under `osmose/engine/**` **including `config.py`**, built by scanning the directory tree — NOT by extending `_EXTRA_ENGINE_SOURCES` (which omits `config.py`).
- **Curated PY_HONORED exclusions the read-clearance guard can't see (must be encoded by hand):** (a) membership/regex reads — `species.biomass.total.sp{idx}` (`background.py:329`) and the six `evolution.trait.{name}.*` patterns (`config.py:1571`, `genetics/trait.py:54`); (b) legacy aliases of read canonicals — `species.lw.condition.factor.sp{idx}` / `species.lw.allpower.sp{idx}` (canonical `species.length2weight.*` read at `config.py:467-468`) and `species.tl.sp{idx}` (canonical `species.trophic.level.sp` read at `background.py:196`). These are genuinely unread under their own name but the feature IS implemented on the Python engine, so "use the Java engine" would misdirect — no warning. `conversion2tons` is NOT excluded (canonical `resource.conversion2tons` is read nowhere → genuinely inert → stays JAVA_ONLY).
- **#120 carve-out = exactly `{"simulation.restart.file", "simulation.restart.enabled"}`** — subtracted from `java_only_keys_set`'s output to avoid double-warning with #120's targeted restart message. Re-verify #120's actual warn-set at rebase.
- **Dedup autouse-clear fixture is MANDATORY** — `_WARNED_JAVA_ONLY_KEYS` is a module-global; a `@pytest.fixture(autouse=True)` clears it before each test (as #120 required for `_WARNED_UNSUPPORTED_RESTART`).
- **Sequencing:** #120 (PR #125) SHOULD merge before #123, so the two restart carve-out keys keep a warning (#120's) rather than none. If #123 lands first, those two keys get no warning until #120 lands — a documented gap, not a defect.
- **Logger:** `config_validation.py` uses the module logger named `log` (`log = setup_logging("osmose.config")`). Use `log.warning`, not `_log`.

---

### Task 1: Partition the allowlist + the two correctness guards

The foundational, correctness-critical task: produce the two frozensets and the executable proof that the split is both complete (no key lost/added) and correct (no read/metadata key is java-only). The classification is DERIVED mechanically, then locked by the guards — never hand-typed from the block comments.

**Files:**
- Modify: `osmose/engine/config_validation.py:45-261` (replace the single `_SUPPLEMENTARY_ALLOWLIST` frozenset with two frozensets + their union)
- Test: `tests/test_issue_123_known_but_unread_keys.py` (new)

**Interfaces:**
- Produces: `_ALLOWLIST_PY_HONORED: frozenset[str]`, `_ALLOWLIST_JAVA_ONLY: frozenset[str]`, and `_SUPPLEMENTARY_ALLOWLIST: frozenset[str]` (= their union, same name as today). Task 2 consumes `_ALLOWLIST_JAVA_ONLY`.
- Consumes (existing, unchanged): `_extract_literal_keys_from_config_py(tree) -> set[str]`, `_compile_regex_for_pattern(pattern) -> re.Pattern` — both already in `config_validation.py`.

- [ ] **Step 1: Capture the pre-refactor 149-key snapshot**

Run this and keep the output — it is the independent reference the test will hardcode (must be captured BEFORE editing `config_validation.py`):

```bash
cd /home/razinka/osmopy && python -c "from osmose.engine.config_validation import _SUPPLEMENTARY_ALLOWLIST as A; import pprint; print(len(A)); pprint.pprint(sorted(A))" 2>/dev/null
```
Expected: first line `149`, then a sorted list of 149 keys. Save this list — you will paste it into the test as `FROZEN_ALLOWLIST_SNAPSHOT`.

- [ ] **Step 2: Write the test file with the snapshot + all guards (red — the two frozensets don't exist yet)**

Create `tests/test_issue_123_known_but_unread_keys.py`. Paste the 149 keys from Step 1 into `FROZEN_ALLOWLIST_SNAPSHOT`:

```python
"""Issue #123 — partition integrity + classification-correctness guards.

The read-clearance guard scans EVERY .py under osmose/engine/** (including config.py) —
a stricter test-only scan than production _EXTRA_ENGINE_SOURCES (which omits config.py).
It couples the JAVA_ONLY classification to the actual engine source, so a future edit that
either mis-buckets a key OR adds a cfg.get for a currently-java-only key turns the suite red.
"""
import ast
import pathlib

from osmose.engine.config_validation import (
    _ALLOWLIST_JAVA_ONLY,
    _ALLOWLIST_PY_HONORED,
    _SUPPLEMENTARY_ALLOWLIST,
    _compile_regex_for_pattern,
    _extract_literal_keys_from_config_py,
)

# Independent reference: the exact 149-key allowlist as of pre-#123 (copied from Step 1 output).
# NOT derived from the source frozensets — that would be circular.
FROZEN_ALLOWLIST_SNAPSHOT = frozenset([
    # <<< paste the 149 sorted keys from Step 1 here, one per line, e.g.:
    # "economic.output.stage",
    # "evolution.trait.{name}.envvar.sp{idx}",
    # ... all 149 ...
])

# Engine-honored dynamic prefixes (verified by grep: movement_maps.py:129, resources.py:143/97).
_STARTSWITH_PREFIXES = ("movement.species.map", "species.type.sp", "ltl.name.rsc")

# AST-INVISIBLE reads (membership / regex-on-iterated-key) that must be PY_HONORED. The read
# scan cannot see these; each carries its read site. Closed at these two families (see spec).
_MEMBERSHIP_EXCLUSIONS = frozenset([
    "species.biomass.total.sp{idx}",          # background.py:329  (total_key in config)
    "evolution.trait.{name}.target",          # config.py:1571 re.match / genetics/trait.py:54
    "evolution.trait.{name}.mean.sp{idx}",    # genetics/trait.py:59-60
    "evolution.trait.{name}.var.sp{idx}",
    "evolution.trait.{name}.envvar.sp{idx}",
    "evolution.trait.{name}.nlocus.sp{idx}",
    "evolution.trait.{name}.nval.sp{idx}",
])

# LEGACY ALIASES of keys the Python engine reads under their CANONICAL spelling. Genuinely unread
# under their own name (so the read-clearance guard can't rescue them), but the feature IS
# implemented on the Python engine — warning "use the Java engine" would misdirect. Must be
# PY_HONORED (no #123 warning). NOTE: conversion2tons is NOT here — its canonical
# resource.conversion2tons is read nowhere, so it is genuinely inert and correctly JAVA_ONLY.
_LEGACY_ALIAS_HONORED = frozenset([
    "species.lw.condition.factor.sp{idx}",    # canonical species.length2weight.condition.factor.sp -> config.py:467
    "species.lw.allpower.sp{idx}",            # canonical species.length2weight.allometric.power.sp -> config.py:468
    "species.tl.sp{idx}",                     # canonical species.trophic.level.sp -> background.py:196
])


def _scan_engine_reads() -> set[str]:
    """Every literal/f-string config key read across osmose/engine/** (incl. config.py).

    _extract_literal_keys_from_config_py returns a MIX: concrete literals from subscript/literal
    cfg.get ('fisheries.movement.file.map0') AND {idx}-pattern forms from f-strings
    (cfg.get(f'ltl.regrowth.rate.rsc{i}') -> 'ltl.regrowth.rate.rsc{idx}').
    """
    reads: set[str] = set()
    engine_root = pathlib.Path(__file__).resolve().parent.parent / "osmose" / "engine"
    for py in engine_root.rglob("*.py"):
        try:
            reads |= _extract_literal_keys_from_config_py(ast.parse(py.read_text(encoding="utf-8")))
        except SyntaxError:
            pass
    return reads


def _is_engine_read(pattern: str, reads: set[str]) -> bool:
    """True iff the Python engine reads any key matching `pattern`."""
    if pattern in reads:  # {idx}-form direct equality: catches f-string reads
        return True
    rx = _compile_regex_for_pattern(pattern)  # {idx}->\d+, {name}->\w+
    if any(rx.match(lit) for lit in reads if "{" not in lit):  # concrete literals: map0, etc.
        return True
    base = pattern.split("{")[0]
    return any(base.startswith(p) or p.startswith(base) for p in _STARTSWITH_PREFIXES)


def test_partition_completeness_against_frozen_snapshot():
    # union == independent snapshot (catches a dropped/added key during the split); disjoint.
    assert _ALLOWLIST_PY_HONORED | _ALLOWLIST_JAVA_ONLY == FROZEN_ALLOWLIST_SNAPSHOT
    assert _ALLOWLIST_PY_HONORED & _ALLOWLIST_JAVA_ONLY == frozenset()
    # source name preserved as the union (build_known_keys unchanged).
    assert _SUPPLEMENTARY_ALLOWLIST == FROZEN_ALLOWLIST_SNAPSHOT


def test_read_clearance_no_java_only_key_is_read():
    reads = _scan_engine_reads()
    offenders = [p for p in _ALLOWLIST_JAVA_ONLY if _is_engine_read(p, reads)]
    assert offenders == [], f"JAVA_ONLY keys the engine actually reads (reclassify PY_HONORED): {offenders}"


def test_membership_exclusion_families_are_py_honored():
    # AST-invisible membership/regex reads must be PY_HONORED (guard can't see them).
    assert _MEMBERSHIP_EXCLUSIONS <= _ALLOWLIST_PY_HONORED


def test_legacy_alias_keys_are_py_honored_not_warned():
    # species.lw.* / species.tl.* are legacy aliases of keys the engine reads under canonical
    # spellings — the Python engine implements the feature, so they must NOT be warned about
    # ("use the Java engine" would misdirect; spec §Out-of-scope requires species.lw.* silent).
    assert _LEGACY_ALIAS_HONORED <= _ALLOWLIST_PY_HONORED
    # conversion2tons is the OPPOSITE case (canonical unread) and stays JAVA_ONLY:
    assert "species.conversion2tons.sp{idx}" in _ALLOWLIST_JAVA_ONLY


def test_metadata_clearance_all_osmose_keys_py_honored():
    # Reader-injected metadata is UNREAD but must be PY_HONORED (else fires on every run).
    metadata = frozenset(k for k in FROZEN_ALLOWLIST_SNAPSHOT if k.startswith("osmose."))
    assert len(metadata) == 21
    assert metadata <= _ALLOWLIST_PY_HONORED
```

- [ ] **Step 3: Run the tests to confirm they fail (red)**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_issue_123_known_but_unread_keys.py -x -q`
Expected: collection/ImportError — `cannot import name '_ALLOWLIST_JAVA_ONLY'` (the frozensets don't exist yet).

- [ ] **Step 4: Derive the two buckets mechanically**

Run this self-contained derivation (a one-time tool — it duplicates the guard logic on purpose so it does not import the not-yet-written frozensets):

```bash
cd /home/razinka/osmopy && python - <<'PY' 2>/dev/null
import ast, pathlib, pprint
from osmose.engine.config_validation import (
    _SUPPLEMENTARY_ALLOWLIST as SNAP, _extract_literal_keys_from_config_py, _compile_regex_for_pattern,
)
reads = set()
for py in (pathlib.Path("osmose") / "engine").rglob("*.py"):
    try:
        reads |= _extract_literal_keys_from_config_py(ast.parse(py.read_text(encoding="utf-8")))
    except SyntaxError:
        pass
STARTSWITH = ("movement.species.map", "species.type.sp", "ltl.name.rsc")
EXCL = {
    # membership/regex reads (AST-invisible):
    "species.biomass.total.sp{idx}",
    "evolution.trait.{name}.target", "evolution.trait.{name}.mean.sp{idx}",
    "evolution.trait.{name}.var.sp{idx}", "evolution.trait.{name}.envvar.sp{idx}",
    "evolution.trait.{name}.nlocus.sp{idx}", "evolution.trait.{name}.nval.sp{idx}",
    # legacy aliases of read canonicals -> feature works on Python, don't warn (conversion2tons
    # is NOT here: its canonical is unread, so it stays JAVA_ONLY):
    "species.lw.condition.factor.sp{idx}", "species.lw.allpower.sp{idx}", "species.tl.sp{idx}",
}
def is_read(p):
    if p in reads: return True
    rx = _compile_regex_for_pattern(p)
    if any(rx.match(l) for l in reads if "{" not in l): return True
    b = p.split("{")[0]
    return any(b.startswith(s) or s.startswith(b) for s in STARTSWITH)
meta = {k for k in SNAP if k.startswith("osmose.")}
py_honored = {k for k in SNAP if is_read(k)} | meta | EXCL
java_only = set(SNAP) - py_honored
print("### PY_HONORED (%d) ###" % len(py_honored)); pprint.pprint(sorted(py_honored))
print("### JAVA_ONLY (%d) ###" % len(java_only)); pprint.pprint(sorted(java_only))
PY
```
Expected: two sorted lists that together total 149. **Eyeball the JAVA_ONLY list** — every entry should read plausibly as a Java-side-only key (output flags config.py doesn't read, forcing filenames, `simulation.ncpu`, `grid.java.classname`, restart params, `mortality.fishing.recruitment.*`, `conversion2tons`, etc.). If any entry looks like something the Python engine should honor, STOP and re-verify by grep before proceeding — but the guard in Step 6 is the backstop.

- [ ] **Step 5: Replace the single allowlist with the two frozensets**

In `osmose/engine/config_validation.py`, replace the entire `_SUPPLEMENTARY_ALLOWLIST: frozenset[str] = frozenset([ ... ])` block (lines ~45-261) with the two derived frozensets and their union. Paste the exact sorted lists from Step 4. Keep the explanatory header comment; drop the now-false "Verified: zero hits …" sub-comment.

```python
# Reader-honored keys the AST walker cannot resolve statically, split into two buckets (#123):
#   _ALLOWLIST_PY_HONORED  — the Python engine reads it, OR reader-injected osmose.* metadata.
#   _ALLOWLIST_JAVA_ONLY   — a real OSMOSE/Java key the Python engine provably does NOT read;
#                            setting it on a Python run has no effect (warned about, see #123).
# The membership of each bucket is proven by tests/test_issue_123_known_but_unread_keys.py
# (read-clearance + metadata-clearance guards) — do NOT edit a key's bucket by eyeballing a
# comment; move it, then let the guard confirm. Their union is byte-identical to the pre-#123
# allowlist, so build_known_keys() and all unknown-key validation are unchanged.
_ALLOWLIST_PY_HONORED: frozenset[str] = frozenset([
    # <<< paste the sorted PY_HONORED list from Step 4 >>>
])

_ALLOWLIST_JAVA_ONLY: frozenset[str] = frozenset([
    # <<< paste the sorted JAVA_ONLY list from Step 4 >>>
])

_SUPPLEMENTARY_ALLOWLIST: frozenset[str] = _ALLOWLIST_PY_HONORED | _ALLOWLIST_JAVA_ONLY
```

- [ ] **Step 6: Run the guard tests to confirm they pass (green)**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_issue_123_known_but_unread_keys.py -v`
Expected: all 4 tests PASS (`test_partition_completeness_against_frozen_snapshot`, `test_read_clearance_no_java_only_key_is_read`, `test_membership_exclusion_families_are_py_honored`, `test_metadata_clearance_all_osmose_keys_py_honored`).

- [ ] **Step 7: Confirm existing validation is unchanged**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_engine_config_validation.py -q`
Expected: all PASS (the `_SUPPLEMENTARY_ALLOWLIST` union is byte-identical, so unknown-key validation behaves identically). If any test imported `_SUPPLEMENTARY_ALLOWLIST` directly, it still resolves (same name).

- [ ] **Step 8: Commit**

```bash
cd /home/razinka/osmopy && git add osmose/engine/config_validation.py tests/test_issue_123_known_but_unread_keys.py
git commit -m "feat(#123): partition allowlist into PY_HONORED/JAVA_ONLY + correctness guards

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `java_only_keys_set` matcher + #120 carve-out

The function that, given a config, returns the java-only keys it actually sets — canonicalizing first (mirrors `validate()`), matching literals + `{idx}`/`{name}` patterns, minus the two #120-owned restart keys.

**Files:**
- Modify: `osmose/engine/config_validation.py` (add `_RESTART_HANDLED_BY_120` + `java_only_keys_set`, after the `_ALLOWLIST_JAVA_ONLY` definition; `canonicalize_config` is imported inside the function as `validate()` does at `config_validation.py:524`)
- Test: `tests/test_issue_123_known_but_unread_keys.py` (extend)

**Interfaces:**
- Consumes: `_ALLOWLIST_JAVA_ONLY` (Task 1), `_compile_regex_for_pattern` (existing), `canonicalize_config(cfg) -> tuple[dict, list]` (from `osmose.config.aliases`).
- Produces: `java_only_keys_set(cfg: dict) -> list[str]` — sorted list of java-only keys the (canonicalized) config sets, excluding the #120 restart keys. `_RESTART_HANDLED_BY_120: frozenset[str]`. Task 3 consumes `java_only_keys_set`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_issue_123_known_but_unread_keys.py`:

```python
from osmose.engine.config_validation import java_only_keys_set, _RESTART_HANDLED_BY_120


def test_java_only_keys_set_matches_literal_and_pattern():
    cfg = {
        "simulation.ncpu": "8",                       # java-only literal
        "output.diet.stage.threshold.sp3": "12",      # java-only {idx} pattern
        "simulation.time.nyear": "15",                # a real read key (not allowlisted) -> ignored
    }
    assert java_only_keys_set(cfg) == ["output.diet.stage.threshold.sp3", "simulation.ncpu"]


def test_java_only_keys_set_excludes_py_honored_and_metadata():
    cfg = {
        "output.tl.enabled": "true",                  # PY_HONORED (config.py:925) — round-4 landmine
        "movement.species.map0": "map.csv",           # PY_HONORED via startswith
        "evolution.trait.imax.target": "1.0",         # PY_HONORED exclusion family
        "ltl.depletable.enabled": "true",             # PY_HONORED (resources.py:74)
        "osmose.version": "4.4.1",                     # reader-injected metadata — must never surface
        "osmose.configuration.background": "x.csv",   # metadata
    }
    assert java_only_keys_set(cfg) == []


def test_java_only_keys_set_excludes_120_restart_carveouts():
    cfg = {"simulation.restart.file": "snap.nc", "simulation.restart.enabled": "true"}
    assert java_only_keys_set(cfg) == []
    assert _RESTART_HANDLED_BY_120 == frozenset({"simulation.restart.file", "simulation.restart.enabled"})


def test_java_only_keys_set_canonicalizes_before_matching():
    # DISCRIMINATING (not vacuous): output.fishery.enabled canonicalizes (RENAMES_440) to the
    # java-only output.fisheries.enabled. The legacy source is not itself allowlisted, so WITHOUT
    # canonicalization this returns [] — the assertion proves canonicalize_config actually ran.
    assert java_only_keys_set({"output.fishery.enabled": "true"}) == ["output.fisheries.enabled"]


def test_java_only_keys_set_empty_when_none():
    assert java_only_keys_set({"simulation.time.nyear": "15"}) == []
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_issue_123_known_but_unread_keys.py -k java_only_keys_set -q`
Expected: FAIL — `cannot import name 'java_only_keys_set'`.

- [ ] **Step 3: Implement `java_only_keys_set` + carve-out**

In `osmose/engine/config_validation.py`, immediately after the `_SUPPLEMENTARY_ALLOWLIST` union line from Task 1:

```python
# #120 already warns on these two restart keys with a targeted message (config.py) — exclude them
# from #123's summary to avoid double-warning. They remain in _ALLOWLIST_JAVA_ONLY for partition
# completeness (they ARE java-only). Re-verify #120's warn-set at rebase (spec §"#120 overlap").
_RESTART_HANDLED_BY_120: frozenset[str] = frozenset(
    {"simulation.restart.file", "simulation.restart.enabled"}
)


def java_only_keys_set(cfg: dict) -> list[str]:
    """Real OSMOSE keys present in `cfg` that the Python engine does not read (inert on a Python
    run). Canonicalizes first (mirrors validate()); matches _ALLOWLIST_JAVA_ONLY literals + {idx}/
    {name} patterns; excludes the #120-owned restart keys. Returns a sorted list."""
    from osmose.config.aliases import canonicalize_config

    cfg, _ = canonicalize_config(cfg)
    java_only = _ALLOWLIST_JAVA_ONLY - _RESTART_HANDLED_BY_120
    literals = frozenset(p for p in java_only if "{idx}" not in p and "{name}" not in p)
    regexes = tuple(
        _compile_regex_for_pattern(p) for p in java_only if "{idx}" in p or "{name}" in p
    )
    hits = [k for k in cfg if k in literals or any(rx.match(k) for rx in regexes)]
    return sorted(hits)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_issue_123_known_but_unread_keys.py -k java_only_keys_set -v`
Expected: all 5 `java_only_keys_set` tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/osmopy && git add osmose/engine/config_validation.py tests/test_issue_123_known_but_unread_keys.py
git commit -m "feat(#123): java_only_keys_set matcher + #120 restart carve-out

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Wire the deduped warning into `_prepare_run` (with the placement guard)

Add the deduped warn wrapper in `config_validation.py`, call it from `PythonEngine._prepare_run` (the Python-run seam), and prove by test that it fires on a run, dedups once per process, and does NOT fire when a config is merely constructed via `from_dict` (the round-1 false-positive guard).

**Files:**
- Modify: `osmose/engine/config_validation.py` (add `_WARNED_JAVA_ONLY_KEYS` + `warn_unread_java_only_keys`)
- Modify: `osmose/engine/__init__.py:74-77` (call it in `_prepare_run`)
- Test: `tests/test_issue_123_known_but_unread_keys.py` (extend, + autouse reset fixture)

**Interfaces:**
- Consumes: `java_only_keys_set` (Task 2).
- Produces: `warn_unread_java_only_keys(cfg: dict) -> list[str]` (emits one deduped `log.warning` if non-empty, returns the java-only keys). Module-global `_WARNED_JAVA_ONLY_KEYS: set[str]`.

- [ ] **Step 1: Write the failing tests + mandatory reset fixture**

Append to `tests/test_issue_123_known_but_unread_keys.py`:

```python
import logging
import pytest

from osmose.engine import config_validation as cv


@pytest.fixture(autouse=True)
def _clear_java_only_warn_cache():
    # MANDATORY: _WARNED_JAVA_ONLY_KEYS persists across tests in a process; clear it so the
    # "exactly one warning" and "deduped once" tests are not order-dependent (as #120 needed).
    cv._WARNED_JAVA_ONLY_KEYS.clear()
    yield
    cv._WARNED_JAVA_ONLY_KEYS.clear()


def _load_minimal_config() -> dict:
    from osmose.config.reader import OsmoseConfigReader
    return dict(OsmoseConfigReader().read("data/minimal/osm_all-parameters.csv"))


def test_warn_emits_one_summary_naming_the_keys(caplog):
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        keys = cv.warn_unread_java_only_keys({"simulation.ncpu": "8", "grid.java.classname": "X"})
    assert keys == ["grid.java.classname", "simulation.ncpu"]
    msgs = [r.getMessage() for r in caplog.records if "issue #123" in r.getMessage()]
    assert len(msgs) == 1
    assert "grid.java.classname" in msgs[0] and "simulation.ncpu" in msgs[0]


def test_warn_dedups_once_per_process(caplog):
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        cv.warn_unread_java_only_keys({"simulation.ncpu": "8"})
        cv.warn_unread_java_only_keys({"simulation.ncpu": "8"})
    assert sum("issue #123" in r.getMessage() for r in caplog.records) == 1


def test_warn_truncates_long_lists_but_counts_all(caplog):
    # Bundled demos set ~20-44 java-only keys; the line names at most _MAX_NAMED_JAVA_ONLY_KEYS
    # and counts the rest, while the returned list and the reported count stay complete.
    cfg = {f"output.diet.stage.threshold.sp{i}": "12" for i in range(15)}  # 15 java-only keys
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        keys = cv.warn_unread_java_only_keys(cfg)
    assert len(keys) == 15
    msg = next(r.getMessage() for r in caplog.records if "issue #123" in r.getMessage())
    assert "15 config key(s)" in msg                              # full count reported
    assert "and 5 more" in msg                                   # 15 - 10 named
    assert msg.count("output.diet.stage.threshold.sp") == 10      # only 10 named inline


def test_warn_silent_when_no_java_only_keys(caplog):
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        cv.warn_unread_java_only_keys({"output.tl.enabled": "true", "simulation.time.nyear": "15"})
    assert not any("issue #123" in r.getMessage() for r in caplog.records)


def test_from_dict_does_not_warn_only_prepare_run_does(caplog):
    # Round-1 placement guard: constructing a config via from_dict (the Fisheries-UI / fmsy-probe
    # pattern) must NOT emit the #123 warning — only an actual run seam does.
    from osmose.engine.config import EngineConfig

    cfg = dict(_load_minimal_config())
    cfg["simulation.ncpu"] = "8"
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        EngineConfig.from_dict(cfg)
    assert not any("issue #123" in r.getMessage() for r in caplog.records)


def test_prepare_run_emits_the_warning(caplog):
    # The run seam DOES warn. _prepare_run builds engine_config/grid/rngs (no simulation loop),
    # so it is cheap to call directly.
    from osmose.engine import PythonEngine

    cfg = dict(_load_minimal_config())
    cfg["simulation.ncpu"] = "8"
    with caplog.at_level(logging.WARNING, logger="osmose.config"):
        PythonEngine()._prepare_run(cfg, seed=0)
    assert any("issue #123" in r.getMessage() and "simulation.ncpu" in r.getMessage()
               for r in caplog.records)
```

The `_load_minimal_config()` helper (shown above the tests) matches the established loader API
(`OsmoseConfigReader().read(path)` — no-arg constructor, path to `read()`; see
`tests/test_baltic_fine_config.py:8`). `data/minimal/osm_all-parameters.csv` is the bundled entry
config.

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_issue_123_known_but_unread_keys.py -k "warn or prepare_run or from_dict" -q`
Expected: FAIL — `module 'osmose.engine.config_validation' has no attribute 'warn_unread_java_only_keys'` / `_WARNED_JAVA_ONLY_KEYS`.

- [ ] **Step 3: Implement the deduped warn wrapper**

In `osmose/engine/config_validation.py`, immediately after `java_only_keys_set` (Task 2):

```python
# Deduped-once-per-process, like #120's _WARNED_UNSUPPORTED_RESTART. Cleared by an autouse test
# fixture (tests/test_issue_123_known_but_unread_keys.py). Placed at the Python-run seam
# (PythonEngine._prepare_run), NOT in from_dict — see spec §3 / Global Constraints.
_WARNED_JAVA_ONLY_KEYS: set[str] = set()
_MAX_NAMED_JAVA_ONLY_KEYS = 10  # cap the listed keys; the rest are counted (bundled demos set ~20-44)


def warn_unread_java_only_keys(cfg: dict) -> list[str]:
    """If `cfg` sets java-only keys, emit ONE deduped summary warning naming (up to
    _MAX_NAMED_JAVA_ONLY_KEYS of) them. Returns the full key list (empty if none). Call only from
    the Python-engine run seam."""
    keys = java_only_keys_set(cfg)
    if not keys:
        return keys
    # Dedup on the FULL key set so two configs differing only in the un-listed tail warn distinctly.
    fingerprint = ",".join(keys)
    if fingerprint not in _WARNED_JAVA_ONLY_KEYS:
        _WARNED_JAVA_ONLY_KEYS.add(fingerprint)
        shown = keys[:_MAX_NAMED_JAVA_ONLY_KEYS]
        more = "" if len(keys) <= _MAX_NAMED_JAVA_ONLY_KEYS else f", and {len(keys) - len(shown)} more"
        log.warning(
            "%d config key(s) are valid OSMOSE keys the Python engine does not implement; on this "
            "engine they have no effect. Use the Java engine if you need them: %s%s (see issue #123).",
            len(keys),
            ", ".join(shown),
            more,
        )
    return keys
```

- [ ] **Step 4: Call it from `_prepare_run`**

In `osmose/engine/__init__.py`, in `_prepare_run` (currently lines ~70-77), add the call right after the existing imports and before `engine_config = EngineConfig.from_dict(config)`:

```python
    def _prepare_run(self, config: dict[str, str], seed: int) -> tuple:
        """Build the (engine_config, grid, rng, movement_rngs, mortality_rngs)
        tuple shared between run() and run_in_memory().
        """
        from osmose.engine.config import EngineConfig
        from osmose.engine.config_validation import warn_unread_java_only_keys
        from osmose.engine.rng import build_rng

        warn_unread_java_only_keys(config)  # #123: warn on Java-only keys inert on the Python engine
        engine_config = EngineConfig.from_dict(config)
        grid = self._resolve_grid(config)
```

- [ ] **Step 5: Run the Task-3 tests to confirm they pass**

Run: `cd /home/razinka/osmopy && python -m pytest tests/test_issue_123_known_but_unread_keys.py -v`
Expected: ALL tests in the file PASS (Task 1 + 2 + 3, including the placement guard and the `_prepare_run` emission test).

- [ ] **Step 6: Whole-suite guard — audit for demo clean-load/run assertions the summary now breaks**

Run: `cd /home/razinka/osmopy && python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -30`
Expected: the pre-existing CI-skip flake `test_trophic_cascade_visible` may fail (it fails on base too — unrelated to #123). NO OTHER failures. If a test fails because it asserts a warning-free / clean load or run of `examples`/`eec`/`benguela`/`baltic` (or runs under `-W error`) and now sees the #123 summary, update THAT test to expect the summary (do not silence globally, do not add filters). If you change any such test, note it in the commit. If the only failure is `test_trophic_cascade_visible`, proceed.

- [ ] **Step 7: Lint**

Run: `cd /home/razinka/osmopy && ruff check osmose/ tests/ && ruff format --check osmose/ tests/`
Expected: clean. If `ruff format --check` reports diffs, run `ruff format osmose/ tests/` and re-check.

- [ ] **Step 8: Commit**

```bash
cd /home/razinka/osmopy && git add osmose/engine/config_validation.py osmose/engine/__init__.py tests/test_issue_123_known_but_unread_keys.py
git commit -m "feat(#123): warn on Java-only keys at the Python-engine run seam

Emits one deduped summary from PythonEngine._prepare_run when a config sets keys
the Python engine does not read. Placement guard test pins that from_dict (the
engine-agnostic analysis path) does NOT warn. Closes #123.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **#120/#125 sequencing (Global Constraints):** if #125 has NOT merged when you start, the two `_RESTART_HANDLED_BY_120` keys will get no warning from either feature until it does. That is the documented gap, acceptable per the spec; do not "fix" it by dropping the carve-out. At rebase onto a #120-merged base, re-verify #120 still warns on exactly `{simulation.restart.file, simulation.restart.enabled}` (`config.py` ~2040/2047) — if it grew, grow the carve-out to match.
- **Do not run any Baltic / `nbackground>0` config on the Java engine.** All tests here use the Python engine (`_prepare_run`, `from_dict`) on the bundled minimal config — no Java subprocess involved.
- **The guard is the authority, not the plan's prose.** If Step 4's derivation surfaces a JAVA_ONLY key you believe is actually read, trust a fresh grep of `osmose/engine/**` over any comment; the read-clearance test will refuse a genuinely-read key regardless.
