# Fix #120 — warn on ignored restart — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the Python engine silently ignoring restart. When a config requests restart, warn loudly (both `simulation.restart.file` = resume and `simulation.restart.enabled` = write, distinct messages), following the `_warn_unsupported_mortality_features` precedent. Does NOT implement restart. Reframe the migration guide's restart trap from silent → loud.

**Architecture:** One warn-check in `EngineConfig.from_dict` after `canonicalize_config` (Python-scoped — Java never calls `from_dict`), deduped via a module-global set cleared by autouse test fixtures. Warn-only — no raise (the strict-raise was dropped in design review: the fisheries UI would swallow it into a silent degrade and it false-positives valid Java-restart configs).

**Tech Stack:** Python 3.12, pytest, ruff, Sphinx 9.1 (`.venv/bin/sphinx-build`).

**Spec:** `docs/superpowers/specs/2026-07-18-issue-120-restart-warning-design.md`

## Global Constraints

- **METHOD RULE:** verify every claim by running the code, not by inference (this whole issue is a silent failure; the design review found the fix originally checked the WRONG key by inference).
- **Two keys, distinct messages:** `simulation.restart.file` set (non-empty, not "null"/"none") = resume request → "will cold-start, not resume from snapshot". `simulation.restart.enabled=true` = write request → "restart output will not be written". `.file` is the more harmful (wrong starting state).
- **Warn-only. No raise anywhere.** `validate()` and the allowlist are UNTOUCHED (restart keys stay known/valid Java-side keys).
- **`_log.warning` channel** (matches the mortality precedent; the dual-channel #120 tripwire catches it). Deduped per-message via `_WARNED_UNSUPPORTED_RESTART`.
- **Every test asserting the warning MUST clear `_WARNED_UNSUPPORTED_RESTART` first** (autouse fixture) — the set is module-global and a bundled config (`data/benguela`) fires the warning in a full run, poisoning it for later files.
- **Do NOT flip `data/benguela`'s restart** — faithful upstream port; the warning correctly surfaces it.
- Branch `fix/issue-120-restart-warning`. Commit after each task. `.venv/bin/python`; cwd resets — absolute paths.

## Prerequisite for Task 2 only (the guide reframe) — the #124 / #121 sequencing dependency

This branch is on master WITHOUT PR #124 (the #121 fix, open). #124 also reframes the guide's restart passages (they overlap on one sentence — the "shim rescues half" arithmetic). **Task 1 (engine + tests) is fully independent of #124 and proceeds now.** **Task 2 (guide reframe) requires #124 merged and this branch rebased on the updated master** — its line numbers below are pre-#121 and WILL shift; re-grep on the rebased base. If #124 is not merged when execution reaches Task 2, STOP and surface it (options: wait for #124, or ship engine+tests now and do the guide reframe as a follow-up once #124 lands).

## File Structure

| File | Change | Task |
|---|---|---|
| `osmose/engine/config.py` | `_WARNED_UNSUPPORTED_RESTART` + `_warn_once_restart` + the two-key check in `from_dict` | 1 |
| `tests/test_issue_120_restart_warning.py` (create) | behavior tests + autouse dedup-clear | 1 |
| `tests/test_r_dialect_migration_claims.py` | flip the #120 tripwire (engine now warns) + autouse dedup-clear | 1 |
| `docs/r-to-python-migration.md` | reframe restart silent → loud (post-#121 base) | 2 |

---

### Task 1: Engine warn (both keys) + behavior tests + flip the tripwire

**Files:**
- Modify: `osmose/engine/config.py` (add dedup set + helper near :37; add check in `from_dict` after :2022)
- Create: `tests/test_issue_120_restart_warning.py`
- Modify: `tests/test_r_dialect_migration_claims.py` (flip the tripwire, add autouse clear)

**Interfaces:**
- Consumes: `_enabled(cfg, key) -> bool` (config.py:167), `_log` (config.py:32), `EngineConfig.from_dict(cfg)`.
- Produces: `_WARNED_UNSUPPORTED_RESTART: set[str]` and `_warn_once_restart(msg)` (module-level in config.py), used by test fixtures.

- [ ] **Step 1: Write the behavior tests (RED)**

Create `tests/test_issue_120_restart_warning.py`:

```python
"""#120: the Python engine warns (does not silently ignore) when a config requests restart."""

import logging

import pytest

from osmose.engine.config import _WARNED_UNSUPPORTED_RESTART
from osmose.engine.config import EngineConfig


@pytest.fixture(autouse=True)
def _clear_restart_warning_cache():
    """Dedup set is process-global; clear before each test so assertions aren't suppressed."""
    _WARNED_UNSUPPORTED_RESTART.clear()
    yield


def _base_cfg() -> dict[str, str]:
    # A fresh raw dict: NO osmose.version, NO native simulation.restart.* — so the old-spelling
    # test (below) actually renames + fires. (data/minimal can't be reused: it stamps
    # osmose.version=4.4.1 AND sets a native simulation.restart.enabled=false, either of which
    # suppresses the old-spelling warning — verified in the design spec.)
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "1",
    }


def _restart_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and "restart" in r.getMessage().lower()]


def test_restart_file_warns_resume(caplog):
    cfg = _base_cfg() | {"simulation.restart.file": "snap.nc"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    msgs = _restart_warnings(caplog)
    assert any("cold-start" in m.lower() and "snap.nc" in m for m in msgs), msgs


def test_restart_enabled_warns_write(caplog):
    cfg = _base_cfg() | {"simulation.restart.enabled": "true"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    msgs = _restart_warnings(caplog)
    assert any("output" in m.lower() for m in msgs), msgs


def test_both_keys_warn_distinctly(caplog):
    cfg = _base_cfg() | {"simulation.restart.file": "snap.nc", "simulation.restart.enabled": "true"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    msgs = _restart_warnings(caplog)
    assert len(set(msgs)) == 2, msgs  # two distinct messages


def test_no_restart_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(_base_cfg())
    assert _restart_warnings(caplog) == []


def test_restart_enabled_false_and_null_file_no_warning(caplog):
    cfg = _base_cfg() | {"simulation.restart.enabled": "false", "simulation.restart.file": "null"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    assert _restart_warnings(caplog) == []


def test_old_spelling_output_restart_enabled_warns(caplog):
    # output.restart.enabled -> (canonicalize) -> simulation.restart.enabled. The post-canonicalize
    # check catches it. Uses the fresh dict (no version, no native key) so the rename fires.
    cfg = _base_cfg() | {"output.restart.enabled": "true"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
    assert any("output" in m.lower() for m in _restart_warnings(caplog))


def test_dedup_warns_once(caplog):
    cfg = _base_cfg() | {"simulation.restart.file": "snap.nc"}
    with caplog.at_level(logging.WARNING):
        EngineConfig.from_dict(cfg)
        EngineConfig.from_dict(cfg)
    # dedup NOT cleared between the two calls (autouse clears only before the test) -> once
    assert len(_restart_warnings(caplog)) == 1
```

- [ ] **Step 2: Run — expect FAIL (import error / no warnings)**

Run: `.venv/bin/python -m pytest tests/test_issue_120_restart_warning.py -q`
Expected: FAIL — `_WARNED_UNSUPPORTED_RESTART` doesn't exist yet (ImportError), and no warnings are emitted. That failure IS the #120 bug.

- [ ] **Step 3: Implement the engine change**

In `osmose/engine/config.py`, near the mortality dedup set (line ~37, after `_WARNED_UNSUPPORTED_MORTALITY: set[str] = set()`), add:

```python
# #120: the Python engine does not implement restart. Warn (don't silently ignore) when a
# config requests it. Deduped per-message for the process lifetime (from_dict reruns per
# calibration candidate). Cleared by autouse fixtures in the restart tests.
_WARNED_UNSUPPORTED_RESTART: set[str] = set()


def _warn_once_restart(msg: str) -> None:
    if msg not in _WARNED_UNSUPPORTED_RESTART:
        _WARNED_UNSUPPORTED_RESTART.add(msg)
        _log.warning("%s", msg)
```

(If `_warn_once_restart` must be defined after `_log` — `_log` is at :32, so this placement at ~:40 is fine.)

In `EngineConfig.from_dict`, immediately after `cfg, _deprecated = canonicalize_config(cfg)` (config.py:2022), insert:

```python
        # #120: warn on requested-but-unsupported restart (post-canonicalize catches the old
        # output.restart.enabled spelling too). Two distinct requests, two distinct messages.
        _restart_file = cfg.get("simulation.restart.file", "").strip()
        if _restart_file and _restart_file.lower() not in ("null", "none"):
            _warn_once_restart(
                f"simulation.restart.file={_restart_file!r} is set, but the Python engine does "
                "not implement restart — the run will COLD-START from scratch instead of "
                "resuming from that snapshot. Use the Java engine to resume (see issue #120)."
            )
        if _enabled(cfg, "simulation.restart.enabled"):
            _warn_once_restart(
                "simulation.restart.enabled is set, but the Python engine does not write "
                "restart/checkpoint output — none will be produced. Use the Java engine for "
                "restart output (see issue #120)."
            )
```

- [ ] **Step 4: Run — expect PASS (7 passed)**

Run: `.venv/bin/python -m pytest tests/test_issue_120_restart_warning.py -v`
Expected: **7 passed.** If `test_old_spelling...` fails, the fresh-dict base wasn't used (a version-stamped or native-restart-carrying dict suppresses the rename — see the docstring).

- [ ] **Step 5: Flip the #120 tripwire in `tests/test_r_dialect_migration_claims.py`**

That file's `test_engine_does_not_yet_warn_on_ignored_restart` (line ~141) currently asserts the engine does NOT warn — the engine change flips it red. Run it to confirm:
`.venv/bin/python -m pytest "tests/test_r_dialect_migration_claims.py::test_engine_does_not_yet_warn_on_ignored_restart" -q`
Expected: **FAIL** (a restart warning now appears) — the designed tripwire firing.

Replace that test with:
```python
def test_engine_warns_on_ignored_restart_after_120(caplog):
    """FORMERLY a tripwire asserting the engine was SILENT on restart; FIXED in #120.

    The Python engine now WARNS (does not implement restart, but no longer silently ignores it).
    Dual-channel capture retained. Restart is still Java-only — this asserts the SILENCE is fixed,
    not that restart is implemented.
    """
    cfg = OsmoseConfigReader().read(MINIMAL_CONFIG)
    cfg["simulation.restart.enabled"] = "true"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING):
            EngineConfig.from_dict(cfg)

    log_hits = [r.getMessage() for r in caplog.records if "restart" in r.getMessage().lower()]
    warn_hits = [str(w.message) for w in caught if "restart" in str(w.message).lower()]
    assert log_hits + warn_hits, "engine no longer warns on ignored restart — #120 regressed"
```

Add an autouse dedup-clear fixture to this file too (a bundled config firing the warning earlier in a full run would otherwise suppress this assertion). Near the top, after imports:
```python
@pytest.fixture(autouse=True)
def _clear_restart_warning_cache():
    from osmose.engine.config import _WARNED_UNSUPPORTED_RESTART
    _WARNED_UNSUPPORTED_RESTART.clear()
    yield
```
Leave `test_strict_mode_is_SILENT_on_unimplemented_restart` (line ~117) UNCHANGED — its assertion is about `validate()` (untouched) and stays true.

- [ ] **Step 6: Run the guard file + confirm no other break**

Run: `.venv/bin/python -m pytest tests/test_r_dialect_migration_claims.py tests/test_issue_120_restart_warning.py -q`
Expected: all pass.
Run: `.venv/bin/python -m pytest tests/test_engine_config.py tests/test_config_validation.py tests/test_engine_config_validation.py tests/test_benguela_bundle.py -q`
Expected: pass. (`test_benguela_bundle` will now emit the restart warning — that's expected; it must not FAIL. `test_config_validation.py`'s `caplog.records == []` assertions use a different logger (`osmose.config`) and cfgs without restart keys, so they're unaffected — confirm.)

- [ ] **Step 7: Lint**

Run: `.venv/bin/python -m ruff check osmose/ tests/ && .venv/bin/python -m ruff format osmose/engine/config.py tests/test_issue_120_restart_warning.py tests/test_r_dialect_migration_claims.py`
Expected: clean.

- [ ] **Step 8: Prove non-tautology (mutation)**

Temporarily comment out the `_warn_once_restart(...)` call for `.file` in config.py; run
`test_restart_file_warns_resume` → expect FAIL; restore; confirm `git diff osmose/` clean and the test passes again.

- [ ] **Step 9: Commit**

```bash
git add osmose/engine/config.py tests/test_issue_120_restart_warning.py tests/test_r_dialect_migration_claims.py
git commit -m "fix(#120): warn when the Python engine ignores a restart request"
```

---

### Task 2: Reframe the migration guide — restart is now loud, not silent

**BLOCKED on the #124 prerequisite** (see top). Do this only after #124 is merged and this branch is rebased on the updated master. **Re-grep for line numbers — the pre-#121 numbers below WILL have shifted.**

**Files:**
- Modify: `docs/r-to-python-migration.md`

- [ ] **Step 1: Confirm the base is post-#121**

```bash
git log --oneline -5 | grep -i "121\|shim rescues"   # confirm PR #124's guide changes are present
grep -n "five of eight arrive" docs/r-to-python-migration.md   # post-#121 arithmetic present?
```
If the post-#121 guide is NOT present, STOP — the base is wrong (rebase on post-#124 master first).

- [ ] **Step 2: Enumerate every restart passage**

```bash
grep -niE "restart|#120" docs/r-to-python-migration.md
```
Reframe each (skip the §5 calibrar `control$restart.file`/`REPORT` passages — those are optimizer resume-on-crash, NOT `simulation.restart.*`; they never mention `simulation.restart.*`):

- **Taxonomy row** for restart: Signal `silent` → `loud (engine warns on load)`. AND fix the description — it says `simulation.restart.enabled (never consumes a restart file)`, but "never consumes a restart file" is the `.file`/resume behavior wrongly pinned on `.enabled` (the write toggle). Reference `simulation.restart.file` (or both keys).
- **"Shim rescues half and strands half"** subsection: it uses `output.restart.enabled → simulation.restart.enabled` as a "engine never reads / fully silent" example, and its count of "fully silent" dead keys includes it. After #120, `restart.enabled` WARNS — so drop it from the "fully silent" set and reduce that count by one, and mark the example "now warns (#120)". (Verify the exact post-#121 wording and count by reading it.)
- **§1 losses** ("nothing consumes a restart file") → still no restart, but now WARNS; Java engine to restart.
- **Honest-gaps row** ("loads and validates clean, does nothing") → "loads clean but now WARNS; still no restart — Java engine"; update its `#120` link.
- **All `#120` references** → past-tense "the silence is fixed", never "restart is implemented". #120 is already CLOSED — do not imply this PR closes it.
- **"strict mode is silent on restart"** stays TRUE (validate() untouched) — optional one-clause note that the ENGINE now warns though the validator stays silent.
- **Leave still-live traps** (spatial `.nc`, missing sub-config, cross-file precedence).

- [ ] **Step 3: Clean build**

```bash
rm -rf docs/_build/html && .venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```
Expected: exit 0.

- [ ] **Step 4: Confirm no now-false claim remains**

```bash
grep -niE "restart.*silent|silently.*ignore.*restart|fully silent" docs/r-to-python-migration.md
```
Each hit must be either accurate (restart is loud now / a still-live non-restart trap) or the §5 calibrar passages. No restart passage may still call restart *silent*.

- [ ] **Step 5: Commit**

```bash
git add docs/r-to-python-migration.md
git commit -m "docs(#120): reframe the restart trap from silent to loud"
```

---

### Task 3: Full verification + PR

- [ ] **Step 1: Full suite + lint + clean docs build**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check osmose/ ui/ tests/ && .venv/bin/python -m ruff format --check osmose/ ui/ tests/
rm -rf docs/_build/html && .venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html && echo "DOCS CLEAN"
```
Expected: all pass; ruff clean; `DOCS CLEAN`. (`test_tutorial_3species.py::test_trophic_cascade_visible` is a known pre-existing `skipif(CI)` local flake unrelated to this change — confirm it's the only failure and skips on CI; do not treat it as a blocker.)

- [ ] **Step 2: End-to-end proof**

```bash
.venv/bin/python -c "
import logging; logging.basicConfig(level=logging.WARNING)
from osmose.engine.config import EngineConfig, _WARNED_UNSUPPORTED_RESTART
_WARNED_UNSUPPORTED_RESTART.clear()
base = {'simulation.time.ndtperyear':'12','simulation.time.nyear':'1','simulation.nspecies':'1',
        'simulation.nschool.sp0':'5','species.name.sp0':'F','species.linf.sp0':'20','species.k.sp0':'0.3',
        'species.t0.sp0':'-0.1','species.egg.size.sp0':'0.1','species.length2weight.condition.factor.sp0':'0.006',
        'species.length2weight.allometric.power.sp0':'3.0','species.lifespan.sp0':'3',
        'species.vonbertalanffy.threshold.age.sp0':'1.0','mortality.subdt':'1',
        'predation.ingestion.rate.max.sp0':'3.5','predation.efficiency.critical.sp0':'0.57',
        'movement.distribution.method.sp0':'random','movement.randomwalk.range.sp0':'1'}
EngineConfig.from_dict(base | {'simulation.restart.file':'snap.nc'})  # should print a COLD-START warning
"
```
Expected: a warning line about `simulation.restart.file` / cold-start.

- [ ] **Step 3: Push + PR**

```bash
git push -u origin fix/issue-120-restart-warning
gh pr create --base master --head fix/issue-120-restart-warning \
  --title "fix(#120): warn when the Python engine ignores a restart request" \
  --body "$(cat <<'BODY'
Follows through on #120 (already closed, but the defect persisted). The Python engine silently cold-started when a config requested restart; it now WARNS loudly instead. It does NOT implement restart (Java engine remains the way to restart).

- `simulation.restart.file=<snapshot>` (the resume trigger — `isRestart()` in the 4.4.1 jar) → warns the run will cold-start instead of resuming.
- `simulation.restart.enabled=true` (the checkpoint-WRITE toggle — `writeRestart`) → warns no restart output will be produced.
- Distinct messages, deduped per-message (mirrors `_warn_unsupported_mortality_features`). `_log.warning` only; `validate()` and the allowlist untouched. Warn-only — no raise (a strict raise would be swallowed by the fisheries UI's broad `except` and false-positive valid Java-restart configs).
- The migration guide's restart trap is reframed from silent → loud. `data/benguela` (a faithful upstream port that sets restart) now surfaces the warning — left as-is intentionally.

Design review corrected the fix twice (verified by jar bytecode): the original checked the write toggle not the resume trigger, and dropped the strict-raise. Spec + plan under `docs/superpowers/`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

---

## Self-Review

**Spec coverage:** engine warn (both keys, distinct messages, dedup) → Task 1 Steps 1-4; tripwire flip + autouse clear → Task 1 Step 5; guide reframe (all passages, taxonomy description fix, shim-rescues-half arithmetic, §5 carve-out) → Task 2; #124 sequencing → Prerequisite + Task 2 gate; benguela left as-is → Global Constraints; old-spelling two-barrier gotcha → Task 1 Step 1 `_base_cfg` docstring; verify + PR + #120-already-closed framing → Task 3. **No gaps.**

**Placeholder scan:** No TBD/TODO. Engine code, test code, and the from_dict placement are exact (verified: `_WARNED_UNSUPPORTED_MORTALITY` at config.py:37, insertion after canonicalize at :2022, `_base_cfg` mirrored from test_engine_yieldn_meansize.py, autouse pattern from test_engine_config_validation.py). Guide line numbers are explicitly flagged pre-#121 with a re-grep instruction (Task 2 is post-rebase).

**Type consistency:** `_WARNED_UNSUPPORTED_RESTART: set[str]` and `_warn_once_restart(msg: str) -> None` used consistently in Task 1's engine code, the new test file's fixture, and the flipped tripwire's fixture. `_enabled(cfg, key) -> bool`, `_log`, `EngineConfig.from_dict` match source.
