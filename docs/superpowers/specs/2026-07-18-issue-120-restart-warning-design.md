# Fix #120 — Python engine silently ignores restart — design

**Date:** 2026-07-18 · **Issue:** [#120](https://github.com/razinkele/osmopy/issues/120)
**Branch:** `fix/issue-120-restart-warning`

> **Revised 2026-07-18 after in-loop review (2 reviewers, all findings verified by execution).**
> Two corrections reshaped the fix, so read the Method note first:
> 1. **The trigger key was WRONG.** The original spec triggered on `simulation.restart.enabled`,
>    but jar bytecode (`fr.ird.osmose.Configuration`) proves that key sets `writeRestart` (the
>    checkpoint-**writing** toggle). The **resume-from-snapshot** trigger — the actual silent
>    failure — is `simulation.restart.file` (→ `isRestart()` → `PopulatingProcess` resume-vs-
>    cold-start). The fix now checks **both**, with **distinct** messages. Schema agrees:
>    `schema/output.py:52` = "Enable restart file **output**"; `schema/simulation.py:79` =
>    "Path to restart file".
> 2. **The strict-mode RAISE is dropped (user-approved after the finding).** It was the sole
>    source of every hard problem: it would be swallowed by `ui/pages/fisheries.py`'s broad
>    `except` into a silent UI degrade, and would false-positive a valid **Java**-restart config
>    read under strict mode. Restart is a *known* key the engine won't honor = capability
>    enforcement, a different axis than `validate()`'s unknown-key job. **Warn-only** fully
>    satisfies #120 and keeps `validate()` (and thus "strict mode is silent on restart")
>    literally true.

## Problem

A config that requests restart loads clean, validates clean, and the Python engine **cold-starts
anyway, silently**. Two distinct requests, both silent today:
- **Resume:** `simulation.restart.file=<snapshot>` — asks to resume from a snapshot. The Python
  engine cold-starts instead, ignoring it. **(The primary, most harmful silent failure — wrong
  starting state.)**
- **Write:** `simulation.restart.enabled=true` — asks to write restart/checkpoint files at end
  of sim. The Python engine writes none. **(Less harmful — a missing output file.)**

Both keys are allowlisted (valid Java-side keys), so `validation.strict.enabled` does not flag
them. R OSMOSE users reach this via `initialize_osmose(type="ncdf")` → a `.file`-based resume.

**Scope: stop the SILENCE, not implement restart.** Implementing restart is a much larger piece
of work and is explicitly NOT this issue.

## Design

### Engine change (`osmose/engine/config.py`) — warn only, both keys

Follows the existing `_warn_unsupported_mortality_features()` precedent (config.py:1946, called
from `__post_init__` :1892 — "warn loudly about features PARSED but NOT applied", `_log.warning`,
module-level dedup set). Restart is the same shape.

`from_dict` is effectively **Python-scoped**: the Java runner (`osmose/runner.py`) never calls
`EngineConfig.from_dict`; only the Python engine (`engine/__init__.py:77`), `fmsy_sweep.py`, and
the UI metadata reader do. So warning at config-load is correct (Java runs are unaffected).

In `EngineConfig.from_dict`, **immediately after `canonicalize_config(cfg)`** (config.py:~2022):

```
# #120: the Python engine does not implement restart. Warn (don't silently ignore) when a
# config requests it. Check AFTER canonicalize so the old spelling output.restart.enabled ->
# simulation.restart.enabled is caught too. Two distinct requests, two distinct messages.
_restart_file = cfg.get("simulation.restart.file", "").strip()
if _restart_file and _restart_file.lower() not in ("", "null", "none"):
    _warn_once_restart(
        f"simulation.restart.file={_restart_file!r} is set, but the Python engine does not "
        "implement restart — the run will COLD-START from scratch instead of resuming from "
        "that snapshot. Use the Java engine to resume from a restart file (see issue #120)."
    )
if _enabled(cfg, "simulation.restart.enabled"):
    _warn_once_restart(
        "simulation.restart.enabled is set, but the Python engine does not write restart/"
        "checkpoint files — no restart output will be produced. Use the Java engine for "
        "restart output (see issue #120)."
    )
```

Add, beside `_WARNED_UNSUPPORTED_MORTALITY`:
```
_WARNED_UNSUPPORTED_RESTART: set[str] = set()

def _warn_once_restart(msg: str) -> None:
    if msg not in _WARNED_UNSUPPORTED_RESTART:   # throttle: from_dict reruns per calibration candidate
        _WARNED_UNSUPPORTED_RESTART.add(msg)
        _log.warning("%s", msg)
```

**No raise anywhere.** `validate()` and the allowlist are untouched — the restart keys stay
KNOWN (they are valid Java-side keys; flagging them unknown would be wrong).

**Per-message dedup** (not a single flag): the `.file` and `.enabled` messages differ, so a
config setting both emits both once. Distinct messages also mean the dedup key naturally
distinguishes different `.file` paths.

**Placement rationale (verified):** `_validate_cfg(cfg,_mode)` → `canonicalize_config` → build.
`output.restart.enabled` canonicalizes to `simulation.restart.enabled` (verified), so the
post-canonicalize check catches both spellings. `_log.warning` (not `warnings.warn`) matches the
precedent and satisfies the dual-channel #120 tripwire.

### Behavior tests (`tests/test_issue_120_restart_warning.py`, new)

**Every test that asserts the warning MUST clear `_WARNED_UNSUPPORTED_RESTART` first** (autouse
fixture), mirroring the mortality precedent which clears `_WARNED_UNSUPPORTED_MORTALITY` in
`tests/test_engine_config_validation.py:15` and `tests/test_engine_selectivity_warning.py:35`.
Without this the module-global dedup makes assertions order-dependent — a bundled config
(`data/benguela`, see below) sets restart and fires the warning first in a full run, poisoning
the set for later positive-assertion tests in *other files*.

- `simulation.restart.file=<path>` → `_log.warning` naming the file + "COLD-START … Java engine". (caplog)
- `simulation.restart.enabled=true` → warning naming "restart output … not produced". (caplog)
- Both set → **both** messages emitted (per-message dedup). 
- No restart key → no warning. `simulation.restart.enabled=false` and empty/`"null"`
  `simulation.restart.file` → no warning.
- **Old spelling** `output.restart.enabled=true` → also warns — **but the test must strip
  `osmose.version` first** (or build from a version-less raw dict): `data/minimal` stamps
  `osmose.version=4.4.1`, and `canonicalize_config` early-returns on the exact-match version, so
  reading MINIMAL_CONFIG then injecting the old key would NOT rename it. Verified gotcha.
- **Dedup:** two `from_dict` calls with the same restart request → message logged once.

### Test coupling — the #120 tripwire flips (by design)

`tests/test_r_dialect_migration_claims.py::test_engine_does_not_yet_warn_on_ignored_restart`
(line ~141) asserts the engine does NOT warn on `simulation.restart.enabled=true` today. This
fix makes it warn, so it goes red — exactly as its docstring predicts. **Update it** to assert
the engine now DOES warn (rename `…_warns_on_ignored_restart_after_120`; keep dual-channel
capture; assert a restart warning IS present). **It too must clear `_WARNED_UNSUPPORTED_RESTART`
first** (autouse or explicit), since it lives in a different file than the new tests and a full
run may have populated the set.

The sibling `test_strict_mode_is_SILENT_on_unimplemented_restart` (line ~117) stays **fully
true and unchanged** — its assertion is about `validate()` (which this fix does not touch), and
under warn-only "strict mode is silent on restart" remains literally true (the non-silence comes
from the *engine* warning, in all modes, not from strict validation). No docstring change needed.

### Guide coupling (`docs/r-to-python-migration.md`)

Restart moves from "**capability absent / silent**" to "**capability absent / loud** (the engine
warns on load)" — same shape as the surveys trap. The engine warns in **all** modes (not just
strict), so a reader running any restart config now sees it. `validate()` stays silent, so the
guide's strict-mode narrative is unchanged. Enumerate and reframe **every** passage (grep, don't
trust a single hit):

- **L196** taxonomy row for restart: Signal `silent` → `loud (engine warns on load)`.
- **L27-29** (§1 losses, "nothing consumes a restart file") → the Python engine still has no
  restart, **but now warns** when a config requests it (fixed #120); Java engine to actually restart.
- **L629** honest-gaps row ("loads and validates clean, does nothing") → "loads clean but now
  **warns**; the engine still doesn't restart — use the Java engine". Update its `#120` link:
  the SILENCE is fixed, so it's no longer an open tracker.
- **All `#120` references** (there are several — L595, L604, L629; verify by grep) → "fixed in
  #120", scoped precisely to the SILENCE, never claiming restart is implemented.
- **L107-108** ("strict mode is silent on the restart gap"): under warn-only this stays **true**
  (`validate()` still doesn't flag the known key). Optional one-clause clarification that the
  *engine* now warns even though the *validator* stays silent — but no correction is required.
- **Leave the still-live traps untouched** (spatial-inputs `.nc`, missing sub-config, cross-file
  precedence — none fixed here).

Clean `-W` sphinx build required (`rm -rf docs/_build/html` first).

### Bundled config: `data/benguela` — LEAVE it, do not flip

`data/benguela/benguela_all-parameters.csv:499` sets `output.restart.enabled ; TRUE`. Benguela
is a **faithful port of upstream `osmose-ben_v4.3_Florance`** bundled as a Python-engine example
(`docs/superpowers/specs/2026-07-06-benguela-example-bundling-design.md`). The `TRUE` is a
fidelity copy from the real Java config; the Python engine ignores it. **Do NOT flip it to false**
— that mutates a faithful port's meaning to silence our own new warning. The deduped warning
correctly surfaces that this upstream setting is a no-op on the Python engine — the fix working
as intended on a real bundled config. `test_benguela_bundle.py` will emit the warning; the
autouse dedup-clear fixtures make that harmless for other tests' assertions.

## Out of scope (explicit)

- **Implementing restart** in the Python engine (the large piece; not #120's ask).
- **A strict-mode raise** — dropped (see the revision note); warn-only.
- **`validate()` / the allowlist** — unchanged; restart is caught in the engine, not the
  unknown-key checker.
- **The Layer-D systemic warning (#123)** — this is a hand-written warning for two specific keys,
  exactly like the mortality precedent; NOT the general mechanism.
- **Flipping `data/benguela`'s restart** — it's a faithful port; leave it.

## Success criteria

- `simulation.restart.file=<snapshot>` on the Python engine → a **loud warning** naming the file
  and the Java-engine resume workaround — verified by running.
- `simulation.restart.enabled=true` → a **distinct** loud warning about missing restart output.
- The old spelling `output.restart.enabled=true` triggers it too (canonicalize path), with the
  `osmose.version` gotcha handled in the test.
- Warnings are deduped per-message (once per process) so calibration isn't spammed.
- The #120 tripwire test is updated to assert the warning; the sibling strict-validation test is
  unchanged (its assertion stays true).
- Every test asserting the warning clears the dedup set first (autouse); the full suite is not
  order-dependent.
- The guide reframes restart from silent to loud across ALL its passages, without claiming
  restart is implemented; still-live traps intact; `-W` build clean.
- `validate()`, the allowlist, and `data/benguela` are untouched; no other config or test breaks.
