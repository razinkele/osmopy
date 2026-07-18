# Fix #120 — Python engine silently ignores restart — design

**Date:** 2026-07-18 · **Issue:** [#120](https://github.com/razinkele/osmopy/issues/120)
**Branch:** `fix/issue-120-restart-warning`

## Problem

`simulation.restart.enabled=true` (and the old spelling `output.restart.enabled`) loads clean,
validates clean, and the Python engine **cold-starts anyway, ignoring the restart request, with
no warning**. The keys are allowlisted (valid Java-side keys), so `validation.strict.enabled`
does not flag them. A user gets plausible output from a run that did not do what the config
asked. R OSMOSE users reach this directly via `initialize_osmose(type="climatology"|"ncdf")`.

**Scope: stop the SILENCE, not implement restart.** Implementing restart in the Python engine
is a much larger piece of work and is explicitly NOT this issue. This fix turns a silent no-op
into a loud one.

## Design

### Engine change (`osmose/engine/config.py`)

Follows the existing `_warn_unsupported_mortality_features()` precedent (config.py:1949 — "warn
loudly about features PARSED but NOT applied", `_log.warning`, module-level dedup set, called
from `__post_init__`). Restart is the same shape: requested but not applied.

In `EngineConfig.from_dict`, **immediately after `canonicalize_config(cfg)`** (config.py:~2022):

```
if _enabled(cfg, "simulation.restart.enabled"):
    msg = ("simulation.restart.enabled is set, but the Python engine does not implement "
           "restart — the run will cold-start from scratch, ignoring the restart snapshot. "
           "Use the Java engine for restart (see issue #120).")
    if _mode == "error":          # validation.strict.enabled=error
        raise ValueError(msg)
    # off / warn: log once (dedup — from_dict reruns per calibration candidate)
    if msg not in _WARNED_UNSUPPORTED_RESTART:
        _WARNED_UNSUPPORTED_RESTART.add(msg)
        _log.warning("%s", msg)
```

Add module-level `_WARNED_UNSUPPORTED_RESTART: set[str] = set()` beside
`_WARNED_UNSUPPORTED_MORTALITY`.

**Why after canonicalize:** a config using the pre-4.4.0 spelling `output.restart.enabled`
is renamed to `simulation.restart.enabled` by `canonicalize_config` (verified). Checking the
canonical key after canonicalization catches BOTH spellings with one check.

**Why `_log.warning` (not `warnings.warn`):** matches the mortality precedent. The #120 tripwire
test is dual-channel (catches either), so logging satisfies it.

**Why error under strict (user-approved):** mirrors `validate()`'s own warn-vs-raise-per-mode
semantics. A user who set `validation.strict.enabled=error` asked to fail on config problems; a
requested-but-unsupported restart is one, and restart changes the run's entire starting state
(snapshot vs cold-start) — more severe than the mortality-feature warnings. `_mode` is already
read at the top of `from_dict` (`_raw_mode = cfg.get("validation.strict.enabled", "off")`).

**Dedup applies only to the warn path.** The error path raises every time (it stops the run;
no throttle needed).

**Not touched:** the allowlist — the restart keys stay KNOWN (they are valid Java-side keys;
flagging them as unknown would be wrong). This fix acts in the engine, not in `validate()`.

### Behavior tests (`tests/test_issue_120_restart_warning.py`, new)

- `simulation.restart.enabled=true`, default (off) mode → `_log.warning` emitted, message names
  the key + the Java-engine workaround. (caplog.)
- `validation.strict.enabled=error` + restart → `EngineConfig.from_dict` raises `ValueError`.
- No restart key → no warning, no raise.
- **Old spelling** `output.restart.enabled=true` (no `simulation.restart.enabled`) → also warns
  (proves the post-canonicalize placement catches both).
- **Dedup:** two `from_dict` calls with restart → the message is logged once. (Reset
  `_WARNED_UNSUPPORTED_RESTART` in the test to isolate; it is module-global.)
- `simulation.restart.enabled=false` → no warning (only `true` triggers).

### Test coupling — the #120 tripwire flips (by design)

`tests/test_r_dialect_migration_claims.py::test_engine_does_not_yet_warn_on_ignored_restart`
(line ~141) asserts the engine does NOT warn on restart today. This fix makes it warn, so the
test goes red — exactly as its own docstring predicts ("when #120 lands and the engine warns,
THIS test goes red"). **Update it** to assert the engine now DOES warn (rename to
`test_engine_warns_on_ignored_restart_after_120`; keep the dual-channel capture; assert a
restart warning IS present).

Also review the sibling `test_strict_mode_is_SILENT_on_unimplemented_restart` (line ~117): its
ASSERTION (`validate()` does not report `simulation.restart.enabled` as unknown) stays TRUE —
this fix does not touch `validate()`. But its docstring's broad framing ("strict mode never
reports it") is now incomplete: under `validation.strict.enabled=error`, `from_dict` raises on
restart via the new engine check (not via `validate()`). Update the docstring to note that the
engine — not the unknown-key validator — is what catches restart post-#120. Do NOT change the
assertion.

### Guide coupling (`docs/r-to-python-migration.md`)

PR #124's reframe explicitly kept restart as a **still-live silent trap**. #120 fixes the
SILENCE, not the capability: restart is still unsupported (Java engine remains the workaround),
but now fails **loudly** — a warning by default, a raise under strict. So in the guide's taxonomy
restart moves from "**capability absent / silent**" to "**capability absent / loud** (warns;
errors under strict)" — the same shape as the surveys trap. Concretely:

- The taxonomy table row for `simulation.restart.enabled` (Signal column: "silent") → "loud
  (warns; errors under strict)".
- Any prose calling restart a *silent* trap → reframe: the engine now warns/errors; restart
  itself is still Java-only. **Do NOT claim restart is implemented.**
- Update the `#120` reference from "tracked/still-live" to "fixed in #120" — but scoped
  precisely to the SILENCE being fixed, not to restart being supported.
- **Leave the still-live traps untouched** (spatial-inputs `.nc`, missing sub-config, cross-file
  precedence — none fixed here).

Clean `-W` sphinx build required (`rm -rf docs/_build/html` first — stale cache skips `-W`).

## Out of scope (explicit)

- **Implementing restart** in the Python engine (the large piece; not #120's ask).
- **The allowlist** — restart keys stay known/valid.
- **`validate()`** — unchanged; restart is caught in the engine, not the unknown-key checker.
- The Layer-D systemic warning (#123) — this is a targeted, hand-written warning for one
  specific feature, exactly like the mortality precedent; it is NOT the general mechanism.

## Success criteria

- A config with `simulation.restart.enabled=true` on the Python engine produces a **loud
  warning** (default) or a **`ValueError`** (strict=error), naming the key and the Java-engine
  workaround — verified by running.
- The old spelling `output.restart.enabled=true` triggers it too (canonicalize path).
- The warning is deduped (once per process) so calibration isn't spammed.
- The #120 tripwire test is updated to assert the warning (not deleted, not left red); the
  sibling strict-validation test's assertion is unchanged.
- The guide reframes restart from silent to loud, without claiming restart is implemented, and
  leaves the still-live traps intact; `-W` build clean.
- `validate()` and the allowlist are untouched; no other config or test silently breaks.
