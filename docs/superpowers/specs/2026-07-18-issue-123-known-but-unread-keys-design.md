# Fix #123 — systemic warning for known-but-unread config keys — design

**Date:** 2026-07-18 · **Issue:** [#123](https://github.com/razinkele/osmopy/issues/123)
**Branch:** `fix/issue-123-known-but-unread-keys`
**Scope:** Layer D of #121 — the "real fix" #120/#121 deferred: warn when a config sets a
key that loads clean, validates clean, and is then silently never read by the **Python** engine.

> **Sequencing dependency (like #120→#124).** This spec is written on a branch off `master`
> (which has #121's allowlist changes — the thing #123 partitions). It references #120's restart
> warning (`_warn_once_restart` in `config.py`, PR #125, currently **open**) for a carve-out. The
> implementation plan must **merge #120/#125 first, then rebase #123**, so the carve-out targets
> live code. If #120 is still open at execution time, that ordering is a hard prerequisite.

## Method — this issue is about false "it's dead / it's read" claims, so every classification is executed

The whole failure mode #123 addresses is a key that *looks* handled but silently no-ops. The
symmetric trap while *building* the fix is trusting an annotation instead of running the code —
#121 already caught the allowlist comments lying (its `config_validation.py:99–100` block claimed
five keys "control the Java engine's output" when they were dead on **both** engines).

**Therefore every one of the 149 `_SUPPLEMENTARY_ALLOWLIST` entries is classified by execution,
never from its neighboring comment.** A key lands in `_ALLOWLIST_JAVA_ONLY` (warns) only after it
is cleared Python-unread across *all* read mechanisms the engine actually uses:

- `cfg.get(...)` / `_enabled(...)` / `_species_*` literal reads (the AST walk already covers these)
- `key.startswith(<prefix>)` — the engine's three dynamic prefixes, verified by grep:
  `movement.species.map` (`movement_maps.py:129`), `species.type.sp` (`resources.py:143`),
  `ltl.name.rsc` (`resources.py:97`)
- `<literal> in cfg` / `in config` membership (`background.py`, `timeseries.py`, `config.py`)
- `for key, val in cfg.items()` iteration consumers (`background.py:156`, `resources.py:142`,
  `movement_maps.py:128`)
- genetics trait reads (`evolution.trait.*`) and ResourceState reads (`ltl.depletable.*`,
  `species.regrowth.*`)

The per-key clearance (which mechanism honors it, or "none found → java-only") is a **reviewable
artifact** the plan produces and the reviewer checks — the same three-front discipline #121 used.

## Architecture — three pieces in the existing validation seam

All three live where the machinery already is; nothing new is threaded through the engine.

### 1. Partition the allowlist (`config_validation.py`)

Replace the single `_SUPPLEMENTARY_ALLOWLIST` frozenset with two named frozensets whose **union is
byte-identical** to today's set:

- `_ALLOWLIST_PY_HONORED` — the Python engine reads it (any mechanism above), **or** it is
  reader-injected metadata (`osmose.version`, `osmose.configuration.*`).
- `_ALLOWLIST_JAVA_ONLY` — a real OSMOSE/Java key the Python engine provably does not read.

```python
_SUPPLEMENTARY_ALLOWLIST = _ALLOWLIST_PY_HONORED | _ALLOWLIST_JAVA_ONLY
```

`build_known_keys()` continues to union `_SUPPLEMENTARY_ALLOWLIST` exactly as before, so **all
existing unknown-key validation is untouched** — this change only *labels* what is already there.

### 2. `java_only_keys_set(cfg) -> list[str]` (`config_validation.py`)

Given a config dict, return the java-only keys it actually sets, sorted. Reuses the existing
literal + `{idx}`/`{name}`-regex matching (`_normalize_key_to_pattern`, the compiled regex pairs)
so `output.ssb.enabled` matches by literal and `output.diet.stage.threshold.sp3` matches by
pattern. Excludes the #120-owned restart keys (see reconciliation below).

```python
def java_only_keys_set(cfg: dict) -> list[str]:
    """Real OSMOSE keys present in cfg that the Python engine does not read."""
    ...  # match each cfg key against _ALLOWLIST_JAVA_ONLY (literals + regexes),
         # minus _RESTART_HANDLED_BY_120, return sorted list
```

### 3. One deduped summary warning in `from_dict` (`config.py`)

Beside #120's restart warn, after `cfg, _ = canonicalize_config(cfg)`. Emit **one** summary line
(not per-key), deduped by a module-global set exactly like `_WARNED_UNSUPPORTED_RESTART`:

```python
_WARNED_JAVA_ONLY_KEYS: set[str] = set()

def _warn_once_java_only(keys: list[str]) -> None:
    fingerprint = ",".join(keys)
    if fingerprint and fingerprint not in _WARNED_JAVA_ONLY_KEYS:
        _WARNED_JAVA_ONLY_KEYS.add(fingerprint)
        _log.warning(
            "%d config key(s) are valid OSMOSE keys the Python engine does not implement; "
            "on this engine they have no effect. Use the Java engine if you need them: %s "
            "(see issue #123).",
            len(keys), ", ".join(keys),
        )
```

**Why `from_dict` = free Python-engine gating.** `from_dict` is called only from
`osmose/engine/__init__.py:77` (the Python run path) and `osmose/validation/fmsy_sweep.py`
(Python-side validation). No Java runner calls it (verified by grep). So the warning fires **only
on Python-engine loads** — it can never fire on an actual Java run, which resolves the issue's
"must not warn when the Java engine reads them" concern mechanically, without an engine flag.

## The keystone — partition-completeness test

This is what makes #123 *systemic* rather than "#120 with a longer list." A test asserts the two
sets exactly partition the allowlist:

```python
assert _ALLOWLIST_PY_HONORED | _ALLOWLIST_JAVA_ONLY == _SUPPLEMENTARY_ALLOWLIST
assert _ALLOWLIST_PY_HONORED & _ALLOWLIST_JAVA_ONLY == frozenset()
```

Every future allowlist addition must be consciously placed in one bucket or the suite goes red —
the drift the issue warned about ("the validator currently cannot tell which") becomes a
compile-time-ish obligation instead of silent rot.

## Two reconciliations the bundled-config audit surfaced

### #120 overlap — carve out the restart keys

`simulation.restart.file` and `simulation.restart.enabled` are java-only in nature, but #120
(PR #125) already warns on them with a **better-targeted** message (`.file` = won't-resume,
`.enabled` = no-checkpoint-output). To avoid double-warning, `java_only_keys_set` subtracts a small
carve-out:

```python
_RESTART_HANDLED_BY_120 = frozenset({"simulation.restart.file", "simulation.restart.enabled"})
```

They remain in `_ALLOWLIST_JAVA_ONLY` for partition-completeness (they *are* java-only); the
carve-out only removes them from #123's summary line. Documented so a future reader doesn't
"fix" the apparent omission.

### `output.diet.stage.threshold.sp{idx}` — classified JAVA_ONLY

Emitted only by `stage_background_for_java` (`java_background_staging.py:182`), which runs on the
**Java-launch path** (`runner.py`) and never through `from_dict`. On a Python run the key is inert
→ JAVA_ONLY. The issue's original "must not warn on those" was conservatism; `from_dict`'s
Python-only scope means it never fires on the Java run where the key *is* honored. Its sibling
`output.diet.stage.structure` classifies the same way (both under the Java `output.diet.stage`
prefix; execution-cleared Python-unread).

## Bundled demos — flat + one line, left as-is (user decision 2026-07-18)

The audit found our own demos already set java-only keys: `output.diet.stage.threshold` in
`data/baltic`, `data/examples`, `data/eec_full`, `data/baltic-fine`; and the broader families
(`output.ssb.enabled`, `temperature.filename`, `simulation.ncpu`, `grid.java.classname`, …) across
`data/examples`, `data/benguela`, `data/eec_full`, `data/baltic`. **Decision: warn on all
java-only keys as one deduped summary line, and do NOT touch the demo configs.** The single line is
honest (the demos do set Java-only keys the Python engine ignores) and low-noise, and the configs
may legitimately be run on the Java engine too. Cleaning them (or harm-tiering the message) was
considered and rejected as scope creep / risk.

**There is a genuine harm gradient inside the java-only set** (recorded for the guide/future, not
acted on here): `temperature.filename` / `oxygen.filename` cause silent *wrong physics* (Python
uses the `temperature.value` scalar instead), whereas `output.*.enabled` flags are low-harm (Python
has its own output system in `osmose/engine/output.py`). The flat one-line message names all of
them; per-key harm ranking is explicitly out of scope (YAGNI).

## Testing

- **Partition completeness (keystone):** union equals `_SUPPLEMENTARY_ALLOWLIST`, intersection
  empty. (Also guards that the refactor preserved the original 149-entry set exactly.)
- **`java_only_keys_set`:** returns java-only keys a config sets (literal match, e.g.
  `output.ssb.enabled`; pattern match, e.g. `output.diet.stage.threshold.sp3`); **excludes**
  py-honored keys (e.g. `movement.species.map0`, `evolution.trait.imax.target`,
  `ltl.depletable.enabled`); **excludes** the #120 restart carve-outs; returns `[]` for a config
  with none.
- **`from_dict` warning:** a raw config setting java-only keys triggers exactly one summary
  `_log.warning` naming them; deduped to once per process for the same key set; **no** warning when
  the config sets none; the #120 restart keys do **not** appear in the #123 summary (no double
  warn).
- **Per-key clearance artifact:** a test (or committed data file) enumerating each
  `_ALLOWLIST_JAVA_ONLY` entry with the grep evidence it is Python-unread — the reviewable artifact
  the Method section requires.
- **Whole-suite guard:** the existing suite must stay green. Audit any test that asserts a
  clean/warning-free load of `examples`/`eec`/`benguela`/`baltic` (or runs pytest under
  `-W error`); the new summary line must not break it. If such an assertion exists, update it to
  expect the summary (do not silence it globally). The known pre-existing CI-skip flake
  `test_trophic_cascade_visible` is unrelated.

## Out of scope (explicit)

- **`conversion2tons` aliasing** — the issue lists aliasing the legacy `species.conversion2tons` /
  `ltl.conversion2tons` forms to canonical `resource.conversion2tons` as a *follow-up*; not #123.
  They stay in `_ALLOWLIST_JAVA_ONLY` (legacy 4.3.x, 0 jar hits, Python-unread) and will surface in
  the summary on a Python run — correct, since they are inert on this engine.
- **Approach B (runtime access-tracking)** and **Approach C (targeted per-family #120-extension)** —
  considered and rejected (B fires post-run, defeating the pre-run value, and is invasive; C is not
  systemic). Approach A chosen.
- **Harm-tiering the message** — one flat summary line; per-key severity is not modeled.
- **`species.lw.*`** — fails *loudly* with a `KeyError` naming the correct key; not a silent no-op,
  not #123-class (stays py-honored/legacy-alias, no warning).

## Success criteria

- A Python-engine load of a config that sets real OSMOSE keys the Python engine ignores (e.g.
  `output.ssb.enabled`, `temperature.filename`) emits exactly one summary warning naming them,
  verified by running.
- The warning **never** fires on a Java-engine run (guaranteed by `from_dict` scope) and never
  double-warns the #120 restart keys.
- `_ALLOWLIST_PY_HONORED` and `_ALLOWLIST_JAVA_ONLY` exactly partition the original allowlist; the
  completeness test enforces it for all future additions.
- Every key in `_ALLOWLIST_JAVA_ONLY` was cleared Python-unread by execution, with a reviewable
  per-key artifact — no classification rests on an allowlist comment.
- No py-honored key (`movement.species.map*`, `evolution.trait.*`, `ltl.depletable.*`,
  background/resources reads) is ever named in the summary — zero false positives on the
  legitimately-read keys the issue was deferred to protect.
- The existing suite stays green; any clean-load assertion over a bundled demo is updated, not
  silenced.
