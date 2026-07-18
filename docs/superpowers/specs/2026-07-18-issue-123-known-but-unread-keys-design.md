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

**Do NOT batch-classify the big "Java-side schema fields" block (`config_validation.py:147–247`) by
its own comment** — its "Verified: zero hits … under osmose/engine/" claim is already stale for at
least two members that the Python engine *does* read, and must go in `_ALLOWLIST_PY_HONORED`:
`species.type.sp{idx}` (`resources.py:143` startswith + `background.py:156` regex) and
`species.biomass.total.sp{idx}` (`background.py:328–330` membership + read). A background-species
config (Baltic seal sp14 / cormorant sp15) legitimately sets `species.biomass.total.sp14`;
misclassifying it JAVA_ONLY would emit a spurious #123 warning. Each entry in that block is cleared
individually, comment notwithstanding.

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

### 3. One deduped summary warning at the Python-engine RUN seam (`PythonEngine._prepare_run`)

Emit **one** summary line (not per-key), deduped by a module-global set exactly like #120's
`_WARNED_UNSUPPORTED_RESTART`:

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

**Placement = `PythonEngine._prepare_run` (`osmose/engine/__init__.py`), NOT `from_dict`.** This
corrects the original spec's foundational error (caught in in-loop review round 1): `from_dict` is a
general-purpose config *constructor*, not a Python-engine marker. The Fisheries UI calls it
**engine-agnostically** to interpret results of *either* engine (`ui/pages/fisheries.py:181,211`),
so a warning there fires when a user opens the Fisheries tab after a **Java** run — telling them to
"use the Java engine" for keys Java just used, the exact false positive #123 exists to prevent.

`_prepare_run` is the correct seam: it is called **only** from `run()` (`__init__.py:99`) and
`run_in_memory()` (`:141`) — i.e. only when the **Python engine actually runs** a simulation — and
the analysis paths (Fisheries UI; `fmsy_sweep` probes at `fmsy_sweep.py:321,404`) call `from_dict`
*directly*, bypassing `_prepare_run`. Verified by grep: `_prepare_run` has exactly those two
callers. This is genuine engine-gating by execution point, not by an inferred property of a shared
constructor.

`java_only_keys_set(cfg)` **canonicalizes internally** (mirrors `validate()`, which calls
`canonicalize_config` at `config_validation.py:526`), so it can be handed the raw `config` that
`_prepare_run` receives and still match post-rename canonical keys.

> **Note (out of scope):** #120's restart warning lives in `from_dict` and so has the same latent
> false-positive on the Fisheries-UI Java-analysis path. That is a pre-existing #120/#125 concern,
> not reopened here; #123 is placed correctly from the start. Flag it as a possible #120 follow-up.

## The keystone — partition-completeness test (against an independent snapshot)

This is what makes #123 *systemic* rather than "#120 with a longer list." **Subtlety caught in
review round 1:** because the source defines `_SUPPLEMENTARY_ALLOWLIST = _ALLOWLIST_PY_HONORED |
_ALLOWLIST_JAVA_ONLY`, asserting `A | B == _SUPPLEMENTARY_ALLOWLIST` is tautological (`(A|B) ==
(A|B)`) and could NOT catch a key accidentally dropped during the split — the union silently drops
to 148, the key becomes "unknown", and `build_known_keys()` starts flagging it. The test therefore
compares the union against an **independent frozen snapshot** of the pre-refactor 149-key set,
captured verbatim in the test file (not derived from the source):

```python
# tests/…: FROZEN_ALLOWLIST_SNAPSHOT is the exact 149-key set copied from the pre-#123 source —
# an independent reference the source definition cannot circularly satisfy.
assert _ALLOWLIST_PY_HONORED | _ALLOWLIST_JAVA_ONLY == FROZEN_ALLOWLIST_SNAPSHOT  # nothing lost/added
assert _ALLOWLIST_PY_HONORED & _ALLOWLIST_JAVA_ONLY == frozenset()                # disjoint
```

The snapshot catches an accidental drop during the initial split; disjointness catches a
double-classification. A *legitimate* future allowlist addition updates both the source (into one
bucket) and the snapshot, consciously — turning the drift the issue warned about ("the validator
currently cannot tell which") into an explicit obligation instead of silent rot.

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

**Exactly two, verified.** #120 warns on precisely these two keys — confirmed on the `#125` branch
at `config.py:2040` (`simulation.restart.file`) and `:2047` (`simulation.restart.enabled`). The
allowlist holds **four other** restart-family keys #120 does *not* touch
(`simulation.restart.spinup.nyear`, `simulation.restart.recordfrequency.ndt`,
`output.restart.recordfrequency.ndt`, `output.restart.spinup`); those stay java-only and #123
**correctly warns** on them (they are inert on the Python engine and nothing else reports them).
Because #120/#125 is not yet merged, the plan MUST re-verify #120's actual warn-set at rebase — if
#120 grew to warn on any additional restart key, the carve-out set grows to match, or #123
double-warns.

### `output.diet.stage.threshold.sp{idx}` — classified JAVA_ONLY

Emitted only by `stage_background_for_java` (`java_background_staging.py:182`), which runs on the
**Java-launch path** (`runner.py`, via `ui/pages/run.py`) and never on the Python-engine run path.
On a Python run the key is inert → JAVA_ONLY. The issue's original "must not warn on those" was
conservatism; the `_prepare_run` run-seam placement (§3) means the summary never fires on the Java
run where the key *is* honored — the Java engine never reaches `_prepare_run`. Its sibling
`output.diet.stage.structure` classifies the same way (both under the Java `output.diet.stage`
prefix; execution-cleared Python-unread).

## Bundled demos — flat + one line, left as-is (user decision 2026-07-18)

The audit found our own demos already set java-only keys (grep-confirmed present):
`simulation.ncpu` (all 4 demo dirs), `output.diet.stage.threshold` (all 4), and
`grid.java.classname` (3). (`output.ssb.enabled` / `temperature.filename` are *not* in any bundled
demo — they are valid java-only keys used elsewhere in this spec as harm examples, not demo
evidence.) **Decision: warn on all
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

- **Partition completeness (keystone):** union equals the **independent `FROZEN_ALLOWLIST_SNAPSHOT`**
  (149 keys, copied verbatim into the test — NOT `_SUPPLEMENTARY_ALLOWLIST`, which would be
  circular); intersection empty. This is what proves the split lost/added no key.
- **`java_only_keys_set`:** returns java-only keys a config sets (literal match, e.g.
  `simulation.ncpu`; pattern match, e.g. `output.diet.stage.threshold.sp3`); **excludes** py-honored
  keys — including `movement.species.map0`, `evolution.trait.imax.target`, `ltl.depletable.enabled`,
  **and `species.biomass.total.sp14`** (the round-1 stale-comment landmine); **excludes** the #120
  restart carve-outs; canonicalizes a legacy-spelled key before matching; returns `[]` for a config
  with none.
- **`_prepare_run` warning (the run seam):** a Python-engine run (`run()` / `run_in_memory()`) whose
  config sets java-only keys triggers exactly one summary `_log.warning` naming them; deduped to
  once per process for the same key set; **no** warning when the config sets none; the #120 restart
  keys do **not** appear in the #123 summary (no double warn).
- **False-positive guard (the round-1 defect):** constructing a config via `EngineConfig.from_dict`
  **directly** (the Fisheries-UI / `fmsy_sweep`-probe pattern) does **NOT** emit the #123 warning —
  only an actual `_prepare_run` does. This test pins the placement so a future refactor can't slide
  the warning back into the constructor and re-introduce the Java-run false positive.
- **Dedup-global reset fixture (MANDATORY):** `_WARNED_JAVA_ONLY_KEYS` is a module-global that
  persists across tests in a process. Add an `@pytest.fixture(autouse=True)` that clears it before
  each test — **exactly as #120 required for `_WARNED_UNSUPPORTED_RESTART`**. Without it, the
  "exactly one warning" and "deduped once per process" tests are mutually order-dependent (whichever
  runs second sees a polluted global) and flake.
- **Per-key clearance artifact:** a test (or committed data file) enumerating each
  `_ALLOWLIST_JAVA_ONLY` entry with the grep evidence it is Python-unread — the reviewable artifact
  the Method section requires.
- **Whole-suite guard:** the existing suite must stay green. Audit any test that asserts a
  clean/warning-free load or **run** of `examples`/`eec`/`benguela`/`baltic` (or runs pytest under
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

- A Python-engine **run** of a config that sets real OSMOSE keys the Python engine ignores (e.g.
  `simulation.ncpu`, `temperature.filename`) emits exactly one summary warning naming them,
  verified by running.
- The warning **never** fires when a config is merely constructed for analysis
  (`EngineConfig.from_dict` from the Fisheries UI / `fmsy_sweep` probes) — guaranteed by the
  `_prepare_run` run-seam placement, verified by the false-positive guard test — and never
  double-warns the #120 restart keys.
- `_ALLOWLIST_PY_HONORED` and `_ALLOWLIST_JAVA_ONLY` exactly partition the original allowlist,
  verified against an independent frozen snapshot; the completeness test enforces it for all future
  additions.
- Every key in `_ALLOWLIST_JAVA_ONLY` was cleared Python-unread by execution, with a reviewable
  per-key artifact — no classification rests on an allowlist comment (whose "zero engine hits"
  claim was already caught stale in round 1).
- No py-honored key (`movement.species.map*`, `evolution.trait.*`, `ltl.depletable.*`,
  `species.type.sp*`, `species.biomass.total.sp*`, and the other background/resources reads) is
  ever named in the summary — zero false positives on the legitimately-read keys the issue was
  deferred to protect.
- The existing suite stays green; any clean-load assertion over a bundled demo is updated, not
  silenced.
