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

- `cfg.get(...)` / `_enabled(...)` / `_species_*` literal reads. **The AST walk covers these only
  for the modules in `_EXTRA_ENGINE_SOURCES`** (`config_validation.py:310–319`) — which is **NOT
  exhaustive**: it omits `incoming_flux.py` and `timeseries.py`, both of which read config directly
  (`incoming_flux.py:90`, `timeseries.py:422–442`). Clearance must grep the **whole** `osmose/engine/`
  + `osmose/genetics/` tree, never assume "AST-known ⇒ read / not-AST-known ⇒ unread."
- `key.startswith(<prefix>)` — the engine's three dynamic prefixes, verified by grep:
  `movement.species.map` (`movement_maps.py:129`), `species.type.sp` (`resources.py:143`),
  `ltl.name.rsc` (`resources.py:97`)
- `<literal> in cfg` / `in config` membership (`background.py`, `timeseries.py`, `config.py`)
- `for key, val in cfg.items()` iteration / `re.match`-on-iterated-key consumers
  (`background.py:156`, `resources.py:142`, `movement_maps.py:128`)
- genetics trait reads (`evolution.trait.*`) and ResourceState reads (`ltl.depletable.*`,
  `species.regrowth.*`)

**Partial-index rule (round-2 finding).** A `{idx}`/`{name}` pattern is stored once but its indices
can have *different* fates: `fisheries.movement.file.map{idx}` is read only at the hardcoded index 0
(`config.py:2098` reads `fisheries.movement.file.map0`; `map1+` are inert). The partition stores one
pattern → one bucket and cannot express "map0 read, map1+ inert." **Resolution: a pattern with ANY
read index is `_ALLOWLIST_PY_HONORED`.** This deliberately accepts a false *negative* (no warning
for the genuinely-inert `map1+`) to guarantee zero false *positives* on the read index — the
asymmetry #123 is built on (the issue was deferred precisely to protect read keys). State the rule
so a reviewer doesn't "correct" `map{idx}` back to java-only.

**Do NOT batch-classify the big "Java-side schema fields" block (`config_validation.py:147–247`) by
its own comment** — its "Verified: zero hits … under osmose/engine/" claim is provably stale for at
least **four** members that the Python engine *does* read (two found in round 1, two more in round 2)
and that must go in `_ALLOWLIST_PY_HONORED`:
- `species.type.sp{idx}` — `resources.py:143` startswith + `background.py:156` regex
- `species.biomass.total.sp{idx}` — `background.py:328–330` membership + read (Baltic seal sp14 /
  cormorant sp15 configs legitimately set `species.biomass.total.sp14`)
- `simulation.incoming.flux.enabled` — `incoming_flux.py:90` `config.get`, honored via
  `get_incoming_schools` (`:191`), on the run path (`simulate.py:1384/1507`); **set `TRUE` in
  `data/benguela/benguela_all-parameters.csv:616`** → a real false-positive if misclassified
- `species.biomass.nsteps.year` — `resources.py:200` `cfg.get` (lower risk: `resources.py` *is*
  AST-scanned, so it is auto-known; still under a "(Java-side)" comment)

Plus the partial-index `fisheries.movement.file.map{idx}` above. Each entry in that block is cleared
individually, comment notwithstanding — and by the **executable guard** below, not eyeballing.

The per-key clearance is a **reviewable, committed artifact** (see Testing: the classification-
correctness test), not prose — the same three-front discipline #121 used, made re-runnable.

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

**But the snapshot test only guards partition *mechanics*, not classification *correctness*
(round-2 finding).** A read key wrongly placed in `_ALLOWLIST_JAVA_ONLY` keeps the union at 149 and
keeps the buckets disjoint — both assertions pass while the false positive ships. That is exactly
how round 1's two landmines and round 2's two more would have slipped through. Mechanics alone are
not enough; correctness needs its own executable guard.

### The classification-correctness guard (the real safety net)

An executable test proves the property the whole feature depends on — **no `_ALLOWLIST_JAVA_ONLY`
key is read by any Python engine module** — instead of trusting a hand-curated list:

```python
# Comprehensive: AST-scan EVERY osmose/engine/**.py + osmose/genetics/**.py (a SUPERSET of the
# production _EXTRA_ENGINE_SOURCES — a stricter TEST-only scan; does NOT change runtime validation).
# _extract_literal_keys_from_config_py returns a MIX of forms: concrete literals from subscript /
# literal cfg.get ("fisheries.movement.file.map0") AND {idx}-pattern forms from f-strings
# (cfg.get(f"ltl.regrowth.rate.rsc{i}") renders to "ltl.regrowth.rate.rsc{idx}"). The guard MUST
# check both forms — verified: a naive concrete-regex-only check is BLIND to the f-string form
# (round-3 finding).
engine_reads        = ast_extract_over_all_engine_modules()   # concrete literals + {idx}-form patterns
startswith_prefixes = ("movement.species.map", "species.type.sp", "ltl.name.rsc")

def _is_read(pattern):
    if pattern in engine_reads:                               # {idx}-form direct equality: f-string reads
        return True
    rx = _compile_regex_for_pattern(pattern)                  # {idx}->\d+, {name}->\w+
    if any(rx.match(lit) for lit in engine_reads if "{" not in lit):  # concrete literals: map0, etc.
        return True
    return any(pattern.split("{")[0].startswith(p) or p.startswith(pattern.split("{")[0])
               for p in startswith_prefixes)

for pattern in _ALLOWLIST_JAVA_ONLY:
    assert not _is_read(pattern), f"{pattern!r} is JAVA_ONLY but the engine reads it — reclassify PY_HONORED"
```

This test **would have caught every landmine both review rounds found**:
`simulation.incoming.flux.enabled` (literal read in the now-scanned `incoming_flux.py`),
`fisheries.movement.file.map0` (concrete literal matching `map{idx}` — the partial-index case),
`species.biomass.nsteps.year`, `species.type.sp*`, and any f-string-read pattern (via the
direct-equality branch). It is the concrete, re-runnable "per-key clearance artifact" — the guard,
not a prose table.

**The guard's ONE structural blind spot — variable-key membership / regex-on-iterated-key reads —
is bounded and enumerated, not open-ended.** No AST scan can see `total_key in config` (background)
or `re.match(pat, key)` loops (genetics). But an exhaustive grep of those read sites
(`background.py:309,329`; `timeseries.py:424–442`; `config.py:1559` `_FISHING_SCENARIOS` membership,
`:2242–2283` fishing reads; `genetics/trait.py:54,60`; `config.py:1571`) shows the honored keys are
either **not in the allowlist at all** (`species.size.proportion.file.sp{idx}`,
`mortality.fishing.catches.*` → a `validate()` unknown-key concern, not #123) or are **exactly the
already-classified** `species.biomass.total.sp{idx}` and `evolution.trait.*`. So the test's curated
exclusion list is closed at those two families — each carrying its `file:line` read evidence — and
the plan re-runs that grep inventory to prove the list stays closed. (Cross-checked: the
Java-only-classified `mortality.fishing.recruitment.age/size.sp{idx}` are genuinely unread — 0
engine hits — so they are correctly JAVA_ONLY, not another membership landmine.)

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
- **Classification-correctness guard (the correctness net, MANDATORY):** the executable test
  described in the keystone section — AST-scan every `osmose/engine/**` + `osmose/genetics/**`
  module (superset of `_EXTRA_ENGINE_SOURCES`, test-only), and assert no `_ALLOWLIST_JAVA_ONLY`
  pattern matches any read literal or startswith-prefix. Membership/regex-read honored keys the scan
  can't see (`species.biomass.total.sp{idx}`, `evolution.trait.*`) go in a curated in-test
  exclusion list, each with its `file:line` evidence. This is what actually prevents shipping a
  read key as java-only; the snapshot/disjointness test does NOT.
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
- Every key in `_ALLOWLIST_JAVA_ONLY` was cleared Python-unread by the **executable
  classification-correctness guard** (comprehensive engine AST-scan + curated dynamic-read
  exclusions), not by an allowlist comment (whose "zero engine hits" claim was already caught stale
  four times across two review rounds).
- No py-honored key is ever named in the summary — zero false positives on the legitimately-read
  keys the issue was deferred to protect. Confirmed-read keys that MUST stay py-honored:
  `movement.species.map*`, `evolution.trait.*`, `ltl.depletable.*`, `species.regrowth.*`,
  `species.type.sp*`, `species.biomass.total.sp*`, `species.biomass.nsteps.year`,
  `simulation.incoming.flux.enabled`, `fisheries.movement.file.map{idx}` (read at index 0), and the
  other background/resources/incoming-flux/timeseries reads.
- The existing suite stays green; any clean-load assertion over a bundled demo is updated, not
  silenced.
