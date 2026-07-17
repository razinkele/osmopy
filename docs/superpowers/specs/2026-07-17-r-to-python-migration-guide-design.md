# R OSMOSE → Python migration guide — design

**Date:** 2026-07-17 · **Status:** Approved, ready for planning
**Deliverable:** `docs/r-to-python-migration.md` (one MyST Markdown file, added to the
`Guides` toctree in `docs/index.md`) + one fixture test.

## Purpose

For an **existing R OSMOSE user with a working config and R driver scripts who has decided to
switch** ("port my model"), prefaced by a short honest "should you switch?" for readers who
haven't committed.

Out of scope: newcomers translating R papers (subsumed); R⇄Python interop (outputs are plain
CSV/NetCDF — near-free, needs no guide).

## Method — read before writing a line

This spec survived four review rounds. The pattern in what survived is the implementer's
working rule, and it is also the guide's thesis:

> **Every claim that survived is one where someone ran the code. Every claim that failed
> reasoned from a grep hit, a line number, or an allowlist entry to a runtime behavior.**

The failures were plausible, not careless, and each was one command from disproof. A retracted
8-row table reasoned from *the allowlist* to "R users set these keys" — 7 of 8 didn't exist. A
retracted claim reasoned from `engine_capabilities.py:32` to "the UI renders a page" — it
doesn't.

**An allowlist entry is not evidence of a read. A line number is not evidence of a behavior. A
schema declaration is not evidence of a consumer.** Verify at the *consumption site*, and for
any "R users write X" claim, grep the actual corpus. Full rationale:
[#121](https://github.com/razinkele/osmopy/issues/121).

## Evidence base

R-side claims are grounded in four public config repos, read at source. Verified 2026-07-17:

| Repo | Ground truth used |
|---|---|
| `osmose-gog` | `run.R`, `runModel.R`, `calibrate.R` |
| `osmose-ben` | `launcher.R`, `osmose-ben.R` (config), `utils/ben-plots.R` |
| `osmose-gom` | `dynamics_gom.R`, `analysis.R` |
| `osmose-eec` | `convert-ltl.R`, `eec_all-parameters.csv` |

The R package left CRAN 2022-10-03; current v4.x installs from GitHub/drat. Readers arrive
with 2022–2025-era configs targeting v3–v4.3, not v4.4.1.

**Every R snippet in the guide MUST cite the real file it came from.** No snippet may rest on
recollection of the R API.

### Verified findings

Counts are **point-in-time, from one config** (`osmose-ben.R`, 2026-07-17). The guide presents
them as a dated exhibit, never as figures to match; the reader's instruction is always "run it
on *your* config."

1. **osmopy parses the R config dialect as-is.** `osmose-ben.R` is `key = value` lines in a
   `.R` file. `SEPARATORS` is `[=;,:\t]` and `#` is a comment char (`config/reader.py:70-71`),
   so it parses with **0 skipped lines**, yielding **844 keys from the file** (`len()` says
   845; `_osmose.config.dir` is injected by `reader.py:91` — quote **844**, since a reader
   greps and counts 844).
2. **"0 skipped" ≠ "clean parse" — this is the thesis in miniature.** The same parse emits 2
   diagnostics: a **failed** sub-config resolution (`missing_subconfig` for
   `input/initial_conditions.osm`) and a `duplicate_key` at line 870. So 844/236 are "keys
   reachable *without* that file" — the counts carry this contingency inline.
3. **The 4.4.0 shim migrates 8 legacy keys — and half land on keys nothing reads.**
   Dead: `output.restart.enabled`, `output.restart.spinup`, `output.fishery.enabled` →
   `output.fisheries.enabled`, `economy.enabled` → `module.bioeconomics.enabled`.
   Live: `fisheries.enabled` (`engine/config.py:2032`), `simulation.bioen.enabled` (:2365),
   `simulation.genetic.enabled` (:2422), `population.initialization.relativebiomass.enabled`
   (:538). Same mechanism, same config, **identical surface behavior**. This is the single
   most useful thing §2 can teach.
4. **844 parsed ≠ 844 supported: 236 unknown** under `validation.strict.enabled=error`.
   "Your config loads" is true *and is the trap*.
5. **`surveys.*` unsupported — 21 keys.** No `surveys.` support in Python source. The in-tree
   Benguela port (PR #100) dropped the block; `data/benguela/benguela_all-parameters.csv` has
   0 such lines. Flagged by strict mode, so **loud but opt-in**. (Say "no support in the
   Python engine", not "anywhere" — `data/eec_full` has one *commented-out* line.)
6. **Python-engine restart is silently ignored.** Allowlisted in `config_validation.py`
   (marked "Java-side"); `engine/initialization.py` exposes only `build_initial_population` /
   `age_structured_population`. Loads clean, validates clean, does nothing. Works on Java.
   → [#120](https://github.com/razinkele/osmopy/issues/120).
7. **"Allowlisted but unread" is a whole class.** Two members verified real against the R
   corpus (below). The class is larger but **could not be safely enumerated** — see the
   retraction note. → [#121](https://github.com/razinkele/osmopy/issues/121).
8. **Phased calibration is NOT a gap.** `calibration/multiphase.py` has `CalibrationPhase` +
   `MultiPhaseCalibrator` with calibrar's exact semantics ("Output of phase N becomes fixed
   params for phase N+1"). Difference is plumbing: calibrar reads a `parphase` CSV column;
   osmopy constructs phases in code.
9. **The Java engine is a real fallback for the Java-side gaps** — verified against the
   vendored jars, not inferred. `fr/ird/osmose/output/Surveys.class` is in **both** 4.3.3 and
   4.4.1; restart is in 4.4.1 via `SchoolSetSnapshot` / `ModularSchoolSetSnapshot` plus
   populator strings `simulation.restart.file`, `isRestart`. Had surveys been dropped in 4.4,
   the guide's prescribed workaround would have been a dead end.

## Gap taxonomy

Gaps fail in **seven directions** — six needing a workaround, plus value coercion (latent; one
sentence, no remedy). An earlier draft claimed three and called the taxonomy complete; two
fresh-eyes rounds found four more. **Assume it is still incomplete.**

| Bucket | Signal | Example | Reader's move |
|---|---|---|---|
| Capability absent | silent | `simulation.restart.enabled` | Java engine |
| Key renamed | silent | `output.tl.enabled` → engine reads `output.meantl.enabled` | Use the right key |
| Shim → dead key | silent | `economy.enabled` → `module.bioeconomics.enabled` | Set the engine's key |
| Unsupported module | loud, opt-in only | `surveys.*` (21 keys) | Java engine |
| Missing sub-config | silent, **invisible to strict mode** | a referenced file that isn't there | `scripts/check_config.py` |
| Cross-file collision | silent, no diagnostic | sub-config overrides master | `check_config.py` won't catch it either |
| Value coercion | silent | `= 1` / `yes` / `T` → `False` | Use `true`/`TRUE` |

"Recognized", "implemented", "implemented under another name", and "present in the file you
think you read" are four different sets, and **no default-mode signal distinguishes any**.

Two, newly verified, change the guide's advice:

- **Missing sub-config is invisible to strict mode, and silence scales with damage.**
  `validate()` sees the already-flattened dict; reader diagnostics live on the reader. A
  config referencing a missing sub-config passes `validate(mode="error")` **clean**. Perversely,
  the missing file's keys never arrive — so strict mode sees *fewer* unknowns and is *more*
  likely to pass. This is the argument for §2's two-tool prescription.
- **Cross-file collisions are silent.** `reader.py:163` builds `seen_keys` per file, so
  `duplicate_key` fires only *within* a file; `_read_recursive` then does `flat.update()`
  across files with no diagnostic. **Sub always beats master, independent of where the
  reference sits** (parent written first in a depth-first walk; among siblings, last-referenced
  wins). OSMOSE configs are split by design, so this is live for every reader.
  ⚠️ **Verify Java's precedence before publishing this rule** — it was not checked, and
  publishing a rule that differs from the engine the reader is migrating *from* would be worse
  than saying nothing.

**Non-gap worth showing for contrast:** some mismatches fail *loudly* — `species.lw.*` without
its `species.length2weight.*` twin raises `KeyError: Required OSMOSE config key missing:
'species.length2weight.condition.factor.sp0'`, naming the exact right key. Show one, so the
reader learns a crash is the *friendly* case and silence is the dangerous one.

### The two verified traps

| R key (provenance) | What happens | Python key actually read |
|---|---|---|
| `output.tl.enabled` (7 R files, case-insensitive; real 4.4.1 jar string) | Loads clean, validates clean, mean-TL output silently absent | `output.meantl.enabled` (osmopy name; 0 R files, 0 jars) |
| `economy.enabled` (`osmose-ben.R:1048`) | Shim → `module.bioeconomics.enabled` (**the correct upstream 4.4.1 key**) → validates clean → engine never runs economics | `simulation.economic.enabled` (osmopy invention; 0 R files, 0 jars) |

**On `economy.enabled`, say which engine is at fault.** `module.bioeconomics.enabled` has 2
hits in the 4.4.1 jar (including `Releases$15`, upstream's own 4.4.0 renames table) and 0 in
4.3.3 — the signature of a genuine rename target. `simulation.economic.enabled` has **0 hits in
either jar and 0 in the R corpus**. The shim is **correct**; osmopy's engine invented a key
(`engine/config.py:2431`). So: `simulation.economic.enabled` works only on Python;
`module.bioeconomics.enabled` is right for Java. Do **not** write "the shim betrays you".

Two things not to write, both from claims that died on execution:
- **Not** "the UI lights up an Economic page." `engine_capabilities.py:32` is the key's only
  consumer and feeds one Run-page **"Will populate:"** label (`ui/pages/run.py:797`).
  `ui/pages/economic.py` gates on `engine_mode`, and honestly says the module isn't
  implemented. True claim: the key adds "Economic" to a label that then doesn't populate.
- **`canonicalize_config` returns a tuple** — `({...}, ['economy.enabled'])` — and injects
  `osmose.version`. Snippets must reproduce.

### Retraction note — do not rebuild this

**A prior draft proposed an 8-row "key-rename table" as the guide's centerpiece. It was
fabricated and is retracted.** It was derived from the `config_validation.py` allowlist and
published under the header *"Key an R/Java config sets"* without checking the R corpus. **7 of
8 rows appear in zero R config files and zero jars** — they came from `data/examples`, osmopy's
own bundled config. For three rows the prescribed "rename to" target (`output.size.enabled`,
`output.recordfrequency.ndt`) is a key R configs **already set**. The two `species.lw.*` rows
were doubly wrong: not R keys, and they fail loudly.

**The general set cannot be safely enumerated.** Cheap derivations fail in **both** directions:
a `startswith` read looks unread to a literal grep (`movement.species.map{idx}` is read at
`engine/movement_maps.py:129`), while an allowlisted key looks read to a validator. Scale: the
R corpus sets ~2,036 distinct tokens; `validate()` calls **910** unknown, normalizing to **271**
families — far past hand-verification.

**Therefore:** ship only traps verified on **both** sides; defer the general case to #121's
tooling. A static table in a doc is the wrong shape for this — the fix is an actionable warning
naming the correct key.

### Caveats — not clean renames, publish no mapping

- `temperature.*` / `oxygen.*` (filename, varname, nsteps.year, factor, offset) → the Python
  engine has **constant-only** forcing (`temperature.value` / `oxygen.value`, gated on
  `bioen_enabled`). A capability **downgrade**, not a rename. Presenting `temperature.value` as
  equivalent to a NetCDF field would be actively misleading.
- `predation.accessibility.stage.*` → Python derives stages from the accessibility CSV's column
  labels (`engine/accessibility.py:50`). Different mechanism. **No mapping.**

### Verified non-gaps — do not list these

- `output.biomass.enabled` / `output.abundance.enabled` / `output.yield.biomass.enabled` are
  unread, but `osmose/engine/output.py` writes those CSVs unconditionally (biomass/abundance
  loop ~46-47, `_write_yield_csv` ~60). Output appears anyway. Cite the **full path** —
  `osmose/schema/output.py` also matches a bare `output.py`. One sentence worth adding: the
  *disable* direction silently doesn't work either (`= FALSE` still yields the CSV).
- `evolution.trait.*` is genuinely read (`engine/genetics/trait.py:49-77`).

## R API surface to cover

Read from the corpus, not recalled:

| R symbol | Source |
|---|---|
| `runOsmose(input, version=, osmose=)` (legacy camelCase) | `osmose-gog/run.R` |
| `run_osmose(input=, output=, osmose=, version=)` | `osmose-ben/launcher.R` |
| `read_osmose(path=, version=)` | `osmose-ben/launcher.R`, `osmose-gog/runModel.R` |
| `read_osmose(...)` → `$biomass`, `$yield` | `osmose-gog/runModel.R` **only** (launcher uses `plot()`/`get_var()`) |
| `get_var(obj, what=, how="list")` | `osmose-ben/launcher.R` |
| `plot(obj, what=, initialYear=, freq=, col=, lwd=)` | `osmose-ben/launcher.R` |
| `initialize_osmose(input=, file=, output=, type="climatology"\|"ncdf", run=)` | `osmose-ben/launcher.R` |
| `.readConfiguration()`, `.getPar()` (internal, dot-prefixed) | `osmose-ben/launcher.R` |
| calibrar: `getCalibrationInfo`, `getObservedData`, `createObjectiveFunction`, `calibrate(phases=, control=)` | `osmose-gog/calibrate.R` |
| user-written `runModel(param, names, ...)` | `osmose-gog/runModel.R` |

`initialize_osmose`, `.readConfiguration` and `.getPar` are **appendix-table only** — no body
section. `initialize_osmose` maps to finding 6 (no Python restart; Java engine); the
dot-prefixed pair are R-internal and have no counterpart by design.

## Document structure

Six sections plus an appendix, in the reader's actual order.

1. **Should you switch?** (~½ page) Gains: no JVM, faster on every benchmarked config,
   calibration stack (NSGA-II / CMA-ES / surrogate-DE / Pareto), Shiny UI. Losses: no surveys
   module, no Python-engine restart, temperature/oxygen forcing downgrades to constant-only, no
   `plot()` one-liners. The Java engine remains available and is the fallback for the
   capability-absent and unsupported-module gaps (finding 9). Renamed keys need no fallback —
   they need the right key name.

2. **Your config already loads — and that's the trap.** Opens with the Benguela exhibit, dated
   and framed as one example, its contingency **inline**: 844 keys, 0 skipped, **236 unknown**,
   and a sub-config resolution that *failed* — so these are keys reachable without that file,
   not the config's key count. That isn't a caveat on the exhibit; it *is* the exhibit.

   Prescribes a **two-tool** first action, in order:
   - **`scripts/check_config.py` first** — the *only* production caller of `format_diagnostics`
     / `diagnostics_have_errors`. Surfaces parse-level damage: missing sub-configs, duplicate
     keys, unparseable lines. Neither `osmose/cli.py` nor the UI reads `reader.diagnostics`.
     (A missing sub-config *does* also emit a `_log.warning` at `reader.py:142` — easy to miss,
     but say "easy to miss", not "silent".)
   - **then `validation.strict.enabled=error`** — key-level: what osmopy doesn't recognize.

   Then the beat that matters: **neither is sufficient, and strict mode is the weaker one.** It
   catches `surveys.*` but is silent on restart, renames, missing sub-configs and cross-file
   collisions — because those keys are *known*, or never arrive. Teach the taxonomy, show the
   two verified traps, and state plainly: a clean strict-mode run means **nothing was
   unrecognized**, not that the config works.

3. **Run.** `runOsmose()` / `run_osmose()` beside `PythonEngine().run()` / `.run_in_memory()`.
   The JVM disappears. → `usage-guide.md` §1.

4. **Read & plot.** `read_osmose()` → `$biomass`/`$yield`, `get_var()` beside `OsmoseResults`.
   Honest: R's `plot(obj, what=…)` one-liners have no single equivalent; point at the plotting
   module and the UI. → `usage-guide.md` §2.

5. **Calibrate** (largest). calibrar's idiom — hand-written `runModel(param, names, ...)` that
   writes params to CSV, runs, reads, returns a named list — beside osmopy's objective/problem
   model. The *shape* differs though the capability is present.

   **Every calibrar symbol in the API table gets a named counterpart or an explicit "no
   equivalent, do X".** Starting map, **to be verified against each module before writing**
   (this is the plausible mapping, not a verified one — and this spec's central lesson is what
   happens when those are confused):
   `calibrate(phases=)` → `MultiPhaseCalibrator` / `CalibrationPhase`; `control$popsize` /
   `control$maxgen` → optimizer args; `getCalibrationInfo` / `getObservedData` →
   `calibration/targets.py` + `objectives.py`; `createObjectiveFunction(aggFn=, aggregate=)` →
   `calibration/losses.py` + `problem.py`; user-written `runModel` → **no counterpart by
   design** (osmopy owns the run/read loop; the user supplies parameters and a loss).
   → `usage-guide.md` §4.

6. **Verify your port — as TWO comparisons.** A naive "run both engines, compare biomass"
   conflates two variables, since the reader has a v3/v4.3-era config and the default jar is
   4.4.1. A mismatch would be unattributable.
   - **Step 1 — isolate the port.** Re-run the reader's *original* jar through osmopy's Java
     engine (`OsmoseRunner.__init__` accepts an arbitrary jar path, `runner.py:123`). Same
     engine, same config, new driver. Any difference is the port.
   - **Step 2 — isolate the engine.** Then compare against Python / default 4.4.1. Any
     difference is the engine or version.

   Set tolerance honestly: `usage-guide.md` §6 documents Python-vs-Java agreeing only "within 1
   order of magnitude" on bundled, verified configs — do not promise bit-equality. Lean on
   `docs/parity-roadmap.md`.

**Appendix:** the R→Python symbol table; the **two verified traps** (deliberately two rows, not
a table of plausible ones); an honest gaps list — `surveys.*`, Python-engine restart,
temperature/oxygen downgrade, `plot()` convenience — each with its workaround (Java engine for
the first two; **none** for the third, say so; plotting module / UI for the fourth).

The two traps appear in §2 (taught in prose) and the appendix (reference). That duplication is
intentional — teach then reference — and is the only content duplicated on purpose.

## Boundaries

The guide's unique content is **the mapping, the traps, and the verification**. Python mechanics
live in `docs/usage-guide.md` (251 lines: run → read → compare → calibrate) and are linked,
never restated.

**The rule that makes this actionable** (§§3–5 must show *some* Python — a mapping with no
target is useless): show the **call signature only**. No parameter walkthroughs, no output
samples, no runnable end-to-end snippets. **If a code block would still make sense with the R
side deleted, it belongs in usage-guide, not here.**

## Keeping the claims true

Three tiers. The spec must not promise more than each delivers.

**Tier 1 — mechanism, pinned by a fixture.** A **synthetic ~20-line R-dialect fixture**,
hand-written, not vendored (avoids vendoring GPL-3.0 content). Exercises: `=` separators, `#`
comments, `TRUE`/`FALSE`, a pre-4.4.0 key the shim migrates, a `surveys.*` key (asserted
unknown), a `simulation.restart.*` key (asserted **not** unknown). Pins that the phenomena
exist.

**Tier 2 — the two verified traps, pinned by a TWO-SIDED assertion.** An earlier draft proposed
"the R key leaves the attribute at its default; the mapped key flips it." **That is vacuous** —
it passes for `banana.enabled`, i.e. any key osmopy doesn't read, real or invented. It would
have shipped all seven fabricated rows green. It pins the *Python* half; the wrong half was the
*R* half. So:
- **Python side:** the R key leaves the attribute at default; the mapped key **changes** it
  (not "flips" — some targets are ints, not bools).
- **Provenance side:** the R key must be greppable in a **named upstream file**, carried as a
  citation in the fixture. A row whose R key appears in no real config cannot pass.

  ⚠️ Honest limit: since we don't vendor R configs, this asserts the *citation is present*, not
  that it is *true*. It forces every row to name a file instead of guessing from the allowlist —
  a real improvement — but re-verification when the guide is next edited remains a human step.
  Do not let the test's existence imply otherwise.

**Tier 3 — could be pinned, DECLINED with reasons.** The jar-classfile claims (`Surveys.class`
in both jars) **are** pinnable — the jars are vendored and `zipfile.ZipFile(...).namelist()`
settles it offline in four lines. We decline: the failure mode (a future jar bump drops
`Surveys.class` and invalidates the workaround) is real but rare, and a test that unzips jars to
protect a doc sentence buys little. Mitigate with a comment at the jar-version sites — there is
no single site: `config/aliases.py:230` (`DEFAULT_TARGET_VERSION`), `runner.py:123`, plus
version strings in `demo.py`, `runner.py`, `calibration/problem.py`.

The Benguela counts (844/236/21/8) are **genuinely un-pinnable** — a one-time read of an
unvendored upstream repo — hence §2's dated-exhibit framing.

Do not claim CI protects what it cannot; and do not call something unprotectable when the truth
is it wasn't worth protecting. Say which.

## Deliberately excluded

- **#120** (silent restart) and **#121** (allowlisted-but-unread class, wrong allowlist
  comments, the `economy.enabled` engine-invented key, dead toggles in `data/examples`) are
  filed separately. This guide documents the terrain; fixing it is an engine change with its own
  test surface. The two verified traps are the reader's workaround **until** #121 lands. If #121
  lands first — particularly its actionable warning naming the correct key — §2 and the appendix
  shrink to a pointer at that warning.
- **No R package install.** Claims are grounded by reading real driver scripts. If a future
  claim needs runtime R behavior, install for that one question.
- **No vendored R config.** See Tier 1.

## Success criteria

- A reader whose config sets `economy.enabled` learns — from the guide, not a support thread —
  that their run has no economics on the Python engine even though the key migrated to
  upstream's correct 4.4.0 name, and that "Will populate: Economic" is a promise it won't keep.
- A reader whose config references a missing sub-config finds out, because the guide sent them
  to `check_config.py` and not to strict mode alone.
- All gap buckets are stated with their differing workarounds; the temperature/oxygen downgrade
  says plainly it has none.
- Every R-side key named is greppable in a named upstream file. **No row exists because the
  allowlist mentioned it.**
- Every R snippet cites a real file in a real repo.
- No Python mechanics restated from `usage-guide.md`.
- Nothing claimed as CI-protected that isn't; nothing called unprotectable that was merely
  declined.
- The fixture passes and would fail if any **Tier 1 or Tier 2** claim stopped being true.
