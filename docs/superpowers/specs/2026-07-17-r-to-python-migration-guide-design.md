# R OSMOSE → Python migration guide — design

**Date:** 2026-07-17
**Status:** Approved, ready for planning
**Deliverable:** `docs/r-to-python-migration.md` (one file, MyST Markdown, added to the `Guides` toctree)

## Purpose

A user-facing guide for an **existing R OSMOSE user with a working config and R driver
scripts who has decided to switch to osmopy** ("port my model"), prefaced by a short honest
"should you switch?" framing for readers who have not yet committed.

Out of scope: newcomers translating R papers (subsumed), and R⇄Python interop (outputs are
plain CSV/NetCDF; interop is near-free and needs no guide).

## Method — read this before writing a line of the guide

This spec went through four review rounds. The pattern in what survived and what didn't is
sharp enough to be the implementer's working rule, because it is also the guide's thesis:

> **Every claim that survived is one where someone ran the code. Every claim that failed is
> one that reasoned from a grep hit, a line number, or an allowlist entry to a runtime
> behavior.**

The failures were not careless — they were plausible. A retracted 8-row table reasoned from
*the allowlist* to "R users set these keys" (7 of 8 didn't exist). A retracted claim reasoned
from `engine_capabilities.py:32` to "the UI renders an Economic page" (it doesn't). A
retracted claim reasoned from `schema/output.py`'s flag list to "something reads this" (nothing
does). Each was checkable in one command, and each survived a self-review that only re-read
the prose.

**An allowlist entry is not evidence of a read. A line number is not evidence of a behavior.
A schema declaration is not evidence of a consumer.** Before any claim enters the guide, run
the thing. This costs seconds and is the only defense that has actually worked here.

## Evidence base

All R-side claims are grounded in the four public config repos, read at source rather than
recalled. Verified 2026-07-17 by cloning at depth 1:

| Repo | Driver scripts used as ground truth |
|---|---|
| `osmose-model/osmose-gog` | `run.R`, `runModel.R`, `calibrate.R` |
| `osmose-model/osmose-ben` | `launcher.R`, `osmose-ben.R` (config), `utils/ben-plots.R` |
| `osmose-model/osmose-gom` | `dynamics_gom.R`, `analysis.R` |
| `osmose-model/osmose-eec` | `convert-ltl.R`, `eec_all-parameters.csv` |

Context: the `osmose` R package was removed from CRAN on 2022-10-03 (unfixed check issues);
current v4.x installs from GitHub or the project drat repo. Readers therefore arrive from
2022–2025-era configs targeting OSMOSE v3–v4.3, not v4.4.1.

Every R snippet in the guide MUST cite the real file it came from. No snippet may rest on
recollection of the R API.

### Verified findings that shape the guide

Measured against the real `osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R`:

All counts below are **point-in-time observations from one real config**, measured
2026-07-17 against `osmose-ben.R` at the cloned default branch. They are illustrative — they
prove the phenomenon, they are not normative constants. The guide MUST present them as a
dated example ("in one real config…"), never as figures the reader should expect to match.
The reader's actual instruction is always *"run strict validation on your own config"*. This
is deliberate: these numbers cannot be regression-pinned (see "Keeping the claims true"), so
the guide must not lean on them as if they could.

1. **osmopy parses the R config dialect as-is.** `osmose-ben.R` is not R code; it is
   `key = value` lines in a `.R` file. `OsmoseConfigReader.SEPARATORS` is
   `re.compile(r"\s*[=;,:\t]\s*")` and `COMMENT_CHARS = {"#", "!"}`, so the file parses with
   **0 skipped lines** and yields **844 keys from the file** (`len()` reports 845; one entry,
   `_osmose.config.dir`, is injected by `reader.py` and is not in the file — the guide must
   quote **844**, since a reader grepping their own config will count 844).
2. **"0 skipped lines" is not "a clean parse" — and that is the thesis in miniature.** The
   same parse emits 2 diagnostics: a *failed* sub-config resolution (`missing_subconfig` for
   `input/initial_conditions.osm`, referenced by `osmose.configuration.initialization`) and a
   `duplicate_key` at line 870 (`output.weight.enabled`). Recursive sub-config resolution was
   attempted and **failed**; had the file been present the counts would shift. The guide
   should cite this as its opening exhibit rather than claim a clean parse.
3. **The 4.4.0 compat shim auto-migrates legacy keys — but migration is not a guarantee of
   arrival, and the guide must not present it as one.** 8 deprecated keys migrated on
   Benguela. **Four of the 8 — half — migrate into keys nothing reads:**
   `output.restart.enabled`, `output.restart.spinup` (finding 6), `output.fishery.enabled` →
   `output.fisheries.enabled`, and `economy.enabled` → `module.bioeconomics.enabled` (see the
   taxonomy). The other four reach the engine: `fisheries.enabled` (`config.py:2032`),
   `simulation.bioen.enabled` (`config.py:2365`), `simulation.genetic.enabled`
   (`config.py:2422`), `population.initialization.relativebiomass.enabled` (`config.py:538`).
   So the same mechanism, on the same config, rescues half and strands half, with **identical
   surface behavior**. That is the single most useful thing §2 can teach.
4. **844 parsed ≠ 844 supported: 236 keys are unknown to osmopy.** The reader is
   permissive; `EngineConfig.from_dict` under `validation.strict.enabled=error` is what
   reports the truth. This reframes the guide: "your config loads" is true *and is the trap*.
   (236 is unaffected by the injected key; verified against a `_`-stripped dict.)
5. **`surveys.*` is unsupported — 21 keys.** `osmose-ben.R` defines a surveys module
   (`surveys.name.sr1 = acousticSurvey`, selectivity, survey movement maps). osmopy has no
   `surveys.` support in any Python source. The in-tree Benguela port (PR #100) dropped the
   block entirely — `data/benguela/benguela_all-parameters.csv` contains 0 `surveys.` lines.
   Flagged as unknown by strict validation, so this gap is **loud but opt-in**. (Say "no
   support in the Python engine", not "anywhere": `data/eec_full/eec_all-parameters.csv`
   carries one *commented-out* `#osmose.configuration.surveys;…` line. Not support, but a
   pedantic reader will find it.)
6. **Python-engine restart is silently ignored.** `simulation.restart.enabled` and siblings
   are allowlisted as valid in `osmose/engine/config_validation.py` (marked "Java-side"),
   but the Python engine does not implement restart — `osmose/engine/initialization.py`
   exposes only `build_initial_population` / `age_structured_population`. The key loads
   clean, validates clean (it is **not** reported unknown), and does nothing. This gap is
   **silent**. Restart works on the Java engine. Tracked as
   [#120](https://github.com/razinkele/osmopy/issues/120).
7. **Restart is not alone: "allowlisted but unread" is a whole CLASS.** A sweep of the
   `config_validation.py` supplementary allowlist found the Python engine sometimes *has* the
   capability but reads a **different key name** — so the R-style key loads clean, validates
   clean, and the feature silently defaults off. **Two members are verified real against the
   R corpus** (`output.tl.enabled`, `economy.enabled` — see the taxonomy). The class is
   almost certainly larger, but its full extent **could not be safely enumerated** (see the
   taxonomy's retraction note: the R corpus's 910 strict-mode unknowns normalize to 271
   families, and cheap derivations fail in *both* directions — `startswith` reads look unread,
   while allowlisted keys look read). The guide ships the verified two and defers the general
   case to tooling.
   Tracked as [#121](https://github.com/razinkele/osmopy/issues/121).
8. **Phased calibration is NOT a gap.** `osmose/calibration/multiphase.py` provides
   `CalibrationPhase` + `MultiPhaseCalibrator` with calibrar's exact semantics: "Output of
   phase N becomes fixed params for phase N+1." The difference is plumbing — calibrar reads
   phases from a `parphase` CSV column; osmopy constructs them in code.

Gaps fail in **seven different directions** — six that need a workaround, plus value coercion,
which is latent and gets one sentence rather than a remedy. An earlier draft claimed three and
asserted the taxonomy was complete; two fresh-eyes rounds found four more, all verified.
Assume it is still incomplete. Teaching the reader to tell the buckets apart is the guide's
core safety contribution — they differ in both signal and remedy:

| Bucket | Signal | Example | What the reader does |
|---|---|---|---|
| **Capability absent** | silent | `simulation.restart.enabled` | Use the Java engine |
| **Key renamed** | silent | `output.tl.enabled` → engine reads `output.meantl.enabled` | Rename — capability is there |
| **Renamed *by the shim* into a dead key** | silent, **and the UI confirms the false belief** | `economy.enabled` → `module.bioeconomics.enabled` (dead); engine reads `simulation.economic.enabled` | Set the engine's key directly |
| **Unsupported module** | loud, but only under opt-in strict mode | `surveys.*` (21 keys) | Use the Java engine |
| **Missing sub-config** | silent — **and invisible to strict mode** | `osmose.configuration.initialization` → a file that isn't there | `scripts/check_config.py` |
| **Cross-file key collision** | silent — no diagnostic at all | sub-config's value silently overrides master's | `scripts/check_config.py` won't catch it either — see below |
| **Value coercion** | silent | `output.x.enabled = 1` / `yes` / `T` → `False` | Use exactly `true`/`TRUE` |

"Recognized", "implemented", "implemented under a different name", and "present in the file
you think you read" are four different sets, and **no default-mode signal distinguishes any
of them**.

Two of these are newly verified and change the guide's advice:

- **Missing sub-config is invisible to strict mode, and silence scales with damage.**
  `validate()` operates on the already-flattened dict; reader diagnostics live on the reader
  object. A config referencing a missing sub-config passes `validate(mode="error")` **clean**
  — verified. And perversely: the missing file's keys never reach the dict, so strict mode
  sees *fewer* unknowns and is *more* likely to pass. The worse the damage, the quieter the
  result. This is the single strongest argument for the two-tool prescription in §2.
- **Cross-file collisions are silent and undocumented.** `reader.py` builds `seen_keys`
  fresh per file, so `duplicate_key` only fires *within* one file; `_read_recursive` then
  does `flat.update(...)` across files with no diagnostic. Verified: master `TRUE`, sub
  `FALSE` → final `FALSE`, diagnostics `[]`. Sub-config beats master. OSMOSE configs are
  split across sub-files **by design**, so this is live for every real reader. The precedence
  direction is untested and undocumented upstream of this spec; the guide should state it and
  #121 should cover the missing diagnostic.

Value coercion is **latent, not live**: R's `TRUE`/`FALSE` do work — `_enabled` is
`cfg.get(key,"false").lower() == "true"` (`config.py:169`), so case is handled. Do **not**
write "the R corpus uses uppercase exclusively" (an earlier draft did; it is checkably
false). The corpus is **mixed and splits by file type**: `.R` dialect files use uppercase,
`.csv` param files use lowercase (`osmose-gog/osm_param-output.csv:43` →
`output.TL.enabled;true`). Both work. Mention in one sentence; do not dramatize.

One coercion case is **live and worth the sentence**, though: `= 1` reads as **enabled in the
UI** (`engine_capabilities.py:25` accepts `("true","1")`) and **disabled in the engine**
(`config.py:169` accepts only `"true"`). Same config, two answers.

Note also a **non-gap direction worth naming for contrast**: some mismatches fail *loudly*
with a `KeyError` naming the exact correct key (e.g. `species.lw.*` without its
`species.length2weight.*` twin). That is the best-case outcome. The guide should show one, so
the reader learns that a crash is the *friendly* case and silence is the dangerous one.

9. **The Java engine is a genuine fallback for the two Java-side gaps — verified against the
   jars**, not
   inferred from the fact that `osmose-ben` ran surveys on 4.3.3. Confirmed by inspecting
   `osmose-java/` directly:
   - `surveys.*`: `fr/ird/osmose/output/Surveys.class` is present in **both**
     `osmose_4.3.3-jar-with-dependencies.jar` and `osmose-4.4.1-jar-with-dependencies.jar`
     (osmopy's default jar). Surveys were not dropped in 4.4.
   - `restart`: implemented in 4.4.1 via `fr/ird/osmose/output/SchoolSetSnapshot.class` and
     `ModularSchoolSetSnapshot.class`, with the populator carrying live strings
     `simulation.restart.file`, `isRestart`, `"Failed to open restart file"`.

   This matters because "use the Java engine" is the workaround the guide prescribes for
   both gaps; had surveys been removed in 4.4, that advice would have been a dead end.

## Gap taxonomy — the known specific traps

**A prior draft of this spec proposed an 8-row "key-rename table" as the guide's centerpiece.
It was wrong and is retracted.** Recording why, because the failure is instructive and a
future implementer will otherwise rebuild it:

The table was derived from the `config_validation.py` allowlist and then published under the
header *"Key an R/Java config sets"* — without checking the R corpus. Measured: **7 of its 8
rows appear in zero R config files and zero jars.** They came from
`data/examples/osm_param-output.csv` — osmopy's *own* bundled example. Worse, for three rows
the prescribed "Python key to use instead" (`output.size.enabled`, `output.recordfrequency.ndt`)
is a key R configs **already set** — the advice told readers to rename a key they don't have
into one they already have. The two `species.lw.*` rows were doubly wrong: not R keys, and
they fail **loudly** (`KeyError: Required OSMOSE config key missing:
'species.length2weight.condition.factor.sp0'`, naming the exact correct key) rather than
silently, so they never belonged in a silent-gap table at all.

**The general set cannot be safely enumerated.** Two independent mechanisms defeat a cheap
derivation, in **opposite directions**, which is why no single grep settles it:

- **False positives** — a key absent from the source as a string literal may still be read.
  Indexed families can be consumed via `startswith`: `movement.species.map{idx}` is read at
  `osmose/engine/movement_maps.py:129` (`if key.startswith("movement.species.map")`), so a
  literal grep calls it unread when it is not. (An earlier draft illustrated this with
  `fisheries.movement.file.map1` — **a self-refuting choice**: the engine reads *only* the
  literal `map0` (`config.py:2095`), with no `{idx}` loop, so `map1`+ really are silently
  ignored. It is a *true* positive and a textbook member of finding 7's class, offered as an
  example of a false one. The `{idx}` entry in the allowlist is hand-maintained, not derived
  from a read — which is the whole trap.)
- **False negatives** — the allowlist marks keys valid that nothing reads, so "validates
  clean" proves nothing.

Scale, by the honest numbers: the R corpus sets ~2,036 distinct key-like tokens; `validate()`
reports **910** of them unknown, normalizing to **271** families. (An earlier draft cited
"~993"; it does not reproduce under any method and is withdrawn — the figure to quote, with
its method, is 910 raw / 271 normalized.) Either number is far past hand-verification.

Every row therefore requires hand-verification **at its consumption site** — not at a grep
hit, not at an allowlist entry — **and** provenance in a named upstream file. There is no
cheap version of this table, and a plausible-looking cheap version is worse than none: it is
precisely the failure the guide exists to warn about, committed by the guide.

**Therefore:** the guide ships only the traps verified real on both sides, and points at
[#121](https://github.com/razinkele/osmopy/issues/121) for the general case. The real fix is
tooling — an actionable warning naming the correct key — not a static table in a doc that
rots. Two rows qualify today:

| R key (provenance) | What happens | Python key actually read |
|---|---|---|
| `output.tl.enabled` (7 R config files, case-insensitive; real 4.4.1 jar string) | Loads clean, validates clean, mean-TL output silently absent | `output.meantl.enabled` (an osmopy name; 0 R files, 0 jars) |
| `economy.enabled` (`osmose-ben.R:1048`) | Shim rewrites → `module.bioeconomics.enabled` (**the correct upstream 4.4.1 key**) → validates clean → engine never runs economics | `simulation.economic.enabled` (an osmopy invention; 0 R files, 0 jars) |

**The `economy.enabled` story is the opposite of what an earlier draft claimed, and the
correction matters because it changes who is at fault.** Verified against the vendored jars:
`module.bioeconomics.enabled` has 2 hits in 4.4.1 (including `Releases$15`, upstream's own
4.4.0 renames table) and 0 in 4.3.3 — exactly the signature of a genuine 4.4.0 rename target.
`simulation.economic.enabled` has **0 hits in either jar and 0 in the R corpus**. So the shim
is **correct**; the defect is that osmopy's Python engine invented a key
(`osmose/engine/config.py:2431`). The guide must therefore say *which engine*:
`simulation.economic.enabled` works only on the Python engine, while
`module.bioeconomics.enabled` is right for Java. Do **not** write "the shim betrays you" —
it doesn't, and #121 was filed with that wrong framing and has been corrected in a comment.

Two further corrections an earlier draft got wrong by reasoning from a citation instead of
running the code — recorded because the guide's whole thesis is that this is how you get hurt:
- **The UI does not "light up an Economic page".** `engine_capabilities.py:32` is the key's
  only consumer, and it feeds one Run-page **"Will populate:"** label (`ui/pages/run.py:797`).
  `ui/pages/economic.py` gates on `engine_mode != "python"`, not on the key, and its body
  honestly says the module isn't implemented yet. The true claim is narrow: the key adds
  "Economic" to a label that then doesn't populate. Write that, not the dramatic version.
- **`canonicalize_config` returns a tuple**, `({...}, ['economy.enabled'])`, and injects
  `osmose.version`. Any snippet in the guide must reproduce; the earlier draft's
  dict-equality shorthand would not have.

`economy.enabled` is **not** the only dead shim-migrated flag:
`fisheries.enabled` → `module.multispecies.fisheries.enabled` (`config.py:2032`),
`simulation.bioen.enabled` (`config.py:2365`) and `simulation.genetic.enabled`
(`config.py:2422`) all reach the engine — but `output.fishery.enabled` →
`output.fisheries.enabled` is dead too (allowlist + `schema/output.py:138`'s
`_OUTPUT_ENABLE_FLAGS`, a declaration list, not a read). See finding 3: it is **4 of 8**.

Two further classes that are **not** clean renames and must be written as caveats, not swaps:

- `temperature.{filename,varname,nsteps.year,factor,offset}` and the `oxygen.*` equivalents →
  the Python engine has **constant-only** forcing (`temperature.value` / `oxygen.value`,
  gated behind `bioen_enabled`). This is a capability *downgrade*, not a rename. Presenting
  `temperature.value` as the equivalent of a NetCDF field would be actively misleading.
- `predation.accessibility.stage.*` → Python derives stages from the accessibility CSV's
  column labels (`accessibility.py:50`), a different mechanism entirely. **Publish no mapping.**

Two things the guide must NOT list as gaps (verified non-gaps):

- `output.biomass.enabled` / `output.abundance.enabled` / `output.yield.biomass.enabled` are
  unread, but `osmose/engine/output.py` writes those CSVs unconditionally — the biomass and
  abundance loop at ~46-47 and `_write_yield_csv` at ~60 are all ungated, so the output
  appears anyway. (Cite the **full path**: `osmose/schema/output.py` also matches a bare
  `output.py` and its nearby lines contain a plausible-looking `simulation.restart.enabled`
  field def. This project has a documented history of file:line drift in review loops.)
  One caveat the guide should note in a sentence: the *disable* direction silently doesn't
  work either — `output.biomass.enabled = FALSE` still yields the CSV. Harmless in effect,
  but it is the same loads-clean-does-nothing phenomenon, so it is worth one line.
- `evolution.trait.*` is genuinely read (`genetics/trait.py:49-77`).

The underlying defects are tracked in [#121](https://github.com/razinkele/osmopy/issues/121)
and are **not** fixed by this guide. If #121 lands first — particularly the actionable
`validation.strict.enabled=warn` message naming the correct key — this table shrinks to a
pointer at that warning, and the guide should say so rather than duplicate it.

## R API surface to cover

Read from the corpus, not recalled:

| R symbol | Source |
|---|---|
| `runOsmose(input, version=, osmose=)` (legacy camelCase) | `osmose-gog/run.R` |
| `run_osmose(input=, output=, osmose=, version=)` | `osmose-ben/launcher.R` |
| `read_osmose(path=, version=)` | `osmose-ben/launcher.R`, `osmose-gog/runModel.R` |
| `read_osmose(...)` → `$biomass`, `$yield` (list access) | `osmose-gog/runModel.R` **only** — `launcher.R` uses `plot()`/`get_var()` instead |
| `get_var(obj, what=, how="list")` | `osmose-ben/launcher.R` |
| `plot(obj, what=, initialYear=, freq=, col=, lwd=)` | `osmose-ben/launcher.R` |
| `initialize_osmose(input=, file=, output=, type="climatology"\|"ncdf", run=)` | `osmose-ben/launcher.R` |
| `.readConfiguration()`, `.getPar()` (internal, dot-prefixed) | `osmose-ben/launcher.R` |
| calibrar: `getCalibrationInfo`, `getObservedData`, `createObjectiveFunction`, `calibrate(phases=, control=)` | `osmose-gog/calibrate.R` |
| user-written `runModel(param, names, ...)` | `osmose-gog/runModel.R` |

## Document structure

Six sections plus an appendix, ordered to match the reader's actual sequence.

1. **Should you switch?** (~½ page)
   Honest gains/losses. Gains: no JVM dependency, faster on every benchmarked config,
   calibration stack (NSGA-II / CMA-ES / surrogate-DE / Pareto explorer), Shiny UI.
   Losses: no surveys module, no Python-engine restart, NetCDF temperature/oxygen forcing
   downgrades to constant-only, no `plot()` one-liner convenience.
   States plainly that the Java engine remains available and is the fallback for the
   capability-absent and unsupported-module gaps (verified: finding 9). Renamed keys need no
   fallback — they need the right key name.

2. **Your config already loads — and that's the trap**
   Opens with the Benguela exhibit, explicitly dated and framed as one example, never as
   figures to match: 844 keys parsed, 0 skipped — **and 236 of them unknown**. The counts
   carry their contingency **inline, not as a footnote**: the same parse's sub-config
   resolution *failed*, so 844/236 are "keys reachable without `input/initial_conditions.osm`",
   not the config's key count. That is not a caveat on the exhibit; it *is* the exhibit.

   Prescribes a **two-tool** first action, in order — one tool is not enough, and this is the
   spec's most important correction to itself:
   - **`scripts/check_config.py` first** — the *only* production caller of
     `format_diagnostics` / `diagnostics_have_errors`. It surfaces parse-level damage:
     missing sub-configs, duplicate keys, unparseable lines. Neither `osmose/cli.py` nor the
     UI ever reads `reader.diagnostics`, so nothing else will tell the reader.
   - **then `validation.strict.enabled=error`** — key-level: what osmopy doesn't recognize.

   Then the crucial beat: **neither tool is sufficient, and strict mode is the weaker one.**
   It catches `surveys.*` but stays silent on restart, on every rename, on missing
   sub-configs, and on cross-file collisions — because those keys are *known*, or never
   arrive. Teaches the taxonomy, shows the two verified traps, and states plainly that a
   clean strict-mode run means **nothing was unrecognized**, not that the config works.

3. **Run**
   `runOsmose()` / `run_osmose()` beside `PythonEngine().run()` and `.run_in_memory()`.
   Notes the JVM disappears; links `usage-guide.md` §1 for mechanics.

4. **Read & plot**
   `read_osmose()` → `$biomass`/`$yield` and `get_var()` beside `OsmoseResults`. States
   honestly that R's `plot(obj, what=…)` one-liners have no single equivalent; points at the
   plotting module and the UI. Links `usage-guide.md` §2.

5. **Calibrate** (largest section)
   calibrar's idiom — hand-written `runModel(param, names, ...)` that writes params to CSV,
   runs, reads outputs, returns a named list — beside osmopy's objective/problem model.
   Explicit that the *shape* differs even though the capability is present.
   Links `usage-guide.md` §4.

   The mapping must be complete — this section is "largest" but was the thinnest-specified in
   earlier drafts, naming counterparts for only `phases` and `control$*`. Each calibrar symbol
   in the R API table gets a named osmopy counterpart **or** an explicit "no equivalent, do X
   instead". At minimum: `calibrate(phases=)` → `MultiPhaseCalibrator` / `CalibrationPhase`;
   `control$popsize` / `control$maxgen` → optimizer args; `getCalibrationInfo` /
   `getObservedData` (CSV-driven target+observation loading) → `osmose/calibration/targets.py`
   + `objectives.py`; `createObjectiveFunction(aggFn=, aggregate=)` →
   `osmose/calibration/losses.py` + `problem.py`; the user-written `runModel` →
   **no counterpart by design** (osmopy owns the run/read loop; the user supplies parameters
   and a loss, not a driver). The implementer must verify each of these against the module
   before writing — they are the plausible mapping, not a verified one, and this spec's
   central lesson is what happens when those are confused.

6. **Verify your port** — as **two** comparisons, not one.
   A naive "run both engines, compare biomass" conflates two independent variables, because
   the reader arrives with a v3/v4.3-era config while osmopy's default jar is 4.4.1: (a) did
   my config port correctly, and (b) did the engine version change my results? A mismatch
   would be unattributable. So the guide prescribes:
   - **Step 1 — isolate the port.** Re-run the reader's *original* jar through osmopy's Java
     engine (`OsmoseRunner` accepts an arbitrary jar path). Same engine, same config, new
     driver. Any difference here is the port.
   - **Step 2 — isolate the engine.** Then compare against the Python engine / default 4.4.1.
     Any difference here is the engine or version, not the port.

   Sets tolerance expectations honestly: `usage-guide.md` §6 already documents that
   Python-vs-Java agree only "within 1 order of magnitude" on bundled, already-verified
   configs, so the reader must not expect bit-equality. Leans on `docs/parity-roadmap.md`
   rather than inventing a method.

**Appendix:** the R→Python symbol table, the **two verified traps** (`output.tl.enabled`,
`economy.enabled` — see the taxonomy; deliberately two rows, not a table of plausible ones),
and an honest gaps list: `surveys.*`,
Python-engine restart, temperature/oxygen forcing downgrade, `plot()` convenience — each
stating its workaround (Java engine for the first two; no workaround for the third, say so;
plotting module / UI for the fourth).

## Boundaries

The guide's unique content is **the mapping, the traps, and the verification**. Python
mechanics live in `docs/usage-guide.md` (251 lines, already covering run → read → compare →
calibrate) and are linked, never restated. This keeps the two documents from drifting.

**The rule that makes this actionable** (sections 3–5 unavoidably show *some* Python, since a
mapping with no target is useless): show the **call signature only** — enough to establish
"this R call becomes that Python call" — with no parameter walkthroughs, no output samples,
no runnable end-to-end snippet. The moment the reader needs those, link `usage-guide.md`.
If a code block in the guide would still make sense with the R side deleted, it belongs in
usage-guide, not here.

## Keeping the claims true

The guide's claims rot silently when the reader, the allowlist, or the default jar changes.
They are **not** equally pinnable, and the spec must not pretend otherwise. Three tiers:

**Tier 1 — pinned by the fixture (mechanism).** A test against a **synthetic ~20-line
R-dialect fixture**, hand-written, not vendored. It must exercise: `=` separators, `#`
comments, `TRUE`/`FALSE` values, a pre-4.4.0 key that the shim migrates, a `surveys.*` key
(asserted unknown), and a `simulation.restart.*` key (asserted **not** unknown). This pins
*that the phenomena exist* — the dialect parses, strict mode catches surveys, strict mode
misses restart. Avoids vendoring third-party GPL-3.0 content.

**Tier 2 — the two verified traps, pinned by a TWO-SIDED assertion.** An earlier draft
proposed: "assert that setting the R/Java key leaves the Python attribute at its default and
that setting the mapped Python key flips it." **That assertion is vacuous** — it passes for
`banana.enabled`, i.e. for any key osmopy doesn't read, real or invented. It pins the *Python*
half of each row; the half that was wrong was the *R* half. It would have shipped all seven
bogus rows green. This is the mechanism by which the retracted table would have survived CI.

The assertion must therefore be two-sided:
- **Python side:** the R key leaves the attribute at its default; the mapped key changes it.
  (Not "flips" — `output.frequency.ndtperyear` → `output.recordfrequency.ndt` is an int, not
  a bool. An earlier draft's boolean-only recipe silently under-specified the non-bool rows.)
- **Provenance side (the one that matters):** the R key must be greppable in a **named
  upstream file** committed to the fixture as a citation. A row whose R key appears in no real
  config cannot pass. This is the assertion that would have caught the retracted table.

**Tier 3 — could be pinned, DECLINED with reasons.** An earlier draft called these "NOT
pinnable" and, in the same breath, warned against "a spec that claims CI protects claims CI
cannot protect." It committed the mirror error: claiming CI *cannot* protect a claim CI
trivially could. Corrected:
- **The jar-classfile claims** (`Surveys.class` in 4.3.3 + 4.4.1) **are pinnable** — the jars
  are vendored in-tree and `zipfile.ZipFile(...).namelist()` settles it offline in four lines.
  We **decline** it: the claim's stated failure mode (a future jar bump silently drops
  `Surveys.class` and invalidates the "use the Java engine" workaround) is real but rare, and
  a test that unzips jars to protect a doc sentence buys little. Mitigate with a comment at
  the jar-version sites — note there is no single site: `aliases.py:230`
  (`DEFAULT_TARGET_VERSION`), `runner.py:123` (`OsmoseRunner.__init__(jar_path)`), plus
  version strings in `demo.py`, `runner.py`, `calibration/problem.py`. This is a cost/benefit
  call, and it is recorded as one — not laundered in behind a real limit.
- **The Benguela counts** (844 / 236 / 21 / 8) are **genuinely un-pinnable**: they come from a
  one-time read of an unvendored upstream repo at its default branch, and drift with both the
  allowlist and upstream. Hence the dated-exhibit framing in §2.

The failure mode being avoided cuts both ways: do not claim CI protects what it cannot, and
do not claim something is unprotectable when the real reason is that protecting it isn't
worth it. Say which one it is.

## Deliberately excluded

- **The silent-restart bug is filed separately, not fixed here** — see
  [issue #120](https://github.com/razinkele/osmopy/issues/120). The Python engine should warn
  when it ignores `simulation.restart.enabled` rather than no-op silently. Documenting a
  silent failure is not the same as accepting it, but the fix is an engine change with its
  own test surface and does not belong in a docs PR. The guide documents the gap as it
  stands today; if #120 lands first, §2 and the appendix should be adjusted to describe the
  warning rather than the silence.
- **The allowlisted-but-unread key class is filed separately, not fixed here** — see
  [issue #121](https://github.com/razinkele/osmopy/issues/121), which covers the
  key-granularity mismatches, the factually wrong allowlist comments (several claim the Java
  engine reads keys absent from the 4.4.1 jar), the `economy.enabled` →
  `module.bioeconomics.enabled` alias routing users into a dead key, and the dead toggles in
  our own bundled configs including `data/examples`. Same reasoning as #120: this guide
  documents the terrain as it stands; fixing the terrain is an engine change with its own
  test surface. The guide's two verified traps are the reader's workaround **until** #121
  lands; the general case is explicitly deferred to #121's tooling rather than approximated
  by a table (see the taxonomy's retraction note for why approximating it is dangerous).
- **No R package install.** Claims are grounded by reading the real driver scripts. Should a
  future claim genuinely require runtime R behavior, install the package for that one
  question only.
- **No vendored R config.** See the synthetic fixture decision above.

## Success criteria

- A reader whose config sets `economy.enabled` learns — from the guide, not from a support
  thread — that their run has no economics on the Python engine even though the key migrated
  to upstream's correct 4.4.0 name, and that the Run page's "Will populate: Economic" label
  is a promise it won't keep. (Not "the UI shows an Economic page" — that claim was false.)
- A reader whose config references a sub-config that isn't there finds out, because the guide
  sent them to `scripts/check_config.py` and not to strict mode alone.
- No claim in the guide is stated as CI-protected unless it actually is; and nothing is
  called unprotectable when the truth is that protecting it wasn't worth the cost (tier 3).
- Every R-side key named in the guide is greppable in a named upstream file. **No row exists
  because the allowlist mentioned it.** (The retracted 8-row table failed exactly here.)
- An R OSMOSE user can load their existing config, discover what osmopy ignores in it, run
  it, read outputs, port their calibration, and verify the port reproduces their R numbers —
  without reading the source.
- Every R snippet cites a real file in a real repo.
- All gap buckets are stated plainly with their (differing) workarounds: capability absent →
  Java engine; renamed key → the right key name; shim-into-dead-key → set the engine's key
  directly; unsupported module → Java engine; missing sub-config and cross-file collision →
  `scripts/check_config.py`. Plus the temperature/oxygen downgrade, which has no workaround
  and says so.
- No Python mechanics are restated from `usage-guide.md`.
- The fixture test passes and would fail if any **Tier 1 or Tier 2** claim stopped being true.
  Not "any load-bearing claim" — Tier 3 deliberately leaves the Benguela counts and the jar
  classfiles unpinned, and the success criteria must not promise more than the tiers deliver.
