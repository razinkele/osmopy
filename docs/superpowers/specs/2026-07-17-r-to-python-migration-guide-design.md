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
3. **The 4.4.0 compat shim auto-migrates legacy keys.** 8 deprecated keys migrated on
   Benguela, e.g. `fisheries.enabled` → `module.multispecies.fisheries.enabled`,
   `economy.enabled` → `module.bioeconomics.enabled`. Note two of the 8 are
   `output.restart.enabled` / `output.restart.spinup` — keys that are migrated *and then*
   silently ignored (see finding 6), which is the trap compounding itself.
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
7. **Restart is not alone: "allowlisted but unread" is a whole CLASS, and the worst members
   are key-granularity mismatches.** A full sweep of the `config_validation.py` supplementary
   allowlist (see "Gap taxonomy" below) found the Python engine frequently *has* the
   capability but reads a **different key name** — so the R/Java-style key loads clean,
   validates clean, and the feature silently defaults off. This is more likely to bite a real
   porter than restart, because it hits routine output toggles that nearly every config sets.
   Tracked as [#121](https://github.com/razinkele/osmopy/issues/121).
8. **Phased calibration is NOT a gap.** `osmose/calibration/multiphase.py` provides
   `CalibrationPhase` + `MultiPhaseCalibrator` with calibrar's exact semantics: "Output of
   phase N becomes fixed params for phase N+1." The difference is plumbing — calibrar reads
   phases from a `parphase` CSV column; osmopy constructs them in code.

Gaps fail in **three different directions**, and the guide's core safety contribution is
teaching the reader to tell them apart:

| Bucket | Signal | Example | What the reader does |
|---|---|---|---|
| **Capability absent** | silent | `simulation.restart.enabled` | Use the Java engine |
| **Key-granularity mismatch** | silent | `output.meansize.enabled` → Python reads `output.size.enabled` | **Rename the key** — capability is there |
| **Unsupported module** | loud, but only under opt-in strict mode | `surveys.*` (21 keys) | Use the Java engine |

"Recognized", "implemented", and "implemented under a different name" are three different
sets, and no default-mode signal distinguishes any of them.

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

## Gap taxonomy — the key-rename table

This is the guide's single most actionable asset: the reader's key loads, validates, and does
nothing, and this table tells them the key to use instead. Every row was verified at its
consumption site (not by grep alone) during the allowlist sweep. Ships in the appendix.

| Key an R/Java config sets | Python key(s) to use instead | Note |
|---|---|---|
| `output.tl.enabled` | `output.meantl.enabled` | The *real* upstream Java name; we ignore it |
| `output.trophiclevel.enabled` | `output.meantl.enabled` | |
| `output.meansize.enabled` | `output.size.enabled` | |
| `output.byage.enabled` | `output.biomass.byage.enabled` **+** `output.abundance.byage.enabled` | Coarse toggle → two finer keys |
| `output.bysize.enabled` | `output.biomass.bysize.enabled` **+** `output.abundance.bysize.enabled` | Coarse toggle → two finer keys |
| `output.frequency.ndtperyear` | `output.recordfrequency.ndt` | |
| `species.lw.condition.factor.sp{idx}` | `species.length2weight.condition.factor.sp{idx}` | Latent today; every config sets both |
| `species.lw.allpower.sp{idx}` | `species.length2weight.allometric.power.sp{idx}` | Latent today; every config sets both |

Two rows that are **not** clean renames and must be written as caveats, not swaps:

- `temperature.{filename,varname,nsteps.year,factor,offset}` and the `oxygen.*` equivalents →
  the Python engine has **constant-only** forcing (`temperature.value` / `oxygen.value`,
  gated behind `bioen_enabled`). This is a capability *downgrade*, not a rename. Presenting
  `temperature.value` as the equivalent of a NetCDF field would be actively misleading.
- `predation.accessibility.stage.*` → Python derives stages from the accessibility CSV's
  column labels (`accessibility.py:50`), a different mechanism entirely. **Publish no mapping.**

Two things the guide must NOT list as gaps (verified non-gaps):

- `output.biomass.enabled` / `output.abundance.enabled` / `output.yield.biomass.enabled` are
  unread, but `output.py:45-56` writes those CSVs unconditionally — the output appears anyway.
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
   capability-absent and unsupported-module gaps (verified: finding 9). Key-granularity
   mismatches need no fallback — they need a rename.

2. **Your config already loads — and that's the trap**
   Opens with the Benguela exhibit, explicitly dated and framed as one example, never as
   figures to match: 844 keys parsed, 0 skipped — **and 236 of them unknown**, plus a
   sub-config resolution that silently *failed*. Prescribes the first action: set
   `validation.strict.enabled=error` **before trusting anything**, and read the unknown-key
   list. Then the crucial next beat — **strict mode is necessary but not sufficient**: it
   catches `surveys.*` but stays silent on restart and on every key-granularity mismatch,
   because those keys are *known*. Teaches the three-bucket taxonomy and points at the
   key-rename table in the appendix.

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
   Maps `phases` → `MultiPhaseCalibrator`, `control$popsize`/`control$maxgen` → optimizer
   args. Explicit that the *shape* differs even though the capability is present.
   Links `usage-guide.md` §4.

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

**Appendix:** the R→Python symbol table, the **key-rename table** (the taxonomy section
above — the guide's most actionable asset), and an honest gaps list: `surveys.*`,
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

**Tier 2 — pinned by a stronger assertion (the rename table).** The fixture as scoped would
stay green even if `output.meansize.enabled` started being read, or if the Python key were
renamed — i.e. the guide's most actionable asset is its least protected. Pin each rename-table
row directly: assert that setting the R/Java key leaves the Python attribute at its default
**and** that setting the mapped Python key flips it. That is a real regression test for the
advice, and it goes red exactly when #121 is fixed — which is the correct signal, since the
table should then shrink to a pointer.

**Tier 3 — NOT pinnable; must be dated in prose.** Two classes cannot be regression-tested
and must therefore never be written as load-bearing:
- **The Benguela counts** (844 / 236 / 21 / 8) come from a one-time read of an upstream repo
  at its default branch. They drift with the allowlist and with upstream. The guide states
  them as a dated exhibit ("measured 2026-07-17 against osmose-ben.R"), never as figures to
  match. The reader's instruction is always "run it on *your* config".
- **The jar-classfile claims** (`Surveys.class` in 4.3.3 + 4.4.1). This project has cut the
  default jar three times (4.3.3 → 4.4.0 → 4.4.1); a future bump could drop `Surveys.class`
  and silently invalidate the "use the Java engine" workaround with nothing catching it.
  Mitigation is a comment at the jar-selection site pointing back at the guide, not a test —
  a test that unzips jars to protect a doc sentence is worse than the drift it prevents.

Tier 3 is not a gap to close; it is a limit to state. The failure mode being avoided is a
spec that *claims* CI protects claims CI cannot protect.

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
  test surface. The guide's rename table is the reader's workaround **until** #121 lands.
- **No R package install.** Claims are grounded by reading the real driver scripts. Should a
  future claim genuinely require runtime R behavior, install the package for that one
  question only.
- **No vendored R config.** See the synthetic fixture decision above.

## Success criteria

- A reader whose config sets `output.byage.enabled` learns — from the guide, not from a
  support thread — why their by-age output is missing and which key to set instead. This is
  the single highest-frequency real outcome; `data/examples`, our own new-user starting
  point, currently exhibits the bug.
- No claim in the guide is stated as CI-protected unless it actually is (see the three tiers).
- An R OSMOSE user can load their existing config, discover what osmopy ignores in it, run
  it, read outputs, port their calibration, and verify the port reproduces their R numbers —
  without reading the source.
- Every R snippet cites a real file in a real repo.
- All three gap buckets are stated plainly with their (differing) workarounds: capability
  absent → Java engine; key-granularity mismatch → rename, per the table; unsupported module
  → Java engine. Plus the temperature/oxygen downgrade, which has no workaround and says so.
- No Python mechanics are restated from `usage-guide.md`.
- The fixture test passes and would fail if any load-bearing claim stopped being true.
