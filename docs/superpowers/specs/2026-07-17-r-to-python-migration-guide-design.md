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

1. **osmopy parses the R config dialect as-is.** `osmose-ben.R` is not R code; it is
   `key = value` lines in a `.R` file. `OsmoseConfigReader.SEPARATORS` is
   `re.compile(r"\s*[=;,:\t]\s*")` and `COMMENT_CHARS = {"#", "!"}`, so the file parses:
   **845 keys, 0 skipped lines**, including recursive sub-config resolution.
2. **The 4.4.0 compat shim auto-migrates legacy keys.** 8 deprecated keys migrated on
   Benguela, e.g. `fisheries.enabled` → `module.multispecies.fisheries.enabled`,
   `economy.enabled` → `module.bioeconomics.enabled`.
3. **845 parsed ≠ 845 supported: 236 keys are unknown to osmopy.** The reader is
   permissive; `EngineConfig.from_dict` under `validation.strict.enabled=error` is what
   reports the truth. This reframes the guide: "your config loads" is true *and is the trap*.
4. **`surveys.*` is unsupported — 21 keys.** `osmose-ben.R` defines a surveys module
   (`surveys.name.sr1 = acousticSurvey`, selectivity, survey movement maps). osmopy has no
   `surveys.` support anywhere. The in-tree Benguela port (PR #100) dropped the block
   entirely — `data/benguela/benguela_all-parameters.csv` contains 0 `surveys.` lines.
   Flagged as unknown by strict validation, so this gap is **loud but opt-in**.
5. **Python-engine restart is silently ignored.** `simulation.restart.enabled` and siblings
   are allowlisted as valid in `osmose/engine/config_validation.py` (marked "Java-side"),
   but the Python engine does not implement restart — `osmose/engine/initialization.py`
   exposes only `build_initial_population` / `age_structured_population`. The key loads
   clean, validates clean (it is **not** reported unknown), and does nothing. This gap is
   **silent**. Restart works on the Java engine.
6. **Phased calibration is NOT a gap.** `osmose/calibration/multiphase.py` provides
   `CalibrationPhase` + `MultiPhaseCalibrator` with calibrar's exact semantics: "Output of
   phase N becomes fixed params for phase N+1." The difference is plumbing — calibrar reads
   phases from a `parphase` CSV column; osmopy constructs them in code.

The two real gaps fail in **opposite directions**: `surveys.*` is flagged but only under
opt-in strict mode; `restart` is never flagged but silently no-ops. "Recognized" and
"implemented" are different sets, and no default-mode signal distinguishes them. Making the
reader aware of this distinction is the guide's core safety contribution.

7. **The Java engine is a genuine fallback for both gaps — verified against the jars**, not
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

## R API surface to cover

Read from the corpus, not recalled:

| R symbol | Source |
|---|---|
| `runOsmose(input, version=, osmose=)` (legacy camelCase) | `osmose-gog/run.R` |
| `run_osmose(input=, output=, osmose=, version=)` | `osmose-ben/launcher.R` |
| `read_osmose(path=, version=)` → `$biomass`, `$yield` | `osmose-ben/launcher.R`, `osmose-gog/runModel.R` |
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
   Losses: no surveys module, no Python-engine restart, no `plot()` one-liner convenience.
   States plainly that the Java engine remains available and is the fallback for both gaps.

2. **Your config already loads — and that's the trap**
   The headline, with the real Benguela numbers (845 parsed / 236 unknown / 8 auto-migrated).
   Prescribes the first action: set `validation.strict.enabled=error` **before trusting
   anything**, and read the unknown-key list. Explains the recognized-vs-implemented
   distinction and names the silent restart case explicitly, because strict mode will not
   catch it.

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

6. **Verify your port**
   Run both engines on the same config, compare biomass, know what tolerance to expect.
   Leans on the existing parity work (`docs/parity-roadmap.md`) rather than inventing a
   method. This is what makes the guide trustworthy rather than aspirational.

**Appendix:** the R→Python symbol table above, plus an honest gaps list — `surveys.*`,
Python-engine restart, `plot()` convenience — each stating its workaround (Java engine for
the first two; plotting module / UI for the third).

## Boundaries

The guide's unique content is **the mapping, the traps, and the verification**. Python
mechanics live in `docs/usage-guide.md` (251 lines, already covering run → read → compare →
calibrate) and are linked, never restated. This keeps the two documents from drifting.

## Keeping the claims true

The load-bearing claims (the R dialect parses; `surveys.*` reports unknown; restart does
not) are exactly the kind that rot silently when the reader or the allowlist changes.

Add a test asserting them against a **synthetic ~20-line R-dialect fixture** — hand-written,
not vendored. It must exercise: `=` separators, `#` comments, `TRUE`/`FALSE` values, at
least one pre-4.4.0 key that the shim migrates, at least one `surveys.*` key (asserted
unknown), and at least one `simulation.restart.*` key (asserted **not** unknown). A
synthetic fixture avoids vendoring third-party GPL-3.0 content and tests the same parser
behavior.

If CI goes red on this fixture, the guide has gone stale — that is the intent.

## Deliberately excluded

- **The silent-restart bug is filed separately, not fixed here** — see
  [issue #120](https://github.com/razinkele/osmopy/issues/120). The Python engine should warn
  when it ignores `simulation.restart.enabled` rather than no-op silently. Documenting a
  silent failure is not the same as accepting it, but the fix is an engine change with its
  own test surface and does not belong in a docs PR. The guide documents the gap as it
  stands today; if #120 lands first, §2 and the appendix should be adjusted to describe the
  warning rather than the silence.
- **No R package install.** Claims are grounded by reading the real driver scripts. Should a
  future claim genuinely require runtime R behavior, install the package for that one
  question only.
- **No vendored R config.** See the synthetic fixture decision above.

## Success criteria

- An R OSMOSE user can load their existing config, discover what osmopy ignores in it, run
  it, read outputs, port their calibration, and verify the port reproduces their R numbers —
  without reading the source.
- Every R snippet cites a real file in a real repo.
- Both gaps (`surveys.*`, Python-engine restart) are stated plainly with workarounds.
- No Python mechanics are restated from `usage-guide.md`.
- The fixture test passes and would fail if any load-bearing claim stopped being true.
