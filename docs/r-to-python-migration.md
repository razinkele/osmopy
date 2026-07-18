# R OSMOSE → Python Migration Guide

This guide is for an **existing R OSMOSE user** — you have a working config (a `.R` file or a
`.csv`/`.osm` bundle) and R driver scripts (`run_osmose`/`runOsmose`, `read_osmose`, calibrar) —
who has decided to port that model to osmopy. It maps the R workflow onto its osmopy
counterpart step by step, and it is honest about the handful of places where a config that
*loads without complaint* is not the same as a config that *works*.

## 1. Should you switch?

If you already have a working R OSMOSE config and driver scripts, here is the trade, stated
plainly, so you can decide before porting anything.

**What you gain:**

- **No JVM dependency.** The Python engine runs as a pure NumPy/Numba process — nothing to
  install beyond the Python package itself, no `java` on `PATH`, no jar to point at.
- **Faster than the Java engine on every benchmarked config.**
- **The calibration stack:** NSGA-II, CMA-ES, surrogate-DE, and a Pareto explorer, built in —
  see §5 for how this maps onto your calibrar workflow.
- **The Shiny UI** — run, read, compare, and calibrate from a browser, with no driver script to
  write at all.

**What you lose:**

- **No surveys module.** `surveys.*` config keys are unsupported by the Python engine (§2).
- **No Python-engine restart.** `simulation.restart.*` loads and validates clean, but the Python
  engine's initialization only builds populations from scratch — nothing in it consumes a
  restart file.
- **Temperature/oxygen forcing downgrades to constant-only** (`temperature.value` /
  `oxygen.value`, gated on `bioen_enabled`). This is **not** a renamed key — it is a genuine
  **capability downgrade**: the Python engine has no path that reads a time-varying NetCDF
  forcing field the way the R/Java side does.
- **No `plot()` one-liner convenience.** R's `plot(obj, what=...)` family has no single Python
  equivalent (§4).

**The Java engine remains available**, and it is the fallback for exactly these two kinds of
gap — the capability-absent one (restart) and the unsupported-module one (surveys). This isn't a
hedge: `fr/ird/osmose/output/Surveys.class` is present in **both** the 4.3.3 and 4.4.1 jars, and
restart is implemented in 4.4.1 via `SchoolSetSnapshot` / `ModularSchoolSetSnapshot`, with
populator strings `simulation.restart.file` and `isRestart`. If your config depends on either,
keep the Java engine in your toolkit for that part of the run.

A renamed key is a different kind of gap from either of these — it needs no fallback engine,
just the right key name (§2 has a verified example).

## 2. Your config already loads — and that's the trap

osmopy will read your R-dialect config, merge its sub-configs, and run — with no complaint at
all — while quietly ignoring parts of it. This section is about the gap between "it loaded" and
"it works", and the two tools that narrow that gap without closing it.

### The exhibit

Measured 2026-07-17 against the real `osmose-ben.R` (`osmose-model/osmose-ben`, the
`osmose-ben_v4.x_develop` config directory): **844 keys parsed, 0 skipped lines** — and **236
of them unknown to osmopy** under
`validation.strict.enabled=error`. That contingency is not a footnote on the exhibit, it *is*
the exhibit: the same parse's sub-config resolution *failed* —
`osmose.configuration.initialization = input/initial_conditions.osm` (`osmose-ben.R:1021`)
points at a file this checkout doesn't have — so 844 is the count of keys reachable *without*
that file, not the config's real key count. (The reader itself parses 845 keys; one of them,
`_osmose.config.dir`, is injected by the loader at `reader.py:91` to record where the config
lives, not read from any line of the file — quote 844, since a person grepping the file counts
844 real lines.)

These numbers are one dated example, not a target. Run the two tools below on **your** config
before you trust either number for it.

### Two tools, and the order matters

osmopy gives you two different tools for two different kinds of damage.

**First, `scripts/check_config.py`.** It is the *only* production caller of the reader's
`format_diagnostics` / `diagnostics_have_errors` — neither `osmose/cli.py` nor any UI page reads
`reader.diagnostics`, so if you don't run this yourself, nothing else in osmopy surfaces a
missing sub-config, a within-file duplicate key, or an unparseable line. It takes `--config`,
not a positional path:

```bash
.venv/bin/python scripts/check_config.py --config data/minimal/osm_all-parameters.csv
# No config issues found.
# exit 0
```

Pointed at a config whose sub-config reference is broken — `osmose-ben.R` above is exactly such
a config — it reports what `validate()` cannot:

```bash
.venv/bin/python scripts/check_config.py --config osmose-ben.R
# osmose-ben.R:870: duplicate_key — output.weight.enabled = TRUE
# osmose-ben.R: missing_subconfig — .../input/initial_conditions.osm (from key osmose.configuration.initialization)
# 2 issue(s): 1 duplicate_key, 1 missing_subconfig
# exit 1
```

It exits 1 here only because `missing_subconfig` is an error-class diagnostic; the duplicate key
alone would still exit 0. (The missing sub-config *does* also produce an ordinary
`_log.warning` the moment the config loads, at `reader.py:141` — so it isn't literally silent,
but it's easy to miss: one `WARNING` line among a run's normal log noise, versus this tool's
explicit, exit-code-bearing report.)

**Then, `validation.strict.enabled=error`.** This is key-level: it tells you which config keys
osmopy has never heard of — a different question entirely from whether your files resolved.

Neither tool is sufficient on its own, and strict mode is the weaker of the two. It catches
`surveys.*` (genuinely unsupported, 21 keys, and loud about it) — but it is silent on the
Python-engine restart gap, on renamed keys, on missing sub-configs, and on cross-file
collisions, because in every one of those cases the key that arrives is *already known* to
osmopy, or the damaged key never arrives at all. **A clean strict-mode run means nothing was
unrecognized — not that your config works.**

And the failure compounds in the wrong direction: a missing sub-config's keys never reach the
flattened dict that `validate()` inspects, so the *more* of your config a broken reference
swallows, the *fewer* unknowns strict mode has left to complain about. The worse the damage, the
quieter the result. That is the argument for running `check_config.py` first — it is the one
tool that looks at file structure, not just the merged key set.

### The shim rescues five keys and strands three

osmopy auto-migrates a legacy pre-4.4.0 config on load. `osmose-ben.R` triggers exactly this,
logging `Loaded a legacy pre-4.4.0 config: 8 deprecated key(s) auto-migrated to 4.4.0 (e.g.
economy.enabled)`. All 8 keys go through the same table-driven rename mechanism
(`osmose/config/aliases.py`), and from the outside a migrated key and a working key look
identical — but three of them still aren't.

Three migrate to a key the **engine never reads**: `output.restart.enabled` →
`simulation.restart.enabled` (allowlisted as Java-side, `config_validation.py:113`, never
consumed by the Python engine); `output.restart.spinup` → `simulation.restart.spinup.nyear`
(`:123`, same story); and `output.fishery.enabled` → `output.fisheries.enabled` (`:126`, no
output-writing code reads it).

Five reach the engine and change its behavior: `fisheries.enabled` →
`module.multispecies.fisheries.enabled`, read at `engine/config.py:2035`; `simulation.bioen.enabled`
→ `module.bioenergetics.enabled`, `:2368`; `simulation.genetic.enabled` → `module.genetics.enabled`,
`:2425`; `population.initialization.relativebiomass.enabled` →
`module.population.initialisation.enabled`, `:538`; and, **fixed in
[#121](https://github.com/razinkele/osmopy/issues/121)**, `economy.enabled` →
`module.bioeconomics.enabled` (`config_validation.py:120`), which used to sit in the dead-key list
above. Before #121, `module.bioeconomics.enabled` was only read at runtime by
`engine_capabilities.py:32`, to build the Run page's "Will populate:" label — so setting it added
"Economic" to that label while nothing actually populated. Since #121, `engine/config.py:2436`
reads `module.bioeconomics.enabled` directly as the real economics switch (see the traps below for
the old, now-fixed, mechanism).

Same mechanism, same config, one log line announcing all 8 migrations — **five of eight arrive,
three don't**, and the three dead ones are fully silent about it. Nothing in that log line tells
you which group you're holding.

### The spatial-inputs trap

Everything above is about config *keys*. The thing that actually breaks a real R port is
*inputs* — and this is the most important trap in this guide, because it is the only one where
**the science silently changes**, rather than an output silently going missing.

`osmose/engine/movement_maps.py` has no NetCDF support at all — `grep -icE
"netcdf|xarray|\.nc\b|Dataset" osmose/engine/movement_maps.py` returns **0**. `_load_csv_grid`
(`:38`) opens every movement map with a plain text-mode `open()`. The handler around the call
site (`:220`) catches `(FileNotFoundError, OSError, ValueError)` — and `UnicodeDecodeError` is a
subclass of `ValueError`, so handing it a binary `.nc` file doesn't crash: it logs
`logger.error("Failed to load movement map file %s: %s", ...)` (`:221`), sets that grid to
`None` (`:222`), and **the run completes and reports success**.

This isn't a hypothetical mismatch. The real `osmose-ben.R` sets `movement.netcdf.enabled = TRUE`
and `movement.distribution.method.sp0 = maps` (`:386`, `:388`), and defines **27**
`movement.file.map{idx}` keys (`map0` through `map26`), every one of them a `.nc` path.

Both prescribed tools are blind to this. `check_config.py` only reports the duplicate-key and
missing-subconfig issues shown above — it never opens a movement map. `validate()` *does* flag
some of `osmose-ben.R`'s 185 `movement.*` lines as unknown (56 of them — legacy Java-side keys
like `movement.netcdf.enabled` and `movement.variable.map{idx}` that osmopy has genuinely never
heard of) — but none of those warnings are about the `.nc` files themselves, and the
`movement.file.map{idx}` keys that actually carry the `.nc` paths are recognized, known keys.
Their *values* are never inspected. So the reader's fish silently stop following whatever
spatial distribution the R config specified, and every tool this guide has recommended so far
says the config is fine.

**The concrete check:** run with `logger.error` visible and grep stderr for `Failed to load
movement map file`. If that string appears, some of your species have no spatial distribution
at all for the run you just trusted.

**Workarounds:** convert each `.nc` map to a semicolon-delimited CSV, or run that part of the
model on the Java engine, which reads NetCDF maps natively. If you convert: the loader flips
rows on load (row 0 in the CSV = the grid's northernmost row), so a converted grid must already
be flipped to match — the in-tree Benguela port hit exactly this and needed every converted map
run through `np.flipud` (`scripts/convert_benguela_maps.py`), or its fish end up on land.

### Eight ways a gap can hide

Config-key and spatial-input gaps fail in **eight** different directions — seven have a
workaround; the last, value coercion, is latent and gets one line. Treat this table as
incomplete: an earlier pass through this material stopped at three buckets and considered the
job done; three further fresh-eyes reviews found five more, the last of which is the
spatial-inputs trap above.

| Bucket | Signal | Example | Reader's move |
|---|---|---|---|
| **Unreadable input file** | **silent — invisible to both tools** | `.nc` movement maps; the loader is CSV-only | Convert to CSV, or use the Java engine |
| Capability absent | silent | `simulation.restart.enabled` (the Python engine never consumes a restart file) | Java engine |
| Key renamed | silent | `output.tl.enabled` → engine invented `output.meantl.enabled` instead (**fixed in #121** — engine now reads the real key first) | Use the right key |
| Shim → dead key | silent (engine); changes one UI label | `economy.enabled` → `module.bioeconomics.enabled` (**fixed in #121** — engine now reads the migrated key) | Set the engine's real key directly |
| Unsupported module | loud, opt-in only | `surveys.*` (21 keys, flagged by strict mode) | Java engine |
| Missing sub-config | silent to strict mode; a log line at load | a referenced sub-config file that isn't there | `scripts/check_config.py` |
| Cross-file collision | silent, no diagnostic at all | a sub-config silently overrides the master | `check_config.py` doesn't catch this either — see below |
| Value coercion | latent | `= 1` / `yes` / `T` are read as `False` (only case-insensitive `true` is truthy) | Use `true` or `TRUE` |

"Recognized", "implemented", "implemented under a different name", "present in the file you think
you loaded", and "present, but in a format osmopy cannot read" are five different situations, and
no default-mode signal distinguishes any of them from a config that actually works.

### Cross-file precedence — and Java disagrees with us

A split config exposes one more divergence, and it changes how you should read §6's port
comparison: osmopy and the Java engine resolve a duplicate key across files **differently**, and
only one of them tells you.

osmopy reads the master file, then recursively reads every sub-config it references, merging
each file's keys into one flat dictionary with a plain `dict.update()` (`reader.py:120`, inside
`_read_recursive`). Because the master's keys are merged first and each referenced sub-config is
merged afterward, **the sub-config always wins, silently, with no diagnostic** — the reader's
duplicate-key check (`reader.py:164` / `:191`) resets its `seen_keys` for every file it reads, so
it never looks *across* files at all.

The Java 4.4.1 engine does the opposite. Its `Configuration.class` carries the string `{0}Parameter
already defined {1}` — Java **warns** when a key is set twice, and (per the upstream
`updateKey` skip-if-exists semantics that this same jar's own 4.4.0 rename table relies on) keeps
the **first**-encountered value rather than the last.

So the same split config, read by both engines, can produce **different parameter values** —
osmopy silently takes whichever file it read last; Java loudly takes whichever file it read
first — and only one of them logs anything. Keep this for §6: if a Python-vs-Java comparison
diverges on a config split across several files, check precedence before concluding the engines
disagree. It may be this, not the engine.

### Two traps, fixed in #121 — worked examples of the class

Two of the gaps in the taxonomy above are worth walking through by name — not because you can
still reproduce them (you can't; see below), but because they are the sharpest worked examples of
the "allowlisted key, dead engine" class this guide keeps warning about. Both were **live silent
traps until [#121](https://github.com/razinkele/osmopy/issues/121) fixed them**: the engine now
reads both keys directly (`engine/config.py:923`, `:2436`), and two regression tests pin the fix —
`test_output_tl_enabled_now_read_after_121` / `test_bioeconomics_enabled_now_read_after_121` in
`tests/test_r_dialect_migration_claims.py`.

**`output.tl.enabled` — was the guide's headline trap; the one that actually bit real configs.**
This is upstream's real key (a literal string in both the 4.3.3 and 4.4.1 jars), and it appears in
7 of this guide's surveyed R config files — set to `true` in **two of them, from two different
upstream models**: `osmose-eec/eec_param-output_papierTROPHIC.csv:54` and
`osmose-gog/osm_param-output.csv:43`. (The key is case-insensitive — the reader lowercases every
key on load — and both cited files actually write it as `output.TL.enabled;true`; this guide's
lowercase form is the normalized spelling, not the on-disk one.) osmopy has always recognized the
key (it's allowlisted — `config_validation.py:219` — so `validate()` says nothing), but before
#121 the engine's actual mean-trophic-level output switch was a different key,
`output.meantl.enabled`, a name that appears in **zero** R config files and **zero** jars — an
osmopy invention. Both of those real users turned on mean-TL output, validated clean, and silently
got none. Since #121, `engine/config.py:923-926` reads `output.tl.enabled` first, falling back to
`output.meantl.enabled` only when the upstream key is absent, so both configs now work as written.

**`economy.enabled` — was a richer mechanism, but latent; also fixed in #121.** Its only
occurrence anywhere in the surveyed corpus is `osmose-ben.R:1048`, where its value is `FALSE` — so
even while broken, it never bit anyone in this corpus. The shim migrates it to
`module.bioeconomics.enabled`, upstream's own genuine 4.4.0 rename target (two hits in the 4.4.1
jar, zero in 4.3.3). Before #121 the engine's actual economics switch was a different key again,
`simulation.economic.enabled` — an osmopy invention with zero hits in either jar or the R corpus —
so a config that set `economy.enabled = TRUE` would have migrated cleanly and still done nothing.
It was worth knowing anyway, because it was the clearest illustration that the shim's rename
target and the engine's real switch could silently diverge — precisely the class this section
warns about. Since #121, `engine/config.py:2436` reads `module.bioeconomics.enabled` directly,
keeping `simulation.economic.enabled` only as a back-compat fallback, so the migrated key now
does exactly what it looks like it does. (`module.bioeconomics.enabled` is still the correct key
if you're aiming at the Java engine instead — `simulation.economic.enabled` is Python-only. The
Run page's "Economic" label (`ui/pages/run.py:797-800`, `engine_capabilities.py:32`) now matches
that: the underlying fleet-economics state genuinely initializes when fleets are configured. The
Economic UI page itself is still a placeholder pending a full economics UI — that part is
unrelated to #121.)

**What still bites you.** Both traps above are closed; the still-live ones are the spatial-inputs
`.nc` trap above (the most important trap in this guide — the science silently changes, not just
an output going missing), a missing sub-config (`check_config.py` catches it, strict mode
doesn't), a cross-file key collision (neither tool catches it, above), and the Python-engine
restart gap ([#120](https://github.com/razinkele/osmopy/issues/120), not fixed — §1 already covers
it). Run `scripts/check_config.py` plus `validation.strict.enabled=error` on **your** config
regardless — the general case of "allowlisted key, dead engine" is still only caught by careful
reading, not by either tool (see the appendix's general-case note), and #121 fixed two named
instances of it, not the class.

### The friendly failure, and the noise you should ignore

Not every mismatch is silent, and it's worth seeing one that isn't — it recalibrates what you
should actually be worried about. Drop `species.length2weight.condition.factor.sp0` from a
config while keeping `species.lw.condition.factor.sp0` (a legacy alias name osmopy still
recognizes for validation but never actually reads), and building the engine config raises:

```
KeyError: "Required OSMOSE config key missing: 'species.length2weight.condition.factor.sp0'"
```

That's the *best* outcome available in this guide: a crash, naming the exact key you need. Every
gap in this section is dangerous precisely because it doesn't do this.

One more signal worth previewing before §3: running a real config through osmopy prints
`UserWarning: Swapping size ratios` — and, on both the real `osmose-ben.R` and this repo's own
bundled Benguela config (`data/benguela/benguela_all-parameters.csv:571`/`:581`), it prints
**exactly 16** of them, and nothing else. These are benign: R and Java configs write
`predation.predprey.sizeratio.min.sp{i}` / `...max.sp{i}` in whatever order made sense to their
authors, and some list the smaller number under `.max` — osmopy just swaps the pair back into
the order it expects. It costs nothing and changes nothing.

It belongs next to the friendly failure above for a reason: this section spends its length
training you that noise is friendly and silence is dangerous. On a first real port, 16 warnings
will be the loudest thing on the screen — and it is the one thing here that is completely safe
to ignore.

## 3. Run

Two calling conventions are both live in real R OSMOSE deployments — an older camelCase form and
the current snake_case one. Both take a config path and an `osmose=jarfile`/`osmose = jarFile`
argument pointing at the OSMOSE `.jar` — exactly the JVM pointer that §1 already named as the
thing that disappears ("no `java` on `PATH`, no jar to point at").

| R call | Source | Python equivalent |
|---|---|---|
| `runOsmose("osm_all-parameters.csv", version=4, osmose=jarfile)` | `osmose-gog/run.R:7` (legacy) | `PythonEngine().run(config=..., output_dir=..., seed=...)` — writes the CSV/NetCDF tree |
| `run_osmose(input = configFile4, output = outputDir4, osmose = jarFile, version = "4.3.3")` | `osmose-ben/launcher.R:20` (current) | `PythonEngine().run(config=..., output_dir=..., seed=...)` |
| *(no R equivalent — R has no in-memory mode)* | — | `PythonEngine().run_in_memory(config=..., seed=...)` — returns an `OsmoseResults` directly, no disk I/O |

Both R calls take a config **path**; `PythonEngine.run`/`run_in_memory` take an already-parsed
config **dict**, and neither has a jar argument at all — there is no JVM step left to point one
at. For the mechanics — building that dict, seeds, ensembles, the `osmose` CLI, and the Java
engine's own `OsmoseRunner` (still jar-based; keep it for the §1/§2 gaps that need it) — see
[usage-guide.md §1](usage-guide.md).

## 4. Read & plot

R drivers read outputs one of two ways, and which one yours uses matters for the port:

| R call | Source | Python equivalent |
|---|---|---|
| `read_osmose(path=outdir, version="v3r2")`, then `data$biomass` / `data$yield` | `osmose-gog/runModel.R:37`, `:41`, `:43` | `OsmoseResults(output_dir, prefix="osm").biomass()` / `.yield_biomass()` |
| `read_osmose(path = outputDir4, version = "4.3.3")` | `osmose-ben/launcher.R:21` | same `OsmoseResults(...)` object — see below, this driver never indexes it directly |
| `get_var(ben, what="biomass", how="list")` | `osmose-ben/launcher.R:53` | `.biomass()` |

`osmose-gog/runModel.R` reads into a list and indexes it directly (`data$biomass` / `data$yield`).
`osmose-ben/launcher.R` also calls `read_osmose` (line 21) but never touches `ben$biomass` — every
value it needs instead comes out through `get_var()` or `plot()`. Both R idioms collapse onto the
same `OsmoseResults` object and its typed accessors. Full accessor list and DataFrame shape:
[usage-guide.md §2](usage-guide.md).

**Plotting does not carry over as one call — be clear about this before you go looking for a
Python `plot()`.** R's `plot(obj, what=...)` is a single function dispatching on the `what=`
string across dozens of chart types:

| R call | Source |
|---|---|
| `plot(ben, initialYear=2000, freq=12)` | `osmose-ben/launcher.R:23` |
| `plot(ben, what = "yield", initialYear=2000)` | `osmose-ben/launcher.R:24` |
| `plot(ben, what="yield.fishery.anchovy", col="red", lwd=2)` | `osmose-ben/launcher.R:47` |
| `plot(ben, what="biomass.acousticSurvey")` | `osmose-ben/launcher.R:44` |

(The first two are separate calls on adjacent lines, not one call — no `plot()` in the corpus
takes both `freq=` and `what=` together.)

No single Python call reproduces this dispatch. osmopy splits it across a plotting-function
module — `osmose/plotting.py`, one function per chart type (`make_stacked_area`,
`make_mortality_breakdown`, ...) — and the Shiny results UI, `ui/pages/results.py`, whose own
`make_timeseries_chart` builds exactly the biomass/yield line charts the first three R calls above
produce. Porting a `plot(ben, what="X")` call means finding the plotting function or UI panel for
that `X`, not calling a Python `plot()`.

The last row has **no Python equivalent at all** — not in `osmose/plotting.py`, not in the UI:
`plot(ben, what="biomass.acousticSurvey")` reads a survey output, and `surveys.*` is unsupported
by the Python *engine* (§2's taxonomy), even though the Java engine still implements it. Run that
part of the model on the Java engine, same as the rest of the surveys module.

## 5. Calibrate

This is the section with the most friction, because the capability calibrar gives you is fully
present in osmopy — NSGA-II, CMA-ES, surrogate-DE, multi-phase sequential calibration, a Pareto
explorer — but the **shape** of the workflow is completely different. Get the shape straight
first; the symbol-by-symbol map below means nothing until you have.

### The R shape: you write `runModel`, calibrar wires it up

In calibrar, **you own the driver**. `runModel(param, names, ...)` (`osmose-gog/runModel.R:10`)
writes the candidate parameters to a CSV, shells out to the jar, reads the outputs back, and
hand-assembles a named list:

```r
runModel  = function(param, names, ...) {
    names(param) = names
    write.table(param, file="./calibration-parameters.csv", sep=";", 
                col.names=FALSE, quote=FALSE)
    ...
    runOsmose(osmose=osmJar, java=javaAp, input="./config.csv", 
              output=outdir, options=NULL, log="osmose.log", 
              verbose=FALSE, clean=TRUE)   
    data = read_osmose(path=outdir, version="v3r2")
    ...
}
```

(`osmose-gog/runModel.R:10`, `:13`, `:18-19`, `:32-34`, `:37` verbatim; the two `...` mark the
elided middle, which reshapes monthly biomass/yield into yearly, per-replicate values before
assembling the returned list — a few adjacent comment/blank lines between those cited numbers
are dropped too, without a `...` marker, which is why the numbers themselves aren't
contiguous.)

`calibrate.R` then chains four calibrar calls around that driver:

```r
calInfo = getCalibrationInfo(path=".", file="calibration_settings.csv")
observed = getObservedData(calInfo, path=".", data.folder="DATA")
objfn = createObjectiveFunction(runModel=runModel, 
                                info=calInfo, 
                                observed=observed, 
                                aggFn=calibrar:::.weighted.sum,
                                aggregate=FALSE,
                                names=row.names(calibData))
control = list()
control$maxgen = c(150, 200, 250, 300)   # maximum number of generations (former gen.max parameter)
control$master = "master/"   # directory that will be copied
control$run = "RUN"   # run directory
control$restart.file = "./calib_restart"   # name of the restart file
control$REPORT = 1    # number of generations to run before saving a restart
control$parallel = FALSE
#control$nCores = 5
control$maxgen = 10   # maximum number of generations (former gen.max parameter)
control$popsize = 15   # population  size (former seed parameter)

cal1 = calibrate(calibData['paropt'], fn=objfn, method='default',
                 lower=calibData['parmin'], upper=calibData['parmax'], 
                 phases=calibData['parphase'], control=control, replicates=2)
```

(`osmose-gog/calibrate.R:17`, `:22`, `:33-38`, `:40-53` verbatim — nothing skipped or reordered.
Two fields never reach `calibrate()`: `control$maxgen` is set twice, so the vector on line 41 is
overwritten by the scalar `10` on line 48; `#control$nCores` on line 47 is commented out. This
guide's mapping below is against the values that actually take effect: `maxgen=10`, `popsize=15`.
`control$master`/`control$run` (directory-copying for a distributed/cluster run) have no bearing
on the calibration algorithm and aren't part of this section's mapping.)

### The osmopy shape: the framework owns the run/read loop

In osmopy there is no `runModel` to write, and that is the point, not a gap.
`osmose.calibration.OsmoseCalibrationProblem` (`osmose/calibration/problem.py:142`) — a pymoo
`Problem` subclass — *is* the driver: `_evaluate_candidate` maps a candidate parameter vector to
config overrides (`:260-265`), `_run_single` dispatches to `_run_python_engine` (in-process,
default) or `_run_java_subprocess` (opt-in, `use_java_engine=True`) (`:394-398`), and the results
are handed straight to your objective callables (`:403-406`):

```python
with results as r:
    obj_values = [float(fn(r)) for fn in self.objective_fns]
```

You supply two things — `free_params: list[FreeParameter]` (what to vary, and its bounds) and
`objective_fns: list[Callable]` (how to score one run) — not a function that writes files and
shells out. The write-CSV / shell-to-jar / read-outputs loop `runModel` used to hand-roll is gone
because the framework already does it, for both engines.

### Verified symbol map

| calibrar | osmopy | Status |
|---|---|---|
| `calibrate(..., phases=calibData['parphase'])` | `osmose.calibration.multiphase.MultiPhaseCalibrator` + `CalibrationPhase` | **Verified** — semantics match: "Output of phase N becomes fixed params for phase N+1" (`multiphase.py:47`, `:56-65`) is exactly calibrar's per-`parphase` sequencing, just constructed in code instead of read from a CSV column. **Correction:** neither class is re-exported from `osmose.calibration` — import from the submodule (`from osmose.calibration.multiphase import MultiPhaseCalibrator, CalibrationPhase`). |
| `control$popsize` / `control$maxgen` | plain keyword arguments on whichever optimizer you call — not a control object | **Corrected.** `CalibrationPhase` only carries `max_iter` (`multiphase.py:23`); `_optimize_phase` forwards it as `differential_evolution(..., maxiter=phase.max_iter)` with **no `popsize=`** (`:101`) — a multi-phase run silently gets scipy's default population. For explicit control, call the optimizers directly: `scipy.optimize.differential_evolution(popsize=, maxiter=)` (standard scipy kwargs), or `osmose.calibration.cmaes_runner.run_cmaes(popsize=, maxiter=)` (`cmaes_runner.py:47-48`). |
| `control$parallel` / `control$nCores` (commented out in the corpus, line 47, but the knob calibrar exposes for it) | `n_parallel=` / `parallel_backend=` on `OsmoseCalibrationProblem` (`problem.py:160-167`); `workers=` on `run_cmaes` (`cmaes_runner.py:51`); `OSMOSE_NSGA2_WORKERS` env var for the process-pool path (`problem.py:105-111`) | **Verified**, split across three knobs instead of one. |
| `control$restart.file` / `control$REPORT` | no equivalent — do this instead | **No counterpart.** `osmose/calibration/checkpoint.py` writes a progress snapshot every N generations for every optimizer, but per its own docstring it is "read by the Shiny dashboard at 1 Hz" (`checkpoint.py:3`) — live telemetry, not a file the optimizer reloads to resume a killed run. There is no resume-on-crash mechanism to point at. |
| `getCalibrationInfo(...)` → `getObservedData(calInfo, ...)` | depends on which objective family you use — no single loader | **Corrected**, it doesn't collapse onto one call. For ICES-band targets, `osmose.calibration.targets.load_targets(path)` (`targets.py:24`) does both steps at once, returning `BiomassTarget` records (species/target/lower/upper/weight/reference_point_type) straight from one CSV. For time-series objectives (`biomass_rmse`, `yield_rmse`, `diet_distance`, ...) there is **no loader at all** — you read your own observed `DataFrame` (e.g. with pandas) and pass it directly to the objective; osmopy has nothing analogous to calibrar's two-step info-then-data indirection. |
| `createObjectiveFunction(runModel=, aggFn=, aggregate=)` | splits three ways | **Corrected**, one calibrar call maps to three separate osmopy pieces: (1) the run/read/dispatch part is `OsmoseCalibrationProblem._run_single` (`problem.py:360`) — no user code, see above; (2) the per-run score is one of `osmose/calibration/objectives.py`'s functions (`biomass_rmse:41`, `diet_distance:55`, `yield_rmse:69`, ...) or the picklable wrapper classes `BiomassRMSEObjective`/`DietDistanceObjective` (`:130`, `:147`, not re-exported from `osmose.calibration` — import from `osmose.calibration.objectives`), passed as `objective_fns=[...]`; (3) the weighted-sum aggregation `aggFn=` performs is `objectives.weighted_multi_objective(objectives, weights)` (`objectives.py:95`, a plain weighted dot product) for the general case, or `osmose.calibration.losses.make_banded_objective(targets, species_names, ...)` (`losses.py:61`), the ready-made ICES-banded aggregate (log-ratio band error + stability penalty + worst-species penalty) — though `scripts/calibrate_baltic.py` itself inlines equivalent logic rather than calling this function directly (it imports only the smaller shared helpers `STABILITY_TYPES`/`quantity_key` from `losses.py:9`, `:14`), a known duplication, not a documentation error. |
| user-written `runModel(param, names, ...)` | **no counterpart, by design** | The write-CSV/shell-to-jar/read-outputs loop is owned by `OsmoseCalibrationProblem` for both engines (`problem.py:432-513`). Writing your own version would duplicate what the framework already does. |

`phases` is deliberately not in this table as a gap — `MultiPhaseCalibrator` is a real, tested
equivalent (`tests/test_multiphase.py`), not a stub. One genuine caveat worth carrying over: it
also declares an `n_replicates` field (`multiphase.py:24`) that reads like calibrar's
`replicates=2`, but `_optimize_phase` never consumes it — as of this writing it has no wired
behavior. Multi-run replication does exist in osmopy, just decoupled from calibration:
`osmose.runner.OsmoseRunner.run_ensemble(n_replicates=...)` (`runner.py:307-334`) runs the Java
engine N times with different seeds.

For the actual how-to — constructing `FreeParameter`s, wiring up `objective_fns`, running the
Baltic calibration driver, reading a finished run back — see
[usage-guide.md §4](usage-guide.md). This section only maps the shape; that one shows the
mechanics.

## 6. Verify your port

A naive check — "run both engines on my config and compare biomass" — conflates two different
variables. You most likely arrive with a v3- or v4.3-era config (§1); osmopy's default jar is
4.4.1. If the two runs disagree, you cannot tell whether that is your port breaking or several
years of upstream OSMOSE changes. Split it into two comparisons instead, so a divergence at each
step means only one thing.

### Before either step: confirm your inputs actually loaded

§2's spatial-inputs trap means a run can complete and report success while some species have no
spatial distribution at all — a silently dropped `.nc` movement map. Check this *before* trusting
either comparison below, or a "divergence" you attribute to the port or the engine may just be
that trap, present on one side and not the other. osmopy logs to stderr
(`osmose/logging.py:25`), so:

```bash
grep "Failed to load movement map file" your_run.log
```

If that string appears, resolve it first — convert the map to CSV, or run that species on the
Java engine (§2) — before chasing a numerical difference that a dropped map fully explains.

### Step 1 — isolate the port: same jar, new driver

Run your **original** jar — whatever version your R config was validated against — through
osmopy's own driver instead of R's. `OsmoseRunner.__init__(self, jar_path: Path, java_cmd: str =
"java")` (`osmose/runner.py:123`) takes an arbitrary jar path; it is not pinned to the bundled
4.4.1 jar. The CLI's `run` subcommand is a thin wrapper around exactly this constructor
(`runner = OsmoseRunner(jar_path=jar_path)`, `osmose/cli.py:64`):

```bash
.venv/bin/osmose run your_config.csv --jar /path/to/your/original-jar-with-dependencies.jar \
    --output output_step1/
```

Verified against this repo's own pre-migration Bay of Biscay config and its bundled 4.3.3 jar:

```bash
.venv/bin/osmose run data/examples_433_orig/osm_all-parameters.csv \
    --jar osmose-java/osmose_4.3.3-jar-with-dependencies.jar --output /tmp/step1
# ... osmose[info] Simulation 0 completed ...
# Complete. Output: /tmp/step1
# exit 0
```

Same engine (your jar), same config, only the driver changed — R's `run_osmose()`/`runOsmose()`
(§3) replaced by osmopy's `OsmoseRunner`. **Any difference between this run and your original
R-driven run, beyond ordinary run-to-run RNG variance, is the port** — a config-reading,
path-resolution, or invocation difference in osmopy's driver — not an OSMOSE engine change,
because the engine binary is identical. OSMOSE is stochastic, so a single unreplicated run on
either side of this comparison can differ from another for no reason but the seed, whether or
not the driver changed; if your config doesn't pin its randomseed keys, compare several
replicates (or their means) rather than one run each, the same discipline §6's tolerance
discussion below asks for across engines.

### Step 2 — isolate the engine: same config, the new engine

Now run the same config through what you'll actually use going forward — the Python engine
(§3's `PythonEngine().run(config, output_dir, seed)`, no jar at all):

```bash
.venv/bin/python -c "
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
config = OsmoseConfigReader().read(Path('your_config.csv'))
PythonEngine().run(config, output_dir=Path('output_step2'), seed=0)
"
```

Verified on the same pre-migration Bay of Biscay config used above — it runs to completion,
`returncode=0`, real biomass/yield CSVs written (the config's one pre-4.4.0 key auto-migrates
via the shim §2 already covers, logged as `Loaded a legacy pre-4.4.0 config: ...`).

If you want a Java-side comparison instead — the same engine language, only the version bumped —
point `.venv/bin/osmose run --jar` at the bundled 4.4.1 jar. Be aware this is not always a like-for-like
swap: the same command above, run against the 4.4.1 jar unmodified, refuses outright rather than
running: `osmose[severe] Your configuration file must be updated. Please run osmose with the
-update option.` (verified). The 4.4.1 jar requires an explicit `-update` conversion of a
pre-4.4.0 config before it will run it at all; the Python engine's shim (§2) does the equivalent
migration automatically, silently, on load. That asymmetry is itself worth knowing before you
read anything into a "the Java 4.4.1 arm won't even start" result.

**Any difference between Step 1's result and this one is the engine or the version — not the
port** — the port was already isolated and cleared in Step 1.

### How much difference is "no difference"

Don't expect bit-equality. `usage-guide.md` §6 documents Python-vs-Java agreeing only **"within
1 order of magnitude"** (per the parity suite: 14/14 EEC, 8/8 Bay of Biscay) — and that figure is
for osmopy's own bundled, already-verified configs, not yours. The same section explains why
tighter agreement isn't the goal: NumPy's PCG64 and Java's `java.util.Random` (MT19937) are
different generators that diverge at the first draw, so don't chase seed-level agreement either.
Full port status, including which configs that 14/14 and 8/8 cover: `docs/parity-roadmap.md`.

### If Step 1 or Step 2 still disagrees

Before concluding "the port is broken" or "the engine changed behavior," rule out two things §2
already found:

- **Is this a split config?** osmopy resolves a key set in both a master file and a sub-config
  silently, last-write-wins; the Java engine warns (`Parameter already defined`) and keeps
  first-write-wins (§2, "Cross-file precedence"). The same split config can legitimately produce
  different parameter values on the two engines — check precedence before concluding they
  disagree about anything else.
- **Did a shim or rename land differently than you expect?** (§2's eight-bucket taxonomy) — a
  key that silently changed meaning changes the run's outcome, and that outcome is neither the
  port nor an engine bug.

## Appendix

### R → Python symbol table

Every symbol below is read from the R corpus (§§3–5), not recalled.

| R symbol | Source | osmopy counterpart |
|---|---|---|
| `runOsmose("osm_all-parameters.csv", version=4, osmose=jarfile)` | `osmose-gog/run.R:7` | `PythonEngine().run(config=, output_dir=, seed=)` (§3) |
| `run_osmose(input=, output=, osmose=, version=)` | `osmose-ben/launcher.R:20` | `PythonEngine().run(config=, output_dir=, seed=)` (§3) |
| `read_osmose(path=, version=)` | `osmose-ben/launcher.R:21`; `osmose-gog/runModel.R:37` | `OsmoseResults(output_dir, prefix=...)` (§4) |
| `read_osmose(...)` → `$biomass` / `$yield` | `osmose-gog/runModel.R:41`, `:43` | `.biomass()` / `.yield_biomass()` (§4) |
| `get_var(obj, what="biomass", how="list")` | `osmose-ben/launcher.R:53` | `.biomass()` (§4) |
| `plot(obj, what=, initialYear=, freq=, col=, lwd=)` | `osmose-ben/launcher.R:23`, `:24`, `:47` | no single call — the plotting-function module `osmose/plotting.py`, or the Shiny results UI's `ui/pages/results.py` chart builders, chosen per `what=` (§4) |
| `plot(obj, what="biomass.acousticSurvey")` | `osmose-ben/launcher.R:44` | **no Python equivalent** — reads a survey output; `surveys.*` is unsupported by the Python engine (§2, §4) |
| `initialize_osmose(input=, file=, output=, type="climatology"\|"ncdf", run=)` | `osmose-ben/launcher.R:32-34`, `:36-38` | **no Python-engine restart** — use the Java engine ([#120](https://github.com/razinkele/osmopy/issues/120)) |
| `.readConfiguration(configFile4)` | `osmose-ben/launcher.R:27` | R-internal, no counterpart by design |
| `.getPar(conf, "osmose.configuration.initialization")` | `osmose-ben/launcher.R:28` | R-internal, no counterpart by design |
| `getCalibrationInfo(path=".", file=)` → `getObservedData(calInfo, path=".", data.folder=)` | `osmose-gog/calibrate.R:17`, `:22` | `osmose.calibration.targets.load_targets(path)` for ICES-band targets (one call does both steps); no loader for time-series objectives — read your own DataFrame (§5) |
| `createObjectiveFunction(runModel=, aggFn=, aggregate=)` | `osmose-gog/calibrate.R:33-38` | splits three ways: the run/read loop is `OsmoseCalibrationProblem._run_single`; the per-run score is a function in `osmose/calibration/objectives.py`; the aggregation is `weighted_multi_objective` or `losses.make_banded_objective` (§5) |
| `calibrate(..., phases=calibData['parphase'], control=control, replicates=2)` | `osmose-gog/calibrate.R:40-53` | `phases` → `MultiPhaseCalibrator`/`CalibrationPhase`; `control$popsize`/`maxgen` → plain optimizer keyword arguments, not a control object; `control$restart.file`/`REPORT` → no counterpart, no resume-on-crash mechanism exists; `replicates` → no calibration-side counterpart, see `OsmoseRunner.run_ensemble` (§5) |
| user-written `runModel(param, names, ...)` | `osmose-gog/runModel.R:10` | **no counterpart, by design** — `OsmoseCalibrationProblem` owns the run/read loop for both engines (§5) |

`initialize_osmose`, `.readConfiguration`, and `.getPar` have no body section above — they're
table-only. `initialize_osmose` is finding 6 restated (no Python restart; use the Java engine);
`.readConfiguration`/`.getPar` are R-internal helpers with no Python counterpart by design.

### The two fixed traps (reference)

Taught in prose in §2 (now fixed); repeated here for lookup. Deliberately two rows, not a table of
plausible ones — the general case can't be safely enumerated this way (below).

| R key (provenance + value) | Bit before #121? | Python key now read |
|---|---|---|
| `output.tl.enabled` — 7 R files, `true` in `osmose-eec/eec_param-output_papierTROPHIC.csv:54` and `osmose-gog/osm_param-output.csv:43` | **Yes — was the guide's headline trap** | **Fixed in [#121](https://github.com/razinkele/osmopy/issues/121).** `engine/config.py:923-926` now reads `output.tl.enabled` first; the osmopy-invented `output.meantl.enabled` remains only as a fallback |
| `economy.enabled` — one occurrence corpus-wide, `osmose-ben.R:1048`, value `FALSE` | **No — was latent.** Would only have bitten a config that set it `TRUE`; none surveyed does | **Fixed in [#121](https://github.com/razinkele/osmopy/issues/121).** `engine/config.py:2436` now reads the shim target `module.bioeconomics.enabled` directly; the osmopy-invented `simulation.economic.enabled` (`:2437`) remains only as a fallback |

The general case — every other allowlisted-but-unread key — is not enumerable this way: cheap
derivations fail in both directions (a `startswith` read looks unread to a literal grep; an
allowlisted key looks read to a validator), and the R corpus alone sets roughly 2,000 distinct
tokens. #121 fixed these two named instances; the systemic fix — tooling that names the correct
key at the point a config is loaded, not a static table that goes stale the next time a key is
renamed — is deferred and tracked in
[#123](https://github.com/razinkele/osmopy/issues/123).

### Honest gaps, and their workaround

| Gap | Workaround |
|---|---|
| `surveys.*` (21 keys) — no support in the Python engine | Java engine — `fr/ird/osmose/output/Surveys.class` is present in both the 4.3.3 and 4.4.1 jars (§1) |
| Python-engine restart (`simulation.restart.*`) — loads and validates clean, does nothing | Java engine ([#120](https://github.com/razinkele/osmopy/issues/120)) |
| Temperature/oxygen forcing — the Python engine has constant-only forcing (`temperature.value` / `oxygen.value`, gated on `bioen_enabled`), not the time-varying NetCDF field R/Java configs supply | **None.** This is a genuine capability downgrade, not a renamed key — there is nothing to point at instead (§1) |
| `plot()` one-liner convenience — no single Python call dispatches on `what=` the way R's `plot()` does | `osmose/plotting.py` (one function per chart type) or the Shiny results UI, `ui/pages/results.py` (§4) |
