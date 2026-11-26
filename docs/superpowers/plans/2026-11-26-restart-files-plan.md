# Restart files (#120 follow-up) — plan

Date: 2026-11-26. Status: approved; owner decisions recorded at the end.

#120 (closed) only made the engine *warn* that restart is Java-only. This plan implements restart
in both the Python and C# engines, with Java 4.4.1's file format, so files move freely between
the three engines and R's `initialize_osmose()` initial-conditions files work too.

## What Java 4.4.1 does (read 2026-11-26)

**Writing** (`Simulation.run`, `SchoolSetSnapshot.makeSnapshot`):
- `simulation.restart.enabled` — **absent means TRUE** (Java warns and writes anyway).
- After step `i` it snapshots when `i >= spinup - 1` and `(i - spinup + 1) % frequency == 0`;
  `simulation.restart.recordfrequency.ndt` (default: never), `simulation.restart.spinup.nstep`
  else `.spinup.nyear * ndt` (default 0). **Always** one more snapshot after the last step.
- File: `restart/<output.file.prefix>_snapshot_step<i>.nc.<rank>` (rank 0 for us).
- Content, one row per school (`nschool` dimension), schools only (no background, no resources):
  `species` (int), `x`, `y` (float cell indices), `abundance` (double), `age` (float years =
  ageDt / ndt), `length` (cm), `weight` (g = t x 1e6), `trophiclevel`; with bioen
  `gonadWeight` (g), `maturity` (0/1); with genetics `genotype` [nschool, ntrait, nlocus, 2] and
  `trait_variance` [nschool, ntrait]. Global attribute `step` = i, plus a species-name list.

**Reading** (`Simulation.init`, `PopulatingProcess`, `NetcdfPopulator`):
- `simulation.restart.file` set => restart: population from that file, and the run continues at
  step `step + 1` (time, seasonality, fishing-by-year, seeding window all continue).
- Otherwise `population.initialization.file` => same reader, run starts at step 0 (R's
  `initialize_osmose()` files).
- Each school: `School(species, x, y, abundance, length, weight, round(age * ndt), TL)`; bioen and
  genetics fields restored; everything else re-derived as for a new school.
- **Not saved, not restored** (Java either): RNG state, the lagged starvation rate / predation
  success, depletable resource biomass, out-of-domain flags, fleet memory. A restart is a
  re-initialisation from a population, not a bit-exact continuation.

## Python engine today

- No restart path; `config.py` warns (#120) on `simulation.restart.file/enabled`.
- `osmose/engine/initialization.py` already builds a non-empty initial `SchoolState` for the
  opt-in warm start (`module.population.initialisation.enabled`): the place to plug the reader.
- `simulate()` always starts at step 0.

## Plan (one PR per item, each Python + C# unless stated)

**R1. Read a restart / initialisation file (Python).**
`osmose/engine/restart.py`: `read_snapshot(path, config) -> (SchoolState, next_step)` with Java's
field conversions (age years -> ageDt rounded, g -> t, cells from x/y); derived fields set as
`build_initial_population` sets them (first-feeding age, egg flag, lifespan, ...). Background
species and resources are not in the file (re-injected / reset by the engine as every step).
`simulate()` gains a start step: loop `step` from `next_step` to `n_steps`; records, seeding
window, fishing-by-year and `output.start.year` all use the true step. File lookup as Java: plain
name, or `<name>.<rank>` (rank 0). Precedence as Java: `simulation.restart.file` over
`population.initialization.file`. Bioen gonad/maturity and genetics genotype restored when on.

**R2. Write snapshots (Python).** `write_snapshot(state, step, path)` in the exact Java layout
(variable names, dtypes, units, `step` attribute). Snapshot schedule as Java. Written from the
shared output path for both engines (C# hands the state back, R4).

**R3. Round trip and Java interoperability.** Tests: write -> read reproduces the school table
(bit-exact where the format allows; float32 fields at float32 precision); a run restarted from
its own mid-run snapshot follows the same trajectory statistically (not bit-exact: RNG and lagged
state are not saved, as in Java); a Java-written snapshot (opt-in `OSMOSE_JAR`) loads in Python
and a Python-written one loads in Java (the format is the contract).

**R4. C# engine.** Read: the bundle carries the initial school table and the start step; the
store is filled from it (the probes already load states this way: `PipelineProbe.LoadSlice`).
Write: at each snapshot step the engine copies the store's school columns into result buffers
(sized from the schedule), and Python writes the NetCDF (no NetCDF code in C#). Flip
`warm_start` (feature 28) or add a `restart` feature; capabilities classify the restart keys.

**R5. Keys and UI.** Move `simulation.restart.*` and `population.initialization.file` from
Java-only to honoured; drop #120's warning; EEC/Benguela warnings go to 0. UI: a "start from
restart file" input on the Run page (optional; could be a later PR).

## Cost and risk
- Writing a snapshot: one NetCDF of a few MB per snapshot (EEC ~30k schools x ~10 fields).
- Biggest risk is the start-step plumbing in `simulate()` (records, windows, seeding, fishing by
  year, output.start.year, regions) and its C# mirror; covered by R3's trajectory tests and by
  "restart from step 0 file == fresh run" (same population => identical outputs given the seed).

## Decisions for the owner
1. **Write by default?** Java writes an end-of-run restart for every run unless
   `simulation.restart.enabled=false`. Mirror that (every run gains `restart/…nc.0`), or write
   only when `simulation.restart.enabled=true` (differs from Java, quieter)?
2. **Scope of this round:** read + write in both engines (R1–R5), or Python first (R1–R3, C#
   refuses restart configs until R4)?
3. **Exact continuation?** Keep Java's format (interoperable, not bit-exact), or also store the
   Python-only state (lagged starvation rate, depletable resources, RNG) as extra variables Java
   would ignore, making a Python restart a bit-exact continuation?

### Owner decisions (2026-11-26)
1. **Mirror Java's default:** absent `simulation.restart.enabled` means write the end-of-run
   snapshot; `false` turns it off.
2. **Both engines this round** (R1–R5 shipped together).
3. **Java format only:** interoperable with Java and R; a restart is a re-initialisation, not a
   bit-exact continuation (as in Java).
