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
- **The calibration stack:** NSGA-II, CMA-ES, surrogate-DE, and a Pareto explorer — calibration
  machinery calibrar does not drive on its own (see §5).
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

Renamed keys (§2 has two verified examples) are a different kind of gap. They need no fallback
engine — they need the right key name, nothing more.

## 2. Your config already loads — and that's the trap

*(filled in Task 5)*

## 3. Run

*(filled in Task 6)*

## 4. Read & plot

*(filled in Task 6)*

## 5. Calibrate

*(filled in Task 7)*

## 6. Verify your port

*(filled in Task 8)*

## Appendix

*(filled in Task 8)*
