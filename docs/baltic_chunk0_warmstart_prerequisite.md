# Chunk 0 follow-on — a warm-start standing IC for a reciprocal-invasion bistability test

> **STATUS: BUILT (2026-07-09).** `osmose/engine/initialization.py` (`build_initial_population` +
> `age_structured_population`), wired into `simulate.py:initialize()`, gated by the canonical flag
> `module.population.initialisation.enabled` (default off → empty init → byte-identical parity; 174
> engine tests unchanged). Init biomass = `population.seeding.biomass.sp{i}`; when the flag is on,
> egg-seeding is disabled (`seeding_max_step=0`) so a suppressed species is not continuously
> re-injected. Smoke (`scripts/warmstart_smoke.py`) builds cod-dominated and clupeid-dominated
> standing stocks with conserved biomass. Plan: `docs/superpowers/plans/2026-07-09-baltic-warmstart-standing-init.md`.
>
> **Honest scope (review finding):** a t=0 standing stock does NOT *manufacture* a second attractor.
> If the model is monostable (as Chunk 0 found), both ICs converge to the same state — the warm-start
> test confirms monostability more rigorously, it doesn't create bistability. What it enables is a
> **reciprocal-invasion test**: seed a settled clupeid-dominated state, introduce a little cod, and see
> whether cod invades/establishes (and vice versa). Genuine bistability still requires the missing
> feedbacks (Chunks C clupeid→cod-egg predation, A2 depletable plankton).
>
> **Follow-on (the actual test):** run the Chunk-0 sweep with `module.population.initialisation.enabled=true`
> so `cod_rich_seeding`/`cod_poor_seeding` become standing stocks, AND add a clupeid-dominated IC pair
> to `scripts/baltic_bistability_chunk0.py`; run each IC to a settled state and check reciprocal invasion.

---

## Why it was the prerequisite (original note)

`osmose/engine/simulate.py:1188` `initialize()` returns `SchoolState.create(n_schools=0)`; every
school is created by the egg-seeding mechanism (SSB==0 → `seeding_biomass` injected as virtual SSB
→ eggs). `restart.file` / `population.initialization.*` are Java-side allowlist keys with no Python
read path. So the Python engine has **no standing-stock population initialization**.

**Consequence for the Chunk-0 bistability experiment.** It can only seed EGGS, filtered through the
swept larval mortality and compressed by cod's Beverton-Holt recruitment — a **conservative** test
that can *confirm* bistability but cannot *rule it out*. Moreover, the single-cod-axis initial
conditions vary only cod seeding, so they cannot initialize the real Baltic **cod↔sprat
regime-shift alternative state** (a clupeid-dominated start). **A MONOSTABLE result therefore does
NOT rule out the sprat-dominated basin.**

**Minimal primitive for a definitive test (future engine sub-chunk).** An age-structured
standing-stock initializer — given a per-species initial biomass, distribute it across the size/age
structure at t=0 (OSMOSE "initialization by relative biomass") — OR a `SchoolState` restart reader
that loads a snapshot written by a prior run. Either lets two genuine adult standing stocks evolve
under fixed parameters. Critically, it must be able to seed a standing ADULT **clupeid-dominated**
state (high herring+sprat, near-zero cod) as well as a cod-dominated one; only then can a
hysteresis / alternative-stable-states test address the real cod↔sprat transition. Estimated
effort: medium (engine + `initialize()` + a snapshot format + parity tests). Do this before
treating any MONOSTABLE result from Chunk 0 as definitive.
