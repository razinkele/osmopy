# Baltic warm-start regime-shift sweep — design

**Date:** 2026-07-09
**Status:** approved (brainstorming), ready for implementation plan
**Depends on:** `feat/baltic-warmstart-standing-init` (PR #101 — the warm-start standing-stock primitive)
**Predecessor:** `docs/baltic_chunk0_results_2026-07-08.md` (egg-only sweep → MONOSTABLE, conservative), `docs/baltic_chunk0_warmstart_prerequisite.md`

## Context / why

Chunk 0 (2026-07-08) swept cod-rich vs cod-poor initial conditions across the larval-mortality
driver and found the deployed Baltic model **MONOSTABLE** — cod's fate is set by the driver alone,
independent of the starting cod stock. But that test was *conservative*: the ICs were **egg-only**
(seeded eggs filtered through the swept larval mortality and Beverton-Holt compensation) and
**single-cod-axis** (only cod seeding varied). It therefore could not construct the real Baltic
**clupeid-dominated (sprat-dominated) alternative state**, so it could rule out an egg-seeded
IC-dependence but could not fully rule out bistability.

The warm-start standing-stock primitive (PR #101) removes that limitation: with
`module.population.initialisation.enabled=true`, `population.seeding.biomass.sp{i}` becomes a genuine
age-structured **standing adult stock at t=0**, and egg-seeding is disabled so a suppressed species is
not continuously re-injected. This lets us initialize two real adult standing stocks — a
cod-dominated one and a clupeid-dominated one — and evolve them under identical parameters.

## Goal

Re-run the Chunk-0 bistability sweep with warm-start **ON** for **two IC contrasts**:

1. **cod-axis** — cod-rich vs cod-poor (the original contrast, now as genuine standing stocks): does
   a standing-stock cod IC alone change the monostable verdict?
2. **regime-shift** — cod-dominated vs clupeid-dominated (new): does the real cod↔sprat regime-shift
   basin exist as an alternative stable state?

Answer, for each contrast and each larval scale, whether the two ICs persist in different basins
(bistable) or converge (monostable), using the same reviewed consensus/stationarity discipline as v3.

## Honest scope (carried from the prerequisite doc)

A t=0 standing stock does **not** manufacture a second attractor. If the model is monostable, both
ICs converge and this test confirms monostability *more rigorously* — it does not create bistability
(that needs the missing endogenous feedbacks: Chunk C clupeid→cod-egg predation, Chunk A2 depletable
plankton). A **positive** (persistent divergence) result would be a genuine surprise and must be
scrutinized, not celebrated.

## Approach (chosen)

**Parameterize the IC pair in the existing v3 harness** (`scripts/baltic_bistability_chunk0.py`) and
reuse its reviewed machinery — `classify_state` (ICES bands), `basins_differ`, seed-consensus
(`aggregate_states`), the stationarity gate (`is_stationary`), and the `_partial`/`on_point`
incremental JSON writer. Generalize `run_bistability_point` / `run_bistability_sweep` to accept a
labelled `(state_A_override, state_B_override)` pair plus a `warmstart` toggle, then drive it twice.

**Rejected:** (a) a separate parallel sweep function — duplicates the four-round-reviewed logic and
lets the two implementations drift; (b) standalone standing-stock config CSVs run directly — bypasses
the consensus/stationarity gating that keeps the instrument honest.

## Components / changes to `scripts/baltic_bistability_chunk0.py`

All changes are additive; the egg-only path (warm-start OFF) stays byte-identical so the 2026-07-08
result remains reproducible.

- **`warmstart` injection.** A helper that, when warm-start is on, merges
  `{"module.population.initialisation.enabled": "true"}` into every override dict. This is the
  canonical flag the committed builder reads (`osmose/engine/initialization.py`,
  `_ENABLE_KEY = "module.population.initialisation.enabled"`).
- **IC builders** (return override dicts of `population.seeding.biomass.sp{i}` values):
  - existing `cod_rich_seeding()` / `cod_poor_seeding()` — unchanged (vary only cod sp0; other
    species keep their default config standing biomass). Note: their `population.seeding.year.max`
    key becomes inert under warm-start (egg-seeding is disabled) — harmless.
  - new `cod_dominated_seeding()` — cod 250,000 (ICES upper), herring 800,000 (lower), sprat 600,000
    (suppressed); sp3–7 default.
  - new `clupeid_dominated_seeding()` — cod 1,000 (remnant/invader), herring 1,500,000 (target),
    sprat 2,500,000 (upper); sp3–7 default.
- **Clupeid-dominance signal.** Record herring+sprat final mean/band at each point (not just cod), so
  the regime-shift verdict can test the clupeid axis.
- **Generalized sweep.** `run_bistability_point(..., ic_a, ic_b, warmstart)` and
  `run_bistability_sweep(..., contrast)` returning per-point `{cod_a_state, cod_b_state,
  clupeid_a_biomass, clupeid_b_biomass, ...}` and a contrast-appropriate verdict.
- **CLI.** `--warmstart` (bool) and `--contrast {cod-axis,regime-shift,both}`. The follow-on run is
  `--warmstart --contrast both`.

## Data flow

For each contrast, for each larval scale, for each seed: build driver override
(`larva_scale_override`) + IC override (A or B) + warm-start flag → `safe_run` → `run_simulation`
(PythonEngine) → stats dict → `classify_state` per axis → `aggregate_states` across seeds →
per-point outcome → sweep verdict. Incremental JSON is written after each point via `on_point`.

## Verdict logic

**cod-axis contrast:** unchanged from v3 — `basins_differ` on the cod band (does cod persist in one IC
but collapse/differ in the other, with gap ≥ threshold?). Reuses `run_bistability_sweep`'s existing
MONOSTABLE / BISTABLE / seed-split / instrument-limited verdicts.

**regime-shift contrast:** a regime shift ("BISTABLE — regime shift confirmed") is called only when
**BOTH axes diverge in the regime-shift direction** (user decision, 2026-07-09). Both checks are
**directional** (not the symmetric `basins_differ`), because a regime shift is a *specific* pattern —
cod down where clupeids are up:
- **cod-collapse axis diverges:** cod *persists* in the cod-dominated IC (consensus band ∈
  {low, in_range, overshoot}) **AND** cod is *collapsed* in the clupeid-dominated IC (consensus band
  == `collapsed`).
- **clupeid-boom axis diverges:** define the clupeid axis as the **summed herring+sprat consensus
  mean biomass**. It diverges when the clupeid-dominated IC's summed clupeid biomass exceeds the
  cod-dominated IC's by `bistability_gap ≥ gap_thresh` (the clupeid-dominated arm the higher one).
  (Summing sidesteps banding two stocks with different ICES bands; the relative-gap test matches how
  the cod axis already measures separation.)

Both axes must diverge, on the same IC assignment (cod-dominated arm = cod-rich/clupeid-poor). If a
gated arm (cod band, or either clupeid) is non-stationary/undetermined/seed-split, that axis is
withheld and the regime-shift call is provisional — same discipline v3 applies to cod.

Outcomes:
- both axes diverge (regime-shift direction) → **regime shift / bistable**;
- only one axis diverges → **partial — not a regime shift** (report which axis moved; the other is
  monostable), NOT a bistable call;
- neither → **monostable**;
- any seed-split or non-stationary/undetermined arm on a gated axis → withhold (instrument-limited /
  provisional), exactly as v3 does for cod.

The stricter conjunction avoids over-calling a regime shift from a clupeid-only or cod-only wobble.

## Pre-flight de-risk (must run before the full sweep)

The PR #101 smoke only *built* an IC (`build_initial_population`); it never ran one forward. With
egg-seeding disabled, an initialized standing stock reproduces from its own adults (Beverton-Holt on
real SSB) with no SSB==0 rescue. Before committing to the ~60-run sweep, run **one** standing-stock IC
(e.g. cod-dominated, seed 0, larva ×1.0) forward ~5 y and confirm:
- the run completes with no crash / NaN / 1e22 blow-up;
- biomass trajectories are finite and do not instantly vanish to zero at t=1.

If a standing stock decays pathologically at t=0 (e.g. every stock crashes in year 1), that is itself
a finding — stop and reassess (it would mean the standing-stock IC is not self-consistent with the
deployed parameters), do not run the full sweep.

## Testing

- **Unit (CI-safe, fake runner):** the generalized `run_bistability_point`/`_sweep` and the new
  regime-shift verdict are unit-tested with the existing fake-runner pattern (no real sim). Cases:
  both-axes-diverge → regime shift; cod-only diverge → partial; clupeid-only diverge → partial;
  neither → monostable; a non-stationary arm → provisional. The existing cod-axis unit tests must
  still pass (parity of the egg-only path).
- **Real-engine (CLI-only, manual, not CI):** the pre-flight run + the two full sweeps. Real Baltic
  emergent runs are non-reproducible across runner cores and are excluded from CI per
  `feedback-ci-fragile-emergent-tests`.
- **Parity:** with `--warmstart` absent the harness produces the same egg-only sweep as v3.

## Outputs

- `docs/diagnostics/baltic_chunk0_warmstart_bistability_cod-axis.json`
- `docs/diagnostics/baltic_chunk0_warmstart_bistability_regime-shift.json`
- `docs/baltic_chunk0_warmstart_results_2026-07-09.md` — write-up mirroring the 2026-07-08 results doc
  (both contrasts' tables, verdicts, and the honest-scope interpretation).

## Runtime

~60 real Baltic runs (2 contrasts × 5 scales × 3 seeds × 2 ICs), minutes each ⇒ order ~1–3 h wall
clock on the Python engine. `--smoke` (1 seed, scales {1.0, 0.1}, 3 y) stays available for a fast
sanity pass.

## Follow-on

If **monostable** (the expected result): the roadmap is unchanged — bistability must be *created*
(Chunk C clupeid→cod-egg predation; Chunk A2 depletable plankton). If a **regime shift** is confirmed:
scrutinize hard (re-run with more seeds, check it is not a seeding/parameter artifact) before treating
the deployed model as genuinely bistable.
