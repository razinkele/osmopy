# Chunk C — clupeid→cod-egg predation — design

**Date:** 2026-07-09
**Status:** approved (brainstorming), ready for implementation plan
**Depends on:** the warm-start regime-shift harness (`scripts/baltic_bistability_chunk0.py`, PR #102) and PR #101 warm-start primitive.
**Predecessor:** `docs/baltic_chunk0_warmstart_results_2026-07-09.md` (deployed Baltic → MONOSTABLE; bistability must be CREATED).

## Context / why

The warm-start reciprocal-invasion test showed the deployed Baltic model is **monostable** — a genuine
sprat-dominated standing stock cannot hold cod down; cod recovers (overshoot at low larval M) or collapses
(deployed ×1.0) driven by larval mortality alone, independent of the starting community. Bistability is not
latent; it must be **created** by adding a missing endogenous feedback.

**Chunk C adds the Baltic cod↔sprat cultivation-depensation feedback:** a booming clupeid (sprat/herring)
stock preys on the eggs and early larvae of recovering cod, suppressing cod recruitment. This is the
classic predator-pit / depensation mechanism for the post-1990 Baltic regime shift (ICES cod R collapsed
11M→1.5M and did not recover despite F falling 0.95→0.015 — depensation, not just fishing).

## Grounded mechanism (verified against the config/engine, 2026-07-09)

- **Deployed accessibility matrix** (`data/baltic/predation-accessibility.csv`, prey rows × predator cols;
  predator order `cod;herring;sprat;flounder;perch;pikeperch;smelt;stickleback;<6 LTL>`). The **cod prey
  row** is `cod;0.05;0;0;0;0;0.05;0.05;0;0;0;0;0;0;0` — cod-as-prey is accessible to cod (0.05,
  cannibalism), pikeperch (0.05), smelt (0.05), and **herring = 0, sprat = 0**. Clupeids currently cannot
  prey on cod at all.
- **Predation is size-ratio + accessibility gated** (`osmose/engine/processes/predation.py`); there is **no
  `is_egg` exclusion** in the prey loop, so egg/larval-stage schools are eligible prey when size overlaps.
- **Size-ratio window** (`predation.predprey.sizeratio.{min,max}`): herring/sprat = `[5, 500]`. Herring
  Linf 27 cm, sprat Linf 16 cm; a ~20 cm herring's prey window is 20/500–20/5 = **0.04–4 cm**, a ~12 cm
  sprat's is **0.024–2.4 cm**. Cod egg = 0.15 cm; cod early larvae < ~3 cm — **both fall in the window**,
  while adult cod (10–110 cm) is size-excluded.
- **Therefore:** setting cod→herring and cod→sprat accessibility to a positive value X enables clupeid
  predation on cod **eggs and early larvae only** — the size mechanism restricts it to the egg/larval stage
  automatically, so **no explicit stage row is needed**.
- **The matrix is a file** loaded via the `predation.accessibility.file` config key (resolved by
  `osmose/engine/config.py` relative to the config dir), so it is **overridable per run** with no engine change.

## Goal

Test — falsifiably — whether enabling clupeid→cod-egg/larval predation (cod→herring/sprat accessibility
X > 0) creates a cod↔sprat **regime-shift bistability**, by re-running the warm-start regime-shift sweep
with Chunk C on across a range of X. A monostable result at all tested X is an equally valid negative.

## Approach (chosen: config-only variant matrix + file override)

Chunk C strength is a **global treatment for a whole sweep** (not swept per point / not per IC arm), so it
is applied by pointing the run's `predation.accessibility.file` at a variant CSV — **no change to the swept
`run_bistability_sweep`/`run_bistability_point` signatures**.

- A helper generates the variant matrix from the deployed one (cod prey row's herring + sprat columns set
  to X; everything else byte-identical).
- `main()` in the harness, when `--chunk-c-strength X` is given, writes the variant CSV and sets
  `base_config["predation.accessibility.file"]` to it before running the regime-shift sweep, then names the
  output per X. Loops over multiple X values for the strength sweep.

**Rejected:**
- (B) engine support for per-cell matrix overrides via config keys — unnecessary engine work; YAGNI.
- (C) an explicit cod egg/larval **stage** row in the matrix — redundant, because the size-ratio window
  already restricts clupeid predation to egg/larval cod; adult cod is size-inaccessible regardless.

## Components / changes

All changes are additive; the deployed bundled Baltic config is **not** modified — Chunk C is applied only
via the experiment's per-run file override.

- **`scripts/chunkc_accessibility.py`** (new, small, importable + CLI) —
  `write_chunkc_matrix(deployed_csv: str, strength: float, out_path: str) -> str`: read the deployed
  accessibility CSV, set the **cod** prey row's **herring** and **sprat** predator columns to `strength`
  (leave every other cell unchanged, preserve header/row labels), write to `out_path`, return `out_path`.
  Pure/deterministic; unit-testable without the engine.
- **`scripts/baltic_bistability_chunk0.py`** —
  - CLI: `--chunk-c-strength FLOAT [FLOAT ...]` (accessibility values to test; e.g. `0.1 0.2 0.4`).
  - When set (requires `--warmstart`; forces `--contrast regime-shift`): for each strength, generate a
    variant CSV (via `write_chunkc_matrix`, to a per-strength path), set
    `base_config["predation.accessibility.file"]` to it, run the regime-shift sweep, and write
    `docs/diagnostics/baltic_chunkc_regime-shift_s{strength}.json`. A small pure helper
    `chunkc_output_name(strength) -> str` builds the filename (unit-testable).
  - The variant CSV path passed to the config: an **absolute** path (into the diagnostics dir or a temp
    dir). Task 1 of the plan **verifies** the override actually swaps the matrix at run time (see Risks).

## Data flow

For each strength X: `write_chunkc_matrix(deployed_csv, X, variant_path)` → set
`base_config["predation.accessibility.file"] = variant_path` → `run_bistability_sweep(..., warmstart=True,
contrast="regime-shift", clupeid_targets=herring+sprat)` → per-point regime-shift outcomes → sweep verdict
→ `docs/diagnostics/baltic_chunkc_regime-shift_s{X}.json`. The X = 0 control is the already-committed
`baltic_chunk0_warmstart_bistability_regime-shift.json` (deployed matrix, MONOSTABLE).

## Experiment

- **Strength sweep:** X ∈ {0.1, 0.2, 0.4} (0 = control, already run). Each is a full regime-shift sweep
  (5 larva scales × 3 seeds × 2 ICs = 30 real runs).
- **Horizon:** extend from 15 y to **25 y** for the Chunk-C sweeps, to reduce the stationarity-gate
  `provisional` points that made the 15 y control instrument-limited. (`--years 25`.)
- **Runtime:** ~2 h per strength on the Python engine; run one strength (e.g. X = 0.2) first as the
  headline test, then fill in the others.

## Success criterion (falsifiable)

Chunk C **creates bistability** if, at some (X, larva-scale), the regime-shift verdict becomes a
determinate **`regime-shift`** — cod *persists* in the cod-dominated IC **and** *collapses* in the
clupeid-dominated IC, **while** the clupeid axis diverges (clupeid-dominated arm higher). Reuses the
existing directional `regime_shift_outcome` / `_regime_shift_verdict` machinery unchanged.

A **monostable / same-basin** result at all tested X is a valid negative: the cultivation-depensation
mechanism, as implemented at these accessibility strengths, does not create an alternative stable state
under the deployed parameters — which would point at Chunk A2 (depletable plankton) as the next lever.

## Secondary: ICES calibration check

For each X, run the deployed config once (larva ×1.0, standard egg-only run, no warm-start) with Chunk C
on and compare cod / herring / sprat mean biomass against the ICES bands, to see whether egg predation
moves the deployed calibration toward or away from ICES. This is diagnostic only — it does **not** adopt
Chunk C into the bundled config.

## Testing

- **Unit (CI-safe, no engine):**
  - `write_chunkc_matrix`: variant has cod→herring = cod→sprat = X, and **every other cell identical** to
    the deployed matrix (assert full-matrix equality except the two target cells); header + row labels
    preserved; cod→cod cannibalism (0.05) untouched.
  - `chunkc_output_name(strength)` returns the expected per-strength filename.
  - CLI wiring: with `--chunk-c-strength`, `base_config["predation.accessibility.file"]` is set to the
    variant path and the regime-shift sweep is invoked (fake-runner, monkeypatched loaders, mirroring the
    existing `test_cli_warmstart_writes_both_contrasts`).
- **Real-engine (CLI-only, not CI):** the strength sweep + ICES check + the Task-1 override-verification
  smoke. Real Baltic emergent runs are excluded from CI per `feedback-ci-fragile-emergent-tests`.

## Risks / open verifications (resolved in the plan's Task 1)

1. **Accessibility-file override resolution.** The `predation.accessibility.file` override must actually
   load the variant matrix at run time (absolute vs config-dir-relative path). Task 1 smokes one Chunk-C
   run and confirms the loaded matrix differs from the deployed one; documented fallback = write the
   variant into the config dir with a temp name + cleanup if absolute paths don't resolve.
2. **Predation actually reaches cod eggs.** Confirm a Chunk-C run produces a **different cod trajectory**
   from the X = 0 control (i.e. clupeids realize cod-egg predation) — via a cod predation-mortality or
   biomass signal in a short real run. If cod is unaffected (e.g. eggs handled outside the predation loop
   after all, or a size-window edge case), Chunk C needs engine work — **surface as a finding, do not
   force**; the plan stops and reassesses rather than proceeding to the full sweep.
3. **Deployed config integrity.** The bundled `predation-accessibility.csv` is never edited; Chunk C lives
   only in generated variant files + per-run overrides.

## Outputs

- `scripts/chunkc_accessibility.py` + harness CLI changes + unit tests.
- `docs/diagnostics/baltic_chunkc_regime-shift_s{0.1,0.2,0.4}.json`.
- `docs/baltic_chunkc_results_2026-07-09.md` — write-up: per-X regime-shift tables + verdicts, the ICES
  check, and the honest interpretation (created bistability, or negative → Chunk A2 next).
