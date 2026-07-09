# Chunk A2 — depletable plankton — design

**Date:** 2026-07-09
**Status:** approved (brainstorming), ready for implementation plan
**Depends on:** the warm-start regime-shift harness (`scripts/baltic_bistability_chunk0.py`, merged #102) and
the Chunk C tooling (#103) for the layered test.
**Predecessors:** `docs/baltic_chunk0_warmstart_results_2026-07-09.md` (deployed Baltic MONOSTABLE),
`docs/baltic_chunkc_results_2026-07-09.md` (top-down egg predation alone insufficient — the overshoot must
be fixed first). Reframe: `docs/baltic_deep_investigation_2026-07-08.md` lever #2.

## Context / why

The deployed Baltic model is monostable, and Chunk C showed top-down clupeid→cod-egg predation cannot
create a predator-pit because cod overshoot is 20–90× the ICES band — the bottom-up plankton firehose
(all six LTL groups at `accessibility2fish = 0.8`, **non-depletable**) overwhelms any top-down control.
Chunk A2 adds the missing **self-limiting feedback**: make plankton a finite, depletable resource that
must regrow after grazing, so heavy fish production draws its own food down and the community can no longer
over-produce without bound.

## Grounded architecture (verified against the engine, 2026-07-09)

- **Within-step depletion already exists.** Both the production Numba predation path
  (`osmose/engine/processes/mortality.py:1035`, `_apply_predation_numba`:
  `rsc_biomass[r_idx, cell_id] = max(0, rsc_biomass[r_idx, cell_id] - eaten_from_prey)`) and the Python
  fallback (`mortality.py:546`, `predation.py:517`) subtract eaten biomass from `resources.biomass`, which
  persists after the mortality step.
- **The only thing erasing depletion is the per-step reset.** `ResourceState.update(step)`
  (`osmose/engine/resources.py:194`) overwrites `self.biomass` from forcing every timestep
  ("resources regenerate from forcing each timestep"): `cell_biomass = forcing × multiplier × accessibility`
  (`resources.py:236`), i.e. the stored biomass is the **accessible** pool.
- **Consequence:** the entire feature lives in `ResourceState.update()` — a plain Python method called once
  per timestep (`simulate.py:1509`), **outside** the Numba predation kernel. No kernel change is needed.
- **Resources are sp8–13** (Diatoms, Dinoflagellates, Microzoo, Mesozoo, Macrozoo, Benthos), loaded via
  `_load_config_species_type` using `species.*.sp{i}` keys.

## Goal

Add an opt-in depletable-plankton mechanism (per-resource logistic regrowth toward the forced carrying
capacity, with a floor), default off and byte-identical when off. Test — falsifiably — whether the
resulting self-limiting feedback (a) brings the community toward the ICES bands and (b) creates the
cod↔sprat regime-shift bistability that starting-conditions and egg-predation alone could not, using the
warm-start regime-shift sweep. A persistent monostable result is a valid negative.

## Mechanism (per-resource logistic + floor)

Each timestep, for each resource `i`, `update(step)` computes `K` exactly as the current reset does
(`K = forcing(step) × multiplier × accessibility`, seasonal and per-cell), then:

- **depletable off (default):** `self.biomass[i] = K` — today's behavior, unchanged.
- **depletable on:** regrow the carried-over (post-grazing) biomass `B` toward `K`:
  - `B = max(self.biomass[i], floor × K)`  (floor seeds recovery so a fully-grazed cell is not a permanent
    dead zone);
  - `B_new = B + rᵢ · B · (1 − B / K)`  (logistic; density-dependent — slow near 0 **and** near K, which is
    the depensation that opens a predator-pit);
  - `B_new = min(B_new, K)`  (cap at carrying capacity);
  - guard: where `K ≤ 0`, `B_new = 0` (no div-by-zero; land/off-season cells stay empty).

Per-resource regrowth rate `rᵢ` and a shared floor fraction are config; **defaults respect the ~15-day
timestep**: phytoplankton (sp8–9) high `r` (fast ≈ chemostat — and fish barely graze them), zooplankton
(sp10–12) moderate `r` (the real food-web feedback), benthos (sp13) low `r`.

**Honest simplification (v1):** the stored biomass already has accessibility baked in, so A2 depletes and
regrows the **accessible** pool (`forcing × access`). This is the minimal consistent extension of the
current storage and composes cleanly with lever #1 (which lowers `K`). A fuller model would deplete the
total stock and re-derive the accessible fraction — deferred to v2; not needed for the first bistability
test.

## Components / changes

- **`osmose/engine/resources.py`:**
  - `ResourceSpeciesInfo`: add `regrowth_rate: float` (per-resource `rᵢ`).
  - `ResourceState`: parse `ltl.depletable.enabled` (bool, default false), `ltl.depletable.floor` (float,
    default 0.05), and `species.regrowth.rate.sp{i}` (float per resource, with global default
    `ltl.regrowth.rate.default`) in the species-type loader.
  - A pure helper `logistic_regrow(biomass, k, rate, floor) -> ndarray` (vectorised over cells; the
    equation above), unit-testable without the engine.
  - `update(step)`: after computing the per-resource `K` array (the existing reset value), branch — full
    overwrite when not depletable (byte-identical), else `self.biomass[i] = logistic_regrow(self.biomass[i],
    K, rᵢ, floor)`.
- **`osmose/engine/config_validation.py`** (if it maintains a known-keys allowlist): register the three new
  keys so validation does not warn/reject them.
- **No change** to `mortality.py` / `predation.py` (the Numba + Python depletion write-backs already exist
  and persist).

## Config keys (new)

- `ltl.depletable.enabled` — bool, default `false` (master switch; off = parity).
- `ltl.depletable.floor` — float, default `0.05` (floor fraction of K, shared).
- `species.regrowth.rate.sp{i}` — float per resource; falls back to `ltl.regrowth.rate.default` (default
  e.g. `1.0`) when unset. Only consulted when depletable.

## Data flow (unchanged except the reset)

`incoming_flux → resources.update(step)` [now: regrow toward K if depletable, else reset] `→ movement →
mortality (predation grazes resources.biomass down, persists) → …`. The depleted biomass carries into the
next step's `update`.

## Testing

- **Unit (CI-safe, no engine):**
  - `logistic_regrow`: (a) with a rate that saturates, `B → K` within a step from a high seed (near-reset);
    (b) partial regrowth leaves `B < K` when `B_carried < K` and `r` small (depletion persists);
    (c) floor: `B_carried = 0` recovers to ≥ `floor × K`; (d) `K = 0` → 0 (no NaN); (e) cap: never exceeds K.
  - `ResourceState.update` gated: with `depletable off` the biomass equals the forced reset value
    (parity); with `depletable on` a pre-depleted `self.biomass` regrows toward K rather than resetting.
- **Parity (critical):** the **entire existing engine test suite passes unchanged** with the default config
  (depletable off). A dedicated test asserts `update()` output is identical to the pre-change reset when the
  flag is absent.
- **Real-engine (CLI-only, not CI):**
  1. **Depletion sanity:** one real Baltic run with `depletable on` — confirm resource biomass is drawn
     below K under grazing and recovers (not stuck at floor, not NaN/blow-up). STOP-gate: if resources
     collapse to floor everywhere or explode, reassess rates before the sweep.
  2. **ICES calibration:** sweep the zooplankton regrowth rate; does depletion bring cod/herring/sprat
     toward the ICES bands (relax the overshoot)?
  3. **Bistability:** warm-start regime-shift sweep with `depletable on` at the calibrating rate — does it
     create a determinate regime-shift? And layered with Chunk C (depletion + egg predation).
  Excluded from CI per `feedback-ci-fragile-emergent-tests`.

## Success criterion (falsifiable)

Chunk A2 **creates bistability** if the warm-start regime-shift sweep (depletable on) yields a determinate
`regime-shift` at one or more (rate, larva-scale). It **relaxes the overshoot** if the ICES check moves the
community from massive overshoot toward the bands. Either is a positive result; a persistent monostable /
still-overshooting outcome across the tested rates is a valid negative that bounds the depletion lever.

## Risks

1. **Parity.** The depletable-off path must be byte-identical. Enforced by keeping the reset branch verbatim
   and gating strictly on the flag; the existing suite is the guard.
2. **Rate calibration.** Too-fast `r` ≈ current firehose (no effect); too-slow ≈ resource collapse. The
   depletion-sanity STOP-gate + a rate sweep bound this before the full experiment.
3. **Accessible-pool-only depletion (v1 simplification).** Documented; revisit only if v1 gives an
   ambiguous result attributable to the hidden inaccessible reserve.

## Outputs

- `osmose/engine/resources.py` change + `tests/` unit tests; `config_validation` key registration.
- `docs/diagnostics/baltic_chunka2_*.json` (rate sweep, regime-shift, ICES).
- `docs/baltic_chunka2_results_2026-07-09.md` — write-up: depletion sanity, ICES relaxation, bistability
  verdict, and the honest interpretation (created bistability / relaxed overshoot / negative → v2 total-pool
  depletion or a combined lever).
