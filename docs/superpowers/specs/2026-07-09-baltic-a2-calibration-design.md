# Baltic A2 calibration — design

**Date:** 2026-07-09
**Status:** approved (brainstorming), ready for implementation plan
**Depends on:** the Chunk A2 depletable-plankton engine feature (PR #104 / branch
`feat/baltic-chunka2-depletable-plankton`) and the existing `scripts/calibrate_baltic.py` DE calibrator.
**Predecessor:** `docs/baltic_chunka2_results_2026-07-09.md` (A2 relaxes the overshoot 76–90% but larval
mortality still sets cod's regime; bistability closed → pivot A2 to a *calibration* lever).

## Context / why

The bistability investigation closed with the deployed Baltic model robustly monostable. But it surfaced a
concrete calibration opportunity: a *uniform* larval mortality cannot calibrate cod and the forage fish
together (cod needs less larval M or it collapses; herring/sprat/percids need more or they overshoot). Two
facts make a real calibration now plausible:

1. `calibrate_baltic.py` already tunes mortality **per species** (16 params: 8 larval + 8 adult, log10
   space) with a banded ICES objective (zero inside `[lower, upper]`, squared-log penalty outside, plus a
   worst-species term and CV/trend stability penalties).
2. Chunk A2 (depletable plankton) is a strong, independent **bottom-up brake** on the forage-fish
   overshoot (herring 18 Mt → in-band at zoo rate ~0.6).

So the DE can plausibly land **cod** in band via its own larval mortality while **A2 + the clupeids' own
mortality** bring the forage fish into band — the simultaneous fit no single lever achieved.

## Goal

Find a Baltic parameter set that lands the maximum number of the 8 focal species in their ICES bands,
with A2 depletion on, and quantify the gain over an A2-off baseline calibration. Deliverable: the
calibrated parameters + a candidate config (not overwriting the deployed one) + a results write-up.

## Approach (chosen)

Run the existing DE calibrator **with A2 depletion enabled in the base config**, co-calibrating the
existing 16 mortality params **plus one shared zooplankton regrowth rate**, against the unchanged ICES
objective. Phytoplankton regrowth is fixed fast (5.0 ≈ chemostat); benthos shares the zooplankton rate.

**Rejected:** (a) 4 separate per-resource zoo/benthos rate params — +4 DE dimensions for little gain, they
would converge similar; (b) a fresh calibrator — the existing one already encodes the ICES banded
objective and the multiprocessing-safe wrapper.

## Components / changes

- **`scripts/calibrate_baltic.py`:**
  - A new phase/param set `get_a2_params()` (or a `--a2` flag on phase 1) that returns the 16 mortality
    params **plus one grouped param** keyed by a sentinel `species.regrowth.rate.zoo` (log10 bounds
    `log10(0.1) = -1.0` to `log10(2.0) ≈ 0.30`, x0 = `log10(0.6)`).
  - Enable A2 in the base config used for calibration: `ltl.depletable.enabled=true`,
    `ltl.depletable.floor=0.05`, `species.regrowth.rate.sp8/9=5.0` (phyto fixed).
  - **Grouped-param expansion:** where the objective wrapper builds `overrides[key] = value`, expand the
    sentinel `species.regrowth.rate.zoo` to the four real keys `species.regrowth.rate.sp{10,11,12,13}`
    (all set to the same value). A small helper `expand_param_overrides(param_keys, values)` (pure,
    unit-testable) does this; the wrapper calls it instead of the inline dict build.
  - A CLI toggle (`--a2`) selects the A2 param set + base-config enablement; without it, behavior is
    unchanged (the existing phase-1 calibration).
- **No engine change** (A2 already shipped).

## Data flow

`get_a2_params()` → bounds/x0 → `differential_evolution(objective, bounds, ...)`; each candidate vector →
`expand_param_overrides` → per-species mortality + 4 zoo/benthos regrowth keys → `run_simulation`
(A2 base config) → banded ICES error. Best vector → write the calibrated params + an in-band report.

## Baseline & success metric

Run the **same** calibration budget **A2-off** (phase-1, no depletion) as the control, and **A2-on** as
the treatment. Report **N/8 species in band** for each best config (a species is "in band" when its mean
biomass ∈ `[lower, upper]`), plus the objective value. Success = A2-on lands **more** species in band than
A2-off, ideally **cod and the two clupeids simultaneously** in band — the fit no prior config achieved. A
null (A2-on no better) is a valid result bounding the calibration lever.

## Compute (bounded — this is the big cost)

DE is hundreds of simulations. Bound it: a small `popsize` (e.g. 12–15) × limited `maxiter` (e.g. 20–30),
`--optimizer surrogate-de` if it helps, `n_years` 15 for the search (stationarity of the last-10-y window
still holds), `workers>1`. Order ~few hours per arm (A2-off baseline + A2-on). Checkpoint/report the
incumbent best periodically. A `--smoke` (tiny popsize/maxiter) validates the wiring fast.

## Deliverable

Do **not** overwrite the deployed `data/baltic/*` config. Write the calibrated result to a **candidate**
sidecar (`docs/diagnostics/baltic_a2_calibrated_params.json`: the per-species larval/adult mortality + zoo
rate + per-species in-band status), and a results doc. Promoting it to a bundled config is a separate,
explicit follow-up after review.

## Testing

- **Unit (CI-safe, no engine):** `expand_param_overrides` — the `species.regrowth.rate.zoo` sentinel
  expands to the four `sp{10..13}` keys with equal values; ordinary mortality keys pass through unchanged;
  log-space conversion (10^x) is applied consistently. `get_a2_params()` returns 17 params with the zoo
  param's bounds/x0 as specified. The existing objective/param unit tests still pass (parity of the
  non-A2 path).
- **Real-engine (CLI-only, not CI):** the `--smoke` wiring check, then the two bounded DE runs (A2-off,
  A2-on) and the in-band comparison. Excluded from CI per `feedback-ci-fragile-emergent-tests`.

## Risks

1. **DE budget vs convergence.** A bounded budget may under-converge. Mitigate with a sensible x0 (R18
   mortality + zoo 0.6) and report the incumbent; a partial improvement is still informative.
2. **Grouped-param expansion** must not corrupt the ordinary per-species overrides — covered by the unit
   test and the parity of the non-A2 path.
3. **Cod may remain uncalibratable** (its collapse↔overshoot knife-edge is narrow). If DE cannot land cod
   in band at any mortality with A2 on, that is itself the finding — it would confirm cod needs a
   structural change (recruitment function), not just retuning.

## Outputs

- `scripts/calibrate_baltic.py` changes + unit tests.
- `docs/diagnostics/baltic_a2_calibrated_params.json` (+ the A2-off baseline for comparison).
- `docs/baltic_a2_calibration_results_2026-07-09.md` — the in-band comparison (A2-off vs A2-on), the
  calibrated params, and whether a deployable ICES-calibrated Baltic is now within reach.
