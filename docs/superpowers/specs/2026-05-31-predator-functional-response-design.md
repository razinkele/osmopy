# Predator Functional Response (aggregate, opt-in) — Design

**Date:** 2026-05-31
**Status:** Design approved, pending spec review → implementation plan
**Author:** brainstormed collaboratively

## Motivation

OSMOSE's current predation is effectively **Holling type-I with a ration ceiling**: each
predator school eats `min(available_prey, max_eatable)` per cell per sub-timestep, where
`max_eatable = biomass × ingestion_rate / (n_dt_per_year × n_subdt)`
(`osmose/engine/processes/predation.py:187,239` numba; `:298-302,360` Python fallback).

From the predator's side this already saturates (it cannot exceed its ration). What is
**linear** is the *per-prey mortality*: when prey is scarce, predators still take a near-constant
*fraction* of available prey — there is **no prey refuge at low density**. This is the classic
type-I weakness and is the knob a type-II / type-III response changes.

This feature adds **selectable, opt-in, per-predator-species functional response forms**
(type-I / type-II / type-III), mirroring the architecture of the just-shipped density-dependent
recruitment feature (Shepherd/B-H, PR #50): a per-species `shape` config key, **default-off**,
**bit-exact-preserving** when off, and a dedicated Baltic calibration phase that validates the
feature moves the objective.

### Scope decisions (locked during brainstorming)

- **Aggregate**, not per-prey-type. The response saturates on *total* accessible prey, not on
  each prey type independently. Per-prey-type **prey-switching** is the scientifically richer
  multi-species behavior but restructures the hot, parity-anchored single-pass kernel and is a
  large parity departure — and PR #50 already diagnosed the percid overshoot (the main
  prey-switching beneficiary) as **grid-under-resolution** (spatial), not a predation-form
  failure. So per-prey-type switching is **deferred** as a documented follow-on, to be revisited
  only if aggregate FR proves insufficient *and* the spatial-resolution issue is addressed.
- **Selectable forms** (type-I / type-II / type-III), per-predator-species, opt-in.

### Honest limitation

Aggregate FR will **not** spare a depleted stock (e.g. cod) when an abundant stock (e.g. percids)
sits in the same cell — there is no switching, so all accessible prey is still depleted
proportionally. Aggregate type-III does introduce a genuine prey refuge at low density and
density-dependent *total* predation pressure, but the species-specific cod/percid control stories
are fundamentally prey-switching problems. This feature delivers the general stabilizing lever and
modeling flexibility; it is not expected to fully resolve the percid overshoot.

## Section 1 — Engine math

Single injection point: replace `eaten_total = min(available, max_eatable)` with
`eaten_total = max_eatable × g(r)`, where `r = available / max_eatable`.

| Shape | `g(r)` | Behavior |
|-------|--------|----------|
| `type1` (default) | `min(r, 1)` | **Literal current branch** — bit-exact, no FP drift |
| `type2` | `r / (r + K)` | Saturating (Holling-disc) |
| `type3` | `r² / (r² + K²)` | Sigmoid, prey refuge at low density |

- `K` = dimensionless **ration-relative** half-saturation per predator species: the
  food-availability (in units of the predator's own max ration) at which the predator achieves
  half its max intake. `K=1` → half-satiated when accessible food equals one full ration.
- **Portability:** because `K` is normalized by each predator's own `max_eatable` (which already
  scales with predator biomass and ingestion rate), the same `K` is meaningful across Baltic /
  EEC / BoB / tutorial configs. An absolute biomass-unit `K` would vary by orders of magnitude per
  cell/species/config and be unbounded for DE — rejected for this reason.
- **DE-friendliness:** `K ∈ [0.1, 5.0]` is a clean, well-scaled bound sitting next to the existing
  recruitment params.
- **type-III Hill exponent fixed at 2** (standard). A general Hill exponent is **not** exposed
  (YAGNI).
- **Bit-exact guarantee:** `type1` dispatches to the *exact existing code path* (`min(available,
  max_eatable)`), NOT the formula evaluated at a limiting `K`. Default-off is therefore provably
  byte-identical.
- **Composition with single-pass depletion:** the FR composes with the existing single-pass
  proportional scheme unchanged. Each predator, in random `pred_order`, sees the remaining
  `available` (prey depleted in place by earlier predators) and applies `g(r)` to its own ration.

### Known confounding (handled in calibration)

Ration-relative `K` is coupled to `ingestion_rate` (since `max_eatable ∝ ingestion_rate`): the
same realized curve can arise from different `(ingestion_rate, K)` pairs. Mitigation: the
calibration phase **holds ingestion rates fixed** at their calibrated values and tunes only `K`.

## Section 2 — Config schema

Two new per-species keys (`osmose/schema/predation.py` + `osmose/engine/config.py`):

- `predation.functional.response.shape.sp{i}` → enum `type1 | type2 | type3`, default `type1`.
- `predation.functional.response.halfsat.sp{i}` → float `K`, range `[0.1, 5.0]`, **required iff**
  `shape ≠ type1`. Strict-validation error if shape is type-II/III and halfsat is missing
  (matching the existing strict-validation style, e.g. the `{name}`-wildcard pattern used for
  `evolution.trait.<name>.*`).

Note on value type: unlike the recruitment `stock.recruitment.shape.sp{i}` key (a **numeric float**
= Shepherd's continuous β exponent), the FR shape is a genuinely **discrete form choice**, so it is
an **enum string**. Same *pattern* (per-species shape key, default-off); different value type
because the semantics differ.

Background predators (GreySeal sp14, Cormorant sp15) read these via the same `background.py`
config-parse path that already reads `predation.ingestion.rate.max.sp{i}` /
`predation.predprey.sizeratio.*`.

## Section 3 — Kernel + Python fallback

- Add two arrays to `_predation_in_cell_numba` (`predation.py:142`): `fr_shape[n_species]` (int
  code: 1=type1, 2=type2, 3=type3) and `fr_halfsat[n_species]` (float).
- Branch on `fr_shape[sp_pred]` at the existing injection point. `==1` keeps the exact existing
  `min()`; `==2` / `==3` apply the formulas above.
- Mirror the branch in the pure-Python fallback (`_predation_in_cell_python`).
- The loop is **not** restructured — one branch at one injection point. The numba kernel recompiles
  but the signature change is purely additive.

## Section 4 — Testing

- **Unit tests** (`tests/test_engine_predation.py` or a new `tests/test_engine_functional_response.py`):
  - each form's curve shape: monotonic in `r`, asymptote → `max_eatable` as `r → ∞`, type-III
    inflection / prey-refuge at low `r`, `g(r) ≤ 1` always;
  - `type1` reproduces `min()` exactly;
  - `halfsat`-required strict-validation error when shape ≠ type1 and key missing;
  - background-predator path picks up FR config.
- **Parity:**
  - default-off run stays **12/12 bit-exact** — Java-parity baseline is **not** regenerated.
  - a **new** opt-in fixture asserts type-II / type-III produce *different* outcomes vs. type-I
    (proves the feature does something). This is a new baseline, not a modification of the
    Java-parity baseline.
- **Determinism:** same-seed reproducibility with FR enabled.

## Section 5 — Calibration (phase-14) + evaluation

- New phase-14 in `scripts/calibrate_baltic.py`, **stacked on the phase-13 Shepherd-calibrated
  baseline**.
- Enables `type3` on the four meaningful Baltic predators — **cod (sp0), pikeperch (sp5), GreySeal
  (sp14), Cormorant (sp15)** — and DE-tunes one `K` each (**4 new params**, bounds `[0.1, 5.0]`).
- **Holds ingestion rates fixed** at calibrated values (resolves the `K`↔ingestion confounding).
- **Fixes shape = type-III** for these predators rather than combinatorially sweeping type-II vs
  type-III per species (which would be 2⁴ DE runs). type-II remains available in the engine/config
  for users; it is simply not explored in this phase.
- **Perch (sp4) excluded** from the FR predator set — weakly piscivorous and flagged grid-under-
  resolved in PR #50, so FR is unlikely to help it and would add a confounded param.
- Reuses the bounded-runtime guards (`--patience 20 --wall-clock-cap-h 12 --checkpoint-every 5`)
  and multi-seed evaluation.
- Extends `scripts/evaluate_calibration_vs_ices.py` to report FR-on vs. phase-13 baseline:
  objective delta + per-species ICES in-range count.

### Success criteria (verdict gate, mirroring PR #50)

- Engine: 12/12 Java parity bit-exact with FR off; new opt-in tests green.
- Calibration: phase-14 converges within wall-clock cap; objective improves vs. phase-13 baseline
  (or is documented as not-improving, in which case the feature ships as mechanism-only and the
  calibration result is reported honestly).
- ICES: in-range species count reported vs. phase-13; not pre-committed to a specific gain given
  the honest aggregate-FR limitation above.

## Deferred / out of scope

- Per-prey-type functional response (prey-switching) — documented follow-on.
- General Hill exponent for type-III.
- type-II exploration in the calibration phase (engine support ships; calibration uses type-III).
- Any change to the bioenergetics ingestion cap (`bioen_predation.py`) — orthogonal, untouched.
