# Density-dependent recruitment: hockey-stick + Shepherd

**Status:** design (approved 2026-05-28, pre-implementation)
**Author:** brainstormed with Claude Code
**Scope:** engine + config + tests (phases 1-2), plus a Baltic calibration experiment (phase 3)

## Motivation

The Baltic calibration repeatedly fails to bring perch and pikeperch into their
ICES biomass envelopes — historically ×100+ overshoots that prior memory flags
as *structural*, not a tuning artifact. Constraints and mortality floors only
masked the symptom (see the cod-floor experiment: DE compensated by dropping
larval mortality 24× and biomass stayed at 6 Mt). The model has Beverton-Holt
and Ricker stock-recruitment (v0.11.0), but neither spans the shape space needed
to flatten an over-productive recruitment curve. Adding **Shepherd** (a
continuous generalization of B-H with a shape exponent β) and **hockey-stick**
(linear then a hard cap at a breakpoint) gives the optimizer the structural lever
to cap recruitment without distorting mortality.

## Goals / non-goals

**Goals**
- Add `hockey_stick` and `shepherd` stock-recruitment forms to the engine,
  available per-species via config like the existing forms.
- Keep the existing B-H / Ricker behavior bit-identical.
- Wire the new Shepherd parameters into the Baltic calibrator.
- Run a Baltic calibration and evaluate whether the new forms put more species in
  ICES range than the B-H baseline.

**Non-goals**
- No change to the linear egg-production formula or the low-SSB regime.
- No new optimizer; reuse the shipped DE + bounded-runtime infrastructure.
- Hockey-stick is implemented and available but is *not* part of the phase-3
  sweep (it would introduce a discrete form-choice dimension into a continuous
  optimizer).

## Design

### Approach

Extend the existing multiplicative-correction function `apply_stock_recruitment`
in place (rejected alternatives: a new strategy module — churn for a ~45-line
function reaching only four forms; a single Shepherd-family kernel — mathematically
wrong because hockey-stick is segmented, not a smooth Shepherd limit). All forms
keep the multiplicative-over-linear framing, so as SSB → 0 every form approaches
the Java-linear `linear_eggs` regime.

### Phase 1 — engine math

`osmose/engine/processes/reproduction.py`, `apply_stock_recruitment()` gains one
parameter and two branches:

```python
def apply_stock_recruitment(
    linear_eggs, ssb, ssb_half, recruitment_type,
    shepherd_beta,   # NEW: (n_sp,) per-species exponent; read only for "shepherd"
) -> NDArray[np.float64]:
```

Per-species correction (applied only where `type != "none"` and `ssb > 0`):

| Form | Correction |
|------|-----------|
| `beverton_holt` | `linear / (1 + ssb/ssb_half)` *(unchanged)* |
| `ricker` | `linear · exp(-ssb/ssb_half)` *(unchanged)* |
| `hockey_stick` | `linear` if `ssb ≤ ssb_half`; else `linear · (ssb_half/ssb)` (flat cap) |
| `shepherd` | `linear / (1 + (ssb/ssb_half)^β)` |

Invariants relied on:
- **Shepherd at β=1 is identically B-H** — exact-equality correctness anchor.
- **Hockey-stick is continuous at the breakpoint** — at `ssb == ssb_half` both
  pieces equal `linear`.
- Existing guards (`type=="none"` skip, `ssb ≤ 0` skip) are retained.
- `ssb_half > 0` and `β > 0` are guaranteed by config validation, so no
  division-by-zero or degenerate `0^β`.

`reproduction()` (line ~124) passes `config.shepherd_beta[:n_sp]` through.

### Phase 1 — config & schema

- **New per-species parameter** in `osmose/schema/species.py`: key
  `stock.recruitment.shape.sp{idx}`, float, **default 1.0**, description: Shepherd
  exponent (β<1 under-compensation, β=1 ≡ B-H, β>1 over-compensation; ignored for
  other forms).
- **Extend the type field** choices (`species.py:249`) and the config allow-set
  (`config.py:533`) from `{none, beverton_holt, ricker}` to add `hockey_stick`,
  `shepherd`.
- **`EngineConfig` (`config.py`):** parse `recruitment_shepherd_beta` via
  `_species_float_optional(cfg, "stock.recruitment.shape.sp{i}", n_sp, default=1.0)`;
  add field `shepherd_beta: NDArray[np.float64]` next to `recruitment_ssb_half`,
  threaded through the focal/background merge and the `from_dict` sites
  (≈ lines 1205, 1521, 1621, 1916).
- **`config_validation.py`:** confirm the AST allowlist walker auto-captures the
  new key; if not, add it to `_SUPPLEMENTARY_ALLOWLIST`. The
  `test_from_dict_warn_mode_clean_on_example_configs[*]` integration test must
  stay warning-free.

### Phase 1 — validation rules (`config.py:539` block, extended)

- Existing: `type != "none"` requires `ssb_half > 0`.
- New: `type == "shepherd"` requires `β > 0` (default 1.0 satisfies it; this
  catches an explicit β ≤ 0).
- β is unused for non-Shepherd forms — present-but-unused is not an error.

### Phase 2 — calibrator wiring (`scripts/calibrate_baltic.py`)

Calibration param-sets are `get_phaseN_params() -> (keys, bounds, x0)` consumed by
`differential_evolution` (bounds usually in log10 space).

- New `get_phase13_shepherd_params()` (sits alongside the existing
  `get_phaseN_params` functions; phase number can be renumbered in the plan) that:
  - Sets `stock.recruitment.type.sp{i} = shepherd` for **all 8 species** during
    config setup — *fixed*, not a DE dimension (keeps DE fully continuous).
  - Adds DE-tunable keys per species: `stock.recruitment.ssbhalf.sp{i}` (log10
    bounds, species-scaled) and `stock.recruitment.shape.sp{i}` = β (linear bounds
    ≈ 0.2–3.0). Up to 16 SR dimensions.
  - Keeps **cod sp0 ssb_half fixed at Bpa = 120 kt** (tune its β only), per the
    existing phase-12 convention.
  - Warm-starts `x0` from the current best phase-12 params where keys overlap.
  - Stacks on a prior phase's mortality/fishing keys.
- Reuse the shipped DE bounded-runtime guards unchanged: `--patience 20
  --wall-clock-cap-h 12 --checkpoint-every 5`, `OSMOSE_DE_WORKERS=16`.

### Phase 3 — calibration experiment (multi-hour, uncertain outcome)

1. **Baseline:** run the current best B-H phase-12 result through
   `scripts/validate_outputs_vs_ices.py`; record the in-ICES-range count and
   per-species magnitude factors. This is the bar.
2. **Run** the Shepherd DE calibration, multi-seed (≈3 seeds), under the runtime
   guards.
3. **Evaluate:**
   - **Primary success criterion:** the best Shepherd config places **strictly
     more of the 8 species inside their ICES envelope** than the baseline.
   - **Secondary (honest diagnostic):** per-species ICES magnitude-factor change,
     especially perch and pikeperch.

## Testing

New unit tests in `tests/test_engine_stock_recruitment.py`:
- Shepherd β=1 ≡ B-H (exact array equality).
- Hockey-stick continuity at `ssb == ssb_half`; flat cap (`≈ α·ssb_half`) for
  `ssb ≫ ssb_half`.
- Shepherd β>1 over-compensates (turns down at high SSB); β<1 under-compensates
  (gentler than B-H) — ordering/direction assertions.
- Low-SSB limit: all forms → `linear_eggs` as `ssb → 0`; non-negativity;
  `type=="none"` untouched.

Config tests (config-validation test file): new key parses with default β=1.0;
`shepherd`/`hockey_stick` accepted; `shepherd` with β≤0 raises; unknown type still
rejected.

Regression / parity guards:
- 14/14 EEC + 8/8 BoB parity stays **bit-exact** (default fixtures don't use the
  new forms; β default only matters under `type=="shepherd"`).
- `test_from_dict_warn_mode_clean_on_example_configs[*]` stays warning-free.

## Risks

- **Large parameter space.** ~16 added SR dimensions on top of the phase-12 set is
  a lot for DE under a 12 h cap; the primary success criterion may not be met even
  with a correct implementation. Phases 1-2 deliver standalone value (engine forms
  + calibrator knobs) regardless of the phase-3 result.
- **Shepherd numerics at extreme SSB.** `(ssb/ssb_half)^β` can overflow to `inf`
  for huge ratios and large β, driving recruitment → 0 (mathematically correct
  over-compensation collapse, but worth an explicit test if it shows up).

## Phasing summary

- **Phase 1** (engine + config + tests): clean, bounded, reviewable PR.
- **Phase 2** (calibrator wiring): small, code-only.
- **Phase 3** (run + demonstrate): experiment; outcome not guaranteed.
