# Predator Functional Response (aggregate, opt-in) — Design

**Date:** 2026-05-31
**Status:** Design approved; revised after round-1 in-loop review (4 angles). Pending round-2 review → implementation plan.
**Author:** brainstormed collaboratively

## Motivation

OSMOSE's current predation is effectively **Holling type-I with a ration ceiling**: each
predator school eats `min(available_prey, max_eatable)` per cell per sub-timestep, where
`max_eatable = biomass × ingestion_rate / (n_dt_per_year × n_subdt)`.

From the predator's side this already saturates (it cannot exceed its ration). What is
**linear** is the *per-prey mortality*: when prey is scarce, predators still take a near-constant
*fraction* of available prey — there is **no prey refuge at low density**. This is the classic
type-I weakness and is the knob a type-II / type-III response changes.

This feature adds **selectable, opt-in, per-predator-species functional-response forms**
(type-I / type-II / type-III), mirroring the architecture of the just-shipped density-dependent
recruitment feature (Shepherd/B-H, PR #50): a per-species `shape` config key, **default-off**,
**bit-exact-preserving** when off, and a dedicated Baltic calibration phase that validates the
feature at the **process level**.

> Code locations in this document are given by **symbol** (`_predation_in_cell_numba`,
> `_predation_on_resources`, etc.). Line numbers cited in round-1 review were verified accurate as
> of 2026-05-31 but are intentionally omitted here because they rot; the implementation plan should
> re-locate by symbol.

### Scope decisions (locked during brainstorming)

- **Aggregate**, not per-prey-type. The response saturates on *total* accessible prey, not on each
  prey type independently. Per-prey-type **prey-switching** restructures the hot, parity-anchored
  single-pass kernel and is a large parity departure — and PR #50 already diagnosed the percid
  overshoot (the main prey-switching beneficiary) as **grid-under-resolution** (spatial), not a
  predation-form failure. Per-prey-type switching is **deferred** as a documented follow-on.
- **Selectable forms** (type-I / type-II / type-III), per-predator-species, opt-in.

### Two consumption sites — and why FR is scoped to school predation

A predator's ration is filled at **two** sequential sites in the engine:

1. **School-to-school predation** (`_predation_in_cell_numba` / `_predation_in_cell_python`): the
   predator eats other schools; this sets `pred_success_rate = eaten/max_eatable`.
2. **Resource predation** (`_predation_on_resources`): runs *after* (1), filling the *leftover*
   appetite `remaining = max_eatable·(1 − pred_success_rate)` from LTL resource pools
   (plankton/benthos), capped `eaten = min(available_resource, remaining)`.

**This feature applies the functional response to site (1) only.** Site (2) stays type-I by design.
The consequence is explicit and *intended*, not a leak: when a type-III refuge suppresses a
predator's fish intake at low fish density, `pred_success_rate` drops, so `remaining` rises and the
predator **compensates by eating more resource prey (benthos/plankton)**. Ecologically this is
**realistic alternative-prey switching**: the refuge protects the *fish stocks we calibrate against
ICES*, while the predator survives on lower-trophic resources rather than starving. For the chosen
Baltic predators this matters mainly for **cod** (eats benthos sp13); seal/cormorant/pikeperch are
near-exclusively piscivorous so their resource backfill is negligible.

Two downstream consequences of FR on site (1), both deliberate and **must be tested** (not just
biomass):
- **Resource backfill** rises (above) — verify total predator ingestion and resource depletion.
- **`pred_success_rate` falls** under type-II/III (`success = g(r) ≤ 1`), which feeds
  starvation/growth bioenergetics. A predator in a moderate-prey cell is scored more food-limited
  → more starvation mortality / less growth. This is part of how FR regulates predators and
  interacts with the PR #50 recruitment curve; the calibration must observe predator condition,
  not only prey biomass.

### Honest limitations

- Aggregate FR will **not** spare a depleted stock when an abundant stock shares its cell — no
  per-stock switching. The single-pass depletion-in-place means aggregate type-III's refuge engages
  only when **total remaining accessible prey across all types in a cell is low** (near whole-cell
  prey collapse), *not* in the realistic "one stock crashed, others fine" case. It is a
  whole-cell-aggregate refuge, not a per-stock refuge.
- The species-specific cod/percid control stories are fundamentally prey-switching problems. This
  feature delivers the general stabilizing lever and modeling flexibility; it is **not** expected to
  resolve the percid overshoot (a spatial-resolution issue per PR #50).

## Section 1 — Engine math

Single injection point **within school predation**: replace `eaten_total = min(available,
max_eatable)` with `eaten_total = max_eatable × g(r)`, where `r = available / max_eatable`.

| Shape | enum / int code | `g(r)` | Behavior |
|-------|-----------------|--------|----------|
| type-I (default) | `type1` / `1` | `min(r, 1)` | **Literal current branch** — bit-exact, no FP drift |
| type-II | `type2` / `2` | `r / (r + K)` | Saturating (Holling-disc); slope 1/K at r=0 (no refuge) |
| type-III | `type3` / `3` | `r² / (r² + K²)` | Sigmoid; zero slope at r=0 → low-density refuge on aggregate |

- `K` = dimensionless **ration-relative** half-saturation per predator species: the
  food-availability (in units of the predator's own per-subdt max ration) at which the predator
  achieves half its max intake. This is a valid algebraic reparameterization of standard Holling:
  with `Imax = max_eatable` and half-saturation prey level `N½`, `f = Imax·N/(N+N½)`; setting
  `r = N/Imax`, `K = N½/Imax` gives `f = Imax·r/(r+K)`.
- **DE-friendliness:** `K ∈ [0.1, 5.0]` is a clean, well-scaled bound sitting next to the recruitment
  params.
- **K is well-scaled, not a transferable constant.** Because `r` depends on local cell prey density,
  predator packing, and the subdt discretization, a K calibrated on Baltic is **not** guaranteed to
  carry the same biological refuge threshold to EEC/BoB. The portability claim is narrow: K is
  *bounded and dimensionless for DE* (unlike absolute-biomass half-saturation), but remains an
  **empirical per-system calibration target**.
- **K does double duty.** type-II/III reduce realized intake at *all* finite r, not only at low
  density (e.g. K=1, r=1 → g=0.5; type-II r=10 → g≈0.91). So switching a predator to type-III at
  K≥1 both adds a low-density refuge **and** depresses mean realized ration. This is accepted as part
  of the calibrated effect and is the reason ingestion rate is held fixed during calibration
  (§5).
- **type-III Hill exponent fixed at 2** (standard). A general Hill exponent is **not** exposed
  (YAGNI).
- **Bit-exact guarantee:** `type1` dispatches to the *exact existing statement* `min(available,
  max_eatable)`, NOT the formula at a limiting K (verified: `r/(r+K)` at small K → 1 everywhere
  r>0, the opposite of `min`). Default-off is provably byte-identical.
- **Division-by-zero is already guarded upstream.** Both backends `continue` when `max_eatable <= 0`
  and when `available <= 0` *before* the injection point, so `r = available/max_eatable` is finite
  and positive. The FR branch must remain strictly **below** the `available <= 0` guard so a future
  refactor cannot introduce NaN.
- **Biomass conservation preserved.** `eaten_total = max_eatable·g(r) ≤ available` must continue to
  hold (so the proportional `share` redistribution stays conservative). For type-II this is
  immediate; for type-III it must be asserted by unit test across the K bound (§4).
- **Composition with single-pass depletion:** unchanged. Each predator, in random `pred_order`, sees
  the remaining `available` and applies `g(r)` to its own ration.

## Section 2 — Config schema

Two new per-species keys (`osmose/schema/predation.py` + `osmose/engine/config.py`), modeled on the
recruitment block (`stock.recruitment.shape.sp{i}` + strict-validation loop):

- `predation.functional.response.shape.sp{i}` → enum `type1 | type2 | type3`, **default `type1`**.
- `predation.functional.response.halfsat.sp{i}` → float `K`, range `[0.1, 5.0]`, **required iff**
  `shape ≠ type1`.

**Strict-validation rule and message.** If shape is `type2`/`type3` and halfsat is absent, raise a
strict-validation error with text:
`"predation.functional.response.halfsat.sp{i} is required when predation.functional.response.shape.sp{i} = {shape}"`.
The §4 test asserts on the `is required when` substring. (Value type differs from recruitment's
numeric-float `shape` because the FR form is a genuinely discrete enum choice; same per-species
default-off *pattern*.)

**Enum→int mapping.** The mapping `type1→1, type2→2, type3→3` is performed at config-parse time.
**Both** parse paths must produce identical int codes into the `fr_shape` array: the focal path
(`config.py`) and the background-predator path (`background.py`, which already parses
`predation.ingestion.rate.max.sp{i}`).

**Applicability to non-predator / prey-only species.** The keys are nominally per-species, but the
kernel only ever consults `fr_shape[sp_pred]` / `fr_halfsat[sp_pred]`, and LTL resource / prey-only
species never occupy the `sp_pred` slot. Therefore:
- Every species index (focal + LTL + background) gets a `fr_shape` entry, **defaulting to `1`
  (type1)**; for prey-only species the entry is allocated but **never consulted** (inert).
- Validation is **uniform**: the enum-membership check and the halfsat-required-iff-shape≠type1
  check fire for any `sp{i}` regardless of whether that species predates. Setting a non-`type1`
  shape on a prey-only species is therefore **accepted but inert** (it requires a halfsat to pass
  validation, but the kernel never reads it). This keeps validation simple and avoids a fragile
  "is this species ever a predator?" determination at parse time.

**Array sizing (critical) — exact layout.** `fr_shape` and `fr_halfsat` must be sized to
`n_total = n_species + n_background` (config.py: `n_total = n_sp + n_bkg`), which is **10 for Baltic**
(8 focal + 2 background), the **same length** as the existing concatenated `ingestion_rate` /
`species_id` arrays (`EngineConfig.__post_init__` rejects any per-species array whose length ≠
n_total). They are indexed by the **runtime** `sp_pred = species_id[p_idx]`.

Critical numbering distinction (the bug this section exists to prevent):
- **LTL resource species (the Baltic config's sp8–13) are NOT `species_id` slots.** They live in a
  separate `ResourceState`, are eaten only via `_predation_on_resources`, and never occupy
  `sp_pred`. They get **no** `fr_shape` entry.
- **Background predators' config-file keys are `sp14` (GreySeal) / `sp15` (Cormorant)** — but their
  **runtime `species_id` is `n_focal + bkg_idx` = 8 / 9** (background.py: `species_id = self._n_focal
  + bkg_idx`). So the config key `predation.functional.response.shape.sp14` is parsed by
  `background.py` and lands in the **background portion of the concatenated array at runtime slot 8**.
- Build the arrays following the **exact `recruitment_shepherd_beta` precedent** (config.py): focal
  values from the focal parse, concatenated with `np.full(n_bkg, ...)` defaults for the background
  portion (shepherd uses `np.ones(n_bkg)`; FR uses `fr_shape` background default = `1`/type1 unless
  the background `sp14`/`sp15` keys override, and a sentinel `fr_halfsat` background default). The
  no-background config path uses the focal arrays directly.

Sizing the arrays to `n_species` (8) or to any "focal+LTL" width would either trip the
`__post_init__` length check or, worse, misalign the background slots so GreySeal/Cormorant FR
**silently no-ops** — which the calibration would not catch.

**Config-validation allowlist (mandatory, per CLAUDE.md).** Read both keys via literal-prefix
f-strings in `config.py` so the AST walker in `config_validation.py` auto-captures them; if built
from a caller-arg `key_pattern` instead, add them to `_SUPPLEMENTARY_ALLOWLIST`. The integration
test `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs[*]`
must stay warning-free.

**Back-compat.** Every existing config on disk (eec_full, baltic, BoB, tutorial) omits both keys;
absence ⇒ `type1` ⇒ unchanged behavior. **No config migration is required.** Explicit `type1` and
absent-key are byte-identical (both hit the literal `min()` path).

**Documentation.** Add both keys to the config reference alongside where the recruitment `shape` key
is documented (in scope for this feature).

## Section 3 — Kernel + Python fallback

- Add two arrays to `_predation_in_cell_numba`: `fr_shape` (`np.int32`, length `n_total =
  n_species + n_background`, codes 1/2/3) and `fr_halfsat` (`np.float64`, length `n_total`). Use
  `np.int32` to match the kernel's existing int arrays (`species_id`, `age_dt`) and avoid a second
  numba specialization. See §2 "Array sizing (critical)" for the exact layout and the
  config-key-`sp14`/`sp15` → runtime-slot-`8`/`9` mapping.
- Branch on `fr_shape[sp_pred]` at the existing injection point. `== 1` keeps the exact existing
  `min()`. For `== 2` / `== 3` apply the formulas using `r*r` (not `r**2`) to stay on the unambiguous
  float path.
- Mirror the branch in the pure-Python fallback `_predation_in_cell_python`.
- The loop is **not** restructured — one branch at one injection point. Numba recompiles once on the
  additive signature change.
- **Call sites:** the two new args are threaded from `predation_for_cell` (which has `config` in
  scope, so `config.fr_shape` / `config.fr_halfsat` are directly available) into the single numba
  call site and the single Python fallback call site. No test imports the private kernels directly.
- `EngineConfig` gains two fields `fr_shape` / `fr_halfsat`, built in `config.py` alongside
  `ingestion_rate` (focal parse + background concatenation), following the `ingestion_rate` build
  path exactly.

## Section 4 — Testing

Unit / acceptance (new file `tests/test_engine_functional_response.py`):
- **Curve shape**, per form: monotonic increasing in `r`; `g(r) → 1` as `r → ∞`.
- **Biomass conservation:** assert `g(r) ≤ min(r, 1)` for all `r` across `K ∈ [0.1, 5.0]` — this is
  the load-bearing invariant (it guarantees `eaten_total = max_eatable·g(r) ≤ available`, since
  `available = r·max_eatable`). Asserting only `g(r) ≤ 1` is **insufficient** (it would not catch a
  form that takes more than the available prey at small `r`).
- **type-I exactness:** `type1` reproduces `min(r, 1)` (hence `min(available, max_eatable)`) exactly.
- **type-III refuge (operationalized):** on a strict `r < K` grid, assert `g(r)/r` is increasing
  (for `g=r²/(r²+K²)`, `d(g/r)/dr = (K²−r²)/(r²+K²)² > 0 ⟺ r < K`), and `g(small r) < r` (type-III
  takes a *smaller* fraction than type-I at low density: the refuge).
- **type-II smoke (it ships user-facing):** a minimal end-to-end run with `type2` on one predator
  completes without error and produces a **robustly detectable** difference from `type1` — assert on
  `preyed_biomass` / `pred_success_rate` (which change directly and immediately), and choose a
  `(K, prey-regime)` where `g(r)` departs clearly from `min(r,1)` (e.g. moderate `r ≈ K`), rather
  than asserting a small end-of-run biomass delta that a loose tolerance could wash out. (type-II's
  only other coverage is the curve unit tests; this is accepted.)
- **Strict validation:** `shape ≠ type1` with missing `halfsat` raises, asserting the `is required
  when` substring; valid config parses; explicit `type1` == absent-key behavior.
- **Array sizing / background path:** FR config set on a **background** predator (sp14/sp15) actually
  changes outcomes — guards against the `n_species`-vs-`n_total` sizing bug.
- **Downstream effects (deliberate):** with type-III on a predator, assert (a) resource-predation
  consumption increases for that predator vs type-I (alternative-prey backfill), and (b)
  `pred_success_rate` decreases in moderate-prey cells (feeds starvation/growth).
- **Determinism:** same-seed reproducibility with FR enabled.

Parity:
- Default-off run stays **12/12 bit-exact** — the Java-parity baseline is **not** regenerated.
- The "type-II/III differs from type-I" assertions above are new opt-in fixtures (new baselines),
  never modifications of the Java-parity baseline.

## Section 5 — Calibration (phase-14) + evaluation

### Phase scaffolding & stacking

- Add a `get_phase14_params()` builder + a `phase == "14"` branch in `run_calibration`
  (`scripts/calibrate_baltic.py`), following the existing flat phase ladder.
- **Stacking mechanism (explicit):** use the **phase-2-style inheritance** pattern (load a prior
  results JSON and inject its params as **fixed `base_config` overrides**), not warm-start. Phase-14
  loads **all 39 phase-13 params** as fixed `base_config` overrides and `get_phase14_params()`
  returns **exactly the 4 new K keys** — the 39 frozen params must live *solely* in `base_config`,
  never in the free `param_keys` list (the phase-2 precedent works because its free set is disjoint
  from the inherited set; phase-14 must preserve that disjointness). The 4-D runtime math below
  holds only under this exact-4-keys condition.
- **Prerequisite artifact:** there is currently **no `phase13_results.json` on disk** (only
  phase1/2/12 exist). Phase-14 therefore requires either (a) running phase-13 first and persisting
  its result JSON, or (b) committing a `phase13_results.json` artifact from the PR #50 run. The plan
  must pick one and make the phase-13 params a concrete file. (Correction to an earlier draft:
  phase-13 optimizes **39** params — 16 mortality + 8 fishing + 7 ssb_half + 8 Shepherd β — not 27;
  27 was phase-12.)

### Parameter space & runtime

- **4 new K params**, bounds `[0.1, 5.0]`, type-III fixed. With all 39 phase-13 params frozen this is
  a clean **4-D DE problem**: `eff_popsize = max(15, 10×4) = 40`; at ~175 evals/h ≈ 14 min/generation
  → fits the 12 h wall-clock cap comfortably under `--patience 20`. (If the 39 were *not* frozen,
  `eff_popsize` would explode to ~430 and not converge in 12 h — hence strict freezing is required,
  not optional.)
- Reuses the bounded-runtime guards (`--patience 20 --wall-clock-cap-h 12 --checkpoint-every 5`) and
  multi-seed re-ranking.

### Predator selection

- FR enabled on **cod (sp0), pikeperch (sp5), GreySeal (sp14), Cormorant (sp15)**.
- **Positive rationale:** cod and pikeperch are the two dominant fish piscivores in the Baltic diet
  matrix (broad, strong prey-access rows); GreySeal and Cormorant are the apex background predators
  whose top-down pressure the feature most directly modulates.
- **Perch (sp4) excluded — corrected rationale.** Perch is **not** weakly piscivorous; the diet
  matrix shows perch preying on smelt/clupeids/stickleback at coefficients within ~30 % of pikeperch.
  It is excluded purely for **parameter economy** and because perch's *own* biomass overshoot is a
  **spatial grid-under-resolution** problem (PR #50), which FR cannot fix. Perch's FR *could* matter
  for controlling its prey; adding it is a documented follow-on, not a correctness gap.
- **Background-predator access note:** GreySeal/Cormorant predation is governed by size-ratio
  windows + default accessibility (they are not focal rows in the accessibility CSV). FR composes
  with this unchanged — FR acts on the resulting `available`, whatever produced it. This is exactly
  why the §2 array sizing (their config keys `sp14`/`sp15` → runtime slots 8/9) is load-bearing for
  these two.

### Confounding & the process-level diagnostic (required)

Fixing ingestion rate removes only the **direct** `K ↔ ingestion_rate` confound. K's mechanism
(reducing realized predation mortality at low prey density) still trades against the **frozen**
phase-13 prey mortality and recruitment params — and a biomass-only objective (last-10-yr mean)
**cannot distinguish** "predator gave prey a refuge" from "prey had lower background mortality" or
"recruitment compensated." This residual confounding is acknowledged and is the core scientific risk.

**Mitigation (mandatory):** add a **process-level diagnostic** reporting each FR predator's *realized
predation mortality at equilibrium*, FR-on vs FR-off. The K effect is **identifiable at the process
level** even though it is **not identifiable from biomass alone**. phase-14 K's are interpreted
**conditional on the frozen phase-13 baseline**, and this is stated in the writeup.

**Concrete data source (first-class plan task, not an evaluation-script afterthought).** The engine
already records per-predator→per-prey eaten biomass in the **diet matrix** (`_predation_in_cell_*`
accumulate it when diet tracking is enabled; resource predation likewise). No new engine output is
required — but `run_simulation` currently returns biomass only, so the diagnostic run must **enable
diet tracking** and surface the diet matrix. Definition:
- **Realized predation mortality** of predator *p* on prey *q* at equilibrium = (Σ eaten biomass of
  *q* by *p* over the last *N* years) / (mean biomass of *q* over the same window), per year.
- The diagnostic runs the **same calibrated config twice with diet tracking on**: FR-off (type-I
  baseline) and FR-on (type-III with the calibrated K's). For each of the four FR predators it
  reports the realized predation mortality on each prey under both, and the FR-on − FR-off delta.
- The **type-III refuge signature** is a *reduction* in realized predation mortality on prey that
  sit at **low density** (where `r` is small → `g(r) < min(r,1)`). "Low vs high density" is read off
  the prey's own equilibrium biomass between the two runs — no arbitrary binning is needed; the
  refuge shows up as a negative delta concentrated on the lower-biomass prey.

This diagnostic is what makes the §"Success criteria" go/no-go falsifiable; the implementation plan
must treat surfacing the diet matrix from the diagnostic run as a concrete task.

### Evaluation script

Extend `scripts/evaluate_calibration_vs_ices.py`:
- Add `shepherd-fr` to the `--mode` `choices` (currently `{bh, shepherd}`). On top of the shepherd
  fixed-config it sets `predation.functional.response.shape.sp{0,5,14,15}=type3` (mode config,
  injected like the `shepherd` branch injects `stock.recruitment.type`) + the calibrated `halfsat`
  values (which flow through from the phase-14 result JSON's `parameters` as overrides).
- **Objective delta requires new capability:** the script currently reports only per-species ICES
  in-range banding, not the objective. Reporting "objective delta vs phase-13" means importing
  `make_objective` / the objective wrapper and evaluating both param sets — budget this as a new
  compare capability, not a one-line flag.
- Emit: objective (FR-on vs phase-13), per-species ICES in-range count delta, and the realized-
  predation-mortality diagnostic specified in §"Confounding" above.

### Success criteria (gate)

**Binding gates (go/no-go):**
- Engine: 12/12 Java parity bit-exact with FR off; all §4 opt-in tests green (incl. type-II smoke and
  the array-sizing/background test).

**Reported, NOT gated (honest outcome, mirroring PR #50's "not pre-committed to a gain"):**
- Calibration objective vs phase-13 baseline. The feature **ships as engine capability** regardless
  of whether the objective improves — but the disposition is split:
  - If phase-14 demonstrates (via the process diagnostic) that **≥1 predator's realized low-density
    predation mortality drops measurably FR-on vs FR-off**, *and* the objective does not regress, it
    ships as a **calibrated Baltic improvement**.
  - Otherwise the calibrated K's are indistinguishable from the type-I baseline, and it ships as
    **engine capability only**, explicitly **not** as a Baltic improvement. This falsifiable
    process-level minimum is what separates "the knob is wired" (defensible to ship) from "the knob
    helped the Baltic fit" (must be earned), matching how PR #50 held its gate to a number.
- "Converges" means phase-14 **terminates via patience/convergence before** the wall-clock cap; if it
  hits the cap, the capped-best is reported honestly as capped (per the bounded-runtime guards).

## Deferred / out of scope

- Per-prey-type functional response (prey-switching) — documented follow-on.
- FR on resource predation (`_predation_on_resources`) — kept type-I by design (§"Two consumption
  sites"); resource backfill is intended alternative-prey switching.
- Perch (sp4) FR — follow-on; excluded here for parameter economy.
- General Hill exponent for type-III.
- type-II exploration in the calibration phase (engine + smoke test ships; calibration uses type-III).
- Any change to the bioenergetics ingestion cap (`bioen_predation.py`) — orthogonal, untouched.

### User-facing caveat to document

type-II functional responses are classically **destabilizing** (paradox of enrichment; can drive
predator-prey limit cycles / prey extinction), whereas type-III is **stabilizing**. The config-key
documentation must note that **type-III is the recommended/validated form** and type-II is offered
for completeness/experimentation.
