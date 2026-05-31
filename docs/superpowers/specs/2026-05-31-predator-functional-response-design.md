# Predator Functional Response (aggregate, opt-in) — Design

**Date:** 2026-05-31
**Status:** Approved direction; **substantially revised after round-4 review** retargeted the feature
from a test-only kernel to the live `mortality.py` kernel and fixed a conservation-math error.
Pending re-review of the rewrite → implementation plan.
**Author:** brainstormed collaboratively

## Motivation

OSMOSE's current predation is effectively **Holling type-I with a ration ceiling**: each predator
school eats `min(total_available, max_eatable)` per cell per sub-timestep, where `max_eatable =
biomass × ingestion_rate / (n_dt_per_year × n_subdt)`. From the predator's side this already
saturates; what is **linear** is the *per-prey mortality* — when prey is scarce, predators still take
a near-constant *fraction* of available prey, with **no prey refuge at low density**. This feature
adds **selectable, opt-in, per-predator-species functional-response forms** (type-I / type-II /
type-III) that change that low-density behavior, mirroring the recruitment feature (Shepherd/B-H, PR
#50): a per-species `shape` config key, **default-off**, **bit-exact-preserving** when off, plus a
dedicated Baltic calibration phase validated at the **process level**.

### THE LIVE KERNEL (corrects an earlier draft)

The production simulation path is `simulate.py:_mortality → mortality.py:mortality() →
_apply_predation_numba` (numba, called from 3 sites) **/** `_apply_predation_for_school` (Python
fallback, 1 site). The `predation.py` module's `predation_for_cell` / `_predation_in_cell_*` /
`_predation_on_resources` are **test-only** (referenced only in docstrings; never called by
`simulate.py`/`mortality.py`). All earlier "single injection point in `predation.py`" framing was
against the test-only path and is **void**.

In the live kernel, **school prey and resource (LTL plankton/benthos) prey are accumulated into one
pooled `total_available`**, and `eaten_total = min(total_available, max_eatable)` is computed **once**
(`mortality.py` Python `:484`, numba `:952`; `success = min(eaten_total/max_eatable, 1.0)` at `:541`
/ `:993`). There is **no** separate resource-consumption site and **no** `remaining`-appetite
backfill in production — those exist only in the test-only `predation.py`.

### Design decision: FR acts on the COMBINED pool

The functional response is applied at the single live injection point with `r = total_available /
max_eatable`, where `total_available` is the **fused fish + resource** pool. We do **not** split the
pool (that would restructure the hot, Java-parity-anchored kernel — high risk, and the fusion is
deliberate).

**What this means ecologically (honest):**
- For **near-pure piscivores** (GreySeal, Cormorant, pikeperch), `total_available ≈ fish`, so the
  functional response behaves as intended on the ICES-calibrated fish stocks: a type-III refuge
  reduces the *fraction* of fish taken at low fish density.
- For **cod**, abundant benthos (sp13) keeps `total_available` (hence `r`) high, so cod is usually
  near-satiated and the refuge **rarely triggers** — the resource pool dilutes the effect. This is a
  real limitation of the combined-pool approach, stated up front. Cod is retained in the calibration
  set but the diagnostic must report it honestly (the effect may be small for cod).
- There is **no prey-switching** (aggregate response; the single proportional `share` depletes all
  pooled prey together) and **no "protect-fish-switch-to-benthos" mechanism** (that was an artifact
  of the test-only kernel). The aggregate refuge engages only when **total** accessible prey is low.

### Scope (locked)

- **Aggregate** (combined-pool), not per-prey-type. Per-prey-type prey-switching is **deferred**
  (restructures the kernel; PR #50 diagnosed the percid overshoot as a *spatial* grid-under-
  resolution problem, not a predation-form failure).
- **Selectable forms** type-I / type-II / type-III, per-predator-species, opt-in, default-off.

## Section 1 — Engine math

Single live injection point (both kernels): replace `eaten_total = min(total_available, max_eatable)`
with `eaten_total = max_eatable × g(r)`, `r = total_available / max_eatable`.

| Shape | enum / int | `g(r)` (before clamp) | Behavior |
|-------|------------|-----------------------|----------|
| type-I (default) | `type1` / `1` | `min(r, 1)` | **Literal existing branch** — bit-exact |
| type-II | `type2` / `2` | `r / (r + K)` | Saturating (Holling-disc) |
| type-III | `type3` / `3` | `r² / (r² + K²)` | Sigmoid; low-density refuge (smaller *fraction* of available taken at low r) |

**Conservation clamp (mandatory — corrects a false claim in an earlier draft).** Conservation
requires `eaten_total ≤ total_available`, i.e. `g(r) ≤ r` (since `total_available = r·max_eatable`),
**not** merely `g(r) ≤ 1`. The raw Holling forms violate this at low K:
- type-II `r/(r+K) ≤ r ⟺ r+K ≥ 1` — fails for `r < 1−K` (any K<1).
- type-III `r²/(r²+K²) ≤ r ⟺ r²−r+K² ≥ 0` — fails for `r ∈ (r₋, r₊)` with roots `(1±√(1−4K²))/2`;
  at K=0.1 that is `r ∈ (≈0.01, ≈0.99)` — almost the entire food-limited range.
  type-III is conservative for **all r when K ≥ 0.5** (negative discriminant).

Therefore the engine applies **`g(r) := min(g_form(r), min(r, 1))`** for type-II/III ("ration-capped
Holling"). This (a) guarantees `eaten_total ≤ total_available` so the proportional `share`
redistribution stays conservative and `preyed_biomass`/`pred_success_rate` are not corrupted, and
(b) preserves the refuge where it genuinely exists (low r, where `g_form(r) < r`). type-I is itself
the cap (`min(r,1)`), so the clamp is consistent across forms.

- `K` = dimensionless **ration-relative** half-saturation per predator species (food in units of the
  predator's per-subdt max ration). Valid Holling reparameterization: with `Imax = max_eatable`,
  half-saturation prey level `N½`, `f = Imax·r/(r+K)` for `r=N/Imax`, `K=N½/Imax`.
- **Config-allowed range `K ∈ [0.1, 5.0]`**; **phase-14 DE bound `[0.5, 5.0]` for type-III** so the
  clamp essentially never engages and the type-III curve is clean (K≥0.5 ⇒ conservative everywhere).
- **K is well-scaled for DE, not a transferable biological constant** (it depends on local cell prey
  density, predator packing, and the subdt discretization; a Baltic K need not mean the same refuge
  threshold on EEC/BoB).
- **K does double duty** (refuge shape *and* a mean-intake scaler — type-II/III reduce realized
  intake at all finite r, e.g. K=1,r=1 → g=0.5). Handled by holding ingestion rate fixed in
  calibration (§5).
- **type-III Hill exponent fixed at 2** (YAGNI: no general exponent).
- **Bit-exact guarantee + fencing (mandatory).** type-I dispatches to the *exact existing two
  statements* — `eaten_total = min(total_available, max_eatable)` and `success =
  min(eaten_total/max_eatable, 1.0)` — preserved **verbatim**. On `fr_shape[sp_pred] == 1`, **neither
  `r` nor any new arithmetic is evaluated** on the path producing `eaten_total`/`success`. Do **not**
  "unify" type-I into `max_eatable·g(r)`: the multiply-then-divide round-trip `max_eatable·(av/max)`
  can differ from `min(av,max)` by 1 ULP and break 12/12 parity. (Verified: `r/(r+K)` at small K → 1
  everywhere r>0, so type-I cannot be emulated by a limiting K either.)
- **Division-by-zero already guarded upstream:** both kernels `continue` when `max_eatable ≤ 0` and
  when `total_available ≤ 0` *before* the injection point, so `r` is finite/positive. The FR branch
  must stay **below** those guards.
- **Determinism preserved:** `g(r)` is a pure pointwise function; no new RNG, ordering, or iteration.

## Section 2 — Config schema

Two new per-species keys (`osmose/schema/predation.py` + `config.py`), modeled on the recruitment
`shape` block:
- `predation.functional.response.shape.sp{i}` → enum `type1 | type2 | type3`, **default `type1`**.
- `predation.functional.response.halfsat.sp{i}` → float `K`, range `[0.1, 5.0]`, **required iff**
  `shape ≠ type1`.

**Strict validation + parse-time bound enforcement.** Raise (a) if shape ∉ enum; (b) if shape ≠ type1
and halfsat absent, with message `"predation.functional.response.halfsat.sp{i} is required when
predation.functional.response.shape.sp{i} = {shape}"` (test asserts the `is required when`
substring); (c) if halfsat present and **outside `[0.1, 5.0]`** — enforce the bound at parse with a
raising error, *not* only as a DE search bound (the recruitment `shepherd_beta>0` precedent checks
only `>0`; we add the upper bound too, so a hand-edited `halfsat=0.0` cannot silently yield `g≡1`).

**Enum→int mapping** `type1→1, type2→2, type3→3`, performed at parse time; **both** the focal path
(`config.py`) and the background-predator path (`background.py`) must produce identical codes.

**Array sizing (critical) — exact layout.** `fr_shape` (`np.int32`) and `fr_halfsat` (`np.float64`)
are sized `n_total = n_species + n_background` (**10 for Baltic** = 8 focal + 2 background) — the same
length as `ingestion_rate` — and indexed by **runtime** `sp_pred = species_id[p_idx]`. Numbering:
- **LTL resources (config sp8–13) are NOT `species_id` slots** (separate `ResourceState`); they get
  **no** `fr_shape` entry.
- **Background predators' config keys are `sp14`/`sp15`** but their **runtime `species_id` =
  `n_focal + bkg_idx` = 8 / 9** (`background.py:378`). The config key `…shape.sp14` is parsed by
  `background.py` into the background portion at runtime slot 8.
- Build via the **`recruitment_shepherd_beta` precedent**: focal values concatenated with
  `np.full(n_bkg, default)` for background (`fr_shape` bkg default `1`; `fr_halfsat` bkg default a
  fixed inert sentinel, since type1 never reads it); no-background path uses focal arrays directly.
- **Register both arrays in `EngineConfig.__post_init__`'s `per_species_arrays` length-check dict** —
  the dict is hardcoded; without registration an `n_species`-length array passes validation and
  GreySeal/Cormorant FR **silently no-ops** (the exact bug this section prevents).

**Config-validation allowlist (per CLAUDE.md).** Read both keys via literal-prefix f-strings so the
AST walker auto-captures them (else add to `_SUPPLEMENTARY_ALLOWLIST`); keep
`test_from_dict_warn_mode_clean_on_example_configs[*]` warning-free.

**Applicability to prey-only species.** Validation is uniform (enum + halfsat-required-iff fire for
any `sp{i}`); a non-`type1` shape on a prey-only/LTL species is **accepted but inert** (the kernel
only reads `fr_shape[sp_pred]`, which prey-only species never occupy).

**Back-compat.** Every existing config omits both keys ⇒ `type1` ⇒ unchanged; explicit `type1` and
absent-key are byte-identical. **No migration.**

**Documentation.** Add both keys to the config reference (alongside the recruitment `shape` key).
Note that **type-II is classically destabilizing** (paradox of enrichment) and **type-III is the
recommended/validated form**.

## Section 3 — Kernel changes (live `mortality.py`)

- Add `fr_shape` (`np.int32[n_total]`) and `fr_halfsat` (`np.float64[n_total]`) as `EngineConfig`
  fields, built alongside `ingestion_rate`.
- Branch on `fr_shape[sp_pred]` at the injection point in **both** kernels — `_apply_predation_numba`
  (numba) and `_apply_predation_for_school` (Python). `== 1` keeps the verbatim existing two
  statements; `== 2/3` apply the clamped formula using `r*r` (not `r**2`).
- Thread the two new args from the **4 call sites** (`_apply_predation_numba` ×3,
  `_apply_predation_for_school` ×1) — `config` is in scope at each, so pass `config.fr_shape` /
  `config.fr_halfsat`.
- The loop is **not** restructured; one clamped branch at one injection point per kernel. Numba
  recompiles once on the additive signature change.

## Section 4 — Testing (new `tests/test_engine_functional_response.py`)

**Curve math (with exact-value anchors, matching the recruitment test house style):**
- type-I reproduces `min(r,1)` exactly; explicit `type1` == absent-key.
- type-II/III exact anchors (e.g. K=1, r=1 → g=0.5 before clamp); monotonic; `g(r) → 1` as `r → ∞`.
- type-II initial slope ≈ 1/K at small r (no-refuge) vs type-III zero slope (refuge) — distinguishes
  the forms.
- **Conservation (load-bearing):** assert the *clamped* `g(r) ≤ min(r, 1)` (⇒ `eaten_total ≤
  total_available`) for all r across **K ∈ {0.1, 0.5, 1, 5}** (exercise both endpoints + the clamp
  region); assert `g(r) ≤ 1` alone is NOT what's tested.
- **type-III refuge (operationalized):** on a strict `r < K` grid, `g_form(r)/r` increasing and
  `g(small r) < r`.

**Config / parse:**
- strict validation (shape enum; halfsat-required substring; halfsat out-of-`[0.1,5.0]` raises).
- enum→int mapping asserted **on both paths**: `shape.sp0=type2 → fr_shape[0]==2`; `shape.sp14=type3
  → fr_shape[8]==3` (background slot).
- **direct sizing assertion** `len(cfg.fr_shape) == n_species + n_background == 10` + slot placement,
  and a `per_species_arrays`-registration regression (mis-sized array must raise).
- accepted-but-inert: non-`type1` on a prey-only species parses and has no runtime effect.

**Kernel behavior:**
- **bit-exact parity:** the existing Java-parity suite passes **unmodified** with FR off (no baseline
  regen); name the suite as the 12/12 enforcement.
- **numba-vs-Python parity with FR on** (type-2 and type-3 configs) — guards single-backend bugs.
- **NaN/guard:** type-3 predator in a cell with `total_available == 0` / no eligible prey → no NaN,
  graceful no-op.
- **background path:** FR on sp14/sp15 changes outcomes (catches the n_species sizing bug).
- **downstream:** with type-3, `pred_success_rate` drops in moderate-prey cells (`r ≈ K`); and a test
  (or explicit spec downgrade) that this feeds starvation/growth — assert the bioenergetic
  consequence, not only the success-rate proxy. **Caveat:** under **bioenergetics mode** growth/
  starvation are driven by `preyed_biomass` re-capped by the allometric `bioen_ingestion_cap`, so the
  `pred_success_rate` path is bypassed and FR composes as a *double-cap*; the Baltic calibration
  config uses neither bioen nor genetics, so this is a documented generic caveat, not a calibration
  bug (see §6).
- **determinism** with FR on.

**Process diagnostic unit tests (the gate's only falsifiable basis — must exist):**
- per-school → per-species diet aggregation by `species_id[p_idx]` that **includes background slots
  8/9** (the stock `aggregate_diet_by_species` excludes them via `focal_mask = species_id <
  n_pred_species` — the diagnostic must NOT use it, or must extend it).
- the diagnostic run calls `enable_diet_tracking` with **column width `n_species + n_background +
  n_resources` (= 16 for Baltic)**, not the production default `n_species + n_background` (= 10) — else
  resource columns ≥10 (cod's benthos = col 13) are silently dropped by the `prey_sp <
  diet_matrix.shape[1]` guard, zeroing exactly the signal the diagnostic needs. Test that
  background-predator and resource columns survive.

## Section 5 — Calibration (phase-14) + evaluation

### Scaffolding & stacking
- `get_phase14_params()` + `phase == "14"` branch in `scripts/calibrate_baltic.py` (existing flat
  phase ladder).
- **Phase-2-style inheritance:** load **all 39 phase-13 params** (16 mortality + 8 fishing + 7
  ssb_half + 8 Shepherd β) as fixed `base_config` overrides; `get_phase14_params()` returns
  **exactly the 4 new K keys** — the 39 frozen params live solely in `base_config` (disjoint from the
  free set, as the phase-2 precedent requires). The 4-D runtime math depends on this.
- **Prerequisite:** no `phase13_results.json` exists on disk. **Decision: commit the PR #50 phase-13
  result as `phase13_results.json`** (recommended over a fresh multi-hour run, unless one is
  independently wanted). This is the first task of the calibration PR.

### Parameter space & runtime
- 4 K params, **type-III fixed**, DE bound `[0.5, 5.0]` (clamp-free type-III). With 39 frozen this is
  4-D: `eff_popsize = max(15, 10×4) = 40`; at ~175 evals/h ≈ 14 min/gen → fits the 12 h cap under
  `--patience 20`. (Unfrozen would be ~430 popsize and not converge — strict freezing is required.)
- Reuses bounded-runtime guards (`--patience 20 --wall-clock-cap-h 12 --checkpoint-every 5`) and
  multi-seed re-ranking.

### Predator selection
- FR on **cod (sp0), pikeperch (sp5), GreySeal (sp14→slot8), Cormorant (sp15→slot9)**.
- Rationale: pikeperch/seal/cormorant are near-pure piscivores where the combined-pool FR acts
  cleanly on fish; cod is the dominant fish piscivore but its refuge is **diluted by benthos**
  (above) — retained, but the diagnostic reports its (possibly small) effect honestly.
- Perch (sp4) excluded for **parameter economy** (it IS a real predator per the diet matrix — not
  "weakly piscivorous"; its own overshoot is the spatial grid-resolution issue, unfixable by FR).

### Confounding & the process diagnostic (required, first-class task)
Fixing ingestion removes only the **direct** `K↔ingestion` confound; K still trades against the
**frozen** phase-13 mortality and recruitment params (the Shepherd β was itself fit under the type-I
predation regime — a dynamical, not just statistical, coupling), and a biomass-only objective cannot
identify the K effect. **Data source:** the diet matrix already records per-(school,prey) eaten
biomass. Realized predation mortality of predator-species p on prey q = (Σ eaten of q by p over the
last 10 yr) / (mean biomass of q over the window), per year. Run the **same calibrated config twice
with diet tracking on at width 16** — FR-off (type-I) vs FR-on (type-III with calibrated K) — and
report each predator's realized mortality on each prey and the FR-on − FR-off delta, using the
per-species aggregation that includes background slots 8/9 (§4).

### Evaluation script
Extend `scripts/evaluate_calibration_vs_ices.py`: add `shepherd-fr` to `--mode` `choices`
(currently `{bh, shepherd}`); inject `shape.sp{0,5,14,15}=type3` (like the `shepherd` branch injects
`stock.recruitment.type`) + calibrated halfsat from the phase-14 JSON `parameters`. Report objective
(FR-on vs phase-13; requires importing the objective wrapper — `make_objective`, exact import to be
pinned in the plan — a new compare capability, not a flag), ICES in-range delta, and the diagnostic.

### Success criteria (gate)
**Binding (go/no-go):** 12/12 Java parity bit-exact with FR off; all §4 opt-in tests green (incl.
numba-vs-Python with FR on, background-sizing, conservation, and the two diagnostic unit tests).

**Reported, not gated (honest, mirroring PR #50):** objective vs phase-13. Disposition:
- Ships as a **calibrated Baltic improvement** iff the objective does not regress **AND** the
  process diagnostic shows a mortality reduction that **exceeds the multi-seed noise band** (analogous
  to PR #50's ±0.012; a bare "some negative delta on the lowest-biomass prey" is NOT sufficient — it
  is structurally guaranteed by the type-III shape and would make the gate non-falsifiable) for ≥1
  predator, **ideally corroborated by movement of that prey toward its ICES range**.
- Otherwise ships as **engine capability only**, explicitly not a Baltic improvement.
- "Converges" = terminates via patience/convergence before the wall-clock cap; if capped, the
  capped-best is reported as capped.

## Section 6 — Cross-feature interactions (notes for the plan)

- **Bioenergetics mode:** not orthogonal. Under bioen, growth/starvation read `preyed_biomass`
  re-capped by `bioen_ingestion_cap`, so FR composes as a **double-cap** and the `pred_success_rate`
  mechanism is bypassed. Baltic calibration uses no bioen → safe; document the generic caveat and do
  not claim FR+bioen is validated.
- **Recruitment (PR #50):** FR changes prey survival → SSB → the frozen Shepherd curve. Acknowledged
  as confounding (§5); the frozen β is itself a top-down-pressure-dependent fit.
- **Ev-OSMOSE / FIE (PR #48):** FR is enabled on cod (sp0), the FIE species in `baltic_ev`. FR + FIE
  on the same species is **unvalidated**; document, do not enable both in calibration.
- **Trophic-level output:** FR shifts the realized diet mix → perturbs the TL diagnostic (minor;
  note).
- **Interleaved 4-cause mortality:** predation competes with starvation/additional/fishing for a
  shared depleting `inst_abd` in a per-school-shuffled order. FR reducing predation's bite leaves
  more abundance for other causes that sub-step. FR composes with this (it acts on the `r` computed
  from current `inst_abd`); note that realized fishing mortality on FR-predators' prey is indirectly
  coupled.
- **Multi-stage feeding / accessibility:** `total_available` is already stage- and accessibility-
  filtered before the injection point; FR composes cleanly.

## Section 7 — Delivery (two PRs, mirroring PR #50 structure)

- **PR-A (engine capability):** schema, focal+background parse, EngineConfig fields + array
  concat/registration, kernel branch in both live kernels, all §4 engine/config/parity tests +
  diagnostic unit tests. Gate: 12/12 parity off + opt-in tests. Self-contained; shippable regardless
  of the Baltic outcome.
- **PR-B (Baltic calibration + science):** commit `phase13_results.json`, phase-14, the FR-on/FR-off
  diagnostic, eval `--mode shepherd-fr`. Gate: the process-diagnostic disposition above.

## Deferred / out of scope
- Per-prey-type functional response (prey-switching) — follow-on.
- Splitting the fused fish+resource pool to apply FR to fish-only `available` — rejected here
  (parity-risk); revisit only if combined-pool proves insufficient for cod.
- Perch (sp4) FR; general Hill exponent; type-II in the calibration phase (engine + smoke ships;
  calibration uses type-III); FR+bioen and FR+FIE validation.
- Any change to `bioen_predation.py`.
