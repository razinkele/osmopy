# OSMOSE Python — Fresh Deep Review v3

**Date:** 2026-04-11
**Commit audited:** `e9dd7c0` (master, post deep-review-v2 + session cleanup)
**Test state at review time:** 2099 passed / 15 skipped / 0 failed, ruff clean, parity 12/12 bit-exact.
**Method:** 6 parallel reviewer subagents, each with a distinct lens and an exclusion list to avoid re-reporting fixes that shipped in the 2026-04-05 plan.

**Status:** Findings document only. **No code changes** were made during this review — it is a discovery artifact, not a remediation.

---

## Executive summary

This review is the first post-v0.6.0 deep pass since the 2026-04-05 deep review v2 plan was finalized. 27 new findings across 6 lenses. After filtering for severity and novelty:

- **3 Critical bugs** — all new, all latent, none detectable with current parity tests. One is a confirmed runtime crash for users combining background species with fishing seasonality or discard rates. Two others are silent "feature requested, file missing, feature disappears" anti-patterns in the engine config loader.
- **1 Critical data-corruption risk** — background NetCDF variable name mismatch silently picks the first variable in the file instead of the named one. Arbitrary wrong biomass assignment with a `logger.debug` that is off by default.
- **~10 Important findings** — unenforced type invariants (SchoolState, EngineConfig bioen coupling), missing test coverage of high-risk branches (`_predation_on_resources`, `_average_step_outputs` multi-element path, `out_mortality`, `reproduction` n_eggs < n_schools branch), and architectural debt (550-line `EngineConfig.from_dict` monolith, dual-path mortality maintained by hand).
- **~14 Minor findings** — dead code (`JavaEngine`, `Engine` Protocol), stale TODOs, near-duplicate per-species CSV loaders, one contested architecture finding (state.dirty placement).

**Load-bearing theme:** the v0.6.0 codebase has strong numerical parity but weak defensive programming at subsystem boundaries. The most dangerous pattern is `_resolve_file(file_key) → None` being indistinguishable from "user didn't set this key" — it converts user typos into silent feature removal. Four Critical findings trace to this one pattern.

---

## Critical findings (fix-or-document before next release)

### C-1. Background species + fishing seasonality crashes mortality

**Severity:** Critical (confirmed runtime crash, not parity-visible)
**Location:** `osmose/engine/processes/mortality.py:638` in `_precompute_effective_rates`
**Verified:** 2026-04-11, main-thread spot-check.

**The pattern:**
```python
if config.fishing_seasonality is not None:
    step_in_year = step % config.n_dt_per_year
    season = config.fishing_seasonality[sp, step_in_year]
```

`sp = work_state.species_id` (line 548) includes background species IDs ≥ `config.n_species`.
`config.fishing_seasonality` has shape `(n_sp, n_dt_per_year)` — **focal species only**. Loaded at `config.py:1108` via `_load_fishing_seasonality(cfg, n_sp, ...)` and **never concatenated for background** in the `if n_bkg > 0:` block (lines 1001–1048).

Contrast with `config.fishing_rate` at `config.py:1033`: `fishing_rate = np.concatenate([fishing, bkg_zeros_f])` — properly extended. `fishing_seasonality` was missed.

**Consequence:** Any config that pairs background species with a non-None `fishing_seasonality` raises `IndexError: index X is out of bounds for axis 0 with size n_sp` on the first step. BoB parity tests pass because BoB has no background species.

**Fix sketch:** Concatenate `np.zeros((n_bkg, n_dt_per_year))` onto `fishing_seasonality` in the `n_bkg > 0` block, matching the `fishing_rate` pattern. One line at the right place.

---

### C-2. Background species + fishing discard rate — same bug, same file

**Severity:** Critical
**Location:** `osmose/engine/processes/mortality.py:647`
**Verified:** 2026-04-11.

```python
if config.fishing_discard_rate is not None:
    fishing_discard = np.where(eff_fishing > 0, config.fishing_discard_rate[sp], 0.0)
```

`fishing_discard_rate` loaded with `n_sp` at `config.py:1111`. Never padded. Same crash vector as C-1.

**Fix sketch:** Concatenate `np.zeros(n_bkg)` onto `fishing_discard_rate` in the background-padding block. One line.

---

### C-3. Missing fisheries catchability file silently zeroes fishing

**Severity:** Critical (silent wrong simulation output)
**Location:** `osmose/engine/config.py:183-186`

```python
catch_file = cfg.get("fisheries.catchability.file", "")
catch_path = _resolve_file(catch_file, _cfg_dir(cfg))
if catch_path is None:
    return fishing_rate, fishing_a50, fishing_sel_type, fishing_l50, fishing_slope
```

`_resolve_file` returns `None` for both "key not set" and "key set, file missing." The caller treats the second case as the first — user configured fishing, typoed the path, simulation runs with all-zero fishing rates, no warning. Simulation results look plausible, are biologically wrong, and there is no log line to grep for.

**Fix sketch:** Introduce `_require_file(file_key, config_dir)` that raises `FileNotFoundError` when `file_key` is non-empty and resolution fails. Convert all "optional feature loaders" to use `_require_file` when the key is present, `_resolve_file` only when the key is truly optional. Findings C-4, C-5, C-6, C-7 (below) are all instances of the same pattern and should be fixed as one batch.

---

### C-4. Species-specific fishing map silently substituted by shared map

**Severity:** Critical (silent wrong simulation output)
**Location:** `osmose/engine/config.py:848-856`

```python
sp_map_file = cfg.get(f"mortality.fishing.spatial.distrib.file.sp{i}", "")
if sp_map_file:
    sp_path = _resolve_file(sp_map_file, _cfg_dir(cfg))
    if sp_path is not None:
        focal_fishing_spatial_maps.append(_load_spatial_csv(sp_path))
    else:
        focal_fishing_spatial_maps.append(shared_fishing_map)
```

Species i's spatial fishing distribution was explicitly set by the user. The file path is typoed. The code silently falls back to the generic shared map. Species i is now fished over the wrong spatial pattern.

**Fix sketch:** Remove the `else: append(shared_fishing_map)` branch — when the user specified a per-species file, a missing file is a config error, not a fallback trigger. Raise, or at minimum `_log.warning` loudly.

---

### C-5, C-6, C-7. Same pattern in three more loaders

All three follow the same anti-pattern (key non-empty → resolve returns None → silently skip). Noted here by location; fix them in one batch with C-3 and C-4.

| ID | File | Feature silently disabled |
|---|---|---|
| C-5 | `config.py:474-484` | `mortality.additional.rate.bytdt.file.sp{i}` — time-varying additional mortality |
| C-6 | `config.py:497-506` | `mortality.additional.rate.spatial.file.sp{i}` — spatial additional mortality |
| C-7 | `config.py:333-341` | `mortality.fishing.rate.byyear.file.sp{i}` — time-varying fishing rate |

Each one converts a user typo into "feature disappears silently." Severity Critical because the user sees no error but gets wrong simulation results.

---

### C-8. Background NetCDF variable name mismatch silently picks first variable

**Severity:** Critical (silent data corruption)
**Location:** `osmose/engine/background.py:305-315`

```python
if stripped in ds:
    da = ds[stripped]
else:
    first_var = list(ds.data_vars)[0]
    da = ds[first_var]
    logger.debug("NetCDF variable '%s' not found ...", stripped, ...)
```

User configures background species "plankton" with NetCDF variable "plankton_biomass." The NetCDF file actually stores the variable as "phytoplankton". The lookup fails, code falls back to the **first** variable in the file — which might be longitude, time, or an unrelated biomass field. Wrong data is silently substituted into simulation biomass. `logger.debug` is off by default in production.

**Fix sketch:** Raise `KeyError(f"NetCDF variable {stripped!r} not found; available: {list(ds.data_vars)}")`. Don't guess which variable the user meant.

---

## Important findings (should fix in a 3-6 task mini-plan)

### I-1. `SchoolState` has no biological invariants

**Location:** `osmose/engine/state.py:30` `__post_init__`
**Reviewer lens:** Type design.

Validation is limited to array lengths + `n_dead` 2D shape. The class allows `abundance < 0`, `biomass < 0`, `feeding_stage >= n_feeding_stages`, `cell_x >= grid.nx`, `is_egg && age_dt > 0`, and `biomass != abundance * weight` — all of which would represent actual simulation corruption.

**Fix sketch:** Add an opt-in `validate(debug=False)` method with biological invariants, called from tests and from an `OSMOSE_DEBUG=1` env hook in `__post_init__`. Zero cost in production, catches corruption at the step it appears in debug mode.

---

### I-2. `EngineConfig.bioen_*` fields are coupled but independently Optional

**Location:** `osmose/engine/config.py:509` + 20+ bioen fields
**Reviewer lens:** Type design.

21 `bioen_*` arrays declared `NDArray | None = None` and gated on `bioen_enabled: bool`. Nothing enforces that `bioen_enabled=True` requires all 21 non-None. A partial config crashes deep in a mortality kernel instead of at config load.

Same issue for the seven `gompertz_*` fields coupled to `"GOMPERTZ" in growth_class`, and several `genetics_*` fields.

**Fix sketch:** In `EngineConfig.__post_init__`, if `bioen_enabled`, assert each `bioen_*` is not None with a helpful error message naming the missing field. Or better — extract a nested `BioenParams | None` dataclass so "bioen on" literally means "this struct exists and all fields are set."

---

### I-3. `EngineConfig.from_dict` is a 550-line monolith with a parallel background branch

**Location:** `osmose/engine/config.py:741-1291`
**Reviewer lens:** Architecture.

The single classmethod reads, parses, and constructs 80+ fields by hand. The `if n_bkg > 0:` branch at lines 1001-1048 manually `np.concatenate`-joins every focal array with a background zeros array. Adding a new per-species parameter requires editing the field declaration + the classmethod + the `n_bkg` branch + `__post_init__` in lockstep with **no compiler guard**. Findings C-1 and C-2 are direct consequences of this — someone added `fishing_seasonality` without updating the background-padding block.

**Fix sketch:** Extract `_pad_for_background(focal_array, n_bkg, pad_value=0)` helper and apply it uniformly to every per-species array. Makes the pattern visible and the omission impossible. Consider splitting `from_dict` by subsystem (growth, mortality, fishing, bioen, genetics) into private loader helpers.

---

### I-4. `_predation_on_resources` has no direct behavioral test

**Location:** `osmose/engine/processes/predation.py`
**Reviewer lens:** Test coverage.

The LTL (lower-trophic-level) resource predation path — where focal fish eat plankton — is only exercised via full `simulate()` runs. No test directly feeds a known resource biomass + one focal school, calls the function, and asserts the correct biomass was removed. A bug here would break trophic mass balance silently.

**Fix sketch:** Add `test_predation_on_resources_removes_biomass`: single-cell state, one focal school, `ResourceState` with known biomass, assert resource biomass decreased by expected amount and school `preyed_biomass` increased correspondingly.

---

### I-5. `_average_step_outputs` multi-element branch has no direct test

**Location:** `osmose/engine/simulate.py:812-835`
**Reviewer lens:** Test coverage (consistent with the Phase 2 review's H3 follow-up note).

`test_average_step_outputs_preserves_distributions` uses `freq=1`, exercising only the `len == 1` short-circuit. The `len > 1` branch sums mortality, sums yield, averages biomass, and snapshots the LAST distribution dict — each of those is a separate contract that could silently regress.

**Fix sketch:** Add a parametrized 3-element test with distinct biomass/mortality/yield, asserting biomass is mean, mortality is sum, yield is sum, distribution dict is the last entry.

---

### I-6. `reproduction`'s "fewer eggs than n_schools" branch untested

**Location:** `osmose/engine/processes/reproduction.py:73-74`
**Reviewer lens:** Test coverage.

No test hits the branch where `n_eggs[sp] < n_new`. A bug here silently skips reproduction for rare spawners or creates schools with zero abundance that downstream code may divide by.

**Fix sketch:** Tiny seeding biomass + 20 schools, assert exactly one new egg school is created with `eggs_per_school == n_eggs[sp]`.

---

### I-7. `out_mortality` and `additional_mortality_by_dt` override paths untested

**Location:** `osmose/engine/processes/natural.py:33-41` and the `is_out=True` path
**Reviewer lens:** Test coverage.

No test sets `is_out=True` on a school and asserts mortality is applied at `M_out / n_dt_per_year`. A regression that drops the denominator or forgets to apply the rate goes unseen. Similarly, no test verifies `additional_mortality_by_dt` actually changes `n_dead` relative to the base rate.

**Fix sketch:** Two tests — one for `is_out=True` with analytical expected mortality, one for `additional_mortality_by_dt[0] = [0, 1.0, 0, 1.0]` asserting alternating zero/positive deaths.

---

### I-8. `MPAZone` doesn't validate grid shape or binary values

**Location:** `osmose/engine/config.py:132` `MPAZone.__post_init__`
**Reviewer lens:** Type design.

`grid` is typed `NDArray[np.float64]` and documented as `1 = protected, 0 = not`. No check that `grid.ndim == 2` or that values are in `{0.0, 1.0}`. A 3D array or a continuous-valued grid silently yields wrong MPA application.

**Fix sketch:** One-line asserts in `__post_init__`. `start_year < 0` should also be rejected.

---

### I-9. `_load_fishing_rate_by_year` vs `_load_additional_mortality_by_dt` — near-duplicate

**Location:** `config.py:322-342` vs `config.py:464-484`
**Reviewer lens:** Dead code / over-abstraction.

Two 11-line functions with identical control flow, differing only in the config key pattern. A fix to one (e.g., C-5 or C-7 above) has to land in both or drift silently.

**Fix sketch:** Extract `_load_per_species_timeseries(cfg, n_species, key_pattern) -> list[ndarray | None]`. The two named wrappers become one-liners.

---

### I-10. `JavaEngine` stub + `Engine` Protocol are unused

**Location:** `osmose/engine/__init__.py:17-29` (Protocol) and `102-123` (stub)
**Reviewer lens:** Dead code.

`Engine` Protocol has zero annotations referencing it (verified by grep). `JavaEngine.run()` and `JavaEngine.run_ensemble()` both unconditionally raise `NotImplementedError`. Only references are the definition and two structural hasattr tests.

**Fix sketch:** Delete both. `PythonEngine` stands alone. Remove `Protocol` / `runtime_checkable` imports.

---

## Minor findings (polish, drift prevention)

- **M-1.** `tests/test_engine_temp_function.py::test_phi_t_degenerate_e_d_equals_e_m_falls_back_to_arrhenius` — added this session. No action.
- **M-2.** Redundant `from pathlib import Path as P` at `engine/__init__.py:54`. Delete.
- **M-3.** `_load_accessibility` and `_load_stage_accessibility` both look up the same config key and are always called together. Consolidate.
- **M-4.** 3 stale TODO comments in `osmose/engine/output.py:289-291, 344-345`. Either link to `docs/parity-roadmap.md` or delete.
- **M-5.** `population.seeding.year.max` is one global value applied to all species uniformly. Java may support per-species override; needs verification against Java source.
- **M-6.** `cell_id = cell_y * (resources.grid.nx if resources else 0) + cell_x` at `mortality.py:381` — garbage value if `resources is None`, never used, misleading expression.
- **M-7.** Movement map uncovered slots warn + fallback to random walk silently. H5 (shipped) made the warning aggregated but didn't escalate to an error. Consider surfacing a blocking Shiny notification at Run time instead of a backend log line.
- **M-8.** `_close_spatial_ds` at `ui/pages/spatial_results.py:145-152` has `except Exception: pass`. Should at least log with `exc_info=True`.
- **M-9.** Several UI pages have zero test files: `movement.py`, `fishing.py`, `forcing.py`, `economic.py`, `spatial_results.py`, `diagnostics.py`, `map_viewer.py`. Factor pure helpers and pin with unit tests.
- **M-10.** Weak test: `test_zero_rate_no_mortality` is near-tautological (rate=0 → D=0 → assertion). Strengthen with a second non-zero school.
- **M-11.** Duplicate `test_parse_label` tests in `test_engine_accessibility.py` and `test_engine_fisheries.py` — consolidate.
- **M-12.** Several construction-only assertions in `test_engine_config_validation.py` that test `__init__` accepting placeholder arrays rather than actual behavior.
- **M-13.** `StepOutput` has 4 distribution fields as independent Optionals; they should be two `AgeDistributions` / `SizeDistributions` sub-dataclasses (pair-invariant).
- **M-14.** `SimulationContext` diet fields (`diet_tracking_enabled`, `diet_matrix`, `tl_weighted_sum`) are three-way coupled — consolidate into a single `DietTracking | None`.

---

## Disputed findings (noted, not included above)

- **D-1.** Architecture reviewer flagged `state.dirty.set(True)` inside `reactive.isolate()` in `ui/pages/forcing.py:136-138` as a smell that may break dirty-badge UI updates. The Phase 2 H8 fix in this session deliberately moved the line *into* the isolate block to match the canonical pattern used by the other 4 sync handlers. Writes inside `reactive.isolate()` should still propagate to downstream readers (isolate only affects reads, not writes), so the H8 pattern is likely correct and the architecture reviewer's concern is likely wrong. **Needs Shiny reactive-semantics verification** to resolve definitively. Not actioning.

- **D-2.** Silent-failure reviewer argued the M2 partial-year spawning season warning (shipped this session) should be `raise ValueError` instead of `logger.warning`. This is a design decision — the M2 followup chose warn+normalize because trimming silently discards user data and raising breaks any config that was working before. Accepting current behavior; documenting the decision here.

---

## Suggested next step

Not a plan, a recommendation: bundle the 8 Critical findings into a focused "deep review v3 critical fixes" plan. Scope:

1. **C-1 + C-2** — one commit, concatenate `fishing_seasonality` and `fishing_discard_rate` for background species in the `n_bkg > 0` block. One-time structural fix.
2. **C-3 through C-7** — one commit each. Introduce `_require_file` helper and convert the 5 sites. Tests: verify each raises FileNotFoundError with a useful message when the key is set and the file is missing.
3. **C-8** — one commit. Raise `KeyError` on NetCDF variable lookup failure with `list(ds.data_vars)` in the message.

Total: ~7 commits, ~1.5 hours subagent-driven. No test-count ramp estimate until the v3 plan is written.

Important findings I-1 through I-10 would make a sensible second phase (~10 tasks, 2-3 hours).

Minor findings are drift-prevention; can be bundled into a single "tidy" commit or deferred indefinitely.

---

## Review metadata

**Reviewer agents dispatched (all parallel, background mode):**

| Agent | Lens | Key findings |
|---|---|---|
| `feature-dev:code-explorer` | Architecture + hot paths | 8 findings incl. I-3, I-9 precursor, one false-positive (D-1) |
| `feature-dev:code-reviewer` | Bugs + logic errors | 6 findings incl. C-1, C-2 (the runtime crashes) |
| `pr-review-toolkit:silent-failure-hunter` | Error handling + fallbacks | 8 findings incl. C-3 through C-8 (the systemic file-missing pattern) |
| `pr-review-toolkit:type-design-analyzer` | Type invariants | 8 findings incl. I-1, I-2, I-8, M-13, M-14 |
| `pr-review-toolkit:pr-test-analyzer` | Test coverage gaps | 12 gaps + 3 anti-findings incl. I-4, I-5, I-6, I-7 |
| `pr-review-toolkit:code-simplifier` | Dead code + duplication | 7 findings incl. I-9, I-10, M-2 through M-4 |

**Total raw findings:** ~49 across all reviewers.
**Deduplicated and consolidated into this document:** 8 Critical + 10 Important + 14 Minor + 2 Disputed = 34 distinct items.

**Duplicate rate:** 6 out of 49 raw findings were duplicates across reviewers (e.g., `_average_step_outputs` multi-element branch flagged by both test-coverage and bugs; `JavaEngine` stub flagged by both dead-code and architecture). That 12% duplicate rate is a useful sanity signal that the lens partitioning was effective.

**Parity impact:** None of the findings in this document required running the parity suite to verify. BoB parity is bit-exact because BoB doesn't exercise the edge cases (background species, fishing seasonality/discards, file-resolution failures) where these bugs live.
