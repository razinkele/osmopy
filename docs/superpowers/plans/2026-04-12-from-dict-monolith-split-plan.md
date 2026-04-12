# I-3: EngineConfig.from_dict Monolith Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `EngineConfig.from_dict` (611 lines) into a coordinator that calls 5 subsystem parsers, each returning a dict of parsed fields.

**Architecture:** Pure refactor — extract inline parsing blocks into private module-level functions in `osmose/engine/config.py`. Each helper receives `cfg: dict[str, str]` plus dimensions (`n_sp`, `n_dt`, etc.) and returns a `dict[str, Any]`. The coordinator calls helpers and unpacks results into the `EngineConfig(...)` constructor. Zero behavioral changes.

**Tech Stack:** Python 3.12, NumPy, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-04-12-from-dict-monolith-split-design.md`

**Baseline:** 2148 tests passed, 15 skipped, 0 failed. Ruff clean. Parity 12/12 bit-exact.

---

## Scope & drift audit

All line numbers verified against master HEAD after deep-review-v3 remediation merge (commit `4b5bfab`). Subagents MUST grep for named code patterns rather than trusting line numbers.

---

## File structure

No new files. All changes are in-place modifications to:

**Production:** `osmose/engine/config.py`
**Tests:** None created — existing 2148 tests + 12 parity tests are the safety net.

---

## Task 1: Extract `_parse_growth_params` (lines 864-920)

**Files:**
- Modify: `osmose/engine/config.py`

**Context:** Lines 864-920 parse Von Bertalanffy growth parameters, egg size, allometry, lifespan_dt, lmax, and related focal-species scalars. They depend on `cfg`, `n_sp`, `n_dt`, and `lifespan_years` (already computed at line 829).

- [ ] **Step 1: Read the target block**

Read `osmose/engine/config.py` lines 864-920. Record every local variable assigned. These become the return dict keys:
- `focal_linf`, `focal_k`, `focal_t0`, `focal_egg_size`, `focal_condition_factor`, `focal_allometric_power`, `focal_vb_threshold_age`, `focal_lifespan_dt`, `focal_delta_lmax_factor`, `focal_lmax`

- [ ] **Step 2: Add the helper function**

Above `from_dict` (after the existing helpers like `_load_per_species_timeseries`), add:

```python
def _parse_growth_params(
    cfg: dict[str, str], n_sp: int, n_dt: int, lifespan_years: NDArray[np.float64]
) -> dict[str, Any]:
    """Parse Von Bertalanffy growth, allometry, and lifespan parameters."""
    linf = _species_float(cfg, "species.linf.sp{i}", n_sp)
    k = _species_float(cfg, "species.k.sp{i}", n_sp)
    t0 = _species_float(cfg, "species.t0.sp{i}", n_sp)
    egg_size = _species_float(cfg, "species.egg.size.sp{i}", n_sp)
    condition_factor = _species_float(
        cfg, "species.length2weight.condition.factor.sp{i}", n_sp
    )
    allometric_power = _species_float(
        cfg, "species.length2weight.allometric.power.sp{i}", n_sp
    )
    vb_threshold_age = _species_float(
        cfg, "species.vonbertalanffy.threshold.age.sp{i}", n_sp
    )
    lifespan_dt = (lifespan_years * n_dt).astype(np.int32)
    delta_lmax_factor = _species_float_optional(
        cfg, "species.delta.lmax.factor.sp{i}", n_sp, default=2.0
    )
    lmax = _species_float_optional(cfg, "species.lmax.sp{i}", n_sp, default=0.0)
    for i in range(n_sp):
        if lmax[i] <= 0:
            lmax[i] = linf[i]
    return {
        "focal_linf": linf,
        "focal_k": k,
        "focal_t0": t0,
        "focal_egg_size": egg_size,
        "focal_condition_factor": condition_factor,
        "focal_allometric_power": allometric_power,
        "focal_vb_threshold_age": vb_threshold_age,
        "focal_lifespan_dt": lifespan_dt,
        "focal_delta_lmax_factor": delta_lmax_factor,
        "focal_lmax": lmax,
    }
```

- [ ] **Step 3: Replace the inline block in `from_dict`**

Replace lines 864-920 (from `focal_species_names = _focal_names` through the lmax defaulting loop) with:

```python
        focal_species_names = _focal_names
        _growth = _parse_growth_params(cfg, n_sp, n_dt, lifespan_years)
        focal_linf = _growth["focal_linf"]
        focal_k = _growth["focal_k"]
        focal_t0 = _growth["focal_t0"]
        focal_egg_size = _growth["focal_egg_size"]
        focal_condition_factor = _growth["focal_condition_factor"]
        focal_allometric_power = _growth["focal_allometric_power"]
        focal_vb_threshold_age = _growth["focal_vb_threshold_age"]
        focal_lifespan_dt = _growth["focal_lifespan_dt"]
        focal_delta_lmax_factor = _growth["focal_delta_lmax_factor"]
        focal_lmax = _growth["focal_lmax"]
```

- [ ] **Step 4: Run parity tests**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q`
Expected: 12 passed. This is the critical gate — bit-exact parity confirms zero behavioral change.

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 2148 passed, 15 skipped.

- [ ] **Step 6: Ruff**

Run: `.venv/bin/ruff check osmose/engine/config.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/config.py
git commit -m "refactor(engine): extract _parse_growth_params from from_dict (I-3 step 1/5)"
```

---

## Task 2: Extract `_parse_reproduction_params` (scattered ~40 lines)

**Files:**
- Modify: `osmose/engine/config.py`

**Context:** Lines 887-914 parse sex_ratio, relative_fecundity, maturity_size, seeding_biomass, seeding_max_step, larva_mortality_rate, maturity_age_dt. These are currently interspersed within the growth block (after Task 1 extraction, they'll be right after the `_growth` call).

- [ ] **Step 1: Identify remaining reproduction lines**

After Task 1, grep for these variables still assigned inline in `from_dict`:
- `focal_sex_ratio`, `focal_relative_fecundity`, `focal_maturity_size`, `focal_seeding_biomass`, `focal_seeding_max_step`, `focal_larva_mortality_rate`, `focal_maturity_age_dt`

- [ ] **Step 2: Add the helper function**

```python
def _parse_reproduction_params(
    cfg: dict[str, str],
    n_sp: int,
    n_dt: int,
    lifespan_years: NDArray[np.float64],
) -> dict[str, Any]:
    """Parse reproduction, seeding, and larva mortality parameters."""
    sex_ratio = _species_float_optional(cfg, "species.sexratio.sp{i}", n_sp, default=0.5)
    relative_fecundity = _species_float_optional(
        cfg, "species.relativefecundity.sp{i}", n_sp, default=500.0
    )
    maturity_size = _species_float_optional(
        cfg, "species.maturity.size.sp{i}", n_sp, default=0.0
    )
    seeding_biomass = _species_float_optional(
        cfg, "population.seeding.biomass.sp{i}", n_sp, default=0.0
    )
    # Seeding max step: explicit override or default to lifespan
    seeding_max_year_str = cfg.get("population.seeding.year.max", "")
    if seeding_max_year_str:
        seeding_max_years = float(seeding_max_year_str)
        seeding_max_step = np.full(n_sp, int(seeding_max_years * n_dt), dtype=np.int32)
    else:
        seeding_max_step = (lifespan_years * n_dt).astype(np.int32)
    larva_mortality_rate = _species_float_optional(
        cfg, "mortality.additional.larva.rate.sp{i}", n_sp, default=0.0
    )
    maturity_age_years = _species_float_optional(
        cfg, "species.maturity.age.sp{i}", n_sp, default=0.0
    )
    maturity_age_dt = (maturity_age_years * n_dt).astype(np.int32)
    return {
        "focal_sex_ratio": sex_ratio,
        "focal_relative_fecundity": relative_fecundity,
        "focal_maturity_size": maturity_size,
        "focal_seeding_biomass": seeding_biomass,
        "focal_seeding_max_step": seeding_max_step,
        "focal_larva_mortality_rate": larva_mortality_rate,
        "focal_maturity_age_dt": maturity_age_dt,
    }
```

- [ ] **Step 3: Replace inline block in `from_dict`**

Replace the 7 variable assignments with:

```python
        _repro = _parse_reproduction_params(cfg, n_sp, n_dt, lifespan_years)
        focal_sex_ratio = _repro["focal_sex_ratio"]
        focal_relative_fecundity = _repro["focal_relative_fecundity"]
        focal_maturity_size = _repro["focal_maturity_size"]
        focal_seeding_biomass = _repro["focal_seeding_biomass"]
        focal_seeding_max_step = _repro["focal_seeding_max_step"]
        focal_larva_mortality_rate = _repro["focal_larva_mortality_rate"]
        focal_maturity_age_dt = _repro["focal_maturity_age_dt"]
```

- [ ] **Step 4: Parity + full suite + ruff**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q` → 12 passed
Run: `.venv/bin/python -m pytest tests/ -q` → 2148 passed
Run: `.venv/bin/ruff check osmose/engine/config.py`

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py
git commit -m "refactor(engine): extract _parse_reproduction_params from from_dict (I-3 step 2/5)"
```

---

## Task 3: Extract `_parse_predation_params` (lines 921-1057)

**Files:**
- Modify: `osmose/engine/config.py`

**Context:** This is the largest block — 136 lines parsing feeding stages, size ratios (with multi-value support), ingestion rate, critical success rate, background species feeding, and 2D array padding. It produces: `focal_ingestion_rate`, `focal_critical_success_rate`, `all_thresholds`, `all_metrics`, `all_ratio_min`, `all_ratio_max`, `n_feeding_stages`, `size_ratio_min_2d`, `size_ratio_max_2d`, `focal_starvation_rate_max`.

- [ ] **Step 1: Read the full block carefully**

Read from the fishing spatial maps section (after Task 1/2 extractions) through the 2D array padding. Note: this block uses `background_list` and `n_bkg` which are computed earlier in `from_dict`. These must be passed as parameters.

- [ ] **Step 2: Add the helper function**

```python
def _parse_predation_params(
    cfg: dict[str, str],
    n_sp: int,
    background_list: list[BackgroundSpeciesInfo],
) -> dict[str, Any]:
    """Parse feeding stages, size ratios, and predation parameters."""
    n_bkg = len(background_list)
    focal_ingestion_rate = _species_float(cfg, "predation.ingestion.rate.max.sp{i}", n_sp)
    focal_critical_success_rate = _species_float(
        cfg, "predation.efficiency.critical.sp{i}", n_sp
    )
    focal_starvation_rate_max = _species_float_optional(
        cfg, "mortality.starvation.rate.max.sp{i}", n_sp, default=0.0
    )

    _VALID_METRICS = {"age", "size", "weight", "tl"}
    global_metric = cfg.get("predation.predprey.stage.structure", "size").strip().lower()
    if global_metric not in _VALID_METRICS:
        raise ValueError(
            f"Unrecognized feeding stage metric: {global_metric!r}. "
            f"Must be one of {sorted(_VALID_METRICS)}."
        )

    all_thresholds: list[list[float]] = []
    all_metrics: list[str] = []
    all_ratio_min: list[list[float]] = []
    all_ratio_max: list[list[float]] = []

    for i in range(n_sp):
        sp_metric = cfg.get(f"predation.predprey.stage.structure.sp{i}", "").strip().lower()
        if not sp_metric:
            sp_metric = global_metric
        elif sp_metric not in _VALID_METRICS:
            raise ValueError(
                f"Unrecognized feeding stage metric for sp{i}: {sp_metric!r}. "
                f"Must be one of {sorted(_VALID_METRICS)}."
            )
        all_metrics.append(sp_metric)

        thresh_raw = cfg.get(f"predation.predprey.stage.threshold.sp{i}", "")
        if not thresh_raw or thresh_raw.strip().lower() == "null":
            sp_thresholds: list[float] = []
        else:
            sp_thresholds = _parse_floats(thresh_raw)
        all_thresholds.append(sp_thresholds)
        n_stages = len(sp_thresholds) + 1

        rmin_raw = cfg.get(f"predation.predprey.sizeratio.min.sp{i}", "1.0")
        rmax_raw = cfg.get(f"predation.predprey.sizeratio.max.sp{i}", "3.5")
        rmin_list = _parse_floats(rmin_raw)
        rmax_list = _parse_floats(rmax_raw)

        if len(rmin_list) != n_stages:
            raise ValueError(
                f"Size ratio min count mismatch for sp{i}: "
                f"got {len(rmin_list)}, expected {n_stages} stages"
            )
        if len(rmax_list) != n_stages:
            raise ValueError(
                f"Size ratio max count mismatch for sp{i}: "
                f"got {len(rmax_list)}, expected {n_stages} stages"
            )

        for s in range(n_stages):
            if rmin_list[s] > rmax_list[s]:
                warnings.warn(
                    f"Swapping size ratios for sp{i} stage {s}: "
                    f"min={rmin_list[s]}, max={rmax_list[s]}",
                    stacklevel=2,
                )
                rmin_list[s], rmax_list[s] = rmax_list[s], rmin_list[s]

        all_ratio_min.append(rmin_list)
        all_ratio_max.append(rmax_list)

    # Background species
    for b in background_list:
        b_idx = b.file_index
        b_metric = cfg.get(
            f"predation.predprey.stage.structure.sp{b_idx}", ""
        ).strip().lower()
        if not b_metric:
            b_metric = global_metric
        all_metrics.append(b_metric)

        thresh_raw = cfg.get(f"predation.predprey.stage.threshold.sp{b_idx}", "")
        if not thresh_raw or thresh_raw.strip().lower() == "null":
            b_thresholds: list[float] = []
        else:
            b_thresholds = _parse_floats(thresh_raw)
        all_thresholds.append(b_thresholds)
        n_stages = len(b_thresholds) + 1

        rmin_list = list(b.size_ratio_min)
        rmax_list = list(b.size_ratio_max)
        if len(rmin_list) == 1 and n_stages > 1:
            rmin_list = rmin_list * n_stages
        if len(rmax_list) == 1 and n_stages > 1:
            rmax_list = rmax_list * n_stages
        for s in range(min(len(rmin_list), len(rmax_list))):
            if rmin_list[s] > rmax_list[s]:
                rmin_list[s], rmax_list[s] = rmax_list[s], rmin_list[s]
        all_ratio_min.append(rmin_list)
        all_ratio_max.append(rmax_list)

    # Build 2D arrays
    n_total = n_sp + n_bkg
    max_stages = max((len(r) for r in all_ratio_min), default=1)
    n_feeding_stages = np.array([len(r) for r in all_ratio_min], dtype=np.int32)
    size_ratio_min_2d = np.zeros((n_total, max_stages), dtype=np.float64)
    size_ratio_max_2d = np.zeros((n_total, max_stages), dtype=np.float64)
    for sp_i in range(n_total):
        n_st = len(all_ratio_min[sp_i])
        for s in range(n_st):
            size_ratio_min_2d[sp_i, s] = all_ratio_min[sp_i][s]
            size_ratio_max_2d[sp_i, s] = all_ratio_max[sp_i][s]
        if n_st > 0 and n_st < max_stages:
            size_ratio_min_2d[sp_i, n_st:] = all_ratio_min[sp_i][-1]
            size_ratio_max_2d[sp_i, n_st:] = all_ratio_max[sp_i][-1]

    return {
        "focal_ingestion_rate": focal_ingestion_rate,
        "focal_critical_success_rate": focal_critical_success_rate,
        "focal_starvation_rate_max": focal_starvation_rate_max,
        "all_thresholds": all_thresholds,
        "all_metrics": all_metrics,
        "n_feeding_stages": n_feeding_stages,
        "size_ratio_min_2d": size_ratio_min_2d,
        "size_ratio_max_2d": size_ratio_max_2d,
    }
```

- [ ] **Step 3: Replace inline block in `from_dict`**

Replace the entire predation/feeding section with:

```python
        _pred = _parse_predation_params(cfg, n_sp, background_list)
        focal_ingestion_rate = _pred["focal_ingestion_rate"]
        focal_critical_success_rate = _pred["focal_critical_success_rate"]
        focal_starvation_rate_max = _pred["focal_starvation_rate_max"]
        all_thresholds = _pred["all_thresholds"]
        all_metrics = _pred["all_metrics"]
        n_feeding_stages = _pred["n_feeding_stages"]
        size_ratio_min_2d = _pred["size_ratio_min_2d"]
        size_ratio_max_2d = _pred["size_ratio_max_2d"]
```

- [ ] **Step 4: Parity + full suite + ruff**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q` → 12 passed
Run: `.venv/bin/python -m pytest tests/ -q` → 2148 passed
Run: `.venv/bin/ruff check osmose/engine/config.py`

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py
git commit -m "refactor(engine): extract _parse_predation_params from from_dict (I-3 step 3/5)"
```

---

## Task 4: Extract `_merge_focal_background` (lines 1085-1164)

**Files:**
- Modify: `osmose/engine/config.py`

**Context:** 79 lines that concatenate ~30 focal-only arrays with background species defaults (zeros for most fields, actual values for ingestion_rate, condition_factor, allometric_power). Has two branches: `if n_bkg > 0:` (concatenation) and `else:` (identity copy). Both must be handled by the helper.

- [ ] **Step 1: Read the block and catalogue all variables**

Read lines 1085-1164. Record every variable assigned in both the `n_bkg > 0` and `else` branches. There should be ~30 variable names assigned in each branch.

- [ ] **Step 2: Add the helper function**

The helper takes all focal arrays plus `background_list` and returns a dict with merged (non-prefixed) names:

```python
def _merge_focal_background(
    focal: dict[str, Any],
    background_list: list[BackgroundSpeciesInfo],
    focal_species_names: list[str],
    focal_fishing_spatial_maps: list[np.ndarray | None],
    focal_movement_method: list[str],
) -> dict[str, Any]:
    """Concatenate focal-species arrays with background defaults.

    Returns a dict with the merged (non-prefixed) variable names used by the
    EngineConfig constructor.
    """
    n_bkg = len(background_list)
    if n_bkg > 0:
        bkg_names = [b.name for b in background_list]
        bkg_ingestion = np.array([b.ingestion_rate for b in background_list])
        bkg_condition_factor = np.array([b.condition_factor for b in background_list])
        bkg_allometric_power = np.array([b.allometric_power for b in background_list])
        bkg_zeros_f = np.zeros(n_bkg, dtype=np.float64)
        bkg_zeros_i = np.zeros(n_bkg, dtype=np.int32)

        return {
            "all_species_names": focal_species_names + bkg_names,
            "linf": np.concatenate([focal["focal_linf"], bkg_zeros_f]),
            "k": np.concatenate([focal["focal_k"], bkg_zeros_f]),
            "t0": np.concatenate([focal["focal_t0"], bkg_zeros_f]),
            "egg_size": np.concatenate([focal["focal_egg_size"], bkg_zeros_f]),
            "condition_factor": np.concatenate([focal["focal_condition_factor"], bkg_condition_factor]),
            "allometric_power": np.concatenate([focal["focal_allometric_power"], bkg_allometric_power]),
            "vb_threshold_age": np.concatenate([focal["focal_vb_threshold_age"], bkg_zeros_f]),
            "lifespan_dt": np.concatenate([focal["focal_lifespan_dt"], bkg_zeros_i]),
            "ingestion_rate": np.concatenate([focal["focal_ingestion_rate"], bkg_ingestion]),
            "critical_success_rate": np.concatenate([focal["focal_critical_success_rate"], bkg_zeros_f]),
            "delta_lmax_factor": np.concatenate([focal["focal_delta_lmax_factor"], bkg_zeros_f]),
            "additional_mortality_rate": np.concatenate([focal["focal_additional_mortality_rate"], bkg_zeros_f]),
            "sex_ratio": np.concatenate([focal["focal_sex_ratio"], bkg_zeros_f]),
            "relative_fecundity": np.concatenate([focal["focal_relative_fecundity"], bkg_zeros_f]),
            "maturity_size": np.concatenate([focal["focal_maturity_size"], bkg_zeros_f]),
            "seeding_biomass": np.concatenate([focal["focal_seeding_biomass"], bkg_zeros_f]),
            "seeding_max_step": np.concatenate([focal["focal_seeding_max_step"], bkg_zeros_i]),
            "larva_mortality_rate": np.concatenate([focal["focal_larva_mortality_rate"], bkg_zeros_f]),
            "maturity_age_dt": np.concatenate([focal["focal_maturity_age_dt"], bkg_zeros_i]),
            "lmax": np.concatenate([focal["focal_lmax"], bkg_zeros_f]),
            "starvation_rate_max": np.concatenate([focal["focal_starvation_rate_max"], bkg_zeros_f]),
            "fishing_rate": np.concatenate([focal["fishing"], bkg_zeros_f]),
            "fishing_selectivity_l50": np.concatenate([focal["focal_fishing_selectivity_l50"], bkg_zeros_f]),
            "fishing_selectivity_a50": np.concatenate([focal["focal_fishing_a50"], np.full(n_bkg, np.nan, dtype=np.float64)]),
            "fishing_selectivity_type": np.concatenate([focal["focal_fishing_sel_type"], np.full(n_bkg, -1, dtype=np.int32)]),
            "fishing_selectivity_slope": np.concatenate([focal["focal_fishing_slope"], bkg_zeros_f]),
            "movement_method": focal_movement_method + ["none"] * n_bkg,
            "random_walk_range": np.concatenate([focal["focal_random_walk_range"], bkg_zeros_i]),
            "out_mortality_rate": np.concatenate([focal["focal_out_mortality_rate"], bkg_zeros_f]),
            "n_schools": np.concatenate([focal["focal_n_schools"], bkg_zeros_i]),
            "fishing_spatial_maps": focal_fishing_spatial_maps + [None] * n_bkg,
        }
    else:
        return {
            "all_species_names": focal_species_names[:],
            "linf": focal["focal_linf"],
            "k": focal["focal_k"],
            "t0": focal["focal_t0"],
            "egg_size": focal["focal_egg_size"],
            "condition_factor": focal["focal_condition_factor"],
            "allometric_power": focal["focal_allometric_power"],
            "vb_threshold_age": focal["focal_vb_threshold_age"],
            "lifespan_dt": focal["focal_lifespan_dt"],
            "ingestion_rate": focal["focal_ingestion_rate"],
            "critical_success_rate": focal["focal_critical_success_rate"],
            "delta_lmax_factor": focal["focal_delta_lmax_factor"],
            "additional_mortality_rate": focal["focal_additional_mortality_rate"],
            "sex_ratio": focal["focal_sex_ratio"],
            "relative_fecundity": focal["focal_relative_fecundity"],
            "maturity_size": focal["focal_maturity_size"],
            "seeding_biomass": focal["focal_seeding_biomass"],
            "seeding_max_step": focal["focal_seeding_max_step"],
            "larva_mortality_rate": focal["focal_larva_mortality_rate"],
            "maturity_age_dt": focal["focal_maturity_age_dt"],
            "lmax": focal["focal_lmax"],
            "starvation_rate_max": focal["focal_starvation_rate_max"],
            "fishing_rate": focal["fishing"],
            "fishing_selectivity_l50": focal["focal_fishing_selectivity_l50"],
            "fishing_selectivity_a50": focal["focal_fishing_a50"],
            "fishing_selectivity_type": focal["focal_fishing_sel_type"],
            "fishing_selectivity_slope": focal["focal_fishing_slope"],
            "movement_method": focal_movement_method,
            "random_walk_range": focal["focal_random_walk_range"],
            "out_mortality_rate": focal["focal_out_mortality_rate"],
            "n_schools": focal["focal_n_schools"],
            "fishing_spatial_maps": focal_fishing_spatial_maps,
        }
```

**IMPORTANT:** The exact variable names in `focal` dict must match what Tasks 1-3 produced. Before writing this helper, grep the current `from_dict` to confirm which variables are still assigned inline vs. returned from helpers. Build the `focal` dict from all the `focal_*` variables that exist at the merge point.

- [ ] **Step 3: Replace inline block in `from_dict`**

Build a `focal` dict from all `focal_*` locals, call the helper, and unpack:

```python
        _focal = {
            "focal_linf": focal_linf, "focal_k": focal_k, "focal_t0": focal_t0,
            "focal_egg_size": focal_egg_size, "focal_condition_factor": focal_condition_factor,
            # ... all focal_* variables ...
        }
        _merged = _merge_focal_background(
            _focal, background_list, focal_species_names, focal_fishing_spatial_maps,
            focal_movement_method,
        )
        all_species_names = _merged["all_species_names"]
        linf = _merged["linf"]
        # ... unpack all merged variables ...
```

- [ ] **Step 4: Parity + full suite + ruff + commit**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q` → 12 passed
Run: `.venv/bin/python -m pytest tests/ -q` → 2148 passed
Run: `.venv/bin/ruff check osmose/engine/config.py`

```bash
git add osmose/engine/config.py
git commit -m "refactor(engine): extract _merge_focal_background from from_dict (I-3 step 4/5)"
```

---

## Task 5: Extract `_parse_output_flags` (return statement block)

**Files:**
- Modify: `osmose/engine/config.py`

**Context:** The output flags are currently passed directly in the `return cls(...)` call at lines 1321-1434. Extract the output-related flags into a helper that returns a dict, then spread that dict into the constructor.

- [ ] **Step 1: Identify output-related fields in the return statement**

Grep for `output_` in the return statement. The target fields are:
- `output_biomass_byage`, `output_biomass_bysize`, `output_abundance_byage`, `output_abundance_bysize`
- `output_size_min`, `output_size_max`, `output_size_incr`
- `output_bioen_ingest`, `output_bioen_maint`, `output_bioen_rho`, `output_bioen_sizeinf`
- `output_record_frequency`, `diet_output_enabled`, `output_step0_include`, `output_cutoff_age`

- [ ] **Step 2: Add the helper function**

```python
def _parse_output_flags(cfg: dict[str, str], n_sp: int, n_bkg: int) -> dict[str, Any]:
    """Parse output recording flags and distribution settings."""
    output_record_freq = int(cfg.get("output.recordfrequency.ndt", "1"))
    diet_output = cfg.get("output.diet.composition.enabled", "false").lower() == "true"
    step0 = cfg.get("output.step0.include", "false").lower() == "true"

    # Output cutoff age
    n_total = n_sp + n_bkg
    cutoff_vals = []
    found_any = False
    for i in range(n_sp):
        val = cfg.get(f"output.cutoff.age.sp{i}", "")
        if val and val.lower() not in ("null", "none", ""):
            cutoff_vals.append(float(val))
            found_any = True
        else:
            cutoff_vals.append(0.0)
    cutoff_vals.extend([0.0] * n_bkg)
    cutoff_age = np.array(cutoff_vals, dtype=np.float64) if found_any else None

    return {
        "output_record_frequency": output_record_freq,
        "diet_output_enabled": diet_output,
        "output_step0_include": step0,
        "output_cutoff_age": cutoff_age,
        "output_biomass_byage": _enabled(cfg, "output.biomass.byage.enabled"),
        "output_biomass_bysize": _enabled(cfg, "output.biomass.bysize.enabled"),
        "output_abundance_byage": _enabled(cfg, "output.abundance.byage.enabled"),
        "output_abundance_bysize": _enabled(cfg, "output.abundance.bysize.enabled"),
        "output_size_min": float(cfg.get("output.distrib.bysize.min", "0")),
        "output_size_max": float(cfg.get("output.distrib.bysize.max", "205")),
        "output_size_incr": float(cfg.get("output.distrib.bysize.incr", "10")),
        "output_bioen_ingest": cfg.get("output.bioen.ingest.enabled", "false").lower() == "true",
        "output_bioen_maint": cfg.get("output.bioen.maint.enabled", "false").lower() == "true",
        "output_bioen_rho": cfg.get("output.bioen.rho.enabled", "false").lower() == "true",
        "output_bioen_sizeinf": cfg.get("output.bioen.sizeinf.enabled", "false").lower() == "true",
    }
```

- [ ] **Step 3: Replace inline assignments and update return statement**

Move the inline output parsing (output_record_freq, diet_output_enabled, output_step0, output_cutoff_age) to the helper call:

```python
        _output = _parse_output_flags(cfg, n_sp, n_bkg)
```

Then in the `return cls(...)` call, replace each output field with `_output["field_name"]`:

```python
            output_record_frequency=_output["output_record_frequency"],
            diet_output_enabled=_output["diet_output_enabled"],
            output_step0_include=_output["output_step0_include"],
            output_cutoff_age=_output["output_cutoff_age"],
            output_biomass_byage=_output["output_biomass_byage"],
            # ... etc for all output fields ...
```

- [ ] **Step 4: Parity + full suite + ruff + commit**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q` → 12 passed
Run: `.venv/bin/python -m pytest tests/ -q` → 2148 passed
Run: `.venv/bin/ruff check osmose/engine/config.py`

```bash
git add osmose/engine/config.py
git commit -m "refactor(engine): extract _parse_output_flags from from_dict (I-3 step 5/5)"
```

---

## Final gate

- [ ] **Parity**: `.venv/bin/python -m pytest tests/test_engine_parity.py -q` → 12 passed, bit-exact
- [ ] **Full suite**: `.venv/bin/python -m pytest tests/ -q` → 2148 passed, 15 skipped, 0 failed
- [ ] **Ruff**: `.venv/bin/ruff check osmose/ ui/ tests/` → clean
- [ ] **Commit count**: `git log --oneline master..HEAD` → 5 commits
- [ ] **Line count check**: `grep -c "def from_dict" osmose/engine/config.py` → still 1; the method should now be ~150-200 lines
- [ ] Invoke `superpowers:finishing-a-development-branch`
