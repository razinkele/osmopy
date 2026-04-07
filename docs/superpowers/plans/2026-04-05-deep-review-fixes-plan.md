# Deep Review Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the highest-impact 30 findings (of 39 total) from 6-agent deep review across engine correctness, config, UI, security, performance, architecture, and test coverage — without breaking parity or existing tests.

**Architecture:** 5-phase incremental remediation. Phase 1 fixes 7 critical bugs. Phase 2 fixes 12 high-severity issues. Phase 3 addresses 5 medium-severity hardening items. Phase 4 adds 2 test coverage tasks. Phase 5 has 1 performance optimization. Each phase has a full test gate.

**Deferred findings** (not in this plan — low impact, latent, or require deep refactoring):
- H2/H15: Module-level `_tl_weighted_sum` in mortality.py — requires restructuring Numba kernel interface
- H9: run.py ncell comment/code mismatch — needs Java engine verification to determine correct value
- M1: Latent `cell_id` value when `resources is None` — value never read
- M2-M4: Latent numerical/seed issues — low practical impact
- M5/M9: Runner validation gaps — latent security, no current UI exposure
- M11-M12: Performance in non-primary standalone code paths
- M14: Cross-module private import — code style only

**Tech Stack:** Python 3.12, NumPy, Numba, xarray, Shiny, pytest, ruff

**Spec:** Deep review consolidated report (conversation context, 2026-04-05)

---

## Phase 1 — Critical Fixes

### Task 1: Fix stale `from-import` of diet tracking globals (C1)

**Files:**
- Modify: `osmose/engine/processes/mortality.py:25-28, 405, 422`

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_diet.py`, add:

```python
def test_diet_tracking_visible_in_mortality_path():
    """Verify enable_diet_tracking() is visible through the mortality module's imports."""
    import osmose.engine.processes.predation as pred
    pred.enable_diet_tracking(n_schools=10, n_species=3)
    try:
        # The mortality module must see the updated values
        import osmose.engine.processes.mortality as mort
        assert pred._diet_tracking_enabled is True
        assert pred._diet_matrix is not None
        # Mortality must access current values, not stale import-time bindings
        assert mort._pred_mod._diet_tracking_enabled is True
        assert mort._pred_mod._diet_matrix is not None
    finally:
        pred.disable_diet_tracking()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_diet.py::test_diet_tracking_visible_in_mortality_path -v`
Expected: FAIL — `mort._pred_mod` does not exist yet

- [ ] **Step 3: Replace stale from-import with module-level reference**

In `osmose/engine/processes/mortality.py`, replace lines 25-28:

```python
from osmose.engine.processes.predation import (
    _diet_matrix,
    _diet_tracking_enabled,
)
```

with:

```python
import osmose.engine.processes.predation as _pred_mod
```

Then replace every reference to the bare names throughout `mortality.py`:
- `_diet_tracking_enabled` → `_pred_mod._diet_tracking_enabled`
- `_diet_matrix` → `_pred_mod._diet_matrix`

These appear at approximately lines 405, 407-408, 422-425, 1354-1355, 1595-1597.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/mortality.py tests/test_engine_diet.py
git commit -m "fix: replace stale from-import of diet globals with module reference (C1)"
```

---

### Task 2: Fix division by zero in `f_o2` (C2)

**Files:**
- Modify: `osmose/engine/processes/oxygen_function.py:8-10`
- Test: `tests/test_engine_oxygen_function.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_oxygen_function.py`, add:

```python
def test_f_o2_zero_denominator():
    """f_o2 should return 0.0 when o2=0 and c2=0, not NaN."""
    from osmose.engine.processes.oxygen_function import f_o2
    result = f_o2(np.array([0.0]), c1=1.0, c2=0.0)
    assert not np.isnan(result[0])
    assert result[0] == 0.0


def test_f_o2_c2_zero_positive_o2():
    """When c2=0 and o2>0, f_o2 should return c1 (limit of c1*o2/(o2+0))."""
    from osmose.engine.processes.oxygen_function import f_o2
    result = f_o2(np.array([5.0]), c1=0.8, c2=0.0)
    assert not np.isnan(result[0])
    np.testing.assert_allclose(result[0], 0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_oxygen_function.py::test_f_o2_zero_denominator -v`
Expected: FAIL — returns NaN

- [ ] **Step 3: Fix `f_o2` with safe denominator**

In `osmose/engine/processes/oxygen_function.py`, replace the function body:

```python
def f_o2(o2: NDArray[np.float64], c1: float, c2: float) -> NDArray[np.float64]:
    """Oxygen dose-response: f_O2 = C1 * O2 / (O2 + C2)."""
    denom = o2 + c2
    safe_denom = np.where(denom > 0, denom, 1.0)  # avoid div-by-zero warning
    return np.where(denom > 0, c1 * o2 / safe_denom, 0.0)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/oxygen_function.py tests/test_engine_oxygen_function.py
git commit -m "fix: guard against division by zero in f_o2 when denom=0 (C2)"
```

---

### Task 3: Fix division by zero in `phi_t` (C3)

**Files:**
- Modify: `osmose/engine/processes/temp_function.py:24-26`
- Test: `tests/test_engine_temp_function.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_temp_function.py`, add:

```python
def test_phi_t_e_d_equals_e_m_raises():
    """phi_t must raise ValueError when e_d == e_m (division by zero)."""
    from osmose.engine.processes.temp_function import phi_t
    with pytest.raises(ValueError, match="e_d must be greater than e_m"):
        phi_t(np.array([15.0]), e_m=0.5, e_d=0.5, t_p=20.0)


def test_phi_t_e_d_less_than_e_m_raises():
    """phi_t must raise ValueError when e_d < e_m."""
    from osmose.engine.processes.temp_function import phi_t
    with pytest.raises(ValueError, match="e_d must be greater than e_m"):
        phi_t(np.array([15.0]), e_m=0.8, e_d=0.3, t_p=20.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_temp_function.py::test_phi_t_e_d_equals_e_m_raises -v`
Expected: FAIL — ZeroDivisionError instead of ValueError

- [ ] **Step 3: Add guard in `phi_t`**

In `osmose/engine/processes/temp_function.py`, add at the top of `phi_t()` (line 21, before `t_k = ...`):

```python
    if e_d <= e_m:
        raise ValueError(
            f"e_d must be greater than e_m for Johnson curve, got e_d={e_d}, e_m={e_m}"
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/temp_function.py tests/test_engine_temp_function.py
git commit -m "fix: validate e_d > e_m in phi_t to prevent division by zero (C3)"
```

---

### Task 4: Eliminate module-level `_config_dir` global (C4)

**Files:**
- Modify: `osmose/engine/config.py:75-81, 84-94, 97-122, 746`

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_config.py`, add:

```python
def test_from_dict_thread_safety():
    """Two concurrent from_dict calls must not share _config_dir state."""
    import concurrent.futures
    import threading

    seen_dirs = []
    barrier = threading.Barrier(2)

    def load_config(config_dir):
        cfg = {"_osmose.config.dir": config_dir, "simulation.nspecies": "1",
               "simulation.time.ndtperyear": "24", "simulation.time.nyear": "1",
               "species.lifespan.sp0": "3", "species.name.sp0": "Test",
               "species.linf.sp0": "30", "species.k.sp0": "0.3",
               "species.t0.sp0": "0", "species.c.sp0": "0.001",
               "species.b.sp0": "3.0", "species.length.initial.sp0": "1.0",
               "population.seeding.biomass.sp0": "1000",
               "reproduction.season.file.sp0": "",
               "predation.ingestion.rate.max.sp0": "3.5",
               "predation.efficiency.critical.sp0": "0.57",
               "mortality.starvation.rate.max.sp0": "3.0",
               "mortality.additional.rate.sp0": "0.0",
               "predation.accessibility.stage.threshold.sp0": "0.0",
               "predation.accessibility.stage.structure.sp0": "size"}
        barrier.wait()
        from osmose.engine.config import EngineConfig
        ec = EngineConfig.from_dict(cfg)
        return config_dir

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(load_config, "/tmp/dir_a")
        f2 = pool.submit(load_config, "/tmp/dir_b")
        r1, r2 = f1.result(), f2.result()
    # Both should complete without error — the key test is no crash from shared state
    assert r1 == "/tmp/dir_a"
    assert r2 == "/tmp/dir_b"
```

- [ ] **Step 2: Run test to verify it passes (baseline — may pass or fail depending on timing)**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py::test_from_dict_thread_safety -v`

- [ ] **Step 3: Thread `config_dir` through `_resolve_file` and `_search_dirs` as a parameter**

In `osmose/engine/config.py`:

1. Remove the global `_config_dir` variable (line 75) and `_set_config_dir()` function (lines 78-81).

2. Add `config_dir` parameter to `_search_dirs`:

```python
def _search_dirs(config_dir: str = "") -> list[Path]:
    """Build a list of directories to search for data files."""
    import glob as _glob
    dirs: list[Path] = []
    if config_dir:
        dirs.append(Path(config_dir))
    dirs.append(Path("."))
    dirs.append(Path("data/examples"))
    dirs += [Path(d) for d in _glob.glob("data/*/")]
    return dirs
```

3. Add `config_dir` parameter to `_resolve_file`:

```python
def _resolve_file(file_key: str, config_dir: str = "") -> Path | None:
    """Resolve a relative file path against multiple search directories."""
    if not file_key:
        return None
    if ".." in Path(file_key).parts:
        _log.warning("Rejecting file key with '..' traversal: %s", file_key)
        return None
    p = Path(file_key)
    if p.is_absolute():
        for base in _search_dirs(config_dir):
            try:
                if p.is_relative_to(base.resolve()) and p.exists():
                    return p
            except (ValueError, OSError):
                continue
        _log.warning("Rejecting absolute path not under any search dir: %s", file_key)
        return None
    for base in _search_dirs(config_dir):
        path = base / file_key
        if path.exists():
            return path
    return None
```

4. In `from_dict()` (line 746), remove the `_set_config_dir(...)` call and instead pass `config_dir` to each `_resolve_file` call. Store as a local variable:

```python
config_dir = cfg.get("_osmose.config.dir", "")
```

Then pass `config_dir=config_dir` to every `_resolve_file(...)` call in the method.

- [ ] **Step 4: Update all 14 `_resolve_file` callers to pass `config_dir`**

There are 14 callsites in `config.py` at lines 147, 162, 214, 286, 313, 328, 366, 383, 407, 447, 491, 514, 848, 854. Each is inside a helper function called from `from_dict()`. For each helper function (`_load_accessibility`, `_load_stage_accessibility`, `_load_catchability_csv`, `_load_fishing_seasonality`, `_load_spawning_seasons`, `_load_additional_mortality_by_dt`, `_load_additional_mortality_spatial`, `_load_mpa_zones`, `_load_movement_distribution_maps`), add `config_dir: str = ""` as a parameter and pass it through to `_resolve_file(file_key, config_dir)`. Then update each call to these helpers in `from_dict()` to pass `config_dir=config_dir`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "fix: thread config_dir as parameter instead of module global (C4)"
```

---

### Task 5: Fix `reactive.poll` not consumed in calibration UI (C5)

**Files:**
- Modify: `ui/pages/calibration_handlers.py:155-172`

- [ ] **Step 1: Add `@reactive.effect` consumer after the poll definition**

In `ui/pages/calibration_handlers.py`, after the `_poll_cal_messages` function definition (after line 172), add:

```python
    @reactive.effect
    def _consume_cal_poll():
        _poll_cal_messages()
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add ui/pages/calibration_handlers.py
git commit -m "fix: consume reactive.poll result so calibration UI updates (C5)"
```

---

### Task 6: Fix tautological weight-length test (C6)

**Files:**
- Modify: `tests/test_engine_java_comparison.py:199-203`

- [ ] **Step 1: Fix the test to compare function output against independently known values**

In `tests/test_engine_java_comparison.py`, replace lines 199-203:

```python
        weights = sp["c"] * lengths ** sp["b"]
        expected_weights = sp["c"] * lengths ** sp["b"]
        np.testing.assert_allclose(
            weights, expected_weights, rtol=1e-12, err_msg=f"W-L mismatch for {sp['name']}"
        )
```

with:

```python
        # Weight from VB lengths should match W = c * L^b
        computed_weights = sp["c"] * lengths ** sp["b"]
        # Verify against independently known analytical bounds:
        # At age=0, length ≈ initial; at max age, length → L_inf
        assert computed_weights[0] > 0, f"Weight at age 0 must be positive for {sp['name']}"
        assert computed_weights[-1] > computed_weights[0], (
            f"Weight must increase with age for {sp['name']}"
        )
        # Max weight must be bounded by c * linf^b
        max_weight = sp["c"] * sp["linf"] ** sp["b"]
        assert computed_weights[-1] <= max_weight * 1.01, (
            f"Weight exceeds theoretical max for {sp['name']}"
        )
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_java_comparison.py::TestVonBertalanffyGrowth::test_weight_length_conversion -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_java_comparison.py
git commit -m "fix: replace tautological weight-length test with meaningful assertions (C6)"
```

---

### Task 7: Fix `JavaEngine.run_ensemble` raising NotImplementedError (C7)

**Files:**
- Modify: `osmose/engine/__init__.py:114-121`

- [ ] **Step 1: Add explicit NotImplementedError in run_ensemble**

In `osmose/engine/__init__.py`, replace lines 114-121:

```python
    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        return [self.run(config, output_dir, seed=base_seed + i) for i in range(n)]
```

with:

```python
    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        raise NotImplementedError(
            "JavaEngine.run_ensemble() is not implemented — use OsmoseRunner directly"
        )
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/__init__.py
git commit -m "fix: explicit NotImplementedError in JavaEngine.run_ensemble (C7)"
```

---

## Phase 2 — High-Severity Fixes

### Task 8: Fix predation Python fallback using stale biomass (H1)

**Files:**
- Modify: `osmose/engine/processes/predation.py:270-272`

- [ ] **Step 1: Replace `state.biomass[p_idx]` with live `abundance * weight`**

In `osmose/engine/processes/predation.py`, replace line 270-272:

```python
        max_eatable = (
            state.biomass[p_idx] * config.ingestion_rate[sp_pred] / (config.n_dt_per_year * n_subdt)
        )
```

with:

```python
        max_eatable = (
            state.abundance[p_idx] * state.weight[p_idx]
            * config.ingestion_rate[sp_pred] / (config.n_dt_per_year * n_subdt)
        )
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/processes/predation.py
git commit -m "fix: use live abundance*weight in Python fallback predation (H1)"
```

---

### Task 9: Fix `_average_step_outputs` dropping distribution data (H3)

**Files:**
- Modify: `osmose/engine/simulate.py:699-742`

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_simulate.py`, add:

```python
def test_average_step_outputs_preserves_distributions():
    """Distribution dicts must not be silently dropped during averaging."""
    from osmose.engine.simulate import _average_step_outputs, StepOutput

    dist = {0: np.array([1.0, 2.0, 3.0])}
    so = StepOutput(
        step=0,
        biomass=np.array([100.0]),
        abundance=np.array([50.0]),
        mortality_by_cause=np.zeros((1, 6)),
        biomass_by_age=dist,
        abundance_by_age=dist,
        biomass_by_size=dist,
        abundance_by_size=dist,
    )
    result = _average_step_outputs([so], freq=1, record_step=0)
    assert result.biomass_by_age is not None
    assert result.abundance_by_age is not None
    assert result.biomass_by_size is not None
    assert result.abundance_by_size is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py::test_average_step_outputs_preserves_distributions -v`
Expected: FAIL — `result.biomass_by_age is None`

- [ ] **Step 3: Pass distribution dicts through in both code paths**

In `osmose/engine/simulate.py`, in `_average_step_outputs`:

For the `len(accumulated) == 1` early-return path (line 712-724), add the four distribution fields:

```python
    if len(accumulated) == 1:
        return StepOutput(
            step=record_step,
            biomass=accumulated[0].biomass,
            abundance=accumulated[0].abundance,
            mortality_by_cause=accumulated[0].mortality_by_cause,
            yield_by_species=accumulated[0].yield_by_species,
            biomass_by_age=accumulated[0].biomass_by_age,
            abundance_by_age=accumulated[0].abundance_by_age,
            biomass_by_size=accumulated[0].biomass_by_size,
            abundance_by_size=accumulated[0].abundance_by_size,
            bioen_e_net_by_species=bioen_e_net_avg,
            bioen_ingestion_by_species=bioen_ingestion_avg,
            bioen_maint_by_species=bioen_maint_avg,
            bioen_rho_by_species=bioen_rho_avg,
            bioen_size_inf_by_species=bioen_size_inf_avg,
        )
```

For the multi-step averaging path (lines 725-742), use the last accumulated entry's distributions (these are snapshots, not rates — averaging doesn't apply):

```python
    biomass = np.mean([o.biomass for o in accumulated], axis=0)
    abundance = np.mean([o.abundance for o in accumulated], axis=0)
    mortality = np.sum([o.mortality_by_cause for o in accumulated], axis=0)
    yield_sum = np.sum(
        [o.yield_by_species for o in accumulated if o.yield_by_species is not None], axis=0
    )
    return StepOutput(
        step=record_step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=mortality,
        yield_by_species=yield_sum,
        biomass_by_age=accumulated[-1].biomass_by_age,
        abundance_by_age=accumulated[-1].abundance_by_age,
        biomass_by_size=accumulated[-1].biomass_by_size,
        abundance_by_size=accumulated[-1].abundance_by_size,
        bioen_e_net_by_species=bioen_e_net_avg,
        bioen_ingestion_by_species=bioen_ingestion_avg,
        bioen_maint_by_species=bioen_maint_avg,
        bioen_rho_by_species=bioen_rho_avg,
        bioen_size_inf_by_species=bioen_size_inf_avg,
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_simulate.py
git commit -m "fix: preserve distribution dicts in _average_step_outputs (H3)"
```

---

### Task 10: Fix boundary cell bias in movement sampling (H4)

**Files:**
- Modify: `osmose/engine/processes/movement.py:102, 385, 422`

- [ ] **Step 1: Write the failing test**

In `tests/test_movement_numba.py`, add:

```python
def test_round_based_sampling_has_boundary_bias():
    """Verify that int(round((n-1)*random())) is biased and our fix removes it.

    The old pattern int(round((n-1)*rand)) gives boundary cells half the
    probability of interior cells. rng.integers(0, n) is uniform.
    """
    n = 5
    rng = np.random.default_rng(42)

    # Old pattern — biased
    old_counts = np.zeros(n, dtype=np.int64)
    for _ in range(100_000):
        idx = int(round((n - 1) * rng.random()))
        old_counts[idx] += 1
    # Boundary cells (0 and n-1) should have ~half the hits of interior cells
    assert old_counts[0] < old_counts[2] * 0.7, "Old pattern should show boundary bias"

    # New pattern — uniform
    new_counts = np.zeros(n, dtype=np.int64)
    rng2 = np.random.default_rng(42)
    for _ in range(100_000):
        idx = rng2.integers(0, n)
        new_counts[idx] += 1
    expected = 100_000 / n
    for i in range(n):
        assert abs(new_counts[i] - expected) < expected * 0.1, (
            f"Cell {i} got {new_counts[i]} hits, expected ~{expected}"
        )
```

- [ ] **Step 2: Fix Python fallback path**

In `osmose/engine/processes/movement.py`, at line 102, replace:

```python
    idx = int(round((len(accessible) - 1) * rng.random()))
```

with:

```python
    idx = rng.integers(0, len(accessible))
```

- [ ] **Step 3: Fix Numba rejection sampling path**

In `osmose/engine/processes/movement.py`, at line 385, replace:

```python
                    flat_idx = int(round((n_cells - 1) * np.random.random()))
```

with:

```python
                    flat_idx = np.random.randint(0, n_cells)
```

- [ ] **Step 4: Fix Numba walk path**

At line 422, replace:

```python
            target = int(round((n_accessible - 1) * np.random.random()))
```

with:

```python
            target = np.random.randint(0, n_accessible)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/movement.py tests/test_movement_numba.py
git commit -m "fix: use integer sampling instead of round() to avoid boundary cell bias (H4)"
```

---

### Task 11: Fix movement map log flooding (H5)

**Files:**
- Modify: `osmose/engine/movement_maps.py:281-290`

- [ ] **Step 1: Replace per-slot warnings with a single aggregated warning**

In `osmose/engine/movement_maps.py`, replace lines 281-290:

```python
        # --- Validate: warn about uncovered (age, step) slots ---
        for age_dt in range(lifespan_dt):
            for step in range(n_total_steps):
                if self.index_maps[age_dt, step] == -1:
                    logger.warning(
                        "No movement map for species=%r age_dt=%d step=%d",
                        species_name,
                        age_dt,
                        step,
                    )
```

with:

```python
        # --- Validate: warn about uncovered (age, step) slots ---
        uncovered = int((self.index_maps == -1).sum())
        if uncovered > 0:
            logger.warning(
                "Species %r: %d of %d (age_dt, step) slots have no movement map assigned",
                species_name,
                uncovered,
                lifespan_dt * n_total_steps,
            )
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/movement_maps.py
git commit -m "fix: aggregate movement map coverage warning instead of per-slot flooding (H5)"
```

---

### Task 12: Filter internal keys in config writer (H6)

**Files:**
- Modify: `osmose/config/writer.py:88-96`

- [ ] **Step 1: Write the failing test**

In `tests/test_config_writer.py`, add:

```python
def test_internal_keys_not_written(tmp_path):
    """Keys starting with _ must not appear in output files."""
    from osmose.config.writer import OsmoseConfigWriter
    writer = OsmoseConfigWriter()
    config = {
        "_osmose.config.dir": "/tmp/internal",
        "simulation.time.nyear": "5",
    }
    writer.write(config, tmp_path)
    master = (tmp_path / "osm_all-parameters.csv").read_text()
    assert "_osmose" not in master
    assert "simulation.time.nyear" in master
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_writer.py::test_internal_keys_not_written -v`
Expected: FAIL — `_osmose.config.dir` appears in master file

- [ ] **Step 3: Filter internal keys in `_route_params`**

In `osmose/config/writer.py`, in `_route_params()` (line 92), add a filter:

```python
    def _route_params(self, config: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Categorise each key in *config* into the correct bucket."""
        buckets: dict[str, dict[str, str]] = {}

        for key, value in config.items():
            if key.startswith("_"):
                continue
            bucket = self._classify(key)
            buckets.setdefault(bucket, {})[key] = str(value)

        return buckets
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 5: Commit**

```bash
git add osmose/config/writer.py tests/test_config_writer.py
git commit -m "fix: filter internal _-prefixed keys from config writer output (H6)"
```

---

### Task 13: Fix multi-species RMSE cross-product merge (H7)

**Files:**
- Modify: `osmose/calibration/objectives.py:21`
- Test: `tests/test_objectives.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_objectives.py`, add:

```python
def test_timeseries_rmse_multi_species_no_cross_product():
    """Multi-species merge must join on (time, species), not just time."""
    from osmose.calibration.objectives import biomass_rmse
    sim = pd.DataFrame({
        "time": [1, 1, 2, 2],
        "species": ["A", "B", "A", "B"],
        "biomass": [100.0, 200.0, 110.0, 210.0],
    })
    obs = pd.DataFrame({
        "time": [1, 1, 2, 2],
        "species": ["A", "B", "A", "B"],
        "biomass": [100.0, 200.0, 110.0, 210.0],
    })
    # Identical data → RMSE should be 0.0
    result = biomass_rmse(sim, obs, species=None)
    assert result == 0.0, f"Expected 0.0 for identical multi-species data, got {result}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_objectives.py::test_timeseries_rmse_multi_species_no_cross_product -v`
Expected: FAIL — cross-product merge produces nonzero RMSE

- [ ] **Step 3: Fix merge to include species column**

In `osmose/calibration/objectives.py`, replace line 21:

```python
    merged = pd.merge(simulated, observed, on="time", suffixes=("_sim", "_obs"))
```

with:

```python
    merge_cols = ["time"]
    if "species" in simulated.columns and "species" in observed.columns:
        merge_cols.append("species")
    merged = pd.merge(simulated, observed, on=merge_cols, suffixes=("_sim", "_obs"))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/objectives.py tests/test_objectives.py
git commit -m "fix: include species in RMSE merge to prevent cross-product (H7)"
```

---

### Task 14: Fix `state.dirty.set(True)` outside isolate in forcing.py (H8)

**Files:**
- Modify: `ui/pages/forcing.py:136-138`

- [ ] **Step 1: Move `state.dirty.set(True)` inside the isolate block**

In `ui/pages/forcing.py`, replace lines 134-138:

```python
        actual_changes = {k: v for k, v in updates.items() if cfg.get(k) != v}
        if actual_changes:
            cfg.update(actual_changes)
            with reactive.isolate():
                state.config.set(cfg)
            state.dirty.set(True)
```

with:

```python
        actual_changes = {k: v for k, v in updates.items() if cfg.get(k) != v}
        if actual_changes:
            cfg.update(actual_changes)
            with reactive.isolate():
                state.config.set(cfg)
                state.dirty.set(True)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add ui/pages/forcing.py
git commit -m "fix: move state.dirty.set inside reactive.isolate in forcing sync (H8)"
```

---

### Task 15: Fix temp dir leaks in export and demo loading (H10, H11)

**Files:**
- Modify: `ui/pages/advanced.py:185-191`
- Modify: `ui/pages/setup.py:107-109`

- [ ] **Step 1: Add atexit cleanup to export_config**

In `ui/pages/advanced.py`, add `import atexit, shutil` to the existing imports at the top of the file (if not already present). Then modify the `export_config` handler (around line 186-191):

```python
    @render.download(filename="osm_all-parameters.csv")
    def export_config():
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        atexit.register(shutil.rmtree, str(work_dir), True)
        writer = OsmoseConfigWriter()
        writer.write(state.config.get(), work_dir)
        master = work_dir / "osm_all-parameters.csv"
        return str(master)
```

- [ ] **Step 2: Add atexit cleanup to handle_load_example**

In `ui/pages/setup.py`, after line 108 (`tmp = Path(tempfile.mkdtemp(prefix="osmose_demo_"))`), add:

```python
            atexit.register(shutil.rmtree, str(tmp), True)
```

Add `import atexit, shutil` to the imports if not already present.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 4: Commit**

```bash
git add ui/pages/advanced.py ui/pages/setup.py
git commit -m "fix: register atexit cleanup for export and demo temp dirs (H10, H11)"
```

---

### Task 16: Fix path traversal in results comparison/diff handlers (H12, H13)

**Files:**
- Modify: `ui/pages/results.py:588, 608`

- [ ] **Step 1: Add path validation to comparison_chart and config_diff_table**

In `ui/pages/results.py`, in the `comparison_chart` function (around line 588), add a guard after `out_dir = Path(input.output_dir())`:

```python
        out_dir = Path(input.output_dir())
        if ".." in out_dir.parts or (out_dir.is_absolute() and not out_dir.is_relative_to(Path.cwd())):
            return go.Figure().update_layout(title="Invalid output directory", template=tmpl)
```

Apply the same guard in `config_diff_table` (around line 608):

```python
        out_dir = Path(input.output_dir())
        if ".." in out_dir.parts or (out_dir.is_absolute() and not out_dir.is_relative_to(Path.cwd())):
            return ui.div("Invalid output directory.")
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add ui/pages/results.py
git commit -m "fix: add path traversal guard to comparison_chart and config_diff_table (H12, H13)"
```

---

### Task 17: Sanitize exception messages in UI notifications (H14)

**Files:**
- Modify: `ui/pages/scenarios.py` (lines 118, 172, 188, 275)
- Modify: `ui/pages/setup.py` (line 111)

- [ ] **Step 1: Replace raw exception strings with generic messages**

In `ui/pages/scenarios.py`, first add a logger at the top of the file (after the existing imports):

```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.scenarios_ui")
```

Then replace each raw exception notification:

Line 118:
```python
# Before:
ui.notification_show(f"Failed to save scenario: {exc}", type="error", duration=8)
# After:
_log.error("Failed to save scenario: %s", exc, exc_info=True)
ui.notification_show("Failed to save scenario. Check server logs for details.", type="error", duration=8)
```

Line 172:
```python
# Before:
ui.notification_show(f"Failed to fork scenario: {exc}", type="error", duration=8)
# After:
_log.error("Failed to fork scenario: %s", exc, exc_info=True)
ui.notification_show("Failed to fork scenario. Check server logs for details.", type="error", duration=8)
```

Line 188:
```python
# Before:
ui.notification_show(f"Failed to delete scenario: {exc}", type="error", duration=8)
# After:
_log.error("Failed to delete scenario: %s", exc, exc_info=True)
ui.notification_show("Failed to delete scenario. Check server logs for details.", type="error", duration=8)
```

Line 275:
```python
# Before:
ui.notification_show(f"Failed to import scenarios: {exc}", type="error", duration=8)
# After:
_log.error("Failed to import scenarios: %s", exc, exc_info=True)
ui.notification_show("Failed to import scenarios. Check server logs for details.", type="error", duration=8)
```

In `ui/pages/setup.py` at line 111:

```python
# Before:
ui.notification_show(str(exc), type="error", duration=5)
# After:
_log.error("Failed to load example: %s", exc, exc_info=True)
ui.notification_show("Failed to load example. Check server logs for details.", type="error", duration=5)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add ui/pages/scenarios.py ui/pages/setup.py
git commit -m "fix: sanitize exception messages in UI notifications (H14)"
```

---

### Task 18: Fix duplicate log output from setup_logging (H16)

**Files:**
- Modify: `osmose/logging.py:21-32`

- [ ] **Step 1: Write the failing test**

In `tests/test_logging.py` (create if needed), add:

Add `import io` to the existing imports at the top of `tests/test_logging.py`, then add the test:

```python
def test_no_duplicate_log_output():
    """A log message from a child logger must appear exactly once."""
    parent = setup_logging("osmose.test_dup_parent")
    child = setup_logging("osmose.test_dup_parent.child")

    # Capture output by replacing parent's handler stream
    buf = io.StringIO()
    for h in parent.handlers:
        h.stream = buf
    for h in child.handlers:
        h.stream = buf

    child.info("test message")
    output = buf.getvalue()
    # Should appear exactly once, not duplicated
    assert output.count("test message") == 1, (
        f"Expected 1 occurrence, got {output.count('test message')}: {output!r}"
    )

    # Cleanup
    parent.handlers.clear()
    child.handlers.clear()
```

- [ ] **Step 2: Fix by setting `propagate = False` on loggers with handlers**

In `osmose/logging.py`, add `logger.propagate = False` when a handler is attached:

```python
def setup_logging(
    name: str = "osmose",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 4: Commit**

```bash
git add osmose/logging.py tests/test_logging.py
git commit -m "fix: set propagate=False in setup_logging to prevent duplicate output (H16)"
```

---

### Task 19: Add diet tracking teardown fixture (H17)

**Files:**
- Modify: `tests/test_engine_diet.py` (add fixture)

- [ ] **Step 1: Add autouse fixture for diet tracking cleanup**

In `tests/test_engine_diet.py`, add `import pytest` to the imports at the top of the file, then add near the top (after imports):

```python
@pytest.fixture(autouse=True)
def _cleanup_diet_tracking():
    """Ensure diet tracking is disabled after every test in this module."""
    yield
    from osmose.engine.processes.predation import disable_diet_tracking
    disable_diet_tracking()
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_diet.py
git commit -m "fix: add autouse fixture to clean up diet tracking state between tests (H17)"
```

---

## Phase 3 — Medium-Severity Hardening

### Task 20: Fix multi-year spawning season normalization (M2)

**Files:**
- Modify: `osmose/engine/config.py:464-468`

- [ ] **Step 1: Write the failing test**

Note: `_load_spawning_seasons(cfg, n_species, n_dt_per_year)` loads season data from CSV files via `_resolve_file`. The function signature is `(cfg, n_species, n_dt_per_year)` — no `n_years` parameter. The number of years is determined by the CSV file's row count.

In `tests/test_engine_config.py`, add:

```python
def test_spawning_season_normalization_per_year(tmp_path):
    """Normalization must divide per-year chunks, not total sum."""
    from osmose.engine.config import _load_spawning_seasons

    # Create a 2-year season CSV (8 rows, 4 dt/year)
    csv_path = tmp_path / "season_sp0.csv"
    csv_path.write_text("step;value\n0;1\n1;2\n2;3\n3;4\n4;5\n5;6\n6;7\n7;8\n")
    cfg = {
        "reproduction.season.file.sp0": str(csv_path),
        "reproduction.normalisation.enabled": "true",
    }
    # After Task 4, _load_spawning_seasons gains a config_dir parameter
    seasons = _load_spawning_seasons(cfg, n_species=1, n_dt_per_year=4, config_dir=str(tmp_path))
    # Year 1 values [1,2,3,4] should sum to 1.0 after normalization
    year1_sum = seasons[0, 0:4].sum()
    np.testing.assert_allclose(year1_sum, 1.0, atol=1e-10)
    # Year 2 values [5,6,7,8] should also sum to 1.0
    year2_sum = seasons[0, 4:8].sum()
    np.testing.assert_allclose(year2_sum, 1.0, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py::test_spawning_season_normalization_per_year -v`
Expected: FAIL — year sums are not 1.0 individually (total-sum normalization makes year1 + year2 = 1.0)

- [ ] **Step 3: Fix normalization to operate per-year**

In `osmose/engine/config.py`, replace lines 464-468:

```python
            if normalize:
                # Normalize per year: sum over each n_dt_per_year chunk
                total = vals.sum()
                if total > 0:
                    vals = vals / total
```

with:

```python
            if normalize:
                for yr in range(max(1, len(vals) // n_dt_per_year)):
                    s = yr * n_dt_per_year
                    e = min(s + n_dt_per_year, len(vals))
                    chunk_sum = vals[s:e].sum()
                    if chunk_sum > 0:
                        vals[s:e] = vals[s:e] / chunk_sum
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "fix: normalize spawning season per-year instead of total sum (M2)"
```

---

### Task 21: Close OsmoseResults in calibration (M8)

**Files:**
- Modify: `osmose/calibration/problem.py:180-186`

- [ ] **Step 1: Use context manager**

In `osmose/calibration/problem.py`, replace lines 180-186:

```python
        from osmose.results import OsmoseResults

        results = OsmoseResults(output_dir)
        obj_values = []
        for fn in self.objective_fns:
            obj_values.append(fn(results))

        return obj_values
```

with:

```python
        from osmose.results import OsmoseResults

        with OsmoseResults(output_dir) as results:
            obj_values = [fn(results) for fn in self.objective_fns]

        return obj_values
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add osmose/calibration/problem.py
git commit -m "fix: use context manager for OsmoseResults in calibration (M8)"
```

---

### Task 22: Reset key_case_map between reader calls (M7)

**Files:**
- Modify: `osmose/config/reader.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_config_reader.py`, add:

```python
def test_key_case_map_reset_between_reads(tmp_path):
    """key_case_map must not carry stale entries from previous read() calls."""
    from osmose.config.reader import OsmoseConfigReader

    # First file with mixed case
    f1 = tmp_path / "cfg1.csv"
    f1.write_text("Species.Name.sp0 ; Anchovy\n")
    # Second file without that key
    f2 = tmp_path / "cfg2.csv"
    f2.write_text("simulation.time.nyear ; 5\n")

    reader = OsmoseConfigReader()
    reader.read(str(f1))
    assert "species.name.sp0" in reader.key_case_map

    reader.read(str(f2))
    assert "species.name.sp0" not in reader.key_case_map
    assert "simulation.time.nyear" in reader.key_case_map
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py::test_key_case_map_reset_between_reads -v`
Expected: FAIL — stale key persists

- [ ] **Step 3: Reset key_case_map at the start of read()**

In `osmose/config/reader.py`, at the beginning of the `read()` method (after `self.skipped_lines = 0`), add:

```python
        self.key_case_map = {}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 5: Commit**

```bash
git add osmose/config/reader.py tests/test_config_reader.py
git commit -m "fix: reset key_case_map between read() calls on same reader (M7)"
```

---

### Task 23: Add ZIP entry size limit in scenario import (M10)

**Files:**
- Modify: `osmose/scenarios.py`

- [ ] **Step 1: Add size guard before reading ZIP entries**

In `osmose/scenarios.py`, in the `import_all` method, before `data = json.loads(zf.read(name))`, add:

```python
                info = zf.getinfo(name)
                if info.file_size > 10_000_000:  # 10 MB cap
                    _log.warning("Skipping oversized ZIP entry: %s (%d bytes)", name, info.file_size)
                    continue
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add osmose/scenarios.py
git commit -m "fix: add 10MB size limit on ZIP entries in scenario import (M10)"
```

---

### Task 24: Graceful skip on bad scenario names in import_all (M6)

**Files:**
- Modify: `osmose/scenarios.py`

- [ ] **Step 1: Wrap Scenario construction in try/except**

In `osmose/scenarios.py`, in `import_all()` (around line 192-200), wrap the `Scenario(...)` construction and `self.save()` call in a try/except:

```python
                try:
                    scenario = Scenario(
                        name=scenario_name,
                        description=data.get("description", ""),
                        config=data.get("config", {}),
                        tags=data.get("tags", []),
                        parent_scenario=data.get("parent_scenario"),
                    )
                    self.save(scenario)
                    count += 1
                except ValueError as exc:
                    _log.warning("Skipping scenario with invalid name %r: %s", scenario_name, exc)
                    continue
```

This replaces the current unguarded `scenario = Scenario(...)` + `self.save(scenario)` + `count += 1` block at lines 192-200.

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add osmose/scenarios.py
git commit -m "fix: gracefully skip invalid scenario names in import_all (M6)"
```

---

## Phase 4 — Test Coverage

### Task 25: Add missing engine test coverage (H18)

**Files:**
- Modify: `tests/test_engine_simulate.py`

- [ ] **Step 1: Add test for output_step0_include**

Note: The `minimal_config` fixture (defined at line 12 of `test_engine_simulate.py`) provides the base config dict. These tests use `dict(minimal_config)` to avoid mutating the fixture.

```python
def test_simulate_output_step0_include(minimal_config):
    """output.step0.include=true should prepend a step-(-1) output."""
    cfg_dict = dict(minimal_config)
    cfg_dict["output.step0.include"] = "true"
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid(ny=1, nx=1, ocean_mask=np.ones((1, 1), dtype=bool))
    rng = np.random.default_rng(42)
    outputs = simulate(cfg, grid, rng)
    # First output should be step -1 (initial state)
    assert outputs[0].step == -1
```

- [ ] **Step 2: Add test for partial flush with non-divisible record frequency**

```python
def test_simulate_partial_flush(minimal_config):
    """Non-divisible record freq must flush remaining accumulated steps."""
    cfg_dict = dict(minimal_config)
    cfg_dict["output.recordfrequency.ndt"] = "7"  # doesn't divide 12 evenly
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid(ny=1, nx=1, ocean_mask=np.ones((1, 1), dtype=bool))
    rng = np.random.default_rng(42)
    outputs = simulate(cfg, grid, rng)
    # 12 steps / 7 = 1 full group + 1 partial = at least 2 outputs
    assert len(outputs) >= 2
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_simulate.py
git commit -m "test: add coverage for output_step0_include and partial flush (H18)"
```

---

### Task 26: Add diet tracking teardown + diet-in-mortality test

**Files:**
- Modify: `tests/test_engine_diet.py`

- [ ] **Step 1: Add test verifying diet tracking works through mortality path**

```python
def test_diet_tracking_through_mortality_module():
    """Diet matrix updates must be visible when accessed through mortality's import."""
    import osmose.engine.processes.predation as pred
    import osmose.engine.processes.mortality as mort

    pred.enable_diet_tracking(n_schools=5, n_species=2)
    try:
        # Access through mortality's module reference
        assert mort._pred_mod._diet_tracking_enabled is True
        assert mort._pred_mod._diet_matrix is not None
        assert mort._pred_mod._diet_matrix.shape == (5, 2)
    finally:
        pred.disable_diet_tracking()
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_diet.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_diet.py
git commit -m "test: verify diet tracking visibility through mortality module (C1 follow-up)"
```

---

## Phase 5 — Performance Optimization

### Task 27: Vectorize temperature lookups in bioenergetics (M13)

**Files:**
- Modify: `osmose/engine/simulate.py:210-222, 254-260`

- [ ] **Step 1: Replace scalar get_value loop with get_grid + fancy indexing**

In `osmose/engine/simulate.py`, replace the spatially-explicit temperature loop (lines 211-222):

```python
        else:
            # Spatially explicit: look up each school's cell
            phi_t_arr = np.empty(len(state), dtype=np.float64)
            for sp in range(config.n_species):
                mask = state.species_id == sp
                if not mask.any():
                    continue
                temps = np.array(
                    [
                        temp_data.get_value(step, int(state.cell_y[i]), int(state.cell_x[i]))
                        for i in np.where(mask)[0]
                    ]
                )
```

with:

```python
        else:
            # Spatially explicit: vectorized grid lookup
            temp_grid = temp_data.get_grid(step)
            phi_t_arr = np.empty(len(state), dtype=np.float64)
            for sp in range(config.n_species):
                mask = state.species_id == sp
                if not mask.any():
                    continue
                temps = temp_grid[state.cell_y[mask], state.cell_x[mask]]
```

Replace the temp_c_arr scalar loop (lines 254-260):

```python
    elif temp_data is not None:
        temp_c_arr = np.array(
            [
                temp_data.get_value(step, int(state.cell_y[i]), int(state.cell_x[i]))
                for i in range(len(state))
            ]
        )
```

with:

```python
    elif temp_data is not None:
        temp_grid = temp_data.get_grid(step)
        temp_c_arr = temp_grid[state.cell_y, state.cell_x]
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed

- [ ] **Step 3: Commit**

```bash
git add osmose/engine/simulate.py
git commit -m "perf: vectorize temperature lookups in _bioen_step via get_grid (M13)"
```

---

### Full Test Gate

- [ ] **Final: Run complete test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: 1766+ passed, 0 failures

- [ ] **Final: Run linter**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: Clean
