# Baltic Recruitment Ceiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cap each enabled focal species' per-step recruitment at its auto-derived unfished-equilibrium level (McGregor et al. 2019), removing the SR-curve's runaway upside that produces the Baltic cod boom/bust — config-gated and inert by default.

**Architecture:** Two parts. (A) A standalone derivation CLI runs the config with fishing zeroed, records per-step per-species recruitment through the engine's existing `step_observer` hook (fresh eggs are the `age_dt == 0` schools — no engine change needed), averages the late-window years by within-year season index, and writes a per-species × per-season sidecar CSV. (B) An engine clamp loads that sidecar (mirroring `_load_rv_gate`) and applies `min(n_eggs, ceiling)` in `reproduction()` right after `apply_stock_recruitment`, skipping seeded steps, bit-identical when the master switch is off.

**Tech Stack:** Python 3.12, NumPy, pandas (CSV I/O), pytest. OSMOSE schema-driven config (`osmose/schema/`), pure-Python engine (`osmose/engine/`).

## Global Constraints

- Branch: `baltic-recruitment-ceiling` (already created, off `master`).
- Run everything with `.venv/bin/python` (system `python` may not exist).
- Line length 100; lint with `.venv/bin/ruff check osmose/ ui/ tests/ scripts/` and `.venv/bin/ruff format --check` the same paths.
- Config keys are lowercase dot-separated; species-indexed keys use `sp{idx}`.
- **Inert by default:** with `reproduction.recruitment.ceiling.enabled=false` (the default) every existing config (Baltic, EEC, Bay of Biscay) must produce bit-identical output. The clamp code must not execute when the loaded ceiling is `None`.
- **Determinism keys** for parity tests: `movement.randomseed.fixed` + `stochastic.mortality.randomseed.fixed` (plus `simulation.rng.fixed=true`).
- `test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs[*]` must stay warning-free.
- New config keys are read as string literals inside `config.py`, so the AST walker in `config_validation.py` captures them automatically (the RV-gate keys are captured this way and are NOT in `_SUPPLEMENTARY_ALLOWLIST`). Verify, don't assume.

---

## File Structure

- `osmose/schema/species.py` — add 3 `OsmoseField` definitions for the ceiling keys (after the RV-gate block, line 511).
- `osmose/engine/config.py` — add `_load_recruitment_ceiling()` (mirror `_load_rv_gate`, line 1081), two `EngineConfig` fields (after line 1401), and `from_dict` wiring (line 2124 + line 2188).
- `osmose/engine/processes/reproduction.py` — add the clamp block after the RV-gate block (after line 166).
- `scripts/derive_recruitment_ceiling.py` — new derivation CLI + recording observer + late-window averaging + stationarity check + sidecar writer.
- `scripts/baltic_recruitment_ceiling_diagnostic.py` — new A/B boom/bust diagnostic.
- `tests/test_recruitment_ceiling.py` — unit + parity + derivation tests.

---

## Task 1: Schema fields for the ceiling config keys

**Files:**
- Modify: `osmose/schema/species.py:511` (insert after the RV-gate `OsmoseField` list entries, before the closing `]` at line 512)
- Test: `tests/test_recruitment_ceiling.py`

**Interfaces:**
- Produces: three registered schema keys — `reproduction.recruitment.ceiling.enabled` (bool), `reproduction.recruitment.ceiling.series.file` (file path), `reproduction.recruitment.ceiling.species.enabled.sp{idx}` (bool, indexed).

- [ ] **Step 1: Write the failing test**

Create `tests/test_recruitment_ceiling.py`:

```python
from osmose.schema import build_registry


def test_ceiling_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "reproduction.recruitment.ceiling.enabled" in keys
    assert "reproduction.recruitment.ceiling.series.file" in keys
    assert "reproduction.recruitment.ceiling.species.enabled.sp{idx}" in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py::test_ceiling_keys_registered -v`
Expected: FAIL — assertion error (keys not in registry).

- [ ] **Step 3: Add the schema fields**

In `osmose/schema/species.py`, insert immediately after the `reproduction.rv.gate.species.enabled.sp{idx}` field (line 511), before the closing `]`:

```python
    # ── Recruitment: unfished-level ceiling (McGregor et al. 2019) ───────────
    OsmoseField(
        key_pattern="reproduction.recruitment.ceiling.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description=(
            "Master switch for the unfished-level recruitment ceiling. When "
            "false the ceiling is inert and output is bit-identical."
        ),
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.recruitment.ceiling.series.file",
        param_type=ParamType.FILE_PATH,
        default="",
        description=(
            "CSV of the per-season unfished-equilibrium recruitment ceiling "
            "(season_idx,ceiling_sp0,...), produced by "
            "scripts/derive_recruitment_ceiling.py."
        ),
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.recruitment.ceiling.species.enabled.sp{idx}",
        param_type=ParamType.BOOL,
        default=False,
        description="Per-species enable for the recruitment ceiling (cod only for Baltic).",
        category="reproduction",
        indexed=True,
        required=False,
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py::test_ceiling_keys_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/schema/species.py tests/test_recruitment_ceiling.py
git commit -m "feat: schema fields for recruitment ceiling config keys"
```

---

## Task 2: `_load_recruitment_ceiling` loader + EngineConfig fields + from_dict wiring

**Files:**
- Modify: `osmose/engine/config.py` — add loader after `_load_rv_gate` (line 1144), two fields after line 1401, wiring at line 2124 + line 2188.
- Test: `tests/test_recruitment_ceiling.py`

**Interfaces:**
- Consumes: `cfg: dict[str, str]`, `n_species: int`, `n_dt: int`, `spawning_season: NDArray | None`.
- Produces:
  - `_load_recruitment_ceiling(cfg, n_species, n_dt, spawning_season) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None]` — returns `(ceiling_by_season, enabled_mask)` where `ceiling_by_season` has shape `(n_cols, n_species)` and `enabled_mask` shape `(n_species,)`; both `None` when the master switch is off.
  - `EngineConfig.recruitment_ceiling_by_season: NDArray[np.float64] | None`
  - `EngineConfig.recruitment_ceiling_enabled: NDArray[np.bool_] | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recruitment_ceiling.py`:

```python
import numpy as np
import pytest

from osmose.engine.config import _load_recruitment_ceiling


def _write_ceiling_csv(path, n_cols, cols):
    # cols: dict {species_index: [value per season_idx]}
    sp_ids = sorted(cols)
    header = "season_idx," + ",".join(f"ceiling_sp{i}" for i in sp_ids)
    lines = [header]
    for s in range(n_cols):
        row = [str(s)] + [f"{cols[i][s]:.6f}" for i in sp_ids]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def _cfg(tmp_path, csv_path, enabled_species=(0,)):
    cfg = {
        "_osmose.config.dir": str(tmp_path),
        "reproduction.recruitment.ceiling.enabled": "true",
        "reproduction.recruitment.ceiling.series.file": csv_path.name,
    }
    for sp in enabled_species:
        cfg[f"reproduction.recruitment.ceiling.species.enabled.sp{sp}"] = "true"
    return cfg


def test_ceiling_off_returns_none(tmp_path):
    ceil, mask = _load_recruitment_ceiling({}, 3, 12, None)
    assert ceil is None and mask is None


def test_ceiling_loads_shape_and_mask(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 12, {0: [10.0] * 12, 1: [20.0] * 12, 2: [30.0] * 12})
    ceil, mask = _load_recruitment_ceiling(_cfg(tmp_path, csv, (0, 2)), 3, 12, None)
    assert ceil.shape == (12, 3)
    assert list(mask) == [True, False, True]
    assert ceil[5, 1] == 20.0


def test_ceiling_row_count_must_match_ncols(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 6, {0: [10.0] * 6})
    with pytest.raises(ValueError, match="season"):
        _load_recruitment_ceiling(_cfg(tmp_path, csv), 1, 12, None)


def test_ceiling_rejects_negative(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 12, {0: [-1.0] + [10.0] * 11})
    with pytest.raises(ValueError, match="negative|NaN"):
        _load_recruitment_ceiling(_cfg(tmp_path, csv), 1, 12, None)


def test_ceiling_requires_enabled_species(tmp_path):
    csv = _write_ceiling_csv(tmp_path / "c.csv", 12, {0: [10.0] * 12})
    cfg = _cfg(tmp_path, csv, enabled_species=())
    with pytest.raises(ValueError, match="no species enabled"):
        _load_recruitment_ceiling(cfg, 1, 12, None)


def test_ceiling_missing_file_raises(tmp_path):
    cfg = _cfg(tmp_path, tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        _load_recruitment_ceiling(cfg, 1, 12, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py -k ceiling_ -v`
Expected: FAIL — `ImportError: cannot import name '_load_recruitment_ceiling'`.

- [ ] **Step 3: Add the loader**

In `osmose/engine/config.py`, immediately after `_load_rv_gate` (ends line 1144), add:

```python
def _load_recruitment_ceiling(
    cfg: dict[str, str],
    n_species: int,
    n_dt: int,
    spawning_season: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None]:
    """Load the unfished-level recruitment ceiling (spec 2026-07-03).

    Returns (ceiling_by_season, enabled_mask). ceiling_by_season has shape
    (n_cols, n_species) where n_cols is the model's within-year season count;
    enabled_mask has shape (n_species,). Both are None when the master switch is
    off. Fail-fast (ValueError / FileNotFoundError) on invalid configuration.
    """
    if cfg.get("reproduction.recruitment.ceiling.enabled", "false").lower() != "true":
        return None, None

    n_cols = spawning_season.shape[1] if spawning_season is not None else n_dt

    file_key = cfg.get("reproduction.recruitment.ceiling.series.file", "")
    if not file_key:
        raise ValueError(
            "Recruitment ceiling enabled but "
            "reproduction.recruitment.ceiling.series.file is empty."
        )
    path = _require_file(
        file_key, _cfg_dir(cfg), "reproduction.recruitment.ceiling.series.file"
    )
    df = pd.read_csv(path)
    if "season_idx" not in df.columns:
        raise ValueError(f"Recruitment ceiling {path} missing 'season_idx' column.")
    seasons = df["season_idx"].to_numpy()
    if not np.array_equal(seasons, np.arange(n_cols)):
        raise ValueError(
            f"Recruitment ceiling {path} season_idx must be 0..{n_cols - 1} "
            f"contiguous (model has {n_cols} season columns), got {seasons.tolist()}."
        )

    ceiling = np.full((n_cols, n_species), np.inf, dtype=np.float64)
    for sp in range(n_species):
        col = f"ceiling_sp{sp}"
        if col in df.columns:
            ceiling[:, sp] = df[col].to_numpy(dtype=np.float64)
    # Disabled species keep the inf sentinel (harmless); only real values are
    # checked. NaN is caught here; the finite-column check for ENABLED species
    # is the last loop below.
    finite = np.isfinite(ceiling)
    if np.any(ceiling[finite] < 0):
        raise ValueError(f"Recruitment ceiling {path} has negative values.")
    if np.any(np.isnan(ceiling)):
        raise ValueError(f"Recruitment ceiling {path} has NaN values.")

    enabled = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        key = f"reproduction.recruitment.ceiling.species.enabled.sp{sp}"
        if cfg.get(key, "false").lower() == "true":
            enabled[sp] = True
    if not enabled.any():
        raise ValueError(
            "Recruitment ceiling enabled but no species enabled "
            "(reproduction.recruitment.ceiling.species.enabled.sp{idx})."
        )
    # An enabled species must have a finite ceiling column.
    for sp in np.where(enabled)[0]:
        if not np.all(np.isfinite(ceiling[:, sp])):
            raise ValueError(
                f"Recruitment ceiling enabled for sp{sp} but no ceiling_sp{sp} "
                f"column in {path}."
            )
    return ceiling, enabled
```

- [ ] **Step 4: Add the EngineConfig fields**

In `osmose/engine/config.py`, after line 1401 (`rv_gate_offset: int`), add:

```python

    # Unfished-level recruitment ceiling (both None when disabled)
    recruitment_ceiling_by_season: NDArray[np.float64] | None  # (n_cols, n_species)
    recruitment_ceiling_enabled: NDArray[np.bool_] | None  # (n_species,) enable mask
```

- [ ] **Step 5: Wire into from_dict**

In `osmose/engine/config.py`, at the `_load_rv_gate` call site (line 2124), add after it:

```python
        _spawning_season = _load_spawning_seasons(cfg, n_sp, n_dt)
        recruitment_ceiling_by_season, recruitment_ceiling_enabled = (
            _load_recruitment_ceiling(cfg, n_sp, n_dt, _spawning_season)
        )
```

Then change the kwargs at line 2188 from
`spawning_season=_load_spawning_seasons(cfg, n_sp, n_dt),`
to
`spawning_season=_spawning_season,`
and add alongside the `rv_gate_*` kwargs (after line 2191):

```python
            recruitment_ceiling_by_season=recruitment_ceiling_by_season,
            recruitment_ceiling_enabled=recruitment_ceiling_enabled,
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py -k ceiling_ -v`
Expected: PASS (all 6 loader tests).

- [ ] **Step 7: Verify config-validation stays clean**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -v`
Expected: PASS (no warnings). If it fails on the new keys, add these three patterns to `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py`: `"reproduction.recruitment.ceiling.enabled"`, `"reproduction.recruitment.ceiling.series.file"`, `"reproduction.recruitment.ceiling.species.enabled.sp{idx}"`.

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/config.py tests/test_recruitment_ceiling.py
git commit -m "feat: _load_recruitment_ceiling loader + EngineConfig wiring"
```

---

## Task 3: The clamp in `reproduction()` + parity

**Files:**
- Modify: `osmose/engine/processes/reproduction.py` — insert after the RV-gate block (after line 166, before "Create new schools from eggs").
- Test: `tests/test_recruitment_ceiling.py`

**Interfaces:**
- Consumes: `config.recruitment_ceiling_by_season` `(n_cols, n_sp)`, `config.recruitment_ceiling_enabled` `(n_sp,)`, the local `n_eggs`, `seeded_this_step`, and `step`.
- Produces: `n_eggs` clamped in place before egg-school creation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recruitment_ceiling.py`. Test the clamp **through the real `reproduction()`** (not by re-implementing the clamp expression), using the same scaffolding as `tests/test_engine_reproduction.py`: build a config dict, `EngineConfig.from_dict`, create a mature `SchoolState`, call `reproduction()`, and sum the abundance of the new egg schools (`is_egg`, `age_dt == 0`) for the target species — that sum is the (possibly clamped) recruitment.

```python
from osmose.engine.config import EngineConfig
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import SchoolState


def _repro_cfg_dict():
    # Single-species config that produces a large per-step n_eggs at step 0.
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "10",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.sexratio.sp0": "0.5",
        "species.relativefecundity.sp0": "800",
        "species.maturity.size.sp0": "12.0",
        "population.seeding.biomass.sp0": "50000",
    }


def _mature_state():
    s = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    return s.replace(
        abundance=np.array([1000.0]),
        length=np.array([15.0]),  # > maturity_size 12
        weight=np.array([20.25]),
        biomass=np.array([20250.0]),
        age_dt=np.array([24], dtype=np.int32),
    )


def _eggs_produced(new_state, sp=0):
    fresh = (new_state.age_dt == 0) & new_state.is_egg & (new_state.species_id == sp)
    return float(new_state.abundance[fresh].sum())


def _enable_ceiling(cfg, tmp_path, n_cols, sp0_ceiling):
    csv = _write_ceiling_csv(tmp_path / "c.csv", n_cols, {0: [sp0_ceiling] * n_cols})
    cfg = dict(cfg)
    cfg["_osmose.config.dir"] = str(tmp_path)
    cfg["reproduction.recruitment.ceiling.enabled"] = "true"
    cfg["reproduction.recruitment.ceiling.series.file"] = csv.name
    cfg["reproduction.recruitment.ceiling.species.enabled.sp0"] = "true"
    return cfg


def test_reproduction_uncapped_baseline(tmp_path):
    cfg = EngineConfig.from_dict(_repro_cfg_dict())
    eggs = _eggs_produced(reproduction(_mature_state(), cfg, step=0, rng=np.random.default_rng(0)))
    assert eggs > 0  # sanity: this state produces eggs


def test_reproduction_clamps_when_above_ceiling(tmp_path):
    base = EngineConfig.from_dict(_repro_cfg_dict())
    uncapped = _eggs_produced(
        reproduction(_mature_state(), base, step=0, rng=np.random.default_rng(0))
    )
    cap = uncapped / 2.0
    cfg = EngineConfig.from_dict(_enable_ceiling(_repro_cfg_dict(), tmp_path, 12, cap))
    capped = _eggs_produced(reproduction(_mature_state(), cfg, step=0, rng=np.random.default_rng(0)))
    assert abs(capped - cap) < 1e-3  # clamped to the ceiling


def test_reproduction_unchanged_when_below_ceiling(tmp_path):
    base = EngineConfig.from_dict(_repro_cfg_dict())
    uncapped = _eggs_produced(
        reproduction(_mature_state(), base, step=0, rng=np.random.default_rng(0))
    )
    cfg = EngineConfig.from_dict(_enable_ceiling(_repro_cfg_dict(), tmp_path, 12, uncapped * 2.0))
    result = _eggs_produced(reproduction(_mature_state(), cfg, step=0, rng=np.random.default_rng(0)))
    assert abs(result - uncapped) < 1e-3  # ceiling above production: identical


def test_reproduction_ceiling_skips_seeded_step(tmp_path):
    # Empty state -> SSB is seeded from population.seeding.biomass; seeded eggs
    # must NOT be clipped even with a tiny ceiling.
    cfg = EngineConfig.from_dict(_enable_ceiling(_repro_cfg_dict(), tmp_path, 12, 1.0))
    empty = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
    eggs = _eggs_produced(reproduction(empty, cfg, step=0, rng=np.random.default_rng(0)))
    assert eggs > 1.0  # seeded bootstrap exceeds the ceiling, proving it was skipped
```

Integration parity test (bit-identical when off) — run the EEC engine with the master switch absent vs explicitly false and compare the full biomass frame. `run_in_memory(cfg, seed=0)` returns an `OsmoseResults`; `.biomass()` is a wide DataFrame (a Time column + one column per species name):

```python
def test_ceiling_off_is_bit_identical():
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    reader = OsmoseConfigReader("data/eec/eec_all-parameters.csv")
    cfg = reader.to_dict()
    cfg["simulation.time.nyear"] = "2"
    cfg["simulation.rng.fixed"] = "true"
    cfg["movement.randomseed.fixed"] = "true"
    cfg["stochastic.mortality.randomseed.fixed"] = "true"

    base = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
    cfg["reproduction.recruitment.ceiling.enabled"] = "false"
    off = PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()
    np.testing.assert_array_equal(base.to_numpy(), off.to_numpy())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py -k "reproduction or bit_identical" -v`
Expected: `test_reproduction_uncapped_baseline` PASSES (no clamp involved); `test_reproduction_clamps_when_above_ceiling` and `test_reproduction_ceiling_skips_seeded_step` FAIL — the clamp block does not exist yet, so `reproduction()` returns the uncapped eggs (the "clamps" test sees `uncapped`, not `cap`). `test_ceiling_off_is_bit_identical` should PASS once Task 2 wiring is in place. If any test errors on an unknown `recruitment_ceiling_*` attribute, Task 2 wiring is incomplete — fix Task 2 first.

- [ ] **Step 3: Add the clamp block**

In `osmose/engine/processes/reproduction.py`, after the RV-gate block (after line 166, the line `n_eggs[sp] *= gate[sp]`), insert:

```python

    # Unfished-level recruitment ceiling (McGregor et al. 2019). Inert unless
    # enabled; caps recruitment at its per-season unfished-equilibrium level.
    # Skipped on seeded steps (bootstrap must not be clipped), like the RV gate.
    if config.recruitment_ceiling_by_season is not None:
        assert config.recruitment_ceiling_enabled is not None  # set together
        n_cols_ceil = config.recruitment_ceiling_by_season.shape[0]
        col = step % n_cols_ceil
        for sp in range(n_sp):
            if config.recruitment_ceiling_enabled[sp] and not seeded_this_step[sp]:
                cap = config.recruitment_ceiling_by_season[col, sp]
                if n_eggs[sp] > cap:
                    n_eggs[sp] = cap
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py -v`
Expected: PASS (all tests, including `test_ceiling_off_is_bit_identical`).

- [ ] **Step 5: Run the broader reproduction suite for regressions**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction.py tests/test_engine_stock_recruitment.py tests/test_rv_recruitment_gate.py -q`
Expected: PASS (no regressions from the new clamp).

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/reproduction.py tests/test_recruitment_ceiling.py
git commit -m "feat: unfished-level recruitment ceiling clamp in reproduction()"
```

---

## Task 4: Derivation CLI (F=0 run → per-season sidecar)

**Files:**
- Create: `scripts/derive_recruitment_ceiling.py`
- Test: `tests/test_recruitment_ceiling.py`

**Interfaces:**
- Produces (importable by tests):
  - `zero_fishing(cfg: dict) -> dict` — returns a copy of `cfg` with both fishing modes disabled.
  - `RecruitmentRecorder` — a callable `step_observer` with `.records: list[tuple[int, np.ndarray]]`; `__call__(step, state, grid, config, map_sets)` appends `(step, per_species_recruitment)`.
  - `per_species_recruitment(state, n_species) -> np.ndarray` — sum of abundance for fresh natural eggs (`is_egg & age_dt == 0 & ~from_seeding`) by species.
  - `late_window_ceiling(records, n_cols, n_species, n_dt, frac=1/3) -> np.ndarray` — `(n_cols, n_species)` mean over the last `frac` of model years, grouped by `step % n_cols`.
  - `write_ceiling_csv(ceiling, path) -> Path`.
  - `main(argv=None) -> int` — CLI: `--config PATH --out PATH [--late-frac 0.333]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recruitment_ceiling.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import derive_recruitment_ceiling as derive  # noqa: E402


def test_zero_fishing_disables_both_modes():
    cfg = {
        "module.multispecies.fisheries.enabled": "true",
        "simulation.fishing.mortality.enabled": "true",
    }
    out = derive.zero_fishing(cfg)
    assert out["module.multispecies.fisheries.enabled"] == "false"
    assert out["simulation.fishing.mortality.enabled"] == "false"
    assert cfg["module.multispecies.fisheries.enabled"] == "true"  # original untouched


def test_per_species_recruitment_counts_fresh_natural_eggs():
    from osmose.engine.state import SchoolState

    s = SchoolState.create(n_schools=3, species_id=np.array([0, 0, 1], dtype=np.int32))
    s = s.replace(
        abundance=np.array([100.0, 50.0, 7.0]),
        age_dt=np.array([0, 1, 0], dtype=np.int32),  # 2nd is last-step egg
        is_egg=np.array([True, True, True]),
        from_seeding=np.array([False, False, False]),
    )
    r = derive.per_species_recruitment(s, n_species=2)
    assert r[0] == 100.0  # only the age_dt==0 school for sp0
    assert r[1] == 7.0


def test_late_window_ceiling_buckets_by_season():
    # 2 seasons/year, 4 years; recruitment = season_idx*10 + noise-free
    records = []
    for step in range(8):  # 4 years * 2 seasons
        col = step % 2
        records.append((step, np.array([10.0 * col + 100.0])))
    ceil = derive.late_window_ceiling(records, n_cols=2, n_species=1, n_dt=2, frac=0.5)
    assert ceil.shape == (2, 1)
    assert ceil[0, 0] == 100.0  # season 0
    assert ceil[1, 0] == 110.0  # season 1


def test_write_ceiling_csv_roundtrips(tmp_path):
    ceil = np.array([[100.0, 200.0], [110.0, 210.0]])
    out = derive.write_ceiling_csv(ceil, tmp_path / "c.csv")
    text = out.read_text().strip().splitlines()
    assert text[0] == "season_idx,ceiling_sp0,ceiling_sp1"
    assert text[1].startswith("0,")
    loaded, mask = _load_recruitment_ceiling(
        {
            "_osmose.config.dir": str(tmp_path),
            "reproduction.recruitment.ceiling.enabled": "true",
            "reproduction.recruitment.ceiling.series.file": out.name,
            "reproduction.recruitment.ceiling.species.enabled.sp0": "true",
        },
        2,
        2,
        None,
    )
    np.testing.assert_array_equal(loaded, ceil)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py -k "zero_fishing or per_species or late_window or roundtrips" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'derive_recruitment_ceiling'`.

- [ ] **Step 3: Write the derivation module**

Create `scripts/derive_recruitment_ceiling.py`:

```python
"""Derive the unfished-level recruitment ceiling from an F=0 reference run.

Runs a config with fishing mortality zeroed, records per-step per-species
recruitment (fresh natural eggs) through the engine's step_observer hook, and
writes a per-season sidecar CSV (season_idx,ceiling_sp0,...) that the engine
loads via reproduction.recruitment.ceiling.series.file. See
docs/superpowers/specs/2026-07-03-baltic-recruitment-ceiling-design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def zero_fishing(cfg: dict) -> dict:
    """Copy of cfg with both fishing modes disabled (rate-based + v4 fisheries)."""
    out = dict(cfg)
    out["module.multispecies.fisheries.enabled"] = "false"
    out["simulation.fishing.mortality.enabled"] = "false"
    return out


def per_species_recruitment(state, n_species: int) -> np.ndarray:
    """Sum abundance of fresh natural eggs (age_dt==0, is_egg, not seeded) by species."""
    out = np.zeros(n_species, dtype=np.float64)
    fresh = (state.age_dt == 0) & state.is_egg
    if state.from_seeding is not None:
        fresh = fresh & (~state.from_seeding)
    if fresh.any():
        np.add.at(out, state.species_id[fresh], state.abundance[fresh])
    return out


class RecruitmentRecorder:
    """step_observer that records (step, per-species recruitment) each step."""

    def __init__(self, n_species: int):
        self.n_species = n_species
        self.records: list[tuple[int, np.ndarray]] = []

    def __call__(self, step, state, grid, config, map_sets):
        self.records.append((step, per_species_recruitment(state, self.n_species)))


def late_window_ceiling(records, n_cols: int, n_species: int, n_dt: int, frac: float) -> np.ndarray:
    """Mean recruitment over the last `frac` of model years, grouped by step % n_cols."""
    if not records:
        raise ValueError("No recruitment records; cannot derive a ceiling.")
    max_step = max(s for s, _ in records)
    n_years = (max_step + 1) / n_dt
    start_year = n_years * (1.0 - frac)
    start_step = int(start_year * n_dt)
    sums = np.zeros((n_cols, n_species), dtype=np.float64)
    counts = np.zeros(n_cols, dtype=np.int64)
    for step, rec in records:
        if step < start_step:
            continue
        col = step % n_cols
        sums[col] += rec
        counts[col] += 1
    if np.any(counts == 0):
        raise ValueError(
            f"Late window covers only {counts.tolist()} steps per season; "
            f"increase run length or late-frac."
        )
    return sums / counts[:, None]


def check_stationarity(records, n_cols, n_species, n_dt, frac, tol=0.25) -> list[str]:
    """Warn if the last-window per-season mean differs from the preceding window
    by more than `tol` (relative). Returns a list of warning strings (empty = ok)."""
    late = late_window_ceiling(records, n_cols, n_species, n_dt, frac)
    # Preceding window of the same width, immediately before the late window.
    max_step = max(s for s, _ in records)
    n_years = (max_step + 1) / n_dt
    lo = int(n_years * (1.0 - 2 * frac) * n_dt)
    hi = int(n_years * (1.0 - frac) * n_dt)
    sums = np.zeros((n_cols, n_species))
    counts = np.zeros(n_cols, dtype=np.int64)
    for step, rec in records:
        if lo <= step < hi:
            sums[step % n_cols] += rec
            counts[step % n_cols] += 1
    warnings: list[str] = []
    if np.any(counts == 0):
        return ["Preceding window empty; cannot assess stationarity (run longer)."]
    prev = sums / counts[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(late - prev) / np.where(late > 0, late, np.nan)
    bad = np.nanmax(rel) if np.isfinite(np.nanmax(rel)) else 0.0
    if bad > tol:
        warnings.append(
            f"Unfished run may not be stationary: max per-season drift "
            f"{bad:.0%} > {tol:.0%}. The derived ceiling may be unreliable."
        )
    return warnings


def write_ceiling_csv(ceiling: np.ndarray, path: Path) -> Path:
    path = Path(path)
    n_cols, n_sp = ceiling.shape
    header = "season_idx," + ",".join(f"ceiling_sp{i}" for i in range(n_sp))
    lines = [header]
    for s in range(n_cols):
        lines.append(str(s) + "," + ",".join(f"{ceiling[s, i]:.6f}" for i in range(n_sp)))
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Derive the unfished recruitment ceiling.")
    ap.add_argument("--config", required=True, help="Path to the all-parameters config CSV.")
    ap.add_argument("--out", required=True, help="Output sidecar CSV path.")
    ap.add_argument("--late-frac", type=float, default=1.0 / 3.0)
    args = ap.parse_args(argv)

    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig

    reader = OsmoseConfigReader(args.config)
    cfg = zero_fishing(reader.to_dict())
    engine_cfg = EngineConfig.from_dict(cfg)
    n_sp = engine_cfg.n_species
    n_dt = engine_cfg.n_dt_per_year
    n_cols = (
        engine_cfg.spawning_season.shape[1]
        if engine_cfg.spawning_season is not None
        else n_dt
    )

    # run() (not run_in_memory) forwards step_observer; it needs an output_dir,
    # so send disk outputs to a throwaway temp dir — we only want the recorder.
    import tempfile

    recorder = RecruitmentRecorder(n_sp)
    with tempfile.TemporaryDirectory() as td:
        PythonEngine().run(cfg, Path(td), seed=0, step_observer=recorder)

    for w in check_stationarity(recorder.records, n_cols, n_sp, n_dt, args.late_frac):
        print("WARNING:", w)
    ceiling = late_window_ceiling(recorder.records, n_cols, n_sp, n_dt, args.late_frac)
    out = write_ceiling_csv(ceiling, args.out)
    print(f"Wrote ceiling ({ceiling.shape[0]} seasons x {ceiling.shape[1]} species) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

API confirmed against `osmose/engine/__init__.py:87-117`: `PythonEngine().run(config, output_dir, seed=0, *, step_observer=...)` forwards `step_observer` to `simulate()` (verified). `run_in_memory()` does NOT forward it, so `main` uses `run()` with a temp `output_dir`. The four pure helpers (`zero_fishing`, `per_species_recruitment`, `late_window_ceiling`, `write_ceiling_csv`) are fully unit-tested independent of the run wiring.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py -k "zero_fishing or per_species or late_window or roundtrips" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Wire `main` to the real run path and smoke-test on EEC**

Confirm the run wiring by generating a ceiling for a fast config:

Run: `.venv/bin/python scripts/derive_recruitment_ceiling.py --config data/eec/eec_all-parameters.csv --out /tmp/eec_ceiling.csv`
Expected: prints `Wrote ceiling (... seasons x ... species)`; `/tmp/eec_ceiling.csv` has a `season_idx` header and one row per season. (Fix the `main` run wiring per the Step-3 note if it errors.)

- [ ] **Step 6: Commit**

```bash
git add scripts/derive_recruitment_ceiling.py tests/test_recruitment_ceiling.py
git commit -m "feat: derive_recruitment_ceiling CLI (F=0 run -> per-season sidecar)"
```

---

## Task 5: A/B boom/bust diagnostic + generate the Baltic ceiling

**Files:**
- Create: `scripts/baltic_recruitment_ceiling_diagnostic.py`
- Create (generated): `data/baltic/baltic_recruitment_ceiling.csv`
- Test: `tests/test_recruitment_ceiling.py`

**Interfaces:**
- Consumes: `derive.late_window_ceiling` / the engine run path (Task 4); the loaded ceiling (Tasks 2-3).
- Produces:
  - `overshoot_ratio(biomass_series) -> float` — a boom/bust metric (max / late-mean) for a species' biomass series.
  - `run_ab(config_path, cod_index) -> dict` — runs Baltic cod ceiling off vs on, returns `{"off": ratio_off, "on": ratio_on}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recruitment_ceiling.py`:

```python
import baltic_recruitment_ceiling_diagnostic as abdiag  # noqa: E402


def test_overshoot_ratio_basic():
    series = np.array([100.0, 300.0, 200.0, 150.0, 150.0, 150.0])
    # max=300, late-mean over last 3 = 150 -> ratio 2.0
    assert abs(abdiag.overshoot_ratio(series, late_frac=0.5) - 2.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py -k overshoot_ratio -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'baltic_recruitment_ceiling_diagnostic'`.

- [ ] **Step 3: Write the diagnostic module**

Create `scripts/baltic_recruitment_ceiling_diagnostic.py`:

```python
"""A/B diagnostic: Baltic cod boom/bust overshoot with the recruitment ceiling
off vs on. This is the go/no-go signal for the ceiling lever (mirrors the RV-gate
diagnostic). See docs/superpowers/specs/2026-07-03-baltic-recruitment-ceiling-design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def overshoot_ratio(biomass_series: np.ndarray, late_frac: float = 1.0 / 3.0) -> float:
    """Boom/bust overshoot = peak biomass / late-window mean biomass."""
    b = np.asarray(biomass_series, dtype=np.float64)
    n = len(b)
    late = b[int(n * (1.0 - late_frac)) :]
    late_mean = float(np.mean(late))
    if late_mean <= 0:
        return float("inf")
    return float(np.max(b)) / late_mean


def run_ab(config_path: str, cod_index: int, cod_name: str = "cod", out_ceiling=None) -> dict:
    """Run Baltic with the ceiling off, derive the ceiling, run with it on for cod,
    and return the cod overshoot ratio in both cases.

    Biomass series come from OsmoseResults.biomass() — a wide frame keyed by
    species NAME (column `cod_name`); the enable key uses the species INDEX.
    """
    import tempfile

    import derive_recruitment_ceiling as derive
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig

    reader = OsmoseConfigReader(config_path)
    base = reader.to_dict()

    # 1. OFF run — cod biomass series (wide frame, species-name column).
    off_series = PythonEngine().run_in_memory(dict(base), seed=0).biomass()[cod_name].to_numpy()

    # 2. Derive the ceiling from an F=0 run (run() forwards step_observer).
    out_ceiling = out_ceiling or str(Path(config_path).parent / "baltic_recruitment_ceiling.csv")
    zero = derive.zero_fishing(dict(base))
    ecfg = EngineConfig.from_dict(zero)
    n_dt = ecfg.n_dt_per_year
    n_cols = ecfg.spawning_season.shape[1] if ecfg.spawning_season is not None else n_dt
    rec = derive.RecruitmentRecorder(ecfg.n_species)
    with tempfile.TemporaryDirectory() as td:
        PythonEngine().run(zero, Path(td), seed=0, step_observer=rec)
    ceiling = derive.late_window_ceiling(rec.records, n_cols, ecfg.n_species, n_dt, 1.0 / 3.0)
    derive.write_ceiling_csv(ceiling, out_ceiling)

    # 3. ON run — enable the ceiling for cod only.
    on_cfg = dict(base)
    on_cfg["reproduction.recruitment.ceiling.enabled"] = "true"
    on_cfg["reproduction.recruitment.ceiling.series.file"] = Path(out_ceiling).name
    on_cfg[f"reproduction.recruitment.ceiling.species.enabled.sp{cod_index}"] = "true"
    on_series = PythonEngine().run_in_memory(on_cfg, seed=0).biomass()[cod_name].to_numpy()

    return {"off": overshoot_ratio(off_series), "on": overshoot_ratio(on_series)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baltic cod recruitment-ceiling A/B diagnostic.")
    ap.add_argument("--config", default="data/baltic/baltic_all-parameters.csv")
    ap.add_argument("--cod-index", type=int, required=True)
    ap.add_argument("--cod-name", default="cod")
    args = ap.parse_args(argv)
    res = run_ab(args.config, args.cod_index, args.cod_name)
    print(f"cod overshoot ratio  OFF={res['off']:.3f}  ON={res['on']:.3f}")
    if res["on"] < res["off"]:
        print("GO: ceiling damps the boom/bust overshoot.")
    else:
        print("NO-GO: ceiling does not damp overshoot (rule out, like the RV gate).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

API confirmed: `PythonEngine().run_in_memory(cfg, seed=0).biomass()` returns a wide DataFrame keyed by species name (see `scripts/baltic_rv_overshoot_diagnostic.py:258-273`, which reads `df["cod"]` exactly this way). `EngineConfig.from_dict(cfg).n_species` gives the species count for the recorder.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py -k overshoot_ratio -v`
Expected: PASS.

- [ ] **Step 5: Determine the cod species index and run the A/B**

Find cod's index:

Run: `grep -n "cod" data/baltic/baltic_param-species.csv | head`
Then run the diagnostic (this also generates `data/baltic/baltic_recruitment_ceiling.csv`):

Run: `.venv/bin/python scripts/baltic_recruitment_ceiling_diagnostic.py --cod-index <COD_IDX>`
Expected: prints `cod overshoot ratio OFF=... ON=...` and a GO/NO-GO verdict. Record the numbers — they are the scientific result of this lever.

- [ ] **Step 6: Commit**

```bash
git add scripts/baltic_recruitment_ceiling_diagnostic.py data/baltic/baltic_recruitment_ceiling.csv tests/test_recruitment_ceiling.py
git commit -m "feat: Baltic cod recruitment-ceiling A/B boom/bust diagnostic + generated sidecar"
```

---

## Task 6: Full-suite gate, lint, and final verification

**Files:** none new (verification only).

- [ ] **Step 1: Lint and format**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ scripts/`
Run: `.venv/bin/ruff format --check osmose/ ui/ tests/ scripts/`
Expected: clean. Fix any findings and re-run.

- [ ] **Step 2: Run the full test module + related suites**

Run: `.venv/bin/python -m pytest tests/test_recruitment_ceiling.py tests/test_engine_config_validation.py tests/test_rv_recruitment_gate.py tests/test_engine_reproduction.py -q`
Expected: all PASS.

- [ ] **Step 3: Confirm inert-by-default across bundled configs**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -q`
Expected: PASS (no warnings for the new keys on EEC / Bay of Biscay / Baltic).

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint + final verification for recruitment ceiling"
```

---

## Self-Review Notes (author, against the spec)

- **Spec §4 Part A (derivation → sidecar):** Task 4. Simplification vs spec: recording uses the *existing* `step_observer` hook + `age_dt==0` fresh-egg filter, so no new engine hot-path hook is added (spec allowed an optional hook; this is strictly less invasive and zero-overhead in production). Documented in the plan header and Task 4.
- **Spec §4 Part B (engine clamp, inert/seeding-safe/bit-identical):** Tasks 2-3.
- **Spec §5 config keys + allowlist:** Task 1 (schema) + Task 2 Step 7 (validation).
- **Spec §6 per-season alignment + stationarity check:** loader validates `season_idx == arange(n_cols)` (Task 2); `check_stationarity` (Task 4).
- **Spec §8 testing (unit / parity / A-B):** Tasks 2-3 (unit + parity), Task 5 (A/B).
- **Scope (cod-only, percids out):** Task 5 enables only `cod_index`.
- **Engine API pinned (no placeholders):** `PythonEngine().run(cfg, output_dir, seed, step_observer=...)` forwards the observer (`osmose/engine/__init__.py:87-117`); `run_in_memory(cfg, seed).biomass()` returns a wide species-name-keyed frame (`scripts/baltic_rv_overshoot_diagnostic.py:258-273`); `EngineConfig.from_dict(cfg).n_species` / `.spawning_season` give the shape. Derivation uses `run()` (observer needs an `output_dir`, sent to a temp dir); A/B and parity use `run_in_memory().biomass()`. All pure logic sits behind unit-tested helpers.
- **One residual discovery step (not a placeholder):** cod's species index for the enable key is found by `grep`-ing the Baltic species config in Task 5 Step 5 — a lookup, not undefined behavior.
