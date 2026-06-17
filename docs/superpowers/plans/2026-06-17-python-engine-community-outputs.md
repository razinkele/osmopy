# Python-engine Community Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Python engine persist the community `{biomass,abundance}DistribBySize` CSVs and a 1D `meanTL` CSV so the Size Spectrum, Sheldon spectrum, and MTL/MTI diagnostics populate on Python-only configs.

**Architecture:** Pure output-layer work — the realized per-school TL is already computed in `mortality.py` (`state.trophic_level`) and per-size data already lives in `StepOutput`. We add one config flag, capture a per-species mean-TL dict into `StepOutput`, and add two builder+writer pairs in `osmose/engine/output.py`. No predation/mortality hot-loop change.

**Tech Stack:** Python 3.12, numpy, pandas, pytest. Run with `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-17-python-engine-community-outputs-design.md`

## Key facts (read before starting)

- `StepOutput` (`osmose/engine/simulate.py:76`) carries `biomass_by_size`/`abundance_by_size: dict[int, NDArray] | None` (focal-species index → per-size-bin array). Constructed in 3 places: `_collect_outputs` (≈1020) and TWO returns in `_average_step_outputs` (the `len(accumulated)==1` passthrough ≈1134 and the averaged path ≈1197). Any new field must be set in all three.
- Realized TL: `state.trophic_level` (`SchoolState`, state.py:49) is the emergent per-school TL maintained by predation in `mortality.py`. Focal schools have `species_id < config.n_species`; background have `species_id >= n_species`.
- Per-species aggregation pattern (mirror it): `_collect_biomass_abundance` uses `np.add.at(arr, state.species_id[mask], …)` with an optional `config.output_cutoff_age` filter (`age_years = state.age_dt/config.n_dt_per_year; include = age_years >= cutoff[species_id]`).
- Config flags live in `osmose/engine/config.py`: the `_output` dict (≈900), `EngineConfig` fields (≈1427), and `from_dict` wiring (≈2079). `output_biomass_bysize`/`output_abundance_bysize` already exist; `output_size_min`/`output_size_incr` give the size-bin geometry. `_enabled(cfg, "<key>")` parses a bool. NO meanTL flag exists yet.
- `write_outputs` (`osmose/engine/output.py:23`) calls family writers (`_write_distribution_csvs`, etc.). The existing `_write_distribution_csvs` writes per-species `biomassBySize_<sp>` files with `df.to_csv(index=False)` (clean header). Keep those unchanged; add the community files alongside.
- Test pattern (sibling `tests/test_engine_distribution_output.py`): `EngineConfig.from_dict(_make_config({...}))`, build `StepOutput(...)`, `write_outputs(outputs, tmp_path, config)`, read CSVs with pandas. `SchoolState.create(n).replace(species_id=…, abundance=…, trophic_level=…, age_dt=…)` builds a test state.

## File structure

- **Modify** `osmose/engine/config.py` — add `output_meantl` flag.
- **Modify** `osmose/engine/simulate.py` — `StepOutput.mean_tl` field; `_collect_mean_tl` helper; capture in `_collect_outputs`; carry through both `_average_step_outputs` returns.
- **Modify** `osmose/engine/output.py` — `_build_meantl_dataframe`/`_write_meantl_csv` and `_build_distrib_bysize_community_dataframes`/`_write_distrib_bysize_community_csvs`; wire both into `write_outputs`.
- **Modify** `data/baltic/baltic_param-output.csv` — enable `output.meanTL.enabled`.
- **Create** `tests/test_engine_community_output.py` — unit + integration tests.

---

## Task 1: Add the `output_meantl` config flag

**Files:** Modify `osmose/engine/config.py`; Create `tests/test_engine_community_output.py`

- [ ] **Step 1: Write the failing test** (create `tests/test_engine_community_output.py`)

```python
"""Tests for Python-engine community outputs: DistribBySize + meanTL."""

import numpy as np
import pandas as pd

from osmose.engine.config import EngineConfig


def _base_config(extra: dict | None = None) -> dict[str, str]:
    base = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "5",
        "simulation.nschool.sp1": "5",
        "species.name.sp0": "Anchovy",
        "species.name.sp1": "Hake",
        "species.linf.sp0": "19.5",
        "species.linf.sp1": "110.0",
        "species.k.sp0": "0.364",
        "species.k.sp1": "0.106",
        "species.t0.sp0": "-0.70",
        "species.t0.sp1": "-0.17",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.005",
        "species.length2weight.allometric.power.sp0": "3.06",
        "species.length2weight.allometric.power.sp1": "3.14",
        "species.lifespan.sp0": "4",
        "species.lifespan.sp1": "12",
        "species.vonbertalanffy.threshold.age.sp0": "0",
        "species.vonbertalanffy.threshold.age.sp1": "0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
    }
    if extra:
        base.update(extra)
    return base


def test_output_meantl_flag_defaults_false():
    config = EngineConfig.from_dict(_base_config())
    assert config.output_meantl is False


def test_output_meantl_flag_enabled():
    config = EngineConfig.from_dict(_base_config({"output.meanTL.enabled": "true"}))
    assert config.output_meantl is True
```

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py -q`
Expected: FAIL with `AttributeError: 'EngineConfig' object has no attribute 'output_meantl'`.

- [ ] **Step 3: Add the flag in `osmose/engine/config.py`**

3a. In the `_output` dict (next to `"output_abundance_bysize": _enabled(cfg, "output.abundance.bysize.enabled"),`), add:
```python
        "output_meantl": _enabled(cfg, "output.meanTL.enabled"),
```

3b. In the `EngineConfig` dataclass distribution-flags block (next to `output_abundance_bysize: bool = False`), add:
```python
    output_meantl: bool = False
```

3c. In `from_dict` (next to `output_abundance_bysize=_output["output_abundance_bysize"],`), add:
```python
            output_meantl=_output["output_meantl"],
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py -q` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py tests/test_engine_community_output.py
git commit -m "feat(engine): add output.meanTL.enabled config flag (output_meantl)"
```

---

## Task 2: Capture per-species mean TL into StepOutput

**Files:** Modify `osmose/engine/simulate.py`; Modify `tests/test_engine_community_output.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.engine.simulate import StepOutput, _collect_mean_tl
from osmose.engine.state import SchoolState


def test_collect_mean_tl_abundance_weighted_excludes_zero_tl():
    config = EngineConfig.from_dict(_base_config())  # 2 focal species, no cutoff age
    # 3 schools: sp0 two schools (TL 4.0 ab 30, TL 2.0 ab 10) -> weighted (4*30+2*10)/40 = 3.5;
    # sp1 one school TL 0.0 (unfed/egg) -> excluded -> sp1 absent from the dict.
    state = SchoolState.create(3).replace(
        species_id=np.array([0, 0, 1], dtype=np.int32),
        abundance=np.array([30.0, 10.0, 50.0], dtype=np.float64),
        trophic_level=np.array([4.0, 2.0, 0.0], dtype=np.float64),
        age_dt=np.array([12, 12, 12], dtype=np.int32),
    )
    out = _collect_mean_tl(state, config)
    assert out[0] == pytest.approx(3.5)
    assert 1 not in out  # sp1's only school has TL 0 -> excluded


def test_collect_mean_tl_empty_state():
    config = EngineConfig.from_dict(_base_config())
    assert _collect_mean_tl(SchoolState.create(0), config) == {}
```

Add `import pytest` at the top of the test file if not already present.

- [ ] **Step 2: Run, verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py -k mean_tl -q`
Expected: FAIL with `ImportError: cannot import name '_collect_mean_tl'`.

- [ ] **Step 3: Implement in `osmose/engine/simulate.py`**

3a. Add the `mean_tl` field to `StepOutput` (after `abundance_by_size`):
```python
    # Per-species realized mean trophic level (sp_idx -> abundance-weighted mean TL), or None
    mean_tl: dict[int, float] | None = None
```

3b. Add the helper (place near `_collect_biomass_abundance`):
```python
def _collect_mean_tl(state: SchoolState, config: EngineConfig) -> dict[int, float]:
    """Abundance-weighted realized mean trophic level per FOCAL species.

    Aggregates the emergent per-school ``state.trophic_level`` (maintained by predation
    in mortality.py). Includes only focal-species schools (species_id < n_species) with
    trophic_level > 0 and abundance > 0 (excludes background and unfed/egg schools whose
    TL is still the 0 sentinel). Applies the same output cutoff-age filter as the biomass
    output. Species with no qualifying school are omitted from the returned dict.
    """
    n_sp = config.n_species
    wsum = np.zeros(n_sp, dtype=np.float64)
    asum = np.zeros(n_sp, dtype=np.float64)
    if len(state) > 0:
        sp = state.species_id
        tl = state.trophic_level
        ab = state.abundance
        mask = (sp < n_sp) & (tl > 0) & (ab > 0)
        if config.output_cutoff_age is not None:
            age_years = state.age_dt.astype(np.float64) / config.n_dt_per_year
            cutoff = config.output_cutoff_age[sp]
            mask = mask & (age_years >= cutoff)
        np.add.at(wsum, sp[mask], ab[mask] * tl[mask])
        np.add.at(asum, sp[mask], ab[mask])
    return {i: float(wsum[i] / asum[i]) for i in range(n_sp) if asum[i] > 0}
```

3c. In `_collect_outputs`, after the `biomass_by_age, … = _collect_distributions(...)` line, add:
```python
    mean_tl = _collect_mean_tl(state, config) if config.output_meantl else None
```
and add `mean_tl=mean_tl,` to the `return StepOutput(...)` (e.g. after `abundance_by_size=abundance_by_size,`).

3d. In `_average_step_outputs`, the `len(accumulated) == 1` return (≈1134): add
```python
            mean_tl=accumulated[0].mean_tl,
```
and in the averaged return (≈1197), first define a local averager near the existing `_avg_spatial`:
```python
    def _avg_scalar_dict(attr: str) -> dict[int, float] | None:
        dicts = [getattr(o, attr) for o in accumulated if getattr(o, attr) is not None]
        if not dicts:
            return None
        keys: set[int] = set().union(*[set(d.keys()) for d in dicts])
        return {k: float(np.mean([d[k] for d in dicts if k in d])) for k in keys}
```
and add `mean_tl=_avg_scalar_dict("mean_tl"),` to that return.

- [ ] **Step 4: Run, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py -k mean_tl -q` → 2 pass.
Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py tests/test_distribution_averaging.py -q` → no regressions (StepOutput still constructs everywhere).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_community_output.py
git commit -m "feat(engine): capture abundance-weighted per-species meanTL into StepOutput"
```

---

## Task 3: Write the meanTL CSV

**Files:** Modify `osmose/engine/output.py`; Modify `tests/test_engine_community_output.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.engine.output import _build_meantl_dataframe, write_outputs


def _step_output(step, biomass, abundance, **kwargs):
    n_sp = len(biomass)
    from osmose.engine.state import MortalityCause

    return StepOutput(
        step=step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=np.zeros((n_sp, len(MortalityCause)), dtype=np.float64),
        **kwargs,
    )


def test_build_meantl_dataframe_wide():
    config = EngineConfig.from_dict(_base_config({"output.meanTL.enabled": "true"}))
    outputs = [
        _step_output(0, np.array([1.0, 1.0]), np.array([1.0, 1.0]), mean_tl={0: 3.5, 1: 4.2}),
        _step_output(12, np.array([1.0, 1.0]), np.array([1.0, 1.0]), mean_tl={0: 3.6}),
    ]
    dfs = _build_meantl_dataframe(outputs, config)
    df = dfs["meanTL"]
    assert list(df.columns) == ["Time", "Anchovy", "Hake"]
    assert df["Anchovy"].tolist() == pytest.approx([3.5, 3.6])
    assert df["Hake"][0] == pytest.approx(4.2)
    assert np.isnan(df["Hake"][1])  # absent that step -> NaN


def test_write_meantl_csv_gated(tmp_path):
    config_off = EngineConfig.from_dict(_base_config())  # flag off
    outputs = [_step_output(0, np.array([1.0, 1.0]), np.array([1.0, 1.0]), mean_tl={0: 3.5})]
    write_outputs(outputs, tmp_path, config_off, prefix="osm")
    assert not (tmp_path / "osm_meanTL_Simu0.csv").exists()

    config_on = EngineConfig.from_dict(_base_config({"output.meanTL.enabled": "true"}))
    write_outputs(outputs, tmp_path, config_on, prefix="osm")
    written = pd.read_csv(tmp_path / "osm_meanTL_Simu0.csv")
    assert "Anchovy" in written.columns and written["Anchovy"][0] == pytest.approx(3.5)
```

- [ ] **Step 2: Run, verify FAIL** (`ImportError: cannot import name '_build_meantl_dataframe'`).

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py -k meantl_dataframe -q`

- [ ] **Step 3: Implement in `osmose/engine/output.py`**

Add (near `_build_distribution_dataframes`):
```python
def _build_meantl_dataframe(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Wide Time + per-species realized mean trophic level. Empty dict when disabled/absent."""
    if not config.output_meantl or not any(o.mean_tl is not None for o in outputs):
        return {}
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    sp_names = config.species_names
    data = np.full((len(outputs), len(sp_names)), np.nan, dtype=np.float64)
    for t_idx, o in enumerate(outputs):
        if o.mean_tl:
            for sp_idx, val in o.mean_tl.items():
                if sp_idx < len(sp_names):
                    data[t_idx, sp_idx] = val
    df = pd.DataFrame(data, columns=sp_names)  # type: ignore[arg-type]
    df.insert(0, "Time", times)
    return {"meanTL": df}


def _write_meantl_csv(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write {prefix}_meanTL_Simu0.csv (clean header; readers auto-detect preamble)."""
    for key, df in _build_meantl_dataframe(outputs, config).items():
        df.to_csv(output_dir / f"{prefix}_{key}_Simu0.csv", index=False)
```

Wire into `write_outputs` after `_write_distribution_csvs(output_dir, prefix, outputs, config)`:
```python
    _write_meantl_csv(output_dir, prefix, outputs, config)
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py -k "meantl_dataframe or meantl_csv" -q` → pass.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/output.py tests/test_engine_community_output.py
git commit -m "feat(engine): write 1D meanTL CSV from captured per-species mean TL"
```

---

## Task 4: Write the community DistribBySize CSVs

**Files:** Modify `osmose/engine/output.py`; Modify `tests/test_engine_community_output.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.engine.output import _build_distrib_bysize_community_dataframes


def test_build_distrib_bysize_community():
    config = EngineConfig.from_dict(
        _base_config(
            {
                "output.biomass.bysize.enabled": "true",
                "output.distrib.bysize.min": "0",
                "output.distrib.bysize.incr": "10",
            }
        )
    )
    # 1 step, 3 size bins; sp0 = [1,2,3], sp1 = [4,5,6].
    outputs = [
        _step_output(
            0,
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            biomass_by_size={0: np.array([1.0, 2.0, 3.0]), 1: np.array([4.0, 5.0, 6.0])},
            abundance_by_size={0: np.array([1.0, 2.0, 3.0]), 1: np.array([4.0, 5.0, 6.0])},
        )
    ]
    dfs = _build_distrib_bysize_community_dataframes(outputs, config)
    df = dfs["biomassDistribBySize"]
    assert list(df.columns) == ["Time", "Size", "Anchovy", "Hake"]
    assert df["Size"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert df["Anchovy"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert df["Hake"].tolist() == pytest.approx([4.0, 5.0, 6.0])
    # abundance flag is off in this config -> only the biomass community file is built
    assert "abundanceDistribBySize" not in dfs


def test_distrib_bysize_community_written_and_readable(tmp_path):
    from osmose.size_spectrum import _read_community_by_size

    config = EngineConfig.from_dict(
        _base_config({"output.biomass.bysize.enabled": "true"})
    )
    outputs = [
        _step_output(
            0,
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            biomass_by_size={0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])},
            abundance_by_size={0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])},
        )
    ]
    write_outputs(outputs, tmp_path, config, prefix="osm")
    wide = _read_community_by_size(tmp_path, "biomassDistribBySize", "osm")
    assert list(wide.columns) == ["Time", "Size", "Anchovy", "Hake"]
```

- [ ] **Step 2: Run, verify FAIL** (`ImportError: cannot import name '_build_distrib_bysize_community_dataframes'`).

- [ ] **Step 3: Implement in `osmose/engine/output.py`**

```python
def _build_distrib_bysize_community_dataframes(
    outputs: list[StepOutput],
    config: EngineConfig,
) -> dict[str, pd.DataFrame]:
    """Community {metric}DistribBySize: wide Time, Size, <species> from the per-size StepOutput
    data, reshaped to the Java community layout (one Size row per bin, one column per species).

    Only the metric(s) whose bysize flag is set are built. Missing per-species/per-bin cells are
    zero-filled (matching the per-species distribution writer).
    """
    result: dict[str, pd.DataFrame] = {}
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    sp_names = config.species_names
    for metric, attr, flag in (
        ("biomassDistribBySize", "biomass_by_size", config.output_biomass_bysize),
        ("abundanceDistribBySize", "abundance_by_size", config.output_abundance_bysize),
    ):
        if not flag:
            continue
        n_bins = 0
        for o in outputs:
            d = getattr(o, attr)
            if d:
                for arr in d.values():
                    n_bins = max(n_bins, len(arr))
        if n_bins == 0:
            continue
        edges = [config.output_size_min + k * config.output_size_incr for k in range(n_bins)]
        rows: list[dict] = []
        for t_idx, o in enumerate(outputs):
            d = getattr(o, attr) or {}
            for k in range(n_bins):
                row: dict[str, float] = {"Time": float(times[t_idx]), "Size": float(edges[k])}
                for sp_idx, sp_name in enumerate(sp_names):
                    arr = d.get(sp_idx)
                    row[sp_name] = float(arr[k]) if arr is not None and k < len(arr) else 0.0
                rows.append(row)
        result[metric] = pd.DataFrame(rows, columns=["Time", "Size", *sp_names])  # type: ignore[arg-type]
    return result


def _write_distrib_bysize_community_csvs(
    output_dir: Path,
    prefix: str,
    outputs: list[StepOutput],
    config: EngineConfig,
) -> None:
    """Write {prefix}_{metric}DistribBySize_Simu0.csv (community layout, clean header)."""
    for key, df in _build_distrib_bysize_community_dataframes(outputs, config).items():
        df.to_csv(output_dir / f"{prefix}_{key}_Simu0.csv", index=False)
```

Wire into `write_outputs` immediately after `_write_distribution_csvs(...)` (and the new `_write_meantl_csv`):
```python
    _write_distrib_bysize_community_csvs(output_dir, prefix, outputs, config)
```

- [ ] **Step 4: Run, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/output.py tests/test_engine_community_output.py
git commit -m "feat(engine): write community biomass/abundanceDistribBySize CSVs"
```

---

## Task 5: Enable in Baltic config + end-to-end integration

**Files:** Modify `data/baltic/baltic_param-output.csv`; Modify `tests/test_engine_community_output.py`

- [ ] **Step 1: Enable 1D meanTL in the Baltic config**

Add a line to `data/baltic/baltic_param-output.csv` (semicolon-separated, matching the existing `output.meanTL.bySize.enabled;true` line):
```
output.meanTL.enabled;true
```
(The by-size biomass/abundance flags are already `true` there, so DistribBySize needs no config change.)

- [ ] **Step 2: Write the end-to-end integration test** (append to `tests/test_engine_community_output.py`)

This runs the REAL engine on the examples config (predation active → TL>0) and confirms the new outputs feed the diagnostics. The setup mirrors `tests/test_engine_parity.py::_run_engine` exactly — the real entry point is `osmose.engine.simulate.simulate(cfg, grid, rng) -> list[StepOutput]`:

```python
def test_community_outputs_feed_diagnostics(tmp_path):
    """Real engine run (examples config, predation active) with the flags on -> the community
    CSVs appear and the size/Sheldon/trophic diagnostics read them instead of degrading."""
    from osmose.community_metrics import compute_sheldon_spectrum, compute_trophic_indicators
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate
    from osmose.size_spectrum import compute_size_spectrum

    project_dir = Path(__file__).parent.parent
    examples_config = project_dir / "data" / "examples" / "osm_all-parameters.csv"
    if not examples_config.exists():
        pytest.skip("No example config for the engine-run integration test")

    raw = OsmoseConfigReader().read(examples_config)
    raw["simulation.time.nyear"] = "2"
    raw["output.biomass.bysize.enabled"] = "true"
    raw["output.abundance.bysize.enabled"] = "true"
    raw["output.meanTL.enabled"] = "true"
    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            project_dir / "data" / "examples" / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        grid = Grid.from_dimensions(
            ny=int(raw.get("grid.nline", "1")), nx=int(raw.get("grid.ncolumn", "1"))
        )

    outputs = simulate(cfg, grid, np.random.default_rng(42))
    write_outputs(outputs, tmp_path, cfg, prefix="osm")

    assert (tmp_path / "osm_biomassDistribBySize_Simu0.csv").exists()
    assert (tmp_path / "osm_meanTL_Simu0.csv").exists()

    # Sheldon spectrum reads the community by-size file (raw config provides a,b).
    spec = compute_sheldon_spectrum(tmp_path, raw, window_years=2)
    assert spec.mass_bin_midpoints  # non-empty -> spectrum built from real by-size data
    # Trophic indicators read meanTL; MTL is a realized TL in a plausible ecological range.
    trophic = compute_trophic_indicators(tmp_path, window_years=2)
    assert trophic.n_species >= 1
    assert 1.0 <= trophic.mtl <= 6.0  # sanity bound on realized community TL
    # Length spectrum (existing diagnostic) also now works on a Python run.
    ls = compute_size_spectrum(tmp_path, window_years=2)
    assert ls.values
```

Requires `import numpy as np` and `from pathlib import Path` at the top of the test file (add if absent). This test SKIPS cleanly when the examples config is gitignored/absent (mirrors the parity test). If a genuine run produces NO school with TL>0 (so `meanTL` is all-NaN and `compute_trophic_indicators` degrades), STOP and report — that is a real finding about the meanTL capture, not something to force-pass.

- [ ] **Step 3: Run, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py -q` → all pass.
If the integration test is slow, mark it consistently with how other engine-run tests are marked in this repo.

- [ ] **Step 4: Commit**

```bash
git add data/baltic/baltic_param-output.csv tests/test_engine_community_output.py
git commit -m "feat(engine): enable meanTL in Baltic config + e2e community-output integration test"
```

---

## Task 6: Parity check + final gates

**Files:** Modify `tests/test_engine_community_output.py` (optional parity assertion)

- [ ] **Step 1: Confirm the meanTL weighting/seed against Java (parity arbiter)**

The spec defaults to abundance-weighting and a `TL>0` filter. Confirm against the Java engine's `MeanTrophicLevel` semantics:
- Search the repo for any Java reference meanTL output or parity fixture (`grep -ril "meanTL" data/ tests/`).
- If a Java reference run exists, compare a Python-engine meanTL series against it within the established parity tolerance and adjust the weight (abundance↔biomass) / seed handling in `_collect_mean_tl` to match. Add a guarded parity test mirroring how existing cross-engine parity tests are written (e.g. `OsmoseCalibrationProblem(use_java_engine=True)` or the parity-roadmap harness).
- If NO Java reference is available in this environment, document that in a comment in the test file and keep the abundance-weighted default; the integration test's plausibility bound (TL ∈ [1,6]) is the available guard. Do NOT fabricate a parity number.

- [ ] **Step 2: Lint, format, type-check (CI parity — lint runs BOTH check and format)**

Run; fix; re-run until clean:
- `.venv/bin/ruff check osmose/engine/config.py osmose/engine/simulate.py osmose/engine/output.py tests/test_engine_community_output.py`
- `.venv/bin/ruff format osmose/engine/config.py osmose/engine/simulate.py osmose/engine/output.py tests/test_engine_community_output.py`
- `.venv/bin/pyright --pythonpath .venv/bin/python osmose/engine/config.py osmose/engine/simulate.py osmose/engine/output.py` → 0 NEW errors (report any pre-existing; pandas `pd.DataFrame(... columns=...)` reductions may need the `# type: ignore[arg-type]` already shown, or `cast`).

- [ ] **Step 3: Engine + config + community-diagnostics regression suites**

Run: `.venv/bin/python -m pytest tests/test_engine_community_output.py tests/test_engine_output.py tests/test_engine_distribution_output.py tests/test_engine_config_validation.py tests/test_community_metrics.py tests/test_size_spectrum.py -q` → all pass.
NOTE on config validation: `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs[*]` must stay warning-free — the new `output.meanTL.enabled` key is parsed via `_enabled(cfg, "output.meanTL.enabled")` (a string literal in config.py), so the AST allowlist walker captures it automatically. If it does NOT (a warning appears), add `"output.meanTL.enabled"` to `_SUPPLEMENTARY_ALLOWLIST` in `osmose/engine/config_validation.py` per the CLAUDE.md guidance.

- [ ] **Step 4: Full suite regression check**

Run: `.venv/bin/python -m pytest -q -m "not e2e"` → no NEW failures (the known `test_runner.py`/`test_study_fullmodel.py` xdist parallel-load flakes may appear; confirm they pass in isolation if they fail, as they're pre-existing).

- [ ] **Step 5: Commit**

```bash
git add tests/test_engine_community_output.py
git commit -m "test(engine): parity check + final gates for community outputs"
```

---

## Notes

- **No hot-loop change:** `state.trophic_level` is already the Java-faithful emergent TL; we only aggregate and serialize it. The single most important correctness check is that a real run produces TL>0 (Task 5 integration) and that the value is plausible.
- **Cutoff-age filter:** `_collect_mean_tl` applies `config.output_cutoff_age` for consistency with the biomass output; if the Java parity check (Task 6) shows Java does NOT apply the cutoff to meanTL, drop the filter — it's a 3-line change isolated to the helper.
- **DistribBySize source of truth:** the community files are reshaped from the SAME `biomass_by_size`/`abundance_by_size` data the per-species `biomassBySize_<sp>` files use; both are written (different consumers). The community layout is what `size_spectrum`/`community_metrics` read.
- **YAGNI:** no 2D `meanTLBySize`/`ByAge`, no NetCDF variants, no per-species-file changes.
