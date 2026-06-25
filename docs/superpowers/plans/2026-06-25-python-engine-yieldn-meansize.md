# Python-engine `yieldN` + `meanSize` outputs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Python engine produce two per-species outputs it currently can't (`yieldN` = fishing catch in numbers; `meanSize` = abundance-weighted mean length), in CSV **and** NetCDF, matching OSMOSE Java v4.4.1.

**Architecture:** Pure output-layer additions mirroring the existing `meanTL`/`yield` templates: two `StepOutput` collectors (gated by config flags) → CSV writers + in-memory build helpers → entries in the combined NetCDF writer, which is also wired into the engine's `write_outputs` entry point (it is currently dormant). Read back via the existing `results.py` readers.

**Tech Stack:** Python 3.12, NumPy, pandas, xarray (NetCDF), pytest.

## Global Constraints

- **Output-only / parity-safe** — no change to engine dynamics. The 14/14 EEC `atol=0` + 8/8 BoB Java-parity suites MUST stay green (they assert biomass, not these outputs).
- **Java v4.4.1-faithful** (verified): `yieldN` = Σ fishing deaths in **numbers** per focal species (`getNdead(FISHING)`, NO `×weight`, NO age cutoff). `meanSize` = **abundance-weighted** mean length cm = `Σ(abundance × length) / Σ(abundance)`, applying `config.output_cutoff_age`, in cm.
- **Wide all-species/NaN frame convention** — both outputs produce a wide frame (Time + one column per FOCAL species), NaN-filled where a species has no qualifying value, identical shape across CSV / in-memory / NetCDF (the `_build_meantl_dataframe` convention). The per-step collector *dict* omits empty species; the wide builder NaN-fills them.
- **Gating keys** (canonical 4.4.0, verified in `schema/output.py`): `yieldN` CSV = `output.yield.abundance.enabled`, `yieldN` NetCDF = `output.yield.abundance.netcdf.enabled` (both already in schema); `meanSize` CSV = `output.size.enabled` (already in schema); `meanSize` NetCDF = `output.size.netcdf.enabled` (**new key, add to schema**).
- **Subdt accumulation**: `yieldN` summed across the record window (catch accumulates; `n_dead` resets each step). `meanSize` averaged via `_avg_scalar_dict` (mean-of-ratios — matches the existing `meanTL`; do NOT change to a sum).
- **Run all commands** with `PYTHONPATH=.` from the worktree root using `.venv/bin/python`. Lint: `.venv/bin/ruff check` + `.venv/bin/ruff format --check` on `osmose/ tests/`.
- Spec: `docs/superpowers/specs/2026-06-25-python-engine-yieldn-meansize-design.md`.

---

### Task 1: Config flags + schema key

**Files:**
- Modify: `osmose/schema/output.py` (add one NetCDF key)
- Modify: `osmose/engine/config.py` (4 flags: parse block ~917, fields ~1453, `from_dict` ~2193)
- Test: `tests/test_engine_yieldn_meansize.py` (new file)

**Interfaces:**
- Produces: `EngineConfig.output_yield_abundance`, `.output_mean_size`, `.output_yield_abundance_netcdf`, `.output_mean_size_netcdf` (all `bool`, default `False`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_yieldn_meansize.py
import numpy as np
import pytest

from osmose.engine.config import EngineConfig


def _base_cfg() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "1",
    }


def test_config_parses_yieldn_meansize_flags():
    cfg = EngineConfig.from_dict({
        **_base_cfg(),
        "output.yield.abundance.enabled": "true",
        "output.size.enabled": "true",
        "output.yield.abundance.netcdf.enabled": "true",
        "output.size.netcdf.enabled": "true",
    })
    assert cfg.output_yield_abundance is True
    assert cfg.output_mean_size is True
    assert cfg.output_yield_abundance_netcdf is True
    assert cfg.output_mean_size_netcdf is True


def test_config_yieldn_meansize_flags_default_false():
    cfg = EngineConfig.from_dict(_base_cfg())
    assert cfg.output_yield_abundance is False
    assert cfg.output_mean_size is False
    assert cfg.output_yield_abundance_netcdf is False
    assert cfg.output_mean_size_netcdf is False
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k config -v`
Expected: FAIL with `AttributeError: 'EngineConfig' object has no attribute 'output_yield_abundance'`.

- [ ] **Step 3: Add the schema key** — in `osmose/schema/output.py`, in the NetCDF key list (after `"output.yield.abundance.netcdf.enabled",`), add:

```python
    "output.size.netcdf.enabled",
```

- [ ] **Step 4: Add the 4 config flags** — in `osmose/engine/config.py`:

(a) parse block (after `"output_yield_biomass_netcdf": _enabled(cfg, "output.yield.biomass.netcdf.enabled"),`):
```python
        "output_yield_abundance": _enabled(cfg, "output.yield.abundance.enabled"),
        "output_mean_size": _enabled(cfg, "output.size.enabled"),
        "output_yield_abundance_netcdf": _enabled(cfg, "output.yield.abundance.netcdf.enabled"),
        "output_mean_size_netcdf": _enabled(cfg, "output.size.netcdf.enabled"),
```

(b) dataclass fields (after `output_meantl: bool = False`):
```python
    output_yield_abundance: bool = False
    output_mean_size: bool = False
    output_yield_abundance_netcdf: bool = False
    output_mean_size_netcdf: bool = False
```

(c) `from_dict` wiring (after `output_meantl=_output["output_meantl"],`):
```python
            output_yield_abundance=_output["output_yield_abundance"],
            output_mean_size=_output["output_mean_size"],
            output_yield_abundance_netcdf=_output["output_yield_abundance_netcdf"],
            output_mean_size_netcdf=_output["output_mean_size_netcdf"],
```

- [ ] **Step 5: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k config -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add osmose/schema/output.py osmose/engine/config.py tests/test_engine_yieldn_meansize.py
git commit -m "feat(output): config flags + schema key for yieldN/meanSize"
```

---

### Task 2: Collectors + StepOutput fields

**Files:**
- Modify: `osmose/engine/simulate.py` (`StepOutput` ~92-100; collectors near `_collect_yield`/`_collect_mean_tl`; build at ~1034 + ~1056; accumulation at ~1178 + ~1230)
- Test: `tests/test_engine_yieldn_meansize.py`

**Interfaces:**
- Consumes: `EngineConfig` flags from Task 1.
- Produces: `StepOutput.yield_n: NDArray|None`, `StepOutput.mean_size: dict[int,float]|None`; `_collect_yield_n(state, config) -> NDArray[float64]` (len n_species), `_collect_mean_size(state, config) -> dict[int,float]`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_engine_yieldn_meansize.py
from osmose.engine.simulate import _collect_mean_size, _collect_yield_n
from osmose.engine.state import MortalityCause, SchoolState


def _two_school_state() -> SchoolState:
    # 2 focal schools of species 0: lengths 10 & 30, abundance 100 & 300,
    # fishing deaths 4 & 6.
    s = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
    s = s.replace(
        length=np.array([10.0, 30.0]),
        abundance=np.array([100.0, 300.0]),
        weight=np.array([0.01, 0.27]),
        biomass=np.array([1.0, 81.0]),
        age_dt=np.array([12, 24], dtype=np.int32),
    )
    nd = s.n_dead.copy()
    nd[:, int(MortalityCause.FISHING)] = np.array([4.0, 6.0])
    return s.replace(n_dead=nd)


class _Cfg:
    n_species = 1
    n_dt_per_year = 12
    output_cutoff_age = None


def test_collect_yield_n_is_fishing_deaths_in_numbers():
    yn = _collect_yield_n(_two_school_state(), _Cfg())
    assert yn.shape == (1,)
    assert yn[0] == pytest.approx(10.0)  # 4 + 6 deaths, NO weight


def test_collect_mean_size_abundance_weighted():
    ms = _collect_mean_size(_two_school_state(), _Cfg())
    # (100*10 + 300*30) / (100+300) = 10000/400 = 25.0
    assert ms[0] == pytest.approx(25.0)


def test_collect_mean_size_applies_cutoff_and_omits_empty():
    class CutCfg(_Cfg):
        output_cutoff_age = np.array([1.5])  # 1.5 yr = 18 dt at ndt=12; excludes the age_dt=12 school
    ms = _collect_mean_size(_two_school_state(), CutCfg())
    # only the age_dt=24 (2 yr) school survives → mean length = 30
    assert ms[0] == pytest.approx(30.0)
    # a state with no qualifying school → species omitted
    empty = _collect_mean_size(SchoolState.create(0), _Cfg())
    assert empty == {}
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k "yield_n or mean_size" -v`
Expected: FAIL with `ImportError: cannot import name '_collect_yield_n'`.

- [ ] **Step 3: Add the collectors** — in `osmose/engine/simulate.py`, immediately after `_collect_yield` (the `def _collect_yield(...)` ending `return yield_by_species`):

```python
def _collect_yield_n(state: SchoolState, config: EngineConfig) -> NDArray[np.float64]:
    """Fishing catch in NUMBERS per focal species (Java yieldN = Σ getNdead(FISHING)).

    Identical to _collect_yield but WITHOUT the × weight, and (like yield) no age cutoff.
    """
    yield_n = np.zeros(config.n_species, dtype=np.float64)
    if len(state) > 0:
        fishing_dead = state.n_dead[:, int(MortalityCause.FISHING)]
        focal_mask = state.species_id < config.n_species
        np.add.at(yield_n, state.species_id[focal_mask], fishing_dead[focal_mask])
    return yield_n


def _collect_mean_size(state: SchoolState, config: EngineConfig) -> dict[int, float]:
    """Abundance-weighted mean length (cm) per FOCAL species — Java MeanSizeOutput:
    ``Σ(abundance × length) / Σ(abundance)``, applying the same output cutoff-age filter as
    meanTL/biomass. Species with no qualifying abundance are omitted (the wide output frame
    NaN-fills them).
    """
    n_sp = config.n_species
    wsum = np.zeros(n_sp, dtype=np.float64)  # Σ abundance*length
    asum = np.zeros(n_sp, dtype=np.float64)  # Σ abundance
    if len(state) > 0:
        sp = state.species_id
        length = state.length
        abd = state.abundance
        mask = (sp < n_sp) & (abd > 0) & (length > 0)
        if config.output_cutoff_age is not None:
            age_years = state.age_dt.astype(np.float64) / config.n_dt_per_year
            cutoff = config.output_cutoff_age[sp]
            mask = mask & (age_years >= cutoff)
        np.add.at(wsum, sp[mask], abd[mask] * length[mask])
        np.add.at(asum, sp[mask], abd[mask])
    return {i: float(wsum[i] / asum[i]) for i in range(n_sp) if asum[i] > 0}
```

- [ ] **Step 4: Add the `StepOutput` fields** — in the `StepOutput` dataclass, after `yield_by_species: NDArray[np.float64] | None = None ...`:

```python
    yield_n: NDArray[np.float64] | None = None  # fishing catch in numbers per species
    mean_size: dict[int, float] | None = None  # abundance-weighted mean length (cm)
```

- [ ] **Step 5: Wire into the per-step build** — in the `StepOutput`-building function (where `mean_tl = _collect_mean_tl(...)` is), after that line add:

```python
    yield_n = (
        _collect_yield_n(state, config)
        if (config.output_yield_abundance or config.output_yield_abundance_netcdf)
        else None
    )
    mean_size = (
        _collect_mean_size(state, config)
        if (config.output_mean_size or config.output_mean_size_netcdf)
        else None
    )
```
and add `yield_n=yield_n, mean_size=mean_size,` to that `return StepOutput(...)` call (next to `mean_tl=mean_tl,`).

- [ ] **Step 6: Wire into the subdt accumulation** — in the accumulation function:

(a) the `len(accumulated) == 1` branch's `StepOutput(...)` — add `yield_n=accumulated[0].yield_n, mean_size=accumulated[0].mean_size,` (next to `mean_tl=accumulated[0].mean_tl,`).

(b) the multi-step branch — after `yield_sum = np.sum(...)` add:
```python
    _yn = [o.yield_n for o in accumulated if o.yield_n is not None]
    yield_n_sum = np.sum(_yn, axis=0) if _yn else None
```
and in that branch's `return StepOutput(...)`, add `yield_n=yield_n_sum, mean_size=_avg_scalar_dict("mean_size"),` (next to `mean_tl=_avg_scalar_dict("mean_tl"),`).

- [ ] **Step 7: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k "yield_n or mean_size" -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/simulate.py tests/test_engine_yieldn_meansize.py
git commit -m "feat(output): yieldN/meanSize collectors + StepOutput fields + accumulation"
```

---

### Task 3: CSV writers + in-memory results wiring

**Files:**
- Modify: `osmose/engine/output.py` (build helpers + CSV writers near `_build_yield_dataframes`/`_write_meantl_csv`; register in `write_outputs` ~62)
- Modify: `osmose/results.py` (`_CROSS_SPECIES_OUTPUT_TYPES` ~229; `_build_dataframes_from_outputs` ~263-285)
- Test: `tests/test_engine_yieldn_meansize.py`

**Interfaces:**
- Consumes: `StepOutput.yield_n`/`.mean_size` (Task 2).
- Produces: `_build_yieldn_dataframes(outputs, config) -> {"yieldN": df}`, `_build_meansize_dataframe(outputs, config) -> {"meanSize": df}` (wide: `["Time"] + config.species_names`, NaN-filled); disk files `{prefix}_yieldN_Simu0.csv` / `{prefix}_meanSize_Simu0.csv`; in-memory cache keys `"yieldN"`/`"meanSize"`.

- [ ] **Step 1: Write the failing test** (end-to-end CSV via a tiny run helper)

```python
# add to tests/test_engine_yieldn_meansize.py
def _run_tiny(tmp_path, extra):
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate
    raw = {**_base_cfg(), "simulation.time.nyear": "1", **extra,
           "mortality.fishing.rate.method.sp0": "constant",
           "mortality.fishing.rate.sp0": "0.2"}
    cfg = EngineConfig.from_dict(raw)
    grid = Grid.from_dimensions(ny=2, nx=2)
    outputs = simulate(cfg, grid, np.random.default_rng(0))
    return cfg, outputs


def test_yieldn_meansize_csv_written_and_readable(tmp_path):
    from osmose.engine.output import write_outputs
    from osmose.results import OsmoseResults
    cfg, outputs = _run_tiny(tmp_path, {"output.yield.abundance.enabled": "true",
                                        "output.size.enabled": "true"})
    write_outputs(outputs, tmp_path, prefix="run", config=cfg, grid=None)
    assert (tmp_path / "run_yieldN_Simu0.csv").exists()
    assert (tmp_path / "run_meanSize_Simu0.csv").exists()
    res = OsmoseResults(output_dir=tmp_path, prefix="run")
    assert not res.yield_abundance().empty
    assert not res.mean_size().empty


def test_yieldn_meansize_csv_matches_in_memory(tmp_path):
    from osmose.engine.output import write_outputs
    from osmose.results import OsmoseResults, _build_dataframes_from_outputs
    from osmose.engine.grid import Grid
    cfg, outputs = _run_tiny(tmp_path, {"output.yield.abundance.enabled": "true",
                                        "output.size.enabled": "true"})
    write_outputs(outputs, tmp_path, prefix="run", config=cfg, grid=None)
    disk = OsmoseResults(output_dir=tmp_path, prefix="run").yield_abundance()
    mem_cache = _build_dataframes_from_outputs(outputs, cfg, Grid.from_dimensions(ny=2, nx=2))
    assert "yieldN" in mem_cache and "meanSize" in mem_cache
    # disk wide frame's species columns sum == in-memory yieldN column sum
    disk_total = disk.drop(columns=[c for c in ("Time", "species") if c in disk]).to_numpy().sum()
    mem_total = mem_cache["yieldN"].drop(columns=[c for c in ("Time", "species") if c in mem_cache["yieldN"]]).to_numpy()
    assert np.isclose(disk_total, np.nan_to_num(mem_total).sum())
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k "csv" -v`
Expected: FAIL (files not written / `"yieldN"` not in cache).

- [ ] **Step 3: Add build helpers + writers** — in `osmose/engine/output.py`, after `_build_yield_dataframes`:

```python
def _build_yieldn_dataframes(
    outputs: list[StepOutput], config: EngineConfig
) -> dict[str, pd.DataFrame]:
    """Wide Time + per-focal-species fishing catch in numbers. Empty when disabled/absent."""
    if not config.output_yield_abundance or not any(o.yield_n is not None for o in outputs):
        return {}
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    data = np.array(
        [o.yield_n if o.yield_n is not None else np.zeros(config.n_species) for o in outputs]
    )
    df = pd.DataFrame(data, columns=list(config.species_names))  # type: ignore[arg-type]
    df.insert(0, "Time", times)
    return {"yieldN": df}


def _build_meansize_dataframe(
    outputs: list[StepOutput], config: EngineConfig
) -> dict[str, pd.DataFrame]:
    """Wide Time + per-focal-species abundance-weighted mean length (cm), NaN where empty."""
    if not config.output_mean_size or not any(o.mean_size is not None for o in outputs):
        return {}
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    sp_names = config.species_names
    data = np.full((len(outputs), len(sp_names)), np.nan, dtype=np.float64)
    for t_idx, o in enumerate(outputs):
        if o.mean_size:
            for sp_idx, val in o.mean_size.items():
                if sp_idx < len(sp_names):
                    data[t_idx, sp_idx] = val
    df = pd.DataFrame(data, columns=sp_names)  # type: ignore[arg-type]
    df.insert(0, "Time", times)
    return {"meanSize": df}


def _write_yieldn_csv(
    output_dir: Path, prefix: str, outputs: list[StepOutput], config: EngineConfig
) -> None:
    for key, df in _build_yieldn_dataframes(outputs, config).items():
        df.to_csv(output_dir / f"{prefix}_{key}_Simu0.csv", index=False)


def _write_meansize_csv(
    output_dir: Path, prefix: str, outputs: list[StepOutput], config: EngineConfig
) -> None:
    for key, df in _build_meansize_dataframe(outputs, config).items():
        df.to_csv(output_dir / f"{prefix}_{key}_Simu0.csv", index=False)
```

- [ ] **Step 4: Register in `write_outputs`** — after `_write_meantl_csv(output_dir, prefix, outputs, config)`:

```python
    _write_yieldn_csv(output_dir, prefix, outputs, config)
    _write_meansize_csv(output_dir, prefix, outputs, config)
```

- [ ] **Step 5: Wire the in-memory cache** — in `osmose/results.py`:

(a) add to `_CROSS_SPECIES_OUTPUT_TYPES`:
```python
    "yieldN",
    "meanSize",
```
(b) in `_build_dataframes_from_outputs`, add the two helpers to the lazy import and the `disk_shape.update(...)` block:
```python
        _build_meansize_dataframe,
        _build_yieldn_dataframes,
```
```python
    disk_shape.update(_build_yieldn_dataframes(outputs, config))
    disk_shape.update(_build_meansize_dataframe(outputs, config))
```

- [ ] **Step 6: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k "csv" -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/output.py osmose/results.py tests/test_engine_yieldn_meansize.py
git commit -m "feat(output): yieldN/meanSize CSV writers + in-memory results wiring"
```

---

### Task 4: NetCDF — writer entries, run-path wiring, reader

**Files:**
- Modify: `osmose/engine/output.py` (`write_outputs_netcdf` `want` ~665 + `data_vars`; wire into `write_outputs` ~85)
- Modify: `osmose/results.py` (`yield_abundance`/`mean_size` `source=` + a `_read_netcdf_species_var` helper)
- Test: `tests/test_engine_yieldn_meansize.py`

**Interfaces:**
- Consumes: `StepOutput.yield_n`/`.mean_size`; the new `.netcdf` flags.
- Produces: a combined `{prefix}_Simu0.nc` with `yieldN`/`meanSize` variables (when their netcdf flag is on); `results.yield_abundance(source="netcdf")` / `mean_size(source="netcdf")` returning a wide frame (`Time` + species columns).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_engine_yieldn_meansize.py
def test_netcdf_written_only_when_flag_on(tmp_path):
    from osmose.engine.output import write_outputs
    # default config: no .nc
    cfg0, out0 = _run_tiny(tmp_path / "a", {})
    (tmp_path / "a").mkdir(parents=True, exist_ok=True)
    write_outputs(out0, tmp_path / "a", prefix="run", config=cfg0, grid=None)
    assert not (tmp_path / "a" / "run_Simu0.nc").exists()
    # netcdf-enabled: .nc with yieldN + meanSize vars
    cfgN, outN = _run_tiny(tmp_path / "b", {"output.yield.abundance.netcdf.enabled": "true",
                                            "output.size.netcdf.enabled": "true"})
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    write_outputs(outN, tmp_path / "b", prefix="run", config=cfgN, grid=None)
    import xarray as xr
    ds = xr.open_dataset(tmp_path / "b" / "run_Simu0.nc")
    assert "yieldN" in ds and "meanSize" in ds


def test_csv_equals_netcdf(tmp_path):
    from osmose.engine.output import write_outputs
    from osmose.results import OsmoseResults
    cfg, outputs = _run_tiny(tmp_path, {
        "output.yield.abundance.enabled": "true", "output.size.enabled": "true",
        "output.yield.abundance.netcdf.enabled": "true", "output.size.netcdf.enabled": "true",
    })
    write_outputs(outputs, tmp_path, prefix="run", config=cfg, grid=None)
    res = OsmoseResults(output_dir=tmp_path, prefix="run")
    for getter in ("yield_abundance", "mean_size"):
        csv_df = getattr(res, getter)()
        nc_df = getattr(res, getter)(source="netcdf")
        cols = [c for c in csv_df.columns if c not in ("Time", "species")]
        np.testing.assert_allclose(
            np.nan_to_num(csv_df[cols].to_numpy()),
            np.nan_to_num(nc_df[cols].to_numpy()), rtol=1e-6,
        )
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k "netcdf or csv_equals" -v`
Expected: FAIL (no `.nc` written; `source` kwarg unknown).

- [ ] **Step 3: Add NetCDF writer entries** — in `write_outputs_netcdf`, in the `want` dict (after `"mortality_by_cause": config.output_mortality_netcdf,`):

```python
        "yieldN": config.output_yield_abundance_netcdf
        and any(o.yield_n is not None for o in outputs),
        "meanSize": config.output_mean_size_netcdf
        and any(o.mean_size is not None for o in outputs),
```
and in the data_vars section (after the `if want["yield"]:` block):
```python
    if want["yieldN"]:
        yn_arr = np.array(
            [
                o.yield_n if o.yield_n is not None else np.full(config.n_species, np.nan)
                for o in outputs
            ]
        )
        data_vars["yieldN"] = (["time", "focal_species"], yn_arr)
        coords["focal_species"] = config.species_names[: yn_arr.shape[1]]
    if want["meanSize"]:
        ms_arr = np.full((len(outputs), n_species), np.nan)
        for t_idx, o in enumerate(outputs):
            if o.mean_size:
                for sp_idx, val in o.mean_size.items():
                    if sp_idx < n_species:
                        ms_arr[t_idx, sp_idx] = val
        data_vars["meanSize"] = (["time", "species"], ms_arr)
```

- [ ] **Step 4: Wire `write_outputs_netcdf` into `write_outputs`** — at the END of `write_outputs` (after the spatial-NetCDF block), add:

```python
    # Combined per-species NetCDF. Inert unless a `.netcdf.enabled` flag is set
    # (write_outputs_netcdf early-returns when no want is true), so default/parity
    # configs write nothing.
    write_outputs_netcdf(outputs, output_dir / f"{prefix}_Simu0.nc", config)
```

- [ ] **Step 5: Add the `source="netcdf"` accessors** — in `osmose/results.py`, replace the `mean_size` and `yield_abundance` methods with `source`-aware versions and add the helper:

```python
    def mean_size(self, species: str | None = None, source: str = "csv") -> pd.DataFrame:
        """Read mean size time series. source='csv' (default) or 'netcdf'."""
        if source == "netcdf":
            return self._read_netcdf_species_var("meanSize", "species", species)
        return self._read_species_output("meanSize", species)

    def yield_abundance(self, species: str | None = None, source: str = "csv") -> pd.DataFrame:
        """Read yield in abundance (numbers caught). source='csv' (default) or 'netcdf'."""
        if source == "netcdf":
            return self._read_netcdf_species_var("yieldN", "focal_species", species)
        return self._read_species_output("yieldN", species)

    def _read_netcdf_species_var(
        self, var: str, species_dim: str, species: str | None
    ) -> pd.DataFrame:
        """Read one (time, species) variable from the combined {prefix}_Simu0.nc into a
        wide frame: ['Time'] + species columns — the same shape as the CSV reader."""
        ds = self.read_netcdf(f"{self.prefix}_Simu0.nc")
        da = ds[var]  # dims (time, <species_dim>)
        sp_names = [str(s) for s in ds.coords[species_dim].values]
        wide = pd.DataFrame(np.asarray(da.values), columns=sp_names)
        wide.insert(0, "Time", np.asarray(ds.coords["time"].values))
        if species is not None and species in wide.columns:
            wide = wide[["Time", species]]
        return wide
```

- [ ] **Step 6: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k "netcdf or csv_equals" -v`
Expected: PASS (2 passed). If `csv_equals` fails on a column-order/naming mismatch, align the netcdf accessor's species columns to the CSV's (both use `config.species_names`).

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/output.py osmose/results.py tests/test_engine_yieldn_meansize.py
git commit -m "feat(output): yieldN/meanSize NetCDF writer + run-path wiring + source=netcdf reader"
```

---

### Task 5: Capability note + regression sweep

**Files:**
- Modify: `osmose/engine_capabilities.py` (`_PYTHON_NOTABLE`)
- Test: `tests/test_engine_yieldn_meansize.py` + run the existing parity/output suites

**Interfaces:** none new.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_engine_yieldn_meansize.py
def test_capability_note_no_longer_lists_yieldn_meansize():
    from osmose.engine_capabilities import _PYTHON_NOTABLE
    assert "yieldN" not in _PYTHON_NOTABLE
    assert "meanSize" not in _PYTHON_NOTABLE
    # still lists the genuinely-unproduced ones
    assert "sizeSpectrum" in _PYTHON_NOTABLE
    assert "fishery-yield" in _PYTHON_NOTABLE
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k capability -v`
Expected: FAIL (`yieldN`/`meanSize` still in the string).

- [ ] **Step 3: Update the note** — in `osmose/engine_capabilities.py`, change `_PYTHON_NOTABLE` to:

```python
_PYTHON_NOTABLE = (
    "Not produced on the Python engine: sizeSpectrum, meanTLByAge, "
    "fishery-yield (run these on the Java engine)."
)
```

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py -k capability -v`
Expected: PASS.

- [ ] **Step 5: Full regression sweep** — confirm no dynamics/output regression:

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_yieldn_meansize.py tests/test_engine_parity.py tests/test_engine_output.py tests/test_baltic_java_compat.py tests/test_engine_community_output.py -q`
Expected: all pass (the new tests + EEC `atol=0` parity + output + BoB + community output suites). Then lint:
Run: `.venv/bin/ruff check osmose/ tests/test_engine_yieldn_meansize.py && .venv/bin/ruff format --check osmose/engine/output.py osmose/engine/simulate.py osmose/engine/config.py osmose/results.py osmose/schema/output.py osmose/engine_capabilities.py tests/test_engine_yieldn_meansize.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add osmose/engine_capabilities.py tests/test_engine_yieldn_meansize.py
git commit -m "feat(output): mark yieldN/meanSize produced; regression sweep green"
```

---

## Notes for the executor

- **Do NOT touch engine dynamics** — these are output collectors only. If any EEC/BoB parity test changes, something is wrong; stop and investigate (do not re-baseline).
- The `_run_tiny` helper writes a 1-species fished config; if `simulate`/`write_outputs` need an arg the helper omits (e.g. `bkg_output`), pass the minimal value the existing output tests use (check `tests/test_engine_output.py` for the call shape) — keep the helper faithful to a real run.
- `meanSize` is **abundance**-weighted (not biomass like meanTL); `yieldN` is deaths in **numbers** (not ×weight like yield). These two one-word differences from the templates are the whole point — get them right.
- Keep the wide all-species/NaN convention across CSV/in-memory/NetCDF so the `csv_equals` test holds.
