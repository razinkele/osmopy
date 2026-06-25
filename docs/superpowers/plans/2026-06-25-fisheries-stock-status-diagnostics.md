# Fisheries stock-status diagnostics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an indicative fisheries stock-status page (Kobe / B/Bmsy / F/Fmsy + the existing F/M bars) to the Shiny app, backed by a new model-SSB engine output and a reference-point resolver.

**Architecture:** A parity-safe engine SSB output (mirrors the merged yieldN/meanSize pattern) feeds a pure `stock_status` computation, which is fed user-supplied `Bmsy` + ICES-auto-filled `Fmsy` from a reference resolver and plotted on a new `Fisheries` page. Everything downstream of the SSB output is read-only analysis.

**Tech Stack:** Python 3.12, NumPy, pandas, xarray, Plotly, Shiny for Python, pytest.

## Global Constraints

- **Parity-safe:** the engine SSB output is gated and additive; the 14/14 EEC `atol=0` + 8/8 BoB suites MUST stay green (they assert biomass, not SSB).
- **Indicative, not authoritative:** soft Kobe shading + a "relative to supplied reference points — not a formal assessment" disclaimer. No fabricated references (no SSB-derived Bmsy auto-fill).
- **SSB = the engine's own conjunction** `length ≥ maturity_size AND age_dt ≥ maturity_age_dt AND abundance > 0` (from `reproduction.py`). Never reconstruct from marginal by-age/by-size.
- **Annual aggregation cadence:** saved series are written every `output.recordfrequency.ndt` steps (Baltic = once/year, pre-summed). Aggregate to annual via `fis.annual_by_year` (groupby `int(Time)` = absolute simulation year) — SSB `how="mean"`, F `how="sum"` — which is correct for ANY record frequency and labels both axes by absolute year. Never positional `np.arange`/`reshape`.
- **B-axis = user `Bmsy` only** (`b_ref_kind ∈ {"bmsy_user","none"}`); **ICES auto-fills `Fmsy` only**, from the deterministic primary tonnes-unit stock (largest `msy_btrigger`, tie → latest `advice_year`).
- **Exploited stage** for model F = the fished stage (`F > _FISHED_TOL`, excluding `Eggs`) with the **largest annual F**; caveat when >1 fished stage.
- **Module paths:** `osmose.validation.{fisheries, stock_status, fisheries_reference}`, `osmose.plotting`, `ui.pages.fisheries`. Reference sidecar: `data/<ecosystem>/reference/fisheries_reference_points.json`.
- **Run all commands** with `PYTHONPATH=.` from the worktree root using `.venv/bin/python`. Lint: `.venv/bin/ruff check` + `ruff format --check`.
- Spec: `docs/superpowers/specs/2026-06-25-fisheries-stock-status-diagnostics-design.md`.

---

### Task 1: Engine SSB output — config flags + collector + StepOutput + accumulation

**Files:**
- Modify: `osmose/engine/config.py` (parse block ~920, fields ~1446, `from_dict` ~2194)
- Modify: `osmose/engine/simulate.py` (`StepOutput` ~103, build ~1113, accumulation ~1237 + ~1305)
- Test: `tests/test_engine_ssb_output.py` (new)

**Interfaces:**
- Produces: `EngineConfig.output_ssb: bool`, `.output_ssb_netcdf: bool`; `StepOutput.ssb: NDArray|None`; `_collect_ssb(state, config) -> NDArray[float64]` (len `n_species`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_ssb_output.py
import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.simulate import _collect_ssb
from osmose.engine.state import SchoolState


def _base_cfg() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "12", "simulation.time.nyear": "1",
        "simulation.nspecies": "1", "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish", "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3", "species.t0.sp0": "-0.1", "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0", "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0", "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5", "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random", "movement.randomwalk.range.sp0": "1",
        "species.maturity.size.sp0": "12.0",
    }


def test_config_parses_ssb_flags():
    cfg = EngineConfig.from_dict({**_base_cfg(),
        "output.ssb.enabled": "true", "output.ssb.netcdf.enabled": "true"})
    assert cfg.output_ssb is True
    assert cfg.output_ssb_netcdf is True
    assert EngineConfig.from_dict(_base_cfg()).output_ssb is False


def test_collect_ssb_uses_maturity_conjunction():
    # 3 schools of sp0, maturity_size=12, maturity_age_dt=0 (size-only):
    # lengths 8/15/20 (school0 immature by size), abundance 100 each, weight 0.01/0.05/0.1
    s = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
    s = s.replace(length=np.array([8.0, 15.0, 20.0]), abundance=np.array([100.0, 100.0, 100.0]),
                  weight=np.array([0.01, 0.05, 0.1]), age_dt=np.array([6, 12, 24], dtype=np.int32))

    class Cfg:
        n_species = 1
        maturity_size = np.array([12.0])
        maturity_age_dt = np.array([0], dtype=np.int32)
    ssb = _collect_ssb(s, Cfg())
    # mature = length>=12 AND age_dt>=0 AND abundance>0 → schools 1,2: 100*0.05 + 100*0.1 = 15.0
    assert ssb[0] == pytest.approx(15.0)
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_ssb_output.py -k "config or collect" -v`
Expected: FAIL (`AttributeError: output_ssb` / `ImportError: _collect_ssb`).

- [ ] **Step 3: Add the config flags** — in `osmose/engine/config.py`:

(a) parse block (after `"output_mean_size_netcdf": ...`):
```python
        "output_ssb": _enabled(cfg, "output.ssb.enabled"),
        "output_ssb_netcdf": _enabled(cfg, "output.ssb.netcdf.enabled"),
```
(b) dataclass fields (after `output_mean_size_netcdf: bool = False`):
```python
    output_ssb: bool = False
    output_ssb_netcdf: bool = False
```
(c) `from_dict` wiring (after `output_mean_size_netcdf=_output["output_mean_size_netcdf"],`):
```python
            output_ssb=_output["output_ssb"],
            output_ssb_netcdf=_output["output_ssb_netcdf"],
```
Also add `"output.ssb.netcdf.enabled"` to `osmose/schema/output.py`'s NetCDF key list (next to `output.size.netcdf.enabled`). (`output.ssb.enabled` is already allowlisted.)

- [ ] **Step 4: Add the collector** — in `osmose/engine/simulate.py`, after `_collect_yield_n`/`_collect_mean_size` (near line 825):

```python
def _collect_ssb(state: SchoolState, config: EngineConfig) -> NDArray[np.float64]:
    """Spawning-stock biomass per focal species — the engine's own maturity conjunction
    (length >= maturity_size AND age_dt >= maturity_age_dt AND abundance > 0), matching
    reproduction.py. SSB = Σ abundance*weight over mature schools (tonnes)."""
    ssb = np.zeros(config.n_species, dtype=np.float64)
    if len(state) > 0:
        mature = (
            (state.length >= config.maturity_size[state.species_id])
            & (state.age_dt >= config.maturity_age_dt[state.species_id])
            & (state.abundance > 0)
            & (state.species_id < config.n_species)
        )
        np.add.at(ssb, state.species_id[mature], state.abundance[mature] * state.weight[mature])
    return ssb
```

- [ ] **Step 5: Add the StepOutput field + build + accumulation** — in `simulate.py`:

(a) `StepOutput` (after `mean_size: dict[int, float] | None = None`):
```python
    ssb: NDArray[np.float64] | None = None  # spawning-stock biomass per species
```
(b) in the per-step build (where `mean_size = _collect_mean_size(...)` is, ~line 1086), add:
```python
    ssb = _collect_ssb(state, config) if (config.output_ssb or config.output_ssb_netcdf) else None
```
and add `ssb=ssb,` to that `return StepOutput(...)` call (next to `mean_size=mean_size,`).

(c) single-step accumulation branch (next to `mean_size=accumulated[0].mean_size,`): add
`ssb=accumulated[0].ssb,`.

(d) multi-step branch — after `_yn = [...]; yield_n_sum = ...` add:
```python
    _ssb = [o.ssb for o in accumulated if o.ssb is not None]
    ssb_avg = np.mean(_ssb, axis=0) if _ssb else None
```
and add `ssb=ssb_avg,` to that `return StepOutput(...)` (next to `mean_size=_avg_scalar_dict("mean_size"),`).
(SSB is **mean**-aggregated across the record window like biomass, not summed.)

- [ ] **Step 6: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_ssb_output.py -k "config or collect" -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add osmose/engine/config.py osmose/engine/simulate.py osmose/schema/output.py tests/test_engine_ssb_output.py
git commit -m "feat(engine): SSB collector + config flags + StepOutput (parity-safe output)"
```

---

### Task 2: SSB output writers + reader

**Files:**
- Modify: `osmose/engine/output.py` (build/writer near `_build_meansize_dataframe` ~256; `write_outputs` ~64; `write_outputs_netcdf` want/data_vars)
- Modify: `osmose/results.py` (`_CROSS_SPECIES_OUTPUT_TYPES`; `_build_dataframes_from_outputs`; new `ssb()` reader)
- Test: `tests/test_engine_ssb_output.py`

**Interfaces:**
- Consumes: `StepOutput.ssb` (Task 1).
- Produces: `{prefix}_SSB_Simu0.csv`; in-memory cache key `"SSB"`; `results.ssb(species=None) -> pd.DataFrame` (wide: `Time` + focal-species columns).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_engine_ssb_output.py
def _step(step, ssb=None, n_sp=1):
    from osmose.engine.simulate import StepOutput
    from osmose.engine.state import MortalityCause
    return StepOutput(step=step, biomass=np.full(n_sp, 100.0), abundance=np.full(n_sp, 1000.0),
                      mortality_by_cause=np.zeros((n_sp, len(MortalityCause)), dtype=np.float64), ssb=ssb)


def test_ssb_csv_and_reader_roundtrip(tmp_path):
    from osmose.engine.output import write_outputs
    from osmose.results import OsmoseResults, _build_dataframes_from_outputs
    from osmose.engine.grid import Grid
    cfg = EngineConfig.from_dict({**_base_cfg(), "output.ssb.enabled": "true"})
    sp = cfg.species_names[0]
    outputs = [_step(0, np.array([40.0])), _step(1, np.array([60.0]))]
    write_outputs(outputs, tmp_path, cfg, prefix="run")
    assert (tmp_path / "run_SSB_Simu0.csv").exists()
    res = OsmoseResults(tmp_path, prefix="run")
    assert res.ssb()[sp].tolist() == [40.0, 60.0]
    mem = _build_dataframes_from_outputs(outputs, cfg, Grid.from_dimensions(ny=1, nx=1))
    assert "SSB" in mem and mem["SSB"][sp].tolist() == [40.0, 60.0]
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_ssb_output.py -k roundtrip -v`
Expected: FAIL (no SSB file / `ssb` reader missing).

- [ ] **Step 3: Add the build helper + CSV writer** — in `osmose/engine/output.py`, after `_build_meansize_dataframe`:

```python
def _build_ssb_dataframe(
    outputs: list[StepOutput], config: EngineConfig
) -> dict[str, pd.DataFrame]:
    """Wide Time + per-focal-species spawning-stock biomass. Empty when disabled/absent."""
    if not config.output_ssb or not any(o.ssb is not None for o in outputs):
        return {}
    times = np.array([o.step / config.n_dt_per_year for o in outputs])
    data = np.array([o.ssb if o.ssb is not None else np.zeros(config.n_species) for o in outputs])
    df = pd.DataFrame(data, columns=list(config.species_names))  # type: ignore[arg-type]
    df.insert(0, "Time", times)
    return {"SSB": df}


def _write_ssb_csv(
    output_dir: Path, prefix: str, outputs: list[StepOutput], config: EngineConfig
) -> None:
    for key, df in _build_ssb_dataframe(outputs, config).items():
        df.to_csv(output_dir / f"{prefix}_{key}_Simu0.csv", index=False)
```

- [ ] **Step 4: Register in `write_outputs` + NetCDF** — in `write_outputs` (after `_write_meansize_csv(...)`):
```python
    _write_ssb_csv(output_dir, prefix, outputs, config)
```
In `write_outputs_netcdf`, add to the `want` dict:
```python
        "SSB": config.output_ssb_netcdf and any(o.ssb is not None for o in outputs),
```
and in data_vars (after the `meanSize` block):
```python
    if want["SSB"]:
        ssb_arr = np.array(
            [o.ssb if o.ssb is not None else np.full(config.n_species, np.nan) for o in outputs]
        )
        data_vars["SSB"] = (["time", "focal_species"], ssb_arr)
        coords.setdefault("focal_species", config.species_names[: ssb_arr.shape[1]])
```

- [ ] **Step 5: Wire the in-memory cache + reader** — in `osmose/results.py`:

(a) add `"SSB"` to `_CROSS_SPECIES_OUTPUT_TYPES`.
(b) in `_build_dataframes_from_outputs`: add `_build_ssb_dataframe` to the lazy import and add
`disk_shape.update(_build_ssb_dataframe(outputs, config))`.
(c) add the reader method (next to `mean_size`):
```python
    def ssb(self, species: str | None = None) -> pd.DataFrame:
        """Read spawning-stock biomass time series (wide: Time + per-species columns)."""
        return self._read_species_output("SSB", species)
```

- [ ] **Step 6: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_ssb_output.py -k roundtrip -v`
Expected: PASS.

- [ ] **Step 7: Parity smoke + commit**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_parity.py tests/test_engine_output.py -q`
Expected: all pass (SSB output is gated; default configs unaffected).
```bash
git add osmose/engine/output.py osmose/results.py tests/test_engine_ssb_output.py
git commit -m "feat(engine): SSB CSV/in-memory/NetCDF writers + results.ssb reader"
```

---

### Task 3: `annual_by_year` aggregator (cadence-correct, absolute-year)

**Files:**
- Modify: `osmose/validation/fisheries.py` (add `annual_by_year`; leave `annual_rate` untouched)
- Test: `tests/test_validation_stock_status.py` (new)

**Why this shape:** the saved series are written every `output.recordfrequency.ndt` steps. Grouping by the integer `Time` (= absolute simulation year) aggregates however many saved rows fall in each year — correct for ANY record frequency without needing a `steps_per_year` reshape — and labels every value by **absolute year** so the two axes intersect correctly (never positionally). F accumulates → `how="sum"`; SSB is a stock level → `how="mean"`.

**Interfaces:**
- Produces: `fisheries.annual_by_year(values, time, *, how) -> dict[int, float]` (`values`/`time` array-likes; `how ∈ {"sum","mean"}`; keys are `int(floor(time))`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_stock_status.py
import numpy as np
import pandas as pd
import pytest

from osmose.validation import fisheries as fis
from osmose.validation import stock_status as ss
from osmose.validation.fisheries_reference import ReferencePoint


def test_annual_by_year_sum_and_mean():
    # 2 saved rows in year 0 (Time 0.0, 0.5), 1 in year 1 (Time 1.0)
    time = [0.0, 0.5, 1.0]
    assert fis.annual_by_year([2.0, 3.0, 10.0], time, how="sum") == {0: 5.0, 1: 10.0}
    assert fis.annual_by_year([2.0, 4.0, 10.0], time, how="mean") == {0: 3.0, 1: 10.0}


def test_annual_by_year_one_row_per_year_identity():
    assert fis.annual_by_year([10.0, 20.0], [0.0, 1.0], how="sum") == {0: 10.0, 1: 20.0}
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validation_stock_status.py -k annual_by_year -v`
Expected: FAIL (`annual_by_year` not defined).

- [ ] **Step 3: Add `annual_by_year`** — in `osmose/validation/fisheries.py` (after `annual_rate`):

```python
def annual_by_year(values, time, *, how: str) -> dict[int, float]:
    """Aggregate a per-saved-step series to one value per ABSOLUTE simulation year.

    Groups by ``int(floor(time))`` so any output.recordfrequency.ndt works. ``how="sum"``
    for accumulating quantities (F), ``how="mean"`` for stock levels (SSB).
    """
    if how not in ("sum", "mean"):
        raise ValueError(f"how must be 'sum' or 'mean', got {how!r}")
    s = pd.Series(np.asarray(values, dtype=float))
    years = np.floor(np.asarray(time, dtype=float)).astype(int)
    grouped = s.groupby(years)
    agg = grouped.sum() if how == "sum" else grouped.mean()
    return {int(y): float(v) for y, v in agg.items()}
```

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validation_stock_status.py -k annual_by_year tests/test_validation_fisheries.py -q`
Expected: PASS (new + the existing F/M tests, which still use `annual_rate`, untouched).

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fisheries.py tests/test_validation_stock_status.py
git commit -m "feat(fisheries): annual_by_year aggregator (cadence-correct, absolute-year)"
```

---

### Task 4: `fisheries_reference.py` — reference-point resolver

**Files:**
- Create: `osmose/validation/fisheries_reference.py`
- Test: `tests/test_validation_fisheries_reference.py` (new)

**Interfaces:**
- Consumes: `osmose.validation.ices.load_snapshot` (returns `IcesSnapshot` with `.manifest`, `.reference_points`).
- Produces: `ReferencePoint` dataclass; `load_reference_points(ref_dir, species_list, *, ices_snapshot_dir=None) -> tuple[dict[str, ReferencePoint], list[str]]` (the dict + unmatched-key warnings); `save_reference_points(ref_dir, refs)`; `ecosystem_of(config_dir) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_fisheries_reference.py
import json
from pathlib import Path

from osmose.validation import fisheries_reference as fr

ICES = Path("data/baltic/reference/ices_snapshots")


def test_autofill_fmsy_from_primary_tonnes_stock():
    refs, unmatched = fr.load_reference_points(
        Path("/nonexistent"), ["sprat", "cod", "perch"], ices_snapshot_dir=ICES)
    # sprat: single tonnes stock spr.27.22-32 fmsy=0.34
    assert refs["sprat"].fmsy == 0.34
    assert refs["sprat"].fmsy_stock == "spr.27.22-32"
    assert refs["sprat"].has_f_axis and not refs["sprat"].has_b_axis
    # cod: cod.27.22-24 (tonnes) chosen over cod.27.24-32 (index, null fmsy)
    assert refs["cod"].fmsy_stock == "cod.27.22-24"
    # perch: empty stock list → no F-axis
    assert not refs["perch"].has_f_axis
    assert refs["perch"].b_ref_kind == "none"


def test_herring_multistock_deterministic_primary_with_caveat():
    refs, _ = fr.load_reference_points(Path("/nonexistent"), ["herring"], ices_snapshot_dir=ICES)
    # 3 tonnes stocks; primary = largest msy_btrigger (her.27.3031 = 613355) — DETERMINISTIC
    assert refs["herring"].fmsy_stock == "her.27.3031"
    assert refs["herring"].fmsy == 0.218
    assert any("stock" in c.lower() for c in refs["herring"].caveats)


def test_user_bmsy_and_override(tmp_path):
    (tmp_path / "fisheries_reference_points.json").write_text(
        json.dumps({"sprat": {"bmsy": 600000.0, "fmsy": 0.4}, "ghostfish": {"fmsy": 1.0}}))
    refs, unmatched = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    assert refs["sprat"].bmsy == 600000.0 and refs["sprat"].b_ref_kind == "bmsy_user"
    assert refs["sprat"].fmsy == 0.4  # user overrides ICES
    assert "ghostfish" in unmatched  # key with no matching species


def test_save_roundtrip(tmp_path):
    refs, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    refs["sprat"].bmsy = 500000.0
    fr.save_reference_points(tmp_path, refs)
    reloaded, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    assert reloaded["sprat"].bmsy == 500000.0


def test_ecosystem_of():
    assert fr.ecosystem_of(Path("/x/data/baltic")) == "baltic"
    assert fr.ecosystem_of(Path("/x/data/eec_full")) == "eec_full"
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validation_fisheries_reference.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `osmose/validation/fisheries_reference.py`:

```python
"""Resolve per-species fisheries reference points: user-supplied Bmsy + ICES-auto-filled Fmsy."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from osmose.validation.ices import load_snapshot

_SIDECAR = "fisheries_reference_points.json"


@dataclass
class ReferencePoint:
    species: str
    fmsy: float | None = None
    bmsy: float | None = None
    fmsy_stock: str | None = None
    fmsy_year: int | None = None
    b_ref_kind: str = "none"  # "bmsy_user" | "none"
    source: str = "none"  # "ices:<stock>@<year>" | "user" | "mixed"
    caveats: list[str] = field(default_factory=list)

    @property
    def has_f_axis(self) -> bool:
        return self.fmsy is not None and self.fmsy > 0

    @property
    def has_b_axis(self) -> bool:
        return self.bmsy is not None and self.bmsy > 0

    @property
    def b_ref_label(self) -> str:
        return "Bmsy [user]"


def ecosystem_of(config_dir: Path | None) -> str:
    """Ecosystem name = the run's config/data dir basename (e.g. 'baltic', 'eec_full')."""
    return config_dir.name if config_dir is not None else "unknown"


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _autofill_fmsy(species: str, snapshot, rp: ReferencePoint) -> None:
    """Set fmsy/fmsy_stock/fmsy_year from the primary tonnes-unit stock (largest msy_btrigger;
    tie → latest advice_year). Caveat when >1 tonnes stock."""
    manifest = snapshot.manifest
    stocks = manifest.get("model_species_to_ices_stocks", {}).get(species, [])
    units = manifest.get("units_by_stock", {})
    years = manifest.get("advice_year_by_stock", {})
    tonnes = [s for s in stocks if units.get(s) == "tonnes"]
    cand = [(s, _float(snapshot.reference_points.get(s, {}).get("fmsy"))) for s in tonnes]
    cand = [(s, f) for s, f in cand if f is not None]
    if not cand:
        return
    def sort_key(s: str):
        bt = _float(snapshot.reference_points.get(s, {}).get("msy_btrigger")) or 0.0
        return (bt, years.get(s, 0))
    primary = max((s for s, _ in cand), key=sort_key)
    rp.fmsy = _float(snapshot.reference_points.get(primary, {}).get("fmsy"))
    rp.fmsy_stock = primary
    rp.fmsy_year = years.get(primary)
    rp.source = f"ices:{primary}@{rp.fmsy_year}"
    if len(tonnes) > 1:
        rp.caveats.append(f"Fmsy from primary stock {primary}; species maps to {len(tonnes)} tonnes stocks")


def load_reference_points(
    ref_dir: Path, species_list: list[str], *, ices_snapshot_dir: Path | None = None
) -> tuple[dict[str, ReferencePoint], list[str]]:
    user: dict[str, dict] = {}
    p = ref_dir / _SIDECAR
    if p.exists():
        user = json.loads(p.read_text())
    unmatched = [k for k in user if k not in species_list]

    snapshot = None
    if ices_snapshot_dir is not None and Path(ices_snapshot_dir).exists():
        snapshot = load_snapshot(Path(ices_snapshot_dir))

    refs: dict[str, ReferencePoint] = {}
    for sp in species_list:
        rp = ReferencePoint(species=sp)
        if snapshot is not None:
            _autofill_fmsy(sp, snapshot, rp)
        u = user.get(sp, {})
        if _float(u.get("fmsy")) is not None:
            rp.fmsy = _float(u["fmsy"])
            rp.source = "user" if rp.fmsy_stock is None else "mixed"
        if _float(u.get("bmsy")) is not None:
            rp.bmsy = _float(u["bmsy"])
            rp.b_ref_kind = "bmsy_user"
        refs[sp] = rp
    return refs, unmatched


def save_reference_points(ref_dir: Path, refs: dict[str, ReferencePoint]) -> None:
    """Persist only USER-owned fields: bmsy always; fmsy ONLY when user-supplied (source
    user/mixed), never the ICES-auto-filled value — so reload re-derives Fmsy from the live
    snapshot rather than freezing a stale auto-fill."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for sp, r in refs.items():
        entry = {}
        if r.bmsy is not None:
            entry["bmsy"] = r.bmsy
        if r.fmsy is not None and r.source in ("user", "mixed"):
            entry["fmsy"] = r.fmsy
        if entry:
            payload[sp] = entry
    (ref_dir / _SIDECAR).write_text(json.dumps(payload, indent=2))
```

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validation_fisheries_reference.py -v`
Expected: PASS (5 passed). If `load_snapshot`'s return shape differs, adapt the `snapshot.reference_points` / `snapshot.manifest` access to its actual attributes (check `osmose/validation/ices.py`).

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fisheries_reference.py tests/test_validation_fisheries_reference.py
git commit -m "feat(validation): fisheries reference-point resolver (user Bmsy + ICES Fmsy)"
```

---

### Task 5: `stock_status.py` — the computation

**Files:**
- Create: `osmose/validation/stock_status.py`
- Test: `tests/test_validation_stock_status.py`

**Interfaces:**
- Consumes: `results.ssb` / `read_mortality` / `annual_by_year` (Tasks 2-3); `ReferencePoint` (Task 4).
- Produces: `StockStatus` dataclass; `compute_stock_status(results, refs, config, *, species_list=None) -> list[StockStatus]`.

- [ ] **Step 1: Write the failing test** (imports `ss`/`ReferencePoint` are already at the top of the file from Task 3 — do NOT re-import here)

```python
# add to tests/test_validation_stock_status.py


class _Cfg:
    n_dt_per_year = 1
    output_record_frequency = 1


class _FakeResults:
    """Minimal results stub: ssb() wide-form (Time + species col)."""
    def __init__(self, ssb_rows, time=(0.0, 1.0)):
        self._ssb = list(ssb_rows)
        self._time = list(time)

    def ssb(self, species=None):
        return pd.DataFrame({"Time": self._time, "cod": self._ssb})


def test_quadrant_and_ratios():
    refs = {"cod": ReferencePoint(species="cod", fmsy=0.3, bmsy=100.0, b_ref_kind="bmsy_user")}
    statuses = ss.compute_stock_status(
        _FakeResults([120.0, 80.0]), refs, _Cfg(), species_list=["cod"],
        _f_override={"cod": {0: 0.15, 1: 0.45}})
    s = statuses[0]
    # year0: B/Bmsy=1.2, F/Fmsy=0.5 → green; year1: 0.8, 1.5 → red
    assert s.b_over_bmsy == [1.2, 0.8]
    assert s.f_over_fmsy == pytest.approx([0.5, 1.5])
    assert s.latest_quadrant == "red"
    assert s.takeaway is not None


def test_ssb_annual_mean_over_subannual_rows():
    # 2 saved rows in year 0 (Time 0.0, 0.5) → MEAN 110, 1 row in year 1 → 80 (NOT last-row)
    res = _FakeResults([100.0, 120.0, 80.0], time=(0.0, 0.5, 1.0))
    refs = {"cod": ReferencePoint(species="cod", bmsy=100.0, b_ref_kind="bmsy_user")}
    s = ss.compute_stock_status(res, refs, _Cfg(), species_list=["cod"])[0]
    assert s.b_over_bmsy == [1.1, 0.8]


def test_data_limited_single_axis():
    refs = {"cod": ReferencePoint(species="cod", fmsy=0.3)}  # no bmsy → no B-axis
    statuses = ss.compute_stock_status(
        _FakeResults([120.0, 80.0]), refs, _Cfg(), species_list=["cod"],
        _f_override={"cod": {0: 0.15, 1: 0.45}})
    s = statuses[0]
    assert all(v is None for v in s.b_over_bmsy)
    assert s.latest_quadrant is None  # needs both axes
    assert any("Bmsy" in c for c in s.caveats)
```

(The `_f_override` (a `{year: F}` dict) keeps the test independent of the mortalityRate CSV format; production derives F via `_exploited_f_by_year` — see Step 3. `test_ssb_annual_mean_over_subannual_rows` pins the cadence-correct MEAN aggregation.)

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validation_stock_status.py -k "quadrant or data_limited" -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `osmose/validation/stock_status.py`:

```python
"""Indicative stock status: per-year SSB/Bmsy and exploited-stage F/Fmsy → Kobe quadrants."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from osmose.validation import fisheries as fis
from osmose.validation.fisheries_reference import ReferencePoint

_EXPLOITABLE = ("Pre-recruits", "Recruits")  # Eggs excluded


@dataclass
class StockStatus:
    species: str
    years: list[int]
    b_over_bmsy: list[float | None]
    f_over_fmsy: list[float | None]
    b_ref_label: str
    latest_quadrant: str | None = None
    takeaway: str | None = None
    caveats: list[str] = field(default_factory=list)


def _quadrant(b: float | None, f: float | None) -> str | None:
    if b is None or f is None:
        return None
    if b >= 1 and f <= 1:
        return "green"
    if b < 1 and f > 1:
        return "red"
    return "yellow" if b >= 1 else "orange"


def _exploited_f_by_year(results, species, caveats) -> dict[int, float] | None:
    """{absolute_year: annual F} on the exploited stage = the fished stage (Eggs excluded)
    with the largest total annual F. Years come from the mortalityRate Time column."""
    from osmose.validation.fisheries import _FISHED_TOL, _mortality_path, read_mortality

    try:
        df = read_mortality(_mortality_path(results.output_dir, results.prefix, species))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"WARN: no mortalityRate for {species!r}: {e}", file=sys.stderr)
        return None
    time = df.iloc[:, 0]  # first column = Time (fractional sim-year)
    per_stage = {
        s: fis.annual_by_year(df[("F", s)].to_numpy(), time.to_numpy(), how="sum")
        for s in _EXPLOITABLE
        if ("F", s) in df.columns
    }
    fished = {s: d for s, d in per_stage.items() if sum(d.values()) > _FISHED_TOL}
    if not fished:
        return None
    stage = max(fished, key=lambda s: sum(fished[s].values()))
    if len(fished) > 1:
        caveats.append(f"F measured on '{stage}'; other fished stages present")
    return fished[stage]


def compute_stock_status(results, refs, config, *, species_list=None, _f_override=None):
    species_list = species_list or list(refs)
    out: list[StockStatus] = []
    for sp in species_list:
        rp: ReferencePoint = refs.get(sp, ReferencePoint(species=sp))
        caveats = list(rp.caveats)

        # F per absolute year (dict {year: annual F})
        if _f_override and sp in _f_override:
            f_map = dict(_f_override[sp])
        else:
            f_map = _exploited_f_by_year(results, sp, caveats) or {}

        # SSB per absolute year — annual MEAN of the saved rows in each year (cadence-correct)
        b_map: dict[int, float] = {}
        try:
            sdf = results.ssb(sp)
            if sp in sdf.columns:
                b_map = fis.annual_by_year(sdf[sp].to_numpy(), sdf["Time"].to_numpy(), how="mean")
        except (FileNotFoundError, KeyError, ValueError):
            caveats.append("SSB unavailable (enable output.ssb.enabled)")

        years = sorted(set(f_map) | set(b_map))
        b_ratio: list[float | None] = []
        f_ratio: list[float | None] = []
        for y in years:
            b, f = b_map.get(y), f_map.get(y)
            b_ratio.append(b / rp.bmsy if (rp.has_b_axis and b is not None) else None)
            f_ratio.append(f / rp.fmsy if (rp.has_f_axis and f is not None) else None)
        if not rp.has_b_axis:
            caveats.append("No Bmsy supplied — B-axis unavailable")
        if not rp.has_f_axis:
            caveats.append("No Fmsy — F-axis unavailable")

        quad = None
        takeaway = None
        for i in range(len(years) - 1, -1, -1):
            quad = _quadrant(b_ratio[i], f_ratio[i])
            if quad is not None:
                takeaway = (
                    f"Indicative: F {'above' if f_ratio[i] > 1 else 'at/below'} Fmsy and "
                    f"SSB {'below' if b_ratio[i] < 1 else 'at/above'} your Bmsy"
                )
                break
        out.append(
            StockStatus(
                species=sp,
                years=years,
                b_over_bmsy=b_ratio,
                f_over_fmsy=f_ratio,
                b_ref_label=rp.b_ref_label,
                latest_quadrant=quad,
                takeaway=takeaway,
                caveats=caveats,
            )
        )
    return out
```

Reuse the module-level `_mortality_path(output_dir, prefix, species)` at `fisheries.py:66` (builds `{output_dir}/Mortality/{prefix}_mortalityRate-{species}_Simu0.csv`); confirm its exact signature before relying on it.

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validation_stock_status.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/stock_status.py tests/test_validation_stock_status.py
git commit -m "feat(validation): indicative stock-status computation (SSB/Bmsy, exploited-stage F/Fmsy)"
```

---

### Task 6: Kobe + ratio-timeseries plots

**Files:**
- Modify: `osmose/plotting.py` (add two functions near `make_fm_ratio_bars` ~337)
- Test: `tests/test_plotting_kobe.py` (new)

**Interfaces:**
- Consumes: `list[StockStatus]` (Task 5).
- Produces: `make_kobe_plot(statuses, *, year=None) -> go.Figure`; `make_ratio_timeseries(statuses, which) -> go.Figure` (`which ∈ {"b","f"}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plotting_kobe.py
import plotly.graph_objects as go

from osmose.plotting import make_kobe_plot, make_ratio_timeseries
from osmose.validation.stock_status import StockStatus


def _status():
    return [StockStatus(species="cod", years=[0, 1], b_over_bmsy=[1.2, 0.8],
                        f_over_fmsy=[0.5, 1.5], b_ref_label="Bmsy [user]", latest_quadrant="red")]


def test_kobe_plot_builds_with_soft_quadrants_and_indicative_note():
    fig = make_kobe_plot(_status())
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) >= 4  # four quadrant rectangles
    txt = " ".join(a.text for a in fig.layout.annotations if a.text)
    assert "ndicative" in txt  # the indicative annotation


def test_kobe_skips_partial_reference_species():
    partial = [StockStatus(species="x", years=[0], b_over_bmsy=[None], f_over_fmsy=[1.0],
                           b_ref_label="Bmsy [user]")]
    fig = make_kobe_plot(partial)
    pts = sum(len(t.x or []) for t in fig.data if isinstance(t, go.Scatter))
    assert pts == 0  # no plottable point (missing B axis)


def test_ratio_timeseries_builds():
    assert isinstance(make_ratio_timeseries(_status(), "f"), go.Figure)
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_plotting_kobe.py -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement** — in `osmose/plotting.py`, after `make_fm_ratio_bars`. Use the module-level `go` import and the `TEMPLATE` constant already imported at the top of `plotting.py` (`from osmose.plotly_theme import PLOTLY_TEMPLATE as TEMPLATE`); do NOT add inline imports or hardcode `"osmose"`.

```python
def make_kobe_plot(statuses, *, year=None):
    """Indicative Kobe scatter: x=B/Bmsy, y=F/Fmsy, soft-shaded quadrants. Points only for
    species with BOTH ratios defined at the selected year (default: each species' latest)."""
    fig = go.Figure()
    xmax, ymax = 2.0, 2.0
    quads = [  # (x0,x1,y0,y1,color) — green healthy bottom-right, red top-left
        (1, xmax, 0, 1, "rgba(0,160,80,0.10)"),
        (0, 1, 1, ymax, "rgba(200,30,30,0.10)"),
        (1, xmax, 1, ymax, "rgba(230,160,0,0.10)"),
        (0, 1, 0, 1, "rgba(230,200,0,0.10)"),
    ]
    for x0, x1, y0, y1, c in quads:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=c,
                      line=dict(width=0), layer="below")
    fig.add_hline(y=1, line=dict(color="grey", dash="dash"))
    fig.add_vline(x=1, line=dict(color="grey", dash="dash"))
    xs, ys, names = [], [], []
    for s in statuses:
        idx = len(s.years) - 1 if year is None else (s.years.index(year) if year in s.years else None)
        if idx is None:
            continue
        b, f = s.b_over_bmsy[idx], s.f_over_fmsy[idx]
        if b is None or f is None:
            continue
        xs.append(b)
        ys.append(f)
        names.append(s.species)
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=names,
                                 textposition="top center", marker=dict(size=12)))
    fig.add_annotation(x=0.5, y=1.06, xref="paper", yref="paper", showarrow=False,
                       text="Indicative — relative to supplied reference points, not a formal assessment")
    label = statuses[0].b_ref_label if statuses else "Bmsy"
    fig.update_layout(template=TEMPLATE, xaxis_title=f"SSB / {label}", yaxis_title="F / Fmsy",
                      xaxis_range=[0, xmax], yaxis_range=[0, ymax])
    return fig


def make_ratio_timeseries(statuses, which):
    """Time-series of B/Bmsy (which='b') or F/Fmsy (which='f') per species."""
    fig = go.Figure()
    for s in statuses:
        vals = s.b_over_bmsy if which == "b" else s.f_over_fmsy
        xy = [(y, v) for y, v in zip(s.years, vals) if v is not None]
        if xy:
            fig.add_trace(go.Scatter(x=[y for y, _ in xy], y=[v for _, v in xy],
                                     mode="lines+markers", name=s.species))
    fig.add_hline(y=1, line=dict(color="grey", dash="dash"))
    title = "SSB / Bmsy" if which == "b" else "F / Fmsy"
    fig.update_layout(template=TEMPLATE, xaxis_title="Year", yaxis_title=title)
    return fig
```

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_plotting_kobe.py -v`
Expected: PASS. (If the `"osmose"` template is not registered in the test process, the existing plotting tests show the pattern — register it or assert on a figure built without the template.)

- [ ] **Step 5: Commit**

```bash
git add osmose/plotting.py tests/test_plotting_kobe.py
git commit -m "feat(plotting): indicative Kobe + B/Bmsy & F/Fmsy ratio timeseries"
```

---

### Task 7: `Fisheries` Shiny page + registration

**Files:**
- Create: `ui/pages/fisheries.py`
- Modify: `app.py` (import; `ui.nav_panel(...)` near line 368; `*_server(...)` near line 656)
- Test: `tests/test_ui_fisheries.py` (new)

**Interfaces:**
- Consumes: `compute_stock_status`, `load_reference_points`/`save_reference_points`/`ecosystem_of`, `make_kobe_plot`/`make_ratio_timeseries`/`make_fm_ratio_bars`, `compute_mortality_balance`, `AppState`.
- Produces: `fisheries_ui()`, `fisheries_server(input, output, session, state)`.

- [ ] **Step 1: Write the failing test** (logic-level — the data assembly the page renders, kept UI-framework-light)

```python
# tests/test_ui_fisheries.py
from osmose.validation.stock_status import StockStatus
from ui.pages.fisheries import build_fisheries_view


def test_empty_state_when_no_run():
    view = build_fisheries_view(None, None, "baltic")
    assert view["kobe_ready"] is False  # no run → CTA, not a blank plot
    assert "Enter a Bmsy" in view["kobe_cta"]
    assert view["lead"] == "fm_bars"  # never leads with an empty Kobe


def test_kobe_gated_until_a_species_has_both_axes(monkeypatch):
    import ui.pages.fisheries as page
    monkeypatch.setattr(page, "load_reference_points", lambda *a, **k: ({}, []))
    cfg = type("C", (), {"species_names": ["cod"]})()
    res = object()
    # F-only status → no quadrant → Kobe NOT ready
    monkeypatch.setattr(page, "compute_stock_status", lambda *a, **k: [
        StockStatus("cod", [0], [None], [0.5], "Bmsy [user]", latest_quadrant=None)])
    assert build_fisheries_view(res, cfg, "baltic")["kobe_ready"] is False
    # both-axis status → quadrant → Kobe ready, save target shown
    monkeypatch.setattr(page, "compute_stock_status", lambda *a, **k: [
        StockStatus("cod", [0], [1.2], [0.5], "Bmsy [user]", latest_quadrant="green")])
    v = build_fisheries_view(res, cfg, "baltic")
    assert v["kobe_ready"] is True
    assert v["save_target"].endswith("baltic/reference")
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ui_fisheries.py -v`
Expected: FAIL (module/function missing).

- [ ] **Step 3: Implement** — create `ui/pages/fisheries.py` with a `build_fisheries_view(...)` pure helper (returns the dict above + the figures/table data) and the `fisheries_ui()`/`fisheries_server()` wrappers. Mirror an existing page (`ui/pages/results.py`) for the `_ui()/_server(input, output, session, state)` shape, `_safe_output_dir`, and `@render.ui`/`@render_widget` usage. Key logic:

```python
from pathlib import Path

from osmose.validation.fisheries_reference import load_reference_points
from osmose.validation.stock_status import compute_stock_status


def build_fisheries_view(results, config, ecosystem, *, ices_snapshot_dir=None):
    """Pure assembly of what the page renders (given an OsmoseResults-like `results` + an
    EngineConfig). Leads with zero-config content; the Kobe is gated on >=1 species having
    BOTH ratios (latest_quadrant is not None) — else an explicit CTA, never a blank plot."""
    view = {"lead": "fm_bars", "kobe_ready": False,
            "kobe_cta": "Enter a Bmsy for >=1 species in the table to populate the Kobe quadrant.",
            "statuses": [], "unmatched": [], "save_target": None}
    if results is None or config is None:
        return view
    ref_dir = Path("data") / ecosystem / "reference"
    view["save_target"] = str(ref_dir)
    species = list(config.species_names)
    refs, unmatched = load_reference_points(ref_dir, species, ices_snapshot_dir=ices_snapshot_dir)
    view["unmatched"] = unmatched
    statuses = compute_stock_status(results, refs, config, species_list=species)
    view["statuses"] = statuses
    view["kobe_ready"] = any(s.latest_quadrant is not None for s in statuses)
    return view
```

Then `fisheries_server` builds `EngineConfig.from_dict(state.config.get())`, an `OsmoseResults(state.output_dir.get(), prefix=...)`, resolves the ecosystem via `ecosystem_of(state.config_dir.get())`, calls `build_fisheries_view(results, config, ecosystem, ices_snapshot_dir=...)`, and renders: F/M bars (`compute_mortality_balance` + `make_fm_ratio_bars`) and the F/Fmsy timeseries first; the Kobe panel only when `view["kobe_ready"]` (else the `kobe_cta`); the editable `bmsy`/`fmsy` numeric inputs per species (showing each species' current mean SSB beside its `bmsy` input for scale); a Save button (`save_reference_points`, showing `view["save_target"]` + "shared across <ecosystem> runs"); the `view["unmatched"]` warning; and the disclaimer banner. Keep `build_fisheries_view` pure (unit-tested above); keep the `@render` wiring thin (exercised by the app-import smoke).

- [ ] **Step 4: Register in `app.py`** — add `from ui.pages.fisheries import fisheries_server, fisheries_ui`; a `ui.nav_panel("Fisheries", fisheries_ui(), value="fisheries")` next to the Results panel (line ~368); and `fisheries_server(input, output, session, state)` in the server body (line ~656).

- [ ] **Step 5: Run test + app import — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ui_fisheries.py -v && PYTHONPATH=. .venv/bin/python -c "import app"`
Expected: PASS + app imports clean.

- [ ] **Step 6: Full sweep + lint + commit**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_engine_ssb_output.py tests/test_validation_stock_status.py tests/test_validation_fisheries_reference.py tests/test_plotting_kobe.py tests/test_ui_fisheries.py tests/test_engine_parity.py tests/test_validation_fisheries.py -q`
Expected: all pass (new suites + EEC parity + the existing F/M tests).
Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/pages/fisheries.py tests/`
Expected: clean.
```bash
git add ui/pages/fisheries.py app.py tests/test_ui_fisheries.py
git commit -m "feat(ui): indicative Fisheries stock-status page (Kobe + F/Fmsy + F/M bars)"
```

---

## Notes for the executor

- **Do NOT touch engine dynamics.** Task 1's SSB collector is read-only over `state`; if any EEC/BoB parity test changes, stop and investigate (do not re-baseline).
- **SSB is mean-aggregated** across the record window in the engine collector (Task 1 step 5d uses `np.mean` for `ssb_avg`); F accumulates. Get these two right where they sit beside `yield_n_sum`.
- **Annual aggregation = `fis.annual_by_year` (groupby `int(Time)`)** — SSB `how="mean"`, F `how="sum"` (Task 3/5). This labels every value by ABSOLUTE simulation year and is correct for ANY `output.recordfrequency.ndt` (multiple saved rows/year collapse correctly). Never use a positional `np.arange`/`reshape`. `test_ssb_annual_mean_over_subannual_rows` pins this.
- The plan reuses the just-merged yieldN/meanSize output as the SSB template — open `git show af19fa0 -- osmose/engine/simulate.py osmose/engine/output.py osmose/results.py` if a wiring location is unclear.
- Verify `ices.load_snapshot`'s real return attributes before Task 4 (the test asserts `.manifest`/`.reference_points`) and `_mortality_path`'s signature before Task 5; adapt if they differ.
- Keep `build_fisheries_view` pure + tested (monkeypatch `load_reference_points`/`compute_stock_status` for the gating test); the Shiny `@render` wiring is thin and exercised by the app-import smoke.
- **Pre-format the code blocks** before committing each task: run `.venv/bin/ruff check --fix` + `.venv/bin/ruff format` on the new/edited files so the Task-7 lint gate (`ruff check osmose/ ui/ tests/` + `ruff format --check`) passes; the plan's blocks are logically correct but may need whitespace/import-order normalization.
