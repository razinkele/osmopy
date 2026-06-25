# Model-internal fishery reference points — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive per-species Fmsy / Bmsy / Blim from an OSMOSE yield-vs-F sweep and feed them to the fisheries page so the Kobe auto-populates without user input.

**Architecture:** A library (`osmose/validation/fmsy_sweep.py`) sweeps one species' fishing rate at a time (others at baseline), reading equilibrium yield + SSB + realized F per run via the in-memory engine on a ProcessPool; a pure `derive_reference_points` turns the curves into reference points; a CLI writes a model sidecar; the existing `fisheries_reference.py` reads it (precedence user > model > ICES).

**Tech Stack:** Python 3.12, NumPy, pandas, the OSMOSE Python engine, `concurrent.futures` ProcessPool, numba, pytest.

## Global Constraints

- **Parity-safe / output-only:** the sweep only RUNS the engine with varied fishing-rate + output-flag config (as calibration does). EEC/BoB parity suites must stay green.
- **THE override crux:** `EngineConfig.from_dict` ignores `mortality.fishing.rate.sp{i}` when `module.multispecies.fisheries.enabled==true AND simulation.nfisheries>0` (BOTH bundled configs). The sweep MUST detect mode and override the **active** knob — fisheries mode → `fisheries.rate.base.fsh{j}` (species→fishery via the catchability matrix), legacy → `mortality.fishing.rate.sp{i}` — and **assert `EngineConfig.from_dict(...).fishing_rate[i]` actually changed** before each run.
- **Readers:** equilibrium yield = `results.yield_biomass(species)` (the in-memory `"yield"` output, tonnes — NOT `results.fishery_yield`, which raises in-memory). SSB = `results.ssb(species)`, which is **gated** — each sweep config must set `output.ssb.enabled=true` (and `output.yield.biomass.enabled=true`).
- **Realized-F basis:** Fmsy is reported on the realized exploited-stage annual F (read from each run's in-memory `results.mortality(species)`, the same extraction `stock_status.py` uses), matching the page's F — not the nominal grid value.
- **Equilibrium:** trailing-window mean (last `window_years=10`) of an `n_years=max(config nyear,30)` run; flag `not_converged` if the last window still trends vs the prior window.
- **Defaults:** `grid=np.linspace(0.0, 2.0, 7)`, `replicates=3`, `window_years=10`; ProcessPool with `numba.set_num_threads(1)` per worker (a single run already saturates cores via the `@njit(parallel=True)` mortality kernel).
- **Precedence:** user > model > ICES. `b_ref_kind` gains `"bmsy_model"`; `b_ref_label` becomes conditional (`"Bmsy [model]"` / `"Bmsy [user]"`).
- **Run all** with `PYTHONPATH=.` from the worktree root using `.venv/bin/python`. Lint: `.venv/bin/ruff check` + `ruff format --check`.
- Spec: `docs/superpowers/specs/2026-06-25-model-internal-reference-points-design.md`.

---

### Task 1: `_fishing_override` — mode detection + species→fishery map (the no-op-trap fix)

**Files:**
- Create: `osmose/validation/fmsy_sweep.py`
- Test: `tests/test_fmsy_sweep.py` (new)

**Interfaces:**
- Produces: `fishing_override(base_config: dict, config: EngineConfig, species_idx: int) -> tuple[str, float]` — returns `(override_key, baseline_value)`: the active fishing key to sweep for species *i* and its current value. Raises `SharedFisheryError` if the species' fishery lands >1 species. Detects fisheries vs legacy mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fmsy_sweep.py
import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.validation.fmsy_sweep import SharedFisheryError, fishing_override

BALTIC = "data/baltic/baltic_all-parameters.csv"


def _baltic_cfg() -> dict[str, str]:
    raw = dict(OsmoseConfigReader().read(BALTIC))
    raw["simulation.time.nyear"] = "2"
    return raw


def test_fisheries_mode_override_actually_changes_fishing_rate():
    raw = _baltic_cfg()
    cfg = EngineConfig.from_dict(dict(raw))
    key, base = fishing_override(raw, cfg, 0)
    assert key.startswith("fisheries.rate.base.fsh")  # baltic is v4 fisheries-mode
    assert base == pytest.approx(cfg.fishing_rate[0])
    # overriding the returned key MUST move fishing_rate[0] (the no-op-trap guard)
    bumped = dict(raw)
    bumped[key] = "9.0"
    assert EngineConfig.from_dict(bumped).fishing_rate[0] == pytest.approx(9.0)
    # and ONLY species 0 if 1:1 (baltic is 1:1)
    assert EngineConfig.from_dict(bumped).fishing_rate[1] == pytest.approx(cfg.fishing_rate[1])


def test_legacy_mode_override():
    raw = {
        "simulation.time.ndtperyear": "12", "simulation.time.nyear": "1",
        "simulation.nspecies": "1", "simulation.nschool.sp0": "5",
        "species.name.sp0": "Fish", "species.linf.sp0": "20.0", "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1", "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0", "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0", "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5", "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random", "movement.randomwalk.range.sp0": "1",
        "mortality.fishing.rate.method.sp0": "constant", "mortality.fishing.rate.sp0": "0.2",
    }
    cfg = EngineConfig.from_dict(dict(raw))
    key, base = fishing_override(raw, cfg, 0)
    assert key == "mortality.fishing.rate.sp0"
    bumped = dict(raw); bumped[key] = "0.9"
    assert EngineConfig.from_dict(bumped).fishing_rate[0] == pytest.approx(0.9)
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py -k override -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement** — create `osmose/validation/fmsy_sweep.py` with the override resolver. Build the species→fishery map by replicating `_parse_fisheries` (osmose/engine/config.py:296-313): read `fisheries.catchability.file` (resolve relative to the config dir), map `species_name.lower() → first fishery column with catchability > 0`.

```python
"""Model-internal fishery reference points via a per-species yield-vs-F sweep."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from osmose.engine.config import EngineConfig


class SharedFisheryError(RuntimeError):
    """Raised when a species' fishery lands >1 species (per-species sweep is ambiguous)."""


def _fisheries_enabled(cfg: dict) -> bool:
    return (
        cfg.get("module.multispecies.fisheries.enabled", "false").lower() == "true"
        and int(cfg.get("simulation.nfisheries", "0")) > 0
    )


def _species_to_fishery(cfg: dict, config_dir: Path) -> dict[str, int]:
    """species_name.lower() -> fishery column index (first catchability column > 0)."""
    catch_rel = cfg.get("fisheries.catchability.file")
    catch_path = (config_dir / catch_rel) if catch_rel else None
    if catch_path is None or not Path(catch_path).exists():
        # fall back: resolve like the engine (cfg dir of the catchability key)
        raise FileNotFoundError(f"catchability file not found: {catch_rel}")
    df = pd.read_csv(catch_path, index_col=0)
    out: dict[str, int] = {}
    for r in range(len(df)):
        name = str(df.index[r]).strip().lower()
        for c in range(len(df.columns)):
            if float(df.iloc[r, c]) > 0:
                out[name] = c
                break
    return out


def fishing_override(base_config: dict, config: EngineConfig, species_idx: int) -> tuple[str, float]:
    """Return (override_key, baseline_value) for the ACTIVE fishing knob of species `species_idx`.

    fisheries mode -> fisheries.rate.base.fsh{j}; legacy -> mortality.fishing.rate.sp{i}.
    Raises SharedFisheryError if the fishery lands >1 species.
    """
    sp_name = config.species_names[species_idx].strip().lower()
    if _fisheries_enabled(base_config):
        config_dir = Path(base_config["__config_dir__"]) if "__config_dir__" in base_config else Path(".")
        s2f = _species_to_fishery(base_config, config_dir)
        fsh = s2f.get(sp_name)
        if fsh is None:
            raise ValueError(f"species {sp_name!r} maps to no fishery")
        sharing = [n for n, j in s2f.items() if j == fsh]
        if len(sharing) > 1:
            raise SharedFisheryError(f"fishery {fsh} lands {len(sharing)} species: {sharing}")
        key = f"fisheries.rate.base.fsh{fsh}"
    else:
        key = f"mortality.fishing.rate.sp{species_idx}"
    return key, float(config.fishing_rate[species_idx])
```

Resolve the config dir: the runner passes `base_config["__config_dir__"]` (set by the CLI from the config path's parent). If the catchability path is absolute or the key already resolves, handle both. Verify `fisheries.catchability.file`'s real value + resolution against a bundled config during implementation; adapt the path join if the engine resolves it differently (`osmose/engine/config.py:_require_file`).

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py -k override -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fmsy_sweep.py tests/test_fmsy_sweep.py
git commit -m "feat(fmsy): fishing-override resolver (mode detection + species->fishery map)"
```

---

### Task 2: `derive_reference_points` — the pure peak/Bmsy/Blim logic

**Files:**
- Modify: `osmose/validation/fmsy_sweep.py`
- Test: `tests/test_fmsy_sweep.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `@dataclass SweepPoint(species, f_nominal, f_realized, yield_eq, ssb_eq, not_converged=False)`;
  `@dataclass ModelReferencePoint(species, fmsy, bmsy, b0, blim, fmsy_at_boundary, multi_peak, caveats)`;
  `derive_reference_points(curves: dict[str, list[SweepPoint]]) -> dict[str, ModelReferencePoint]`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_fmsy_sweep.py
from osmose.validation.fmsy_sweep import (
    ModelReferencePoint, SweepPoint, derive_reference_points)


def _curve(species, fs, yields, ssbs, frealized=None):
    fr = frealized or fs
    return [SweepPoint(species, fn, r, y, s) for fn, r, y, s in zip(fs, fr, yields, ssbs)]


def test_single_peak_fmsy_bmsy_b0_blim():
    # yield rises then falls; peak at f_nominal=0.6 (realized 0.5); SSB declines with F
    fs = [0.0, 0.3, 0.6, 0.9, 1.2]
    rs = [0.0, 0.25, 0.5, 0.75, 1.0]
    yields = [0.0, 8.0, 10.0, 7.0, 3.0]
    ssbs = [1000.0, 700.0, 500.0, 300.0, 150.0]
    rp = derive_reference_points({"cod": _curve("cod", fs, yields, ssbs, rs)})["cod"]
    assert rp.fmsy == pytest.approx(0.5)        # realized F at the yield peak
    assert rp.bmsy == pytest.approx(500.0)      # SSB at the peak
    assert rp.b0 == pytest.approx(1000.0)       # SSB at F=0
    assert rp.blim == pytest.approx(200.0)      # 0.2 * B0
    assert not rp.fmsy_at_boundary and not rp.multi_peak


def test_monotone_increasing_is_boundary():
    fs = [0.0, 0.5, 1.0]
    rp = derive_reference_points({"x": _curve("x", fs, [0.0, 5.0, 9.0], [900.0, 500.0, 200.0])})["x"]
    assert rp.fmsy_at_boundary and any("boundary" in c.lower() for c in rp.caveats)


def test_monotone_decreasing_no_fmsy():
    fs = [0.0, 0.5, 1.0]
    rp = derive_reference_points({"x": _curve("x", fs, [9.0, 5.0, 1.0], [900.0, 500.0, 200.0])})["x"]
    assert rp.fmsy is None and any("no" in c.lower() for c in rp.caveats)


def test_two_peaks_flagged():
    fs = [0.0, 0.25, 0.5, 0.75, 1.0]
    rp = derive_reference_points({"x": _curve("x", fs, [0, 9, 2, 8, 1], [900, 700, 500, 300, 150])})["x"]
    assert rp.multi_peak


def test_b0_nonpositive_no_blim():
    fs = [0.0, 0.5]
    rp = derive_reference_points({"x": _curve("x", fs, [0.0, 5.0], [0.0, -1.0])})["x"]
    assert rp.blim is None
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py -k "peak or boundary or monotone or b0" -v`
Expected: FAIL (names not defined).

- [ ] **Step 3: Implement** — add to `osmose/validation/fmsy_sweep.py`:

```python
from dataclasses import dataclass, field


@dataclass
class SweepPoint:
    species: str
    f_nominal: float
    f_realized: float
    yield_eq: float
    ssb_eq: float
    not_converged: bool = False


@dataclass
class ModelReferencePoint:
    species: str
    fmsy: float | None
    bmsy: float | None
    b0: float | None
    blim: float | None
    fmsy_at_boundary: bool = False
    multi_peak: bool = False
    caveats: list[str] = field(default_factory=list)
    curve: list = field(default_factory=list)  # the SweepPoints (for the CLI/debug)


def _count_interior_peaks(y: list[float]) -> int:
    return sum(1 for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1])


def derive_reference_points(curves: dict[str, list[SweepPoint]]) -> dict[str, ModelReferencePoint]:
    out: dict[str, ModelReferencePoint] = {}
    for sp, pts in curves.items():
        pts = sorted(pts, key=lambda p: p.f_nominal)
        caveats: list[str] = []
        b0 = next((p.ssb_eq for p in pts if p.f_nominal == 0.0), None)
        blim = 0.2 * b0 if (b0 is not None and b0 > 0) else None
        if b0 is not None and b0 <= 0:
            caveats.append("B0 <= 0; no Blim")
        ys = [p.yield_eq for p in pts]
        imax = max(range(len(ys)), key=lambda i: ys[i]) if ys else None
        multi_peak = _count_interior_peaks(ys) > 1
        if multi_peak:
            caveats.append("multi-peak yield curve; Fmsy ambiguous")
        rp = ModelReferencePoint(sp, None, None, b0, blim, multi_peak=multi_peak, caveats=caveats)
        if imax is None or ys[imax] <= 0:
            caveats.append("no positive-yield F; no Fmsy")
        elif imax == 0:
            caveats.append("yield maximal at F=0 (over-fished at baseline); no valid Fmsy")
        elif imax == len(ys) - 1:
            rp.fmsy = pts[imax].f_realized
            rp.bmsy = pts[imax].ssb_eq
            rp.fmsy_at_boundary = True
            caveats.append("Fmsy at the last grid F (boundary); extend the grid")
        else:
            rp.fmsy = pts[imax].f_realized
            rp.bmsy = pts[imax].ssb_eq
            if pts[imax].not_converged:
                caveats.append("Fmsy grid point not converged; lower confidence")
        out[sp] = rp
    return out
```

(v1 reports Fmsy at the grid argmax's realized F; parabolic sub-grid refinement is deferred — §9 of the spec — the 7-point grid + realized-F basis makes it a minor precision nicety.)

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py -k "peak or boundary or monotone or b0" -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fmsy_sweep.py tests/test_fmsy_sweep.py
git commit -m "feat(fmsy): derive_reference_points (peak/boundary/multi-peak/B0/Blim, pure)"
```

---

### Task 3: `run_yield_f_sweep` + `compute_model_reference_points` — the engine sweep

**Files:**
- Modify: `osmose/validation/fmsy_sweep.py`
- Test: `tests/test_fmsy_sweep.py`

**Interfaces:**
- Consumes: `fishing_override` (Task 1), `derive_reference_points` (Task 2), `PythonEngine.run_in_memory`, `results.yield_biomass`/`.ssb`/`.mortality`.
- Produces: `equilibrium_mean(df, sp, window_years) -> (value, not_converged)`; `realized_exploited_f(results, sp) -> float`; `run_yield_f_sweep(base_config, config, species_list, *, grid, n_years, replicates, window_years, max_workers, seed0) -> dict[str, list[SweepPoint]]`; `compute_model_reference_points(base_config, *, grid=None, n_years=None, replicates=3, window_years=10, max_workers=None) -> dict[str, ModelReferencePoint]`.

- [ ] **Step 1: Write the failing test** (integration — a tiny fast config; small grid; 1 replicate)

```python
# add to tests/test_fmsy_sweep.py
from osmose.validation.fmsy_sweep import compute_model_reference_points


def _tiny_fished_legacy_cfg() -> dict[str, str]:
    # a legacy-mode single-species config (nfisheries unset), ndt=12, 6 yr, seeded
    raw = {
        "simulation.time.ndtperyear": "12", "simulation.time.nyear": "6",
        "simulation.nspecies": "1", "simulation.nschool.sp0": "20",
        "species.name.sp0": "Fish", "species.linf.sp0": "20.0", "species.k.sp0": "0.5",
        "species.t0.sp0": "-0.1", "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0", "species.lifespan.sp0": "4",
        "species.vonbertalanffy.threshold.age.sp0": "1.0", "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5", "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random", "movement.randomwalk.range.sp0": "1",
        "mortality.fishing.rate.method.sp0": "constant", "mortality.fishing.rate.sp0": "0.3",
        "population.seeding.biomass.sp0": "100.0",
    }
    return raw


def test_sweep_end_to_end_tiny_legacy():
    refs = compute_model_reference_points(
        _tiny_fished_legacy_cfg(), grid=np.array([0.0, 0.4, 0.8, 1.2]),
        n_years=6, replicates=1, window_years=2, max_workers=2)
    rp = refs["Fish"]
    # the F=0 run must have the largest SSB (unfished), so b0 > the SSB at higher F
    assert rp.b0 is not None and rp.b0 > 0
    # yield is non-zero at some fished F (the yield reader + forced output worked)
    assert any(p.yield_eq > 0 for p in rp.curve) if hasattr(rp, "curve") else True
```

(If running the real engine in the unit test is too slow/heavy for CI, mark it `@pytest.mark.slow` and additionally add a `monkeypatch`-based test that stubs `PythonEngine.run_in_memory` to return canned `yield_biomass`/`ssb`/`mortality` frames and asserts the sweep assembles curves + forces the output flags + applies `fishing_override`. Keep at least one stubbed fast test in the default suite.)

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py -k sweep_end_to_end -v`
Expected: FAIL (`compute_model_reference_points` not defined).

- [ ] **Step 3: Implement** — add to `osmose/validation/fmsy_sweep.py`. Reuse the calibration ProcessPool pattern (`osmose/calibration/problem.py:347` builds a `ProcessPoolExecutor`; `:441` runs `PythonEngine().run_in_memory(cfg, seed=run_id)`). The realized-F extraction reuses the stock-status logic on the in-memory `results.mortality(sp)` frame (verify its column shape — the `("F", stage)` MultiIndex; reuse `osmose/validation/stock_status._EXPLOITABLE` + `fisheries.annual_by_year(..., how="sum")`).

```python
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from osmose.engine import PythonEngine
from osmose.validation import fisheries as fis

_DEFAULT_GRID = np.linspace(0.0, 2.0, 7)
_FORCE_OUTPUTS = {"output.ssb.enabled": "true", "output.yield.biomass.enabled": "true"}


def equilibrium_mean(df, sp, window_years):
    """Trailing-window mean of a wide Time+species frame; not_converged if last vs prior window differ."""
    if sp not in df.columns:
        return 0.0, True
    s = df[[c for c in ("Time", sp) if c in df.columns]].copy()
    by_year = fis.annual_by_year(s[sp].to_numpy(), s["Time"].to_numpy(), how="mean")
    years = sorted(by_year)
    vals = [by_year[y] for y in years]
    if len(vals) < 2 * window_years:
        return (float(np.mean(vals)) if vals else 0.0), True
    last = np.mean(vals[-window_years:])
    prior = np.mean(vals[-2 * window_years : -window_years])
    not_conv = abs(last - prior) > 0.05 * (abs(prior) + 1e-9)
    return float(last), bool(not_conv)


def realized_exploited_f(results, sp, window_years):
    """Realized annual exploited-stage F from the in-memory mortalityRate (matches the page basis)."""
    from osmose.validation.stock_status import _EXPLOITABLE
    try:
        df = results.mortality(sp)
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return 0.0
    time = df.iloc[:, 0]
    per_stage = {
        s: fis.annual_by_year(df[("F", s)].to_numpy(), time.to_numpy(), how="sum")
        for s in _EXPLOITABLE if ("F", s) in df.columns
    }
    fished = {s: d for s, d in per_stage.items() if sum(d.values()) > 0}
    if not fished:
        return 0.0
    stage = max(fished, key=lambda s: sum(fished[s].values()))
    years = sorted(fished[stage])[-window_years:]
    return float(np.mean([fished[stage][y] for y in years])) if years else 0.0


def _run_one(args):
    base_config, override_key, f_val, seed, sp_name, window_years = args
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass
    cfg = dict(base_config)
    cfg.update(_FORCE_OUTPUTS)
    cfg[override_key] = str(f_val)
    res = PythonEngine().run_in_memory(cfg, seed=seed)
    y, _ = equilibrium_mean(res.yield_biomass(), sp_name, window_years)
    b, nc = equilibrium_mean(res.ssb(), sp_name, window_years)
    fr = realized_exploited_f(res, sp_name, window_years)
    return (f_val, fr, y, b, nc)


def run_yield_f_sweep(base_config, config, species_list, *, grid, n_years, replicates,
                      window_years, max_workers, seed0=0):
    base = dict(base_config)
    base["simulation.time.nyear"] = str(n_years)
    tasks = []
    meta = []
    for sp_idx, sp_name in species_list:
        key, _ = fishing_override(base, config, sp_idx)
        # no-op-trap guard: the override must actually move fishing_rate[sp_idx]
        probe = dict(base); probe[key] = str(float(config.fishing_rate[sp_idx]) + 1.0)
        if EngineConfig.from_dict(probe).fishing_rate[sp_idx] == config.fishing_rate[sp_idx]:
            raise RuntimeError(f"override key {key!r} does not move fishing_rate[{sp_idx}]")
        for f_val in grid:
            for r in range(replicates):
                tasks.append((base, key, float(f_val), seed0 + r, sp_name, window_years))
                meta.append((sp_name, float(f_val)))
    results = [None] * len(tasks)
    workers = max_workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(_run_one, tasks)):
            results[i] = out
    # group by (species, f) → mean over replicates
    curves: dict[str, dict[float, list]] = {}
    for (sp_name, f_val), (fv, fr, y, b, nc) in zip(meta, results):
        curves.setdefault(sp_name, {}).setdefault(fv, []).append((fr, y, b, nc))
    out: dict[str, list[SweepPoint]] = {}
    for sp_name, byf in curves.items():
        pts = []
        for fv, reps in sorted(byf.items()):
            fr = float(np.mean([r[0] for r in reps]))
            y = float(np.mean([r[1] for r in reps]))
            b = float(np.mean([r[2] for r in reps]))
            nc = any(r[3] for r in reps)
            pts.append(SweepPoint(sp_name, fv, fr, y, b, nc))
        out[sp_name] = pts
    return out


def compute_model_reference_points(base_config, *, grid=None, n_years=None, replicates=3,
                                   window_years=10, max_workers=None):
    config = EngineConfig.from_dict(dict(base_config))
    grid = _DEFAULT_GRID if grid is None else np.asarray(grid, dtype=float)
    n_years = max(config.n_years if hasattr(config, "n_years") else int(base_config.get(
        "simulation.time.nyear", "30")), 30) if n_years is None else n_years
    species_list = list(enumerate(config.species_names))
    curves = run_yield_f_sweep(base_config, config, species_list, grid=grid, n_years=n_years,
                               replicates=replicates, window_years=window_years,
                               max_workers=max_workers)
    refs = derive_reference_points(curves)
    for sp, rp in refs.items():
        rp.curve = curves.get(sp, [])  # attach for the CLI/debug
    return refs
```

Verify against source before relying on it: `config.n_years` field name (else use `simulation.time.nyear`); `results.mortality(sp)`'s real column shape (the `("F", stage)` MultiIndex); that `_FORCE_OUTPUTS` keys are valid (they are — `osmose/schema/output.py`). Handle a `SharedFisheryError`/`ValueError` from `fishing_override` by skipping that species with a caveat rather than crashing the batch.

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py -k sweep -v`
Expected: PASS (the stubbed fast test always; the real-engine one may be `slow`-marked).

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fmsy_sweep.py tests/test_fmsy_sweep.py
git commit -m "feat(fmsy): yield-vs-F sweep runner + compute_model_reference_points (ProcessPool, realized-F)"
```

---

### Task 4: CLI `scripts/compute_model_reference_points.py`

**Files:**
- Create: `scripts/compute_model_reference_points.py`
- Test: `tests/test_fmsy_sweep.py`

**Interfaces:**
- Consumes: `compute_model_reference_points` (Task 3), `OsmoseConfigReader`.
- Produces: a `data/<ecosystem>/reference/fisheries_model_reference_points.json` sidecar (with `_meta` + per-species `fmsy/bmsy/b0/blim/fmsy_at_boundary/multi_peak`); a `write_model_sidecar(refs, out_path, meta)` helper (importable + unit-tested).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_fmsy_sweep.py
import json


def test_write_model_sidecar(tmp_path):
    from osmose.validation.fmsy_sweep import ModelReferencePoint
    from scripts.compute_model_reference_points import write_model_sidecar
    refs = {"cod": ModelReferencePoint("cod", fmsy=0.3, bmsy=118000.0, b0=410000.0,
                                       blim=82000.0, fmsy_at_boundary=False, multi_peak=False)}
    out = tmp_path / "fisheries_model_reference_points.json"
    write_model_sidecar(refs, out, meta={"grid": [0.0, 1.0], "replicates": 3})
    d = json.loads(out.read_text())
    assert d["cod"]["fmsy"] == 0.3 and d["cod"]["blim"] == 82000.0
    assert d["_meta"]["replicates"] == 3
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py -k sidecar -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `scripts/compute_model_reference_points.py`:

```python
"""CLI: compute model-internal fishery reference points (Fmsy/Bmsy/Blim) by a yield-vs-F sweep."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.validation.fmsy_sweep import compute_model_reference_points


def write_model_sidecar(refs, out_path: Path, meta: dict) -> None:
    payload = {"_meta": meta}
    for sp, rp in refs.items():
        payload[sp] = {
            "fmsy": rp.fmsy, "bmsy": rp.bmsy, "b0": rp.b0, "blim": rp.blim,
            "fmsy_at_boundary": rp.fmsy_at_boundary, "multi_peak": rp.multi_peak,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to the master parameters file")
    ap.add_argument("--grid", type=float, nargs="*", default=None, help="absolute F grid")
    ap.add_argument("--n-years", type=int, default=None)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cfg_path = Path(a.config)
    base = dict(OsmoseConfigReader().read(str(cfg_path)))
    base["__config_dir__"] = str(cfg_path.parent)
    grid = np.asarray(a.grid, dtype=float) if a.grid else None
    n_grid = len(grid) if grid is not None else 7
    from osmose.engine.config import EngineConfig
    n_sp = EngineConfig.from_dict(dict(base)).n_species
    n_years = a.n_years or max(int(base.get("simulation.time.nyear", "30")), 30)
    print(f"Sweep: {n_sp} species x {n_grid} F x {a.replicates} reps = "
          f"{n_sp * n_grid * a.replicates} runs of {n_years} yr each (this is offline; expect "
          f"tens of minutes to hours).")
    t0 = time.time()
    refs = compute_model_reference_points(base, grid=grid, n_years=a.n_years,
                                          replicates=a.replicates, max_workers=a.workers)
    ecosystem = cfg_path.parent.name
    out = Path(a.out) if a.out else (
        Path("data") / ecosystem / "reference" / "fisheries_model_reference_points.json")
    meta = {"grid": (grid.tolist() if grid is not None else None), "n_years": n_years,
            "replicates": a.replicates, "window_years": 10, "f_basis": "realized_exploited_stage"}
    write_model_sidecar(refs, out, meta)
    print(f"Wrote {out} ({len(refs)} species) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py -k sidecar -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/compute_model_reference_points.py tests/test_fmsy_sweep.py
git commit -m "feat(fmsy): CLI to compute + write the model reference-point sidecar"
```

---

### Task 5: read the model sidecar in `fisheries_reference.py` (precedence user > model > ICES)

**Files:**
- Modify: `osmose/validation/fisheries_reference.py`
- Test: `tests/test_fisheries_reference.py`

**Interfaces:**
- Consumes: the model sidecar (Task 4).
- Produces: `load_reference_points` now fills `fmsy`/`bmsy` from the model where present (`source="model"`, `b_ref_kind="bmsy_model"`); `b_ref_label` conditional.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_fisheries_reference.py
import json
from pathlib import Path

from osmose.validation import fisheries_reference as fr

ICES = Path("data/baltic/reference/ices_snapshots")


def _write_model(ref_dir, payload):
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "fisheries_model_reference_points.json").write_text(json.dumps(payload))


def test_model_fills_fmsy_and_bmsy(tmp_path):
    _write_model(tmp_path, {"sprat": {"fmsy": 0.5, "bmsy": 600000.0}})
    refs, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    assert refs["sprat"].fmsy == 0.5 and refs["sprat"].bmsy == 600000.0
    assert refs["sprat"].b_ref_kind == "bmsy_model"
    assert refs["sprat"].b_ref_label == "Bmsy [model]"
    assert "model" in refs["sprat"].source


def test_precedence_user_over_model_over_ices(tmp_path):
    _write_model(tmp_path, {"sprat": {"fmsy": 0.5, "bmsy": 600000.0}})
    (tmp_path / "fisheries_reference_points.json").write_text(
        json.dumps({"sprat": {"bmsy": 999000.0}}))  # user Bmsy wins
    refs, _ = fr.load_reference_points(tmp_path, ["sprat"], ices_snapshot_dir=ICES)
    assert refs["sprat"].bmsy == 999000.0 and refs["sprat"].b_ref_kind == "bmsy_user"
    assert refs["sprat"].fmsy == 0.5  # model Fmsy kept (no user/ICES override of fmsy here? user none)
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fisheries_reference.py -k "model or precedence" -v`
Expected: FAIL.

- [ ] **Step 3: Implement** — in `osmose/validation/fisheries_reference.py`: (a) add `"bmsy_model"` handling; (b) make `b_ref_label` conditional:

```python
    @property
    def b_ref_label(self) -> str:
        return "Bmsy [model]" if self.b_ref_kind == "bmsy_model" else "Bmsy [user]"
```

(c) in `load_reference_points`, after the ICES auto-fill and BEFORE the user-file merge, read the model sidecar and apply it (model overrides ICES; the subsequent user merge overrides model):

```python
    model_path = ref_dir / "fisheries_model_reference_points.json"
    model = json.loads(model_path.read_text()) if model_path.exists() else {}
    for sp in species_list:
        m = model.get(sp, {})
        if _float(m.get("fmsy")) is not None:
            rp = refs[sp]
            rp.fmsy = _float(m["fmsy"])
            rp.source = "model" if rp.fmsy_stock is None else "mixed"
        if _float(m.get("bmsy")) is not None:
            refs[sp].bmsy = _float(m["bmsy"])
            refs[sp].b_ref_kind = "bmsy_model"
```

Then the existing user-file loop (which sets `bmsy` + `b_ref_kind="bmsy_user"`) runs AFTER, so a user `bmsy` overrides the model one. Verify the exact ordering in the current `load_reference_points` and place the model block so precedence is user > model > ICES.

- [ ] **Step 4: Run test — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fisheries_reference.py -v`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fisheries_reference.py tests/test_fisheries_reference.py
git commit -m "feat(fmsy): load model reference sidecar (precedence user>model>ICES; conditional label)"
```

---

### Task 6: surface the source in the Fisheries page

**Files:**
- Modify: `ui/pages/fisheries.py`
- Test: `tests/test_ui_fisheries.py`

**Interfaces:**
- Consumes: the model-aware `load_reference_points` (Task 5).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ui_fisheries.py
def test_view_reports_model_source(tmp_path, monkeypatch):
    import ui.pages.fisheries as page
    from osmose.validation.fisheries_reference import ReferencePoint
    from osmose.validation.stock_status import StockStatus
    monkeypatch.setattr(page, "load_reference_points", lambda *a, **k: (
        {"cod": ReferencePoint(species="cod", fmsy=0.3, bmsy=100.0,
                               b_ref_kind="bmsy_model", source="model")}, []))
    monkeypatch.setattr(page, "compute_stock_status", lambda *a, **k: [
        StockStatus("cod", [0], [1.2], [0.5], "Bmsy [model]", latest_quadrant="green")])
    cfg = type("C", (), {"species_names": ["cod"]})()
    view = page.build_fisheries_view(object(), cfg, "baltic")
    assert view["kobe_ready"] is True
    assert any(getattr(r, "source", "") == "model" for r in
               page.load_reference_points(None, ["cod"])[0].values())
```

- [ ] **Step 2: Run test — verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ui_fisheries.py -k model_source -v`
Expected: FAIL (or trivially pass if build_fisheries_view already surfaces it) — adjust the page so the reference-table render includes a `source` column and the methodology note mentions the model sidecar + how to regenerate it via the CLI.

- [ ] **Step 3: Implement** — in `ui/pages/fisheries.py`, add a `source` column to the reference-point table render (read `rp.source` per species) and a one-line note: "Reference points may be model-internal (run `scripts/compute_model_reference_points.py --config <cfg>`), ICES-auto-filled, or user-entered; user values override model, model overrides ICES." Keep `build_fisheries_view` unchanged in contract (it already returns `statuses`/`kobe_ready` from `load_reference_points` which now includes model values).

- [ ] **Step 4: Run test + app import — verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ui_fisheries.py -v && PYTHONPATH=. .venv/bin/python -c "import app"`
Expected: PASS + clean import.

- [ ] **Step 5: Full sweep + lint + commit**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_fmsy_sweep.py tests/test_fisheries_reference.py tests/test_ui_fisheries.py tests/test_validation_stock_status.py tests/test_engine_parity.py -q`
Expected: all pass (parity untouched).
Run: `.venv/bin/ruff check osmose/ ui/ scripts/ tests/ && .venv/bin/ruff format --check osmose/validation/fmsy_sweep.py scripts/compute_model_reference_points.py`
Expected: clean.
```bash
git add ui/pages/fisheries.py tests/test_ui_fisheries.py
git commit -m "feat(ui): show reference-point source (model/ICES/user) on the Fisheries page"
```

---

## Notes for the executor

- **THE no-op trap:** the override key differs by config mode. `run_yield_f_sweep` asserts the override moves `fishing_rate[i]` before running — never delete that guard. Both bundled configs are v4 fisheries-mode (`fisheries.rate.base.fsh{j}`).
- **Readers:** `results.yield_biomass` (NOT `fishery_yield`), `results.ssb` (force `output.ssb.enabled`), `results.mortality` (realized F). Verify `results.mortality()`'s in-memory `("F", stage)` column shape before wiring the realized-F helper.
- **Parallelization:** `numba.set_num_threads(1)` inside each ProcessPool worker; the engine's mortality kernel is already `prange`. Don't run threads — use processes.
- **Cost:** real-engine sweep tests must be tiny (few years, small grid, 1 replicate, 2 workers) or `slow`-marked; keep a stubbed fast test in the default suite. The full sweep is an offline CLI batch (tens of min–hours), not a unit test.
- **Parity:** the sweep only varies fishing + output-flag config; do not touch engine dynamics. EEC/BoB parity must stay green.
- The realized-F extraction mirrors `osmose/validation/stock_status.py` (`_EXPLOITABLE`, `annual_by_year(how="sum")`) — reuse, don't re-derive.
