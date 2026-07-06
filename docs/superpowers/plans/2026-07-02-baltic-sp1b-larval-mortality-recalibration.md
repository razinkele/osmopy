# SP1b — Baltic cod larval-mortality recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recalibrate cod's larval mortality rate so that enabling the SP1 spatial egg-survival term leaves the mean cod biomass unchanged vs SP1-off (mean-neutral), isolating SP1 as a purely spatial intervention.

**Architecture:** A pure 1-D root-finder (`solve_larva_rate`: coarse grid scan over `[0, d0]` → sign-change feasibility gate → bisection) in `osmose/calibration/larva_recal.py`, driven by an injected `run_mean_on(rate)` closure so it is engine-free and stub-testable. A CLI runs it against the real Baltic engine to obtain the recalibrated rate, which is hand-frozen into a `RECAL_RATE` constant consumed by an `sp1_on_config` overlay helper. A mean-neutrality test acts as the drift guard. The Baltic default config is never modified — SP1 stays inert-by-default.

**Tech Stack:** Python 3.12, NumPy, xarray, pytest, numba (thread pinning), the existing `osmose.engine.PythonEngine` and `osmose.config.OsmoseConfigReader`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-02-baltic-sp1b-larval-mortality-recalibration-design.md`.
- Target: mean-neutral vs the SP1-off baseline; **not** ICES-absolute, **not** overshoot-gating.
- Baseline = mean cod biomass over years index `[3:15]`, finite & >0, seed 0 (identical to `scripts/rv_field_diagnostic.py:_mean_cod`). Tolerance `tol = 0.02`.
- The knob is the **resolved per-cohort** `mortality.additional.larva.rate.sp0` (base file stores 360/yr; reader ÷ndt=24 → engine reads 15.0; `d0 = 15.0`). Overriding it on an already-read in-memory cfg dict sets the resolved value directly (no re-migration).
- **Determinism is required** for a well-defined `f(d)`: config keys `movement.randomseed.fixed=true` + `stochastic.mortality.randomseed.fixed=true` (`osmose/engine/config.py:2054-2056`), AND runtime `numba.set_num_threads(1)` (precedent `osmose/validation/fmsy_sweep.py:244`) — set by **both** the CLI and the neutrality test.
- Only cod (sp0) is touched. Baltic runs use `simulation.time.nyear=15`. Real-engine tests run FOREGROUND with a generous timeout; never background them.
- Lint/format gate: `.venv/bin/ruff check osmose/ tests/` and `.venv/bin/ruff format --check`; types: `.venv/bin/pyright <changed files>`.
- Work on branch `baltic-rv-spatial-egg-survival` (SP1b builds on SP1; PR #97).

---

### Task 1: Solver core — `solve_larva_rate` + `RecalResult`

**Files:**
- Create: `osmose/calibration/larva_recal.py`
- Test: `tests/test_sp1b_recalibration.py`

**Interfaces:**
- Produces: `RecalResult` dataclass and
  `solve_larva_rate(baseline: float, run_mean_on: Callable[[float], float], *, grid_points: Sequence[float], tol: float = 0.02, max_iter: int = 20) -> RecalResult`.
  `run_mean_on(rate)` returns the SP1-on mean cod biomass for that larva rate. `d0` is `max(grid_points)`.

- [ ] **Step 1: Write the failing tests (pure, stub-driven — no engine)**

Create `tests/test_sp1b_recalibration.py`:

Only import what the tests actually use — ruff's default `F401` fails the lint gate on unused imports.
`numpy`/`pytest`/`RecalResult` are NOT used by any Task 1 test (`pytest` is added in Task 5 where
`pytest.skip` first appears; later tasks extend this one import line).

```python
import math

from osmose.calibration.larva_recal import solve_larva_rate

GRID = [0.0, 5.0, 10.0, 15.0]


def test_solve_bisects_to_interior_root():
    # mean decreases with rate: mean(d) = 100 - 4d; baseline 70 -> root at d=7.5.
    # grid f = mean-baseline = [30, 10, -10, -30] -> exactly one crossing in (5, 10).
    r = solve_larva_rate(70.0, lambda d: 100.0 - 4.0 * d, grid_points=GRID, tol=1e-4)
    assert r.feasible and r.converged
    assert r.rate is not None and abs(r.rate - 7.5) < 0.05
    assert abs(r.mean_on - 70.0) / 70.0 <= 1e-4


def test_solve_near_zero_shortcircuit_returns_grid_point():
    # grid includes the exact root 7.5 -> short-circuit, iters=0.
    r = solve_larva_rate(70.0, lambda d: 100.0 - 4.0 * d, grid_points=[0.0, 7.5, 15.0], tol=0.02)
    assert r.feasible and r.converged and r.iters == 0
    assert r.rate == 7.5


def test_solve_d0_already_within_tol_means_no_recalibration():
    # SP1 barely moved the mean: mean(d0=15)=40 == baseline 40 -> near-zero hit at the last
    # grid point -> rate == d0, no recalibration. (mean(0)=100, mean(7.5)=70 are far off.)
    r = solve_larva_rate(40.0, lambda d: 100.0 - 4.0 * d, grid_points=[0.0, 7.5, 15.0], tol=0.02)
    assert r.feasible and r.converged and r.rate == 15.0


def test_solve_infeasible_zero_crossings():
    # every grid mean is far below baseline -> baseline unreachable -> feasible=False.
    r = solve_larva_rate(200.0, lambda d: 50.0 - d, grid_points=GRID, tol=0.02)
    assert not r.feasible and r.rate is None and "0 sign change" in r.message


def test_solve_infeasible_multiple_crossings():
    # f = 30*cos(d), baseline 100: grid values give two sign changes -> ambiguous.
    r = solve_larva_rate(
        100.0, lambda d: 100.0 + 30.0 * math.cos(d), grid_points=[1.0, 2.5, 4.0, 5.5], tol=0.02
    )
    assert not r.feasible and r.rate is None and "2 sign change" in r.message


def test_solve_max_iter_not_converged_reports_best():
    # baseline 71 -> root at d=7.25 (NON-dyadic, so bisection midpoints never hit it exactly);
    # impossibly tight tol + tiny max_iter -> feasible bracket but converged=False.
    r = solve_larva_rate(
        71.0, lambda d: 100.0 - 4.0 * d, grid_points=GRID, tol=1e-12, max_iter=2
    )
    assert r.feasible and not r.converged and r.rate is not None
    assert 5.0 < r.rate < 10.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sp1b_recalibration.py -q`
Expected: FAIL / collection error (`osmose.calibration.larva_recal` does not exist).

- [ ] **Step 3: Implement the solver**

Create `osmose/calibration/larva_recal.py`:

```python
"""SP1b — mean-neutral recalibration of cod larval mortality for the SP1 spatial term.

Pure 1-D root finder: coarse grid scan over [0, d0] -> sign-change feasibility gate ->
bisection. Engine-free (the caller injects run_mean_on); see the SP1b design spec.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class RecalResult:
    rate: float | None            # solved larva rate, or None when infeasible
    baseline: float               # SP1-off mean cod biomass
    mean_on: float | None         # SP1-on mean at `rate` (None when infeasible)
    rel_err: float | None         # |mean_on - baseline| / baseline
    converged: bool               # reached tol
    feasible: bool                # exactly one sign change (or a near-zero grid hit)
    grid: list[tuple[float, float]]  # (rate, mean) at each grid point
    iters: int                    # bisection iterations used
    message: str


def solve_larva_rate(
    baseline: float,
    run_mean_on: Callable[[float], float],
    *,
    grid_points: Sequence[float],
    tol: float = 0.02,
    max_iter: int = 20,
) -> RecalResult:
    """Find the larva rate whose SP1-on mean cod biomass matches `baseline` within `tol`.

    No monotonicity is assumed: the grid measures the shape. Exactly one sign change of
    f(d) = run_mean_on(d) - baseline is required for a solve; zero or >=2 -> infeasible.
    A grid point already within tol short-circuits (rate == max grid point == "no change").
    """
    grid = sorted({float(g) for g in grid_points})
    evals = [(d, float(run_mean_on(d))) for d in grid]

    def rel(m: float) -> float:
        return abs(m - baseline) / baseline

    # (b) near-zero short-circuit BEFORE sign counting (makes f=0 well-posed).
    for d, m in evals:
        if rel(m) <= tol:
            return RecalResult(d, baseline, m, rel(m), True, True, evals, 0,
                               f"grid point {d:.4g} already within tol")

    # (c) feasibility gate: count sign changes of f = m - baseline.
    fs = [(d, m - baseline) for d, m in evals]
    crossings = [(fs[i], fs[i + 1]) for i in range(len(fs) - 1) if fs[i][1] * fs[i + 1][1] < 0.0]
    if len(crossings) != 1:
        return RecalResult(None, baseline, None, None, False, False, evals, 0,
                           f"{len(crossings)} sign changes on grid (need exactly 1); "
                           "baseline unreachable or ambiguous/multi-root")

    # (d) bisection on the single sign-changing sub-interval.
    (a, fa), (b, _fb) = crossings[0]
    iters = 0
    while iters < max_iter:
        mid = 0.5 * (a + b)
        m_mid = float(run_mean_on(mid))
        f_mid = m_mid - baseline
        iters += 1
        if rel(m_mid) <= tol:
            return RecalResult(mid, baseline, m_mid, rel(m_mid), True, True, evals, iters,
                               "converged")
        if fa * f_mid < 0.0:
            b = mid
        else:
            a, fa = mid, f_mid

    mid = 0.5 * (a + b)
    m_mid = float(run_mean_on(mid))
    return RecalResult(mid, baseline, m_mid, rel(m_mid), False, True, evals, iters,
                       f"max_iter={max_iter} reached, rel_err={rel(m_mid):.3f} > tol={tol}")
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sp1b_recalibration.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint/types, commit**

```bash
cd /home/razinka/osmose/osmose-python && .venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format osmose/calibration/larva_recal.py tests/test_sp1b_recalibration.py && .venv/bin/pyright osmose/calibration/larva_recal.py
git -C /home/razinka/osmose/osmose-python add osmose/calibration/larva_recal.py tests/test_sp1b_recalibration.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: SP1b solver core (grid scan + feasibility gate + bisection)"
```

---

### Task 2: Analytical first-guess — `e_clip_first_guess`

**Files:**
- Modify: `osmose/calibration/larva_recal.py`
- Test: `tests/test_sp1b_recalibration.py`

**Interfaces:**
- Produces: `e_clip_first_guess(field_path: str | Path, spawn_path: str | Path, d0: float) -> tuple[float, float]` returning `(d1_analytical, e_clip)`, where `d1 = clip(d0 + ln(E[clip]), 0, d0)` and `E[clip]` is the presence-weighted mean of `clip(RV_timemean/RV_ref)` over `cod_spawning` cells. First-guess only (seeds the grid); the empirical solve makes the final rate weighting-agnostic.

- [ ] **Step 1: Write the failing test (uses the real SP1 field)**

First, **edit the top import line** of `tests/test_sp1b_recalibration.py` to add the new symbol
(all test imports stay at the top — appending an `import` mid-file trips ruff E402):

```python
from osmose.calibration.larva_recal import e_clip_first_guess, solve_larva_rate
```

Then append the module-level constants + test (constants are not imports, so mid-file is fine):

```python
SP_FIELD = "data/baltic/forcing/baltic_rv_field.nc"
SPAWN = "data/baltic/maps/cod_spawning.csv"


def test_e_clip_first_guess_bounds():
    d1, e_clip = e_clip_first_guess(SP_FIELD, SPAWN, d0=15.0)
    assert 0.0 < e_clip < 1.0          # some but not all viable
    assert 0.0 <= d1 <= 15.0           # a valid rate inside the bracket
    # d1 = d0 + ln(e_clip); ln(e_clip) < 0 so d1 < d0
    assert d1 < 15.0
    assert abs(d1 - max(0.0, 15.0 + math.log(e_clip))) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sp1b_recalibration.py::test_e_clip_first_guess_bounds -q`
Expected: collection error (ImportError: cannot import name `e_clip_first_guess`) — the whole file
fails to collect until Step 3 implements it. That is the intended "red" for this step.

- [ ] **Step 3: Implement**

Add to `osmose/calibration/larva_recal.py` (add `import math`, `from pathlib import Path`, `import numpy as np` at the top import block):

```python
def e_clip_first_guess(
    field_path: str | Path, spawn_path: str | Path, d0: float
) -> tuple[float, float]:
    """Analytical first-guess rate d1 = clip(d0 + ln E[clip], 0, d0), and E[clip].

    E[clip] = presence-weighted mean of clip(RV_timemean(cell)/RV_ref) over cod_spawning
    cells. This restores the *instantaneous* egg-weighted average survival; it is only a
    grid seed (the empirical solve finds the true equilibrium root, which — because the
    biomass effect is buffered by density dependence — is usually much closer to d0).
    """
    import xarray as xr

    da = xr.open_dataset(field_path)["reproductive_volume"]
    rv = da.values.mean(axis=0)  # time-mean (nlat, nlon), north-first
    ref = float(da.attrs["RV_ref"])
    spawn = np.flipud(np.genfromtxt(spawn_path, delimiter=";")) > 0
    e_clip = float(np.clip(rv[spawn] / ref, 0.0, 1.0).mean())
    d1 = d0 + math.log(e_clip) if e_clip > 0.0 else 0.0
    return max(0.0, min(d0, d1)), e_clip
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sp1b_recalibration.py::test_e_clip_first_guess_bounds -q`
Expected: PASS.

- [ ] **Step 5: Lint/types, commit**

```bash
cd /home/razinka/osmose/osmose-python && .venv/bin/ruff check osmose/ tests/ && .venv/bin/pyright osmose/calibration/larva_recal.py
git -C /home/razinka/osmose/osmose-python add osmose/calibration/larva_recal.py tests/test_sp1b_recalibration.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: SP1b analytical first-guess e_clip_first_guess"
```

---

### Task 3: Overlay helpers — `with_determinism`, `sp1_on_config`, `RECAL_RATE`

**Files:**
- Modify: `osmose/calibration/larva_recal.py`
- Modify: `tests/test_rv_spatial_egg_survival.py` (replace the test-local `_baltic_gate_cfg`)
- Test: `tests/test_sp1b_recalibration.py`

**Interfaces:**
- Produces:
  - `RECAL_RATE: float | None` (module constant, initially `None`; hand-set in Task 4).
  - `with_determinism(cfg: dict[str, str]) -> dict[str, str]` — returns a copy with the two fixed-seed keys set to `"true"`.
  - `sp1_on_config(base_cfg: dict[str, str], field_path: str | Path, *, larva_rate: float | None | _UseRecal = _USE_RECAL) -> dict[str, str]` — SP1 flags + determinism keys + (recalibrated rate unless `None`). Default reads the *current* module `RECAL_RATE` at call time. `_UseRecal` is a dedicated sentinel class (typed, so `isinstance` narrows the union and pyright accepts `float(rate)`).

- [ ] **Step 1: Write the failing tests**

**Add the new imports to the top of `tests/test_sp1b_recalibration.py`** (keep imports at top for
E402): add `from osmose.calibration import larva_recal` and extend the existing
`from osmose.calibration.larva_recal import ...` line to include `sp1_on_config, with_determinism`:

```python
from osmose.calibration import larva_recal
from osmose.calibration.larva_recal import (
    e_clip_first_guess,
    solve_larva_rate,
    sp1_on_config,
    with_determinism,
)
```

Then append the constants + tests:

```python
RATE_KEY = "mortality.additional.larva.rate.sp0"
DET_KEYS = ("movement.randomseed.fixed", "stochastic.mortality.randomseed.fixed")


def test_with_determinism_sets_both_keys_without_mutating():
    base = {"a": "1"}
    out = with_determinism(base)
    assert all(out[k] == "true" for k in DET_KEYS)
    assert "a" in out and base == {"a": "1"}  # original untouched


def test_sp1_on_config_flags_and_determinism():
    cfg = sp1_on_config({"x": "y"}, SP_FIELD, larva_rate=None)
    assert cfg["reproduction.rv.spatial.enabled"] == "true"
    assert cfg["reproduction.rv.spatial.field.file"] == SP_FIELD
    assert cfg["reproduction.rv.spatial.species.enabled.sp0"] == "true"
    assert all(cfg[k] == "true" for k in DET_KEYS)


def test_sp1_on_config_none_omits_rate_key():
    cfg = sp1_on_config({}, SP_FIELD, larva_rate=None)
    assert RATE_KEY not in cfg  # infeasible path: base d0 stands


def test_sp1_on_config_value_sets_rate_key():
    cfg = sp1_on_config({}, SP_FIELD, larva_rate=12.5)
    assert float(cfg[RATE_KEY]) == 12.5


def test_sp1_on_config_default_reads_current_recal_rate(monkeypatch):
    monkeypatch.setattr(larva_recal, "RECAL_RATE", 9.0)
    cfg = sp1_on_config({}, SP_FIELD)  # default -> current module RECAL_RATE
    assert float(cfg[RATE_KEY]) == 9.0
    monkeypatch.setattr(larva_recal, "RECAL_RATE", None)
    cfg2 = sp1_on_config({}, SP_FIELD)
    assert RATE_KEY not in cfg2
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sp1b_recalibration.py -k "determinism or sp1_on_config" -q`
Expected: collection error (ImportError: cannot import name `sp1_on_config`) until Step 3 implements
the helpers. That is the intended "red".

- [ ] **Step 3: Implement**

Add to `osmose/calibration/larva_recal.py`:

```python
RECAL_RATE: float | None = None  # set from scripts/recalibrate_sp1b.py output (SP1b Task 4)


class _UseRecal:
    """Sentinel type for sp1_on_config's default (typed so isinstance narrows the union)."""


_USE_RECAL = _UseRecal()  # read the current module RECAL_RATE at call time

_DET_KEYS = {
    "movement.randomseed.fixed": "true",
    "stochastic.mortality.randomseed.fixed": "true",
}


def with_determinism(cfg: dict[str, str]) -> dict[str, str]:
    """Return a copy of cfg with the two fixed-seed keys set (required for a reproducible
    solve; the runtime numba single-thread pin is set separately by the caller)."""
    return {**cfg, **_DET_KEYS}


def sp1_on_config(
    base_cfg: dict[str, str],
    field_path: str | Path,
    *,
    larva_rate: float | None | _UseRecal = _USE_RECAL,
) -> dict[str, str]:
    """SP1-on config: SP1 flags + determinism keys + recalibrated cod larva rate.

    larva_rate=None omits the rate key (base d0 stands — the infeasible path); a float sets
    it; the default reads the current module RECAL_RATE at call time (so freezing RECAL_RATE
    in Task 4 takes effect without editing this default).
    """
    rate: float | None = RECAL_RATE if isinstance(larva_rate, _UseRecal) else larva_rate
    cfg = with_determinism(base_cfg)
    cfg["reproduction.rv.spatial.enabled"] = "true"
    cfg["reproduction.rv.spatial.field.file"] = str(field_path)
    cfg["reproduction.rv.spatial.species.enabled.sp0"] = "true"
    if rate is not None:
        cfg["mortality.additional.larva.rate.sp0"] = repr(float(rate))  # resolved per-cohort value
    return cfg
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sp1b_recalibration.py -k "determinism or sp1_on_config" -q`
Expected: PASS.

- [ ] **Step 5: Replace the test-local `_baltic_gate_cfg` in the SP1 tests**

In `tests/test_rv_spatial_egg_survival.py`, delete the local `_baltic_gate_cfg` function and route its one caller through the production helper. Add near the other imports:

```python
from osmose.calibration.larva_recal import sp1_on_config
```

Replace the `_baltic_gate_cfg` definition and its use in `test_spatial_on_changes_cod`:

```python
def test_spatial_on_changes_cod():
    off = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    on_cfg = sp1_on_config(_baltic_cfg(), SP_FIELD, larva_rate=None)  # SP1 on, no recal
    on = PythonEngine().run_in_memory(on_cfg, seed=0).biomass()["cod"].to_numpy()
    assert not np.allclose(off, on)  # the spatial term changes cod
```

(`larva_rate=None` keeps the un-recalibrated rate so this test checks the raw SP1 effect, as before. The added determinism keys on the on-side do not affect the assertion — cod still changes.)

- [ ] **Step 6: Run the SP1 test file to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -q`
Expected: PASS (20 tests; `test_spatial_on_changes_cod` still passes via `sp1_on_config`).

- [ ] **Step 7: Lint/types, commit**

```bash
cd /home/razinka/osmose/osmose-python && .venv/bin/ruff check osmose/ tests/ && .venv/bin/pyright osmose/calibration/larva_recal.py
git -C /home/razinka/osmose/osmose-python add osmose/calibration/larva_recal.py tests/test_sp1b_recalibration.py tests/test_rv_spatial_egg_survival.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: SP1b overlay helpers (with_determinism, sp1_on_config, RECAL_RATE)"
```

---

### Task 4: CLI — run the solve, obtain and freeze `RECAL_RATE`

**Files:**
- Create: `scripts/recalibrate_sp1b.py`
- Modify: `osmose/calibration/larva_recal.py` (add the shared `mean_cod` helper; set `RECAL_RATE` from the CLI output)
- Modify: `scripts/rv_field_diagnostic.py` (replace its local `_mean_cod` with the shared helper — DRY, per the spec's "extract rather than re-derive")

**Interfaces:**
- Produces: `mean_cod(cfg: dict[str, str], *, seed: int = 0) -> float` — years `[3:15]`, finite & >0 mean of cod biomass (the shared window helper). Used by the CLI, the neutrality test, and the SP1 diagnostic.

- [ ] **Step 1: Add the shared `mean_cod` helper (engine-coupled) to the module, and adopt it in the SP1 diagnostic**

Add to `osmose/calibration/larva_recal.py`:

```python
def mean_cod(cfg: dict[str, str], *, seed: int = 0) -> float:
    """Mean cod biomass over years index [3:15] (finite & >0), matching the SP1 diagnostic."""
    from osmose.engine import PythonEngine

    b = PythonEngine().run_in_memory(cfg, seed=seed).biomass()["cod"].to_numpy()
    w = b[3:15]
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.mean())
```

Then make `scripts/rv_field_diagnostic.py` the single-source consumer: delete its local `_mean_cod`
(lines ~21-26), add `from osmose.calibration.larva_recal import mean_cod` to its imports, **and delete
its now-orphaned `from osmose.engine import PythonEngine` import** (it was used only inside `_mean_cod`
— leaving it trips ruff `F401`). Replace its two call sites `off_mean = _mean_cod(base)` /
`on_mean = _mean_cod(on)` with `mean_cod(base)` / `mean_cod(on)`. The window logic is byte-identical,
so the SP1 diagnostic output is unchanged.
(Sanity-check after: `.venv/bin/python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/rv_field_diagnostic.py').read_text())"` parses clean.)

- [ ] **Step 2: Write the CLI**

Create `scripts/recalibrate_sp1b.py`:

```python
#!/usr/bin/env python
"""SP1b — solve cod's larval rate so SP1-on mean cod biomass matches the SP1-off baseline.

Usage: PYTHONPATH=. .venv/bin/python scripts/recalibrate_sp1b.py
Prints the grid, the RecalResult, and a ready-to-paste `RECAL_RATE = ...` line for
osmose/calibration/larva_recal.py. Two-plus 15-yr Baltic runs — foreground only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

from osmose.calibration.larva_recal import (
    e_clip_first_guess,
    mean_cod,
    solve_larva_rate,
    sp1_on_config,
    with_determinism,
)
from osmose.config import OsmoseConfigReader

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
FIELD = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"
SPAWN = ROOT / "data" / "baltic" / "maps" / "cod_spawning.csv"
D0 = 15.0


def _base_cfg() -> dict[str, str]:
    cfg = dict(OsmoseConfigReader().read(str(CONFIG)))
    cfg["simulation.time.nyear"] = "15"
    return cfg


def main() -> int:
    numba.set_num_threads(1)  # runtime determinism pin
    base = _base_cfg()

    # Noise check: under the fixed-seed keys + single thread, f(d) must be reproducible,
    # else the solve chases noise. Require bit-identical repeat means before trusting it.
    off = with_determinism(base)
    baseline = mean_cod(off)
    baseline_again = mean_cod(off)
    if abs(baseline_again - baseline) / baseline > 1e-9:
        print(f"NON-DETERMINISTIC baseline ({baseline} vs {baseline_again}) — determinism pins "
              "not effective; fix before solving.", file=sys.stderr)
        return 1

    d1, e_clip = e_clip_first_guess(FIELD, SPAWN, D0)
    grid = sorted({0.0, D0, *np.linspace(0.0, D0, 5).tolist(), max(0.0, min(D0, d1))})
    print(f"baseline (SP1-off) mean cod = {baseline:.1f}; E[clip]={e_clip:.3f}, "
          f"d1_analytical={d1:.3f}; grid={[round(g, 2) for g in grid]}")

    def run_mean_on(rate: float) -> float:
        return mean_cod(sp1_on_config(base, FIELD, larva_rate=rate))

    res = solve_larva_rate(baseline, run_mean_on, grid_points=grid, tol=0.02)
    print("grid (rate, mean):", [(round(d, 3), round(m, 1)) for d, m in res.grid])
    print(f"result: feasible={res.feasible} converged={res.converged} rate={res.rate} "
          f"mean_on={res.mean_on} rel_err={res.rel_err} iters={res.iters} :: {res.message}")
    if res.feasible and res.rate is not None:
        print(f"\nPASTE into osmose/calibration/larva_recal.py:\nRECAL_RATE = {res.rate!r}  "
              f"# SP1b solved {res.message}")
    else:
        print("\nINFEASIBLE — leave RECAL_RATE = None; record the grid in the diagnostic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Lint/types the new code, then commit the CLI + helper (before running)**

```bash
cd /home/razinka/osmose/osmose-python && .venv/bin/ruff check osmose/ scripts/recalibrate_sp1b.py scripts/rv_field_diagnostic.py && .venv/bin/ruff format osmose/calibration/larva_recal.py scripts/recalibrate_sp1b.py scripts/rv_field_diagnostic.py && .venv/bin/pyright osmose/calibration/larva_recal.py scripts/recalibrate_sp1b.py
git -C /home/razinka/osmose/osmose-python add osmose/calibration/larva_recal.py scripts/recalibrate_sp1b.py scripts/rv_field_diagnostic.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: SP1b recalibration CLI + shared mean_cod helper (DRY with SP1 diagnostic)"
```

- [ ] **Step 4: Run the solve (SLOW — foreground, generous timeout)**

Run: `cd /home/razinka/osmose/osmose-python && PYTHONPATH=. .venv/bin/python scripts/recalibrate_sp1b.py`
Expected: prints the baseline, grid `(rate, mean)` pairs, and a `RecalResult`. Two outcomes are both valid:
- **feasible** → a `RECAL_RATE = <value>` paste line;
- **infeasible** (`feasible=False`) → record the grid; `RECAL_RATE` stays `None` (a legitimate negative result — mean-neutrality unreachable via this knob).

Do NOT tune anything to force feasibility. If the run errors, STOP and report BLOCKED with the traceback rather than guessing.

- [ ] **Step 5: Freeze the result**

If feasible, edit `osmose/calibration/larva_recal.py` and replace `RECAL_RATE: float | None = None` with the printed value, keeping the type annotation:

```python
RECAL_RATE: float | None = <printed value>  # SP1b solved <message>
```

If infeasible, leave `RECAL_RATE = None` and note the grid in the commit message.

- [ ] **Step 6: Commit the frozen constant**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/calibration/larva_recal.py
git -C /home/razinka/osmose/osmose-python commit -m "chore: freeze SP1b RECAL_RATE from solve (<feasible|infeasible: grid ...>)"
```

---

### Task 5: Mean-neutrality drift-guard test + diagnostic

**Files:**
- Modify: `tests/test_sp1b_recalibration.py`
- Create: `scripts/sp1b_diagnostic.py`
- Create (generated): `docs/diagnostics/sp1b_recalibration.md`

**Interfaces:**
- Consumes: `RECAL_RATE`, `sp1_on_config`, `with_determinism`, `mean_cod` (Tasks 3–4); the SP1 field + `cod_spawning` map; `OsmoseResults.biomass("cod")`.

- [ ] **Step 1: Write the mean-neutrality test**

**Add the new imports to the top of `tests/test_sp1b_recalibration.py`** (E402): add `import numba`
and `import pytest` to the third-party group (`pytest` is first used here, by `pytest.skip`),
`from osmose.config import OsmoseConfigReader`, and extend the
`from osmose.calibration.larva_recal import (...)` line to include `RECAL_RATE, mean_cod`. Then append
the constant + helper + test:

```python
BALTIC = "data/baltic/baltic_all-parameters.csv"


def _baltic_15yr():
    cfg = dict(OsmoseConfigReader().read(BALTIC))
    cfg["simulation.time.nyear"] = "15"
    return cfg


def test_sp1b_mean_neutral_drift_guard():
    numba.set_num_threads(1)  # runtime determinism pin (config keys added by the helpers)
    if RECAL_RATE is None:
        pytest.skip("SP1b infeasible: RECAL_RATE is None (see docs/diagnostics/sp1b_recalibration.md)")
    base = _baltic_15yr()
    baseline = mean_cod(with_determinism(base))
    on = mean_cod(sp1_on_config(base, SP_FIELD))  # default larva_rate -> RECAL_RATE
    assert abs(on - baseline) / baseline <= 0.02
```

- [ ] **Step 2: Run it (SLOW — foreground)**

Run: `.venv/bin/python -m pytest tests/test_sp1b_recalibration.py::test_sp1b_mean_neutral_drift_guard -q`
Expected: PASS if `RECAL_RATE` was frozen to a feasible value (mean within ±2%); SKIP if infeasible. If it FAILS (mean outside ±2%), the frozen constant is stale vs the field/engine — re-run Task 4 Step 4 and re-freeze.

- [ ] **Step 3: Write the diagnostic**

Create `scripts/sp1b_diagnostic.py`:

```python
#!/usr/bin/env python
"""SP1b diagnostic: records the recalibrated rate, achieved rel-err, and the cod overshoot
ratio SP1-on-recalibrated vs SP1-off (measured, not gated — does mean-neutral spatial
egg-survival damp the boom/bust?)."""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

from osmose.calibration.larva_recal import RECAL_RATE, mean_cod, sp1_on_config, with_determinism
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
FIELD = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"


def _overshoot(cfg) -> float:
    b = PythonEngine().run_in_memory(cfg, seed=0).biomass()["cod"].to_numpy()[3:15]
    b = b[np.isfinite(b) & (b > 0)]
    return float(b.max() / b.mean()) if b.size and b.mean() > 0 else float("nan")


def main() -> int:
    numba.set_num_threads(1)
    base = dict(OsmoseConfigReader().read(str(CONFIG)))
    base["simulation.time.nyear"] = "15"
    off = with_determinism(base)

    baseline = mean_cod(off)
    over_off = _overshoot(off)
    if RECAL_RATE is None:
        lines = [
            "# SP1b recalibration diagnostic",
            "",
            "RESULT: INFEASIBLE — mean-neutrality not achievable via the cod larva rate alone.",
            f"SP1-off baseline mean cod = {baseline:.1f}; overshoot(off) = {over_off:.2f}",
            "RECAL_RATE = None. See the solve grid in the recalibrate_sp1b commit.",
        ]
    else:
        on = sp1_on_config(base, FIELD)
        mean_on = mean_cod(on)
        over_on = _overshoot(on)
        lines = [
            "# SP1b recalibration diagnostic",
            "",
            f"RECAL_RATE = {RECAL_RATE:.4f}  (cod larval mortality, resolved per-cohort; d0=15.0)",
            f"mean cod: off={baseline:.1f}  on_recal={mean_on:.1f}  "
            f"rel_err={abs(mean_on / baseline - 1):.3f}  (target <= 0.02)",
            "",
            "## Overshoot (max/mean over years 3-14) — measured, NOT gated",
            f"off={over_off:.2f}  on_recal={over_on:.2f}  "
            f"ratio={over_on / over_off:.2f}  "
            f"({'damps' if over_on < over_off else 'does not damp'} the boom/bust)",
        ]
    print("\n".join(lines))
    out = ROOT / "docs" / "diagnostics" / "sp1b_recalibration.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the diagnostic (SLOW — foreground)**

Run: `cd /home/razinka/osmose/osmose-python && PYTHONPATH=. .venv/bin/python scripts/sp1b_diagnostic.py`
Expected: writes `docs/diagnostics/sp1b_recalibration.md` with the recalibrated rate (or the infeasible note), the achieved rel-err, and the SP1-on-recalibrated vs SP1-off overshoot ratio. Report the overshoot finding honestly either way.

- [ ] **Step 5: Lint/types, commit**

```bash
cd /home/razinka/osmose/osmose-python && .venv/bin/ruff check osmose/ tests/ scripts/sp1b_diagnostic.py && .venv/bin/ruff format tests/test_sp1b_recalibration.py scripts/sp1b_diagnostic.py && .venv/bin/pyright scripts/sp1b_diagnostic.py
git -C /home/razinka/osmose/osmose-python add tests/test_sp1b_recalibration.py scripts/sp1b_diagnostic.py docs/diagnostics/sp1b_recalibration.md
git -C /home/razinka/osmose/osmose-python commit -m "feat: SP1b mean-neutrality drift-guard test + recalibration diagnostic"
```

---

## Final verification

- [ ] Full SP1b test file green: `.venv/bin/python -m pytest tests/test_sp1b_recalibration.py -v` (solver stub tests fast; the mean-neutrality test passes or skips-if-infeasible).
- [ ] SP1 test file still green: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -q` (20 tests; `sp1_on_config` swap did not regress).
- [ ] Inert-by-default parity still green: `.venv/bin/python -m pytest -k "parity or cross_engine" -q` (SP1b touches no default config; the engine path is unchanged when SP1 is off).
- [ ] ruff check + `ruff format --check` on all changed files; pyright clean on `osmose/calibration/larva_recal.py` and the two scripts.
- [ ] `docs/diagnostics/sp1b_recalibration.md` records the recalibrated rate (or the infeasible grid), the achieved rel-err, and the measured overshoot ratio.
- [ ] If infeasible, that negative result is recorded honestly (mean-neutrality unreachable via the cod larva rate alone); do NOT tune to force a number.
- [ ] Side-fix (optional, non-blocking): correct CLAUDE.md's stale `simulation.rng.fixed=true` reproducibility note to the two real keys (`movement.randomseed.fixed` + `stochastic.mortality.randomseed.fixed`); can be a standalone one-line commit.
