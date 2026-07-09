# Baltic Chunk 0 — Bistability De-risk Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Supersedes:** `2026-07-08-baltic-chunk0-bistability-derisk.md` (v1). This v2 folds in a 21-finding adversarial review of v1 plus a feasibility check of the engine.

**Goal:** Deliver the two Chunk-0 de-risk experiments with a *trustworthy* instrument: (A) a clean plankton-accessibility A/B whose verdict cannot be faked by a food-web collapse, and (B) a best-effort bistability/response-curve probe that is honest about the one thing the Python engine cannot currently do (warm-started standing initial conditions), so a MONOSTABLE reading is never mistaken for proof.

**Architecture:** One script (`scripts/baltic_bistability_chunk0.py`) of pure helpers + two experiment runners taking an **injectable model-runner** (default `calibrate_baltic.run_simulation`). Every model call goes through a `safe_run` wrapper that turns a crash or empty/partial output into a `_failed` sentinel (never a silent `cod_mean=0`). Every classification is **gated on stationarity** (using the `{sp}_cv`/`{sp}_trend` that `run_simulation` already returns) and expressed against the **ICES bands** (collapsed / low / in-range / overshoot), not a single collapse line. Experiment B seeds two initial states (cod-rich, cod-poor) via a fixed short seeding window and reports per-seed basins plus the continuous response curve; it is explicitly documented as a *conservative* bistability test (Beverton-Holt compensation + egg-only seeding bias it toward MONOSTABLE), with a definitive test deferred to an engine warm-start primitive. Unit tests use a fake runner — no real sims in CI.

**Tech Stack:** Python 3, numpy, pytest; `scripts/calibrate_baltic.py` (`run_simulation`, `load_targets`, `BALTIC_CONFIG`), `osmose.config.reader.OsmoseConfigReader`, `osmose.engine.PythonEngine`. No matplotlib (not a project dep; outputs are JSON + printed verdicts).

## Feasibility finding that shapes this plan (verified in the repo)

- **No standing-stock / warm-start init.** `osmose/engine/simulate.py:1188-1195` `initialize()` returns `SchoolState.create(n_schools=0)`; the docstring states all schools come from the seeding mechanism (SSB==0 → `seeding_biomass` injected as virtual SSB → eggs). `restart.file` / `population.initialization.*` appear only in the `config_validation.py` allowlist (Java-side), with no read path in the Python engine. **Consequence:** the only initial condition available is egg-seeding, which is filtered through the larval-mortality driver being swept and compressed by cod's Beverton-Holt recruitment (`ssbhalf.sp0=120000`). A rigorous alternative-stable-states / hysteresis test needs a standing-adult IC — engine work (a warm-start/restart primitive) tracked as a **prerequisite**, not part of Chunk 0. Task 8 records this so the roadmap carries the true cost.

## Changes from v1 (must-fixes from the review → resolution)

| v1 defect (review) | Resolution in v2 |
|---|---|
| Warm-start not available; egg-seeding ≠ standing IC; seeding-window "raise until diverged" manufactures artifacts | Do **not** chase the window. Fixed 2-y window; per-scale **establishment + stationarity gate**; bistability framed as *conservative*; definitive test deferred to Task 8 engine primitive. |
| 40-y horizon runs past the ~15-y stability ceiling → both ICs collapse → false MONOSTABLE | Default `--years 15` (the shipped `nyear`); optional horizon sweep; every classification gated on stationarity so a still-drifting run is `undetermined`, not a basin. |
| Accessibility verdict fires on any biomass drop / an extinction reads as "relaxes" | Verdict gated on **no new extinctions** AND (in-range count up **and** summed log-overshoot down); distinct "COLLAPSES the web" verdict otherwise. Metric excludes under-target species and total biomass. |
| Bistability keys only on crossing 6000 t; ignores the computed gap | `basins_differ` uses ICES-band state **and** the continuous gap; two distinct non-collapsed equilibria count as bistable. |
| No stationarity check; cv/trend discarded | `is_stationary(cv, trend)` gates every classification; non-stationary → `undetermined`. |
| Crash indistinguishable from collapse; `returncode` guard is dead; one failure aborts the grid | `safe_run` sentinel (`_failed`) distinct from a real 0; failed points excluded from votes; **per-scale JSON persistence**. |
| 3 seeds median/majority hides basin scatter | Report **per-seed** states + present-fraction; `--seeds` default `0 1 2`, raiseable; a split is surfaced, not voted away. |
| Accessibility override hits all sp8–13 (Benthos/Dinoflagellates) at 0.05 (below the 0.1 floor) | Default `resource_indices=(8,10,11,12)` (the calibrated plankton groups), `low_value=0.1` (the realistic floor). |
| Doc says base larva sp0=360; reader migrates to 15 (÷ndt=24) | Read rates at runtime (never hardcode); Self-Review states the post-migration value 15.0. |
| Dangling PNG / matplotlib | Removed; outputs are JSON + printed verdicts only. |
| No runtime budget | Documented below (~18 sims at 15 y). |

## Global Constraints

- **Python-engine only** (never Java; `run_simulation` already uses `PythonEngine`).
- Interpreter `.venv/bin/python`; scripts in `scripts/` use bare `from calibrate_baltic import ...` (scripts/ is `sys.path[0]`); tests insert `scripts/` on `sys.path` (per `tests/test_fr_diagnostic.py`).
- Config keys are **lowercase**; override values are `str(...)`.
- **No unit test calls the real engine** (inject a fake runner); heavy runs are CLI-only.
- Deployed config `data/baltic/baltic_all-parameters.csv`; cod=sp0..stickleback=sp7; LTL sp8(Diatoms),sp9(Dino),sp10(Micro),sp11(Meso),sp12(Macro),sp13(Benthos). Plankton (calibrated) groups = **sp8,10,11,12**.
- **Runtime read** of larva rates yields the post-4.4.1-migration per-dt value (`sp0`→15.0=360/24), not the raw 360.
- ICES bands from `biomass_targets.csv` (cod target 120000, lower 60000, upper 250000). Collapse = mean < `0.05×target`.
- Stationarity thresholds: `cv_max=0.30` (matches calibrate_baltic's `stable = mean_cv < 0.3`), `trend_max=0.05` (normalized `|slope|/(mean+1)`).
- **Runtime budget:** Exp B = 5 scales × 2 ICs × 3 seeds = 30 sims; Exp A = 2 × 3 = 6; probe = 2 → ~38 sims at **15 y** (default), run serially (a single Baltic sim already multithreads via Numba, so outer parallelism is intentionally avoided). Budget ~30–60 min.

---

## File Structure

- **Create `scripts/baltic_bistability_chunk0.py`** — pure helpers (stationarity, state classification, basins, overshoot metric, override builders, safe_run) + two experiment runners (injectable) + config loaders + CLI. **All module imports at the top of the file** (avoids E402).
- **Create `tests/test_baltic_bistability_chunk0.py`** — unit tests, fake runner, CI-safe.
- **Outputs (CLI runs, not committed):** `docs/diagnostics/baltic_chunk0_bistability.json` (written incrementally per scale), `docs/diagnostics/baltic_chunk0_accessibility_ab.json`.

---

### Task 1: Stationarity, ICES-band state, and basin comparison (pure)

**Files:** Create `scripts/baltic_bistability_chunk0.py`; Test `tests/test_baltic_bistability_chunk0.py`.

**Interfaces (Produces):** `is_stationary(cv, trend, cv_max=0.30, trend_max=0.05) -> bool`; `classify_state(mean, cv, trend, target, lower, upper, collapse_frac=0.05) -> str` in `{"undetermined","collapsed","low","in_range","overshoot"}`; `basins_differ(rich_state, poor_state, gap, gap_thresh=0.5) -> bool`; `bistability_gap(a, b) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baltic_bistability_chunk0.py
import sys
from collections import namedtuple
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import baltic_bistability_chunk0 as c0  # noqa: E402

Tgt = namedtuple("Tgt", "species target lower upper")
COD = dict(target=120000.0, lower=60000.0, upper=250000.0)


def test_is_stationary():
    assert c0.is_stationary(0.1, 0.01) is True
    assert c0.is_stationary(0.5, 0.01) is False   # cv too high
    assert c0.is_stationary(0.1, 0.2) is False    # trend too steep


def test_classify_state_bands_and_stationarity():
    # non-stationary -> undetermined regardless of mean
    assert c0.classify_state(120000, 0.5, 0.01, **COD) == "undetermined"
    assert c0.classify_state(3000, 0.1, 0.01, **COD) == "collapsed"     # < 6000
    assert c0.classify_state(30000, 0.1, 0.01, **COD) == "low"          # 6000..60000
    assert c0.classify_state(120000, 0.1, 0.01, **COD) == "in_range"
    assert c0.classify_state(400000, 0.1, 0.01, **COD) == "overshoot"


def test_basins_differ():
    assert c0.basins_differ("in_range", "collapsed", 0.9) is True     # different bands
    assert c0.basins_differ("overshoot", "overshoot", 0.9) is False   # both degenerate-high
    assert c0.basins_differ("collapsed", "collapsed", 0.9) is False   # both collapsed
    assert c0.basins_differ("in_range", "in_range", 0.8) is True      # same band, large gap
    assert c0.basins_differ("in_range", "in_range", 0.1) is False     # same band, small gap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'baltic_bistability_chunk0'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/baltic_bistability_chunk0.py
"""Chunk 0 de-risk harness (v2) for the Baltic OSMOSE improvement roadmap.

Two experiments, both hardened per the v1 adversarial review:
  * accessibility — plankton-accessibility A/B; verdict cannot be faked by a
    food-web collapse (gated on no new extinctions + real over-production metric).
  * bistability  — CONSERVATIVE response-curve / initial-condition probe. The
    Python engine has no standing-stock init (initialize() -> 0 schools; all
    schools come from egg-seeding), so ICs are egg-seeded and filtered through the
    swept larval mortality and Beverton-Holt compensation. A BISTABLE reading is
    therefore strong evidence; a MONOSTABLE reading means "no bistability
    detectable by this method" — a definitive test needs an engine warm-start
    primitive (see the plan's Task 8). Every classification is gated on stationarity.

Pure helpers + injectable runner; unit-tested with a fake runner. Real Baltic runs
are CLI-only (Python engine, minutes each).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagnostics"
_DEFAULT_SCALES = [0.03, 0.1, 0.3, 0.5, 1.0]
_DEFAULT_SEEDS = [0, 1, 2]
_PLANKTON_GROUPS = (8, 10, 11, 12)   # calibrated plankton; excludes sp9 Dino, sp13 Benthos
_SEEDING_WINDOW_Y = 2                 # fixed; NOT tuned to force a result


def is_stationary(cv: float, trend: float, cv_max: float = 0.30, trend_max: float = 0.05) -> bool:
    """True if the last-window biomass is settled (low inter-annual CV and flat trend)."""
    return cv <= cv_max and trend <= trend_max


def classify_state(mean, cv, trend, target, lower, upper, collapse_frac: float = 0.05) -> str:
    """Map a species' stationary mean onto an ICES band; 'undetermined' if not stationary."""
    if not is_stationary(cv, trend):
        return "undetermined"
    if mean < collapse_frac * target:
        return "collapsed"
    if mean < lower:
        return "low"
    if mean > upper:
        return "overshoot"
    return "in_range"


def bistability_gap(a: float, b: float) -> float:
    return abs(a - b) / (max(a, b) + 1.0)


def basins_differ(rich_state, poor_state, gap, gap_thresh: float = 0.5) -> bool:
    """Two stationary equilibria occupy different basins (evidence of bistability)."""
    if rich_state == "undetermined" or poor_state == "undetermined":
        return False
    if rich_state == poor_state == "collapsed":
        return False
    if rich_state == poor_state == "overshoot":
        return False
    if rich_state != poor_state:
        return True
    return gap >= gap_thresh
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v2 stationarity + ICES-band state + basin compare"
```

---

### Task 2: Over-production metric + accessibility verdict (pure)

**Files:** Modify `scripts/baltic_bistability_chunk0.py`; Test same.

**Interfaces (Produces):** `overshoot_metric(stats, targets) -> dict` with keys `overshoot_log`, `in_range`, `extinct`; `accessibility_verdict(baseline, lowered) -> tuple[bool, str]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_overshoot_metric():
    targets = [Tgt("cod", 120000, 60000, 250000), Tgt("sprat", 1_500_000, 800_000, 2_500_000)]
    stats = {"cod_mean": 30000.0, "sprat_mean": 25_000_000.0}  # cod low, sprat 10x over
    m = c0.overshoot_metric(stats, targets)
    assert m["in_range"] == 0
    assert m["extinct"] == 0
    assert abs(m["overshoot_log"] - 1.0) < 1e-6   # log10(25e6/2.5e6)=1


def test_accessibility_verdict_relax_vs_collapse():
    # genuine relaxation: no new extinctions, more in-range, less overshoot
    good = c0.accessibility_verdict(
        {"overshoot_log": 2.0, "in_range": 1, "extinct": 0},
        {"overshoot_log": 0.3, "in_range": 3, "extinct": 0},
    )
    assert good[0] is True and "relax" in good[1].lower()
    # collapse masquerading as relaxation: overshoot gone but extinctions rose
    bad = c0.accessibility_verdict(
        {"overshoot_log": 2.0, "in_range": 3, "extinct": 0},
        {"overshoot_log": 0.0, "in_range": 1, "extinct": 2},
    )
    assert bad[0] is False and "collapse" in bad[1].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'overshoot_metric'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def overshoot_metric(stats: dict, targets: list) -> dict:
    """Over-production-specific metric: summed log10 overshoot of OVER species,
    count in-range, count extinct. Under-target species and total biomass are
    deliberately excluded (starving them is not 'relaxation')."""
    overshoot_log = 0.0
    in_range = 0
    extinct = 0
    for t in targets:
        m = float(stats.get(f"{t.species}_mean", 0.0))
        if m <= 0:
            extinct += 1
            continue
        if m > t.upper:
            overshoot_log += math.log10(m / t.upper)
        elif m >= t.lower:
            in_range += 1
    return {"overshoot_log": overshoot_log, "in_range": in_range, "extinct": extinct}


def accessibility_verdict(baseline: dict, lowered: dict) -> tuple[bool, str]:
    """Relaxation ONLY if no new extinctions AND (more in-range AND less overshoot)."""
    if lowered["extinct"] > baseline["extinct"]:
        return False, (
            f"Lowering accessibility COLLAPSES the web, not relaxes it "
            f"(extinct {baseline['extinct']} -> {lowered['extinct']}). NOT evidence for A1."
        )
    relaxed = lowered["in_range"] >= baseline["in_range"] and lowered["overshoot_log"] < baseline["overshoot_log"]
    if relaxed:
        return True, (
            f"Lowering plankton accessibility relaxes over-production toward ICES bands "
            f"(in-range {baseline['in_range']} -> {lowered['in_range']}; summed log-overshoot "
            f"{baseline['overshoot_log']:.2f} -> {lowered['overshoot_log']:.2f}). A1 is a real lever."
        )
    return False, (
        f"No clean relaxation (in-range {baseline['in_range']} -> {lowered['in_range']}; "
        f"log-overshoot {baseline['overshoot_log']:.2f} -> {lowered['overshoot_log']:.2f}). "
        f"Reconsider A1 before building it."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v2 extinction-gated accessibility verdict"
```

---

### Task 3: Override builders + safe_run failure sentinel

**Files:** Modify both.

**Interfaces (Produces):** `larva_scale_override(scale, base_rates) -> dict`; `accessibility_override(value, resource_indices=_PLANKTON_GROUPS) -> dict`; `cod_rich_seeding(window=_SEEDING_WINDOW_Y) -> dict`; `cod_poor_seeding(window=_SEEDING_WINDOW_Y) -> dict`; `safe_run(runner, config, overrides, n_years, seed) -> dict` (adds `_failed=True` on crash/empty).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_overrides_scope_and_values():
    assert c0.accessibility_override(0.1) == {
        "species.accessibility2fish.sp8": "0.1",
        "species.accessibility2fish.sp10": "0.1",
        "species.accessibility2fish.sp11": "0.1",
        "species.accessibility2fish.sp12": "0.1",
    }  # sp9 (Dino) and sp13 (Benthos) deliberately excluded
    assert c0.larva_scale_override(0.1, {0: 15.0})["mortality.additional.larva.rate.sp0"] == "1.5"
    rich, poor = c0.cod_rich_seeding(), c0.cod_poor_seeding()
    assert float(rich["population.seeding.biomass.sp0"]) > float(poor["population.seeding.biomass.sp0"])
    assert rich["population.seeding.year.max"] == "2" and poor["population.seeding.year.max"] == "2"


def test_safe_run_sentinel():
    assert c0.safe_run(lambda *a: {"cod_mean": 5.0}, {}, {}, 5, 0) == {"cod_mean": 5.0}
    assert c0.safe_run(lambda *a: {}, {}, {}, 5, 0)["_failed"] is True          # empty
    assert c0.safe_run(lambda *a: {"herring_mean": 1.0}, {}, {}, 5, 0)["_failed"] is True  # no cod_mean
    def boom(*a):
        raise RuntimeError("blowup")
    assert c0.safe_run(boom, {}, {}, 5, 0)["_failed"] is True                   # crash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'larva_scale_override'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def larva_scale_override(scale: float, base_rates: dict) -> dict:
    return {f"mortality.additional.larva.rate.sp{i}": str(r * scale) for i, r in base_rates.items()}


def accessibility_override(value: float, resource_indices=_PLANKTON_GROUPS) -> dict:
    return {f"species.accessibility2fish.sp{i}": str(value) for i in resource_indices}


def cod_rich_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    return {
        "population.seeding.biomass.sp0": "300000",
        "population.seeding.biomass.sp1": "800000",
        "population.seeding.biomass.sp2": "600000",
        "population.seeding.year.max": str(window),
    }


def cod_poor_seeding(window: int = _SEEDING_WINDOW_Y) -> dict:
    return {
        "population.seeding.biomass.sp0": "1000",
        "population.seeding.biomass.sp1": "1500000",
        "population.seeding.biomass.sp2": "1500000",
        "population.seeding.year.max": str(window),
    }


def safe_run(runner, config, overrides, n_years, seed) -> dict:
    """Run the model; return a `_failed` sentinel on crash or empty/partial output.

    A failed run is DISTINCT from a real cod_mean==0 collapse; callers exclude
    `_failed` points from basin votes rather than scoring them 'collapsed'.
    """
    try:
        stats = runner(config, overrides, n_years, seed)
    except Exception as exc:  # noqa: BLE001 — diagnostic must not abort the whole grid
        return {"_failed": True, "_error": repr(exc)}
    if not stats or "cod_mean" not in stats:
        return {"_failed": True, "_error": "empty or partial stats (no cod_mean)"}
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v2 override builders (scoped) + safe_run sentinel"
```

---

### Task 4: Bistability point + sweep runner (per-seed, stationarity-gated, gap-based)

**Files:** Modify both.

**Interfaces (Produces):** `run_bistability_point(scale, base_config, base_rates, cod_bands, seeds, *, runner, n_years) -> dict`; `run_bistability_sweep(scales, base_config, base_rates, cod_bands, seeds, *, runner, n_years, on_point=None) -> dict`. `cod_bands` = `{"target","lower","upper"}`. `on_point(point)` optional callback for per-scale persistence.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def _bands():
    return {"target": 120000.0, "lower": 60000.0, "upper": 250000.0}

def _runner_bistable(config, overrides, n_years, seed):
    """Stationary; at scale 0.3 the IC decides cod fate (rich in-range, poor collapsed)."""
    scale = float(overrides["mortality.additional.larva.rate.sp0"]) / 15.0
    seeded = float(overrides.get("population.seeding.biomass.sp0", "0"))
    cod = 120000.0 if (abs(scale - 0.3) < 1e-9 and seeded >= 100000) else (
        120000.0 if scale < 0.9 else 100.0)
    return {"cod_mean": cod, "cod_cv": 0.05, "cod_trend": 0.01}

def test_bistability_point_flags_basin_split():
    pt = c0.run_bistability_point(0.3, {}, {0: 15.0}, _bands(), [0, 1, 2],
                                  runner=_runner_bistable, n_years=15)
    assert pt["rich_state"] == "in_range"
    assert pt["poor_state"] == "collapsed"
    assert pt["bistable"] is True
    assert pt["per_seed_rich"] == ["in_range", "in_range", "in_range"]

def test_bistability_sweep_verdict_and_persistence():
    seen = []
    out = c0.run_bistability_sweep([0.1, 0.3, 1.0], {}, {0: 15.0}, _bands(), [0],
                                   runner=_runner_bistable, n_years=15, on_point=seen.append)
    assert out["bistable"] is True and 0.3 in out["bistable_scales"]
    assert "conservative" in out["verdict"].lower()   # honest caveat present
    assert len(seen) == 3                              # per-scale persistence fired
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'run_bistability_point'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def _cod_state(stats: dict, bands: dict) -> tuple[str, float]:
    if stats.get("_failed"):
        return "failed", 0.0
    mean = float(stats.get("cod_mean", 0.0))
    st = classify_state(mean, float(stats.get("cod_cv", 1.0)), float(stats.get("cod_trend", 1.0)),
                        bands["target"], bands["lower"], bands["upper"])
    return st, mean


def run_bistability_point(scale, base_config, base_rates, cod_bands, seeds, *, runner, n_years) -> dict:
    driver = larva_scale_override(scale, base_rates)
    rich_states, poor_states, rich_means, poor_means = [], [], [], []
    for seed in seeds:
        r = safe_run(runner, base_config, {**driver, **cod_rich_seeding()}, n_years, seed)
        p = safe_run(runner, base_config, {**driver, **cod_poor_seeding()}, n_years, seed)
        rs, rm = _cod_state(r, cod_bands)
        ps, pm = _cod_state(p, cod_bands)
        rich_states.append(rs); poor_states.append(ps)
        rich_means.append(rm); poor_means.append(pm)

    def _agg(states, means):
        valid = [m for s, m in zip(states, means) if s not in ("failed", "undetermined")]
        med = statistics.median(valid) if valid else 0.0
        # modal state among valid seeds (deterministic tie-break: prefer the state of the median)
        good = [s for s in states if s not in ("failed", "undetermined")]
        state = classify_state(med, 0.0, 0.0, cod_bands["target"], cod_bands["lower"],
                               cod_bands["upper"]) if good else "undetermined"
        return state, med, good

    rich_state, rich_med, rich_good = _agg(rich_states, rich_means)
    poor_state, poor_med, poor_good = _agg(poor_states, poor_means)
    gap = bistability_gap(rich_med, poor_med)
    determinable = bool(rich_good) and bool(poor_good)
    bistable = determinable and basins_differ(rich_state, poor_state, gap)
    return {
        "scale": scale,
        "rich_state": rich_state, "poor_state": poor_state,
        "rich_cod_median": rich_med, "poor_cod_median": poor_med, "gap": gap,
        "per_seed_rich": rich_states, "per_seed_poor": poor_states,
        "rich_present_fraction": (sum(s in ("in_range", "overshoot", "low") for s in rich_good) / len(rich_good)) if rich_good else None,
        "determinable": determinable, "bistable": bistable,
    }


def run_bistability_sweep(scales, base_config, base_rates, cod_bands, seeds, *, runner, n_years, on_point=None) -> dict:
    points = []
    for s in scales:
        pt = run_bistability_point(s, base_config, base_rates, cod_bands, seeds,
                                   runner=runner, n_years=n_years)
        points.append(pt)
        if on_point is not None:
            on_point(pt)   # per-scale persistence so a later crash keeps earlier results
    bistable_scales = [p["scale"] for p in points if p["bistable"]]
    undetermined = [p["scale"] for p in points if not p["determinable"]]
    if bistable_scales:
        verdict = (f"BISTABLE (conservative test) — different cod basins from the two ICs at "
                   f"larva-scale(s) {bistable_scales}. Egg-seeding + Beverton-Holt bias this test "
                   f"toward MONOSTABLE, so a positive result is strong evidence the model already "
                   f"supports alternative stable states. Confirm with a warm-start standing IC (Task 8).")
    else:
        verdict = (f"MONOSTABLE by this method — no basin split at any scale "
                   f"(undetermined scales: {undetermined}). This is a CONSERVATIVE test (egg-only "
                   f"ICs filtered through the swept larval M + B-H compensation), so it CANNOT rule "
                   f"out bistability; it means the roadmap should either add the engine warm-start "
                   f"primitive (Task 8) for a definitive test, or proceed to Chunks C & A2 to CREATE "
                   f"a self-locking bistability. Read the response curve (rich/poor cod vs scale) for "
                   f"transition sharpness.")
    return {"points": points, "bistable": bool(bistable_scales),
            "bistable_scales": bistable_scales, "undetermined_scales": undetermined, "verdict": verdict}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v2 stationarity-gated, gap-based bistability sweep"
```

---

### Task 5: Accessibility A/B runner (stationarity-gated, real metric)

**Files:** Modify both.

**Interfaces (Produces):** `run_accessibility_ab(base_config, targets, seeds, *, runner, n_years, low_value=0.1) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_accessibility_ab_collapse_not_relax():
    targets = [Tgt("cod", 120000, 60000, 250000), Tgt("sprat", 1_500_000, 800_000, 2_500_000),
               Tgt("herring", 1_500_000, 800_000, 3_000_000)]
    def runner(config, overrides, n_years, seed):
        low = "species.accessibility2fish.sp11" in overrides
        base = {"cod_cv": 0.05, "cod_trend": 0.01}
        if low:  # starving planktivores to extinction
            return {**base, "cod_mean": 30000.0, "sprat_mean": 0.0, "herring_mean": 0.0}
        return {**base, "cod_mean": 30000.0, "sprat_mean": 25_000_000.0, "herring_mean": 20_000_000.0}
    out = c0.run_accessibility_ab({}, targets, [0], runner=runner, n_years=15)
    assert out["relaxed"] is False and "collapse" in out["verdict"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'run_accessibility_ab'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def _median_metric(metrics: list) -> dict:
    keys = ("overshoot_log", "in_range", "extinct")
    return {k: statistics.median([m[k] for m in metrics]) for k in keys}


def run_accessibility_ab(base_config, targets, seeds, *, runner, n_years, low_value: float = 0.1) -> dict:
    base_m, low_m = [], []
    for seed in seeds:
        s_base = safe_run(runner, base_config, {}, n_years, seed)
        s_low = safe_run(runner, base_config, accessibility_override(low_value), n_years, seed)
        base_m.append(overshoot_metric({} if s_base.get("_failed") else s_base, targets))
        low_m.append(overshoot_metric({} if s_low.get("_failed") else s_low, targets))
    baseline = _median_metric(base_m)
    lowered = _median_metric(low_m)
    relaxed, verdict = accessibility_verdict(baseline, lowered)
    return {"baseline": baseline, "lowered": lowered, "low_value": low_value,
            "relaxed": relaxed, "verdict": verdict}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v2 accessibility A/B runner"
```

---

### Task 6: Config loaders + CLI + smoke gate

**Files:** Modify both.

**Interfaces (Produces):** `read_base_config() -> dict`; `read_base_larva_rates(base_config, n_focal=8) -> dict`; `read_cod_bands(targets) -> dict`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_read_base_larva_rates_and_bands():
    cfg = {f"mortality.additional.larva.rate.sp{i}": str(i + 1) for i in range(8)}
    rates = c0.read_base_larva_rates(cfg)
    assert rates[0] == 1.0 and rates[7] == 8.0 and set(rates) == set(range(8))
    bands = c0.read_cod_bands([Tgt("cod", 120000, 60000, 250000)])
    assert bands == {"target": 120000.0, "lower": 60000.0, "upper": 250000.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'read_base_larva_rates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def read_base_config() -> dict:
    from calibrate_baltic import BALTIC_CONFIG
    from osmose.config.reader import OsmoseConfigReader
    return OsmoseConfigReader().read(str(BALTIC_CONFIG))


def read_base_larva_rates(base_config: dict, n_focal: int = 8) -> dict:
    rates = {}
    for i in range(n_focal):
        key = f"mortality.additional.larva.rate.sp{i}"
        if key in base_config:
            rates[i] = float(base_config[key])   # post-4.4.1 migration => per-dt (~15 for cod)
    return rates


def read_cod_bands(targets) -> dict:
    t = next(t for t in targets if t.species == "cod")
    return {"target": float(t.target), "lower": float(t.lower), "upper": float(t.upper)}


def _default_runner(config, overrides, n_years, seed):
    from calibrate_baltic import run_simulation
    return run_simulation(config, overrides, n_years=n_years, seed=seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baltic Chunk 0 de-risk experiments (v2)")
    ap.add_argument("--experiment", choices=["bistability", "accessibility", "both"], default="both")
    ap.add_argument("--years", type=int, default=15)   # ships nyear=15; stability ceiling
    ap.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS)
    ap.add_argument("--scales", type=float, nargs="+", default=_DEFAULT_SCALES)
    ap.add_argument("--low-accessibility", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    from calibrate_baltic import load_targets
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    scales = [1.0, 0.1] if args.smoke else args.scales
    years = 3 if args.smoke else args.years

    base_config = read_base_config()
    base_rates = read_base_larva_rates(base_config)
    targets = load_targets()
    cod_bands = read_cod_bands(targets)
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"base larva rates (post-migration, per-dt): {base_rates}")

    if args.experiment in ("bistability", "both"):
        out_path = _DIAG_DIR / "baltic_chunk0_bistability.json"
        state = {"points": [], "note": "written incrementally"}

        def _persist(pt):
            state["points"].append(pt)
            out_path.write_text(json.dumps(state, indent=2))

        result = run_bistability_sweep(scales, base_config, base_rates, cod_bands, seeds,
                                       runner=_default_runner, n_years=years, on_point=_persist)
        print("\n=== BISTABILITY (conservative) ===")
        for p in result["points"]:
            print(f"  larva x{p['scale']:<5} rich={p['rich_state']:<12} poor={p['poor_state']:<12} "
                  f"gap={p['gap']:.3f} rich_seeds={p['per_seed_rich']} -> "
                  f"{'BISTABLE' if p['bistable'] else ('undet' if not p['determinable'] else 'same basin')}")
        print(f"\nVERDICT: {result['verdict']}")
        out_path.write_text(json.dumps(result, indent=2))

    if args.experiment in ("accessibility", "both"):
        result = run_accessibility_ab(base_config, targets, seeds, runner=_default_runner,
                                      n_years=years, low_value=args.low_accessibility)
        print("\n=== ACCESSIBILITY A/B ===")
        print(f"  baseline: {result['baseline']}")
        print(f"  lowered({args.low_accessibility}): {result['lowered']}")
        print(f"\nVERDICT: {result['verdict']}")
        (_DIAG_DIR / "baltic_chunk0_accessibility_ab.json").write_text(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Lint, then smoke-run end-to-end**

Run: `.venv/bin/ruff check scripts/baltic_bistability_chunk0.py`
Expected: no E402 (all imports are at the top) — clean, or only intended `# noqa`.

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment both --smoke`
Expected: prints the read base larva rates, a BISTABILITY block (2 scales) and an ACCESSIBILITY block with numeric verdicts; writes both JSON files; exits 0. Scientific meaning is NOT expected at 3 y — this only proves the harness runs against the real engine and that `safe_run` handled every point.

- [ ] **Step 6: Commit**

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v2 CLI, loaders, incremental persistence, smoke gate"
```

---

### Task 7: Establishment / horizon validity check (the instrument gate)

Instead of the v1 "raise the seeding window until diverged" chase (which manufactures artifacts), validate the instrument by confirming that at low larval-mortality scales cod-rich reaches a **stationary non-collapsed** state (so seeding successfully bootstrapped a self-sustaining stock), and that the chosen horizon is long enough for that.

**Files:** Modify both.

**Interfaces (Produces):** `establishment_report(base_config, base_rates, cod_bands, *, runner, n_years, seed, low_scale=0.1) -> dict` returning `{"cod_rich_state","cod_rich_mean","established"}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baltic_bistability_chunk0.py

def test_establishment_report():
    def runner(config, overrides, n_years, seed):
        return {"cod_mean": 120000.0, "cod_cv": 0.05, "cod_trend": 0.01}
    out = c0.establishment_report({}, {0: 15.0}, _bands(), runner=runner, n_years=15, seed=0)
    assert out["established"] is True and out["cod_rich_state"] == "in_range"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: FAIL — `AttributeError: ... 'establishment_report'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/baltic_bistability_chunk0.py

def establishment_report(base_config, base_rates, cod_bands, *, runner, n_years, seed, low_scale: float = 0.1) -> dict:
    """At a low larval-mortality scale, can egg-seeding bootstrap a stationary,
    non-collapsed cod stock? If not, the instrument (not the biology) is the limit."""
    driver = larva_scale_override(low_scale, base_rates)
    stats = safe_run(runner, base_config, {**driver, **cod_rich_seeding()}, n_years, seed)
    state, mean = _cod_state(stats, cod_bands)
    return {"cod_rich_state": state, "cod_rich_mean": mean,
            "established": state in ("low", "in_range", "overshoot")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q`
Expected: PASS (12 tests).

- [ ] **Step 5: Wire the check into the CLI and commit**

Add near the top of `main()` (after `base_rates` is read), before the bistability block:

```python
        est = establishment_report(base_config, base_rates, cod_bands,
                                   runner=_default_runner, n_years=years, seed=seeds[0])
        print(f"establishment @ scale {0.1}: cod_rich -> {est['cod_rich_state']} "
              f"(established={est['established']})")
        if not est["established"]:
            print("WARNING: cod-rich cannot bootstrap a stationary stock even at low larval M — "
                  "the egg-seeding instrument, not the biology, may be the limit; a definitive "
                  "bistability test needs the engine warm-start primitive (see plan Task 8). "
                  "Interpret a MONOSTABLE verdict with caution.")
```

Run: `.venv/bin/python -m pytest tests/test_baltic_bistability_chunk0.py -q` → PASS (12).

```bash
git add scripts/baltic_bistability_chunk0.py tests/test_baltic_bistability_chunk0.py
git commit -m "feat(baltic): chunk0 v2 establishment/instrument-validity check"
```

---

### Task 8: Record the warm-start prerequisite (docs only)

The definitive bistability/hysteresis test needs a standing-adult initial condition the Python engine cannot currently provide. Capture the finding and the minimal primitive so the roadmap carries the cost.

**Files:** Create `docs/baltic_chunk0_warmstart_prerequisite.md`.

- [ ] **Step 1: Write the note** (no test; docs task)

```markdown
# Chunk 0 follow-on — warm-start standing IC is a prerequisite for a definitive bistability test

`osmose/engine/simulate.py:1188` `initialize()` returns `SchoolState.create(n_schools=0)`; all
schools come from egg-seeding (SSB==0 -> `seeding_biomass` as virtual SSB -> eggs). There is no
standing-stock / restart population init in the Python engine (`restart.file` /
`population.initialization.*` are Java-side allowlist keys only). Therefore Chunk 0's bistability
experiment can only seed EGGS, filtered through the swept larval mortality and compressed by cod's
Beverton-Holt recruitment — a CONSERVATIVE test that can confirm bistability but cannot rule it out.

**Minimal primitive for a definitive test (future engine sub-chunk):** an age-structured
standing-stock initializer — given a per-species initial biomass, distribute it across the
size/age structure at t=0 (OSMOSE "initialization by relative biomass") OR a restart reader that
loads a `SchoolState` snapshot written by a prior run. Either lets two genuine adult standing
stocks (cod-rich, cod-poor) evolve under fixed parameters, which is what a hysteresis/alternative-
stable-states test requires. Estimated effort: medium (engine + `initialize()` + a snapshot format
+ parity tests). Do this before claiming a MONOSTABLE result is definitive.
```

- [ ] **Step 2: Commit**

```bash
git add docs/baltic_chunk0_warmstart_prerequisite.md
git commit -m "docs(baltic): record warm-start prerequisite for a definitive bistability test"
```

---

## The real experiment runs (post-implementation, not CI)

~38 sims at 15 y (~30–60 min, serial). Read printed verdicts + the two JSON files.

1. **Instrument + establishment check** is printed by every run; if `established=False`, treat a MONOSTABLE result as inconclusive (needs Task-8 warm-start).
2. **Both experiments at the stability horizon:**
   `PYTHONPATH=. .venv/bin/python scripts/baltic_bistability_chunk0.py --experiment both --years 15 --seeds 0 1 2`
3. **Optional horizon sensitivity:** re-run bistability at `--years 30` and compare; a basin split that survives both horizons is stronger, a MONOSTABLE that only appears at 30 y is likely the documented long-horizon collapse, not structure.

Record verdicts in `docs/baltic_chunk0_results_YYYY-MM-DD.md`; they steer whether Phase 2 emphasizes *creating* bistability (Chunks C, A2), adding the warm-start primitive (Task 8) for a definitive test, or building A1.

---

## Self-Review

**Spec coverage.** Every v1 must-fix maps to a task (see the Changes table): stationarity (Task 1), extinction-gated accessibility (Task 2), scoped overrides + failure sentinel (Task 3), gap/per-seed/persistence bistability (Task 4), real accessibility metric (Task 5), 15-y horizon + loaders + smoke + lint (Task 6), instrument-validity/establishment (Task 7), warm-start prerequisite (Task 8). Values are the real deployed ones read at runtime; **larva base is 15.0 (=360/24 after the 4.4.1 read migration), not 360** — read, never hardcoded. No matplotlib/PNG.

**Placeholder scan.** No "TBD"/"handle edge cases"/unshown code; every step has runnable code and an exact command with expected output. The one honest deferral (a definitive warm-start test) is an explicit documented prerequisite (Task 8), not a hidden gap.

**Type consistency.** `runner(config, overrides, n_years, seed) -> dict` (matching `run_simulation`) is used identically everywhere, wrapped by `safe_run`. `classify_state(mean, cv, trend, target, lower, upper, ...)`, `_cod_state`, `basins_differ`, `overshoot_metric`, `accessibility_verdict` keep one signature across tasks. Stats keys `{sp}_mean`/`{sp}_cv`/`{sp}_trend` match `run_simulation`'s output; the `_failed` sentinel is the sole non-`{sp}_*` key and is checked before any `.get("cod_mean")`.
