# Fisheries Stock-Status Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute per-species fisheries stock-status diagnostics (F/M, B/Bmsy, F/Fmsy, Kobe quadrant) from a finished OSMOSE run, using the ICES reference points already loaded by `osmose/validation/ices.py`, and surface them via a library module, a Kobe plot, and a CLI.

**Architecture:** A new `osmose/validation/fisheries.py` reuses `ices.py`'s `IcesSnapshot`/`load_snapshot`/`model_biomass_window_mean` and the snapshot manifest (species→stock mapping + `units_by_stock` + `reference_points`). It adds mortality-rate helpers (annual F and M from the `mortalityRate` CSV), a Kobe-quadrant classifier, a `FisheriesStatus` dataclass, `compute_fisheries_status()`, and a markdown report. Plotting functions go in `osmose/plotting.py`; a CLI in `scripts/compute_fisheries_diagnostics.py` mirrors `scripts/validate_outputs_vs_ices.py`. No engine changes, no calibration runs.

**Tech Stack:** Python 3.12, pandas, NumPy, Plotly (via `osmose/plotly_theme.py` `PLOTLY_TEMPLATE`), pytest, ruff. Run tests with `.venv/bin/python -m pytest`.

**Reference spec:** `docs/superpowers/specs/2026-06-03-fisheries-diagnostics-design.md`.

---

## Verified facts (from reconnaissance — use exactly)

- `osmose/validation/ices.py` exposes: `IcesSnapshot(manifest, assessments, reference_points, snapshot_dir)`; `load_snapshot(dir)`; `model_biomass_window_mean(results, species, window_years)` → float tonnes.
- `snapshot.manifest["model_species_to_ices_stocks"]` = `{species: [stock_keys]}`; `snapshot.manifest["units_by_stock"]` = `{stock: "tonnes"|"index"}`; `snapshot.reference_points[stock]` = dict with keys `fmsy`, `blim`, `bpa`, `msy_btrigger` (values may be `None`).
- `results.mortality(species)` reads `{prefix}_mortalityRate-{species}_Simu0.csv`. That CSV has a **2-row column header**: row A = cause (`Mpred`,`Mstarv`,`Madd`,`F`,`Zout`,`Mfor`,`Mdis`,`Mage`, each repeated 3×), row B = life-stage (`Eggs`,`Pre-recruits`,`Recruits`); first column is `Time` (per saving timestep). "To get annual rates, sum within one year." (`osmose/engine/output.py:210`.)
- `results.biomass(species)` → long-form `time, species, value` (tonnes).
- `osmose/plotting.py` imports `from osmose.plotly_theme import PLOTLY_TEMPLATE as TEMPLATE`; every `make_*` fn returns a `go.Figure` and sets `template=TEMPLATE`.
- CLI precedent `scripts/validate_outputs_vs_ices.py`: argparse with `--results-dir --snapshots-dir --window-years --prefix --report --json`; `PROJECT_ROOT`/`DEFAULT_SNAPSHOT_DIR` constants; `dataclasses.asdict` for JSON.

## File Structure

- Create: `osmose/validation/fisheries.py` — helpers, dataclass, compute, report.
- Modify: `osmose/plotting.py` — add `make_kobe_plot`, `make_fm_ratio_bars`.
- Create: `scripts/compute_fisheries_diagnostics.py` — CLI.
- Create: `tests/test_validation_fisheries.py` — unit tests.
- Modify: a feature/usage doc (find via `grep -rl "validate_outputs_vs_ices" docs/`) — add a short CLI note.

---

## Task 1: Mortality-rate helpers (annual F and M)

**Files:**
- Create: `osmose/validation/fisheries.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Write failing tests with a synthetic mortality DataFrame**

The helpers take a *DataFrame already shaped like `results.mortality()` returns* so they're unit-testable without a run. First, in `tests/test_validation_fisheries.py`, build a synthetic mortality frame matching the real 2-row header and assert annual aggregation:

```python
import numpy as np
import pandas as pd
import pytest
from osmose.validation import fisheries as fz


def _synthetic_mortality_df(n_years=3, steps_per_year=2):
    """Mimic results.mortality(): MultiIndex columns (cause, stage) + a 'Time' column.
    F/Recruits = 0.1 per step → annual F = 0.1*steps_per_year. Mpred/Recruits=0.2,
    Mstarv/Recruits=0.05, Madd/Recruits=0.01 per step → annual M = 0.26*steps_per_year."""
    causes = ["Mpred", "Mstarv", "Madd", "F"]
    stages = ["Eggs", "Pre-recruits", "Recruits"]
    cols = pd.MultiIndex.from_product([causes, stages])
    n = n_years * steps_per_year
    data = np.zeros((n, len(cols)))
    per_step = {("F", "Recruits"): 0.1, ("Mpred", "Recruits"): 0.2,
                ("Mstarv", "Recruits"): 0.05, ("Madd", "Recruits"): 0.01}
    for (c, s), v in per_step.items():
        data[:, cols.get_loc((c, s))] = v
    df = pd.DataFrame(data, columns=cols)
    df[("Time", "")] = np.arange(1, n + 1, dtype=float)
    return df


def test_annual_fishing_mortality_sums_within_year():
    df = _synthetic_mortality_df(n_years=3, steps_per_year=2)
    # annual F per year = 0.1 * 2 = 0.2; windowed mean over last 2 years = 0.2
    assert fz._annual_fishing_mortality(df, steps_per_year=2, window_years=2) == pytest.approx(0.2)


def test_annual_natural_mortality_sums_causes_and_steps():
    df = _synthetic_mortality_df(n_years=3, steps_per_year=2)
    # annual M per year = (0.2+0.05+0.01)*2 = 0.52
    assert fz._annual_natural_mortality(df, steps_per_year=2, window_years=2) == pytest.approx(0.52)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k "annual_" -v`
Expected: FAIL (`module 'fisheries' has no attribute '_annual_fishing_mortality'`).

- [ ] **Step 3: Implement the helpers**

Create `osmose/validation/fisheries.py`:

```python
"""Fisheries stock-status diagnostics for OSMOSE outputs.

Computes per-species F/M, B/Bmsy, F/Fmsy and Kobe-quadrant status from a
finished run, using ICES reference points loaded by `osmose.validation.ices`.
OSMOSE has no native MSY; B/Bmsy uses ICES `msy_btrigger` as the Bmsy proxy
and F/Fmsy uses ICES `fmsy`. Reference-point ratios are computed only for
species whose ICES sub-stocks are all tonnes-unit with non-null fmsy (mixed
or index-unit stocks → no ratio, reported honestly).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from osmose.validation.ices import IcesSnapshot, model_biomass_window_mean

if TYPE_CHECKING:
    from osmose.results import OsmoseResults

_NATURAL_CAUSES = ("Mpred", "Mstarv", "Madd")
_RECRUITS = "Recruits"


def _recruits_series(df: pd.DataFrame, cause: str) -> pd.Series:
    """Extract the (cause, 'Recruits') per-timestep column from a mortalityRate frame.

    Handles both a MultiIndex-column frame and a flattened frame whose columns
    were joined (e.g. 'F Recruits' / 'F_Recruits'). Returns a float Series.
    """
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        if (cause, _RECRUITS) in cols:
            return df[(cause, _RECRUITS)].astype(float)
        raise KeyError(f"({cause!r}, 'Recruits') not in mortality columns")
    # flattened fallback: find a column containing both the cause and 'Recruits'
    for c in cols:
        name = str(c)
        if cause in name and _RECRUITS in name:
            return df[c].astype(float)
    raise KeyError(f"no flattened '{cause} Recruits' column in {list(cols)}")


def _windowed_annual_mean(per_step: pd.Series, steps_per_year: int, window_years: int) -> float:
    """Sum a per-timestep rate within each year, then mean over the trailing window."""
    n = len(per_step)
    n_years = n // steps_per_year
    if n_years == 0:
        raise ValueError("mortality series shorter than one year")
    vals = per_step.to_numpy(dtype=float)[: n_years * steps_per_year]
    annual = vals.reshape(n_years, steps_per_year).sum(axis=1)
    w = min(window_years, n_years)
    return float(annual[-w:].mean())


def _annual_fishing_mortality(df: pd.DataFrame, steps_per_year: int, window_years: int) -> float:
    return _windowed_annual_mean(_recruits_series(df, "F"), steps_per_year, window_years)


def _annual_natural_mortality(df: pd.DataFrame, steps_per_year: int, window_years: int) -> float:
    total = None
    for cause in _NATURAL_CAUSES:
        s = _recruits_series(df, cause)
        total = s if total is None else total + s
    return _windowed_annual_mean(total, steps_per_year, window_years)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k "annual_" -v`
Expected: PASS (both).

- [ ] **Step 5: Verify the real CSV shape matches `_recruits_series`**

Confirm `results.mortality()`'s actual column form so `_recruits_series` handles it. Run:
`.venv/bin/python -c "from osmose.results import OsmoseResults; r=OsmoseResults('data/baltic/output'); m=r.mortality('cod'); print(type(m.columns)); print(list(m.columns)[:6])"`
If columns are a `MultiIndex` → the primary branch handles it. If flattened strings → the fallback handles it. If neither matches (e.g. header rows became data), adjust `_recruits_series` to read the CSV directly with `pd.read_csv(path, header=[1,2])` via a `_read_mortality_csv(results, species)` helper, and add a test for that path. Document what you found in the commit message.

- [ ] **Step 6: Commit**

```bash
git add osmose/validation/fisheries.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): annual F and M helpers from mortalityRate output"
```

---

## Task 2: Kobe-quadrant classifier

**Files:**
- Modify: `osmose/validation/fisheries.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Write failing tests**

```python
def test_kobe_quadrant_classification():
    assert fz.kobe_quadrant(1.5, 0.5) == "green"    # healthy: B>=1, F<=1
    assert fz.kobe_quadrant(0.5, 1.5) == "red"      # overfished + overfishing
    assert fz.kobe_quadrant(1.5, 1.5) == "orange"   # overfishing, not yet overfished
    assert fz.kobe_quadrant(0.5, 0.5) == "yellow"   # overfished, not overfishing
    # on-the-line edges count as the healthy side (>= / <=)
    assert fz.kobe_quadrant(1.0, 1.0) == "green"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k kobe -v`
Expected: FAIL (`no attribute 'kobe_quadrant'`).

- [ ] **Step 3: Implement**

Add to `osmose/validation/fisheries.py`:

```python
def kobe_quadrant(b_over_bmsy: float, f_over_fmsy: float) -> str:
    """Standard Kobe stock-status quadrant.

    green  = B/Bmsy >= 1 and F/Fmsy <= 1 (healthy)
    orange = B/Bmsy >= 1 and F/Fmsy > 1  (overfishing, not yet overfished)
    yellow = B/Bmsy < 1 and F/Fmsy <= 1  (overfished, not overfishing)
    red    = B/Bmsy < 1 and F/Fmsy > 1   (overfished and overfishing)
    """
    healthy_b = b_over_bmsy >= 1.0
    healthy_f = f_over_fmsy <= 1.0
    if healthy_b and healthy_f:
        return "green"
    if healthy_b and not healthy_f:
        return "orange"
    if not healthy_b and healthy_f:
        return "yellow"
    return "red"
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k kobe -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fisheries.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): Kobe-quadrant classifier"
```

---

## Task 3: `FisheriesStatus` dataclass + `compute_fisheries_status`

**Files:**
- Modify: `osmose/validation/fisheries.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Write failing tests with synthetic results + snapshot**

```python
class _FakeResults:
    """Minimal stand-in for OsmoseResults exposing biomass() and mortality()."""
    def __init__(self, biomass_by_sp, mort_by_sp):
        self._b = biomass_by_sp      # {species: pd.DataFrame(time,species,value)}
        self._m = mort_by_sp         # {species: synthetic mortality df}
        self.output_dir = "<fake>"
    def biomass(self, species=None):
        return self._b[species]
    def mortality(self, species=None):
        return self._m[species]


def _biomass_df(value):
    return pd.DataFrame({"time": [1.0, 2.0, 3.0], "species": ["x"] * 3, "value": [value] * 3})


def _snapshot(ref_points, units, mapping):
    return IcesSnapshot(
        manifest={"model_species_to_ices_stocks": mapping, "units_by_stock": units},
        assessments={}, reference_points=ref_points, snapshot_dir=Path("<fake>"),
    )


def test_compute_status_with_reference_points():
    # sprat: tonnes stock with fmsy=0.34, msy_btrigger=500000
    mort = {"sprat": _synthetic_mortality_df(n_years=3, steps_per_year=2)}  # annual F=0.2, M=0.52
    res = _FakeResults({"sprat": _biomass_df(1_000_000)}, mort)
    snap = _snapshot(
        ref_points={"spr.27.22-32": {"fmsy": 0.34, "msy_btrigger": 500_000}},
        units={"spr.27.22-32": "tonnes"},
        mapping={"sprat": ["spr.27.22-32"]},
    )
    out = {s.species: s for s in fz.compute_fisheries_status(res, snap, window_years=2, steps_per_year=2)}
    s = out["sprat"]
    assert s.has_reference_points is True
    assert s.fishing_mortality == pytest.approx(0.2)
    assert s.natural_mortality == pytest.approx(0.52)
    assert s.f_over_m == pytest.approx(0.2 / 0.52)
    assert s.b_over_bmsy == pytest.approx(1_000_000 / 500_000)   # 2.0
    assert s.f_over_fmsy == pytest.approx(0.2 / 0.34)
    assert s.kobe_quadrant == "green"                            # B>1, F<1


def test_compute_status_no_reference_points_coastal():
    # perch: no stock mapping → F/M only, no ratios
    mort = {"perch": _synthetic_mortality_df()}
    res = _FakeResults({"perch": _biomass_df(3_000_000)}, mort)
    snap = _snapshot(ref_points={}, units={}, mapping={"perch": []})
    s = {x.species: x for x in fz.compute_fisheries_status(res, snap, window_years=2, steps_per_year=2)}["perch"]
    assert s.has_reference_points is False
    assert s.b_over_bmsy is None and s.f_over_fmsy is None and s.kobe_quadrant is None
    assert s.f_over_m == pytest.approx(0.2 / 0.52)               # F/M still computed
    assert "no ICES reference point" in s.note.lower()


def test_compute_status_mixed_unit_excluded():
    # cod: western tonnes (fmsy set) + eastern index (no fmsy) → mixed → no ratio
    mort = {"cod": _synthetic_mortality_df()}
    res = _FakeResults({"cod": _biomass_df(2_000_000)}, mort)
    snap = _snapshot(
        ref_points={"cod.27.22-24": {"fmsy": 0.26, "msy_btrigger": 30_000},
                    "cod.27.24-32": {"fmsy": None, "msy_btrigger": None}},
        units={"cod.27.22-24": "tonnes", "cod.27.24-32": "index"},
        mapping={"cod": ["cod.27.22-24", "cod.27.24-32"]},
    )
    s = {x.species: x for x in fz.compute_fisheries_status(res, snap, window_years=2, steps_per_year=2)}["cod"]
    assert s.has_reference_points is False
    assert s.b_over_bmsy is None
    assert "mixed" in s.note.lower() or "index" in s.note.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k compute_status -v`
Expected: FAIL (`no attribute 'compute_fisheries_status'`).

- [ ] **Step 3: Implement the dataclass + compute**

Add to `osmose/validation/fisheries.py`:

```python
@dataclass(frozen=True)
class FisheriesStatus:
    species: str
    biomass_t: float
    fishing_mortality: float
    natural_mortality: float
    f_over_m: float | None
    b_over_bmsy: float | None = None
    f_over_fmsy: float | None = None
    kobe_quadrant: str | None = None
    has_reference_points: bool = False
    note: str = ""


def _reference_points_for(snapshot: IcesSnapshot, stocks: list[str]):
    """Return (bmsy, fmsy, note) for a species' stocks, or (None, None, note) if
    not all stocks are tonnes-unit with non-null fmsy + msy_btrigger.

    Bmsy proxy = sum of sub-stock msy_btrigger. Fmsy = simple mean of sub-stock fmsy.
    """
    units = snapshot.manifest.get("units_by_stock", {})
    if not stocks:
        return None, None, "no ICES reference point (no stock mapped)"
    index_stocks = [s for s in stocks if units.get(s) != "tonnes"]
    rps = [snapshot.reference_points.get(s, {}) for s in stocks]
    fmsys = [rp.get("fmsy") for rp in rps]
    btrigs = [rp.get("msy_btrigger") for rp in rps]
    if index_stocks:
        return None, None, f"no ICES reference point (index-unit/mixed: {index_stocks})"
    if any(f is None for f in fmsys) or any(b is None for b in btrigs):
        return None, None, "no ICES reference point (null fmsy/msy_btrigger)"
    bmsy = float(sum(btrigs))
    fmsy = float(sum(fmsys) / len(fmsys))
    return bmsy, fmsy, ""


def compute_fisheries_status(
    results: OsmoseResults,
    snapshot: IcesSnapshot,
    *,
    window_years: int = 10,
    steps_per_year: int | None = None,
) -> list[FisheriesStatus]:
    """Per-species fisheries stock-status diagnostics.

    steps_per_year: mortality-output saving steps per year. If None, infer from the
    mortality frame length and the biomass frame length (steps = mort_rows / n_years
    where n_years = biomass_rows, since biomass is annual). Callers may pass it explicitly.
    """
    mapping = snapshot.manifest.get("model_species_to_ices_stocks", {})
    out: list[FisheriesStatus] = []
    for species, stocks in mapping.items():
        try:
            biomass = model_biomass_window_mean(results, species, window_years=window_years)
            mort = results.mortality(species=species)
        except (KeyError, ValueError) as e:
            print(f"WARN: skipping {species!r}: {e}", file=sys.stderr)
            continue
        spy = steps_per_year if steps_per_year is not None else _infer_steps_per_year(results, species, mort)
        f = _annual_fishing_mortality(mort, spy, window_years)
        m = _annual_natural_mortality(mort, spy, window_years)
        f_over_m = (f / m) if m > 0 else None
        bmsy, fmsy, note = _reference_points_for(snapshot, stocks)
        if bmsy is not None and fmsy is not None and bmsy > 0 and fmsy > 0:
            b_ratio = biomass / bmsy
            f_ratio = f / fmsy
            out.append(FisheriesStatus(
                species=species, biomass_t=biomass, fishing_mortality=f, natural_mortality=m,
                f_over_m=f_over_m, b_over_bmsy=b_ratio, f_over_fmsy=f_ratio,
                kobe_quadrant=kobe_quadrant(b_ratio, f_ratio), has_reference_points=True, note="",
            ))
        else:
            out.append(FisheriesStatus(
                species=species, biomass_t=biomass, fishing_mortality=f, natural_mortality=m,
                f_over_m=f_over_m, has_reference_points=False, note=note,
            ))
    return out


def _infer_steps_per_year(results, species, mort) -> int:
    """Infer mortality saving-steps-per-year as len(mortality) / len(biomass-years)."""
    try:
        n_years = len(results.biomass(species=species))
        n_steps = len(mort)
        spy = max(1, round(n_steps / n_years)) if n_years else 1
        return spy
    except Exception:
        return 1
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k compute_status -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fisheries.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): FisheriesStatus + compute_fisheries_status (ref-point rule)"
```

---

## Task 4: Markdown report

**Files:**
- Modify: `osmose/validation/fisheries.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Write failing test**

```python
def test_format_report_includes_ratios_and_gap_note():
    statuses = [
        fz.FisheriesStatus("sprat", 1_000_000, 0.2, 0.52, 0.385, 2.0, 0.588, "green", True, ""),
        fz.FisheriesStatus("perch", 3_000_000, 0.2, 0.52, 0.385, None, None, None, False,
                           "no ICES reference point (no stock mapped)"),
    ]
    md = fz.format_fisheries_report(statuses)
    assert "sprat" in md and "perch" in md
    assert "green" in md
    assert "B/Bmsy" in md and "F/Fmsy" in md and "F/M" in md
    assert "no ICES reference point" in md
    assert "1/2 species with ICES reference points" in md or "1/2" in md
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k format_report -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add to `osmose/validation/fisheries.py`:

```python
def format_fisheries_report(statuses: list[FisheriesStatus], *, window_years: int = 10) -> str:
    """Markdown table of fisheries stock-status diagnostics."""
    lines = [
        "# OSMOSE fisheries stock-status diagnostics",
        "",
        f"Model window: last {window_years} years. B/Bmsy uses ICES MSY Btrigger as the "
        "Bmsy proxy; F/Fmsy uses ICES Fmsy. Ratios shown only where all ICES sub-stocks "
        "are tonnes-unit with non-null reference points.",
        "",
        "| species | B (t) | F | M | F/M | B/Bmsy | F/Fmsy | Kobe | note |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    n_ref = 0
    for s in statuses:
        fm = f"{s.f_over_m:.2f}" if s.f_over_m is not None else "—"
        if s.has_reference_points:
            n_ref += 1
            bb = f"{s.b_over_bmsy:.2f}"
            ff = f"{s.f_over_fmsy:.2f}"
            kobe = s.kobe_quadrant or "—"
            note = "—"
        else:
            bb = ff = kobe = "—"
            note = s.note or "—"
        lines.append(
            f"| {s.species} | {s.biomass_t:,.0f} | {s.fishing_mortality:.3f} | "
            f"{s.natural_mortality:.3f} | {fm} | {bb} | {ff} | {kobe} | {note} |"
        )
    lines += ["", f"**Summary:** {n_ref}/{len(statuses)} species with ICES reference points.", ""]
    return "\n".join(lines)
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -v` (all pass).
```bash
git add osmose/validation/fisheries.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): markdown stock-status report"
```

---

## Task 5: Plotting — Kobe plot + F/M bars

**Files:**
- Modify: `osmose/plotting.py`
- Test: `tests/test_validation_fisheries.py` (smoke: figures build)

- [ ] **Step 1: Write failing smoke tests**

```python
def test_plots_build():
    from osmose import plotting
    statuses = [
        fz.FisheriesStatus("sprat", 1_000_000, 0.2, 0.52, 0.385, 2.0, 0.588, "green", True, ""),
        fz.FisheriesStatus("perch", 3_000_000, 0.2, 0.52, 0.385, None, None, None, False, "no ref"),
    ]
    kobe = plotting.make_kobe_plot(statuses)
    bars = plotting.make_fm_ratio_bars(statuses)
    assert kobe is not None and bars is not None
    # Kobe shows only species WITH reference points (1 of 2)
    assert sum(len(t.x) for t in kobe.data if hasattr(t, "x") and t.x is not None) >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k plots_build -v`
Expected: FAIL (`no attribute 'make_kobe_plot'`).

- [ ] **Step 3: Implement in `osmose/plotting.py`**

Add (after an existing `make_*`; the module already has `import plotly.graph_objects as go` and `from osmose.plotly_theme import PLOTLY_TEMPLATE as TEMPLATE`):

```python
def make_kobe_plot(statuses) -> go.Figure:
    """Kobe plot: B/Bmsy (x) vs F/Fmsy (y), four stock-status quadrants.

    Only species with ICES reference points (has_reference_points) are plotted.
    """
    pts = [s for s in statuses if getattr(s, "has_reference_points", False)]
    fig = go.Figure()
    # quadrant shading (axes 0..2 each)
    quad = [(0, 1, 1, 2, "rgba(214,39,40,0.12)"),   # red: B<1,F>1 (NW)
            (1, 2, 1, 2, "rgba(255,127,14,0.12)"),  # orange: B>1,F>1 (NE)
            (0, 1, 0, 1, "rgba(255,221,0,0.12)"),   # yellow: B<1,F<1 (SW)
            (1, 2, 0, 1, "rgba(44,160,44,0.12)")]   # green: B>1,F<1 (SE)
    for x0, x1, y0, y1, color in quad:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color,
                      line=dict(width=0), layer="below")
    fig.add_hline(y=1.0, line=dict(dash="dash", width=1))
    fig.add_vline(x=1.0, line=dict(dash="dash", width=1))
    if pts:
        fig.add_trace(go.Scatter(
            x=[s.b_over_bmsy for s in pts], y=[s.f_over_fmsy for s in pts],
            mode="markers+text", text=[s.species for s in pts], textposition="top center",
            marker=dict(size=12), name="stocks",
        ))
    fig.update_layout(
        title=dict(text="Kobe plot — stock status (B/Bmsy vs F/Fmsy)"),
        xaxis=dict(title="B / Bmsy", range=[0, 2]),
        yaxis=dict(title="F / Fmsy", range=[0, 2]),
        template=TEMPLATE,
    )
    return fig


def make_fm_ratio_bars(statuses) -> go.Figure:
    """F/M ratio per species (available for all species; reference line at F/M=1)."""
    valid = [s for s in statuses if getattr(s, "f_over_m", None) is not None]
    fig = go.Figure(go.Bar(x=[s.species for s in valid], y=[s.f_over_m for s in valid], name="F/M"))
    fig.add_hline(y=1.0, line=dict(dash="dash", width=1))
    fig.update_layout(title=dict(text="Fishing vs natural mortality (F/M)"),
                      xaxis=dict(title="species"), yaxis=dict(title="F / M"), template=TEMPLATE)
    return fig
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k plots_build -v` (PASS).
Run: `.venv/bin/ruff check osmose/plotting.py osmose/validation/fisheries.py tests/test_validation_fisheries.py`.
```bash
git add osmose/plotting.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): Kobe plot + F/M bar chart"
```

---

## Task 6: CLI — `scripts/compute_fisheries_diagnostics.py`

**Files:**
- Create: `scripts/compute_fisheries_diagnostics.py`
- Test: `tests/test_validation_fisheries.py` (argparse smoke)

- [ ] **Step 1: Implement the CLI (mirror `validate_outputs_vs_ices.py`)**

```python
#!/usr/bin/env python3
"""Compute fisheries stock-status diagnostics (F/M, B/Bmsy, F/Fmsy, Kobe) for a run.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compute_fisheries_diagnostics.py \\
        --results-dir <path> \\
        [--snapshots-dir data/baltic/reference/ices_snapshots] \\
        [--window-years 10] [--prefix osm] \\
        [--report out.md] [--json out.json] [--plot out_prefix]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "baltic" / "reference" / "ices_snapshots"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    p.add_argument("--window-years", type=int, default=10)
    p.add_argument("--prefix", type=str, default="osm")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--plot", type=str, default=None, help="output path prefix for kobe/fm plots (html)")
    args = p.parse_args(argv)

    from osmose.results import OsmoseResults
    from osmose.validation.ices import load_snapshot
    from osmose.validation import fisheries as fz

    results = OsmoseResults(args.results_dir, prefix=args.prefix)
    snapshot = load_snapshot(args.snapshots_dir)
    statuses = fz.compute_fisheries_status(results, snapshot, window_years=args.window_years)

    report = fz.format_fisheries_report(statuses, window_years=args.window_years)
    if args.report:
        args.report.write_text(report)
    print(report)

    if args.json:
        args.json.write_text(json.dumps([asdict(s) for s in statuses], indent=2))

    if args.plot:
        from osmose import plotting
        plotting.make_kobe_plot(statuses).write_html(f"{args.plot}_kobe.html")
        plotting.make_fm_ratio_bars(statuses).write_html(f"{args.plot}_fm.html")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

(Confirm the `OsmoseResults(dir, prefix=...)` constructor signature matches `validate_outputs_vs_ices.py`'s usage — `grep -n "OsmoseResults(" scripts/validate_outputs_vs_ices.py`; adjust the prefix kwarg if it differs.)

- [ ] **Step 2: Argparse smoke test**

```python
def test_cli_parses_and_help():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cfd", "scripts/compute_fisheries_diagnostics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.main(["--help"])
```

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k cli -v`
Expected: PASS.

- [ ] **Step 3: Real-run smoke (if a Baltic output dir exists)**

If `data/baltic/output/` has a finished run, run the CLI end-to-end:
`PYTHONPATH=. .venv/bin/python scripts/compute_fisheries_diagnostics.py --results-dir data/baltic/output --prefix baltic --window-years 10 2>&1 | tail -20`
Expected: a markdown table with F/M for all 8 species and B/Bmsy + F/Fmsy + Kobe for the ~4 ICES-covered stocks; coastal species show "no ICES reference point". If `--prefix` differs for the Baltic run, find it (`ls data/baltic/output/*biomass*`). Confirm no crash and the F values are plausible (0–2 yr⁻¹). Document the observed output in the commit.

- [ ] **Step 4: Commit**

```bash
git add scripts/compute_fisheries_diagnostics.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): compute_fisheries_diagnostics CLI"
```

---

## Task 7: Docs + finalize

**Files:**
- Modify: a usage/feature doc (find: `grep -rl "validate_outputs_vs_ices" docs/`)

- [ ] **Step 1: Add a CLI note**

Document `scripts/compute_fisheries_diagnostics.py` alongside `validate_outputs_vs_ices.py`: what it computes (F/M, B/Bmsy, F/Fmsy, Kobe), the Bmsy=MSY-Btrigger proxy caveat, and the honest coverage gap (only ~4 ICES tonnes-unit Baltic stocks get ratios; coastal species + eastern cod show F/M + biomass only).

- [ ] **Step 2: Full FR-validation suite + lint**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -v` (all pass; report count).
Run: `.venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/ tests/` (clean — `scripts/` is not CI-format-checked; `ruff check` it separately).
Run: `.venv/bin/python -m pytest tests/test_validation_ices.py -q` (the sibling validator still green — confirm the new module didn't perturb `ices.py`).

- [ ] **Step 3: Commit + finish**

```bash
git add docs/
git commit -m "docs(fisheries): document the fisheries-diagnostics CLI"
```

Use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** §architecture lib → T1–T4 (`fisheries.py`: F/M helpers, Kobe classifier, dataclass+compute, report); plotting → T5 (Kobe + F/M bars); CLI → T6; tests → T1–T6; docs → T7. Reference-point mixed-unit rule (spec's explicit rule) → T3 `_reference_points_for` (index/null → no ratio) + `test_compute_status_mixed_unit_excluded`. Bmsy=MSY-Btrigger proxy → T3 + report text + T7 doc. Coverage-gap honesty → T3 (`has_reference_points`/`note`) + T4 report + plots only-with-refs (T5). Shiny deferred → not in plan (YAGNI, per spec). ✅

**Placeholder scan:** no TBD/TODO; every code step has complete code. Two verification steps (T1 Step5 real-CSV shape, T6 Step3 real-run smoke) are explicit "verify and adjust" against real data with named fallbacks — not placeholders. ✅

**Type consistency:** `FisheriesStatus` fields (biomass_t, fishing_mortality, natural_mortality, f_over_m, b_over_bmsy, f_over_fmsy, kobe_quadrant, has_reference_points, note) identical across T3 dataclass, T4 report, T5 plots, T6 CLI. `kobe_quadrant()` returns {green,orange,yellow,red} consistently (T2 ↔ T3 ↔ T5 quadrant colors). `_annual_fishing_mortality`/`_annual_natural_mortality`/`_recruits_series`/`_windowed_annual_mean` signatures consistent T1↔T3. `steps_per_year` threaded consistently (compute → helpers; inferred when None). ✅
