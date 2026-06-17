# Community Size-Spectrum Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the canonical Sheldon (body-mass) normalized biomass spectrum plus community-level indicators (Mean Trophic Level / Marine Trophic Index, community totals, size diversity, Warwick ABC W-statistic) to the OSMOSE diagnostics.

**Architecture:** A new pure module `osmose/community_metrics.py` holds three independent compute units + an orchestrator + a markdown formatter, reusing `osmose/size_spectrum.py`'s readers and `osmose/results.OsmoseResults`. Two plot helpers go in `osmose/plotting.py`; the Diagnostics page (`ui/pages/results.py`) gains two chart entries and a metrics panel.

**Tech Stack:** Python 3.12, pandas, numpy (via `osmose.analysis`), plotly, Shiny for Python; pytest. Run everything with `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-17-community-size-spectrum-extension-design.md`

## Key facts (read before starting)

- The community by-size file is wide `Time, Size, <species…>`; `Size` is a LENGTH (cm) lower-edge axis. Reader: `osmose.size_spectrum._read_community_by_size(output_dir, output_type, prefix)` (raises `FileNotFoundError` if absent). Bin width helper: `osmose.size_spectrum._infer_bin_width(edges)`. Window helper: `osmose.size_spectrum._window_by_time(df, time_col, window_years)` (keeps rows with `time_col > tmax - window_years`; raises `ValueError` if `window_years < 1`).
- `OsmoseResults(output_dir, prefix="osm", strict=True)` 1D accessors `.biomass()`, `.abundance()`, `.mean_trophic_level()` each return a **WIDE** frame: a `Time` column, one column per species (species NAME as the column), and a filename-derived `species` meta column (value `"all"` for the combined file). With `strict=True` a missing file raises `FileNotFoundError`; with `strict=False` it returns an empty `DataFrame`. (Do NOT assume long-form `time/species/value` — these are wide.)
- `osmose.analysis.size_spectrum_slope(df[size, abundance]) -> (slope, intercept, r2)`; raises `ValueError` if fewer than 2 strictly-positive `(size, abundance)` pairs.
- Config is a flat `dict[str, str]`: `simulation.nspecies`, `species.name.sp{i}`, `species.length2weight.condition.factor.sp{i}` (= a), `species.length2weight.allometric.power.sp{i}` (= b).
- Synthetic test fixtures write a clean-header CSV (no preamble), e.g. `pd.DataFrame(rows, columns=[...]).to_csv(path, index=False)`. The by-size file must be named `{prefix}_{type}_Simu0.csv` (e.g. `osm_biomassDistribBySize_Simu0.csv`); the 1D files `osm_biomass_Simu0.csv`, `osm_abundance_Simu0.csv`, `osm_meanTL_Simu0.csv`. `_matches_output_type` prevents `osm_biomass*` from also matching `osm_biomassDistribBySize*`, so both can live in the same dir.

## File structure

- **Create** `osmose/community_metrics.py` — shared helpers, three compute units, orchestrator, formatter.
- **Create** `tests/test_community_metrics.py` — unit + degradation + real-data tests.
- **Modify** `osmose/plotting.py` — add `make_sheldon_spectrum_plot`, `make_abc_plot`.
- **Modify** `tests/test_plotting.py` — tests for the two new plot helpers.
- **Modify** `ui/pages/results.py` — two new chart rtypes + a Community Metrics panel.
- **Modify** `tests/test_trophic_network.py`? No. Wiring assertion goes in `tests/test_community_metrics.py`.

---

## Task 1: Shared helpers + module skeleton

**Files:**
- Create: `osmose/community_metrics.py`
- Create: `tests/test_community_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_metrics.py
from __future__ import annotations

import math

import pandas as pd
import pytest

from osmose.community_metrics import (
    _per_species_window_mean,
    _species_columns,
    _species_lw_coeffs,
    _to_float,
)


def test_to_float_handles_bad_values():
    assert _to_float("1.5") == 1.5
    assert _to_float(None) is None
    assert _to_float("abc") is None


def test_species_columns_excludes_meta():
    df = pd.DataFrame(columns=["Time", "Size", "species", "cod", "herring"])
    assert _species_columns(df) == ["cod", "herring"]


def test_per_species_window_mean_windows_and_means():
    # Time 1..12; window_years=3 keeps Time > 12-3 = 9 -> {10,11,12}. cod mean of 10,20,30 = 20.
    df = pd.DataFrame(
        {
            "Time": [float(t) for t in range(1, 13)],
            "cod": [0.0] * 9 + [10.0, 20.0, 30.0],
            "species": ["all"] * 12,
        }
    )
    out = _per_species_window_mean(df, window_years=3)
    assert out == {"cod": pytest.approx(20.0)}


def test_per_species_window_mean_empty_df():
    assert _per_species_window_mean(pd.DataFrame(), window_years=10) == {}


def test_species_lw_coeffs_reads_config_and_skips_bad():
    config = {
        "simulation.nspecies": "2",
        "species.name.sp0": "cod",
        "species.length2weight.condition.factor.sp0": "0.01",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.name.sp1": "herring",
        "species.length2weight.condition.factor.sp1": "0",  # non-positive -> skip
        "species.length2weight.allometric.power.sp1": "3.0",
    }
    out = _species_lw_coeffs(config)
    assert out == {"cod": (0.01, 3.0)}


def test_species_lw_coeffs_empty_without_nspecies():
    assert _species_lw_coeffs({}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.community_metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# osmose/community_metrics.py
"""Community-level ecosystem-state diagnostics from OSMOSE output.

Adds the canonical Sheldon (body-mass) normalized biomass spectrum (NBSS) and a
suite of community indicators — Mean Trophic Level / Marine Trophic Index,
community totals + size diversity, and the Warwick Abundance-Biomass Comparison
(ABC) W-statistic — on top of the length spectrum in osmose.size_spectrum.

The Sheldon spectrum needs body MASS; OSMOSE writes by-LENGTH size classes, so we
convert per species via the config length-weight law W = a * L^b. Each unit fails
soft (records a note, returns degraded values) rather than raising past the
orchestrator. See docs/superpowers/specs/2026-06-17-community-size-spectrum-extension-design.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from osmose.analysis import size_spectrum_slope
from osmose.results import OsmoseResults
from osmose.size_spectrum import _infer_bin_width, _read_community_by_size, _window_by_time

_META_COLS = {"Time", "time", "Size", "size", "species", "Simu", "simu"}


def _to_float(value) -> float | None:
    """float(value) or None if value is None / non-numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _species_columns(df: pd.DataFrame) -> list[str]:
    """Column names that are species (exclude Time/Size/species/Simu meta columns)."""
    return [c for c in df.columns if c not in _META_COLS]


def _per_species_window_mean(df: pd.DataFrame, window_years: int) -> dict[str, float]:
    """{species: trailing-window mean value} from a WIDE 1D OsmoseResults frame.

    `df` has a Time column + one column per species (+ a 'species' meta column).
    Empty/Time-less frame -> {}. Non-numeric cells coerce to NaN before the mean.
    """
    if df.empty or "Time" not in df.columns:
        return {}
    windowed = _window_by_time(df, "Time", window_years)
    out: dict[str, float] = {}
    for c in _species_columns(windowed):
        out[c] = float(pd.to_numeric(windowed[c], errors="coerce").mean())
    return out


def _species_lw_coeffs(config: dict) -> dict[str, tuple[float, float]]:
    """{species_name: (a, b)} from config length-weight keys.

    Skips a species whose name is missing or whose a/b is missing or non-positive
    (a non-positive coefficient can't define a usable W = a * L^b mapping).
    """
    out: dict[str, tuple[float, float]] = {}
    n = _to_float(config.get("simulation.nspecies"))
    if n is None:
        return out
    for i in range(int(n)):
        name = config.get(f"species.name.sp{i}")
        a = _to_float(config.get(f"species.length2weight.condition.factor.sp{i}"))
        b = _to_float(config.get(f"species.length2weight.allometric.power.sp{i}"))
        if name and a is not None and b is not None and a > 0 and b > 0:
            out[str(name)] = (a, b)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/community_metrics.py tests/test_community_metrics.py
git commit -m "feat(community-metrics): shared helpers (window-mean, species cols, L-W coeffs)"
```

---

## Task 2: Sheldon mass spectrum + spectrum-derived metrics

**Files:**
- Modify: `osmose/community_metrics.py`
- Modify: `tests/test_community_metrics.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_community_metrics.py`)

```python
from osmose.community_metrics import SheldonSpectrum, compute_sheldon_spectrum


def _write_csv(path, rows, cols):
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def _sheldon_fixture(out_dir):
    # One species "cod", a=1, b=1 so mass == length-midpoint.
    # Size lower-edges 0 and 10 -> inferred width 10 -> midpoints 5 and 15 -> masses 5, 15.
    # w_ref=5: octave k(5)=floor(log2(1))=0, k(15)=floor(log2(3))=1.
    # biomass-at-size: 20 at Size 0, 10 at Size 10 -> bin biomass {k0:20, k1:10}.
    # widths {k0: 5*2^0=5, k1: 5*2^1=10} -> NBSS {4.0, 1.0}; midpoints {7.071, 14.142}.
    # slope of log10(NBSS) vs log10(midpoint) = (0 - 0.602)/(1.150 - 0.849) = -2.0.
    _write_csv(
        out_dir / "osm_biomassDistribBySize_Simu0.csv",
        [(1.0, 0.0, 20.0), (1.0, 10.0, 10.0)],
        ["Time", "Size", "cod"],
    )
    # Totals come from the 1D biomass/abundance files: total B=30, total A=6 -> mean mass=5.
    _write_csv(out_dir / "osm_biomass_Simu0.csv", [(1.0, 30.0)], ["Time", "cod"])
    _write_csv(out_dir / "osm_abundance_Simu0.csv", [(1.0, 6.0)], ["Time", "cod"])


_CONFIG = {
    "simulation.nspecies": "1",
    "species.name.sp0": "cod",
    "species.length2weight.condition.factor.sp0": "1.0",
    "species.length2weight.allometric.power.sp0": "1.0",
}


def test_sheldon_spectrum_bins_and_slope(tmp_path):
    _sheldon_fixture(tmp_path)
    spec = compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)
    assert isinstance(spec, SheldonSpectrum)
    assert spec.mass_bin_midpoints == pytest.approx([5 * 2 ** 0.5, 5 * 2 ** 1.5], rel=1e-6)
    assert spec.nbss_values == pytest.approx([4.0, 1.0], rel=1e-6)
    assert spec.slope == pytest.approx(-2.0, abs=1e-6)
    assert spec.n_bins_fit == 2
    assert spec.dropped_species == []


def test_sheldon_totals_and_diversity(tmp_path):
    _sheldon_fixture(tmp_path)
    spec = compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)
    assert spec.total_biomass == pytest.approx(30.0)
    assert spec.total_abundance == pytest.approx(6.0)
    assert spec.mean_body_mass == pytest.approx(5.0)
    # biomass shares [20/30, 10/30]; H = 0.6365; evenness = H/ln(2) = 0.9183.
    assert spec.size_diversity == pytest.approx(0.9183, abs=1e-3)


def test_sheldon_drops_species_without_coeffs(tmp_path):
    _write_csv(
        tmp_path / "osm_biomassDistribBySize_Simu0.csv",
        [(1.0, 0.0, 20.0, 5.0)],
        ["Time", "Size", "cod", "herring"],
    )
    spec = compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)  # config has cod only
    assert spec.dropped_species == ["herring"]
    assert "herring" in spec.note


def test_sheldon_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_sheldon_spectrum(tmp_path, _CONFIG, window_years=10)


def test_sheldon_bad_metric_raises(tmp_path):
    with pytest.raises(ValueError):
        compute_sheldon_spectrum(tmp_path, _CONFIG, metric="nonsense")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k sheldon -q`
Expected: FAIL with `ImportError: cannot import name 'SheldonSpectrum'`.

- [ ] **Step 3: Write minimal implementation** (append to `osmose/community_metrics.py`)

```python
@dataclass(frozen=True)
class SheldonSpectrum:
    metric: str
    mass_bin_edges: list[float]
    mass_bin_midpoints: list[float]
    nbss_values: list[float]
    slope: float | None
    intercept: float | None
    r_squared: float | None
    n_bins_fit: int
    size_diversity: float
    total_biomass: float
    total_abundance: float
    mean_body_mass: float
    window_years: int
    n_timesteps_used: int
    dropped_species: list[str]
    note: str


def _shannon_evenness(values: list[float]) -> float:
    """Shannon evenness H/ln(S) over positive shares; NaN if fewer than 2 positive bins."""
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return float("nan")
    total = sum(positive)
    shares = [v / total for v in positive]
    h = -sum(p * math.log(p) for p in shares)
    return float(h / math.log(len(positive)))


def compute_sheldon_spectrum(
    output_dir,
    config: dict,
    *,
    metric: str = "biomass",
    prefix: str = "osm",
    window_years: int = 10,
) -> SheldonSpectrum:
    """Canonical Sheldon NBSS over equal log2 (octave) body-mass bins + derived metrics.

    Reads the per-species {metric}DistribBySize file (does NOT sum species), converts
    each species' length midpoints to mass via config W = a*L^b, bins biomass into
    octaves, normalizes by bin width, and log-log fits the slope. Totals come from the
    1D biomass/abundance outputs (read non-strict; absent -> 0/NaN). Raises
    FileNotFoundError if the by-size file is missing; ValueError on a bad metric.
    """
    if metric not in ("biomass", "abundance"):
        raise ValueError("metric must be 'biomass' or 'abundance'")
    coeffs = _species_lw_coeffs(config or {})
    wide = _read_community_by_size(Path(output_dir), f"{metric}DistribBySize", prefix)
    windowed = _window_by_time(wide, "Time", window_years)
    n_steps = int(cast(pd.Series, windowed["Time"]).nunique())

    sp_cols = _species_columns(windowed)
    sizes = sorted({float(s) for s in windowed["Size"].unique()})
    width_len = _infer_bin_width(sizes)
    per_size = windowed.groupby("Size")[sp_cols].mean()  # index=Size lower-edge, cols=species

    notes: list[str] = []
    dropped: list[str] = []
    masses: list[float] = []
    vals: list[float] = []
    for sp in sp_cols:
        if sp not in coeffs:
            dropped.append(sp)
            continue
        a, b = coeffs[sp]
        for size_lo, value in per_size[sp].items():
            mid_len = float(size_lo) + width_len / 2.0
            mass = a * mid_len**b
            v = float(value)
            if mass > 0 and v > 0:
                masses.append(mass)
                vals.append(v)
    if dropped:
        notes.append(f"dropped {len(dropped)} species without usable a,b: {', '.join(dropped)}.")

    edges: list[float] = []
    midpoints: list[float] = []
    nbss: list[float] = []
    if masses:
        w_ref = min(masses)
        binned: dict[int, float] = defaultdict(float)
        for m, v in zip(masses, vals):
            k = int(math.floor(math.log2(m / w_ref)))
            binned[k] += v
        for k in sorted(binned):
            lower = w_ref * 2.0**k
            edges.append(lower)
            midpoints.append(w_ref * 2.0 ** (k + 0.5))
            nbss.append(binned[k] / lower)  # octave linear width == lower edge value
    else:
        notes.append("no positive mass bins (no species with usable a,b and data).")

    slope = intercept = r2 = None
    n_fit = sum(1 for m, v in zip(midpoints, nbss) if m > 0 and v > 0)
    if n_fit >= 2:
        try:
            slope, intercept, r2 = size_spectrum_slope(
                pd.DataFrame({"size": midpoints, "abundance": nbss})
            )
        except ValueError:
            notes.append("NBSS slope fit failed.")
    else:
        notes.append("fewer than 2 positive mass bins; NBSS slope undefined.")

    size_diversity = _shannon_evenness([b for b in nbss])

    res = OsmoseResults(Path(output_dir), prefix=prefix, strict=False)
    bm = _per_species_window_mean(res.biomass(), window_years)
    ab = _per_species_window_mean(res.abundance(), window_years)
    total_biomass = float(sum(bm.values()))
    total_abundance = float(sum(ab.values()))
    mean_body_mass = total_biomass / total_abundance if total_abundance > 0 else float("nan")

    return SheldonSpectrum(
        metric=metric,
        mass_bin_edges=edges,
        mass_bin_midpoints=midpoints,
        nbss_values=nbss,
        slope=slope,
        intercept=intercept,
        r_squared=r2,
        n_bins_fit=n_fit,
        size_diversity=size_diversity,
        total_biomass=total_biomass,
        total_abundance=total_abundance,
        mean_body_mass=mean_body_mass,
        window_years=window_years,
        n_timesteps_used=n_steps,
        dropped_species=dropped,
        note=" ".join(notes),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k sheldon -q`
Expected: PASS (5 sheldon tests). Then run the full file: `.venv/bin/python -m pytest tests/test_community_metrics.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/community_metrics.py tests/test_community_metrics.py
git commit -m "feat(community-metrics): Sheldon NBSS mass spectrum + size diversity + totals"
```

---

## Task 3: Trophic indicators (MTL / MTI)

**Files:**
- Modify: `osmose/community_metrics.py`
- Modify: `tests/test_community_metrics.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.community_metrics import TrophicIndicators, compute_trophic_indicators


def test_trophic_mtl_and_mti(tmp_path):
    # cod TL 4.0 biomass 30; herring TL 3.0 biomass 10.
    # MTL = (4*30 + 3*10)/40 = 3.75. MTI (cutoff 3.25) keeps cod only -> 4.0.
    _write_csv(tmp_path / "osm_meanTL_Simu0.csv", [(1.0, 4.0, 3.0)], ["Time", "cod", "herring"])
    _write_csv(tmp_path / "osm_biomass_Simu0.csv", [(1.0, 30.0, 10.0)], ["Time", "cod", "herring"])
    ind = compute_trophic_indicators(tmp_path, window_years=10)
    assert isinstance(ind, TrophicIndicators)
    assert ind.mtl == pytest.approx(3.75)
    assert ind.mti == pytest.approx(4.0)
    assert ind.n_species == 2
    assert ind.n_species_above_cutoff == 1


def test_trophic_equal_weights_without_biomass(tmp_path):
    # No biomass file -> equal weights. MTL = (4+3)/2 = 3.5.
    _write_csv(tmp_path / "osm_meanTL_Simu0.csv", [(1.0, 4.0, 3.0)], ["Time", "cod", "herring"])
    ind = compute_trophic_indicators(tmp_path, window_years=10)
    assert ind.mtl == pytest.approx(3.5)
    assert "equal weights" in ind.note


def test_trophic_missing_meanTL_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_trophic_indicators(tmp_path, window_years=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k trophic -q`
Expected: FAIL with `ImportError: cannot import name 'TrophicIndicators'`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
@dataclass(frozen=True)
class TrophicIndicators:
    mtl: float
    mti: float
    mti_tl_cutoff: float
    n_species: int
    n_species_above_cutoff: int
    window_years: int
    note: str


def compute_trophic_indicators(
    output_dir,
    *,
    prefix: str = "osm",
    window_years: int = 10,
    mti_tl_cutoff: float = 3.25,
) -> TrophicIndicators:
    """Biomass-weighted community Mean Trophic Level + Marine Trophic Index.

    MTL = biomass-weighted mean of per-species mean TL. MTI = same but only over
    species with mean TL >= mti_tl_cutoff (Pauly & Watson; default 3.25). meanTL is
    required (strict read raises FileNotFoundError if absent); biomass is the weight
    (read non-strict; if absent, equal weights with a note).
    """
    res = OsmoseResults(Path(output_dir), prefix=prefix, strict=True)
    tl = _per_species_window_mean(res.mean_trophic_level(), window_years)
    res_soft = OsmoseResults(Path(output_dir), prefix=prefix, strict=False)
    bm = _per_species_window_mean(res_soft.biomass(), window_years)

    species = [s for s in tl if not math.isnan(tl[s])]
    if not species:
        return TrophicIndicators(
            float("nan"), float("nan"), mti_tl_cutoff, 0, 0, window_years,
            "no usable meanTL values.",
        )

    note = ""
    weights = {s: bm.get(s, 0.0) for s in species}
    if sum(weights.values()) <= 0:
        weights = {s: 1.0 for s in species}
        note = "no biomass output; trophic indices use equal weights."

    wsum = sum(weights.values())
    mtl = sum(tl[s] * weights[s] for s in species) / wsum
    above = [s for s in species if tl[s] >= mti_tl_cutoff]
    if above:
        wabove = sum(weights[s] for s in above)
        mti = sum(tl[s] * weights[s] for s in above) / wabove
    else:
        mti = float("nan")

    return TrophicIndicators(
        float(mtl), float(mti), mti_tl_cutoff, len(species), len(above), window_years, note
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k trophic -q`
Expected: PASS (3 trophic tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/community_metrics.py tests/test_community_metrics.py
git commit -m "feat(community-metrics): Mean Trophic Level + Marine Trophic Index"
```

---

## Task 4: ABC / W-statistic

**Files:**
- Modify: `osmose/community_metrics.py`
- Modify: `tests/test_community_metrics.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.community_metrics import ABCResult, compute_abc


def test_abc_undisturbed_w_positive(tmp_path):
    # biomass dominated by cod (80/20) vs more-even abundance (55/45):
    # cumB=[80,100], cumA=[55,100]; W = (80-55 + 0)/(50*1) = 0.5.
    _write_csv(tmp_path / "osm_biomass_Simu0.csv", [(1.0, 80.0, 20.0)], ["Time", "cod", "herring"])
    _write_csv(tmp_path / "osm_abundance_Simu0.csv", [(1.0, 55.0, 45.0)], ["Time", "cod", "herring"])
    abc = compute_abc(tmp_path, window_years=10)
    assert isinstance(abc, ABCResult)
    assert abc.w_statistic == pytest.approx(0.5)
    assert abc.n_species == 2
    assert abc.cum_biomass_pct == pytest.approx([80.0, 100.0])
    assert abc.cum_abundance_pct == pytest.approx([55.0, 100.0])


def test_abc_disturbed_w_negative(tmp_path):
    _write_csv(tmp_path / "osm_biomass_Simu0.csv", [(1.0, 55.0, 45.0)], ["Time", "cod", "herring"])
    _write_csv(tmp_path / "osm_abundance_Simu0.csv", [(1.0, 80.0, 20.0)], ["Time", "cod", "herring"])
    assert compute_abc(tmp_path, window_years=10).w_statistic == pytest.approx(-0.5)


def test_abc_even_w_zero(tmp_path):
    _write_csv(tmp_path / "osm_biomass_Simu0.csv", [(1.0, 50.0, 50.0)], ["Time", "cod", "herring"])
    _write_csv(tmp_path / "osm_abundance_Simu0.csv", [(1.0, 50.0, 50.0)], ["Time", "cod", "herring"])
    assert compute_abc(tmp_path, window_years=10).w_statistic == pytest.approx(0.0)


def test_abc_single_species_undefined(tmp_path):
    _write_csv(tmp_path / "osm_biomass_Simu0.csv", [(1.0, 50.0)], ["Time", "cod"])
    _write_csv(tmp_path / "osm_abundance_Simu0.csv", [(1.0, 50.0)], ["Time", "cod"])
    abc = compute_abc(tmp_path, window_years=10)
    assert math.isnan(abc.w_statistic)
    assert abc.n_species == 1


def test_abc_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_abc(tmp_path, window_years=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k abc -q`
Expected: FAIL with `ImportError: cannot import name 'ABCResult'`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
@dataclass(frozen=True)
class ABCResult:
    w_statistic: float
    ranks: list[int]
    cum_biomass_pct: list[float]
    cum_abundance_pct: list[float]
    n_species: int
    window_years: int
    note: str


def compute_abc(output_dir, *, prefix: str = "osm", window_years: int = 10) -> ABCResult:
    """Warwick Abundance-Biomass Comparison W-statistic + cumulative dominance curves.

    Ranks species separately by biomass and by abundance (descending), builds the two
    cumulative %-dominance curves, and computes W = sum(Bi - Ai) / (50*(S-1)) over the
    curves (Warwick 1986). W > 0 => biomass-dominated (undisturbed); W < 0 => disturbed.
    Both 1D outputs are required (strict read raises FileNotFoundError if absent).
    """
    res = OsmoseResults(Path(output_dir), prefix=prefix, strict=True)
    bm = _per_species_window_mean(res.biomass(), window_years)
    ab = _per_species_window_mean(res.abundance(), window_years)
    species = sorted(set(bm) & set(ab))
    n = len(species)
    if n < 2:
        return ABCResult(float("nan"), [], [], [], n, window_years,
                         "need >= 2 species for ABC; W undefined.")

    b_sorted = sorted((bm[s] for s in species), reverse=True)
    a_sorted = sorted((ab[s] for s in species), reverse=True)
    bt, at = sum(b_sorted), sum(a_sorted)
    if bt <= 0 or at <= 0:
        return ABCResult(float("nan"), [], [], [], n, window_years,
                         "zero total biomass or abundance; W undefined.")

    cum_b: list[float] = []
    cum_a: list[float] = []
    cb = ca = 0.0
    for bv, av in zip(b_sorted, a_sorted):
        cb += bv
        ca += av
        cum_b.append(100.0 * cb / bt)
        cum_a.append(100.0 * ca / at)
    w = sum(b - a for b, a in zip(cum_b, cum_a)) / (50.0 * (n - 1))
    return ABCResult(float(w), list(range(1, n + 1)), cum_b, cum_a, n, window_years, "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k abc -q`
Expected: PASS (5 abc tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/community_metrics.py tests/test_community_metrics.py
git commit -m "feat(community-metrics): Warwick ABC W-statistic + dominance curves"
```

---

## Task 5: Orchestrator + markdown formatter

**Files:**
- Modify: `osmose/community_metrics.py`
- Modify: `tests/test_community_metrics.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.community_metrics import (
    CommunityDiagnostics,
    community_report,
    format_community_report,
)


def _full_fixture(out_dir):
    _sheldon_fixture(out_dir)  # writes biomassDistribBySize + biomass + abundance (cod)
    _write_csv(out_dir / "osm_meanTL_Simu0.csv", [(1.0, 4.0)], ["Time", "cod"])


def test_community_report_assembles_all(tmp_path):
    _full_fixture(tmp_path)
    diag = community_report(tmp_path, _CONFIG, window_years=10)
    assert isinstance(diag, CommunityDiagnostics)
    assert diag.sheldon is not None and diag.sheldon.slope == pytest.approx(-2.0, abs=1e-6)
    assert diag.trophic is not None and diag.trophic.mtl == pytest.approx(4.0)
    # one species -> ABC undefined (kept, not None)
    assert diag.abc is not None and math.isnan(diag.abc.w_statistic)


def test_community_report_without_config_skips_sheldon(tmp_path):
    _full_fixture(tmp_path)
    diag = community_report(tmp_path, None, window_years=10)
    assert diag.sheldon is None
    assert any("config" in n.lower() for n in diag.notes)
    assert diag.trophic is not None  # trophic still computed


def test_community_report_missing_outputs_degrade(tmp_path):
    # empty dir: every unit's required file is absent -> all None, notes recorded, no raise.
    diag = community_report(tmp_path, _CONFIG, window_years=10)
    assert diag.sheldon is None and diag.trophic is None and diag.abc is None
    assert len(diag.notes) >= 1


def test_format_community_report_renders_present_sections(tmp_path):
    _full_fixture(tmp_path)
    md = format_community_report(community_report(tmp_path, _CONFIG, window_years=10))
    assert "# OSMOSE community diagnostics" in md
    assert "Sheldon" in md and "Mean Trophic Level" in md and "W-statistic" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k "report or format" -q`
Expected: FAIL with `ImportError: cannot import name 'CommunityDiagnostics'`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
@dataclass(frozen=True)
class CommunityDiagnostics:
    sheldon: SheldonSpectrum | None
    trophic: TrophicIndicators | None
    abc: ABCResult | None
    notes: list[str]


def community_report(
    output_dir,
    config: dict | None = None,
    *,
    prefix: str = "osm",
    window_years: int = 10,
    metric: str = "biomass",
) -> CommunityDiagnostics:
    """Assemble the full community diagnostics bundle; each unit degrades to None on
    missing input (recording a top-level note) rather than raising."""
    notes: list[str] = []

    sheldon: SheldonSpectrum | None = None
    if config:
        try:
            sheldon = compute_sheldon_spectrum(
                output_dir, config, metric=metric, prefix=prefix, window_years=window_years
            )
        except FileNotFoundError as exc:
            notes.append(f"Sheldon (mass) spectrum unavailable: {exc}")
    else:
        notes.append("No config provided; Sheldon spectrum, size diversity and totals skipped.")

    trophic: TrophicIndicators | None = None
    try:
        trophic = compute_trophic_indicators(output_dir, prefix=prefix, window_years=window_years)
    except FileNotFoundError as exc:
        notes.append(f"Trophic indicators unavailable: {exc}")

    abc: ABCResult | None = None
    try:
        abc = compute_abc(output_dir, prefix=prefix, window_years=window_years)
    except FileNotFoundError as exc:
        notes.append(f"ABC / W-statistic unavailable: {exc}")

    return CommunityDiagnostics(sheldon=sheldon, trophic=trophic, abc=abc, notes=notes)


def _fmt(value: float | None, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def format_community_report(diag: CommunityDiagnostics) -> str:
    """Markdown summary of whichever community-diagnostic sections are present."""
    lines = ["# OSMOSE community diagnostics", ""]

    if diag.sheldon is not None:
        s = diag.sheldon
        lines += [
            "## Sheldon (body-mass) normalized biomass spectrum",
            "",
            f"- NBSS slope: {_fmt(s.slope)} "
            f"(intercept {_fmt(s.intercept)}, R²={_fmt(s.r_squared)}, n_bins_fit={s.n_bins_fit})",
            f"- Size diversity (Shannon evenness over mass bins): {_fmt(s.size_diversity)}",
            f"- Community total biomass: {_fmt(s.total_biomass, '.6g')}; "
            f"total abundance: {_fmt(s.total_abundance, '.6g')}; "
            f"mean body mass: {_fmt(s.mean_body_mass, '.6g')}",
            f"- Window: last {s.window_years} yr ({s.n_timesteps_used} timesteps)",
        ]
        if s.note:
            lines.append(f"- _Note: {s.note}_")
        lines.append("")

    if diag.trophic is not None:
        t = diag.trophic
        lines += [
            "## Trophic indicators",
            "",
            f"- Mean Trophic Level (biomass-weighted): {_fmt(t.mtl)}",
            f"- Marine Trophic Index (TL ≥ {_fmt(t.mti_tl_cutoff, '.2f')}): {_fmt(t.mti)} "
            f"({t.n_species_above_cutoff}/{t.n_species} species)",
        ]
        if t.note:
            lines.append(f"- _Note: {t.note}_")
        lines.append("")

    if diag.abc is not None:
        a = diag.abc
        lines += [
            "## Abundance-Biomass Comparison (ABC)",
            "",
            f"- W-statistic: {_fmt(a.w_statistic)} "
            f"({'biomass-dominated / undisturbed' if (a.w_statistic == a.w_statistic and a.w_statistic > 0) else 'abundance-dominated / disturbed' if (a.w_statistic == a.w_statistic and a.w_statistic < 0) else 'n/a'})",
            f"- Species ranked: {a.n_species}",
        ]
        if a.note:
            lines.append(f"- _Note: {a.note}_")
        lines.append("")

    if diag.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in diag.notes]

    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add osmose/community_metrics.py tests/test_community_metrics.py
git commit -m "feat(community-metrics): community_report orchestrator + markdown formatter"
```

---

## Task 6: Plot helpers

**Files:**
- Modify: `osmose/plotting.py`
- Modify: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_plotting.py`)

```python
def test_make_sheldon_spectrum_plot_builds():
    from osmose.community_metrics import SheldonSpectrum
    from osmose.plotting import make_sheldon_spectrum_plot

    spec = SheldonSpectrum(
        metric="biomass",
        mass_bin_edges=[5.0, 10.0],
        mass_bin_midpoints=[7.07, 14.14],
        nbss_values=[4.0, 1.0],
        slope=-2.0,
        intercept=1.0,
        r_squared=1.0,
        n_bins_fit=2,
        size_diversity=0.9,
        total_biomass=30.0,
        total_abundance=6.0,
        mean_body_mass=5.0,
        window_years=10,
        n_timesteps_used=1,
        dropped_species=[],
        note="",
    )
    fig = make_sheldon_spectrum_plot(spec)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # NBSS scatter (+ regression line)


def test_make_sheldon_spectrum_plot_empty():
    from osmose.community_metrics import SheldonSpectrum
    from osmose.plotting import make_sheldon_spectrum_plot

    spec = SheldonSpectrum("biomass", [], [], [], None, None, None, 0, float("nan"),
                           0.0, 0.0, float("nan"), 10, 0, [], "empty")
    assert isinstance(make_sheldon_spectrum_plot(spec), go.Figure)


def test_make_abc_plot_builds():
    from osmose.community_metrics import ABCResult
    from osmose.plotting import make_abc_plot

    abc = ABCResult(0.5, [1, 2], [80.0, 100.0], [55.0, 100.0], 2, 10, "")
    fig = make_abc_plot(abc)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # biomass + abundance dominance curves
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_plotting.py -k "sheldon or abc" -q`
Expected: FAIL with `ImportError: cannot import name 'make_sheldon_spectrum_plot'`.

- [ ] **Step 3: Write minimal implementation** (append to `osmose/plotting.py`; uses the existing `_empty_figure` helper and `go`/`np` already imported in the module)

```python
def make_sheldon_spectrum_plot(spec) -> go.Figure:
    """Log-log Sheldon NBSS scatter (mass-bin midpoint vs normalized biomass) + fit line."""
    title = "Sheldon (mass) spectrum — NBSS"
    mids = [m for m, v in zip(spec.mass_bin_midpoints, spec.nbss_values) if m > 0 and v > 0]
    vals = [v for m, v in zip(spec.mass_bin_midpoints, spec.nbss_values) if m > 0 and v > 0]
    if not mids:
        return _empty_figure(title)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mids, y=vals, mode="markers", name="NBSS"))
    if spec.slope is not None and spec.intercept is not None:
        log_mid = np.log10(np.asarray(mids, dtype=float))
        fitted = 10 ** (spec.slope * log_mid + spec.intercept)
        fig.add_trace(go.Scatter(x=mids, y=fitted, mode="lines", name="Regression",
                                 line=dict(dash="dash")))
        fig.add_annotation(text=f"Slope = {spec.slope:.2f}", showarrow=False,
                           xref="paper", yref="paper", x=0.95, y=0.95)
    fig.update_layout(
        title=title,
        xaxis=dict(title="Body mass (bin midpoint)", type="log"),
        yaxis=dict(title="Normalized biomass (per mass width)", type="log"),
    )
    return fig


def make_abc_plot(abc) -> go.Figure:
    """Cumulative %-dominance curves (biomass vs abundance) over species rank — ABC."""
    title = "Abundance-Biomass Comparison (ABC)"
    if not abc.ranks:
        return _empty_figure(title)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=abc.ranks, y=abc.cum_biomass_pct, mode="lines+markers",
                             name="Biomass"))
    fig.add_trace(go.Scatter(x=abc.ranks, y=abc.cum_abundance_pct, mode="lines+markers",
                             name="Abundance"))
    w_txt = "n/a" if abc.w_statistic != abc.w_statistic else f"{abc.w_statistic:.3f}"
    fig.add_annotation(text=f"W = {w_txt}", showarrow=False, xref="paper", yref="paper",
                       x=0.95, y=0.05)
    fig.update_layout(
        title=title,
        xaxis=dict(title="Species rank"),
        yaxis=dict(title="Cumulative dominance (%)"),
    )
    return fig
```

NOTE: confirm `_empty_figure`, `go`, and `np` are module-level in `osmose/plotting.py` (they are — `make_size_spectrum_plot` uses all three). If `_empty_figure` is named differently, match the existing helper.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_plotting.py -k "sheldon or abc" -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add osmose/plotting.py tests/test_plotting.py
git commit -m "feat(plotting): Sheldon NBSS + ABC dominance-curve charts"
```

---

## Task 7: Wire into the Diagnostics page

**Files:**
- Modify: `ui/pages/results.py`
- Modify: `tests/test_community_metrics.py`

Context: `ui/pages/results.py` registers result types in a selector and renders one plotly figure per `rtype` inside the `@render_plotly def results_chart():` function, where `tmpl = _tpl(input)` is the plotly template. The existing length-spectrum entry appears as `"size_spectrum": "Size Spectrum"` in the selector choices (≈ lines 311 and 638) and is rendered by a branch `if rtype == "size_spectrum":` (≈ line 703). The loaded config dict is `state.config.get()` (`dict[str, str]`, carries the species length-weight `a,b`); the current run output directory is `state.output_dir.get()` (a `Path`). `state` is the app-state object already imported and used throughout this module. This task mirrors the `size_spectrum` branch for two new chart types and adds a markdown metrics panel.

- [ ] **Step 1: Write the failing wiring test** (append to `tests/test_community_metrics.py`)

```python
def test_results_page_wires_community_metrics():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py").read_text()
    assert "sheldon_spectrum" in src  # new chart rtype / id
    assert "abc_curve" in src  # new chart rtype / id
    assert "make_sheldon_spectrum_plot" in src
    assert "make_abc_plot" in src
    assert "community_report" in src  # the metrics panel builder
    assert "format_community_report" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k wires -q`
Expected: FAIL (none of the new symbols are referenced yet).

- [ ] **Step 3: Implement the wiring**

3a. Add the two chart types to BOTH the `title_map` dict (≈ line 638) and the selector `choices` dict (≈ line 311), next to the existing `"size_spectrum": "Size Spectrum"` entry:

```python
            "size_spectrum": "Size Spectrum",
            "sheldon_spectrum": "Sheldon (Mass) Spectrum",
            "abc_curve": "ABC (W-statistic)",
```

3b. Add render branches mirroring the `size_spectrum` branch (≈ line 703). Place immediately after it. `tmpl` is already in scope (`tmpl = _tpl(input)`); config and run dir come from app state:

```python
        if rtype == "sheldon_spectrum":
            from osmose.community_metrics import compute_sheldon_spectrum
            from osmose.plotting import _empty_figure, make_sheldon_spectrum_plot

            cfg = state.config.get()
            if not cfg:
                fig = _empty_figure("Sheldon (mass) spectrum — load a config for length-weight a,b")
            else:
                try:
                    spec = compute_sheldon_spectrum(state.output_dir.get(), cfg)
                    fig = make_sheldon_spectrum_plot(spec)
                except FileNotFoundError:
                    fig = _empty_figure("Sheldon (mass) spectrum — no by-size output for this run")
            fig.update_layout(template=tmpl)
            return fig

        if rtype == "abc_curve":
            from osmose.community_metrics import compute_abc
            from osmose.plotting import _empty_figure, make_abc_plot

            try:
                abc = compute_abc(state.output_dir.get())
                fig = make_abc_plot(abc)
            except FileNotFoundError:
                fig = _empty_figure("ABC — needs biomass and abundance outputs")
            fig.update_layout(template=tmpl)
            return fig
```

NOTE: `_empty_figure` is importable from `osmose.plotting` (the same helper the module uses internally). Confirm `state` is the in-scope app-state name in `results.py` (it is — used for `state.output_dir`, `state.results_loaded`, etc.).

3c. Add a Community Metrics markdown panel. In the page's output-UI section (near where other `@render.ui`/`@render.text` outputs live), add a `ui.output_ui("community_metrics_panel")` to the Diagnostics layout, and in the server function add:

```python
    @render.ui
    def community_metrics_panel():
        from osmose.community_metrics import community_report, format_community_report

        cfg = state.config.get()
        try:
            diag = community_report(state.output_dir.get(), cfg or None)
        except FileNotFoundError:
            return ui.markdown("_Community metrics unavailable — run a simulation first._")
        return ui.markdown(format_community_report(diag))
```

Keep the placement consistent with the existing diagnostics layout — do not restructure unrelated UI. `community_report` itself never raises on missing files (each unit degrades to `None`); the `try/except` guards only an unexpected `state.output_dir` being empty/invalid.

- [ ] **Step 4: Run the wiring test + the page imports clean**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k wires -q`
Expected: PASS.
Run: `.venv/bin/python -c "import ui.pages.results"`
Expected: no error (the page module imports successfully).

- [ ] **Step 5: Commit**

```bash
git add ui/pages/results.py tests/test_community_metrics.py
git commit -m "feat(results): wire Sheldon spectrum, ABC chart + community-metrics panel into Diagnostics"
```

---

## Task 8: Real-data smoke test + final gates

**Files:**
- Modify: `tests/test_community_metrics.py`

- [ ] **Step 1: Write the real-data smoke test** (append; guarded so it skips when the gitignored eec_full output is absent)

```python
from tests._data_guards import require_eec_output


def test_community_report_eec_real():
    require_eec_output("*DistribBySize*")
    from pathlib import Path

    # eec_full has no in-memory config here; exercise the config-less path (Sheldon skipped)
    # plus the trophic + ABC units against real output.
    diag = community_report(Path("data/eec_full/output"), None, window_years=10)
    assert diag.trophic is not None
    assert diag.abc is not None
    # W-statistic is a bounded index in [-1, 1].
    assert math.isnan(diag.abc.w_statistic) or -1.0 <= diag.abc.w_statistic <= 1.0
    md = format_community_report(diag)
    assert "# OSMOSE community diagnostics" in md
```

- [ ] **Step 2: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py -k eec_real -q`
Expected: PASS, or SKIP if `data/eec_full/output/*DistribBySize*` is absent (both acceptable).

- [ ] **Step 3: Run the full new test module + touched suites**

Run: `.venv/bin/python -m pytest tests/test_community_metrics.py tests/test_plotting.py tests/test_size_spectrum.py -q`
Expected: all PASS.

- [ ] **Step 4: Lint, format, type-check (CI parity — lint job runs BOTH check and format)**

Run each, fix any reported issues, re-run until clean:
- `.venv/bin/ruff check osmose/community_metrics.py osmose/plotting.py ui/pages/results.py tests/test_community_metrics.py tests/test_plotting.py`
- `.venv/bin/ruff format osmose/community_metrics.py osmose/plotting.py ui/pages/results.py tests/test_community_metrics.py tests/test_plotting.py`
- `.venv/bin/pyright osmose/community_metrics.py osmose/plotting.py` (pandas reductions may need `np.asarray(..., dtype=float)` casts or `cast(...)`; resolve to 0 errors on the new module)

- [ ] **Step 5: Run the full suite once for regressions**

Run: `.venv/bin/python -m pytest -q -m "not e2e"`
Expected: PASS (no regressions; coverage stays ≥ 90%).

- [ ] **Step 6: Commit**

```bash
git add tests/test_community_metrics.py
git commit -m "test(community-metrics): real-data eec smoke + final gate pass"
```

---

## Notes

- **Units:** OSMOSE biomass is in tonnes and abundance in numbers, so `mean_body_mass` (total biomass / total abundance) is tonnes-per-individual; the formatter reports it with `.6g` without forcing a unit conversion. The Sheldon mass axis is in the config's W = a·L^b units (typically grams). These are reported as-is — consistent within a run, which is what trends/comparison need.
- **DRY:** all three units share `_per_species_window_mean`; the Sheldon fit reuses `analysis.size_spectrum_slope`; readers reuse `size_spectrum`'s `_read_community_by_size` / `_infer_bin_width` / `_window_by_time`.
- **YAGNI:** no per-timestep time series of the new metrics, no cross-run comparison UI (Scenario Diff already exists), no new config keys.
- **Out of scope:** an e2e assertion for the new Diagnostics panels (the wiring test + `import ui.pages.results` cover regressions cheaply; a full Playwright pass is optional follow-up).
