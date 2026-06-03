# Result Delta-Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute per-species output deltas (absolute + %) between a baseline and a variant OSMOSE run, ranked by magnitude, exposed via `osmose/analysis.py`, a diverging-bar chart, and a CLI.

**Architecture:** A shape-robust normalizer turns a run's `biomass()/yield_biomass()/abundance()` output into a `{species: windowed_mean}` dict (the disk output is WIDE — `Time` + per-species columns + a constant `species="all"`; a long `time/species/value` shape is also handled). `run_delta` unions both runs' species, computes abs/pct change + a `from_zero` flag, sorts by magnitude, returns the top-N as `SpeciesDelta` records. A markdown report + `make_run_delta_chart` (plotting.py) + `scripts/compare_runs.py` CLI complete it. No engine changes, no calibration runs.

**Tech Stack:** Python 3.12, pandas, NumPy, Plotly (`osmose/plotly_theme.py` `PLOTLY_TEMPLATE`), pytest, ruff. Tests: `.venv/bin/python -m pytest`.

**Reference spec:** `docs/superpowers/specs/2026-06-03-result-delta-tracking-design.md`.

---

## Verified facts (executed against real data — use exactly)

- `OsmoseResults(output_dir, prefix="osm", strict=False)`. Two runs = two instances.
- `results.biomass()`, `results.yield_biomass()`, `results.abundance()` (no-arg) return a **WIDE** frame: `Time` + one column per species + a constant `species` column whose only value is `"all"`. Per-species values are the COLUMNS. `biomass(species="cod")` returns **0 rows** — do NOT use the `species=` filter for these global 1D outputs; do NOT expect a `value` column.
- Metric→accessor: `"biomass"`→`biomass()`, `"yield"`→`yield_biomass()`, `"abundance"`→`abundance()`.
- A long-form shape (`time, species, value` with a real per-species `species` column) MAY occur for some outputs; the normalizer detects both (presence of a `value` column + a non-constant `species` column ⇒ long).
- `osmose/analysis.py` exists: `import numpy as np`, `import pandas as pd`, helper `_require_columns`, functions `ensemble_stats`/`summary_table`/`shannon_diversity`/`mean_tl_catch`/`size_spectrum_slope`. Add the new code here, same style.
- `osmose/plotting.py`: `import plotly.graph_objects as go`; `from osmose.plotly_theme import PLOTLY_TEMPLATE as TEMPLATE`.

## File Structure

- Modify: `osmose/analysis.py` — `_per_species_window_mean`, `SpeciesDelta`, `run_delta`, `format_delta_report`.
- Modify: `osmose/plotting.py` — `make_run_delta_chart`.
- Create: `scripts/compare_runs.py` — CLI.
- Create: `tests/test_analysis_delta.py` — unit tests (+ a real EEC biomass fixture for the wide path).
- Modify: a usage doc (Task 6).

---

## Task 1: Per-species windowed-mean normalizer (wide + long)

**Files:**
- Modify: `osmose/analysis.py`
- Test: `tests/test_analysis_delta.py` ; fixture `tests/fixtures/biomass_wide_sample.csv`

- [ ] **Step 1: Capture a real wide fixture + write failing tests**

`cp data/eec_full/output/eec_biomass_Simu0.csv tests/fixtures/biomass_wide_sample.csv` (confirm with `ls data/eec_full/output/*biomass*`). It's the wide format (Time + per-species cols + species=all).

```python
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from osmose import analysis as az

_WIDE_FIXTURE = Path(__file__).parent / "fixtures" / "biomass_wide_sample.csv"


class _FakeResults:
    """Stand-in for OsmoseResults exposing the three metric accessors."""
    def __init__(self, frames):  # frames: {"biomass": df, ...}
        self._frames = frames
        self.output_dir = "<fake>"
    def biomass(self, species=None):
        return self._frames["biomass"]
    def yield_biomass(self, species=None):
        return self._frames["yield"]
    def abundance(self, species=None):
        return self._frames["abundance"]


def _wide(**species_to_series):
    n = len(next(iter(species_to_series.values())))
    d = {"Time": list(range(1, n + 1))}
    d.update(species_to_series)
    d["species"] = ["all"] * n
    return pd.DataFrame(d)


def test_window_mean_wide_format():
    df = _wide(cod=[10.0, 20.0, 30.0], herring=[100.0, 100.0, 100.0])
    res = _FakeResults({"biomass": df, "yield": df, "abundance": df})
    means = az._per_species_window_mean(res, "biomass", window_years=2)
    assert means["cod"] == pytest.approx(25.0)       # mean(20,30)
    assert means["herring"] == pytest.approx(100.0)
    assert "species" not in means and "Time" not in means

def test_window_mean_long_format():
    long = pd.DataFrame({
        "time": [1, 2, 3, 1, 2, 3],
        "species": ["cod", "cod", "cod", "sprat", "sprat", "sprat"],
        "value": [10.0, 20.0, 30.0, 1.0, 1.0, 1.0],
    })
    res = _FakeResults({"biomass": long, "yield": long, "abundance": long})
    means = az._per_species_window_mean(res, "biomass", window_years=2)
    assert means["cod"] == pytest.approx(25.0)
    assert means["sprat"] == pytest.approx(1.0)

def test_window_mean_real_wide_fixture():
    df = pd.read_csv(_WIDE_FIXTURE)
    res = _FakeResults({"biomass": df, "yield": df, "abundance": df})
    means = az._per_species_window_mean(res, "biomass", window_years=10)
    assert "cod" in means and means["cod"] > 0
    assert "species" not in means  # the constant 'species' artifact column is excluded

def test_window_mean_uses_years_not_row_count():
    # 3 years at 2 rows/year. window=1 must take the LAST YEAR (Time>2.0 → rows at 2.5,3.0),
    # NOT the last ROW. cod last-year rows = [30,40] → mean 35 (a row-count tail(1) would give 40).
    df = _wide(cod=[10.0, 10.0, 20.0, 20.0, 30.0, 40.0])
    df["Time"] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    res = _FakeResults({"biomass": df, "yield": df, "abundance": df})
    means = az._per_species_window_mean(res, "biomass", window_years=1)
    assert means["cod"] == pytest.approx(35.0)   # by-year window; tail(1)=40 would be WRONG
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_analysis_delta.py -k window_mean -v`
Expected: FAIL (`no attribute '_per_species_window_mean'`).

- [ ] **Step 3: Implement the normalizer**

Add to `osmose/analysis.py`:

```python
_METRIC_ACCESSOR = {"biomass": "biomass", "yield": "yield_biomass", "abundance": "abundance"}
_NON_SPECIES_COLS = {"Time", "time", "species"}


def _trailing_window(df, time_col: str, window_years: int):
    """Rows whose time is within the last `window_years` (time is in YEARS).

    Filters by the time COLUMN, not by row count — so the window is correct
    regardless of how many rows-per-year the output was saved at (a row-count
    `tail` silently takes the last N *rows*, which is wrong for sub-annual output).
    """
    tmax = float(df[time_col].max())
    return df[df[time_col] > tmax - window_years]


def _per_species_window_mean(results, metric: str, window_years: int) -> dict[str, float]:
    """Per-species mean of `metric` over the trailing `window_years` of a run.

    Handles both output shapes:
    - WIDE (the disk default): `Time` + one column per species + a constant `species`
      column. Per-species values are the columns; mean each over the trailing window.
    - LONG: `time, species, value` with a real per-species `species` column; group by
      species and mean `value` over the trailing window per species.

    The window is selected by the time column (years), NOT by row count, so it is
    correct for sub-annual output cadences (recordfrequency.ndt < ndtPerYear).
    """
    if metric not in _METRIC_ACCESSOR:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(_METRIC_ACCESSOR)}")
    df = getattr(results, _METRIC_ACCESSOR[metric])()
    if df is None or len(df) == 0:
        return {}
    cols = set(df.columns)
    # LONG iff a value column + a species column are present. (The WIDE global frame
    # has a `species` column too — but no `value` column — so this discriminates
    # correctly even for a single-species long frame, where a row-count heuristic fails.)
    is_long = "value" in cols and "species" in cols
    if is_long:
        time_col = "time" if "time" in cols else "Time"
        out: dict[str, float] = {}
        for sp, g in df.groupby("species"):
            win = _trailing_window(g.sort_values(time_col), time_col, window_years)
            out[str(sp)] = float(win["value"].mean())
        return out
    # WIDE: species are the non-Time/non-species columns
    time_col = "Time" if "Time" in cols else "time"
    species_cols = [c for c in df.columns if c not in _NON_SPECIES_COLS]
    win = _trailing_window(df, time_col, window_years)
    return {str(c): float(win[c].mean()) for c in species_cols}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_analysis_delta.py -k window_mean -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add osmose/analysis.py tests/test_analysis_delta.py tests/fixtures/biomass_wide_sample.csv
git commit -m "feat(delta): per-species windowed-mean normalizer (wide + long output shapes)"
```

---

## Task 2: `SpeciesDelta` + `run_delta`

**Files:**
- Modify: `osmose/analysis.py`
- Test: `tests/test_analysis_delta.py`

- [ ] **Step 1: Write failing tests**

```python
def test_run_delta_ranks_by_pct():
    base = _FakeResults({"biomass": _wide(cod=[100.0, 100.0], herring=[50.0, 50.0]),
                         "yield": _wide(cod=[1.0, 1.0]), "abundance": _wide(cod=[1.0, 1.0])})
    var = _FakeResults({"biomass": _wide(cod=[110.0, 110.0], herring=[100.0, 100.0]),
                        "yield": _wide(cod=[1.0, 1.0]), "abundance": _wide(cod=[1.0, 1.0])})
    deltas = az.run_delta(base, var, metric="biomass", window_years=2)
    by = {d.species: d for d in deltas}
    assert by["cod"].abs_delta == pytest.approx(10.0)
    assert by["cod"].pct_delta == pytest.approx(0.10)
    assert by["herring"].pct_delta == pytest.approx(1.0)   # 50→100 = +100%
    # ranked by |pct_delta| desc → herring (1.0) before cod (0.10)
    assert [d.species for d in deltas][:2] == ["herring", "cod"]

def test_run_delta_top_n():
    # Names chosen so ALPHABETICAL order (a,b,c) DISAGREES with the pct ranking — a broken
    # sort that just preserves alpha order would fail this. a=+10%, b=+100%, c=+50%.
    base = _FakeResults({"biomass": _wide(a=[1.0], b=[1.0], c=[1.0]), "yield": _wide(a=[1.0]), "abundance": _wide(a=[1.0])})
    var = _FakeResults({"biomass": _wide(a=[1.1], b=[2.0], c=[1.5]), "yield": _wide(a=[1.0]), "abundance": _wide(a=[1.0])})
    deltas = az.run_delta(base, var, metric="biomass", window_years=1, top_n=2)
    assert len(deltas) == 2
    assert [d.species for d in deltas] == ["b", "c"]   # +100%, +50% — NOT alphabetical

def test_run_delta_from_zero_ranks_above_finite():
    # from-zero recovery must outrank a finite +200% mover.
    base = _FakeResults({"biomass": _wide(cod=[0.0], herring=[1.0]), "yield": _wide(cod=[0.0]), "abundance": _wide(cod=[0.0])})
    var = _FakeResults({"biomass": _wide(cod=[10.0], herring=[3.0]), "yield": _wide(cod=[0.0]), "abundance": _wide(cod=[0.0])})
    deltas = az.run_delta(base, var, metric="biomass", window_years=1)
    assert deltas[0].species == "cod" and deltas[0].from_zero is True
    assert deltas[1].species == "herring"

def test_run_delta_both_zero_ranks_last():
    # A 0->0 "dead" species (pct None but NOT from_zero) must rank LAST, never as a top mover.
    base = _FakeResults({"biomass": _wide(cod=[1.0], ghost=[0.0]), "yield": _wide(cod=[0.0]), "abundance": _wide(cod=[0.0])})
    var = _FakeResults({"biomass": _wide(cod=[2.0], ghost=[0.0]), "yield": _wide(cod=[0.0]), "abundance": _wide(cod=[0.0])})
    deltas = az.run_delta(base, var, metric="biomass", window_years=1)
    assert deltas[0].species == "cod"           # +100% mover on top
    assert deltas[-1].species == "ghost"        # 0->0 dead species last
    ghost = deltas[-1]
    assert ghost.pct_delta is None and ghost.from_zero is False and ghost.abs_delta == 0.0

def test_run_delta_from_zero():
    base = _FakeResults({"biomass": _wide(cod=[0.0, 0.0]), "yield": _wide(cod=[0.0]), "abundance": _wide(cod=[0.0])})
    var = _FakeResults({"biomass": _wide(cod=[5.0, 5.0]), "yield": _wide(cod=[0.0]), "abundance": _wide(cod=[0.0])})
    d = az.run_delta(base, var, metric="biomass", window_years=2)[0]
    assert d.species == "cod"
    assert d.baseline_mean == 0.0 and d.variant_mean == pytest.approx(5.0)
    assert d.pct_delta is None and d.from_zero is True
    assert d.abs_delta == pytest.approx(5.0)

def test_run_delta_union_species_present_in_one_run():
    base = _FakeResults({"biomass": _wide(cod=[10.0]), "yield": _wide(cod=[1.0]), "abundance": _wide(cod=[1.0])})
    var = _FakeResults({"biomass": _wide(cod=[10.0], newsp=[7.0]), "yield": _wide(cod=[1.0]), "abundance": _wide(cod=[1.0])})
    by = {d.species: d for d in az.run_delta(base, var, metric="biomass", window_years=1)}
    assert by["newsp"].baseline_mean == 0.0 and by["newsp"].variant_mean == pytest.approx(7.0)
    assert by["newsp"].from_zero is True

def test_run_delta_metric_switch():
    # yield differs while biomass is identical → metric="yield" must pick up the change
    base = _FakeResults({"biomass": _wide(cod=[10.0]), "yield": _wide(cod=[2.0]), "abundance": _wide(cod=[1.0])})
    var = _FakeResults({"biomass": _wide(cod=[10.0]), "yield": _wide(cod=[4.0]), "abundance": _wide(cod=[1.0])})
    d = {x.species: x for x in az.run_delta(base, var, metric="yield", window_years=1)}["cod"]
    assert d.pct_delta == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_analysis_delta.py -k run_delta -v`
Expected: FAIL (`no attribute 'run_delta'`).

- [ ] **Step 3: Implement**

Add to `osmose/analysis.py`. **Add `from dataclasses import dataclass` at the top — it is NOT currently imported** (verified; `@dataclass` will `NameError` without it):

```python
@dataclass(frozen=True)
class SpeciesDelta:
    species: str
    baseline_mean: float
    variant_mean: float
    abs_delta: float
    pct_delta: float | None   # None when baseline_mean == 0
    from_zero: bool           # baseline_mean == 0 and variant_mean > 0


def run_delta(baseline, variant, *, metric: str = "biomass", window_years: int = 10,
              top_n: int | None = None) -> list[SpeciesDelta]:
    """Per-species delta of `metric` (windowed mean) between two runs, ranked by |% change|.

    Species set = union of both runs (a species absent from one contributes mean 0 there).
    pct_delta is None for a zero-baseline species (reported via from_zero + abs_delta).
    Sorted by |pct_delta| desc (from-zero species sort to the top); ties by |abs_delta| desc.
    """
    bmeans = _per_species_window_mean(baseline, metric, window_years)
    vmeans = _per_species_window_mean(variant, metric, window_years)
    species = sorted(set(bmeans) | set(vmeans))
    deltas: list[SpeciesDelta] = []
    for sp in species:
        b = bmeans.get(sp, 0.0)
        v = vmeans.get(sp, 0.0)
        abs_d = v - b
        pct = (abs_d / b) if b != 0 else None
        deltas.append(SpeciesDelta(species=sp, baseline_mean=b, variant_mean=v,
                                   abs_delta=abs_d, pct_delta=pct,
                                   from_zero=(b == 0.0 and v > 0.0)))

    def _key(d: SpeciesDelta):
        # Genuine from-zero recoveries rank at top (inf). A 0->0 "dead" species also has
        # pct_delta None but is NOT a mover — it must rank LAST (0.0), not at the top.
        # Finite pct ranks by |pct|; ties by |abs|.
        if d.pct_delta is None:
            primary = float("inf") if d.from_zero else 0.0
        else:
            primary = abs(d.pct_delta)
        return (primary, abs(d.abs_delta))

    deltas.sort(key=_key, reverse=True)
    return deltas[:top_n] if top_n is not None else deltas
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_analysis_delta.py -k run_delta -v`
Expected: PASS (all five).

- [ ] **Step 5: Commit**

```bash
git add osmose/analysis.py tests/test_analysis_delta.py
git commit -m "feat(delta): SpeciesDelta + run_delta (union, abs/pct, from-zero, ranked)"
```

---

## Task 3: Markdown report

**Files:**
- Modify: `osmose/analysis.py`
- Test: `tests/test_analysis_delta.py`

- [ ] **Step 1: Write failing test**

```python
def test_format_delta_report():
    deltas = [
        az.SpeciesDelta("herring", 50.0, 100.0, 50.0, 1.0, False),
        az.SpeciesDelta("cod", 0.0, 5.0, 5.0, None, True),
    ]
    md = az.format_delta_report(deltas, metric="biomass", window_years=10)
    assert "herring" in md and "cod" in md
    assert "biomass" in md
    assert "+100.0%" in md or "100.0%" in md     # herring pct
    assert "from 0" in md                        # cod from-zero note
    assert "B/Bmsy" not in md                     # sanity: not the fisheries report
```

- [ ] **Step 2: Run (FAIL) → implement → pass**

Add to `osmose/analysis.py`:

```python
def format_delta_report(deltas: list[SpeciesDelta], *, metric: str = "biomass",
                        window_years: int = 10) -> str:
    """Markdown table of per-species deltas, ranked (as returned by run_delta)."""
    lines = [
        f"# OSMOSE run delta — {metric} (variant vs baseline)",
        "",
        f"Per-species mean {metric} over the last {window_years} years; ranked by |% change|. "
        "Δ% is undefined for a zero-baseline species (shown 'from 0').",
        "",
        "| species | baseline | variant | Δ | Δ% |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in deltas:
        if d.pct_delta is None:
            pct = "— (from 0)" if d.from_zero else "—"
        else:
            pct = f"{d.pct_delta * 100:+.1f}%"
        lines.append(f"| {d.species} | {d.baseline_mean:.3g} | {d.variant_mean:.3g} | "
                     f"{d.abs_delta:+.3g} | {pct} |")
    n_moved = sum(1 for d in deltas if d.abs_delta != 0.0)
    lines += ["", f"**Summary:** {n_moved} of {len(deltas)} species changed.", ""]
    return "\n".join(lines)
```

Run: `.venv/bin/python -m pytest tests/test_analysis_delta.py -k format_delta_report -v` (PASS).
```bash
git add osmose/analysis.py tests/test_analysis_delta.py
git commit -m "feat(delta): markdown delta report"
```

---

## Task 4: Diverging-bar chart

**Files:**
- Modify: `osmose/plotting.py`
- Test: `tests/test_analysis_delta.py`

- [ ] **Step 1: Write failing smoke test**

```python
def test_delta_chart_builds():
    from osmose import plotting
    from osmose import analysis as az
    deltas = [
        az.SpeciesDelta("herring", 50.0, 100.0, 50.0, 1.0, False),
        az.SpeciesDelta("cod", 100.0, 90.0, -10.0, -0.10, False),
        az.SpeciesDelta("sprat", 0.0, 5.0, 5.0, None, True),  # from-zero: no finite bar
    ]
    fig = plotting.make_run_delta_chart(deltas)
    assert fig is not None
    # EXACTLY the 2 finite-pct species are barred (herring, cod); the from-zero sprat is NOT.
    assert sum(len(t.x) for t in fig.data if hasattr(t, "x") and t.x is not None) == 2
    assert "sprat" not in [s for t in fig.data if t.y is not None for s in t.y]
```

- [ ] **Step 2: Run (FAIL) → implement in `osmose/plotting.py` → pass**

```python
def make_run_delta_chart(deltas, *, metric: str = "biomass") -> go.Figure:
    """Horizontal diverging bar of per-species % change (variant vs baseline).

    Only species with a finite pct_delta are barred (positive=green, negative=red);
    from-zero species are listed in the title note (no finite bar).
    """
    finite = [d for d in deltas if getattr(d, "pct_delta", None) is not None]
    # Ascending by |Δ%| so the biggest mover sits at the TOP of the horizontal chart
    # (plotly plots data order bottom-to-top), matching run_delta's magnitude ranking.
    finite = sorted(finite, key=lambda d: abs(d.pct_delta))
    colors = ["#2ca02c" if d.pct_delta >= 0 else "#d62728" for d in finite]
    fig = go.Figure(go.Bar(
        x=[d.pct_delta * 100 for d in finite], y=[d.species for d in finite],
        orientation="h", marker=dict(color=colors), name="Δ%",
    ))
    fig.add_vline(x=0.0, line=dict(width=1))
    from_zero = [d.species for d in deltas if getattr(d, "from_zero", False)]
    title = f"Run delta — {metric} (% change, variant vs baseline)"
    if from_zero:
        title += f"  ·  from 0: {', '.join(from_zero)}"
    fig.update_layout(title=dict(text=title), xaxis=dict(title="Δ%"),
                      yaxis=dict(title="species"), template=TEMPLATE)
    return fig
```

Run: `.venv/bin/python -m pytest tests/test_analysis_delta.py -k delta_chart -v` (PASS).
Run: `.venv/bin/ruff check osmose/analysis.py osmose/plotting.py tests/test_analysis_delta.py`.
```bash
git add osmose/plotting.py tests/test_analysis_delta.py
git commit -m "feat(delta): diverging-bar run-delta chart"
```

---

## Task 5: CLI `scripts/compare_runs.py`

**Files:**
- Create: `scripts/compare_runs.py`
- Test: `tests/test_analysis_delta.py`

- [ ] **Step 1: Implement the CLI**

```python
#!/usr/bin/env python3
"""Compare two finished OSMOSE runs: per-species output delta, ranked by % change.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compare_runs.py \\
        --baseline <dir> --variant <dir> [--prefix osm] \\
        [--baseline-prefix P] [--variant-prefix P] \\
        [--metric biomass|yield|abundance] [--window-years 10] [--top-n N] \\
        [--report out.md] [--json out.json] [--plot out_prefix]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--variant", required=True, type=Path)
    p.add_argument("--prefix", type=str, default="osm")
    p.add_argument("--baseline-prefix", type=str, default=None)
    p.add_argument("--variant-prefix", type=str, default=None)
    p.add_argument("--metric", type=str, default="biomass", choices=["biomass", "yield", "abundance"])
    p.add_argument("--window-years", type=int, default=10)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args(argv)

    from osmose.results import OsmoseResults
    from osmose import analysis as az

    bpref = args.baseline_prefix or args.prefix
    vpref = args.variant_prefix or args.prefix
    baseline = OsmoseResults(args.baseline, prefix=bpref, strict=False)
    variant = OsmoseResults(args.variant, prefix=vpref, strict=False)
    deltas = az.run_delta(baseline, variant, metric=args.metric,
                          window_years=args.window_years, top_n=args.top_n)
    report = az.format_delta_report(deltas, metric=args.metric, window_years=args.window_years)
    if args.report:
        args.report.write_text(report)
    print(report)
    if args.json:
        args.json.write_text(json.dumps([asdict(d) for d in deltas], indent=2))
    if args.plot:
        from osmose import plotting
        plotting.make_run_delta_chart(deltas, metric=args.metric).write_html(f"{args.plot}_delta.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: CLI test (exercises deferred imports, not just --help)**

```python
def test_cli_self_comparison_is_all_zero(tmp_path):
    import importlib.util, json
    spec = importlib.util.spec_from_file_location("cr", "scripts/compare_runs.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = tmp_path / "delta.json"
    # compare a real run against ITSELF → every delta must be exactly 0
    rc = mod.main(["--baseline", "data/eec_full/output", "--variant", "data/eec_full/output",
                   "--prefix", "eec", "--metric", "biomass", "--window-years", "10",
                   "--json", str(out)])
    assert rc == 0
    rows = json.loads(out.read_text())
    assert len(rows) > 0                                  # species were actually compared
    assert all(r["abs_delta"] == 0.0 for r in rows)       # genuine self-comparison → zero deltas
    assert all(r["pct_delta"] == 0.0 for r in rows)       # 0/x = 0% (no from-zero on a self-compare)
```

Run: `.venv/bin/python -m pytest tests/test_analysis_delta.py -k cli -v`
Expected: PASS (self-comparison runs; the JSON proves every Δ is exactly 0, not just rc==0).
(If eec_full's prefix differs, find it with `ls data/eec_full/output/*biomass*`. PYTHONPATH: the importlib load inherits pytest's sys.path which has the repo root; the real-run usage in the docstring shows `PYTHONPATH=.`.)

- [ ] **Step 3: Real perturbed-run smoke (prove a non-zero delta surfaces)**

Confirm the delta surfaces a real change by comparing a run against a programmatically-perturbed copy of its biomass output:
```bash
PYTHONPATH=. .venv/bin/python -c "
import shutil, pandas as pd, tempfile, os
src='data/eec_full/output'; dst=tempfile.mkdtemp()
shutil.copytree(src, dst, dirs_exist_ok=True)
# halve cod biomass in the variant
import glob
f=glob.glob(os.path.join(dst,'*biomass*Simu0.csv'))[0]
df=pd.read_csv(f); df['cod']=df['cod']*0.5; df.to_csv(f,index=False)
from osmose.results import OsmoseResults
from osmose import analysis as az
d=az.run_delta(OsmoseResults(src,prefix='eec',strict=False), OsmoseResults(dst,prefix='eec',strict=False), metric='biomass', window_years=10)
print(az.format_delta_report(d, metric='biomass'))
"
```
Expected: cod shows ≈ −50% (the top mover), other species ≈ 0%. Report the table. (If the biomass file has a header/preamble that `pd.read_csv` mangles, adjust the perturbation read accordingly — but `OsmoseResults` reads it fine, so the delta itself is what matters.)

- [ ] **Step 4: Commit**

```bash
git add scripts/compare_runs.py tests/test_analysis_delta.py
git commit -m "feat(delta): compare_runs CLI"
```

---

## Task 6: Docs + finalize

**Files:**
- Modify: a usage doc (find: `grep -rln "compute_mortality_balance\|validate_outputs_vs_ices\|Scripts" docs/ | head`; prefer the same doc the fisheries CLI was added to — likely `docs/baltic_example.md`).

- [ ] **Step 1: Document the CLI**

Document `scripts/compare_runs.py`: what it computes (per-species % change in biomass/yield/abundance between a baseline and variant run, ranked by magnitude), example usage, the `--metric`/`--window-years`/`--top-n` flags, and the honest limitations (windowed-mean comparison, no significance/multi-seed band; zero-baseline shown as "from 0", not infinity). One line: per-period and per-cell deltas are a deferred follow-on (see the spec).

- [ ] **Step 2: Full verification**

Run: `.venv/bin/python -m pytest tests/test_analysis_delta.py -v` (all pass; report count).
Run: `.venv/bin/python -m pytest tests/ -k "analysis or plotting or results" -q` (no regressions — `analysis.py`/`plotting.py` additions are additive; report counts, note pre-existing vs new).
Run: `.venv/bin/ruff check osmose/ tests/ scripts/compare_runs.py && .venv/bin/ruff format --check osmose/ tests/` (clean on touched files; `scripts/` is `ruff check`-only — note pre-existing unrelated `scripts/` errors are out of scope).

- [ ] **Step 3: Commit + finish**

```bash
git add docs/
git commit -m "docs(delta): document compare_runs CLI"
```

Use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** normalizer (wide+long, the corrected shape) → T1; `SpeciesDelta`/`run_delta` (union, abs/pct, from_zero, sort by |pct| with from-zero-at-top, top_n, metric switch) → T2; `format_delta_report` → T3; `make_run_delta_chart` (diverging bar, finite-pct only, from-zero in title) → T4; `scripts/compare_runs.py` (baseline/variant, prefix + per-run overrides, metric/window/top-n, report/json/plot) → T5; docs + deferred-note → T6. Deferred (per-period, per-cell, age/size bins, UI) → not in plan, per spec. Honest limitations (windowed-mean, no significance band; zero-baseline=from_zero) → T3 report + T6 docs. ✅

**Placeholder scan:** no TBD/TODO; every code step complete. T1 Step1 + T5 Step3 use a REAL fixture / real perturbed run (named, with fallback notes) — not placeholders. ✅

**Type consistency:** `SpeciesDelta(species, baseline_mean, variant_mean, abs_delta, pct_delta, from_zero)` identical across T2/T3/T4/T5. `_per_species_window_mean(results, metric, window_years) -> dict[str,float]` consistent T1↔T2. `run_delta(baseline, variant, *, metric, window_years, top_n)` consistent T2↔T5. `_METRIC_ACCESSOR` maps biomass/yield/abundance → biomass()/yield_biomass()/abundance() consistently. ✅
