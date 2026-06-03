# Fishing-vs-natural Mortality (F/M) Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute a per-species F/M (fishing vs natural mortality) ratio from a finished OSMOSE run — for ALL species, no ICES reference points — surfaced via a library function, a bar chart, and a CLI; plus fix the pre-existing `mortalityRate` CSV read bug it depends on.

**Architecture:** A dedicated reader parses the real `mortalityRate-{sp}` CSV (1 preamble line + 2 header rows + trailing comma). `osmose/validation/fisheries.py` aggregates the `('F','Recruits')` and natural-cause columns to annual rates (using a config-derived `steps_per_year`, never row-ratio inference), computes F/M per species, and formats a report. A bar chart goes in `osmose/plotting.py`; a CLI in `scripts/compute_mortality_balance.py`. No engine changes, no calibration runs, no ICES reference-point math (B/Bmsy/F/Fmsy/Kobe are a documented deferred follow-up).

**Tech Stack:** Python 3.12, pandas, NumPy, Plotly (`osmose/plotly_theme.py` `PLOTLY_TEMPLATE`), pytest, ruff. Tests: `.venv/bin/python -m pytest`.

**Reference spec:** `docs/superpowers/specs/2026-06-03-fisheries-diagnostics-design.md` (rescoped after a 4-angle in-loop review).

---

## Verified facts (executed against real code/data — use exactly)

- Mortality file path: `{output_dir}/Mortality/{prefix}_mortalityRate-{species}_Simu0.csv` (confirmed baltic + eec_full).
- That CSV: line 0 = description preamble; line 1 = cause header (`Mpred`,`Mstarv`,`Madd`,`F`,`Zout`,`Mfor`,`Mdis`,`Mage`, each repeated for 3 stages; first field `Time`); line 2 = stage (`Eggs`,`Pre-recruits`,`Recruits`); data rows carry a **trailing comma** (26 fields vs 25 header). `results.mortality()` **raises `pandas.errors.ParserError`** on it today. Correct read: `pd.read_csv(path, skiprows=1, header=[0,1])` then drop the all-NaN trailing column → `(cause, stage)` MultiIndex; `df[("F","Recruits")]` etc. accessible.
- Output cadence: biomass and mortality both saved every `output.recordfrequency.ndt`; equal row counts; for shipped configs `recordfrequency.ndt == ndtPerYear == 24` → 1 row/year, each mortality row already an annual sum. So default `steps_per_year = 1` for shipped configs; do NOT infer from row counts.
- Config keys for steps-per-year: `simulation.time.ndtPerYear` and `output.recordfrequency.ndt` (in `{prefix}_param-simulation.csv` / `{prefix}_param-output.csv`). `steps_per_year = ndtPerYear // recordfrequency_ndt`.
- `OsmoseResults(output_dir, prefix="osm", strict=False)` — `osmose/results.py:239`; mirror `strict=False` from `validate_outputs_vs_ices.py:102`. We do NOT rely on `results.mortality()`.
- `osmose/plotting.py`: `import plotly.graph_objects as go`; `from osmose.plotly_theme import PLOTLY_TEMPLATE as TEMPLATE`; plotly 6.5.2 has `add_hline`/`write_html`.

## File Structure

- Create: `osmose/validation/fisheries.py` — reader, `annual_rate`, `MortalityBalance`, `compute_mortality_balance`, `format_mortality_report`.
- Modify: `osmose/results.py` — fix the `mortalityRate` read bug (Task 6, low-risk, guarded by checking other callers).
- Modify: `osmose/plotting.py` — add `make_fm_ratio_bars`.
- Create: `scripts/compute_mortality_balance.py` — CLI.
- Create: `tests/test_validation_fisheries.py` — real-file-fixture tests.
- Modify: a usage doc (Task 7).

---

## Task 1: Mortality CSV reader (real 2-row header + trailing comma)

**Files:**
- Create: `osmose/validation/fisheries.py`
- Test: `tests/test_validation_fisheries.py`
- Test fixture: copy a real CSV (see Step 1).

- [ ] **Step 1: Capture a real fixture + write the failing test**

Copy a real mortality CSV into the test tree so the test runs against the true format:
`mkdir -p tests/fixtures && cp data/eec_full/output/Mortality/eec_mortalityRate-cod_Simu0.csv tests/fixtures/mortalityRate_sample.csv`
(If that path is absent, use `data/baltic/output/Mortality/baltic_mortalityRate-cod_Simu0.csv`. Confirm with `ls data/*/output/Mortality/*mortalityRate-cod*`.)

```python
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from osmose.validation import fisheries as fz

_FIXTURE = Path(__file__).parent / "fixtures" / "mortalityRate_sample.csv"

def test_read_mortality_recruits_real_csv():
    df = fz.read_mortality_recruits(_FIXTURE)
    # MultiIndex (cause, stage); the F/Recruits and natural-cause Recruits columns exist
    assert ("F", "Recruits") in df.columns
    assert ("Mpred", "Recruits") in df.columns
    assert ("Mstarv", "Recruits") in df.columns
    assert ("Madd", "Recruits") in df.columns
    # values are finite floats, one row per saved step
    assert len(df) > 0
    assert df[("F", "Recruits")].notna().all()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py::test_read_mortality_recruits_real_csv -v`
Expected: FAIL (`module 'fisheries' has no attribute 'read_mortality_recruits'`).

- [ ] **Step 3: Implement the reader**

Create `osmose/validation/fisheries.py`:

```python
"""Fishing-vs-natural mortality (F/M) diagnostics for OSMOSE outputs.

Computes per-species F/M (realized fishing mortality vs natural mortality) from a
finished run — for all species, no ICES reference points. F is OSMOSE's Recruits-stage
instantaneous fishing mortality summed to annual; M = Mpred + Mstarv + Madd likewise.
F/M > 1 means fishing removes more than natural processes (an overexploitation signal).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_NATURAL_CAUSES = ("Mpred", "Mstarv", "Madd")


def read_mortality_recruits(path: Path) -> pd.DataFrame:
    """Read a `mortalityRate-{sp}` CSV into a (cause, stage) MultiIndex frame.

    The real file has a 1-line description preamble, a cause header row, a stage
    header row, and data rows with a trailing comma (one extra field). We skip the
    preamble, read the two header rows as a MultiIndex, and drop the all-NaN trailing
    column the trailing comma produces.
    """
    df = pd.read_csv(path, skiprows=1, header=[0, 1])
    # Drop any fully-NaN column (the trailing-comma artifact).
    df = df.dropna(axis=1, how="all")
    return df
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py::test_read_mortality_recruits_real_csv -v`
Expected: PASS. (If `("F","Recruits")` isn't found, inspect `list(df.columns)` — the second header level may carry whitespace; `.rename(columns=str.strip)` per level if needed, and assert the cleaned names.)

- [ ] **Step 5: Commit**

```bash
git add osmose/validation/fisheries.py tests/test_validation_fisheries.py tests/fixtures/mortalityRate_sample.csv
git commit -m "feat(fisheries): mortalityRate CSV reader (2-row header + trailing comma)"
```

---

## Task 2: `annual_rate` aggregation

**Files:**
- Modify: `osmose/validation/fisheries.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Write failing tests**

```python
def test_annual_rate_steps_per_year_1():
    # 3 rows, spy=1 → each row already a yearly value; window of 2 → mean of last 2
    s = pd.Series([0.1, 0.2, 0.3])
    assert fz.annual_rate(s, steps_per_year=1, window_years=2) == pytest.approx(0.25)

def test_annual_rate_steps_per_year_2():
    # 6 rows, spy=2 → annual = [0.3, 0.7, 1.1]; window 2 → mean(0.7,1.1)=0.9
    s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert fz.annual_rate(s, steps_per_year=2, window_years=2) == pytest.approx(0.9)

def test_annual_rate_drops_trailing_partial_year():
    # 5 rows, spy=2 → 2 full years [0.3,0.7]; trailing partial (row 4) dropped
    s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    assert fz.annual_rate(s, steps_per_year=2, window_years=2) == pytest.approx(0.5)  # mean(0.3,0.7)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k annual_rate -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add to `osmose/validation/fisheries.py`:

```python
def annual_rate(per_step: pd.Series, steps_per_year: int, window_years: int) -> float:
    """Sum a per-saved-step rate within each year, then mean over the trailing window.

    A trailing *partial* year (len not a multiple of steps_per_year) is dropped so the
    window only averages complete years.
    """
    if steps_per_year < 1:
        raise ValueError(f"steps_per_year must be >= 1, got {steps_per_year}")
    vals = np.asarray(per_step, dtype=float)
    n_years = len(vals) // steps_per_year
    if n_years == 0:
        raise ValueError("mortality series shorter than one full year")
    annual = vals[: n_years * steps_per_year].reshape(n_years, steps_per_year).sum(axis=1)
    w = min(window_years, n_years)
    return float(annual[-w:].mean())
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k annual_rate -v` (PASS).
```bash
git add osmose/validation/fisheries.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): annual_rate aggregation (per-step → annual, windowed)"
```

---

## Task 3: `MortalityBalance` + `compute_mortality_balance`

**Files:**
- Modify: `osmose/validation/fisheries.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Write failing tests (real fixture for one species; synthetic for edge cases)**

```python
def test_compute_balance_real_fixture(tmp_path):
    # Lay out a fake results dir with the real fixture as one species' mortality file.
    mort_dir = tmp_path / "Mortality"
    mort_dir.mkdir(parents=True)
    (mort_dir / "osm_mortalityRate-cod_Simu0.csv").write_bytes(_FIXTURE.read_bytes())
    out = fz.compute_mortality_balance(
        tmp_path, prefix="osm", species_list=["cod"], steps_per_year=1, window_years=5
    )
    b = {x.species: x for x in out}["cod"]
    assert b.fishing_mortality >= 0.0
    assert b.natural_mortality >= 0.0
    if b.natural_mortality > 0:
        assert b.f_over_m == pytest.approx(b.fishing_mortality / b.natural_mortality)
        assert b.overexploited == (b.f_over_m > 1.0)

def test_compute_balance_m_zero_gives_none(tmp_path, monkeypatch):
    # Force a zero-M / known-F frame via the reader.
    def fake_reader(path):
        cols = pd.MultiIndex.from_tuples([("F", "Recruits"), ("Mpred", "Recruits"),
                                          ("Mstarv", "Recruits"), ("Madd", "Recruits")])
        return pd.DataFrame([[0.4, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]], columns=cols)
    monkeypatch.setattr(fz, "read_mortality_recruits", fake_reader)
    (tmp_path / "Mortality").mkdir()
    (tmp_path / "Mortality" / "osm_mortalityRate-x_Simu0.csv").write_text("stub")
    b = fz.compute_mortality_balance(tmp_path, prefix="osm", species_list=["x"],
                                     steps_per_year=1, window_years=2)[0]
    assert b.natural_mortality == 0.0
    assert b.f_over_m is None
    assert b.overexploited is False

def test_compute_balance_skips_missing_species(tmp_path):
    (tmp_path / "Mortality").mkdir()
    out = fz.compute_mortality_balance(tmp_path, prefix="osm", species_list=["ghost"],
                                       steps_per_year=1, window_years=2)
    assert out == []   # missing mortality file → WARN-skip, not in output

def test_discover_species_from_mortality_dir(tmp_path):
    d = tmp_path / "Mortality"; d.mkdir()
    for sp in ("cod", "sprat"):
        (d / f"osm_mortalityRate-{sp}_Simu0.csv").write_text("stub")
    assert sorted(fz.discover_species(tmp_path, prefix="osm")) == ["cod", "sprat"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k "compute_balance or discover_species" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add to `osmose/validation/fisheries.py`:

```python
@dataclass(frozen=True)
class MortalityBalance:
    species: str
    fishing_mortality: float
    natural_mortality: float
    f_over_m: float | None
    overexploited: bool


def _mortality_path(output_dir: Path, prefix: str, species: str) -> Path:
    return Path(output_dir) / "Mortality" / f"{prefix}_mortalityRate-{species}_Simu0.csv"


def discover_species(output_dir: Path, prefix: str) -> list[str]:
    """Species names with a mortalityRate file in {output_dir}/Mortality."""
    mdir = Path(output_dir) / "Mortality"
    stem = f"{prefix}_mortalityRate-"
    out = []
    for p in sorted(mdir.glob(f"{stem}*_Simu0.csv")):
        out.append(p.name[len(stem):].rsplit("_Simu0.csv", 1)[0])
    return out


def compute_mortality_balance(
    output_dir: Path,
    *,
    prefix: str,
    species_list: list[str] | None = None,
    steps_per_year: int,
    window_years: int = 10,
) -> list[MortalityBalance]:
    """Per-species F/M from the mortalityRate outputs. steps_per_year is REQUIRED
    (config-derived by the caller; never inferred from row counts)."""
    species = species_list if species_list is not None else discover_species(output_dir, prefix)
    out: list[MortalityBalance] = []
    for sp in species:
        path = _mortality_path(output_dir, prefix, sp)
        if not path.exists():
            print(f"WARN: no mortality file for {sp!r} at {path}", file=sys.stderr)
            continue
        try:
            df = read_mortality_recruits(path)
            f = annual_rate(df[("F", "Recruits")], steps_per_year, window_years)
            m_series = sum(df[(c, "Recruits")] for c in _NATURAL_CAUSES)
            m = annual_rate(m_series, steps_per_year, window_years)
        except (KeyError, ValueError) as e:
            print(f"WARN: skipping {sp!r}: {e}", file=sys.stderr)
            continue
        f_over_m = (f / m) if m > 0 else None
        out.append(MortalityBalance(
            species=sp, fishing_mortality=f, natural_mortality=m,
            f_over_m=f_over_m, overexploited=(f_over_m is not None and f_over_m > 1.0),
        ))
    return out
```

- [ ] **Step 4: Run + commit**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k "compute_balance or discover_species" -v` (PASS).
```bash
git add osmose/validation/fisheries.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): MortalityBalance + compute_mortality_balance (F/M)"
```

---

## Task 4: Markdown report

**Files:**
- Modify: `osmose/validation/fisheries.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Write failing test**

```python
def test_format_report_renders_with_none_fm():
    bals = [
        fz.MortalityBalance("cod", 0.4, 0.2, 2.0, True),
        fz.MortalityBalance("x", 0.4, 0.0, None, False),  # M=0 → "—"
    ]
    md = fz.format_mortality_report(bals)
    assert "cod" in md and "x" in md
    assert "F/M" in md
    assert "2.00" in md and "—" in md
    assert "Recruits-stage" in md          # honest-limitation note present
    assert "1 overexploited" in md or "1/2" in md
```

- [ ] **Step 2: Run to verify failure → implement → pass**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k format_report -v` (FAIL).

Add to `osmose/validation/fisheries.py`:

```python
def format_mortality_report(balances: list[MortalityBalance], *, window_years: int = 10) -> str:
    """Markdown table of per-species F/M (fishing vs natural mortality)."""
    lines = [
        "# OSMOSE fishing-vs-natural mortality (F/M)",
        "",
        f"Model window: last {window_years} years. F is OSMOSE's Recruits-stage instantaneous "
        "fishing mortality summed to annual (not an ICES Fbar); M = Mpred + Mstarv + Madd. "
        "F/M > 1 means fishing removes more than natural processes.",
        "",
        "| species | F | M | F/M | overexploited |",
        "|---|---:|---:|---:|:---:|",
    ]
    n_over = 0
    for b in balances:
        fm = f"{b.f_over_m:.2f}" if b.f_over_m is not None else "—"
        over = "✓" if b.overexploited else "—"
        if b.overexploited:
            n_over += 1
        lines.append(f"| {b.species} | {b.fishing_mortality:.3f} | {b.natural_mortality:.3f} | {fm} | {over} |")
    lines += ["", f"**Summary:** {n_over} overexploited (F/M > 1) of {len(balances)} species.", ""]
    return "\n".join(lines)
```

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k format_report -v` (PASS).
```bash
git add osmose/validation/fisheries.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): F/M markdown report"
```

---

## Task 5: F/M bar chart

**Files:**
- Modify: `osmose/plotting.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Write failing smoke test**

```python
def test_fm_bar_chart_builds():
    from osmose import plotting
    bals = [fz.MortalityBalance("cod", 0.4, 0.2, 2.0, True),
            fz.MortalityBalance("x", 0.1, 0.5, 0.2, False)]
    fig = plotting.make_fm_ratio_bars(bals)
    assert fig is not None
    # both species with a finite F/M are plotted
    assert sum(len(t.x) for t in fig.data if hasattr(t, "x") and t.x is not None) >= 1
    # a reference line at y=1 exists
    assert any(getattr(s, "y0", None) == 1.0 or getattr(s, "y", None) == 1.0
               for s in fig.layout.shapes)
```

- [ ] **Step 2: Run (FAIL) → implement in `osmose/plotting.py` → pass**

Add to `osmose/plotting.py`:

```python
def make_fm_ratio_bars(balances) -> go.Figure:
    """F/M (fishing vs natural mortality) per species; reference line at F/M=1.

    Bars above 1 (fishing exceeds natural mortality) are highlighted.
    """
    valid = [b for b in balances if getattr(b, "f_over_m", None) is not None]
    colors = ["#d62728" if b.f_over_m > 1.0 else "#2ca02c" for b in valid]
    fig = go.Figure(go.Bar(
        x=[b.species for b in valid], y=[b.f_over_m for b in valid],
        marker=dict(color=colors), name="F/M",
    ))
    fig.add_hline(y=1.0, line=dict(dash="dash", width=1))
    fig.update_layout(title=dict(text="Fishing vs natural mortality (F/M)"),
                      xaxis=dict(title="species"), yaxis=dict(title="F / M"), template=TEMPLATE)
    return fig
```

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k fm_bar -v` (PASS).
Run: `.venv/bin/ruff check osmose/validation/fisheries.py osmose/plotting.py tests/test_validation_fisheries.py`.
```bash
git add osmose/plotting.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): F/M bar chart"
```

---

## Task 6: Fix `results.mortality()` (pre-existing ParserError) + CLI

**Files:**
- Modify: `osmose/results.py`
- Create: `scripts/compute_mortality_balance.py`
- Test: `tests/test_validation_fisheries.py`

- [ ] **Step 1: Regression test for the results.mortality() bug**

```python
def test_results_mortality_reads_real_csv(tmp_path):
    from osmose.results import OsmoseResults
    mdir = tmp_path / "Mortality"; mdir.mkdir()
    (mdir / "osm_mortalityRate-cod_Simu0.csv").write_bytes(_FIXTURE.read_bytes())
    # a minimal biomass file so OsmoseResults(strict=False) constructs
    r = OsmoseResults(tmp_path, prefix="osm", strict=False)
    df = r.mortality("cod")       # must NOT raise ParserError
    assert df is not None and len(df) > 0
```

- [ ] **Step 2: Run to verify it fails (ParserError today)**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k results_mortality -v`
Expected: FAIL with `ParserError` (the bug).

- [ ] **Step 3: Fix the reader — but FIRST audit other callers**

Run: `grep -rn "\.mortality(\|_read_species_output(\"mortalityRate\"\|mortalityRate" osmose/ ui/ scripts/ tests/ | grep -v test_validation_fisheries`. List every caller of `results.mortality()` / `mortality_rate()`. The fix must not break them. In `osmose/results.py`, make the `mortalityRate` read path use `skiprows=1, header=[0,1]` + drop the all-NaN trailing column (the same logic as `read_mortality_recruits`), returning a `(cause, stage)` MultiIndex frame. If existing callers depend on the OLD (broken or single-header) shape, instead route ONLY `mortalityRate` through the new logic and leave other species-outputs untouched; if a caller truly needs a flat frame, add a `flatten=` option rather than changing every caller. Document in the commit which callers you checked and why the change is safe. (If the audit shows the fix is risky, SKIP the results.py change — the feature already uses its own `read_mortality_recruits` — and mark this regression test `xfail` with a reason, keeping the bug documented.)

- [ ] **Step 4: Run the regression test**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -k results_mortality -v`
Expected: PASS (or documented `xfail` if Step 3 deemed the shared fix risky).

- [ ] **Step 5: Implement the CLI `scripts/compute_mortality_balance.py`**

```python
#!/usr/bin/env python3
"""Per-species F/M (fishing vs natural mortality) diagnostics for a finished run.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compute_mortality_balance.py \\
        --results-dir <path> [--prefix osm] [--window-years 10] \\
        [--steps-per-year N] [--config <param-dir-or-file>] \\
        [--species cod sprat ...] [--report out.md] [--json out.json] [--plot out_prefix]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_steps_per_year(args) -> int:
    if args.steps_per_year is not None:
        return args.steps_per_year
    # Try to derive from a config: ndtPerYear // recordfrequency.ndt.
    # Look for *_param-simulation.csv / *_param-output.csv near results or --config.
    search = [args.config] if args.config else []
    search += [args.results_dir, args.results_dir.parent]
    ndt = rec = None
    for base in search:
        if base is None:
            continue
        base = Path(base)
        for f in list(base.glob("*param-simulation.csv")) + list(base.glob("*param-output.csv")):
            for line in f.read_text().splitlines():
                if "ndtPerYear" in line:
                    ndt = int(line.split(";")[-1].split(",")[-1].strip() or 0) or ndt
                if "recordfrequency.ndt" in line:
                    rec = int(line.split(";")[-1].split(",")[-1].strip() or 0) or rec
    if ndt and rec:
        spy = max(1, ndt // rec)
        print(f"steps_per_year = {spy} (ndtPerYear={ndt} / recordfrequency.ndt={rec})")
        return spy
    print("WARNING: could not derive steps_per_year from config; defaulting to 1 "
          "(correct iff output.recordfrequency.ndt == simulation.time.ndtPerYear). "
          "Pass --steps-per-year if record frequency is finer than annual.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--prefix", type=str, default="osm")
    p.add_argument("--window-years", type=int, default=10)
    p.add_argument("--steps-per-year", type=int, default=None)
    p.add_argument("--config", type=Path, default=None, help="dir/file with *param-simulation/output.csv")
    p.add_argument("--species", nargs="*", default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--plot", type=str, default=None, help="output path prefix for the F/M bar chart (html)")
    args = p.parse_args(argv)

    from osmose.validation import fisheries as fz

    spy = _resolve_steps_per_year(args)
    balances = fz.compute_mortality_balance(
        args.results_dir, prefix=args.prefix, species_list=args.species,
        steps_per_year=spy, window_years=args.window_years,
    )
    report = fz.format_mortality_report(balances, window_years=args.window_years)
    if args.report:
        args.report.write_text(report)
    print(report)
    if args.json:
        args.json.write_text(json.dumps([asdict(b) for b in balances], indent=2))
    if args.plot:
        from osmose import plotting
        plotting.make_fm_ratio_bars(balances).write_html(f"{args.plot}_fm.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: CLI tests (exercise deferred imports, not just --help)**

```python
def test_cli_runs_on_empty_dir(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmb", "scripts/compute_mortality_balance.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    (tmp_path / "Mortality").mkdir()
    rc = mod.main(["--results-dir", str(tmp_path), "--prefix", "osm", "--steps-per-year", "1"])
    assert rc == 0   # no species → empty report, imports exercised, no crash
```

- [ ] **Step 7: Real-run smoke (use a config that actually has output)**

Run on a config with a finished run (eec_full has populated output; baltic_biomass may be empty — check `ls -la data/*/output/*biomass*`):
`PYTHONPATH=. .venv/bin/python scripts/compute_mortality_balance.py --results-dir data/eec_full/output --prefix eec --window-years 10 2>&1 | tail -25`
Expected: a markdown table with F, M, F/M per species, the "Recruits-stage" note, and the overexploited count; `steps_per_year = 1 (...)` printed. Confirm F/M values are plausible (0–several). Document the observed output.

- [ ] **Step 8: Commit**

```bash
git add osmose/results.py scripts/compute_mortality_balance.py tests/test_validation_fisheries.py
git commit -m "feat(fisheries): fix mortalityRate reader + compute_mortality_balance CLI"
```

---

## Task 7: Docs + finalize

**Files:**
- Modify: a usage doc (find: `grep -rl "validate_outputs_vs_ices\|compute_fisheries\|diagnostics" docs/ | head`; if none, add a short section to `docs/baltic_example.md` near the outputs/validation discussion).

- [ ] **Step 1: Document the CLI + the deferred follow-up**

Document `scripts/compute_mortality_balance.py`: what F/M means (fishing vs natural mortality, available for all species), the `--steps-per-year` caveat (default 1 = correct when record frequency == ndtPerYear; pass explicitly otherwise), and the honest limitation (F is OSMOSE Recruits-stage instantaneous F, not an ICES Fbar). Add a one-line **deferred follow-up** note: B/Bmsy + F/Fmsy + Kobe were scoped out — they need a config with broad ICES tonnes-unit coverage, a defensible Bmsy (not the MSY-Btrigger proxy, which overstates health), and an Fbar-aligned F; not worthwhile for sprat-only on Baltic.

- [ ] **Step 2: Full suite + lint**

Run: `.venv/bin/python -m pytest tests/test_validation_fisheries.py -v` (all pass; report count).
Run: `.venv/bin/python -m pytest tests/test_validation_ices.py tests/test_results.py -q` (sibling validator + results tests still green — confirm the results.py mortality fix didn't perturb them; if `test_results.py` doesn't exist, run `-k results`).
Run: `.venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/ tests/` (clean; `scripts/` is `ruff check`-only).

- [ ] **Step 3: Commit + finish**

```bash
git add docs/
git commit -m "docs(fisheries): document F/M diagnostics CLI + deferred Kobe follow-up"
```

Use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** reader bug + fix → T1 (dedicated reader) + T6 (results.mortality fix, guarded); `annual_rate` config-derived steps_per_year (no row-ratio inference) → T2 + T6 `_resolve_steps_per_year`; `MortalityBalance`/`compute_mortality_balance` (F/M, M==0→None, WARN-skip, species discovery) → T3; report w/ Recruits-stage note → T4; F/M bar chart (ref line, all species) → T5; CLI (strict=False via no `results.mortality` dependency in compute; config-derived spy, fail-loud-ish default-1-with-warning) → T6; docs + deferred-Kobe note → T7. DROPPED B/Bmsy/F/Fmsy/Kobe/ref-point-math → not in plan (deferred, per rescoped spec). ✅

**Placeholder scan:** no TBD/TODO; every code step has complete code. T1 Step1 / T6 Step7 use a REAL fixture/real run (named, with fallback paths) — not placeholders. T6 Step3 has an explicit audit-then-decide (fix or xfail) — a guarded decision, not a placeholder. ✅

**Type consistency:** `MortalityBalance(species, fishing_mortality, natural_mortality, f_over_m, overexploited)` identical across T3 dataclass, T4 report, T5 plot, T6 CLI. `read_mortality_recruits`/`annual_rate`/`discover_species`/`compute_mortality_balance` signatures consistent T1↔T2↔T3↔T6. `steps_per_year` is an explicit required arg everywhere (never inferred from row counts — the review's BLOCKER-4 fix). MultiIndex access `(cause, "Recruits")` consistent T1↔T3. ✅
