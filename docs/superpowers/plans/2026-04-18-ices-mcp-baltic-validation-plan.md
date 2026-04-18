# Baltic Calibration ICES Cross-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Precondition:** Execute in a session that has the `ices` MCP server reachable. MCP config lives at `/home/razinka/osmose/osmose-python/.mcp.json` — **Claude Code must be launched from `osmose-python/` (not its parent `/home/razinka/osmose/`)** or the `.mcp.json` will not be discovered and the `ices` tools will be unavailable. The server is stdio-based (`uv run --directory /home/razinka/ices-mcp-server server.py`); MCP boots automatically at session start when the cwd matches.

**Goal:** Cross-validate Baltic OSMOSE calibration inputs — biomass targets, fishing mortality rates, and implicit reference points — against ICES SAG (Stock Assessment Graphs) data for the 2018–2022 advice window. Produce reproducible JSON snapshots, a diff report, and a regression test that trips when config drifts outside the ICES-assessed envelope.

**Architecture:** Two-layer design. Layer 1 (interactive/MCP-driven): pull raw ICES data via the MCP tools into `data/baltic/reference/ices_snapshots/*.json` — one snapshot per stock, taken once per advice year. Layer 2 (offline/deterministic): a script + pytest that compare `baltic_param-fishing.csv` and `biomass_targets.csv` against those snapshots. The snapshots are committed so validation is reproducible without re-querying ICES (keeps tests fast and independent of network).

**ICES MCP response shape (verified against `/home/razinka/ices-mcp-server/ices/sag.py:80-132`):**
- `get_stock_assessment` returns a **list of dicts** with lowercase keys: `year`, `ssb`, `recruitment`, `f`, `catches`, `landings`, `discards`, `low_ssb`, `high_ssb`. No uppercase variants; no dict-of-parallel-lists shape.
- `get_reference_points` returns a **dict** with lowercase keys: `flim`, `fpa`, `fmsy`, `blim`, `bpa`, `msy_btrigger`, `f_age_range`, `recruitment_age`, optionally `note`. No `MSYBtrigger` / `Blim` etc. in the MCP output (those are only in the raw ICES response, before MCP flattening).

Every downstream piece of code in this plan reads MCP output, so **use lowercase keys consistently**.

**Unit assumption (tonnes):** the raw ICES SAG payload attaches `StockSizeUnits` per row but the MCP flattening at `sag.py:89-102` drops it. The validator therefore **assumes all Baltic finfish stocks report SSB in tonnes** — the ICES convention for these stocks. This holds for every stock in the `model_species_to_ices_stocks` manifest (verified against the ICES fixture `"StockSizeUnits": "tonnes"` for `cod.27.24-32`). If a future stock is added to the manifest with a different unit, the envelope check silently drifts by the unit ratio (e.g. 1000× for "1000 tonnes"). Mitigation: Task 5 must spot-check one year of each snapshot against the published ICES advice sheet DOI before trusting the envelope table.

**Tech Stack:** ICES MCP server (9 tools wrapping SAG, DATRAS, SD, Eggs & Larvae APIs — see `/home/razinka/ices-mcp-server/server.py:40-96`), Python 3.12, pandas, httpx, pytest, ruff.

**Out of scope:** DATRAS survey CPUE comparison (useful for growth/recruitment calibration — separate plan), eggs & larvae spawning-ground cross-check (separate plan), adjusting the calibration objective weights in `scripts/calibrate_baltic.py:141` (belongs to a calibration-tuning plan, not this validation plan).

**Pre-flight:**
- Baseline: `.venv/bin/python -m pytest -q` must report 2432 passed before starting.
- Lint baseline: `.venv/bin/ruff check osmose/ scripts/ tests/ ui/` must be clean.
- MCP check: inside the session, invoke `list_stocks(year=2023, area_filter="27\\.2[0-9]")` and confirm results include at least `cod.27.24-32`, `her.27.25-2932`, `spr.27.22-32`, `fle.27.24-32`. If that fails, the plan is blocked — do NOT proceed with snapshots. **Also confirm the exact western-flounder stock-key label** — this plan assumes `fle.27.2223` but ICES occasionally publishes it hyphenated as `fle.27.22-23`. Whichever the `list_stocks` payload shows is canonical; update `index.json` and Task 3's stock list before the snapshot pull.
- All work lands in the current working tree (the deferred Baltic-data commit from the separate front is fine — files touched below do not intersect).

---

## Stocks Under Test

| Model species | ICES stock key(s)                                               | Rationale                                               |
|---------------|-----------------------------------------------------------------|---------------------------------------------------------|
| cod           | `cod.27.24-32` (eastern), `cod.27.22-24` (western)              | Two-stock Baltic cod; model aggregates.                 |
| herring       | `her.27.25-2932`, `her.27.28`, `her.27.3031`, `her.27.20-24`    | Four herring stocks; model aggregates (simplification). |
| sprat         | `spr.27.22-32`                                                  | Single Baltic sprat stock.                              |
| flounder      | `fle.27.24-32`, `fle.27.2223`                                   | Two flounder stocks.                                    |
| plaice        | `ple.27.24-32`                                                  | Reserved; model currently has no plaice species.        |
| perch / pikeperch / smelt / stickleback | **no ICES SAG assessment**                            | Coastal stocks — documented as such in the report.      |

These stock keys drive every ICES MCP call below.

---

## File Structure

**Created:**

- `data/baltic/reference/ices_snapshots/README.md` — how the snapshots were produced + MCP tool + call arguments.
- `data/baltic/reference/ices_snapshots/{stock_key}.assessment.json` — one file per stock; raw result of `get_stock_assessment`.
- `data/baltic/reference/ices_snapshots/{stock_key}.reference_points.json` — one file per stock; raw result of `get_reference_points`.
- `data/baltic/reference/ices_snapshots/index.json` — manifest mapping model species → stock keys + snapshot paths.
- `scripts/validate_baltic_vs_ices_sag.py` — offline validator; reads snapshots + model config, emits diff table + report.
- `tests/test_baltic_ices_validation.py` — pytest regression guard over the snapshots.
- `docs/baltic_ices_validation_2026-04-18.md` — one-shot report generated by the script.

**Modified (outside this plan, but may follow):** `data/baltic/reference/biomass_targets.csv`, `data/baltic/baltic_param-fishing.csv` — if validation finds drift, a follow-up data-update task handles edits. Not done by this plan.

---

## Task 1: MCP connectivity smoke

**Files:**
- Test: invoke directly in conversation; no file

- [ ] **Step 1: Probe the ICES MCP server is live**

Invoke the MCP tool `list_stocks` with `year=2023` and `area_filter="27\\.[234][0-9]"` (Baltic subdivisions). Expected: the result list contains keys starting with `cod.27.`, `her.27.`, `spr.27.`, `fle.27.`, `ple.27.`. If the MCP call errors or returns empty, stop here — the rest of the plan is unreachable.

- [ ] **Step 2: Probe a single-stock assessment**

Invoke `get_stock_assessment(stock_key="spr.27.22-32", year=2023)`. Expected: a list of year-row dicts with lowercase keys — at minimum `year`, `ssb`, `f`, `recruitment` with values across the 1990s-2022 window. Confirm the exact field names (MCP flattens the raw ICES response per `sag.py:89-102`, but a future MCP-server version could change this) — Task 4 matches on them.

---

## Task 2: Snapshot directory scaffold + manifest

**Files:**
- Create: `data/baltic/reference/ices_snapshots/README.md`
- Create: `data/baltic/reference/ices_snapshots/index.json`

- [ ] **Step 1: Create the snapshot directory**

```bash
mkdir -p data/baltic/reference/ices_snapshots
```

- [ ] **Step 2: Write the README describing the snapshot workflow**

Write to `data/baltic/reference/ices_snapshots/README.md`:

```markdown
# ICES SAG Snapshots

Frozen JSON copies of ICES Stock Assessment Graphs (SAG) data used to
validate Baltic OSMOSE calibration inputs (biomass targets, fishing
mortality rates). Taken via the `ices` MCP server:

- `get_stock_assessment(stock_key, year)` → `{stock}.assessment.json`
- `get_reference_points(stock_key, year)` → `{stock}.reference_points.json`

Advice year: 2023 (covers the 2018-2022 window used for calibration
targets in `biomass_targets.csv`).

See `index.json` for the model-species → stock-key mapping and the
"How to Refresh" section below for regenerating snapshots when ICES
publishes a new advice year. Refresh is MCP-driven (Tasks 1-3 of the
validation plan) — the offline validator `scripts/validate_baltic_vs_ices_sag.py`
only reads snapshots; it does not re-query ICES.
```

- [ ] **Step 3: Write the manifest**

Write to `data/baltic/reference/ices_snapshots/index.json`:

```json
{
  "advice_year": 2023,
  "created": "2026-04-18",
  "model_species_to_ices_stocks": {
    "cod": ["cod.27.24-32", "cod.27.22-24"],
    "herring": [
      "her.27.25-2932",
      "her.27.28",
      "her.27.3031",
      "her.27.20-24"
    ],
    "sprat": ["spr.27.22-32"],
    "flounder": ["fle.27.24-32", "fle.27.2223"],
    "perch": [],
    "pikeperch": [],
    "smelt": [],
    "stickleback": []
  }
}
```

`perch`/`pikeperch`/`smelt`/`stickleback` intentionally map to `[]` — they are coastal stocks not assessed by ICES SAG. The validator must tolerate the empty list without erroring.

- [ ] **Step 4: Commit**

```bash
git add data/baltic/reference/ices_snapshots/README.md data/baltic/reference/ices_snapshots/index.json
git commit -m "docs(baltic): scaffold ICES SAG snapshot directory + manifest"
```

---

## Task 3: Pull the 2023 advice snapshots

**Files:**
- Create: `data/baltic/reference/ices_snapshots/*.assessment.json` (9 files)
- Create: `data/baltic/reference/ices_snapshots/*.reference_points.json` (9 files)

**Nine** stock keys need snapshots: `cod.27.24-32`, `cod.27.22-24`, `her.27.25-2932`, `her.27.28`, `her.27.3031`, `her.27.20-24`, `spr.27.22-32`, `fle.27.24-32`, `fle.27.2223` (confirm the flounder label in Task 1 Step 1; substitute `fle.27.22-23` if that is what ICES returns). Per stock, pull both the assessment time series and the reference points.

- [ ] **Step 1: For each stock key, invoke `get_stock_assessment`**

For every `{stock}` in the list above, call the MCP tool:

```
get_stock_assessment(stock_key="{stock}", year=2023)
```

Extract the `result` field from the MCP response and write it verbatim to `data/baltic/reference/ices_snapshots/{stock}.assessment.json`. Use 2-space JSON indentation so diffs are readable. The MCP payload is a list of dicts, one per year, with lowercase keys: `year`, `ssb`, `recruitment`, `f`, `catches`, `landings`, `discards`, `low_ssb`, `high_ssb` (see `/home/razinka/ices-mcp-server/ices/sag.py:89-102`).

- [ ] **Step 2: For each stock key, invoke `get_reference_points`**

For every `{stock}` in the list above, call:

```
get_reference_points(stock_key="{stock}", year=2023)
```

Write the result to `data/baltic/reference/ices_snapshots/{stock}.reference_points.json`. MCP-flattened fields (lowercase, see `sag.py:117-127`): `flim`, `fpa`, `fmsy`, `blim`, `bpa`, `msy_btrigger`, `f_age_range`, `recruitment_age`, optionally `note`. Missing points are kept as null.

- [ ] **Step 3: Verify snapshot shapes on disk**

```bash
.venv/bin/python -c "
import json
from pathlib import Path
snap = Path('data/baltic/reference/ices_snapshots')
stocks = ['cod.27.24-32', 'cod.27.22-24',
          'her.27.25-2932', 'her.27.28', 'her.27.3031', 'her.27.20-24',
          'spr.27.22-32', 'fle.27.24-32', 'fle.27.2223']
for s in stocks:
    a = json.loads((snap / f'{s}.assessment.json').read_text())
    r = json.loads((snap / f'{s}.reference_points.json').read_text())
    print(f'{s}: assessment={type(a).__name__} rp={type(r).__name__}')
"
```

Expected: nine lines, each with non-empty types. If any file is missing, repeat Steps 1-2 for that stock.

- [ ] **Step 4: Commit**

```bash
git add data/baltic/reference/ices_snapshots/
git commit -m "data(baltic): freeze 2023 ICES SAG snapshots for nine Baltic stocks"
```

---

## Task 4: Offline validator script

**Files:**
- Create: `scripts/validate_baltic_vs_ices_sag.py`
- Test: `tests/test_baltic_ices_validation.py`

**Design notes** — the script does three comparisons:

1. **F-rate check.** For each model species with ≥ 1 ICES stock, compute SSB-weighted average of ICES `F` over 2018-2022 across all linked stocks. Compare to the model's `fisheries.rate.base.fsh{idx}` value from `data/baltic/baltic_param-fishing.csv`. Flag if the model F is outside `[0.5 * F_ices, 1.5 * F_ices]`.
2. **Biomass envelope check.** For each model species, compute ICES SSB range (`min, max` across 2018-2022, summed across linked stocks). Compare against `biomass_targets.csv` `lower_tonnes`/`upper_tonnes`. Flag if model target envelope does not overlap ICES envelope.
3. **Reference points.** Record ICES `Fmsy`, `Blim`, `Bpa`, `MSYBtrigger` per stock. Model does not consume these directly — report them as context for future ICES-aware objectives.

- [ ] **Step 1: Write the failing smoke test first**

Write to `tests/test_baltic_ices_validation.py`:

```python
"""Regression guards: Baltic model config stays consistent with ICES snapshots."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "baltic" / "reference" / "ices_snapshots"
MANIFEST = SNAPSHOT_DIR / "index.json"
VALIDATOR_SCRIPT = PROJECT_ROOT / "scripts" / "validate_baltic_vs_ices_sag.py"

# Skip every test in this module when the snapshots haven't been pulled
# (i.e. Tasks 2-3 haven't run yet). Prevents `FileNotFoundError` noise
# during partial plan execution.
pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="ICES SAG snapshots not yet pulled (run Tasks 2-3 first)",
)


@pytest.fixture(scope="module")
def validator_module():
    """Load the validator script by path (scripts/ has no __init__.py)."""
    spec = importlib.util.spec_from_file_location(
        "validate_baltic_vs_ices_sag", VALIDATOR_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate_baltic_vs_ices_sag"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("validate_baltic_vs_ices_sag", None)


@pytest.fixture(scope="module")
def report(validator_module):
    """A cached validator report — avoids re-running the comparisons
    for every fence test."""
    return validator_module.run(write_report=False)


def test_manifest_exists_and_is_readable():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["advice_year"] == 2023
    assert "model_species_to_ices_stocks" in manifest
    assert manifest["model_species_to_ices_stocks"]["sprat"] == ["spr.27.22-32"]


def test_every_assessed_species_has_snapshot_files():
    manifest = json.loads(MANIFEST.read_text())
    for species, stocks in manifest["model_species_to_ices_stocks"].items():
        for stock in stocks:
            assert (SNAPSHOT_DIR / f"{stock}.assessment.json").exists(), \
                f"Missing assessment snapshot for {species}/{stock}"
            assert (SNAPSHOT_DIR / f"{stock}.reference_points.json").exists(), \
                f"Missing reference-points snapshot for {species}/{stock}"


def test_validator_produces_report_for_sprat(report):
    """The sprat row — the only single-stock, well-assessed Baltic species —
    must always produce a numeric comparison. If this row is `None`, the
    lowercase-key pathway is silently broken again."""
    assert "f_rates" in report
    assert "biomass_envelopes" in report
    assert "reference_points" in report
    f_rows = {r["species"]: r for r in report["f_rates"]}
    b_rows = {r["species"]: r for r in report["biomass_envelopes"]}
    assert "sprat" in f_rows, "F-rates table missing sprat"
    assert f_rows["sprat"]["ices_f_weighted"] is not None, \
        "sprat has linked stock but no F computed — check key casing in _series_by_year"
    assert b_rows["sprat"]["ices_min_ssb"] is not None, \
        "sprat has linked stock but no SSB envelope computed"
```

- [ ] **Step 2: Run test to confirm FAIL (script doesn't exist yet)**

```bash
.venv/bin/python -m pytest tests/test_baltic_ices_validation.py -v
```

Expected: FAIL on `test_validator_script_runs_clean` with `FileNotFoundError` for the script. The two snapshot tests should PASS because they only touch the files from Tasks 2-3.

- [ ] **Step 3: Write the validator script**

Write to `scripts/validate_baltic_vs_ices_sag.py`:

```python
#!/usr/bin/env python3
"""Baltic OSMOSE calibration vs ICES SAG 2023 snapshots validator.

Reads frozen ICES snapshots from data/baltic/reference/ices_snapshots/ and
compares:
  1. Fishing mortality rates (model vs SSB-weighted ICES F, 2018-2022).
  2. Biomass targets (model envelope vs ICES SSB envelope, 2018-2022).
  3. Reference points (Fmsy/Blim/Bpa/MSYBtrigger — reported, not enforced).

Usage:
    .venv/bin/python scripts/validate_baltic_vs_ices_sag.py            # run + print report
    .venv/bin/python scripts/validate_baltic_vs_ices_sag.py --report   # + write markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "baltic" / "reference" / "ices_snapshots"
TARGETS_CSV = PROJECT_ROOT / "data" / "baltic" / "reference" / "biomass_targets.csv"
FISHING_CSV = PROJECT_ROOT / "data" / "baltic" / "baltic_param-fishing.csv"
REPORT_MD = PROJECT_ROOT / "docs" / "baltic_ices_validation_2026-04-18.md"

WINDOW_YEARS = range(2018, 2023)  # 2018..2022 inclusive
F_TOLERANCE = (0.5, 1.5)  # model F must land within [0.5x, 1.5x] of ICES F


@dataclass
class FComparison:
    species: str
    model_f: float
    ices_f_weighted: float | None
    in_tolerance: bool | None


@dataclass
class BiomassComparison:
    species: str
    model_lower: float
    model_upper: float
    ices_min_ssb: float | None
    ices_max_ssb: float | None
    envelopes_overlap: bool | None


def _load_manifest() -> dict:
    return json.loads((SNAPSHOT_DIR / "index.json").read_text())


def _load_assessment(stock_key: str) -> list[dict]:
    return json.loads((SNAPSHOT_DIR / f"{stock_key}.assessment.json").read_text())


def _load_reference_points(stock_key: str) -> dict:
    return json.loads((SNAPSHOT_DIR / f"{stock_key}.reference_points.json").read_text())


def _series_by_year(assessment: list[dict], field: str) -> dict[int, float]:
    """Extract {year: value} for an ICES SAG field from an MCP `get_stock_assessment` response.

    The MCP response is a list of dicts with lowercase keys (`year`, `ssb`,
    `recruitment`, `f`, `catches`, `landings`, `discards`, ...) — see
    `/home/razinka/ices-mcp-server/ices/sag.py:89-102`. String values are
    coerced via float(); None/NaN values are dropped.
    """
    out: dict[int, float] = {}
    for row in assessment or []:
        y = row.get("year")
        v = row.get(field)
        if y is None or v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        out[int(y)] = fv
    return out


def _ssb_weighted_f(stocks: list[str]) -> float | None:
    """Compute SSB-weighted mean F across linked stocks, window 2018-2022."""
    num = 0.0
    den = 0.0
    for stock in stocks:
        assessment = _load_assessment(stock)
        f_series = _series_by_year(assessment, "f")
        ssb_series = _series_by_year(assessment, "ssb")
        for y in WINDOW_YEARS:
            if y in f_series and y in ssb_series:
                num += f_series[y] * ssb_series[y]
                den += ssb_series[y]
    return num / den if den > 0 else None


def _ices_ssb_envelope(stocks: list[str]) -> tuple[float | None, float | None]:
    """Sum SSB across stocks per year, then return (min, max) across the window.

    Only include years where **all** stocks in the list have SSB data; drop
    years with partial coverage to avoid an artificially narrow envelope
    (e.g. a stock that was only assessed from 2020 onwards would otherwise
    undercount the 2018-2019 totals). Returns `(None, None)` if no year
    satisfies the full-coverage requirement — caller then reports `—`.
    """
    per_stock_series: list[dict[int, float]] = [
        _series_by_year(_load_assessment(stock), "ssb") for stock in stocks
    ]
    if not per_stock_series:
        return None, None
    full_coverage_years = [
        y for y in WINDOW_YEARS if all(y in s for s in per_stock_series)
    ]
    if not full_coverage_years:
        return None, None
    yearly_totals = [
        sum(s[y] for s in per_stock_series) for y in full_coverage_years
    ]
    return min(yearly_totals), max(yearly_totals)


def _parse_model_fishing_rates() -> dict[int, float]:
    """Return {fsh_idx: F_base} from baltic_param-fishing.csv."""
    rates: dict[int, float] = {}
    with FISHING_CSV.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("fisheries.rate.base.fsh"):
                continue
            key, _, value = line.partition(";")
            idx = int(key.rsplit(".fsh", 1)[1])
            rates[idx] = float(value.strip())
    return rates


def _parse_model_targets() -> list[dict]:
    """Return list of rows from biomass_targets.csv."""
    targets: list[dict] = []
    with TARGETS_CSV.open() as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        targets.append(row)
    return targets


# Model species → fsh index (from baltic_param-fishing.csv fisheries.name.fshN rows).
_SPECIES_FSH_INDEX = {
    "cod": 0,
    "herring": 1,
    "sprat": 2,
    "flounder": 3,
    "perch": 4,
    "pikeperch": 5,
    "smelt": 6,
    "stickleback": 7,
}


def _compare_f_rates(manifest: dict, model_fsh: dict[int, float]) -> list[FComparison]:
    out: list[FComparison] = []
    for species, stocks in manifest["model_species_to_ices_stocks"].items():
        idx = _SPECIES_FSH_INDEX.get(species)
        if idx is None:
            continue
        model_f = model_fsh.get(idx)
        if model_f is None:
            continue
        if not stocks:
            out.append(FComparison(species, model_f, None, None))
            continue
        ices_f = _ssb_weighted_f(stocks)
        if ices_f is None:
            out.append(FComparison(species, model_f, None, None))
            continue
        lo, hi = F_TOLERANCE
        in_tol = lo * ices_f <= model_f <= hi * ices_f
        out.append(FComparison(species, model_f, ices_f, in_tol))
    return out


def _compare_biomass(manifest: dict, targets: list[dict]) -> list[BiomassComparison]:
    out: list[BiomassComparison] = []
    by_species = {row["species"]: row for row in targets}
    for species, stocks in manifest["model_species_to_ices_stocks"].items():
        row = by_species.get(species)
        if row is None:
            continue
        lower = float(row["lower_tonnes"])
        upper = float(row["upper_tonnes"])
        if not stocks:
            out.append(BiomassComparison(species, lower, upper, None, None, None))
            continue
        ices_min, ices_max = _ices_ssb_envelope(stocks)
        if ices_min is None or ices_max is None:
            out.append(BiomassComparison(species, lower, upper, None, None, None))
            continue
        overlap = not (upper < ices_min or lower > ices_max)
        out.append(BiomassComparison(species, lower, upper, ices_min, ices_max, overlap))
    return out


def _collect_reference_points(manifest: dict) -> dict[str, dict]:
    rp: dict[str, dict] = {}
    for stocks in manifest["model_species_to_ices_stocks"].values():
        for stock in stocks:
            rp[stock] = _load_reference_points(stock)
    return rp


def run(*, write_report: bool = True) -> dict:
    manifest = _load_manifest()
    model_fsh = _parse_model_fishing_rates()
    targets = _parse_model_targets()

    report = {
        "f_rates": [f.__dict__ for f in _compare_f_rates(manifest, model_fsh)],
        "biomass_envelopes": [b.__dict__ for b in _compare_biomass(manifest, targets)],
        "reference_points": _collect_reference_points(manifest),
    }

    if write_report:
        _write_markdown_report(report)
    return report


def _write_markdown_report(report: dict) -> None:
    lines = [
        "# Baltic OSMOSE vs ICES SAG (2023 advice) — Validation Report",
        "",
        f"Generated {Path(__file__).name} from snapshots in `{SNAPSHOT_DIR.relative_to(PROJECT_ROOT)}/`.",
        "",
        "## Fishing Mortality Rates (2018-2022 window)",
        "",
        "| species | model F | ICES F (SSB-weighted) | in [0.5x, 1.5x] tolerance? |",
        "|---|---:|---:|:---:|",
    ]
    for r in report["f_rates"]:
        f_model = f"{r['model_f']:.3f}"
        f_ices = "—" if r["ices_f_weighted"] is None else f"{r['ices_f_weighted']:.3f}"
        flag = "—" if r["in_tolerance"] is None else ("✓" if r["in_tolerance"] else "✗")
        lines.append(f"| {r['species']} | {f_model} | {f_ices} | {flag} |")
    lines += [
        "",
        "## Biomass Envelope Overlap (SSB 2018-2022, summed across stocks)",
        "",
        "| species | model [lower, upper] | ICES SSB [min, max] | overlap? |",
        "|---|---:|---:|:---:|",
    ]
    for r in report["biomass_envelopes"]:
        mdl = f"[{r['model_lower']:.0f}, {r['model_upper']:.0f}]"
        ices = "—" if r["ices_min_ssb"] is None else f"[{r['ices_min_ssb']:.0f}, {r['ices_max_ssb']:.0f}]"
        flag = "—" if r["envelopes_overlap"] is None else ("✓" if r["envelopes_overlap"] else "✗")
        lines.append(f"| {r['species']} | {mdl} | {ices} | {flag} |")
    lines += [
        "",
        "## Reference Points (per ICES stock)",
        "",
        "| stock | Blim | Bpa | Fmsy | MSYBtrigger |",
        "|---|---:|---:|---:|---:|",
    ]
    for stock, rp in sorted(report["reference_points"].items()):
        cells = [_format_rp_cell(rp, k) for k in ("blim", "bpa", "fmsy", "msy_btrigger")]
        lines.append(f"| `{stock}` | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n")


def _format_rp_cell(rp: dict, key: str) -> str:
    """Format a reference-point value cell — '—' for missing, 4sf otherwise.

    Kept at module scope to avoid ruff B023 (late-binding closure in loop)
    when called from the for-loop inside `_write_markdown_report`.
    """
    v = rp.get(key) if isinstance(rp, dict) else None
    if v is None:
        return "—"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help=f"Write markdown report to {REPORT_MD.relative_to(PROJECT_ROOT)}")
    args = parser.parse_args()
    report = run(write_report=args.report)
    manifest = _load_manifest()
    assessed = {
        s for s, stocks in manifest["model_species_to_ices_stocks"].items() if stocks
    }

    for r in report["f_rates"]:
        print(f"F[{r['species']}]: model={r['model_f']:.3f} ices={r['ices_f_weighted']} tol={r['in_tolerance']}")
        if r["species"] in assessed and r["ices_f_weighted"] is None:
            print(
                f"  WARN: {r['species']} has linked ICES stocks but no F data in "
                f"window {WINDOW_YEARS.start}-{WINDOW_YEARS.stop - 1} — check snapshots.",
                file=sys.stderr,
            )
    for r in report["biomass_envelopes"]:
        print(f"B[{r['species']}]: overlap={r['envelopes_overlap']} model=[{r['model_lower']:.0f},{r['model_upper']:.0f}] "
              f"ices=[{r['ices_min_ssb']},{r['ices_max_ssb']}]")
        if r["species"] in assessed and r["ices_min_ssb"] is None:
            print(
                f"  WARN: {r['species']} has linked ICES stocks but no full-coverage "
                f"SSB year in window — check snapshots or broaden window.",
                file=sys.stderr,
            )
    # Exit 1 if any flagged miss (so CI can catch drift).
    failures = [r for r in report["f_rates"] if r["in_tolerance"] is False]
    failures += [r for r in report["biomass_envelopes"] if r["envelopes_overlap"] is False]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test**

```bash
.venv/bin/python -m pytest tests/test_baltic_ices_validation.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Run the validator with `--report`**

```bash
.venv/bin/python scripts/validate_baltic_vs_ices_sag.py --report
```

Expected: stdout shows F/B comparison lines per species; `docs/baltic_ices_validation_2026-04-18.md` is created. Exit code is 0 if all in-tolerance, 1 if drift is detected — **both outcomes are fine** at this stage; the report is the deliverable, the exit code is informational.

- [ ] **Step 6: Lint**

```bash
.venv/bin/ruff check scripts/validate_baltic_vs_ices_sag.py tests/test_baltic_ices_validation.py
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/validate_baltic_vs_ices_sag.py tests/test_baltic_ices_validation.py docs/baltic_ices_validation_2026-04-18.md
git commit -m "feat(calibration): ICES SAG validator for Baltic F rates and biomass envelopes"
```

---

## Task 5: Flag-and-document drift

**Files:**
- Modify: `docs/baltic_ices_validation_2026-04-18.md` (append findings section)

- [ ] **Step 1: Review the report + unit sanity-check**

Read `docs/baltic_ices_validation_2026-04-18.md`. Note every `✗` in the F-rate and biomass-envelope tables; these are drift hits.

Before trusting the envelope table, **spot-check units**: for one stock per species (e.g. `cod.27.24-32`, `her.27.25-2932`, `spr.27.22-32`, `fle.27.24-32`), open the ICES advice sheet DOI from the snapshot's source row and confirm the published SSB magnitude for one year in the window matches the corresponding `ssb` value in `data/baltic/reference/ices_snapshots/{stock}.assessment.json` to within 1%. If any stock is off by ~1000×, the MCP flattening dropped a non-tonnes unit and the envelope comparison must be corrected before publishing findings.

- [ ] **Step 2: Append a findings section to the report**

Append to `docs/baltic_ices_validation_2026-04-18.md`:

```markdown
## Findings and Recommended Follow-ups

If the tables above contain **no `✗` marks**, write a single line
`No drift detected against 2023 ICES advice; all assessed species within
[0.5×, 1.5×] F tolerance and biomass envelope overlap.` and skip the
template below.

Otherwise, for each `✗` row, instantiate this template (replace every
`{token}` with the actual value — they are fill-ins, not f-string
literals):

- **{species}:** model {F|B} is {above|below} ICES by {pct}%. Likely cause:
  {retrospective-vs-current | aggregation across stocks | calibration pressure
  against other species | genuine model mis-specification}. Follow-up:
  {update config | document as scenario choice | re-calibrate | none}.

## Known Limitations

- `model_species_to_ices_stocks` aggregates four independently assessed herring
  stocks into one `herring` species in the model. Summed SSB/F is a
  simplification — spatial distribution differs between stocks and per-stock
  Fmsy values cannot be combined linearly. Document this as a modeling
  choice, not drift.
- Flounder has two ICES stocks with different area coverage; the same caveat
  applies.
- Coastal species (perch, pikeperch, smelt, stickleback) have no ICES SAG
  assessment — validation for these is **by construction impossible** through
  this plan. See `docs/superpowers/plans/*-coastal-validation-plan.md` (to be
  written) for DATRAS-CPUE / national-survey alternatives.
- The SSB envelope uses the intersection of years for which **all** linked
  stocks report SSB. A stock assessed only from 2020 will drop 2018-2019
  from the envelope for its species rather than undercounting them. If the
  resulting envelope window shrinks below three years the result is likely
  noise — report it as "insufficient overlap" rather than drift.
- `_ssb_weighted_f` returning `None` for an assessed species triggers a
  `WARN` on stderr; these should be investigated (missing F in the most
  recent advice year is common for data-limited stocks).
```

- [ ] **Step 3: Commit**

```bash
git add docs/baltic_ices_validation_2026-04-18.md
git commit -m "docs(baltic): summarize ICES cross-validation findings and known limits"
```

---

## Task 6: Regression guard for config drift

**Files:**
- Modify: `tests/test_baltic_ices_validation.py` (append drift-fence tests)

**Rationale:** snapshots are frozen at 2023 advice. If someone edits `baltic_param-fishing.csv` or `biomass_targets.csv` in a way that *increases* drift against ICES, CI should flag it. This is a fence, not a gate — authors can bump the tolerance or refresh snapshots with justification.

- [ ] **Step 1: Append drift-fence tests**

Append to `tests/test_baltic_ices_validation.py`:

```python
def test_no_severe_f_rate_drift(report):
    """Model F must stay within [0.25x, 4x] of ICES F — a wider fence than
    the validator's [0.5x, 1.5x] tolerance. Tighter fence = more false
    positives on benign tweaks; this catches only order-of-magnitude drift.

    Uses the module-scoped `report` fixture defined at the top of this
    file so the comparisons run only once per pytest invocation.
    """
    severe = []
    for r in report["f_rates"]:
        if r["ices_f_weighted"] is None:
            continue
        ratio = r["model_f"] / r["ices_f_weighted"]
        if ratio < 0.25 or ratio > 4.0:
            severe.append((r["species"], ratio))
    assert not severe, (
        f"Severe F-rate drift (order-of-magnitude) vs ICES 2023: {severe}. "
        "Either refresh snapshots with a note or correct the model config."
    )


def test_biomass_envelope_overlaps_ices_for_assessed_species(report):
    """For every species with ICES stocks linked, the model's biomass
    envelope must overlap the ICES SSB envelope (2018-2022). Species with
    no linked stocks are allowed to have any envelope.
    """
    non_overlapping = [
        r["species"] for r in report["biomass_envelopes"]
        if r["envelopes_overlap"] is False
    ]
    assert not non_overlapping, (
        f"Model biomass envelope does not overlap ICES SSB for: {non_overlapping}. "
        "Either broaden the target envelope or re-justify (e.g. total biomass "
        "vs SSB distinction — document in the report)."
    )
```

- [ ] **Step 2: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_baltic_ices_validation.py -v
```

Expected: all PASS. If `test_biomass_envelope_overlaps_ices_for_assessed_species` fails for **cod**, that is expected given known post-2015 collapse dynamics — in that case, edit the test to allow cod as a known exception with a reference to `biomass_targets.csv:22` comment explaining "Post-2015 collapse state; total biomass ~1.5-2x SSB". Do NOT silently weaken the test — add an explicit `cod` allowlist:

```python
    KNOWN_EXCEPTIONS = {"cod"}  # see biomass_targets.csv:22 — aggregation mismatch SSB/total
    non_overlapping = [
        r["species"] for r in report["biomass_envelopes"]
        if r["envelopes_overlap"] is False and r["species"] not in KNOWN_EXCEPTIONS
    ]
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_baltic_ices_validation.py
git commit -m "test(calibration): regression fence for F-rate and biomass-envelope drift vs ICES"
```

---

## Task 7: Refresh path documentation

**Files:**
- Modify: `data/baltic/reference/ices_snapshots/README.md`

- [ ] **Step 1: Append a refresh section**

Append to `data/baltic/reference/ices_snapshots/README.md`:

```markdown
## How to Refresh

Snapshots freeze the 2023 ICES advice. When ICES publishes a new advice year
(typically every May/June), refresh via:

1. Start a Claude Code session in this repo so the `ices` MCP server loads.
2. Update `index.json`: bump `advice_year` and `created`.
3. Re-run every `get_stock_assessment` and `get_reference_points` call for
   every stock listed in `index.json`, writing the new payloads to
   `{stock}.assessment.json` / `{stock}.reference_points.json`.
4. **Update hardcoded constants in `scripts/validate_baltic_vs_ices_sag.py`:**
   - `WINDOW_YEARS` (currently `range(2018, 2023)`) — shift to the last
     five years covered by the new advice (`range(advice_year - 5, advice_year)`).
   - `REPORT_MD` filename (currently contains `2026-04-18`) — update the
     date stub to reflect the refresh date.
   - The `"(2023 advice)"` label in the validator docstring and report
     header should also match the new advice year.
5. Run `.venv/bin/python scripts/validate_baltic_vs_ices_sag.py --report` and
   review `docs/baltic_ices_validation_<date>.md` for new drift.
6. If drift has changed materially, open a separate PR to adjust
   `baltic_param-fishing.csv` / `biomass_targets.csv` (this repo's
   calibration-tuning plan, NOT the validation plan).
```

- [ ] **Step 2: Commit**

```bash
git add data/baltic/reference/ices_snapshots/README.md
git commit -m "docs(baltic): document ICES snapshot refresh workflow"
```

---

## Task 8: Final validation

- [ ] **Step 1: Full test suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: `2432 + 5 = 2437 passed` (five new tests — 3 from Task 4, 2 from Task 6).

- [ ] **Step 2: Full lint**

```bash
.venv/bin/ruff check osmose/ scripts/ tests/ ui/
```

Expected: `All checks passed!`

- [ ] **Step 3: Skim the report**

Open `docs/baltic_ices_validation_2026-04-18.md`. Confirm it has all three sections (F rates, biomass envelopes, reference points) populated for the assessed species.

- [ ] **Step 4: Skim git log**

```bash
git log --oneline -10
```

Expected: six commits from Tasks 2-7 (scaffold, snapshots, validator, findings, fence, refresh docs).

- [ ] **Step 5: Update CHANGELOG**

Append under `## [Unreleased]` in `CHANGELOG.md`:

```markdown
### Added

- **calibration:** ICES SAG 2023-advice snapshots for nine Baltic stocks (cod, herring ×4, sprat, flounder ×2) under `data/baltic/reference/ices_snapshots/`.
- **calibration:** `scripts/validate_baltic_vs_ices_sag.py` compares model F rates and biomass envelopes against ICES snapshots; writes `docs/baltic_ices_validation_2026-04-18.md`.
- **tests:** regression fence for severe F-rate and biomass-envelope drift vs ICES (`tests/test_baltic_ices_validation.py`).
```

- [ ] **Step 6: Final commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for Baltic ICES cross-validation"
```

---

## Self-review checklist

- **Spec coverage:** every ask from `.remember/remember.md` ("cross-check F rates against ICES SAG", "reference points for cod.27.24-32 / her.27.25-2932 / spr.27.22-32", "biomass targets against SSB time series") is addressed: F rates → Task 4 `_compare_f_rates` + Task 6 fence, reference points → Task 4 `_collect_reference_points`, SSB time series → Task 4 `_compare_biomass` + Task 6 fence. Flounder is added because its F rate is model-relevant.
- **Placeholders:** scanned — every code block is complete; no "TBD"/"similar to earlier task".
- **Type consistency:** `FComparison` and `BiomassComparison` dataclasses used identically in Task 4; snapshot filename convention `{stock_key}.assessment.json` / `{stock_key}.reference_points.json` used in Tasks 2, 3, 4, 7.
- **MCP dependency:** Task 1 gates the plan on MCP availability; Tasks 4-6 run purely offline against committed snapshots. Validation is reproducible without a live ICES connection.
- **Drift tolerance:** `[0.5x, 1.5x]` soft tolerance (validator report flag) vs `[0.25x, 4x]` hard fence (regression test). Documented, adjustable, two-tier.
- **Coastal species:** explicitly excluded with rationale (Task 2 manifest, Task 5 findings).

---

## Execution Handoff

This plan is **not to be executed in the current session** — Task 1 requires a Claude Code session with the `ices` MCP server active, which only boots at session start. Execute in a fresh session where `.mcp.json` has been loaded.

Recommended follow-up:

1. Restart Claude Code at repo root so `ices` MCP boots.
2. Run `/loop 5m` or dispatch a subagent via `superpowers:subagent-driven-development` with this plan.
3. Review the produced `docs/baltic_ices_validation_2026-04-18.md` once Task 4 completes; the report content drives decisions in Task 5.
