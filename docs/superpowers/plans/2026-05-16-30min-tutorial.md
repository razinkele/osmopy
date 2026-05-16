# 30-Minute 3-Species Tutorial — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a markdown walkthrough tutorial under `docs/tutorials/` that takes a Python-fluent newcomer from "I installed OSMOSE" to "I built and perturbed a working 3-species ecosystem" in ~30 minutes. Backed by a regression test that pins the trophic-pyramid + cascade behaviour the tutorial promises.

**Architecture:** Single Markdown file with a ~90-line Python code block the reader pastes into `tutorial.py`. Real OSMOSE config keys (no pretend-API). One file is the **source of truth** (`tests/_tutorial_config.py`); the tutorial's code block is a transcribed copy of it. Three test assertions pin the load-bearing observations (script runs, trophic pyramid emerges, cascade is visible after perturbation); three more pin documentation hygiene (markdown parses, perturbation instructions findable, headless fallback produces values).

**Tech Stack:** Python 3.12, NumPy, xarray, plotly, h5netcdf, pytest, `osmose.engine.PythonEngine` + `OsmoseResults`.

**Spec:** `docs/superpowers/specs/2026-05-16-30min-tutorial-design.md` (converged through 6 in-loop review rounds — read it first; this plan inherits all design decisions verbatim).

---

## Conventions used in this plan

- **`.venv/bin/python` and `.venv/bin/pytest`** are the canonical invocations (see CLAUDE.md). Never use bare `python`/`pytest`.
- Tests use **`-xvs`** during development (`-x` stop at first failure, `-v` verbose, `-s` show prints) so that subagents see the actual error.
- **Commit per step that produces a logically-complete unit** — at minimum after each task. No batching across tasks.
- Ruff line-length is **100** (per CLAUDE.md). Use `.venv/bin/ruff format` after each Python file write; the format step is shown explicitly in each task that writes Python.
- Working directory is **`/home/razinka/osmose/osmose-python`** throughout. All file paths are relative to that unless prefixed `/`.

---

## Task 1: Stub `tests/_tutorial_config.py`

**Goal:** Author the source-of-truth module as stubs that satisfy import + signature contracts but fail when actually called. This lets Task 3 write the test in final form against a module that exists.

**Files:**
- Create: `tests/_tutorial_config.py`

- [ ] **Step 1.1: Confirm `tests/` is importable as a package**

Run: `.venv/bin/python -c "import tests"`

Expected: succeeds silently (no error). If `ModuleNotFoundError`, check that `tests/__init__.py` exists. Per round-5 review, the project already has tests as a package; this step is verification, not creation.

If the import fails, create the empty file:
```bash
touch tests/__init__.py
```

- [ ] **Step 1.2: Write the stub module**

Create `tests/_tutorial_config.py` exactly as below:

```python
"""Canonical source of truth for the 3-species tutorial.

The tutorial markdown at docs/tutorials/30-minute-ecosystem.md transcribes
the contents of `build_config`, `ACCESSIBILITY_CSV`, and `build_ltl` into
its main code block. The regression test at tests/test_tutorial_3species.py
imports them directly.

If anything here changes, update docs/tutorials/30-minute-ecosystem.md to
match. Drift is caught at PR review time + by the markdown-parses test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

# The exact substring the tutorial tells the reader to find in accessibility.csv,
# and the exact substring they replace it with. Used by the perturbation test
# to confirm the instruction is robust to CSV-format edits.
BASELINE_PERTURBATION: tuple[str, str] = ("Forager;0.8;0;0", "Forager;0.1;0;0")

# Canonical accessibility CSV. ;-separated; rows = prey labels, cols = predator
# labels (focal species only). Loaded by osmose/engine/accessibility.py:72-131.
ACCESSIBILITY_CSV: str = """;Predator;Forager;PlanktonEater
Predator;0;0;0
Forager;0.8;0;0
PlanktonEater;0;0.8;0
Plankton;0;0.2;0.8
"""


def build_ltl(work_dir: Path) -> Path:
    """Write the constant-Plankton LTL forcing NetCDF to work_dir.

    Returns the path to the written file. Filled in by Task 4.
    """
    raise NotImplementedError("Filled in by Task 4")


def build_config(work_dir: Path) -> dict:
    """Return the engine config dict with paths resolved against work_dir.

    `species.file.sp3` and `predation.accessibility.file` get resolved to
    `work_dir / "ltl.nc"` and `work_dir / "accessibility.csv"` respectively
    via `.as_posix()`. Filled in by Task 4.
    """
    raise NotImplementedError("Filled in by Task 4")
```

- [ ] **Step 1.3: Format with ruff**

Run: `.venv/bin/ruff format tests/_tutorial_config.py`

Expected: `1 file reformatted` or `1 file already formatted`.

- [ ] **Step 1.4: Lint with ruff**

Run: `.venv/bin/ruff check tests/_tutorial_config.py`

Expected: `All checks passed!`

- [ ] **Step 1.5: Confirm import works**

Run: `.venv/bin/python -c "from tests._tutorial_config import build_config, build_ltl, ACCESSIBILITY_CSV, BASELINE_PERTURBATION; print('ok')"`

Expected: `ok`

- [ ] **Step 1.6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add tests/_tutorial_config.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(tutorial): stub _tutorial_config helper module"
```

---

## Task 2: Probe `result.biomass()` column shape

**Goal:** Verify (not assume) what `PythonEngine().run_in_memory().biomass()` actually returns. The spec's melt logic depends on the wide-form shape `["Time", <sp_names>, "species"="all"]`; we confirm before encoding it into the test.

This task produces a throwaway script and prints output. **Do not commit the probe script.** Capture its output as evidence in the task report.

**Files:**
- None committed. Throwaway script written to `/tmp/probe_biomass.py`.

- [ ] **Step 2.1: Write the probe script**

Create `/tmp/probe_biomass.py`:

```python
"""Throwaway: verify result.biomass() column shape for the tutorial spec."""
from osmose.engine import PythonEngine
from osmose.config.reader import OsmoseConfigReader

reader = OsmoseConfigReader()
cfg = reader.read("data/minimal/osm_all-parameters.csv")
# Trim to 2 years so the probe runs fast
cfg["simulation.time.nyear"] = "2"

result = PythonEngine().run_in_memory(config=cfg, seed=42)
bio = result.biomass()
print("--- columns ---")
print(list(bio.columns))
print("--- dtypes ---")
print(bio.dtypes)
print("--- head ---")
print(bio.head())
print("--- tail ---")
print(bio.tail())
print("--- shape ---")
print(bio.shape)
print("--- unique species values ---")
print(bio["species"].unique())
```

- [ ] **Step 2.2: Run the probe**

Run: `cd /home/razinka/osmose/osmose-python && .venv/bin/python /tmp/probe_biomass.py`

Expected output (approximately — exact species names depend on `data/minimal/`):
- `columns` list contains `"Time"`, the focal species names, and `"species"`
- `shape` is `(48, K)` where K = 2 + n_focal_species (48 = 2 yr × 24 dt)
- `unique species values` is `["all"]`

If the actual shape is materially different (e.g., long-form already, or `time` lowercase), the test's melt logic in Task 3 must be adjusted accordingly. Record the observed shape and update Task 3 if needed.

- [ ] **Step 2.3: Delete the probe**

```bash
rm /tmp/probe_biomass.py
```

No commit (no tracked files changed).

---

## Task 3: Author `tests/test_tutorial_3species.py` in final form

**Goal:** Write all 6 assertions with placeholder margins. Run the test and confirm assertions #1-#3 go RED (helper stubs raise `NotImplementedError`); assertions #4-#6 may pass trivially against the current files (#4 requires the markdown to exist, which it doesn't yet — it'll fail with a clean `FileNotFoundError`, also expected RED).

**Files:**
- Create: `tests/test_tutorial_3species.py`

- [ ] **Step 3.1: Write the test file**

Create `tests/test_tutorial_3species.py`:

```python
"""Regression test for the 30-minute 3-species tutorial.

If `build_config`, `ACCESSIBILITY_CSV`, or `build_ltl` in
`tests/_tutorial_config.py` change, update
`docs/tutorials/30-minute-ecosystem.md` to match. Drift in the markdown
code block's syntax is caught by test_markdown_code_block_parses_and_runs below.
Drift in the dict's actual values is caught by manual reconciliation at
PR review.

The pyramid + cascade margins (assertions #2 and #3) are set by *measuring*
the equilibrium with seed=42 during authorship (Tasks 6 and 7), then
encoding the measured values ±20%. Do not hand-pick.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

from osmose.engine import PythonEngine

from ._tutorial_config import (
    ACCESSIBILITY_CSV,
    BASELINE_PERTURBATION,
    build_config,
    build_ltl,
)

FOCAL_SPECIES = ["Predator", "Forager", "PlanktonEater"]

TUTORIAL_MD_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "tutorials"
    / "30-minute-ecosystem.md"
)


def _melt_to_long(bio_wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape biomass() output from wide to tidy long form."""
    return (
        bio_wide.drop(columns=["species"])
        .melt(id_vars="Time", var_name="species", value_name="biomass")
    )


def _equilibrium_means(bio_long: pd.DataFrame) -> pd.Series:
    """Mean biomass per focal species over years 45-50."""
    window = bio_long[bio_long["Time"] >= 45]
    focal = window[window["species"].isin(FOCAL_SPECIES)]
    return focal.groupby("species")["biomass"].mean()


@pytest.fixture
def tutorial_workdir(tmp_path: Path) -> Path:
    """Materialise the tutorial's on-disk artifacts in tmp_path."""
    (tmp_path / "accessibility.csv").write_text(ACCESSIBILITY_CSV)
    build_ltl(tmp_path)
    return tmp_path


@pytest.fixture
def baseline_run(tutorial_workdir: Path, numba_warmup: None) -> pd.DataFrame:
    """Run the engine with the baseline config; return tidy biomass."""
    cfg = build_config(tutorial_workdir)
    cfg["validation.strict.enabled"] = "error"
    result = PythonEngine().run_in_memory(config=cfg, seed=42)
    bio_wide = result.biomass()
    bio_long = _melt_to_long(bio_wide)
    return bio_long[bio_long["species"].isin(FOCAL_SPECIES)].reset_index(drop=True)


@pytest.fixture
def perturbed_run(tutorial_workdir: Path, numba_warmup: None) -> pd.DataFrame:
    """Run the engine with the Beat-6 perturbation applied; return tidy biomass."""
    find, replace = BASELINE_PERTURBATION
    perturbed_csv = ACCESSIBILITY_CSV.replace(find, replace)
    (tutorial_workdir / "accessibility.csv").write_text(perturbed_csv)

    cfg = build_config(tutorial_workdir)
    cfg["validation.strict.enabled"] = "error"
    result = PythonEngine().run_in_memory(config=cfg, seed=42)
    bio_wide = result.biomass()
    bio_long = _melt_to_long(bio_wide)
    return bio_long[bio_long["species"].isin(FOCAL_SPECIES)].reset_index(drop=True)


EXPECTED_ROWS_PER_SPECIES = 50 * 24  # 1200; n_year × n_dt_per_year

# Pre-pinned cascade thresholds. Task 7 confirms these hold via measurement.
# If measurement shows they don't, the implementer escalates per Task 7 step 7.2 —
# they do NOT silently loosen these thresholds.
_CASCADE_FORAGER_MIN_RATIO: float = 2.0  # mean(F_pert) / mean(F_base) >= this
_CASCADE_PLANKTONEATER_MAX_RATIO: float = 0.6  # mean(PE_pert) / mean(PE_base) <= this

# Equilibrium bands per focal species. Wide-default in Task 3 (covers any plausible
# value within 15 orders of magnitude). Task 6 narrows to MEASURED equilibrium ± 20%.
_PYRAMID_BOUNDS: dict[str, tuple[float, float]] = {
    "Predator": (1.0, 1.0e15),
    "Forager": (1.0, 1.0e15),
    "PlanktonEater": (1.0, 1.0e15),
}


# === Assertion #1: the script runs to completion ===
def test_script_runs_to_completion(baseline_run: pd.DataFrame) -> None:
    """run_in_memory returns valid biomass; strict-key check passes; exact row count."""
    assert not baseline_run.empty, "biomass DataFrame is empty"
    assert set(baseline_run["species"].unique()) == set(FOCAL_SPECIES), (
        f"Expected exactly {FOCAL_SPECIES} in species column, "
        f"got {sorted(baseline_run['species'].unique())}"
    )
    per_species_rows = baseline_run.groupby("species").size()
    # Engine default output.recordfrequency.ndt=1 → one row per dt → 50×24=1200 per species.
    # Exact equality catches engine output-frequency drift (e.g., if a default flips).
    assert (per_species_rows == EXPECTED_ROWS_PER_SPECIES).all(), (
        f"Expected exactly {EXPECTED_ROWS_PER_SPECIES} rows per species "
        f"(50 yr × 24 dt); got {dict(per_species_rows)}"
    )


# === Assertion #2: biomass pyramid at equilibrium ===
def test_biomass_pyramid_emerges(baseline_run: pd.DataFrame) -> None:
    """Two layers: (a) strict ordering PlanktonEater > Forager > Predator at equilibrium —
    always tested, RED while helper stubbed. (b) ±20% bands around measured equilibrium —
    wide-default in Task 3, tightened in Task 6 from measurement."""
    means = _equilibrium_means(baseline_run)

    # Layer (a): strict pyramid ordering. This is the load-bearing narrative
    # promise — if measurement disagrees, parameters need tuning (per Task 6),
    # not loosening this assertion.
    assert means["PlanktonEater"] > means["Forager"] > means["Predator"], (
        f"Pyramid violated: PE={means['PlanktonEater']:.3e}, "
        f"F={means['Forager']:.3e}, P={means['Predator']:.3e}. "
        f"Expected PE > F > P at equilibrium."
    )

    # Layer (b): equilibrium bands. Tightened in Task 6.
    for sp, (lo, hi) in _PYRAMID_BOUNDS.items():
        assert lo <= means[sp] <= hi, (
            f"{sp} equilibrium mean {means[sp]:.3e} outside expected band "
            f"[{lo:.3e}, {hi:.3e}]"
        )


# === Assertion #3: trophic cascade visible under perturbation ===
def test_trophic_cascade_visible(
    baseline_run: pd.DataFrame, perturbed_run: pd.DataFrame
) -> None:
    """Two layers: (a) direction of change (Forager↑, PlanktonEater↓) — qualitative;
    (b) magnitude ratios ≥2.0× and ≤0.6× — pre-pinned in Task 3, validated in Task 7."""
    base = _equilibrium_means(baseline_run)
    pert = _equilibrium_means(perturbed_run)

    forager_ratio = pert["Forager"] / base["Forager"]
    pe_ratio = pert["PlanktonEater"] / base["PlanktonEater"]

    # Layer (a): direction of change.
    assert forager_ratio > 1.0, (
        f"Forager perturbed/baseline = {forager_ratio:.2f}; "
        f"expected > 1 (release from predation)."
    )
    assert pe_ratio < 1.0, (
        f"PlanktonEater perturbed/baseline = {pe_ratio:.2f}; "
        f"expected < 1 (cascade reached the bottom)."
    )

    # Layer (b): magnitude. 8× drop in accessibility should produce a strong response.
    # Pre-pinned thresholds — do not silently loosen.
    assert forager_ratio >= _CASCADE_FORAGER_MIN_RATIO, (
        f"Forager perturbed/baseline = {forager_ratio:.2f}, expected >= "
        f"{_CASCADE_FORAGER_MIN_RATIO}. Cascade visible but weak. "
        f"See Task 7 step 7.2 — adjust parameters in build_config or escalate."
    )
    assert pe_ratio <= _CASCADE_PLANKTONEATER_MAX_RATIO, (
        f"PlanktonEater perturbed/baseline = {pe_ratio:.2f}, expected <= "
        f"{_CASCADE_PLANKTONEATER_MAX_RATIO}. Cascade did not fully propagate."
    )


# === Assertion #4: the tutorial's markdown code block parses + runs ===
def test_markdown_code_block_parses_and_runs(tmp_path: Path, numba_warmup: None) -> None:
    """Extract the first ```python fence from the tutorial markdown, ast.parse it,
    then exec it in a subprocess with a 90 s timeout. Catches semantic drift —
    e.g., a renamed import (PythonEngine → OsmoseEngine) parses fine but fails to run."""
    import subprocess
    import sys

    assert TUTORIAL_MD_PATH.exists(), f"Tutorial markdown not found at {TUTORIAL_MD_PATH}"
    text = TUTORIAL_MD_PATH.read_text()
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    assert match is not None, "No ```python fence found in tutorial markdown"
    code = match.group(1)

    # Layer (a): syntactic.
    ast.parse(code)

    # Layer (b): runs to completion. Write to tmp_path/tutorial.py and exec in a
    # subprocess. Numba is warm via the numba_warmup fixture, so this is ~3-5 s.
    script_path = tmp_path / "tutorial.py"
    script_path.write_text(code)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"Markdown tutorial.py failed to execute (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Layer (c): the script writes biomass.html. Confirm non-trivial size.
    html_path = tmp_path / "tutorial-work" / "biomass.html"
    assert html_path.exists(), f"biomass.html not written at {html_path}"
    assert html_path.stat().st_size > 100_000, (
        f"biomass.html is suspiciously small ({html_path.stat().st_size} bytes); "
        f"plotly may have produced a malformed file."
    )


# === Assertion #5: the perturbation instruction is findable + replace string is disjoint ===
def test_perturbation_instruction_is_findable() -> None:
    """The Beat-6 instruction says 'find Forager;0.8;0;0, change to Forager;0.1;0;0'.
    Confirm both substrings are unique-and-disjoint in the canonical CSV."""
    find, replace = BASELINE_PERTURBATION
    assert find in ACCESSIBILITY_CSV, (
        f"Tutorial tells reader to find {find!r}, but it's not in the canonical CSV. "
        f"CSV format has drifted."
    )
    assert replace not in ACCESSIBILITY_CSV, (
        f"Replace string {replace!r} is already in the canonical CSV. "
        f"BASELINE_PERTURBATION is not a real change."
    )


# === Assertion #6: headless fallback produces meaningful equilibrium means ===
def test_headless_fallback_produces_equilibrium(baseline_run: pd.DataFrame) -> None:
    """The tutorial prints equilibrium means. Confirm: 3 species, finite, non-collapsed."""
    means = _equilibrium_means(baseline_run)
    assert len(means) == 3, f"Expected 3 species in equilibrium summary; got {len(means)}"
    assert means.notna().all(), f"Some equilibrium means are NaN: {means.to_dict()}"
    assert (means > 0).all(), f"Some equilibrium means are zero or negative: {means.to_dict()}"
    # Spread check: at least 2× separation between max and min. Catches the
    # "all species collapsed to the same tiny biomass" failure mode that would
    # otherwise satisfy the finite-and-positive check trivially.
    spread = means.max() / means.min()
    assert spread >= 2.0, (
        f"Equilibrium means are collapsed (max/min = {spread:.2f}, expected >= 2.0): "
        f"{means.to_dict()}. Food chain likely has not differentiated."
    )
```

- [ ] **Step 3.2: Format with ruff**

Run: `.venv/bin/ruff format tests/test_tutorial_3species.py`

Expected: `1 file reformatted` or `1 file already formatted`.

- [ ] **Step 3.3: Lint with ruff**

Run: `.venv/bin/ruff check tests/test_tutorial_3species.py`

Expected: `All checks passed!`. If ruff complains about unused imports, ensure every imported symbol is used in at least one test or fixture.

- [ ] **Step 3.4: Run the test, confirm RED**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py -xvs`

Expected: tests fail. Specifically:
- `test_script_runs_to_completion`: FAILS at `build_ltl(tmp_path)` with `NotImplementedError: Filled in by Task 4`
- `test_biomass_pyramid_emerges`: same (uses same fixture)
- `test_trophic_cascade_visible`: same
- `test_markdown_code_block_parses_and_runs`: FAILS at `TUTORIAL_MD_PATH.exists()` with `AssertionError: Tutorial markdown not found...` (expected — written in Task 9)
- `test_perturbation_instruction_is_findable`: PASSES (string lookups against in-memory constants, no helper needed)
- `test_headless_fallback_produces_equilibrium`: FAILS via fixture (NotImplementedError)

5 fail, 1 pass — exactly the expected RED state. The 5 failures are *real RED* (not gated behind a `pytest.fail` skip): they exercise the test's actual predicates and fail because the helper is stubbed or the markdown file doesn't exist yet.

- [ ] **Step 3.5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add tests/test_tutorial_3species.py
git -C /home/razinka/osmose/osmose-python commit -m "test(tutorial): regression test in final form (RED until Tasks 4-9)"
```

---

## Task 4: Populate `build_config` and `build_ltl` end-to-end

**Goal:** Fill in the helper functions so that the engine runs end-to-end with the synthetic 3-species model. Use a 5-year smoke (`N_YR=5`) for fast feedback during key-enumeration; assertion #1 of the test should pass when the dict is complete.

This is the largest task. Allow 30-60 minutes for the engineer to enumerate the mandatory keys and converge.

**Files:**
- Modify: `tests/_tutorial_config.py`

- [ ] **Step 4.1: Enumerate mandatory keys from the engine**

Run, capture, study:

```bash
.venv/bin/python -c "
import re, pathlib
src = pathlib.Path('osmose/engine/config.py').read_text()
# Find every _get(cfg, '...') call — these raise on missing keys
gets = re.findall(r'_get\(cfg,\s*[\"\\']([\w.{}]+)[\"\\']', src)
# Find every _species_float(cfg, '...sp{i}...') call — also mandatory per species
sps = re.findall(r'_species_float\(\s*cfg,\s*[\"\\']([\w.{}]+)[\"\\']', src)
print('=== Globally mandatory ({}) ==='.format(len(set(gets))))
for k in sorted(set(gets)):
    print(' ', k)
print('=== Per-species mandatory ({}) ==='.format(len(set(sps))))
for k in sorted(set(sps)):
    print(' ', k)
"
```

Expected: a list of ~30-50 unique keys. The per-species list typically includes `species.linf.sp{i}`, `species.k.sp{i}`, `species.t0.sp{i}`, `species.lifespan.sp{i}`, `species.egg.size.sp{i}`, `species.length2weight.condition.factor.sp{i}`, `species.length2weight.allometric.power.sp{i}`, `species.vonbertalanffy.threshold.age.sp{i}`, `predation.ingestion.rate.max.sp{i}`, `predation.efficiency.critical.sp{i}`. The global list includes `simulation.time.nyear`, `simulation.time.ndtperyear`, `simulation.nspecies`, `grid.nlon`, `grid.nlat`.

Save the output. You will populate `build_config` against this list.

- [ ] **Step 4.2: Fill in `build_ltl` first (independent of config keys)**

Open `tests/_tutorial_config.py` and replace `build_ltl`:

```python
def build_ltl(work_dir: Path) -> Path:
    """Write the constant-Plankton LTL forcing NetCDF to work_dir.

    Dims: (time, latitude, longitude). 24 timesteps (engine wraps step %
    n_dt_per_year). Constant 10000 t/cell across the 4x4 grid.

    Variable name "Plankton" matches species.name.sp3 in build_config
    (case-sensitive — osmose/engine/resources.py:216 does an exact match).
    """
    n_lon, n_lat, n_dt = 4, 4, 24
    ds = xr.Dataset(
        {
            "Plankton": (
                ("time", "latitude", "longitude"),
                np.full((n_dt, n_lat, n_lon), 10000.0),
            )
        },
        coords={
            "time": np.arange(n_dt),
            "latitude": np.arange(n_lat),
            "longitude": np.arange(n_lon),
        },
    )
    out = work_dir / "ltl.nc"
    ds.to_netcdf(out)
    return out
```

- [ ] **Step 4.3: Fill in `build_config` — minimal first attempt**

Add this function below `build_ltl`. **Start with N_YR=5** for fast feedback:

```python
def build_config(work_dir: Path) -> dict:
    """Return the engine config dict with paths resolved against work_dir.

    Synthetic 3-species model: Predator (sp0), Forager (sp1), PlanktonEater (sp2),
    plus a Plankton LTL group (sp3). See the spec at
    docs/superpowers/specs/2026-05-16-30min-tutorial-design.md
    section "The synthetic model (concrete values)" for the rationale of each
    value choice.
    """
    accessibility_path = (work_dir / "accessibility.csv").as_posix()
    ltl_path = (work_dir / "ltl.nc").as_posix()

    cfg: dict = {
        # === Simulation globals ===
        "simulation.time.nyear": 50,
        "simulation.time.ndtperyear": 24,
        "simulation.nspecies": 3,
        "simulation.nresource": 1,
        "simulation.nfisheries": 0,
        "simulation.fishing.mortality.enabled": "false",
        "mortality.subdt": 10,
        # === Grid ===
        "grid.nlon": 4,
        "grid.nlat": 4,
        # === Predation (global) ===
        "predation.accessibility.file": accessibility_path,
    }

    # Per-focal-species params: round-number toy values, pre-tuned per round-1 review.
    # larva_rate: 6/8/10 yr⁻¹ — higher for fast-cyclers to prevent boom-bust without
    # fishing or B-H stock-recruit (cf. project_phase12_cod_floor.md in MEMORY.md).
    # adult_rate: 0.15 yr⁻¹ on Predator only — proxy for what fishing would do in a
    # real config; without it the apex layer grows unbounded.
    focal_params = [
        # (name, linf, k, t0, lifespan, egg, mat, ingest, seed, larva_rate, adult_rate)
        ("Predator", 150.0, 0.12, -0.3, 25, 0.25, 50.0, 4.0, 200.0, 6.0, 0.15),
        ("Forager", 25.0, 0.40, -0.5, 8, 0.15, 12.0, 5.0, 1000.0, 8.0, 0.0),
        ("PlanktonEater", 8.0, 0.70, -0.3, 4, 0.10, 4.0, 6.0, 4000.0, 10.0, 0.0),
    ]
    for i, (
        name, linf, k, t0, lifespan, egg, mat, ingest, seed, larva_rate, adult_rate,
    ) in enumerate(focal_params):
        cfg[f"species.name.sp{i}"] = name
        cfg[f"species.linf.sp{i}"] = linf
        cfg[f"species.k.sp{i}"] = k
        cfg[f"species.t0.sp{i}"] = t0
        cfg[f"species.lifespan.sp{i}"] = lifespan
        cfg[f"species.egg.size.sp{i}"] = egg
        cfg[f"species.maturity.size.sp{i}"] = mat
        cfg[f"species.relativefecundity.sp{i}"] = 500.0
        cfg[f"species.sexratio.sp{i}"] = 0.5
        cfg[f"species.length2weight.condition.factor.sp{i}"] = 0.0070
        cfg[f"species.length2weight.allometric.power.sp{i}"] = 3.05
        cfg[f"species.vonbertalanffy.threshold.age.sp{i}"] = 1.0
        cfg[f"predation.ingestion.rate.max.sp{i}"] = ingest
        cfg[f"predation.efficiency.critical.sp{i}"] = 0.57
        cfg[f"predation.predprey.sizeratio.min.sp{i}"] = 2.0
        cfg[f"predation.predprey.sizeratio.max.sp{i}"] = 1000.0
        cfg[f"movement.distribution.method.sp{i}"] = "random"
        cfg[f"movement.randomwalk.range.sp{i}"] = 1
        cfg[f"mortality.additional.larva.rate.sp{i}"] = larva_rate
        if adult_rate > 0.0:
            cfg[f"mortality.additional.rate.sp{i}"] = adult_rate
        cfg[f"population.seeding.biomass.sp{i}"] = seed

    cfg["population.seeding.year.max"] = 20

    # LTL Plankton (sp3)
    cfg["species.name.sp3"] = "Plankton"
    cfg["species.type.sp3"] = "resource"
    cfg["species.file.sp3"] = ltl_path
    cfg["species.size.min.sp3"] = 0.0002
    cfg["species.size.max.sp3"] = 0.5
    cfg["species.trophic.level.sp3"] = 1.0
    cfg["species.accessibility2fish.sp3"] = 0.99

    return cfg
```

- [ ] **Step 4.4: Format + lint**

```bash
.venv/bin/ruff format tests/_tutorial_config.py
.venv/bin/ruff check tests/_tutorial_config.py
```

Expected: format reflows cleanly, `All checks passed!`.

- [ ] **Step 4.5: Smoke-run with N_YR=5 for fast iteration**

Write `/tmp/smoke.py`:

```python
"""Throwaway: smoke-test the tutorial config end-to-end with N_YR=5."""
import tempfile
from pathlib import Path

from osmose.engine import PythonEngine
from tests._tutorial_config import ACCESSIBILITY_CSV, build_config, build_ltl

with tempfile.TemporaryDirectory() as td:
    work = Path(td)
    (work / "accessibility.csv").write_text(ACCESSIBILITY_CSV)
    build_ltl(work)

    cfg = build_config(work)
    cfg["simulation.time.nyear"] = 5  # smoke
    cfg["validation.strict.enabled"] = "error"

    result = PythonEngine().run_in_memory(config=cfg, seed=42)
    bio = result.biomass()
    print("SHAPE:", bio.shape)
    print(bio.head())
    print(bio.tail())
```

Run: `cd /home/razinka/osmose/osmose-python && .venv/bin/python /tmp/smoke.py`

Expected: one of three outcomes:
- **(a)** Engine raises `ValueError: unknown configuration keys: [...]` — a config key in your dict is wrong/typo. Read the message, fix the typo, re-run.
- **(b)** Engine raises `KeyError: 'species.X.sp0'` or `ValueError: Missing required parameter ...` — a mandatory key is missing. Add it (cross-reference Step 4.1's enumeration), re-run.
- **(c)** Engine runs to completion, prints a biomass DataFrame. Move on.

**Iterate until outcome (c).** Each cold-start pays ~25 s of Numba JIT; subsequent runs are <2 s warm. Allow up to 10-15 iterations before stepping back to re-read `osmose/engine/config.py:1428-1500` for a missed key family.

- [ ] **Step 4.5b: Checkpoint commit at first green smoke**

The moment outcome (c) succeeds — engine runs end-to-end without errors — make a checkpoint commit. Long iterations + context-budget exhaustion = lost work if you don't:

```bash
git -C /home/razinka/osmose/osmose-python add tests/_tutorial_config.py
git -C /home/razinka/osmose/osmose-python commit -m "wip(tutorial): build_config runs end-to-end on 5-yr smoke"
```

You'll squash or replace this in Step 4.9 once assertions #1, #5, #6 are all green.

- [ ] **Step 4.6: Run the regression test — confirm assertion #1 passes**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py::test_script_runs_to_completion -xvs`

Expected: PASSES. If it fails with a different error (e.g., DataFrame shape mismatch), reconcile with the probe output from Task 2.

- [ ] **Step 4.7: Confirm assertions #5 and #6 also pass**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py::test_perturbation_instruction_is_findable tests/test_tutorial_3species.py::test_headless_fallback_produces_equilibrium -xvs`

Expected: both PASS. (Assertions #2 and #3 still RED-fail by design — measured in Tasks 6/7. Assertion #4 still RED-fails — markdown written in Task 9.)

- [ ] **Step 4.8: Delete the smoke script**

```bash
rm /tmp/smoke.py
```

- [ ] **Step 4.9: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add tests/_tutorial_config.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(tutorial): populate build_config + build_ltl; assertions #1 #5 #6 GREEN"
```

---

## Task 5: Add `numba_warmup` session-scoped fixture

**Goal:** Pay the Numba JIT compile cost once per pytest session, not once per test class. This brings the regression test from "cold ~30 s" to "warm ~5 s" for the second-and-onward run within a session.

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 5.1: Read the existing conftest**

Run: `.venv/bin/python -c "import pathlib; print(pathlib.Path('tests/conftest.py').read_text())"`

Note any existing fixture names so you don't collide. The project's conftest already has fixtures like `tmp_results_dir`, `synthetic_two_species_targets`, `synthetic_stats_in_band`, `synthetic_stats_sp_b_out_of_band` (per memory).

- [ ] **Step 5.2: Append the warmup fixture**

Append to `tests/conftest.py` (add the imports at the top if not already present):

```python
import pytest


@pytest.fixture(scope="session")
def numba_warmup():
    """Warm Numba's JIT cache once per pytest session.

    The OSMOSE engine compiles ~20-25 s of native code on first run; subsequent
    runs are <2 s. Tests that run the engine should request this fixture in their
    signature so the JIT cost is paid once per session, not once per test.

    OPT-IN (not autouse) — tests that don't run the engine (schema-only,
    MCP-credential, etc.) shouldn't pay this cost. The tutorial test requests
    it via its `baseline_run` and `perturbed_run` fixtures.

    Runs a minimal 1-year simulation (16 cells, 3 species, 1 LTL group). The
    warmup output is discarded.
    """
    import tempfile
    from pathlib import Path

    from osmose.engine import PythonEngine

    # Local import to avoid circular dependency at conftest load time
    from tests._tutorial_config import (
        ACCESSIBILITY_CSV,
        build_config,
        build_ltl,
    )

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        (work / "accessibility.csv").write_text(ACCESSIBILITY_CSV)
        build_ltl(work)
        cfg = build_config(work)
        cfg["simulation.time.nyear"] = 1  # smallest meaningful run

        PythonEngine().run_in_memory(config=cfg, seed=0)
    # No yield — one-shot setup with no teardown.
```

- [ ] **Step 5.3: Format + lint**

```bash
.venv/bin/ruff format tests/conftest.py
.venv/bin/ruff check tests/conftest.py
```

- [ ] **Step 5.4: Re-run the tutorial test, confirm warmup runs once**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py -xvs`

Expected:
- First run shows ~25-30 s wall-clock for the whole file (warmup + 2 engine sims)
- The warmup itself doesn't print anything; you'll just notice startup is slow
- Assertions #1, #5, #6 PASS; #2, #3, #4 FAIL by design

- [ ] **Step 5.5: Confirm warmup doesn't break the existing suite**

Run: `.venv/bin/pytest tests/ -x --ignore=tests/test_tutorial_3species.py -q 2>&1 | tail -30`

Expected: existing tests pass with **no change in wall-clock**. The fixture is `autouse=False`, opt-in — it only runs when a test requests it in its signature. The tutorial test does; the rest of the suite doesn't.

If existing tests break (e.g., they don't expect `_tutorial_config` to be importable), audit the fixture: it should only fail loudly if `_tutorial_config` doesn't exist, which it does by this point.

- [ ] **Step 5.6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add tests/conftest.py
git -C /home/razinka/osmose/osmose-python commit -m "test(tutorial): session-scoped numba_warmup fixture in conftest"
```

---

## Task 6: Measure equilibrium and encode pyramid margins

**Goal:** Run the full 50-year baseline simulation, measure the year-45-50 mean biomass per focal species, and encode those numbers ±20% into `_PYRAMID_BOUNDS` in the test.

**Files:**
- Modify: `tests/test_tutorial_3species.py`

- [ ] **Step 6.1: Write the measurement script**

Create `/tmp/measure.py`:

```python
"""Throwaway: measure equilibrium biomass with the locked build_config."""
import tempfile
from pathlib import Path

from osmose.engine import PythonEngine
from tests._tutorial_config import ACCESSIBILITY_CSV, build_config, build_ltl

with tempfile.TemporaryDirectory() as td:
    work = Path(td)
    (work / "accessibility.csv").write_text(ACCESSIBILITY_CSV)
    build_ltl(work)

    cfg = build_config(work)  # N_YR=50 from build_config default

    result = PythonEngine().run_in_memory(config=cfg, seed=42)
    bio_wide = result.biomass().drop(columns=["species"])
    bio_long = bio_wide.melt(id_vars="Time", var_name="species", value_name="biomass")
    focal = ["Predator", "Forager", "PlanktonEater"]
    bio_long = bio_long[bio_long["species"].isin(focal)]

    equilibrium = (
        bio_long[bio_long["Time"] >= 45]
        .groupby("species")["biomass"]
        .mean()
    )
    print("\n=== Equilibrium biomass (years 45-50 mean, baseline seed=42) ===")
    print(equilibrium)
    print("\n=== Margins (mean ± 20%) ===")
    for sp, mean in equilibrium.items():
        lo, hi = mean * 0.8, mean * 1.2
        print(f"  '{sp}': ({lo:.3e}, {hi:.3e}),")
```

- [ ] **Step 6.2: Run the measurement**

Run: `cd /home/razinka/osmose/osmose-python && .venv/bin/python /tmp/measure.py`

Expected: ~30 s wall-clock (cold-start JIT + 50 yr simulation). Output should look like:

```
=== Equilibrium biomass (years 45-50 mean, baseline seed=42) ===
species
Forager           XXXXX.XX
PlanktonEater     XXXXX.XX
Predator          XXXXX.XX
Name: biomass, dtype: float64

=== Margins (mean ± 20%) ===
  'Forager': (X.XXXe+XX, X.XXXe+XX),
  'PlanktonEater': (X.XXXe+XX, X.XXXe+XX),
  'Predator': (X.XXXe+XX, X.XXXe+XX),
```

**Sanity check:** the ordering should be `PlanktonEater > Forager > Predator` (biomass pyramid). If it's a different ordering — e.g., the Predator dominates because of a life-history quirk — **read the spec's §"Open items" #11 (in r6: §"Implementation discipline" step 6)**: the narrative is load-bearing. If measurement disagrees with the narrative, adjust the parameters (seeding biomass, larva mortality) in `build_config` until measurement matches the narrative. **Do not flip the narrative.** Re-run the measurement after each parameter adjustment.

- [ ] **Step 6.3: Copy the measured margins into the test**

Open `tests/test_tutorial_3species.py`. Find the wide-default `_PYRAMID_BOUNDS` block:

```python
_PYRAMID_BOUNDS: dict[str, tuple[float, float]] = {
    "Predator": (1.0, 1.0e15),
    "Forager": (1.0, 1.0e15),
    "PlanktonEater": (1.0, 1.0e15),
}
```

Replace with the measured bounds. Use the values printed in Step 6.2:

```python
# Equilibrium means (years 45-50, seed=42) ± 20%. Measured 2026-05-16; re-measure if
# build_config values or engine version changes (see scripts/measure_tutorial_baseline.py if checked in).
_PYRAMID_BOUNDS: dict[str, tuple[float, float]] = {
    "Predator": (X.XXXe+XX, X.XXXe+XX),   # measured mean × {0.8, 1.2}
    "Forager": (X.XXXe+XX, X.XXXe+XX),
    "PlanktonEater": (X.XXXe+XX, X.XXXe+XX),
}
```

**No `_PYRAMID_BOUNDS_MEASURED` flag** — the pyramid assertion's strict-ordering layer (a) is always tested and was GREEN from Task 4 onward; this step tightens the magnitude bands (layer b) from measurement.

- [ ] **Step 6.4: Run assertion #2 — confirm GREEN**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py::test_biomass_pyramid_emerges -xvs`

Expected: PASSES.

If it FAILS: the equilibrium values you measured don't match the values the test now sees. This shouldn't happen — `seed=42` makes the run deterministic. If it does, re-run Step 6.2 and confirm the printed values match what you encoded.

- [ ] **Step 6.5: Delete the measurement script**

```bash
rm /tmp/measure.py
```

- [ ] **Step 6.6: Commit**

If Step 6.2's sanity check forced parameter adjustments in `build_config` (e.g., bumping a `larva.rate` value), include those in a separate commit BEFORE the test-encoding commit:

```bash
git -C /home/razinka/osmose/osmose-python add tests/_tutorial_config.py
git -C /home/razinka/osmose/osmose-python commit -m "fix(tutorial): tune build_config for measured equilibrium ordering"
```

Then commit the test-encoding change:

```bash
git -C /home/razinka/osmose/osmose-python add tests/test_tutorial_3species.py
git -C /home/razinka/osmose/osmose-python commit -m "test(tutorial): encode measured equilibrium margins; assertion #2 tightened"
```

---

## Task 7: Measure perturbation and confirm cascade margins

**Goal:** Run the perturbed simulation (Forager;0.8 → Forager;0.1 in accessibility.csv), measure the year-45-50 mean ratios, and confirm assertion #3's pre-set 2.0× / 0.6× margins hold. If they don't, the parameters need adjusting (the cascade is too weak).

**Files:**
- No code edits required if the pre-pinned 2.0×/0.6× thresholds hold. Conditional edit to `tests/_tutorial_config.py` if parameter tuning is needed (see Step 7.2's decision tree).

- [ ] **Step 7.1: Write the perturbation measurement script**

Create `/tmp/cascade.py`:

```python
"""Throwaway: measure the cascade response to the Beat-6 perturbation."""
import tempfile
from pathlib import Path

from osmose.engine import PythonEngine
from tests._tutorial_config import (
    ACCESSIBILITY_CSV,
    BASELINE_PERTURBATION,
    build_config,
    build_ltl,
)


def equilibrium_means(csv_text: str) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        (work / "accessibility.csv").write_text(csv_text)
        build_ltl(work)
        cfg = build_config(work)
        result = PythonEngine().run_in_memory(config=cfg, seed=42)
        bio_wide = result.biomass().drop(columns=["species"])
        bio_long = bio_wide.melt(
            id_vars="Time", var_name="species", value_name="biomass"
        )
        focal = ["Predator", "Forager", "PlanktonEater"]
        bio_long = bio_long[bio_long["species"].isin(focal)]
        return (
            bio_long[bio_long["Time"] >= 45]
            .groupby("species")["biomass"]
            .mean()
            .to_dict()
        )


find, replace = BASELINE_PERTURBATION
baseline = equilibrium_means(ACCESSIBILITY_CSV)
perturbed = equilibrium_means(ACCESSIBILITY_CSV.replace(find, replace))

print("\n=== Baseline (seed=42, accessibility 0.8) ===")
for k, v in baseline.items():
    print(f"  {k}: {v:.3e}")
print("\n=== Perturbed (seed=42, accessibility 0.1) ===")
for k, v in perturbed.items():
    print(f"  {k}: {v:.3e}")
print("\n=== Ratios (perturbed / baseline) ===")
for k in baseline:
    ratio = perturbed[k] / baseline[k]
    print(f"  {k}: {ratio:.3f}")
print("\nAssertion #3 thresholds:")
print(f"  Forager ratio >= 2.0  (got {perturbed['Forager']/baseline['Forager']:.3f})")
print(
    f"  PlanktonEater ratio <= 0.6  "
    f"(got {perturbed['PlanktonEater']/baseline['PlanktonEater']:.3f})"
)
```

- [ ] **Step 7.2: Run and verify**

Run: `cd /home/razinka/osmose/osmose-python && .venv/bin/python /tmp/cascade.py`

Expected: ~60 s wall-clock (two 50-yr runs cold). Output shows ratios.

Decision tree:
- **If `Forager ratio >= 2.0` AND `PlanktonEater ratio <= 0.6`:** Margins hold. Proceed.
- **If `Forager ratio` is between 1.5 and 2.0:** Cascade is visible but weak. Likely Predator's pressure on Forager isn't dominant enough at baseline. Tweak in `build_config`: try increasing the Predator's `predation.ingestion.rate.max.sp0` from 4.0 to 6.0, OR increase Predator seeding biomass to 400. Re-measure pyramid (Task 6) AND cascade until both hold.
- **If `Forager ratio < 1.5`:** Cascade is barely transmitting. Inspect the size-ratio kernel — confirm Predator/Forager = 6× exceeds `sizeratio.min.sp0=2.0`. Confirm Forager/PlanktonEater = 3.1× exceeds `sizeratio.min.sp1=2.0`. Also confirm `predation.predprey.sizeratio.max.sp{i}=1000.0` is in effect (otherwise engine default 3.5× would block large-spread predation).
- **If `PlanktonEater ratio > 0.6` (cascade didn't reach the bottom):** Forager population didn't expand enough to depress PlanktonEater, OR PlanktonEater is bottom-feeder-saturated by the 10000 t/cell plankton. Try reducing PlanktonEater's seeding biomass from 4000 to 1000, OR reducing plankton biomass from 10000 to 5000 in `build_ltl`.
- **If Predator equilibrium mean is < 1.0 t (effectively extinct):** the combined adult+larva mortality compressed Predator recruitment too hard. Try reducing `mortality.additional.rate.sp0` from 0.15 to 0.05, OR remove it entirely from `build_config` and rely on larva mortality alone. Predator extinction also breaks the pyramid assertion (#2 layer a), so Task 6's measurement would have caught this first.

**Each iteration costs ~60 s of wall-clock.** **Hard stopping criterion: 8 iterations max.** If after 8 iterations the 2.0×/0.6× margins still don't hold:
- **DO NOT silently loosen** `_CASCADE_FORAGER_MIN_RATIO` or `_CASCADE_PLANKTONEATER_MAX_RATIO` in the test. The pre-pinned thresholds are spec-promised behaviour.
- Commit your best-attempt `build_config` to a separate branch and open an issue describing: (a) the measured ratios after 8 iterations, (b) what was tried, (c) which decision-tree branch each iteration landed in.
- Surface the failure in your task report. Escalate to the spec author with: "the synthetic model as specified does not produce a 2.0×/0.6× cascade under any reasonable parameter perturbation we explored. Spec may need either (a) different parameters, (b) looser margins, or (c) a different perturbation magnitude."

This is the documented failure mode for the "implementer tunes parameters to pass the test" risk in the spec (round-1 review). The thresholds stay rigid; the parameters get tuned within bounds; if the bounds don't work, the design needs revisiting, not the test.

- [ ] **Step 7.3: No flag-flip needed**

Assertion #3 is fully active (no `_CASCADE_BOUNDS_MEASURED` flag) and will run whenever pytest exercises it. Step 7.2's measurement validates that the pre-pinned thresholds (`_CASCADE_FORAGER_MIN_RATIO=2.0`, `_CASCADE_PLANKTONEATER_MAX_RATIO=0.6`) hold — no edit to the test required if measurement passes.

- [ ] **Step 7.4: Run assertion #3 — confirm GREEN**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py::test_trophic_cascade_visible -xvs`

Expected: PASSES.

- [ ] **Step 7.5: Run the full test file — confirm all 5 non-markdown assertions GREEN**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py -xvs --deselect tests/test_tutorial_3species.py::test_markdown_code_block_parses_and_runs`

Expected: 5 passed.

- [ ] **Step 7.6: Delete the cascade script**

```bash
rm /tmp/cascade.py
```

- [ ] **Step 7.7: Commit (conditional)**

If Step 7.2 required parameter tweaks to `tests/_tutorial_config.py` (which is the normal case if pre-tuning in Task 4 was insufficient), commit them:

```bash
git -C /home/razinka/osmose/osmose-python add tests/_tutorial_config.py
git -C /home/razinka/osmose/osmose-python commit -m "fix(tutorial): tune build_config for visible cascade margins"
```

You may also need to re-encode the pyramid bounds in `tests/test_tutorial_3species.py` if the tweaks shifted equilibrium values by more than the ±20% bands already encoded — re-run Step 6.2's measurement script after any `build_config` edit, and if any species' mean moves more than 15% from the Task-6 measurement, re-encode the bounds:

```bash
git -C /home/razinka/osmose/osmose-python add tests/test_tutorial_3species.py
git -C /home/razinka/osmose/osmose-python commit -m "test(tutorial): re-encode pyramid bounds after cascade tuning"
```

If the pre-pinned thresholds held without tweaks, no commit is needed here — assertion #3 was already GREEN from Task 4 onward (the wide-default bounds + pre-pinned cascade ratios). Skip to Task 8.

---

## Task 8: Verify h5netcdf in `[dev]` extras

**Goal:** Confirm the NetCDF backend the tutorial requires is in `pyproject.toml`'s `[dev]` extras (the tutorial says `pip install -e ".[dev]"` is sufficient). If missing, add it.

**Files:**
- Conditional modify: `pyproject.toml`

- [ ] **Step 8.1: Check current deps**

Run: `grep -A 30 '\[project.optional-dependencies\]\|\[tool.uv\]\|dev.*=' pyproject.toml | head -50`

Look for `h5netcdf` in the dev extras section.

- [ ] **Step 8.2: Verify import works in current venv**

Run: `.venv/bin/python -c "import h5netcdf; print(h5netcdf.__version__)"`

Expected: prints a version number (e.g., `1.3.0`). If it works, h5netcdf is installed; check pyproject.toml to confirm it's a declared dep (not just transitively pulled in).

- [ ] **Step 8.3: If missing from `[dev]` extras, add it**

Find the section, append `"h5netcdf>=1.0"` to the list. Example diff:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
    # ... existing entries ...
    "h5netcdf>=1.0",  # required by the tutorial's xr.Dataset.to_netcdf()
]
```

Run: `.venv/bin/pip install -e ".[dev]"` to re-resolve.

- [ ] **Step 8.4: Confirm xarray-to-NetCDF round-trip works**

Run: `.venv/bin/python -c "
import xarray as xr
import numpy as np
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 'test.nc'
    xr.Dataset({'X': (('t',), np.arange(5))}).to_netcdf(p)
    ds = xr.open_dataset(p)
    print('roundtrip ok:', list(ds.data_vars))
"`

Expected: `roundtrip ok: ['X']`

- [ ] **Step 8.5: Commit if pyproject changed**

```bash
git -C /home/razinka/osmose/osmose-python status --short pyproject.toml
```

If `pyproject.toml` is dirty:

```bash
git -C /home/razinka/osmose/osmose-python add pyproject.toml
git -C /home/razinka/osmose/osmose-python commit -m "build(deps): declare h5netcdf in [dev] extras (tutorial NetCDF I/O)"
```

If unchanged, no commit.

---

## Task 9: Write tutorial preamble + Beat 1 (the main paste-and-run code block)

**Goal:** Ship the first half of `docs/tutorials/30-minute-ecosystem.md`: the front-matter, audience statement, preamble, and the Beat-1 paste-and-run code block (the most load-bearing single block in the whole tutorial).

**Files:**
- Create: `docs/tutorials/30-minute-ecosystem.md` (this task writes the first ~250 lines)

The full tutorial is two tasks: Task 9 ships preamble + Beat 1, Task 10 ships Beats 2-6 + closing + troubleshooting.

- [ ] **Step 9.1: Get the version stamp values**

Run, capture:

```bash
.venv/bin/python -c "from osmose import __version__; print(__version__)"
git -C /home/razinka/osmose/osmose-python rev-parse --short HEAD
```

Note the version (e.g., `0.12.0`) and short SHA (e.g., `1585599`). Use them in Step 9.2's `<VERSION>` and `<SHA>` placeholders.

- [ ] **Step 9.2: Create the tutorial markdown file**

Create `docs/tutorials/30-minute-ecosystem.md`:

```markdown
# Build a 3-Species Ecosystem in 30 Minutes

> Tutorial validated against OSMOSE Python <VERSION> (commit <SHA>). If your install is on a different version and the code doesn't run, see the troubleshooting table at the bottom.

This is a hands-on tutorial that takes you from "I just installed OSMOSE" to "I built and perturbed a working 3-species food web." You'll paste one ~90-line Python script, run it, and watch a biomass plot. Then you'll read what each piece does. At the end, you'll change one number and see a trophic cascade.

**Audience:** Python-fluent, broadly scientific. You do **not** need to know fisheries vocabulary — we define every term inline at first use.

**Time:** ~30 minutes. First run takes about 25 seconds (Numba JIT-compiling native code); every subsequent run is under 2 seconds. The reading + understanding portion is ~20 minutes.

## Before you start

1. **Install OSMOSE** if you haven't: from the repo root, `pip install -e ".[dev]"`. This pulls `osmose`, `xarray`, `plotly`, and `h5netcdf` (the NetCDF backend) into your venv.
2. **Activate your venv:** `source .venv/bin/activate`.
3. **Heads-up on the first run:** Numba will JIT-compile native code for ~25 s. Subsequent runs are <2 s. If the script appears to hang on the first run, wait.

## Beat 1 — Run something

Paste the following block into a new file `tutorial.py`. We'll walk through every section in Beats 2-6; for now just paste, run, and look.

```python
"""3-species OSMOSE ecosystem tutorial — see docs/tutorials/30-minute-ecosystem.md."""
from pathlib import Path

import numpy as np
import xarray as xr
import plotly.express as px

from osmose.engine import PythonEngine

WORK = Path("tutorial-work").absolute()
WORK.mkdir(exist_ok=True)
N_LON, N_LAT, N_YR, N_DT = 4, 4, 50, 24

# [Beat 5] Predation accessibility CSV — Beat 6 perturbs the Forager;0.8 entry ↓
ACCESSIBILITY_CSV = """;Predator;Forager;PlanktonEater
Predator;0;0;0
Forager;0.8;0;0
PlanktonEater;0;0.8;0
Plankton;0;0.2;0.8
"""
(WORK / "accessibility.csv").write_text(ACCESSIBILITY_CSV)

# [Beat 4] LTL plankton forcing — constant 10000 t/cell over one annual cycle ↓
xr.Dataset(
    {
        "Plankton": (
            ("time", "latitude", "longitude"),
            np.full((N_DT, N_LAT, N_LON), 10000.0),
        )
    },
    coords={
        "time": np.arange(N_DT),
        "latitude": np.arange(N_LAT),
        "longitude": np.arange(N_LON),
    },
).to_netcdf(WORK / "ltl.nc")

# [Beat 2-5] The model: grid, clock, 3 focal species + 1 LTL group, no fishing ↓
config = {
    # [Beat 2] grid + time
    "grid.nlon": N_LON,
    "grid.nlat": N_LAT,
    "simulation.time.nyear": N_YR,
    "simulation.time.ndtperyear": N_DT,
    "simulation.nspecies": 3,
    "simulation.nresource": 1,
    "simulation.nfisheries": 0,
    "simulation.fishing.mortality.enabled": "false",
    "mortality.subdt": 10,
    # [Beat 3] the four keys you should understand: linf, k, lifespan, egg.size ↓
    "species.name.sp0": "Predator",
    "species.linf.sp0": 150.0,
    "species.k.sp0": 0.12,
    "species.t0.sp0": -0.3,
    "species.lifespan.sp0": 25,
    "species.egg.size.sp0": 0.25,
    "species.maturity.size.sp0": 50.0,
    "species.relativefecundity.sp0": 500.0,
    "species.sexratio.sp0": 0.5,
    "species.length2weight.condition.factor.sp0": 0.0070,
    "species.length2weight.allometric.power.sp0": 3.05,
    "species.vonbertalanffy.threshold.age.sp0": 1.0,
    "predation.ingestion.rate.max.sp0": 4.0,
    "predation.efficiency.critical.sp0": 0.57,
    "predation.predprey.sizeratio.min.sp0": 2.0,
    "predation.predprey.sizeratio.max.sp0": 1000.0,
    "movement.distribution.method.sp0": "random",
    "movement.randomwalk.range.sp0": 1,
    "mortality.additional.larva.rate.sp0": 6.0,
    "mortality.additional.rate.sp0": 0.15,
    "population.seeding.biomass.sp0": 200.0,
    # sp1 — Forager
    "species.name.sp1": "Forager",
    "species.linf.sp1": 25.0,
    "species.k.sp1": 0.40,
    "species.t0.sp1": -0.5,
    "species.lifespan.sp1": 8,
    "species.egg.size.sp1": 0.15,
    "species.maturity.size.sp1": 12.0,
    "species.relativefecundity.sp1": 500.0,
    "species.sexratio.sp1": 0.5,
    "species.length2weight.condition.factor.sp1": 0.0070,
    "species.length2weight.allometric.power.sp1": 3.05,
    "species.vonbertalanffy.threshold.age.sp1": 1.0,
    "predation.ingestion.rate.max.sp1": 5.0,
    "predation.efficiency.critical.sp1": 0.57,
    "predation.predprey.sizeratio.min.sp1": 2.0,
    "predation.predprey.sizeratio.max.sp1": 1000.0,
    "movement.distribution.method.sp1": "random",
    "movement.randomwalk.range.sp1": 1,
    "mortality.additional.larva.rate.sp1": 8.0,
    "population.seeding.biomass.sp1": 1000.0,
    # sp2 — PlanktonEater
    "species.name.sp2": "PlanktonEater",
    "species.linf.sp2": 8.0,
    "species.k.sp2": 0.70,
    "species.t0.sp2": -0.3,
    "species.lifespan.sp2": 4,
    "species.egg.size.sp2": 0.10,
    "species.maturity.size.sp2": 4.0,
    "species.relativefecundity.sp2": 500.0,
    "species.sexratio.sp2": 0.5,
    "species.length2weight.condition.factor.sp2": 0.0070,
    "species.length2weight.allometric.power.sp2": 3.05,
    "species.vonbertalanffy.threshold.age.sp2": 1.0,
    "predation.ingestion.rate.max.sp2": 6.0,
    "predation.efficiency.critical.sp2": 0.57,
    "predation.predprey.sizeratio.min.sp2": 2.0,
    "predation.predprey.sizeratio.max.sp2": 1000.0,
    "movement.distribution.method.sp2": "random",
    "movement.randomwalk.range.sp2": 1,
    "mortality.additional.larva.rate.sp2": 10.0,
    "population.seeding.biomass.sp2": 4000.0,
    # [Beat 4] LTL Plankton (sp3)
    "species.name.sp3": "Plankton",
    "species.type.sp3": "resource",
    "species.file.sp3": (WORK / "ltl.nc").as_posix(),
    "species.size.min.sp3": 0.0002,
    "species.size.max.sp3": 0.5,
    "species.trophic.level.sp3": 1.0,
    "species.accessibility2fish.sp3": 0.99,
    # [Beat 5] Predation accessibility CSV
    "predation.accessibility.file": (WORK / "accessibility.csv").as_posix(),
    "population.seeding.year.max": 20,
}

# Run + reshape biomass to tidy form for plotly
result = PythonEngine().run_in_memory(config=config, seed=42)
# biomass() returns wide-form ["Time", <sp_names>, "species"="all"]; melt to tidy.
bio_wide = result.biomass().drop(columns=["species"])
bio_long = bio_wide.melt(id_vars="Time", var_name="species", value_name="biomass")
focal = ["Predator", "Forager", "PlanktonEater"]
bio_long = bio_long[bio_long["species"].isin(focal)]

# Plot + headless fallback
fig = px.line(
    bio_long,
    x="Time",
    y="biomass",
    color="species",
    title=f"3-species ecosystem biomass over {N_YR} years",
    template="plotly_white",
)
fig.write_html(WORK / "biomass.html")
print(f"Open: {WORK / 'biomass.html'}")
print("Equilibrium biomass (years 45-50 mean):")
print(bio_long[bio_long["Time"] >= 45].groupby("species")["biomass"].mean())
```

Run it:

```bash
python tutorial.py
```

The first run takes ~25 seconds (Numba compiles native code). When it finishes, it prints a path to `tutorial-work/biomass.html` and a 3-line summary of equilibrium biomass.

**You should see** three biomass trajectories diverging from similar starts and stabilizing by year 50. The qualitative ordering matches whatever the model's equilibrium produces (most commonly: PlanktonEater highest, Forager middle, Predator lowest — a textbook biomass pyramid).

**If your plot looks substantially different** — flat zeros, runaway exponentials, or one species missing entirely — **STOP and consult the troubleshooting table at the bottom of this tutorial** before reading on.

Continued in [Beats 2-6 below](#beat-2--the-grid-and-the-clock).
```

Replace `<VERSION>` and `<SHA>` at the top with the actual values from Step 9.1.

- [ ] **Step 9.3: Run the markdown-parses test — expect GREEN**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py::test_markdown_code_block_parses_and_runs -xvs`

Expected: PASSES. The test extracts the first ```python fence and runs `ast.parse` on it.

If it fails with `SyntaxError`, copy the parser's error message verbatim, locate the matching line in the markdown, fix.

- [ ] **Step 9.4: Manually run the script — confirm it executes**

Extract the markdown's code block into a runnable file in an isolated cwd, then run it:

```bash
mkdir -p /tmp/tut9 && cd /tmp/tut9
.venv/bin/python -c "
import re, pathlib
md = pathlib.Path('/home/razinka/osmose/osmose-python/docs/tutorials/30-minute-ecosystem.md').read_text()
script = re.search(r'\`\`\`python\n(.*?)\n\`\`\`', md, re.DOTALL).group(1)
pathlib.Path('/tmp/tut9/tutorial.py').write_text(script)
"
cd /tmp/tut9
/home/razinka/osmose/osmose-python/.venv/bin/python tutorial.py
```

Expected: completes in ~25-30 s (cold JIT); prints `Open: ...biomass.html` and a 3-line equilibrium summary. The `biomass.html` file exists.

If it errors with a config-key issue, the markdown's transcription drifted from `tests/_tutorial_config.py`. Reconcile by hand.

- [ ] **Step 9.5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add docs/tutorials/30-minute-ecosystem.md
git -C /home/razinka/osmose/osmose-python commit -m "docs(tutorial): preamble + Beat 1 paste-and-run code block"
```

---

## Task 10: Write Beats 2-6 + closing + troubleshooting

**Goal:** Append the explanatory beats, closing pointers, and troubleshooting table to the tutorial. Total tutorial size after this task: ~700-900 lines.

**Files:**
- Modify: `docs/tutorials/30-minute-ecosystem.md` (append)

- [ ] **Step 10.1: Append Beats 2-6 + closing**

Open `docs/tutorials/30-minute-ecosystem.md` and append (after the Beat-1 section):

```markdown

## Beat 2 — The grid and the clock

Look at the dict block in `tutorial.py`. The first cluster of keys (under the `# [Beat 2] grid + time ↓` anchor) sets up *where* and *when* the simulation lives:

- `grid.nlon=4`, `grid.nlat=4` — a 4×4 grid of ocean cells. Real OSMOSE configs use 50×40 or larger; 4×4 is the smallest defensible setup for a learning model. The engine creates a rectangular all-ocean grid via `Grid.from_dimensions` — no NetCDF mask file needed.
- `simulation.time.nyear=50` — run the simulation for 50 model-years.
- `simulation.time.ndtperyear=24` — each year is divided into **24 timesteps** (`dt`). Fortnightly resolution. This is the OSMOSE default; enough temporal resolution for spring/summer phenology.
- `simulation.nspecies=3` and `simulation.nresource=1` — 3 focal fish species + 1 low-trophic-level resource (Plankton).
- `simulation.nfisheries=0` + `simulation.fishing.mortality.enabled="false"` — no fishing in this toy model. The Beat-6 perturbation is predation, not extraction.
- `mortality.subdt=10` — within each dt, mortality processes are integrated using 10 sub-timesteps (the OSMOSE default; finer for numerical stability).

**Quasi-equilibrium** (a term you'll see often): biomass trajectories that have stopped trending and only fluctuate around a stable mean. The model reaches quasi-equilibrium around year 20-30 for this toy; we run to 50 to have a clean equilibrium window.

## Beat 3 — The three fish

The species block has 20+ keys per focal species; **four of them matter for understanding what the model is doing**:

- `species.linf.sp{i}` — asymptotic maximum length `L∞` (in centimeters). The fish grows toward this length but never reaches it.
- `species.k.sp{i}` — von Bertalanffy growth rate. Higher K = faster approach to `L∞`.
- `species.lifespan.sp{i}` — how long an individual lives (in years).
- `species.egg.size.sp{i}` — size of newly-spawned eggs (in cm); reproduction adds new fish at this size.

The chosen values are round-number toy values, not literature-sourced:

| Species | Linf | K | Lifespan | Role |
|---|---:|---:|---:|---|
| Predator (sp0) | 150 cm | 0.12 | 25 yr | apex, large + slow |
| Forager (sp1) | 25 cm | 0.40 | 8 yr | mid-trophic |
| PlanktonEater (sp2) | 8 cm | 0.70 | 4 yr | small + fast |

The other ~16 keys per species are growth-curve and length-to-weight conversion parameters (`length2weight.condition.factor`, `length2weight.allometric.power`, `vonbertalanffy.threshold.age`, `t0`) plus mortality and seeding parameters (`mortality.additional.larva.rate`, `population.seeding.biomass`). They use OSMOSE-EEC defaults except `mortality.additional.larva.rate` (6/8/10 yr⁻¹ for Predator/Forager/PlanktonEater — higher for fast-cyclers to prevent boom-bust without fishing) and `mortality.additional.rate.sp0=0.15` on Predator only (proxy for what fishing would do; without it the apex layer grows unbounded). If you're curious, the [Baltic example documentation](../baltic_example.md) walks through every parameter family with literature provenance.

**Von Bertalanffy growth** (one paragraph): `L∞` is the asymptotic max length; `K` is how fast the fish approaches `L∞`. Large + slow fish (Predator: Linf=150 cm, K=0.12) live long; small + fast (PlanktonEater: Linf=8 cm, K=0.7) live short. The actual length-at-age curve is `L(t) = L∞ × (1 - exp(-K × (t - t0)))`.

## Beat 4 — Plankton: the bottom of the food chain

Look at the `# [Beat 4] LTL plankton forcing` block and the `# [Beat 4] LTL Plankton (sp3)` block. Together they declare a fourth species, `Plankton`, that lives at **trophic level 1.0** (primary producer; TL=2.0 is herbivores, TL=3.0+ is carnivores) and whose biomass is **fixed forcing** rather than dynamic.

Why fixed forcing? OSMOSE doesn't model phytoplankton or zooplankton dynamics — those happen on faster timescales than fish populations and are usually handled by upstream biogeochemistry models. The lowest trophic level in OSMOSE is **exogenous**: you provide a NetCDF with biomass time series, and the fish eat from it.

The `xr.Dataset(...).to_netcdf(...)` line writes a tiny NetCDF with one variable (`Plankton`), three dimensions (`time, latitude, longitude`), and shape `(24, 4, 4)`. **Time length is 24 (one annual cycle), not 50×24** — the engine wraps `step % n_dt_per_year` when reading forcing, so you only need one cycle. The biomass value is **10000 t/cell**, constant in space and time.

The variable name (`Plankton`) must exactly match `species.name.sp3` (case-sensitive — the engine does a direct dict lookup at `osmose/engine/resources.py:216`).

**Toy-model framing:** Real OSMOSE configs use spatiotemporal CMEMS-derived forcing on 50×40 grids (see `mcp_servers/copernicus/server.py` for how the Baltic example sources its forcing). This tutorial uses constant, single-group, 4×4 forcing because we're teaching the *shape* of the configuration, not the science. For the science, see [the Baltic example documentation](../baltic_example.md).

## Beat 5 — Who eats whom

The predation accessibility matrix is loaded from a CSV (`tutorial-work/accessibility.csv`), not from dict keys. Look at it:

```
;Predator;Forager;PlanktonEater
Predator;0;0;0
Forager;0.8;0;0
PlanktonEater;0;0.8;0
Plankton;0;0.2;0.8
```

Rows are prey labels (3 focal species + 1 LTL resource). Columns are predator labels (focal species only — LTL groups can't be predators). The semantics: row `Forager`, column `Predator` value `0.8` means "when Predator encounters Forager, 80% of the Forager's biomass is accessible to be eaten."

**Accessibility** ≠ encounter probability. Encounter probability comes from the size-ratio kernel (next paragraph) plus spatial overlap. Accessibility is what's left: the fraction of *encountered* prey biomass that the predator can actually consume given vertical overlap, gut hardware, and behavioural rejection.

**The trophic chain backbone** in this CSV is Plankton → PlanktonEater → Forager → Predator. Notice that Predator's column has `0.8` against Forager and `0` against PlanktonEater — Predator is a Forager specialist; it can't reach down to the PlanktonEater. This choice avoids the "predator generalist" pattern that would create a feedback loop where Predator-on-Forager perturbations get short-circuited.

**The size-ratio kernel** (`predation.predprey.sizeratio.min.sp{i}=2.0`): even if accessibility is 1.0, a predator must be at least 2× longer than its prey to eat it. The values we use:

| Predation leg | Predator Linf | Prey Linf | Ratio |
|---|---:|---:|---:|
| Predator → Forager | 150 cm | 25 cm | 6× ✓ |
| Forager → PlanktonEater | 25 cm | 8 cm | 3.1× ✓ |
| PlanktonEater → Plankton | 8 cm | ≤ 0.5 cm | ≥ 16× ✓ |

All three legs comfortably exceed the floor of 2.0, so the cascade can transmit. The accessibility matrix governs *how much* is eaten; the size-ratio kernel governs *whether* eating is possible at all.

## Beat 6 — Perturb and watch the cascade

The tutorial's punchline. Open `tutorial-work/accessibility.csv` in any text editor. Find the line:

```
Forager;0.8;0;0
```

(Be careful: there's another `0.8` on the line `Plankton;0;0.2;0.8` — do not touch that one. Search for the literal string `Forager;0.8` to find the right line.)

Change `0.8` to `0.1`. The line should now read:

```
Forager;0.1;0;0
```

Save the file. Re-run:

```bash
python tutorial.py
```

The second run is **<2 seconds** (Numba JIT is warm-cached from the first run).

Open the new `tutorial-work/biomass.html` (or look at the printed equilibrium summary). Compare with the baseline:

- **Forager biomass climbs** — it's no longer being heavily predated.
- **PlanktonEater biomass falls** — more Foragers means more pressure on the trophic level below.

This is a **trophic cascade**: an indirect effect that propagates down the food chain. The same mechanism explains real-world phenomena: when wolves were reintroduced to Yellowstone, elk numbers fell *and* willow trees rebounded (the elk weren't eating as many young willows). OSMOSE models reveal these indirect effects automatically — you don't program "willow trees benefit from wolves," it emerges from the size-structured predation kernel.

## Where next

- **Real-world Baltic example** with 8 species, CMEMS-derived forcing, and full literature provenance: [`docs/baltic_example.md`](../baltic_example.md).
- **Engine internals, Java parity, and calibration:** [`docs/parity-roadmap.md`](../parity-roadmap.md) and the Shiny UI (`shiny run app.py --host 0.0.0.0 --port 8000`).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: osmose` | venv not activated or repo not installed | `source .venv/bin/activate && pip install -e ".[dev]"` |
| `ValueError: NetCDF backend not found` | `h5netcdf` missing | `pip install h5netcdf` |
| First run "hangs" ~25 s (much longer on slow VMs / cold laptops) | Numba JIT compilation | Wait. Subsequent runs are <2 s. |
| `ValueError: unknown configuration keys` | Dict mis-typed | Compare your dict against the canonical block at the top of this tutorial. |
| Plot file path printed but won't open | Headless server | The script also prints the equilibrium values; read those. |
| Want to start over from scratch | Stale artifacts in `tutorial-work/` | `rm -rf tutorial-work/` then re-run. |
| `FileNotFoundError: ltl.nc` after editing accessibility CSV | Re-ran from a different CWD | `cd` back to the directory where `tutorial.py` lives. Paths are absolute under `tutorial-work/`. |
| Beat 6 perturbation didn't change anything | Edited the wrong `0.8` | The CSV contains *two* `0.8`s — one on `Forager;0.8;0;0` (target) and one on `Plankton;0;0.2;0.8` (last column; do NOT edit). Search for the literal string `Forager;0.8` to find the right line. |
```

- [ ] **Step 10.2: Re-run the markdown-parses test**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py::test_markdown_code_block_parses_and_runs -xvs`

Expected: PASSES. The Beats 2-6 markdown doesn't add new Python code fences (only one ```python block at the top, the rest are illustrative shell + CSV snippets).

- [ ] **Step 10.3: Reading-pass — manually skim the appended sections**

Open `docs/tutorials/30-minute-ecosystem.md` and read top-to-bottom in one pass. Look for:
- Any anchor comment in the Beat 1 code block that's referenced in Beats 2-5 but not present (or vice versa)
- Any term referenced in a beat that isn't defined or pointed at
- Any line that wraps awkwardly on a typical-width terminal (markdown is rendered fluid, but absurdly long lines hurt source readability)

Fix in place. No new commit per fix; bundle into Step 10.4.

- [ ] **Step 10.4: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add docs/tutorials/30-minute-ecosystem.md
git -C /home/razinka/osmose/osmose-python commit -m "docs(tutorial): Beats 2-6 + closing pointers + troubleshooting table"
```

---

## Task 11: Cross-references — README + tutorials index

**Goal:** Add the top-of-page "New here?" callout to `README.md`, add the documentation-index row, and create `docs/tutorials/README.md` as a one-paragraph tutorials index.

**Files:**
- Create: `docs/tutorials/README.md`
- Modify: `README.md` (two edits — callout + doc index row)

- [ ] **Step 11.1: Create `docs/tutorials/README.md`**

Write:

```markdown
# OSMOSE Tutorials

Hands-on tutorials. Self-contained, single-session. Each tutorial is a markdown walkthrough you read top-to-bottom with code you copy-paste into one Python file.

- [30-minute-ecosystem.md](30-minute-ecosystem.md) — build, run, and perturb a 3-species food web. **Start here.**
```

- [ ] **Step 11.2: Add the top-of-page callout to `README.md`**

First, verify the `## Status` header exists and capture its line number:

```bash
grep -n '^## Status' README.md
```

Expected: one match, e.g., `7:## Status`. If the result is empty or different, the README structure has drifted from what the spec described — update the spec target before editing.

Open `README.md`. Find the `## Status` header line. Insert *immediately above* it (between the one-paragraph intro and the Status table):

```markdown

> 🚀 **New here?** Build a 3-species ecosystem in 30 minutes: [Tutorial](docs/tutorials/30-minute-ecosystem.md).

```

Make sure there's a blank line before the blockquote and after it.

- [ ] **Step 11.3: Add the documentation-index row**

In `README.md`, find the `## Documentation index` table. The first row of the body is currently something like:

```markdown
| Run an existing example | [Quick start](#quick-start) above |
```

Insert a new row immediately above it:

```markdown
| Learn OSMOSE by building a 3-species ecosystem in 30 min | [`docs/tutorials/30-minute-ecosystem.md`](docs/tutorials/30-minute-ecosystem.md) |
```

- [ ] **Step 11.4: Verify all README links resolve**

Run (one-liner that checks for broken relative links):

```bash
.venv/bin/python -c "
import re, pathlib

readme = pathlib.Path('README.md').read_text()
for href in re.findall(r'\]\((?!http)([^)#]+)', readme):
    target = pathlib.Path(href.lstrip('./'))
    if not target.exists():
        print(f'BROKEN: {href}')
    else:
        print(f'  ok: {href}')
"
```

Expected: every link prints `ok: ...`. If any prints `BROKEN: ...`, fix the path.

- [ ] **Step 11.5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add docs/tutorials/README.md README.md
git -C /home/razinka/osmose/osmose-python commit -m "docs(tutorial): top-of-page README callout + doc-index row + tutorials index"
```

---

## Task 12: Final regression-test pass + acceptance verification

**Goal:** Run the entire regression test, manually walk through the tutorial in a fresh `/tmp/` cwd as if you were a new reader, confirm acceptance criteria 1-6 from the spec.

**Files:** None modified. This task is verification.

- [ ] **Step 12.1: Run the regression test, no deselections**

Run: `.venv/bin/pytest tests/test_tutorial_3species.py -v`

Expected: **6 passed in under 45 seconds** (cold). All 6 assertions GREEN.

- [ ] **Step 12.2: Run the wider test suite to confirm no regressions**

Run: `.venv/bin/pytest tests/ -q --ignore=tests/test_tutorial_3species.py 2>&1 | tail -10`

Expected: pre-existing passing tests still pass. The `numba_warmup` fixture is `autouse=False`, opt-in — pre-existing tests don't request it, so their wall-clock is unchanged.

- [ ] **Step 12.3: Acceptance #1 — fresh-venv smoke test**

Simulate a fresh reader. From any directory. **CRITICAL: do NOT remove the `cd /tmp/ac1-fresh` step** — the tutorial's `WORK = Path("tutorial-work").absolute()` resolves against the current working directory; running from the repo root would pollute the repo with `tutorial-work/`.

```bash
mkdir -p /tmp/ac1-fresh && cd /tmp/ac1-fresh
.venv/bin/python -c "
import re, pathlib
md = pathlib.Path('/home/razinka/osmose/osmose-python/docs/tutorials/30-minute-ecosystem.md').read_text()
script = re.search(r'\`\`\`python\n(.*?)\n\`\`\`', md, re.DOTALL).group(1)
pathlib.Path('tutorial.py').write_text(script)
"
/home/razinka/osmose/osmose-python/.venv/bin/python tutorial.py
```

Expected: completes; prints `Open: ...biomass.html`; prints the equilibrium summary; `tutorial-work/biomass.html` exists.

Open the HTML in a browser (or capture its size to confirm it's a real plotly HTML — `wc -c tutorial-work/biomass.html` should print something in the millions of bytes).

- [ ] **Step 12.4: Acceptance #2 — perturbation in the fresh shell**

Still in `/tmp/ac1-fresh`:

```bash
sed -i 's/Forager;0.8;0;0/Forager;0.1;0;0/' tutorial-work/accessibility.csv
/home/razinka/osmose/osmose-python/.venv/bin/python tutorial.py
```

Expected: re-runs in ~2 s (warm Numba); prints a different equilibrium summary. Forager biomass should be visibly higher, PlanktonEater visibly lower.

Compare numerically:
```bash
grep -A 3 'Equilibrium biomass' tutorial-work/*.html 2>/dev/null || .venv/bin/python -c "
# (just re-run the script and capture print output by piping; the summary
# was already printed in Step 12.4 above)
"
```

Visual check: the difference between baseline and perturbed equilibrium summaries should be obvious without ambiguity.

- [ ] **Step 12.5: Acceptance #3 — regression test passes <45 s**

Already covered by Step 12.1. Re-confirm wall-clock:

```bash
time .venv/bin/pytest tests/test_tutorial_3species.py -q
```

Expected: passes in under 45 s (the wall time printed at the bottom).

- [ ] **Step 12.6: Acceptance #4 — all README links resolve**

Run the script from Step 11.4 again. Expected: all `ok:`.

- [ ] **Step 12.7: Acceptance #5 — strict-key validation produces no warnings**

The test already runs with `cfg["validation.strict.enabled"] = "error"`. If Step 12.1 passed, this is satisfied.

Defensive double-check: confirm by reading the assertion's `baseline_run` fixture in `tests/test_tutorial_3species.py` — the `cfg["validation.strict.enabled"] = "error"` line should be present.

- [ ] **Step 12.8: Acceptance #6 — troubleshooting covers 8 symptoms**

Open the tutorial markdown's troubleshooting table. Confirm exactly 8 rows:
1. ModuleNotFoundError
2. NetCDF backend not found
3. First run "hangs"
4. unknown configuration keys
5. Plot file path printed but won't open
6. Want to start over
7. FileNotFoundError: ltl.nc (wrong CWD)
8. Beat 6 perturbation didn't change anything

If any row is missing, add it.

- [ ] **Step 12.9: Clean up the smoke directory**

```bash
rm -rf /tmp/ac1-fresh
```

- [ ] **Step 12.10: Final commit (memory + acceptance)**

If any acceptance step required a fix, those would be amend-worthy commits in Tasks 9-11. Assuming no fixes needed:

```bash
git -C /home/razinka/osmose/osmose-python log --oneline -15
```

Expected: 12 commits since `2cfb168` (the prior session's HEAD), one per task. If the count is off (some tasks needed multiple commits or some bundled together), that's fine — the count is a sanity check, not a requirement.

The tutorial is ready to ship. Post-merge bookkeeping (not in this plan):
- Add a memory entry `project_30min_tutorial_shipped.md`
- Add a one-line entry under `MEMORY.md`'s "Status + roadmap" pointing at the memory file
- Optional: tag the commit with the next version bump (deferred to a separate release task)

---

## Self-review checklist for the implementer

Before declaring the plan complete, the implementer should confirm:

- [ ] All 6 regression assertions pass: `.venv/bin/pytest tests/test_tutorial_3species.py -v` shows `6 passed`.
- [ ] The tutorial markdown is syntactically valid Python in its main code block (the `test_markdown_code_block_parses_and_runs` assertion).
- [ ] Manual smoke of the tutorial in `/tmp/` produces a `biomass.html` plot (Step 12.3).
- [ ] The Beat-6 perturbation visibly changes the plot (Step 12.4).
- [ ] All README links resolve (Step 12.6).
- [ ] No new ruff failures: `.venv/bin/ruff check osmose/ ui/ tests/` returns 0.
- [ ] No regression in the wider suite: `.venv/bin/pytest tests/ -q` passes (modulo any pre-existing skipped/deselected).
- [ ] Each task's commit message follows project conventions (`docs(tutorial): ...`, `feat(tutorial): ...`, `test(tutorial): ...`).
- [ ] Each commit is reviewable in isolation (no "fix typo" follow-ups bundled into feature commits).
