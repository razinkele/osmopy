"""Regression test for the 30-minute 3-species tutorial.

Two layers of assertion (per round-1 review):
- Always-on layer (ordering, direction-of-change, ratios): tests load-bearing
  qualitative behaviour. RED while helper is stubbed; GREEN as soon as Task 4
  fills the helper.
- Tightening layer (equilibrium ±20% bands): pre-set to wide-default in Task 3,
  narrowed in Task 6 from MEASURED values. Catches engine-behaviour drift.

If `build_config`, `ACCESSIBILITY_CSV`, or `build_ltl` in
`tests/_tutorial_config.py` change, update
`docs/tutorials/30-minute-ecosystem.md` to match.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
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
EXPECTED_ROWS_PER_SPECIES = 50 * 24  # 1200; n_year × n_dt_per_year

TUTORIAL_MD_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "30-minute-ecosystem.md"
)

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


def _melt_to_long(bio_wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape biomass() output from wide to tidy long form."""
    return bio_wide.drop(columns=["species"]).melt(
        id_vars="Time", var_name="species", value_name="biomass"
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
            f"{sp} equilibrium mean {means[sp]:.3e} outside expected band [{lo:.3e}, {hi:.3e}]"
        )


# === Assertion #3: trophic cascade visible under perturbation ===
def test_trophic_cascade_visible(baseline_run: pd.DataFrame, perturbed_run: pd.DataFrame) -> None:
    """Two layers: (a) direction of change (Forager↑, PlanktonEater↓) — qualitative;
    (b) magnitude ratios ≥2.0× and ≤0.6× — pre-pinned in Task 3, validated in Task 7."""
    base = _equilibrium_means(baseline_run)
    pert = _equilibrium_means(perturbed_run)

    forager_ratio = pert["Forager"] / base["Forager"]
    pe_ratio = pert["PlanktonEater"] / base["PlanktonEater"]

    # Layer (a): direction of change.
    assert forager_ratio > 1.0, (
        f"Forager perturbed/baseline = {forager_ratio:.2f}; expected > 1 (release from predation)."
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
