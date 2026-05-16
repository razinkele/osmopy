"""Regression test for the 30-minute 3-species tutorial (Baltic substrate).

Two layers of assertion (per round-1 review):
- Always-on layer (ordering, direction-of-change, ratios): tests load-bearing
  qualitative behaviour.
- Tightening layer (equilibrium ±20% bands): pre-set to wide-default in Task 3,
  narrowed in Task 6 from MEASURED values. Catches engine-behaviour drift.

The tutorial uses the data/baltic/ 8-species calibrated config with cod, sprat,
and stickleback highlighted for the trophic cascade narrative.

Cascade mechanics note (Baltic-substrate finding):
  The dominant cascade signal is: drop cod-sprat accessibility → cod has less
  food → cod starvation increases slightly → cod biomass stays lower/declines
  → stickleback experiences less cod predation → stickleback UP.
  The sprat signal is small (<2 %) because cod is a minor predator of sprat in
  the Baltic (bottom-up controlled ecosystem).  Thresholds are set to match the
  measured cascade from smoke runs; they are tight enough to detect regression
  but do not pre-suppose a cascade magnitude stronger than the model produces.

If `build_config` or `BASELINE_PERTURBATION` in `tests/_tutorial_config.py`
change, update `docs/tutorials/30-minute-ecosystem.md` to match.
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
    BALTIC_DIR,
    ACCESSIBILITY_CSV_RELPATH,
    BASELINE_PERTURBATION,
    build_config,
)

FOCAL_SPECIES = ["cod", "sprat", "stickleback"]

# Baltic output.recordfrequency.ndt = 24 (annual records).
# For a 30-year run: exactly 30 rows per species.
EXPECTED_ROWS_PER_SPECIES = 30  # n_year (not n_year × 24; Baltic records annually)

TUTORIAL_MD_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "30-minute-ecosystem.md"
)

# Equilibrium window: years 5-25. Cod still exists and shows cascade dynamics
# during this window; beyond year 25 cod collapses in the uncalibrated state.
_EQ_WINDOW_START: float = 5.0
_EQ_WINDOW_END: float = 25.0

# Measured cascade thresholds (from smoke runs on data/baltic/ with seed=42,
# 30 yr, perturbation sprat;0.4 -> sprat;0.05, window years 5-25):
#   sprat:       0.994x  (barely changes; cod is a minor sprat predator)
#   stickleback: 1.133x  (rises as cod predation pressure decreases)
# Thresholds are set conservatively below/above the measured values so they
# catch genuine regressions but are not brittle to small RNG variation.
#
# Interpretation: cascade direction is STICKLEBACK UP (not down as a
# classical top-down cascade would predict), because Baltic cod is in a
# bottom-up-controlled state where removing its sprat access starves cod
# slightly, reducing cod's predation on stickleback.
_CASCADE_STICKLEBACK_MIN_RATIO: float = 1.02  # mean(S_pert) / mean(S_base) >= this
_CASCADE_SPRAT_MAX_DELTA: float = 0.10        # |mean(Sp_pert)/mean(Sp_base) - 1| <= this

# Equilibrium bands per focal species. Measured from equilibrium window
# (years 5-25, seed=42) and encoded as ± 20%. Values are (lower, upper) in tonnes.
# Measured 2026-05-17 against the Baltic substrate. Re-measure if build_config
# values or engine version change.
_PYRAMID_BOUNDS: dict[str, tuple[float, float]] = {
    "cod":          (7.238e+02, 1.086e+03),
    "sprat":        (4.418e+06, 6.627e+06),
    "stickleback":  (4.342e+05, 6.513e+05),
}


def _melt_to_long(bio_wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape biomass() output from wide to tidy long form.

    Baltic biomass() returns a wide DataFrame with columns
    [Time, cod, herring, ..., species] where the 'species' column holds the
    constant value 'all'.  Drop 'species' before melting.
    """
    drop_cols = [c for c in ["species"] if c in bio_wide.columns]
    return bio_wide.drop(columns=drop_cols).melt(
        id_vars="Time", var_name="species", value_name="biomass"
    )


def _equilibrium_means(bio_long: pd.DataFrame) -> pd.Series:
    """Mean biomass per focal species over the equilibrium window (years 5-25)."""
    window = bio_long[
        (bio_long["Time"] >= _EQ_WINDOW_START) & (bio_long["Time"] <= _EQ_WINDOW_END)
    ]
    focal = window[window["species"].isin(FOCAL_SPECIES)]
    return focal.groupby("species")["biomass"].mean()


@pytest.fixture
def baseline_run(tmp_path: Path, numba_warmup: None) -> pd.DataFrame:
    """Run the engine with the baseline Baltic config; return tidy biomass.

    Uses tmp_path/base/ as the workdir to avoid collision with perturbed_run
    when both fixtures are requested by the same test function.
    """
    workdir = tmp_path / "base"
    workdir.mkdir()
    cfg = build_config(workdir)
    cfg["simulation.rng.fixed"] = "true"
    result = PythonEngine().run_in_memory(config=cfg, seed=42)
    bio_wide = result.biomass()
    bio_long = _melt_to_long(bio_wide)
    return bio_long[bio_long["species"].isin(FOCAL_SPECIES)].reset_index(drop=True)


@pytest.fixture
def perturbed_run(tmp_path: Path, numba_warmup: None) -> pd.DataFrame:
    """Run the engine with the Beat-6 perturbation applied; return tidy biomass.

    Uses tmp_path/pert/ as the workdir to avoid collision with baseline_run.
    The perturbation edits predation-accessibility.csv in the workdir copy
    (never touches data/baltic/).
    """
    import shutil  # noqa: PLC0415

    workdir = tmp_path / "pert"
    workdir.mkdir()
    target = workdir / "baltic"
    shutil.copytree(BALTIC_DIR, target)

    # Apply perturbation to the copied CSV.
    acc_path = target / ACCESSIBILITY_CSV_RELPATH
    find, replace = BASELINE_PERTURBATION
    original = acc_path.read_text()
    assert find in original, (
        f"Perturbation find-string {find!r} not found in copied accessibility CSV. "
        f"CSV format may have changed."
    )
    acc_path.write_text(original.replace(find, replace))

    # Load config directly to avoid a second copytree call.
    from osmose.config.reader import OsmoseConfigReader  # noqa: PLC0415
    reader = OsmoseConfigReader()
    cfg = reader.read(str(target / "baltic_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "30"
    cfg["simulation.rng.fixed"] = "true"

    result = PythonEngine().run_in_memory(config=cfg, seed=42)
    bio_wide = result.biomass()
    bio_long = _melt_to_long(bio_wide)
    return bio_long[bio_long["species"].isin(FOCAL_SPECIES)].reset_index(drop=True)


# === Assertion #1: the script runs to completion ===
def test_script_runs_to_completion(baseline_run: pd.DataFrame) -> None:
    """run_in_memory returns valid biomass; Baltic 3 focal species present; exact row count.

    Baltic records biomass annually (output.recordfrequency.ndt = 24) so for a
    30-year run we expect exactly 30 rows per focal species.
    """
    assert not baseline_run.empty, "biomass DataFrame is empty"
    assert set(baseline_run["species"].unique()) == set(FOCAL_SPECIES), (
        f"Expected exactly {FOCAL_SPECIES} in species column, "
        f"got {sorted(baseline_run['species'].unique())}"
    )
    per_species_rows = baseline_run.groupby("species").size()
    assert (per_species_rows == EXPECTED_ROWS_PER_SPECIES).all(), (
        f"Expected exactly {EXPECTED_ROWS_PER_SPECIES} rows per species "
        f"(30 yr × 1 annual record); got {dict(per_species_rows)}"
    )


# === Assertion #2: biomass pyramid at equilibrium ===
def test_biomass_pyramid_emerges(baseline_run: pd.DataFrame) -> None:
    """Two layers: (a) strict ordering sprat > stickleback > cod at equilibrium —
    always tested. (b) ±20% bands around measured equilibrium —
    wide-default in Task 3, tightened in Task 6 from measurement.

    Pyramid rationale: sprat is a large-biomass planktivore (~5-7 Mt);
    stickleback is a smaller forage fish (~100K-1Mt in early years);
    cod is a TL4 predator with low biomass in the Baltic overfished state.
    """
    means = _equilibrium_means(baseline_run)

    # Layer (a): strict pyramid ordering (measured from smoke runs, seed=42).
    assert means["sprat"] > means["stickleback"] > means["cod"], (
        f"Pyramid violated: sprat={means['sprat']:.3e}, "
        f"stickleback={means['stickleback']:.3e}, cod={means['cod']:.3e}. "
        f"Expected sprat > stickleback > cod at equilibrium (years {_EQ_WINDOW_START}-{_EQ_WINDOW_END})."
    )

    # Layer (b): equilibrium bands. Tightened in Task 6.
    for sp, (lo, hi) in _PYRAMID_BOUNDS.items():
        assert lo <= means[sp] <= hi, (
            f"{sp} equilibrium mean {means[sp]:.3e} outside expected band [{lo:.3e}, {hi:.3e}]"
        )


# === Assertion #3: trophic cascade visible under perturbation ===
def test_trophic_cascade_visible(baseline_run: pd.DataFrame, perturbed_run: pd.DataFrame) -> None:
    """Two layers: (a) direction of change — stickleback UP when cod-sprat acc drops;
    (b) magnitude — stickleback ratio >= 1.02, sprat ratio within ±10% of 1.0.

    Baltic cascade mechanics (measured from smoke runs):
      Reducing cod-sprat accessibility starves cod slightly (less food) which
      reduces cod biomass, lowering predation pressure on stickleback.
      Result: stickleback UP ~7-13%, sprat essentially unchanged (<2%).
      This bottom-up-controlled cascade is the ecologically realistic signal
      for a Baltic Sea in an overfished state.

    Pre-pinned thresholds from smoke run measurements (years 5-25, seed=42):
      stickleback ratio: 1.133x measured → threshold 1.02 (conservative)
      sprat ratio: 0.994x measured → |delta| <= 0.10 (sprat barely moves)
    """
    base = _equilibrium_means(baseline_run)
    pert = _equilibrium_means(perturbed_run)

    stickleback_ratio = pert["stickleback"] / base["stickleback"]
    sprat_ratio = pert["sprat"] / base["sprat"]

    # Layer (a): direction of change — stickleback goes UP.
    assert stickleback_ratio > 1.0, (
        f"Stickleback perturbed/baseline = {stickleback_ratio:.3f}; expected > 1.0 "
        f"(cod starvation releases stickleback from predation). "
        f"base={base['stickleback']:.3e}, pert={pert['stickleback']:.3e}"
    )

    # Layer (a): sprat signal is near zero (cod is a minor sprat predator).
    sprat_delta = abs(sprat_ratio - 1.0)
    assert sprat_delta <= _CASCADE_SPRAT_MAX_DELTA, (
        f"Sprat perturbed/baseline = {sprat_ratio:.3f} (|delta|={sprat_delta:.3f}); "
        f"expected |delta| <= {_CASCADE_SPRAT_MAX_DELTA} (sprat should barely change). "
        f"base={base['sprat']:.3e}, pert={pert['sprat']:.3e}"
    )

    # Layer (b): magnitude check.
    assert stickleback_ratio >= _CASCADE_STICKLEBACK_MIN_RATIO, (
        f"Stickleback perturbed/baseline = {stickleback_ratio:.3f}, expected >= "
        f"{_CASCADE_STICKLEBACK_MIN_RATIO}. Cascade visible but weaker than measured. "
        f"See Task 7 — check if Baltic config or engine has changed."
    )


# === Assertion #4: the tutorial's markdown code block parses + runs ===
def test_markdown_code_block_parses_and_runs(tmp_path: Path, numba_warmup: None) -> None:
    """Extract the first ```python fence from the tutorial markdown, ast.parse it,
    then exec it in a subprocess with a 90 s timeout. Catches semantic drift —
    e.g., a renamed import (PythonEngine -> OsmoseEngine) parses fine but fails to run."""
    assert TUTORIAL_MD_PATH.exists(), f"Tutorial markdown not found at {TUTORIAL_MD_PATH}"
    text = TUTORIAL_MD_PATH.read_text()
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    assert match is not None, "No ```python fence found in tutorial markdown"
    code = match.group(1)

    # Layer (a): syntactic.
    ast.parse(code)

    # Layer (b): runs to completion.
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
    """The Beat-6 instruction says 'find sprat;0.4;, change to sprat;0.05;'.
    Confirm find-string is present and replace-string is absent in the canonical CSV."""
    find, replace = BASELINE_PERTURBATION
    canonical_csv = (BALTIC_DIR / ACCESSIBILITY_CSV_RELPATH).read_text()

    assert find in canonical_csv, (
        f"Tutorial tells reader to find {find!r}, but it's not in the canonical CSV. "
        f"CSV format has drifted.\nCanonical CSV path: {BALTIC_DIR / ACCESSIBILITY_CSV_RELPATH}"
    )
    assert replace not in canonical_csv, (
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
    # Spread check: at least 10× separation between max and min.
    # Baltic's sprat is ~5M tonnes, cod is ~1.5K tonnes -> spread >> 100.
    # The threshold is set loose (10×) to avoid brittleness but ensures the
    # food chain has differentiated beyond a single biomass level.
    spread = means.max() / means.min()
    assert spread >= 10.0, (
        f"Equilibrium means are collapsed (max/min = {spread:.2f}, expected >= 10.0): "
        f"{means.to_dict()}. Food chain likely has not differentiated."
    )
