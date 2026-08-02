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

If `build_config` or `apply_cod_sprat_perturbation` in `tests/_tutorial_config.py`
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
    FOCAL_SPECIES,
    add_total_cod,
    apply_cod_sprat_perturbation,
    build_config,
)

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

# Cascade thresholds. The perturbation (drop cod->sprat accessibility 0.4 -> 0.05)
# lowers cod predation pressure on stickleback. Only the QUALITATIVE, environment-
# robust claims are asserted here:
#   - stickleback stays the same order of magnitude (does not crash or explode)
#   - sprat barely moves (cod is a minor sprat predator)
#   - the biomass pyramid ordering sprat > stickleback > cod survives (see the test)
#
# The cascade MAGNITUDE is NOT reproducible across environments and must not be
# asserted tightly: same code + same seed gives stickleback ratio ~0.99 (a 2-core
# CI runner), ~1.13 (an 8-core dev box), and ~1.35 (an i9-10940X dev box). This is
# NOT thread count and NOT the @njit mortality kernel: a within-machine Numba thread
# sweep (N=1..28) and a re-run at the commit where the band was first pinned
# (94106a7) each reproduce a given machine's value bit-for-bit, and forcing Numba to
# a generic SSE2 target reproduces the host AVX-512 value exactly. The spread is the
# numerical library/CPU environment amplifying tiny FP differences over the 30-year
# chaotic run — bit-reproducibility across machines is not achievable, so assert the
# signal, not the number. (Earlier "thread-topology sensitive" comments were wrong.)
_CASCADE_STICKLEBACK_MIN_RATIO: float = 0.5  # does not crash (within ~2x down)
_CASCADE_STICKLEBACK_MAX_RATIO: float = 2.0  # ...nor explode (within ~2x up)
# Sprat RELEASE band. Measured 1.241 (2026-08-02, #129) — the direct cascade link.
# Deliberately wide: magnitude is environment-specific (see the note above), direction is not.
_CASCADE_SPRAT_MIN_RATIO: float = 1.05  # sprat must go UP when cod loses access...
_CASCADE_SPRAT_MAX_RATIO: float = 1.60  # ...without running away

# Equilibrium bands per focal series. Measured from the window (years 5-25, seed=42) and
# encoded as ± 20%. Values are (lower, upper) in tonnes.
#
# RE-MEASURED 2026-08-02 (#129). Every band moved, and not only because "cod" was retargeted
# to total_cod: sprat fell ~5x (3.5-7.5e6 -> 9.14e5) and stickleback ~5x (3.0-6.5e5 -> 7.49e4)
# against bands last set 2026-06-24. The Baltic recalibration since then moved cod up two
# orders of magnitude (aggregate ~1-2.4 kt -> total 156 kt, consistent with cod_east now
# sitting inside its 60-85 kt ICES envelope). Per-stock: cod_west 10,519 t, cod_east 145,419 t
# — the split is heavily eastern, so following cod_west alone would track 6.7% of the stock.
#
# CAVEAT on the window: years 5-25 of a 30-year run is NOT equilibrium for this config.
# Certification uses the final decade of 50 years and puts cod_east at ~83 kt; this window
# shows 145 kt because it still contains the seeding transient. These bands characterise the
# tutorial run, not the certified equilibrium — do not cite them as the latter.
# Re-measured 2026-06-24 after the egg-retention fix (94f1bfb): gating predation
# on the released egg fraction lets more eggs survive, so cod recovers to ~2x its
# prior equilibrium (sprat/stickleback barely move). The fix mechanism was
# Java-validated on EEC (14 species within 0.807-1.724x of the 4.3.3 engine).
# Re-measure if build_config values or engine version change.
# NOTE: bands are intentionally generous. These emergent equilibria are numerically
# non-reproducible across environments — identical code+seed gives cod ~1.6e3 on some
# machines and ~2.0e3 on others (a 2-core CI runner). The spread is the library/CPU
# environment amplifying tiny FP differences over the 30-year chaotic run, NOT thread
# count and NOT the @njit mortality kernel (both proven invariant here; see the cascade
# constants above). The load-bearing assertion is the pyramid ORDERING (sprat >
# stickleback > cod); these bands are a coarse order-of-magnitude guard spanning the
# observed range.
_PYRAMID_BOUNDS: dict[str, tuple[float, float]] = {
    "total_cod": (1.248e05, 1.871e05),
    "sprat": (7.313e05, 1.097e06),
    "stickleback": (5.989e04, 8.983e04),
}


def _melt_to_long(bio_wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape biomass() output from wide to tidy long form.

    Baltic biomass() returns a wide DataFrame with columns
    [Time, cod_west, herring, ..., species] where the 'species' column holds the
    constant value 'all'.  Drop 'species' before melting. Callers pass the frame
    through add_total_cod() first so 'total_cod' is present.
    """
    drop_cols = [c for c in ["species"] if c in bio_wide.columns]
    return bio_wide.drop(columns=drop_cols).melt(
        id_vars="Time", var_name="species", value_name="biomass"
    )


def _equilibrium_means(bio_long: pd.DataFrame) -> pd.Series:
    """Mean biomass per focal species over the equilibrium window (years 5-25)."""
    window = bio_long[(bio_long["Time"] >= _EQ_WINDOW_START) & (bio_long["Time"] <= _EQ_WINDOW_END)]
    focal = window[window["species"].isin(FOCAL_SPECIES)]
    return focal.groupby("species")["biomass"].mean()


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory: pytest.TempPathFactory, numba_warmup: None) -> pd.DataFrame:
    """Run the engine with the baseline Baltic config; return tidy biomass.

    Module-scoped: the 30-year Baltic sim is shared across every test that
    consumes it (consumers only read the frame, never mutate it). Uses a
    factory-allocated workdir to avoid colliding with perturbed_run.
    """
    workdir = tmp_path_factory.mktemp("base")
    cfg = build_config(workdir)
    cfg["validation.strict.enabled"] = "error"
    result = PythonEngine().run_in_memory(config=cfg, seed=42)
    bio_wide = add_total_cod(result.biomass())
    bio_long = _melt_to_long(bio_wide)
    return bio_long[bio_long["species"].isin(FOCAL_SPECIES)].reset_index(drop=True)


@pytest.fixture(scope="module")
def perturbed_run(tmp_path_factory: pytest.TempPathFactory, numba_warmup: None) -> pd.DataFrame:
    """Run the engine with the Beat-6 perturbation applied; return tidy biomass.

    Module-scoped to mirror baseline_run. Uses a factory-allocated workdir to
    avoid collision with baseline_run. The perturbation edits
    predation-accessibility.csv in the workdir copy (never touches data/baltic/).
    """
    import shutil  # noqa: PLC0415

    workdir = tmp_path_factory.mktemp("pert")
    target = workdir / "baltic"
    shutil.copytree(BALTIC_DIR, target)

    # Apply perturbation to the copied CSV (never touches data/baltic/).
    # Edits BOTH cod stocks by column name — see apply_cod_sprat_perturbation for why the
    # old positional replace silently missed cod_east.
    acc_path = target / ACCESSIBILITY_CSV_RELPATH
    before = apply_cod_sprat_perturbation(acc_path)
    assert set(before) == {"cod_west", "cod_east"} and all(v > 0.05 for v in before.values()), (
        f"Expected both cod stocks to start above the perturbed value; got {before}"
    )

    # Load config directly to avoid a second copytree call.
    from osmose.config.reader import OsmoseConfigReader  # noqa: PLC0415

    reader = OsmoseConfigReader()
    cfg = reader.read(str(target / "baltic_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "30"
    cfg["validation.strict.enabled"] = "error"

    result = PythonEngine().run_in_memory(config=cfg, seed=42)
    bio_wide = add_total_cod(result.biomass())
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
    """Two layers: (a) prey biomass exceeds predator biomass — always tested.
    (b) ±20% bands around the measured window (re-measured 2026-08-02, #129).

    **The claim is sprat > total_cod, and only that.** The prior assertion was
    sprat > stickleback > cod, which broke once "cod" correctly meant both stocks
    (total_cod 1.56e5 > stickleback 7.49e4). It was never a pyramid statement anyway:
    stickleback sits at roughly sprat's trophic level, so ordering the two forage fish
    against each other says "stickleback is scarce here", not anything trophic.

    What IS a biomass-pyramid statement, and what this asserts: the forage fish
    outweighs its predator, sprat 9.14e5 vs total_cod 1.56e5 (~5.9x).
    """
    means = _equilibrium_means(baseline_run)

    # Layer (a): prey outweighs predator — the actual trophic-pyramid claim.
    assert means["sprat"] > means["total_cod"], (
        f"Prey does not outweigh predator: sprat={means['sprat']:.3e}, "
        f"total_cod={means['total_cod']:.3e} over years "
        f"{_EQ_WINDOW_START}-{_EQ_WINDOW_END}. "
        f"(stickleback={means['stickleback']:.3e}, reported for context only — it is not "
        f"part of the ordering claim.)"
    )

    # Layer (b): equilibrium bands. Tightened in Task 6.
    for sp, (lo, hi) in _PYRAMID_BOUNDS.items():
        assert lo <= means[sp] <= hi, (
            f"{sp} equilibrium mean {means[sp]:.3e} outside expected band [{lo:.3e}, {hi:.3e}]"
        )


# === Assertion #3: trophic cascade visible under perturbation ===
def test_trophic_cascade_visible(baseline_run: pd.DataFrame, perturbed_run: pd.DataFrame) -> None:
    """Environment-robust cascade signal (the magnitude is NOT reproducible — see
    the constants above for why: it varies ~0.99/1.13/1.35 across machines for
    identical code+seed, and that spread is library/CPU, not thread count).

    Dropping BOTH cod stocks' accessibility to sprat releases sprat directly. The
    load-bearing, cross-environment claims are qualitative:
      (a) stickleback stays the same order of magnitude (does not crash or explode),
      (b) sprat is RELEASED — it goes up, the direct first link of the cascade,
      (c) prey still outweighs predator after the perturbation (structural signal).

    **Re-measured 2026-08-02 (#129) and the story changed.** The old test asserted
    "sprat barely moves (|delta| <= 0.10)" and the prose promised stickleback would rise.
    With both cod stocks perturbed (the old positional edit silently missed cod_east, which
    holds the HIGHER sprat accessibility at 0.5) the measurement is the other way round:
    sprat +24.1%, stickleback -0.1%. The direct release is now the visible signal and the
    indirect stickleback pathway is not detectable.
    """
    base = _equilibrium_means(baseline_run)
    pert = _equilibrium_means(perturbed_run)

    stickleback_ratio = pert["stickleback"] / base["stickleback"]
    sprat_ratio = pert["sprat"] / base["sprat"]

    # (a) stickleback stays within an order of magnitude — does not crash or explode.
    # NOT a magnitude claim: the exact ratio is environment-specific and unpinnable.
    assert _CASCADE_STICKLEBACK_MIN_RATIO <= stickleback_ratio <= _CASCADE_STICKLEBACK_MAX_RATIO, (
        f"Stickleback perturbed/baseline = {stickleback_ratio:.3f}; expected within "
        f"[{_CASCADE_STICKLEBACK_MIN_RATIO}, {_CASCADE_STICKLEBACK_MAX_RATIO}] "
        f"(order-of-magnitude guard). base={base['stickleback']:.3e}, pert={pert['stickleback']:.3e}"
    )

    # (b) sprat is released upward — the direct cascade link. Measured 1.241; the band is
    # deliberately wide because magnitude is environment-specific (see constants above),
    # but the DIRECTION is the claim and must hold.
    assert _CASCADE_SPRAT_MIN_RATIO <= sprat_ratio <= _CASCADE_SPRAT_MAX_RATIO, (
        f"Sprat perturbed/baseline = {sprat_ratio:.3f}; expected release within "
        f"[{_CASCADE_SPRAT_MIN_RATIO}, {_CASCADE_SPRAT_MAX_RATIO}]. Sprat should INCREASE "
        f"when both cod stocks lose access to it. "
        f"base={base['sprat']:.3e}, pert={pert['sprat']:.3e}"
    )

    # (c) prey still outweighs predator in both runs — the structural, environment-robust
    # signal. Releasing sprat only widens this gap, so it must survive the perturbation.
    for label, means in (("baseline", base), ("perturbed", pert)):
        assert means["sprat"] > means["total_cod"], (
            f"Prey no longer outweighs predator in {label} run: "
            f"sprat={means['sprat']:.3e}, total_cod={means['total_cod']:.3e} "
            f"(stickleback={means['stickleback']:.3e}, context only)"
        )


# === Assertion #4: the tutorial's markdown code block parses + runs ===
def test_markdown_code_block_parses_and_runs(tmp_path: Path, numba_warmup: None) -> None:
    """Extract the first ```python fence from the tutorial markdown, ast.parse it,
    then exec it in a subprocess with a 300 s timeout. Catches semantic drift —
    e.g., a renamed import (PythonEngine -> OsmoseEngine) parses fine but fails to run.

    The subprocess is a fresh interpreter, so it pays full Numba JIT cost (the
    numba_warmup fixture only warms this process); 90 s was too tight on CI."""
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
        timeout=300,
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


# === Assertion #5: the perturbation targets both cod stocks and is a real change ===
def test_perturbation_targets_both_cod_stocks(tmp_path: Path) -> None:
    """Beat 6 drops BOTH cod stocks' sprat accessibility to 0.05.

    Guards the bug that motivated #129's retarget: the old positional replace
    ("sprat;0.4;" -> "sprat;0.05;") could only reach the FIRST predator column, so after the
    cod split it silently left cod_east untouched — and cod_east has the higher accessibility
    to sprat (0.5 vs 0.4), i.e. the perturbation was missing the larger half of the effect.
    """
    import pandas as pd  # noqa: PLC0415

    canonical = BALTIC_DIR / ACCESSIBILITY_CSV_RELPATH
    scratch = tmp_path / ACCESSIBILITY_CSV_RELPATH
    scratch.write_text(canonical.read_text())

    before = apply_cod_sprat_perturbation(scratch)
    assert before == {"cod_west": 0.4, "cod_east": 0.5}, (
        f"Canonical sprat accessibility drifted from the documented 0.4/0.5: {before}"
    )

    after = pd.read_csv(scratch, sep=";", index_col=0)
    assert after.loc["sprat", "cod_west"] == 0.05
    assert after.loc["sprat", "cod_east"] == 0.05, (
        "cod_east was not perturbed — this is exactly the silent miss #129 fixed."
    )
    # Everything else must be untouched: the edit is surgical, not a rewrite.
    orig = pd.read_csv(canonical, sep=";", index_col=0)
    other = [c for c in orig.columns if c not in ("cod_west", "cod_east")]
    pd.testing.assert_frame_equal(after[other], orig[other])


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
