"""Python-vs-Java cross-check for the egg-retention fix: the FIXED Python engine
must agree with Java within ~1 order of magnitude per species (Java implements
graduated egg release). If Python DIVERGES from Java, the fix's direction is
wrong — STOP. Opt-in via OSMOSE_JAR; excluded from the default suite.

Parity band: 0.1 <= py/java <= 10.0 per species (1 OoM — same as the
calibration-problem cross-engine test at tests/test_calibration_problem_python_engine.py:135).

Fallback: if the Java run fails to load the EEC config (known 4.3.3 compat issue),
fall back to the BoB (Bay of Biscay) config which is reliably Java-runnable and
is also affected by the egg-retention fix.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate
from osmose.results import OsmoseResults
from osmose.runner import OsmoseRunner

pytestmark = pytest.mark.slow

PROJECT_DIR = Path(__file__).parent.parent
EEC_CONFIG = PROJECT_DIR / "data" / "eec_full" / "eec_all-parameters.csv"
EEC_BASE_DIR = PROJECT_DIR / "data" / "eec_full"
BOB_CONFIG = PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv"
BOB_BASE_DIR = PROJECT_DIR / "data" / "examples"

_JAR = os.environ.get("OSMOSE_JAR")

_N_YEARS = 1
_SEED = 42
_PARITY_LO = 0.1
_PARITY_HI = 10.0


def _run_python_engine(
    config_path: Path, base_dir: Path, n_years: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    """Run the Python engine; return (final_year_mean_biomass, species_names)."""
    reader = OsmoseConfigReader()
    raw = reader.read(config_path)
    raw["simulation.time.nyear"] = str(n_years)

    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            base_dir / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw.get("grid.nline", "1"))
        nx = int(raw.get("grid.ncolumn", "1"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)
    outputs = simulate(cfg, grid, rng)

    # Mean biomass over the final year's steps
    n_steps = len(outputs)
    steps_per_year = n_steps // n_years
    final_steps = outputs[n_steps - steps_per_year :]
    final_bio = np.array([o.biomass for o in final_steps])
    mean_bio = final_bio.mean(axis=0)

    return mean_bio, cfg.species_names


def _run_java_engine(
    jar_path: Path,
    config_path: Path,
    output_dir: Path,
    prefix: str,
    n_years: int,
) -> tuple[np.ndarray, list[str]] | None:
    """Run the Java engine; return (final_year_mean_biomass, species_names) or None on failure.

    OsmoseResults.biomass() returns a wide-format DataFrame:
    columns = [Time, sp0_name, sp1_name, ..., species='all']
    where species columns are the actual species names.
    """
    runner = OsmoseRunner(jar_path)

    result = asyncio.run(
        runner.run(
            config_path,
            output_dir=output_dir,
            overrides={"simulation.time.nyear": str(n_years)},
            quiet=True,
        )
    )

    # Detect Java config-load failure (stack trace → non-zero exit or error in output)
    if result.returncode != 0:
        return None

    combined = result.stdout + result.stderr
    load_failure_indicators = ["Exception in thread", "ERROR", "not found"]
    if any(ind in combined for ind in load_failure_indicators):
        return None

    try:
        results = OsmoseResults(output_dir, prefix=prefix, strict=False)
        bio_df = results.biomass()
    except FileNotFoundError:
        return None

    if bio_df is None or bio_df.empty:
        return None

    # bio_df is wide: [Time, sp0, sp1, ..., 'species'='all']
    # Extract numeric species columns (drop Time, species, and any other non-numeric)
    numeric_cols = bio_df.select_dtypes(include=["number"]).columns.tolist()
    species_cols = [c for c in numeric_cols if c != "Time"]

    if not species_cols:
        return None

    # Mean biomass over the final n_steps_per_year rows (= ndt time steps)
    ndt = int(bio_df.shape[0] // n_years)
    final_bio = bio_df[species_cols].iloc[-ndt:].mean().values.astype(np.float64)

    return final_bio, species_cols


def _assert_within_band(
    py_bio: np.ndarray,
    java_bio: np.ndarray,
    py_species: list[str],
    java_species: list[str],
    config_label: str,
) -> None:
    """Assert all common species stay within the 0.1–10× parity band."""
    java_set = set(java_species)
    py_set = set(py_species)
    common = sorted(java_set & py_set)
    assert common, f"No species in common between Java ({java_species}) and Python ({py_species})"

    java_map = dict(zip(java_species, java_bio))
    py_map = dict(zip(py_species, py_bio))

    failures = []
    ratios = {}
    for sp in common:
        j = java_map[sp]
        p = py_map[sp]
        if j <= 0 and p <= 0:
            ratios[sp] = 1.0  # both extinct — trivially in-band
            continue
        if j <= 0:
            ratios[sp] = float("inf")
            failures.append(f"  {sp}: Java=0, Python={p:.4g} (Python non-zero, Java extinct)")
            continue
        ratio = p / j
        ratios[sp] = ratio
        if not (_PARITY_LO <= ratio <= _PARITY_HI):
            failures.append(
                f"  {sp}: Python={p:.4g}, Java={j:.4g}, ratio={ratio:.3f} "
                f"(outside [{_PARITY_LO}, {_PARITY_HI}])"
            )

    ratio_report = "\n".join(f"  {sp}: py/java={ratios[sp]:.3f}" for sp in common)
    assert not failures, (
        f"Python DIVERGES from Java ({config_label}) for {len(failures)} species "
        f"— egg-retention fix direction needs review:\n"
        + "\n".join(failures)
        + f"\nAll per-species ratios (py/java):\n{ratio_report}"
    )

    print(f"\nJava cross-check PASSED ({config_label}, {len(common)} species):")
    print(ratio_report)


@pytest.mark.skipif(not _JAR, reason="set OSMOSE_JAR to the 4.3.3 jar to run")
def test_python_matches_java_eec_biomass():
    """Fixed Python engine must agree with Java within the 0.1–10× parity band.

    Tries EEC first (14 species, spatially resolved). If the 4.3.3 jar
    can't load the EEC config (known compat issue), the test is skipped
    — run test_python_matches_java_bob_biomass instead.
    """
    jar_path = Path(_JAR)
    assert jar_path.exists(), f"JAR not found: {jar_path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        java_result = _run_java_engine(
            jar_path,
            EEC_CONFIG,
            Path(tmpdir),
            prefix="eec",
            n_years=_N_YEARS,
        )

    if java_result is None:
        pytest.skip(
            "Java 4.3.3 could not load EEC config (known compat issue); "
            "use test_python_matches_java_bob_biomass for the cross-check, "
            "or re-run with a Java version that supports EEC."
        )
        return

    java_bio, java_species = java_result
    py_bio, py_species = _run_python_engine(EEC_CONFIG, EEC_BASE_DIR, _N_YEARS, _SEED)
    _assert_within_band(py_bio, java_bio, py_species, java_species, "EEC")


@pytest.mark.skipif(not _JAR, reason="set OSMOSE_JAR to the 4.3.3 jar to run")
def test_python_matches_java_bob_biomass():
    """Fallback cross-check on BoB (reliably Java-runnable; also affected by the fix)."""
    jar_path = Path(_JAR)
    assert jar_path.exists(), f"JAR not found: {jar_path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        java_result = _run_java_engine(
            jar_path,
            BOB_CONFIG,
            Path(tmpdir),
            prefix="osm",
            n_years=_N_YEARS,
        )

    if java_result is None:
        pytest.skip("Java could not load BoB config — cross-check blocked")
        return

    java_bio, java_species = java_result
    py_bio, py_species = _run_python_engine(BOB_CONFIG, BOB_BASE_DIR, _N_YEARS, _SEED)
    _assert_within_band(py_bio, java_bio, py_species, java_species, "BoB")
