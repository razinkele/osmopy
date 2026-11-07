"""Python-vs-Java cross-check: the Python engine must agree with Java within ~1 order of
magnitude per species, on the EEC and Bay of Biscay (BoB) configs. Written for the
egg-retention fix (Java implements graduated egg release): if Python DIVERGES from Java,
that fix's direction is wrong — STOP.

Opt-in via OSMOSE_JAR (a 4.3.3 or 4.4.x jar; ``osmose-java/`` holds both). Without it both
tests skip. With it, a Java failure FAILS the test: it is never converted into a skip.

Parity band: 0.1 <= py/java <= 10.0 per species (1 OoM — same as the
calibration-problem cross-engine test at tests/test_calibration_problem_python_engine.py:135).

Java gets the config staged for the jar's version, the way the Run tab stages it
(``ui.pages.run.stage_config_for_java``, version parsed from the jar's filename). The bundled
configs are native 4.4.x, and neither jar loads them raw: 4.4.1 needs the ``species.biomass.*``
resource-forcing keys the writer synthesizes, and 4.3.3 also needs
``species.biomass.nsteps.year``. Until 2026-11-07 the raw config went to Java and every
failure became a skip, so both tests skipped on both jars. The BoB test had never run: it also
read the wrong output prefix (``osm``; BoB writes ``biscay``) and let BoB's
``output.start.year ; 20`` suppress all output of a 1-year run.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from osmose.config.aliases import _numeric_version, target_version_for_jar
from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate
from osmose.results import OsmoseResults
from osmose.runner import OsmoseRunner
from scripts.cross_engine_parity_440 import inject_java_resource_nsteps_year
from ui.pages.run import stage_config_for_java

pytestmark = pytest.mark.slow

PROJECT_DIR = Path(__file__).parent.parent
EEC_CONFIG = PROJECT_DIR / "data" / "eec_full" / "eec_all-parameters.csv"
EEC_BASE_DIR = PROJECT_DIR / "data" / "eec_full"
BOB_CONFIG = PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv"
BOB_BASE_DIR = PROJECT_DIR / "data" / "examples"

_JAR = os.environ.get("OSMOSE_JAR")
_SKIP_NO_JAR = (
    "set OSMOSE_JAR to a 4.3.3 or 4.4.x OSMOSE jar (osmose-java/ has both) to run the "
    "Python-vs-Java cross-check; run it after any change to the engine's dynamics"
)

_N_YEARS = 1
_SEED = 42
_PARITY_LO = 0.1
_PARITY_HI = 10.0

# Java seeds its RNG from the clock unless told otherwise, and the config writer maps neither
# version's switch. Unseeded, BoB Hake's 1-yr Java biomass ranged 6.5k-9.2k t over six runs
# (py/java 5.2-7.3); seeded, repeat runs are bit-identical on both jars (2026-11-07).
_JAVA_FIXED_SEED_44X = {"simulation.fixed.seed.enabled": "true"}
_JAVA_FIXED_SEED_433 = {
    "movement.randomseed.fixed": "true",
    "stochastic.mortality.randomseed.fixed": "true",
}


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
    jar_path: Path, config_path: Path, work_dir: Path, n_years: int
) -> tuple[np.ndarray, list[str]]:
    """Stage the config for the jar, run Java; return (final_year_mean_biomass, species_names).

    Fails the test, with Java's own output, if Java exits non-zero or writes no biomass rows.
    OsmoseResults.biomass() is wide: [Time, <one column per species>].
    """
    raw = dict(OsmoseConfigReader().read(str(config_path)))
    target = target_version_for_jar(jar_path)
    master, overrides = stage_config_for_java(
        raw, work_dir / "stage", source_dir=config_path.parent, target_version=target
    )
    is_44x = _numeric_version(target) >= _numeric_version("4.4.0")
    if not is_44x:
        # 4.3.3's ResourceForcing requires species.biomass.nsteps.year, which nothing on the
        # writer's reverse-alias (<4.4.0) branch synthesizes; the parity harness injects it too.
        inject_java_resource_nsteps_year(master, raw)
    overrides = {
        **overrides,
        **(_JAVA_FIXED_SEED_44X if is_44x else _JAVA_FIXED_SEED_433),
        "simulation.time.nyear": str(n_years),
        # BoB starts recording at year 20, which would leave a 1-year run with no output rows.
        "output.start.year": "0",
    }
    out_dir = work_dir / "out"
    result = asyncio.run(
        OsmoseRunner(jar_path).run(master, output_dir=out_dir, overrides=overrides, quiet=True)
    )
    # Java prints its exception to STDOUT; stderr carries only the SLF4J banner.
    java_log = f"--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-1000:]}"
    if result.returncode != 0:
        pytest.fail(
            f"Java {target} exited {result.returncode} on staged {config_path.name}:\n{java_log}"
        )

    prefix = raw.get("output.file.prefix", config_path.parent.name)
    bio_df = OsmoseResults(out_dir, prefix=prefix, strict=False).biomass()
    if bio_df is None or bio_df.empty:
        pytest.fail(f"Java {target} wrote no {prefix}_biomass*.csv rows to {out_dir}:\n{java_log}")

    numeric_cols = bio_df.select_dtypes(include=["number"]).columns.tolist()
    species_cols = [c for c in numeric_cols if c != "Time"]

    # Mean biomass over the final year's rows (rows per year = ndtperyear / recordfrequency)
    rows_per_year = int(bio_df.shape[0] // n_years)
    final_bio = bio_df[species_cols].iloc[-rows_per_year:].mean().values.astype(np.float64)

    return final_bio, species_cols


def _assert_within_band(
    py_bio: np.ndarray,
    java_bio: np.ndarray,
    py_species: list[str],
    java_species: list[str],
    config_label: str,
) -> None:
    """Assert every Python species has a Java column and stays within the 0.1–10× band."""
    missing = sorted(set(py_species) - set(java_species))
    assert not missing, (
        f"Java ({config_label}) wrote no biomass column for Python species {missing}; "
        f"Java columns: {java_species}"
    )

    java_map = dict(zip(java_species, java_bio))
    py_map = dict(zip(py_species, py_bio))

    failures = []
    ratios = {}
    for sp in py_species:
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

    ratio_report = "\n".join(f"  {sp}: py/java={ratios[sp]:.3f}" for sp in py_species)
    assert not failures, (
        f"Python DIVERGES from Java ({config_label}) for {len(failures)} species "
        f"— egg-retention fix direction needs review:\n"
        + "\n".join(failures)
        + f"\nAll per-species ratios (py/java):\n{ratio_report}"
    )

    print(f"\nJava cross-check PASSED ({config_label}, {len(py_species)} species):")
    print(ratio_report)


@pytest.mark.skipif(not _JAR, reason=_SKIP_NO_JAR)
def test_python_matches_java_eec_biomass(tmp_path):
    """Python must agree with Java within the 0.1–10× band on EEC (14 species)."""
    jar_path = Path(_JAR)
    assert jar_path.exists(), f"JAR not found: {jar_path}"

    java_bio, java_species = _run_java_engine(jar_path, EEC_CONFIG, tmp_path, _N_YEARS)
    py_bio, py_species = _run_python_engine(EEC_CONFIG, EEC_BASE_DIR, _N_YEARS, _SEED)
    _assert_within_band(py_bio, java_bio, py_species, java_species, "EEC")


@pytest.mark.skipif(not _JAR, reason=_SKIP_NO_JAR)
def test_python_matches_java_bob_biomass(tmp_path):
    """Python must agree with Java within the 0.1–10× band on the Bay of Biscay (8 species)."""
    jar_path = Path(_JAR)
    assert jar_path.exists(), f"JAR not found: {jar_path}"

    java_bio, java_species = _run_java_engine(jar_path, BOB_CONFIG, tmp_path, _N_YEARS)
    py_bio, py_species = _run_python_engine(BOB_CONFIG, BOB_BASE_DIR, _N_YEARS, _SEED)
    _assert_within_band(py_bio, java_bio, py_species, java_species, "BoB")
