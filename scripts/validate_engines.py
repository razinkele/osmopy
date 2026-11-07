#!/usr/bin/env python3
"""Validate Python OSMOSE engine against Java engine.

Runs both engines on the Bay of Biscay example configuration and
produces a comparison report. Java gets the config staged for the jar's
version, as the Run tab stages it; works with the 4.4.1 and 4.3.3 jars.

Usage:
    .venv/bin/python scripts/validate_engines.py [--years N] [--seed S]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
JAR_DIR = PROJECT_DIR / "osmose-java"
EXAMPLES_CONFIG = PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv"


def resolve_jar(explicit: str | None = None) -> Path | None:
    """Locate the OSMOSE JAR: --jar, then $OSMOSE_JAR, then any jar in osmose-java/.

    This used to be a single hard-coded filename
    (``osmose_4.3.3-jar-with-dependencies.jar``, with an underscore), which made
    the script unrunnable for anyone whose JAR was named anything else — and the
    names genuinely differ across this repo: apptainer/osmose.def downloads
    ``osmose-4.3.3-...`` with a HYPHEN, while the 4.4.1 references in
    baltic_440_smoke.py / baltic_stability_certify.py / ui/state.py use a hyphen
    too. The JAR is gitignored, so nothing in-tree pins the true spelling. Glob
    instead of guessing, and honour $OSMOSE_JAR like osmose/cli.py and
    baltic_stability_certify.py already do.
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("OSMOSE_JAR")
    if env:
        return Path(env)
    found = sorted(JAR_DIR.glob("*.jar"))
    return found[0] if found else None


def run_java(
    jar_path: Path, output_dir: Path, n_years: int, config_path: Path = EXAMPLES_CONFIG
) -> float:
    """Run Java OSMOSE on *config_path*, staged for the jar's version. Returns elapsed seconds.

    Neither jar loads the bundled native-4.4.x config raw: 4.4.1 needs the
    ``species.biomass.*`` resource keys the writer synthesizes, 4.3.3 needs
    ``species.biomass.nsteps.year``. So the config goes through ``stage_config_for_java``, the
    Run tab's staging, into a ``<output_dir>_stage`` sibling directory.
    """
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))  # ui/ is not an installed package
    from osmose.config.aliases import target_version_for_jar
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import stage_config_for_java

    output_dir.mkdir(parents=True, exist_ok=True)
    raw = dict(OsmoseConfigReader().read(config_path))
    master, stage_overrides = stage_config_for_java(
        raw,
        output_dir.with_name(f"{output_dir.name}_stage"),
        config_path.parent,
        target_version=target_version_for_jar(jar_path),
    )
    cmd = [
        "java",
        "-Xmx2g",
        "-jar",
        str(jar_path),
        str(master),
        f"-Poutput.dir.path={output_dir}",
        f"-Psimulation.time.nyear={n_years}",
        "-Poutput.start.year=0",
        *(f"-P{key}={value}" for key, value in stage_overrides.items()),
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(_java_failure_report(result))
        sys.exit(1)
    return elapsed


# Lines that carry no diagnostic value but crowd out the ones that do. The JVM
# prints JAVA_TOOL_OPTIONS to stderr on every start, and it is long enough on a
# proxied container to fill a 500-char tail by itself.
_NOISE_PREFIXES = ("Picked up JAVA_TOOL_OPTIONS", "SLF4J:")


def _java_failure_report(result: subprocess.CompletedProcess[str], tail: int = 25) -> str:
    """Explain a failed Java run, reading BOTH streams.

    This used to be ``result.stderr[-500:]``, which reliably hid the actual cause:
    OSMOSE logs its own fatal errors (``osmose[severe] ...`` plus the stack trace)
    to **stdout**, so the stderr tail showed only the JVM's JAVA_TOOL_OPTIONS echo
    and SLF4J's binder warning while the real message — e.g. "NETCDF_BIOMASS
    resource forcing is used but parameters are missing" — was never printed.
    Measured 2026-09-27 against a 4.4.1 jar.
    """
    lines: list[str] = []
    for stream, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        kept = [
            ln
            for ln in (text or "").splitlines()
            if ln.strip() and not ln.startswith(_NOISE_PREFIXES)
        ]
        if kept:
            lines.append(f"  --- {stream} (last {min(tail, len(kept))} of {len(kept)} lines) ---")
            lines.extend(f"  {ln}" for ln in kept[-tail:])

    # Surface the engine's own fatal lines first: they may sit above the tail when a
    # long stack trace follows, and they are the sentence worth reading.
    severe = [
        ln
        for text in (result.stdout, result.stderr)
        for ln in (text or "").splitlines()
        if "osmose[severe]" in ln
    ]
    header = [f"  Java FAILED (exit {result.returncode})."]
    if severe:
        header.append("  Engine reported:")
        header.extend(f"    {ln.strip()}" for ln in severe[:5])
    return "\n".join(header + lines)


def run_python(output_dir: Path, n_years: int, seed: int) -> float:
    """Run Python OSMOSE engine. Returns elapsed seconds."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    reader = OsmoseConfigReader()
    config = reader.read(EXAMPLES_CONFIG)
    config["simulation.time.nyear"] = str(n_years)

    engine = PythonEngine()
    start = time.time()
    engine.run(config=config, output_dir=output_dir, seed=seed)
    elapsed = time.time() - start
    return elapsed


def _biomass_file(output_dir: Path) -> Path | None:
    """The single ``<prefix>_biomass_Simu0.csv`` in *output_dir*, or None (with a message).

    Each engine names it by its own prefix: Java by ``output.file.prefix`` (``biscay`` for the
    bundled config), the Python engine by its writer's default (``osm``). A hard-coded
    ``osmose_biomass_Simu0.csv`` matched neither, so no comparison was ever produced.
    """
    found = sorted(output_dir.glob("*_biomass_Simu0.csv"))
    if len(found) != 1:
        print(
            f"  Expected one *_biomass_Simu0.csv in {output_dir}, found {[p.name for p in found]}"
        )
        return None
    return found[0]


def _final_year_mean(bio: pd.DataFrame, species: str) -> float:
    """Mean of *species* over the final simulated year, or 0.0 if the column is absent.

    Rows are picked by ``Time``, so any record frequency works, and so do both stamping
    conventions: Java labels a step by its end (0.0417 ... 5.0), Python by its start
    (0.0 ... 4.9583), and ``Time > last - 1`` selects the same final-year rows from each.
    """
    if species not in bio.columns:
        return 0.0
    time = bio["Time"]
    return float(bio.loc[time > time.max() - 1 + 1e-9, species].mean())


def compare_biomass(java_dir: Path, python_dir: Path) -> pd.DataFrame:
    """Compare the engines' final-year mean biomass, per species.

    Not the last step: a species can be absent at a single step in both engines while they
    disagree all year. Bay of Biscay Hake on a 5-yr run is 0 vs 0 at the final step, which
    this used to score as agreement, but 247 t (Java) vs 41,540 t (Python) averaged over year
    5 (2026-11-07).
    """
    java_file = _biomass_file(java_dir)
    python_file = _biomass_file(python_dir)

    if java_file is None or python_file is None:
        print("  Missing biomass files!")
        return pd.DataFrame()

    java_bio = pd.read_csv(java_file, skiprows=1)
    python_bio = pd.read_csv(python_file, skiprows=1)

    species = [c for c in java_bio.columns if c != "Time"]
    results = []

    for sp in species:
        j_mean = _final_year_mean(java_bio, sp)
        p_mean = _final_year_mean(python_bio, sp)

        if j_mean > 0 and p_mean > 0:
            log_ratio = np.log10(p_mean / j_mean)
        elif j_mean == 0 and p_mean == 0:
            log_ratio = 0.0  # absent from both engines for the whole final year
        else:
            log_ratio = float("inf")

        results.append(
            {
                "Species": sp,
                "Java_biomass": j_mean,
                "Python_biomass": p_mean,
                "Log10_ratio": log_ratio,
                "Within_1_OoM": abs(log_ratio) <= 1.0,
            }
        )

    return pd.DataFrame(results)


def compare_mortality(java_dir: Path, python_dir: Path, species: list[str]) -> None:
    """Compare mortality outputs."""
    java_mort_dir = java_dir / "Mortality"
    python_mort_dir = python_dir / "Mortality"

    if not java_mort_dir.exists() or not python_mort_dir.exists():
        print("  Mortality directories not found — skipping")
        return

    for sp in species[:3]:  # Sample first 3 species
        java_files = list(java_mort_dir.glob(f"*{sp}*"))
        python_files = list(python_mort_dir.glob(f"*{sp}*"))
        if java_files and python_files:
            print(f"  {sp}: Java has {len(java_files)} file(s), Python has {len(python_files)}")


def main() -> None:
    """Run both engines on the bundled config and print a per-species comparison.

    Prerequisites are checked before anything runs, and the resolved jar path is
    printed, because ``resolve_jar`` globs ``osmose-java/`` and may pick a different
    version than the caller assumed.

    This is a single-run comparison of final-year mean biomass. For a statistical verdict (seed
    ensembles, equivalence tests) use ``scripts/cross_engine_parity_440.py``; see
    docs/java_cross_check_2026-09-27.md.
    """
    parser = argparse.ArgumentParser(description="Validate Python vs Java OSMOSE engine")
    parser.add_argument("--years", type=int, default=5, help="Simulation years (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Python RNG seed (default: 42)")
    parser.add_argument(
        "--jar",
        default=None,
        help="Path to the OSMOSE JAR (default: $OSMOSE_JAR, else osmose-java/*.jar)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("OSMOSE Engine Validation: Python vs Java")
    print("=" * 70)
    print("Config: Bay of Biscay (8 species + 6 resources)")
    print(f"Years:  {args.years}")
    print(f"Seed:   {args.seed}")
    print()

    # Check prerequisites
    jar_path = resolve_jar(args.jar)
    if jar_path is None or not jar_path.exists():
        where = str(jar_path) if jar_path else f"no *.jar in {JAR_DIR}"
        print(f"ERROR: Java JAR not found ({where}).")
        print("  Pass --jar PATH, set $OSMOSE_JAR, or drop the jar into osmose-java/.")
        print("  The jar is gitignored; see apptainer/osmose.def for the upstream release URL.")
        sys.exit(1)
    print(f"JAR:    {jar_path}")
    if not EXAMPLES_CONFIG.exists():
        print(f"ERROR: Config not found at {EXAMPLES_CONFIG}")
        sys.exit(1)

    base = Path("/tmp/osmose_validation_run")
    java_dir = base / "java"
    python_dir = base / "python"

    # Run Java
    print("[1/4] Running Java engine...")
    java_time = run_java(jar_path, java_dir, args.years)
    print(f"  Done in {java_time:.1f}s ({java_time / args.years:.2f}s/year)")

    # Run Python
    print("[2/4] Running Python engine...")
    python_time = run_python(python_dir, args.years, args.seed)
    print(f"  Done in {python_time:.1f}s ({python_time / args.years:.2f}s/year)")

    # Compare biomass
    print("[3/4] Comparing final-year mean biomass...")
    df = compare_biomass(java_dir, python_dir)
    if df.empty:
        print("  No comparison data available")
        sys.exit(1)

    print()
    print(f"{'Species':<20} {'Java (t)':>15} {'Python (t)':>15} {'Log10':>8} {'Parity':>8}")
    print("-" * 68)
    for _, row in df.iterrows():
        parity = "YES" if row["Within_1_OoM"] else "no"
        lr = f"{row['Log10_ratio']:.1f}" if np.isfinite(row["Log10_ratio"]) else "inf"
        print(
            f"{row['Species']:<20} {row['Java_biomass']:>15.0f} "
            f"{row['Python_biomass']:>15.0f} {lr:>8} {parity:>8}"
        )

    n_parity = df["Within_1_OoM"].sum()
    n_total = len(df)
    print("-" * 68)
    print(f"Species within 1 order of magnitude: {n_parity}/{n_total}")

    # Compare mortality
    print()
    print("[4/4] Checking mortality outputs...")
    species = df["Species"].tolist()
    compare_mortality(java_dir, python_dir, species)

    # Performance
    print()
    print("=== Performance ===")
    print(f"  Java:   {java_time:.1f}s total, {java_time / args.years:.2f}s/year")
    print(f"  Python: {python_time:.1f}s total, {python_time / args.years:.2f}s/year")
    speedup = java_time / python_time if python_time > 0 else float("inf")
    print(f"  Python is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than Java")

    # Verdict
    print()
    print("=" * 70)
    if n_parity >= 6:
        print(f"VERDICT: PASS — {n_parity}/{n_total} species within 1 OoM")
    elif n_parity >= 3:
        print(f"VERDICT: PARTIAL — {n_parity}/{n_total} species within 1 OoM")
    else:
        print(f"VERDICT: FAIL — only {n_parity}/{n_total} species within 1 OoM")
    print("=" * 70)


if __name__ == "__main__":
    main()
