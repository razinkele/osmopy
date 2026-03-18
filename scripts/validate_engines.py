#!/usr/bin/env python3
"""Validate Python OSMOSE engine against Java engine.

Runs both engines on the Bay of Biscay example configuration and
produces a comparison report.

Usage:
    .venv/bin/python scripts/validate_engines.py [--years N] [--seed S]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
JAR_PATH = PROJECT_DIR / "osmose-java" / "osmose_4.3.3-jar-with-dependencies.jar"
EXAMPLES_CONFIG = PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv"


def run_java(output_dir: Path, n_years: int) -> float:
    """Run Java OSMOSE engine. Returns elapsed seconds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-Xmx2g",
        "-jar",
        str(JAR_PATH),
        str(EXAMPLES_CONFIG),
        f"-Poutput.dir.path={output_dir}",
        f"-Psimulation.time.nyear={n_years}",
        "-Poutput.start.year=0",
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"  Java FAILED: {result.stderr[-500:]}")
        sys.exit(1)
    return elapsed


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


def compare_biomass(java_dir: Path, python_dir: Path) -> pd.DataFrame:
    """Compare biomass time series between engines."""
    java_file = list(java_dir.glob("*biomass*Simu0.csv"))
    python_file = python_dir / "osmose_biomass_Simu0.csv"

    if not java_file or not python_file.exists():
        print("  Missing biomass files!")
        return pd.DataFrame()

    java_bio = pd.read_csv(java_file[0], skiprows=1)
    python_bio = pd.read_csv(python_file, skiprows=1)

    species = [c for c in java_bio.columns if c != "Time"]
    results = []

    for sp in species:
        j_final = java_bio[sp].iloc[-1] if sp in java_bio.columns else 0
        p_final = python_bio[sp].iloc[-1] if sp in python_bio.columns else 0

        if j_final > 0 and p_final > 0:
            log_ratio = np.log10(p_final / j_final)
        elif j_final == 0 and p_final == 0:
            log_ratio = 0.0
        else:
            log_ratio = float("inf")

        results.append(
            {
                "Species": sp,
                "Java_biomass": j_final,
                "Python_biomass": p_final,
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
    parser = argparse.ArgumentParser(description="Validate Python vs Java OSMOSE engine")
    parser.add_argument("--years", type=int, default=5, help="Simulation years (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Python RNG seed (default: 42)")
    args = parser.parse_args()

    print("=" * 70)
    print("OSMOSE Engine Validation: Python vs Java")
    print("=" * 70)
    print("Config: Bay of Biscay (8 species + 6 resources)")
    print(f"Years:  {args.years}")
    print(f"Seed:   {args.seed}")
    print()

    # Check prerequisites
    if not JAR_PATH.exists():
        print(f"ERROR: Java JAR not found at {JAR_PATH}")
        sys.exit(1)
    if not EXAMPLES_CONFIG.exists():
        print(f"ERROR: Config not found at {EXAMPLES_CONFIG}")
        sys.exit(1)

    base = Path("/tmp/osmose_validation_run")
    java_dir = base / "java"
    python_dir = base / "python"

    # Run Java
    print("[1/4] Running Java engine...")
    java_time = run_java(java_dir, args.years)
    print(f"  Done in {java_time:.1f}s ({java_time / args.years:.2f}s/year)")

    # Run Python
    print("[2/4] Running Python engine...")
    python_time = run_python(python_dir, args.years, args.seed)
    print(f"  Done in {python_time:.1f}s ({python_time / args.years:.2f}s/year)")

    # Compare biomass
    print("[3/4] Comparing biomass...")
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
