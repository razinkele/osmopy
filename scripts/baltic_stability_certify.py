"""SP-A certification — 50 yr × multi-seed persistence/in-envelope table for a candidate config.

Given a parameter set (from the ε-sweep JSON, or "current" for the un-recalibrated baseline), runs the
Python engine 50 yr × 5 seeds and reports, per focal species: min biomass, final-decade mean, whether
it persists (min > 0.1·ICES-lower) and stays in-envelope (lower ≤ final-decade mean ≤ upper). With
--java, a single Java 4.4.1 cross-check (via the C2 background staging) compares survivor sets.

    PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current
    PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params <sweep.json> --java

Baseline sanity: `--params current` must reproduce the known collapse (few survivors), proving the
harness actually detects instability.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine

_JAR = Path(os.environ.get(
    "OSMOSE_JAR",
    str(Path(__file__).resolve().parents[1] / "osmose-java" / "osmose-4.4.1-jar-with-dependencies.jar"),
))

# ICES envelopes (data/baltic/reference/biomass_targets.csv): (lower, upper) tonnes
ENVELOPE = {
    "cod_west": (4000, 25000), "cod_east": (60000, 85000),
    "herring": (800000, 3000000), "sprat": (800000, 2500000),
    "flounder": (20000, 100000), "perch": (8000, 50000), "pikeperch": (4000, 25000),
    "smelt": (20000, 120000), "stickleback": (50000, 500000),
}
FOCAL = list(ENVELOPE)
CERT_SEEDS = (42, 123, 7, 999, 2024)


def _load_params(source: str) -> dict[str, str]:
    """'current' -> no overrides; otherwise a sweep JSON path -> its best in-envelope point's params."""
    if source == "current":
        return {}
    data = json.loads(Path(source).read_text())
    # pick the tightest-epsilon point that recorded params (front is loose->tight)
    candidates = [p for p in data if p.get("params")]
    if not candidates:
        raise SystemExit(f"no front point with params in {source}")
    best = candidates[-1]
    return {k: str(v) for k, v in best["params"].items()}


def _species_row(bio, sp: str) -> dict:
    lo, hi = ENVELOPE[sp]
    v = np.asarray(bio[sp].values, float) if sp in bio.columns else np.array([0.0])
    late = v[-10:] if len(v) >= 10 else v
    vmin, late_mean = float(v.min()), float(np.mean(late))
    persists = vmin > 0.1 * lo
    in_env = lo <= late_mean <= hi
    return {"min": vmin, "late_mean": late_mean, "persists": persists, "in_envelope": in_env}


def certify_python(params: dict[str, str], n_years: int, seeds) -> dict:
    """Run Python n_years x seeds; aggregate per-species persist/in-envelope (worst-case across seeds)."""
    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    cfg["simulation.time.nyear"] = str(n_years)
    cfg.update(params)
    per_seed = []
    for seed in seeds:
        bio = PythonEngine().run_in_memory(cfg, seed=seed).biomass()
        per_seed.append({sp: _species_row(bio, sp) for sp in FOCAL})
    # worst-case: a species passes only if it persists AND is in-envelope in EVERY seed
    table = {}
    for sp in FOCAL:
        rows = [ps[sp] for ps in per_seed]
        table[sp] = {
            "persists": all(r["persists"] for r in rows),
            "in_envelope": all(r["in_envelope"] for r in rows),
            "min_biomass": min(r["min"] for r in rows),
            "late_mean_range": [min(r["late_mean"] for r in rows), max(r["late_mean"] for r in rows)],
        }
    return table


def certify_java(params: dict[str, str], n_years: int, seed: int = 42) -> dict | None:
    """Single Java 4.4.1 run (staged via the C2 background recipe) -> per-species table, or None if
    the jar is missing / the run fails. A coarse cross-engine consistency check (Baltic is not
    bit-equal cross-engine), so a single seed is used."""
    if not _JAR.exists():
        print(f"(Java cross-check skipped: jar not found at {_JAR})")
        return None
    from osmose.java_background_staging import stage_background_for_java
    from osmose.java_config_reconcile import reconcile_config_for_java
    from ui.pages.run import write_temp_config

    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    cfg.update(params)  # bake the recalibrated params into the config
    stage = tmp / "stage"
    write_temp_config(cfg, stage, source_dir=res["config_file"].parent, target_version="4.4.1")
    master = stage / "osm_all-parameters.csv"
    overrides = stage_background_for_java(stage, cfg)  # incl. output.cutoff.enabled=false
    # Java 4.4.1 strips '_'/'-' from species.name but not from name-based references; the
    # disaggregated config (cod_west/cod_east) also left the discards matrix stale. Reconcile the
    # STAGED copy so Java can resolve names and the fishery matrices are consistent (no-op on a
    # clean aggregate config). See osmose/java_config_reconcile.py.
    reconcile_config_for_java(stage)
    odir = tmp / "out"
    odir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java", "-Xmx2g", "-jar", str(_JAR), str(master),
        f"-Poutput.dir.path={odir}",
        f"-Psimulation.time.nyear={n_years}",
        "-Poutput.start.year=0",
        *[f"-P{k}={v}" for k, v in overrides.items()],
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"(Java run failed, exit {r.returncode}):")
        print("\n".join((r.stdout or "").splitlines()[-8:]))
        return None
    # Read the 4.4.1 Java biomass CSV directly (comma-separated, first line a "Mean..." comment,
    # species NAMES as columns). OsmoseResults does not map the Java biomass to focal-named columns.
    import csv as csv_mod

    series: dict[str, list[float]] = {}
    for f in sorted(odir.rglob("*biomass_Simu*.csv")):
        rows = [ln for ln in f.read_text().splitlines() if not ln.lstrip().startswith('"Mean')]
        if len(rows) < 2:
            continue
        header = next(csv_mod.reader([rows[0]]))
        for ln in rows[1:]:
            cells = next(csv_mod.reader([ln]))
            for ci, col in enumerate(header[1:], start=1):
                sp = col.strip().strip('"')
                if sp and ci < len(cells):
                    try:
                        series.setdefault(sp, []).append(float(cells[ci].strip().strip('"')))
                    except ValueError:
                        pass
    if not series:
        print("(Java biomass output not found or unparseable)")
        return None

    table = {}
    for sp in FOCAL:
        lo, hi = ENVELOPE[sp]
        v = np.asarray(series.get(sp, [0.0]), float)
        late = v[-10:] if len(v) >= 10 else v
        vmin, late_mean = float(v.min()), float(np.mean(late))
        table[sp] = {
            "persists": vmin > 0.1 * lo,
            "in_envelope": lo <= late_mean <= hi,
            "min_biomass": vmin,
            "late_mean_range": [late_mean, late_mean],
        }
    return table


def _print_table(engine: str, table: dict) -> int:
    print(f"\n=== {engine}: per-species certification ===")
    ok = 0
    for sp in FOCAL:
        t = table[sp]
        good = t["persists"] and t["in_envelope"]
        ok += good
        flag = "PASS" if good else ("persists" if t["persists"] else "COLLAPSE")
        print(f"  {sp:12s} {flag:9s} min={t['min_biomass']:.2e} late_mean={t['late_mean_range']}")
    print(f"  --> {ok}/8 persistent & in-envelope")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default="current", help="'current' or a stability_sweep.json path")
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CERT_SEEDS))
    ap.add_argument("--java", action="store_true", help="also run a single Java 4.4.1 cross-check")
    ap.add_argument("--out", default="docs/baltic_stability_certification_2026-07-01.md")
    args = ap.parse_args()

    params = _load_params(args.params)
    py_table = certify_python(params, args.years, tuple(args.seeds))
    py_ok = _print_table("Python", py_table)

    lines = [
        "# Baltic stability — SP-A certification\n",
        f"**Params:** {args.params}  ·  **horizon:** {args.years} yr  ·  **seeds:** {args.seeds}\n",
        "| species | persists | in-envelope | min biomass | final-decade mean range |",
        "|---|---|---|---|---|",
    ]
    for sp in FOCAL:
        t = py_table[sp]
        lines.append(
            f"| {sp} | {'✓' if t['persists'] else '✗'} | {'✓' if t['in_envelope'] else '✗'} "
            f"| {t['min_biomass']:.2e} | {t['late_mean_range']} |"
        )
    n_focal = len(FOCAL)
    verdict = (
        f"\n**Python verdict: {py_ok}/{n_focal} persistent & in-envelope.** "
        + (f"All {n_focal} pass — candidate is certifiable; verify value round-trip before writing data/baltic."
           if py_ok == n_focal
           else f"Not {n_focal}/{n_focal} — SP-B gate: the failing species (not PASS above) are candidates params alone "
                "cannot stabilise; record whether sweeping their params moved them (structural vs tunable).")
    )
    lines.append(verdict)

    if args.java:
        print("\n=== Java 4.4.1 cross-check (single seed, staged via C2) ===")
        j_table = certify_java(params, args.years, seed=args.seeds[0])
        if j_table:
            j_ok = _print_table("Java 4.4.1", j_table)
            py_surv = {sp for sp in FOCAL if py_table[sp]["persists"]}
            j_surv = {sp for sp in FOCAL if j_table[sp]["persists"]}
            agree = py_surv == j_surv
            lines.append(
                f"\n**Java cross-check: {j_ok}/8 persistent (single seed).** Survivor sets "
                f"{'AGREE' if agree else 'DIFFER'} with Python — Python {sorted(py_surv)}, "
                f"Java {sorted(j_surv)}. Coarse consistency check only (Baltic is not bit-equal "
                "cross-engine); a DIFFER is a flag to inspect, not an automatic failure."
            )
        else:
            lines.append("\n_Java cross-check unavailable (jar missing or run failed) — see console._")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote certification note to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
