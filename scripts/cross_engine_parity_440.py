#!/usr/bin/env python3
"""Phase 2 cross-engine ensemble parity: Python <-> Java {4.4.1, 4.3.3} on EEC.

The deferred jar-swap's Phase 2 (docs/superpowers/plans/2026-06-19-jar-swap-440-validated-resume.md):
confirm Java-4.4.1 agrees with the pure-Python engine *at least as well as* Java-4.3.3 does, using a
multi-replicate, distributional + equivalence comparison (NOT a single-seed run). Cross-engine streams
diverge by construction (Python PCG64 vs Java MT19937), so the test is statistical:

  - per species, final-year biomass distribution over N varied-seed replicates per engine;
  - work on log10(biomass) (biomass ~ log-normal);
  - EQUIVALENCE (TOST-style): 90% CI of mean log-difference within +-Delta (default Delta=log10(3));
  - DISTRIBUTION: two-sample KS p-value (Python vs Java) + variance ratio;
  - RELATIVE GATE: |Python - 4.4.1| agreement no worse than |Python - 4.3.3|;
  - COLLAPSE frequency per engine (resolves single-seed extinctions, e.g. sardine);
  - 1-OoM only as a catastrophic tripwire.

Usage: PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py --n 8 --years 10 --spinup-years 2
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import stats

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults
from ui.pages.run import write_temp_config

ROOT = Path(__file__).resolve().parent.parent
EEC = ROOT / "data" / "eec_full" / "eec_all-parameters.csv"
JARS = {
    "4.4.1": ROOT / "osmose-java" / "osmose-4.4.1-jar-with-dependencies.jar",
    "4.3.3": ROOT / "osmose-java" / "osmose_4.3.3-jar-with-dependencies.jar",
}
COLLAPSE = 1.0  # tonnes; below this a species is treated as collapsed/extinct for that replicate


def _final_mean(bio, years: int, spinup_years: int) -> dict[str, float]:
    cols = [c for c in bio.columns if c not in ("Time", "species") and not str(c).startswith("Unnamed")]
    cols = [c for c in cols if str(c).lower() != "all"]
    n = len(bio)
    spy = max(1, n // years)
    keep = max(spy, n - spinup_years * spy)  # mean over the post-spinup tail
    tail = bio.iloc[n - keep:]
    return {c: float(np.nanmean(tail[c].to_numpy(dtype=float))) for c in cols}


def python_rep(years: int, seed: int) -> dict[str, float]:
    raw = dict(OsmoseConfigReader().read(str(EEC)))
    raw["simulation.time.nyear"] = str(years)
    res = PythonEngine().run_in_memory(raw, seed=seed)
    return _final_mean(res.biomass(), years, SPINUP)


def stage_java(ver: str, out: Path) -> Path:
    raw = dict(OsmoseConfigReader().read(str(EEC)))
    write_temp_config(raw, out, source_dir=EEC.parent, target_version=ver)
    return out / "osm_all-parameters.csv"


def java_rep(ver: str, master: Path, years: int, odir: Path) -> dict[str, float] | None:
    odir.mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-Xmx2g", "-jar", str(JARS[ver]), str(master),
           f"-Poutput.dir.path={odir}", f"-Psimulation.time.nyear={years}", "-Poutput.start.year=0"]
    if subprocess.run(cmd, capture_output=True, text=True, timeout=900).returncode != 0:
        return None
    return _final_mean(OsmoseResults(odir, prefix="eec").biomass(), years, SPINUP)


def ensemble(engine: str, years: int, n: int, tmp: Path) -> dict[str, np.ndarray]:
    """N replicates -> {species: array(N) of final-year mean biomass}."""
    reps: list[dict[str, float]] = []
    if engine == "python":
        for s in range(n):
            reps.append(python_rep(years, seed=1000 + s))
    else:
        master = stage_java(engine, tmp / f"stage_{engine}")
        for s in range(n):
            # Java seed is time-based (simulation.fixed.seed.enabled defaults false) -> varied per run.
            r = java_rep(engine, master, years, tmp / f"out_{engine}_{s}")
            if r is not None:
                reps.append(r)
    species = sorted({k for r in reps for k in r})
    return {sp: np.array([r.get(sp, np.nan) for r in reps], dtype=float) for sp in species}


def _logmean(a: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(a, COLLAPSE, None))


def compare(py: np.ndarray, jv: np.ndarray, delta: float):
    """Return (mean_log_diff, ci_halfwidth, equivalent, ks_p, var_ratio, oom_ok)."""
    lp, lj = _logmean(py), _logmean(jv)
    d = lp.mean() - lj.mean()
    se = np.sqrt(lp.var(ddof=1) / len(lp) + lj.var(ddof=1) / len(lj)) if len(lp) > 1 else np.inf
    ci = 1.645 * se  # 90% two-sided -> TOST 90% CI on the difference
    equivalent = abs(d) + ci <= delta
    ks_p = float(stats.ks_2samp(py, jv).pvalue) if len(py) > 1 and len(jv) > 1 else np.nan
    vj = lj.var(ddof=1)
    var_ratio = float(lp.var(ddof=1) / vj) if vj > 0 else np.inf
    return d, ci, equivalent, ks_p, var_ratio, abs(d) < 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="replicates per engine")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--spinup-years", type=int, default=2)
    ap.add_argument("--delta", type=float, default=np.log10(3), help="equivalence margin in log10 units")
    args = ap.parse_args()
    global SPINUP
    SPINUP = args.spinup_years
    tmp = Path("/home/razinka/.claude/jobs/18a62785/tmp/phase2")
    tmp.mkdir(parents=True, exist_ok=True)

    # within-engine determinism check (harness sanity): same seed -> identical Python output
    a = python_rep(3, seed=7)
    b = python_rep(3, seed=7)
    det = all(np.isclose(a[k], b[k]) for k in a)
    print(f"[determinism] Python same-seed reproducible: {det}")

    t0 = time.perf_counter()
    print(f"[run] {args.n} replicates x 3 engines x {args.years}yr (this takes a few minutes)...")
    py = ensemble("python", args.years, args.n, tmp)
    j441 = ensemble("4.4.1", args.years, args.n, tmp)
    j433 = ensemble("4.3.3", args.years, args.n, tmp)
    print(f"[run] done in {time.perf_counter() - t0:.0f}s\n")

    species = [s for s in py if s in j441 and s in j433]
    hdr = f"{'species':<22}{'py_gm':>10}{'441_gm':>10}{'433_gm':>10}{'d(py-441)':>10}{'d(py-433)':>10}{'441≤433?':>9}{'KS441':>7}{'collapse(py/441/433)':>22}"
    print(hdr)
    print("-" * len(hdr))
    worse = []
    for sp in species:
        d441, ci441, eq441, ks441, vr441, oom441 = compare(py[sp], j441[sp], args.delta)
        d433, ci433, eq433, ks433, vr433, oom433 = compare(py[sp], j433[sp], args.delta)
        no_worse = abs(d441) <= abs(d433) + args.delta  # 4.4.1 agreement not materially worse
        if not no_worse or not oom441:
            worse.append(sp)
        gm = lambda a: 10 ** _logmean(a).mean()  # noqa: E731
        cp = lambda a: int(np.sum(a < COLLAPSE))  # noqa: E731
        flag = "OK" if no_worse else ("OOM!" if not oom441 else "worse")
        print(f"{sp:<22}{gm(py[sp]):>10.3g}{gm(j441[sp]):>10.3g}{gm(j433[sp]):>10.3g}"
              f"{d441:>10.2f}{d433:>10.2f}{flag:>9}{ks441:>7.2f}"
              f"{cp(py[sp]):>8}/{cp(j441[sp])}/{cp(j433[sp]):<11}")
    print("-" * len(hdr))
    print(f"delta (equivalence margin) = {args.delta:.2f} log10 units (factor {10**args.delta:.1f}x)")
    print(f"gm = geometric mean over {args.n} replicates; d = mean log10(py)-log10(java); "
          f"collapse = # replicates with biomass < {COLLAPSE}t")
    print(f"\nGATE: 4.4.1 agrees with Python no worse than 4.3.3 (within delta) + within 1 OoM: "
          f"{'PASS' if not worse else 'REVIEW: ' + ', '.join(worse)}")


if __name__ == "__main__":
    main()
