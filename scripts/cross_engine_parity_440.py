#!/usr/bin/env python3
"""Phase 2 cross-engine ensemble parity: Python <-> Java {4.4.1, 4.3.3} on EEC.

The deferred jar-swap's Phase 2 (docs/superpowers/plans/2026-06-19-jar-swap-440-validated-resume.md):
confirm Java-4.4.1 agrees with the pure-Python engine *at least as well as* Java-4.3.3 does, using a
multi-replicate, distributional + equivalence comparison (NOT a single-seed run). Cross-engine streams
diverge by construction (Python PCG64 vs Java MT19937), so the test is statistical:

  - per species and per metric (biomass, yield, abundance, and mean individual weight =
    biomass/abundance as a size-structure proxy), final-year mean over N varied-seed reps;
  - work on log10 (these are ~ log-normal);
  - EQUIVALENCE (formal TOST, two one-sided t-tests vs +-Delta; Lakens & Delacre 2020);
  - DISTRIBUTION: two-sample KS p-value; variance ratio;
  - COMMUNITY SKILL: MEF (Nash-Sutcliffe modelling efficiency) + Spearman on the per-species vector;
  - RELATIVE GATE: |Python - 4.4.1| no worse than |Python - 4.3.3| (within Delta);
  - COLLAPSE frequency per engine; 1-OoM only as a catastrophic tripwire;
  - precision: achieved 90% CI half-width per species = the minimum detectable difference at this N.

Size-structure is covered via mean individual weight (biomass/abundance, derived from the two
collected ensembles). Metrics still deferred (need output-flag enablement + Java support, not in
EEC's default output set): F (fishing mortality), mean trophic level, full size spectra.

Usage: PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py --n 16 --years 10
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
METRICS = ("biomass", "yield", "abundance")
COLLAPSE = 1.0  # tonnes/numbers floor for the collapse count + log clamp


def _reader(results, metric: str):
    return {
        "biomass": results.biomass,
        "yield": results.yield_biomass,
        "abundance": results.abundance,
    }[metric]()


def _final_mean(df, years: int, spinup: int) -> dict[str, float]:
    cols = [
        c for c in df.columns if c not in ("Time", "species") and not str(c).startswith("Unnamed")
    ]
    cols = [c for c in cols if str(c).lower() != "all"]
    n = len(df)
    spy = max(1, n // years)
    tail = df.iloc[max(0, n - max(spy, n - spinup * spy)) :]
    return {c: float(np.nanmean(tail[c].to_numpy(dtype=float))) for c in cols}


def python_rep(years: int, seed: int, spinup: int) -> dict[str, dict[str, float]]:
    raw = dict(OsmoseConfigReader().read(str(EEC)))
    raw["simulation.time.nyear"] = str(years)
    res = PythonEngine().run_in_memory(raw, seed=seed)
    return {m: _final_mean(_reader(res, m), years, spinup) for m in METRICS}


def java_rep(ver: str, master: Path, years: int, odir: Path, spinup: int):
    odir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-Xmx2g",
        "-jar",
        str(JARS[ver]),
        str(master),
        f"-Poutput.dir.path={odir}",
        f"-Psimulation.time.nyear={years}",
        "-Poutput.start.year=0",
    ]
    if subprocess.run(cmd, capture_output=True, text=True, timeout=900).returncode != 0:
        return None
    res = OsmoseResults(odir, prefix="eec")
    return {m: _final_mean(_reader(res, m), years, spinup) for m in METRICS}


def ensemble(engine: str, years: int, n: int, spinup: int, tmp: Path):
    """-> {metric: {species: array(N)}}."""
    reps = []
    if engine == "python":
        for s in range(n):
            reps.append(python_rep(years, 1000 + s, spinup))
    else:
        raw = dict(OsmoseConfigReader().read(str(EEC)))
        master = tmp / f"stage_{engine}"
        write_temp_config(raw, master, source_dir=EEC.parent, target_version=engine)
        master = master / "osm_all-parameters.csv"
        for s in range(n):
            r = java_rep(engine, master, years, tmp / f"out_{engine}_{s}", spinup)
            if r is not None:
                reps.append(r)
    out: dict[str, dict[str, np.ndarray]] = {}
    for m in METRICS:
        species = sorted({k for r in reps for k in r[m]})
        out[m] = {sp: np.array([r[m].get(sp, np.nan) for r in reps], dtype=float) for sp in species}
    return out


def _log(a, floor: float = COLLAPSE):
    return np.log10(np.clip(a, floor, None))


def tost(py, jv, delta: float, floor: float = COLLAPSE):
    """Formal TOST: returns (mean_log_diff, ci90_halfwidth, p_tost, equivalent, ks_p, var_ratio)."""
    lp, lj = _log(py, floor), _log(jv, floor)
    n1, n2 = len(lp), len(lj)
    d = lp.mean() - lj.mean()
    se = np.sqrt(lp.var(ddof=1) / n1 + lj.var(ddof=1) / n2)
    df = n1 + n2 - 2
    if se == 0:
        eq = abs(d) <= delta
        return d, 0.0, 0.0 if eq else 1.0, eq, np.nan, np.nan
    p_lower = stats.t.sf((d + delta) / se, df)  # H0: d <= -delta
    p_upper = stats.t.cdf((d - delta) / se, df)  # H0: d >= +delta
    p_tost = max(p_lower, p_upper)
    ci = stats.t.ppf(0.95, df) * se
    ks_p = float(stats.ks_2samp(py, jv).pvalue)
    vr = lp.var(ddof=1) / lj.var(ddof=1) if lj.var(ddof=1) > 0 else np.inf
    return d, ci, p_tost, p_tost < 0.05, ks_p, vr


def mef_spearman(py_vec, jv_vec, floor: float = COLLAPSE):
    """Community skill on the per-species log-gm vectors: MEF (Java=obs, Python=pred) + Spearman."""
    obs, pred = _log(np.array(jv_vec), floor), _log(np.array(py_vec), floor)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    mef = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rho = float(stats.spearmanr(py_vec, jv_vec).statistic)
    return mef, rho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--spinup-years", type=int, default=2)
    ap.add_argument(
        "--delta", type=float, default=np.log10(3), help="equivalence margin, log10 units"
    )
    args = ap.parse_args()
    tmp = Path("/home/razinka/.claude/jobs/18a62785/tmp/phase2v2")
    tmp.mkdir(parents=True, exist_ok=True)

    a, b = python_rep(3, 7, 1), python_rep(3, 7, 1)
    det = all(np.isclose(a["biomass"][k], b["biomass"][k]) for k in a["biomass"])
    print(f"[determinism] Python same-seed reproducible: {det}")
    t0 = time.perf_counter()
    print(f"[run] {args.n} reps x 3 engines x {args.years}yr x {len(METRICS)} metrics ...")
    py = ensemble("python", args.years, args.n, args.spinup_years, tmp)
    j441 = ensemble("4.4.1", args.years, args.n, args.spinup_years, tmp)
    j433 = ensemble("4.3.3", args.years, args.n, args.spinup_years, tmp)
    print(
        f"[run] done in {time.perf_counter() - t0:.0f}s  (delta={args.delta:.2f} log10 = {10**args.delta:.1f}x)\n"
    )

    # Derive a size-structure metric: mean individual weight = biomass / abundance, paired per
    # replicate (both come from the same run). Captures growth/larval-units shifts that biomass alone
    # hides. floor 1e-9 t/ind (biomass/abundance are in tonnes/numbers).
    for eng in (py, j441, j433):
        eng["mean_weight"] = {
            sp: eng["biomass"][sp] / np.clip(eng["abundance"][sp], 1e-9, None)
            for sp in eng["biomass"]
            if sp in eng["abundance"]
        }
    analysis_metrics = METRICS + ("mean_weight",)
    floors = {"mean_weight": 1e-9}

    overall_fail = []
    for m in analysis_metrics:
        floor = floors.get(m, COLLAPSE)
        sp_all = [s for s in py[m] if s in j441[m] and s in j433[m]]
        print(f"==================== METRIC: {m} ====================")
        hdr = f"{'species':<22}{'d(py-441)':>10}{'±CI90':>7}{'TOST_p':>8}{'equiv':>6}{'KS':>6}{'441≤433':>8}{'coll p/441/433':>16}"
        print(hdr)
        for sp in sp_all:
            d1, ci1, p1, eq1, ks1, vr1 = tost(py[m][sp], j441[m][sp], args.delta, floor)
            d3, _, _, _, _, _ = tost(py[m][sp], j433[m][sp], args.delta, floor)
            no_worse = abs(d1) <= abs(d3) + args.delta
            cp = lambda a: int(np.sum(np.asarray(a) < floor))  # noqa: E731
            if not no_worse or abs(d1) >= 1.0:
                overall_fail.append(f"{m}:{sp}")
            print(
                f"{sp:<22}{d1:>10.2f}{ci1:>7.2f}{p1:>8.3f}{'Y' if eq1 else 'n':>6}{ks1:>6.2f}"
                f"{'Y' if no_worse else 'N':>8}{cp(py[m][sp]):>6}/{cp(j441[m][sp])}/{cp(j433[m][sp]):<7}"
            )
        # community skill on per-species geometric-mean vectors
        v_py = [10 ** _log(py[m][s], floor).mean() for s in sp_all]
        v441 = [10 ** _log(j441[m][s], floor).mean() for s in sp_all]
        v433 = [10 ** _log(j433[m][s], floor).mean() for s in sp_all]
        mef1, rho1 = mef_spearman(v_py, v441, floor)
        mef3, rho3 = mef_spearman(v_py, v433, floor)
        n_eq = sum(tost(py[m][s], j441[m][s], args.delta, floor)[3] for s in sp_all)
        print(
            f"  community: Py~441 MEF={mef1:.2f} Spearman={rho1:.2f} | Py~433 MEF={mef3:.2f} "
            f"Spearman={rho3:.2f} | TOST-equivalent {n_eq}/{len(sp_all)} species\n"
        )

    print(
        f"GATE (4.4.1 no worse than 4.3.3 + within 1 OoM, all metrics): "
        f"{'PASS' if not overall_fail else 'REVIEW: ' + ', '.join(overall_fail)}"
    )


if __name__ == "__main__":
    main()
