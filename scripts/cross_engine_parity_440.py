#!/usr/bin/env python3
"""Cross-engine ensemble parity: Python <-> Java {4.4.1, 4.3.3} for any config (--config).

Cross-engine RNG streams diverge by construction (Python PCG64 vs Java MT19937), so the test is
statistical, per species and per metric (biomass, yield, abundance, and mean individual weight =
biomass/abundance as a size-structure proxy), on the final-year mean over N varied-seed reps
(log10-scaled, ~log-normal):

  - GATE (per species): ABSOLUTE equivalence via TOST (two one-sided t-tests vs +-Delta;
    Lakens & Delacre 2020) against Java-4.4.1, PLUS a 1-OoM catastrophic-divergence tripwire.
  - Java-4.3.3 is a REPORTED reference only (|Python - 4.3.3| shown per species, not gated).
  - Also reported per species: two-sample KS p-value, variance ratio, collapse frequency, and
    the 90% CI half-width (minimum detectable difference at this N).
  - --engines selects which arms run (a config loads on a specific jar set); --persist-results
    keeps one OsmoseResults dir per selected Java arm for downstream (Phase 3) reuse.

Usage: PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py \\
         --config data/eec_full/eec_all-parameters.csv --n 16 --years 10
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import stats

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import OsmoseResults
from ui.pages.run import write_temp_config

ROOT = Path(__file__).resolve().parent.parent
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


def _read_prefix(config: Path) -> str:
    """The Java output-file prefix (OsmoseResults globs '{prefix}_{type}*.csv')."""
    raw = dict(OsmoseConfigReader().read(str(config)))
    return raw.get("output.file.prefix", config.parent.name)


def python_rep(config: Path, years: int, seed: int, spinup: int) -> dict[str, dict[str, float]]:
    raw = dict(OsmoseConfigReader().read(str(config)))
    raw["simulation.time.nyear"] = str(years)
    res = PythonEngine().run_in_memory(raw, seed=seed)
    return {m: _final_mean(_reader(res, m), years, spinup) for m in METRICS}


def java_rep(ver: str, master: Path, years: int, odir: Path, spinup: int, prefix: str):
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
    res = OsmoseResults(
        odir, prefix=prefix, strict=False
    )  # strict=False: report-empty, don't crash
    return {m: _final_mean(_reader(res, m), years, spinup) for m in METRICS}


def ensemble(engine: str, config: Path, prefix: str, years: int, n: int, spinup: int, tmp: Path):
    """-> {metric: {species: array(N)}}."""
    reps = []
    if engine == "python":
        for s in range(n):
            reps.append(python_rep(config, years, 1000 + s, spinup))
    else:
        raw = dict(OsmoseConfigReader().read(str(config)))
        master = tmp / f"stage_{engine}"
        write_temp_config(raw, master, source_dir=config.parent, target_version=engine)
        master = master / "osm_all-parameters.csv"
        for s in range(n):
            r = java_rep(engine, master, years, tmp / f"out_{engine}_{s}", spinup, prefix)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--spinup-years", type=int, default=2)
    ap.add_argument(
        "--delta", type=float, default=np.log10(3), help="equivalence margin, log10 units"
    )
    ap.add_argument(
        "--config", type=Path, default=ROOT / "data" / "eec_full" / "eec_all-parameters.csv"
    )
    ap.add_argument("--engines", default="python,4.4.1,4.3.3", help="comma list subset")
    ap.add_argument("--persist-results", type=Path, default=None)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="xengine_"))
    prefix = _read_prefix(args.config)
    engines = [e.strip() for e in args.engines.split(",")]

    if "python" in engines:  # determinism check
        a, b = python_rep(args.config, 3, 7, 1), python_rep(args.config, 3, 7, 1)
        det = all(np.isclose(a["biomass"][k], b["biomass"][k]) for k in a["biomass"])
        print(f"[determinism] Python same-seed reproducible: {det}")
    t0 = time.perf_counter()
    print(
        f"[run] {args.n} reps x {len(engines)} engines x {args.years}yr x {len(METRICS)} metrics ..."
    )

    ens = {}  # only the selected engines (each config supports a specific set)
    for e in engines:
        ens[e] = ensemble(e, args.config, prefix, args.years, args.n, args.spinup_years, tmp)
    py = ens.get("python")
    j441 = ens.get("4.4.1")
    j433 = ens.get("4.3.3")
    print(
        f"[run] done in {time.perf_counter() - t0:.0f}s  (delta={args.delta:.2f} log10 = {10**args.delta:.1f}x)\n"
    )

    # Derive a size-structure metric: mean individual weight = biomass / abundance, paired per
    # replicate (both come from the same run). Captures growth/larval-units shifts that biomass alone
    # hides. floor 1e-9 t/ind (biomass/abundance are in tonnes/numbers).
    for eng in filter(None, (py, j441, j433)):
        eng["mean_weight"] = {
            sp: eng["biomass"][sp] / np.clip(eng["abundance"][sp], 1e-9, None)
            for sp in eng["biomass"]
            if sp in eng["abundance"]
        }
    analysis_metrics = METRICS + ("mean_weight",)
    floors = {"mean_weight": 1e-9}

    if py is None:
        print("no python arm — nothing to gate")
        return
    present = [v for v in ("4.4.1", "4.3.3") if ens.get(v) is not None]
    # A selected Java arm that produced ZERO comparable species (every replicate errored) must be
    # reported, not silently omitted — otherwise present/sp_all filter it out and the gate prints a
    # vacuous PASS (the design's "a dropped arm degrades the reference invisibly" hazard).
    empty_arms = [v for v in present if not any(ens[v][m] for m in analysis_metrics)]
    for v in empty_arms:
        print(f"[warn] engine {v} produced ZERO comparable species — dropped arm, NOT a clean PASS")
    overall_fail = []
    for m in analysis_metrics:
        floor = floors.get(m, COLLAPSE)
        sp_all = [s for s in py[m] if all(s in ens[v][m] for v in present)]
        print(f"==================== METRIC: {m} ====================")
        for sp in sp_all:
            row = f"{sp:<22}"
            if j441 is not None:
                d1, ci1, p1, eq1, ks1, vr1 = tost(py[m][sp], j441[m][sp], args.delta, floor)
                if not eq1 or abs(d1) >= 1.0:  # PRIMARY gate: absolute equivalence + 1-OoM tripwire
                    overall_fail.append(f"{m}:{sp}")
                row += f"  441 d={d1:>6.2f} eq={'Y' if eq1 else 'n'} KS={ks1:.2f}"
            if j433 is not None:  # reference / reported only, never gated
                d3, _, _, eq3, _, _ = tost(py[m][sp], j433[m][sp], args.delta, floor)
                row += f"  | 433 d={d3:>6.2f} eq={'Y' if eq3 else 'n'}"
            print(row)
    tag = (
        "absolute Python<->4.4.1 equivalence + within 1 OoM"
        if j441 is not None
        else "reference run (no 4.4.1 arm — not gated)"
    )
    if empty_arms:
        verdict = f"FAIL (dropped arm(s) with zero comparable species: {', '.join(empty_arms)})"
    elif overall_fail:
        verdict = "REVIEW: " + ", ".join(overall_fail)
    else:
        verdict = "PASS"
    print(f"GATE ({tag}): {verdict}")

    if args.persist_results:
        args.persist_results.mkdir(parents=True, exist_ok=True)
        for ver in [e for e in engines if e != "python"]:  # only selected java arms
            raw = dict(OsmoseConfigReader().read(str(args.config)))
            st = tmp / f"persist_stage_{ver}"
            write_temp_config(raw, st, source_dir=args.config.parent, target_version=ver)
            java_rep(
                ver,
                st / "osm_all-parameters.csv",
                args.years,
                args.persist_results / f"{prefix}_{ver}",
                args.spinup_years,
                prefix,
            )
        # Python arm is in-memory; Task 12 recomputes it directly (no persist needed).


if __name__ == "__main__":
    main()
