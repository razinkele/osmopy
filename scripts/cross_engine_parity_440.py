#!/usr/bin/env python3
"""Cross-engine ensemble parity: Python <-> Java {4.4.1, 4.3.3} for any config (--config).

Cross-engine RNG streams diverge by construction (Python PCG64 vs Java MT19937), so the test is
statistical, per species and per metric (biomass, yield, abundance, and mean individual weight =
biomass/abundance as a size-structure proxy; plus mean_size when both engines actually wrote it),
on the post-spinup mean over N varied-seed reps (log10-scaled, ~log-normal) -- see ``_final_mean``:
at the default --years/--spinup-years it is NOT just the final year, it is everything from
year (years - spinup) onward (e.g. years=5, spinup=2 averages years 3-5, the last 60% of the run):

  - GATE (per species): ABSOLUTE equivalence via TOST (two one-sided t-tests vs +-Delta;
    Lakens & Delacre 2020) against the selected --gate-engine (default 4.4.1), PLUS a 1-OoM
    catastrophic-divergence tripwire.
  - The other Java arm is a REPORTED reference only (|Python - other| shown per species, not
    gated).
  - Also reported per species: two-sample KS p-value, variance ratio, collapse frequency, and
    the 90% CI half-width (minimum detectable difference at this N).
  - --engines selects which arms run (a config loads on a specific jar set); --persist-results
    keeps one OsmoseResults dir per selected Java arm for downstream (Phase 3) reuse.
  - --require-nondegenerate fails the run (before the equivalence numbers are even meaningful)
    if any species collapsed to ~zero in >=10% of reps on the Python arm or the gate arm.

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
OPTIONAL_METRICS = ("mean_size",)  # only promoted into the report if actually produced (see main())
COLLAPSE = 1.0  # tonnes/numbers floor for the collapse count + log clamp


def _reader(results, metric: str):
    return {
        "biomass": results.biomass,
        "yield": results.yield_biomass,
        "abundance": results.abundance,
        "mean_size": results.mean_size,
    }[metric]()


def _final_mean(df, years: int, spinup: int) -> dict[str, float]:
    """Post-spinup per-column mean, NOT a final-year mean.

    ``spy`` = rows/year; the tail is the last ``max(spy, n - spinup*spy)`` rows, i.e. everything
    from year ``(years - spinup)`` onward. At Gate B's own parameters (years=5, spinup=2, 24
    rows/year -> spy=24, n=120) that tail is 72 rows = years 3-5, not the final year alone --
    more rows averaged than "final year" suggests, which is why several TOST rows below show a
    90% CI half-width of 0.00 (see MINOR-1 in the task-9 review / the diagnostics doc).
    """
    cols = [
        c for c in df.columns if c not in ("Time", "species") and not str(c).startswith("Unnamed")
    ]
    cols = [c for c in cols if str(c).lower() != "all"]
    n = len(df)
    spy = max(1, n // years)
    tail = df.iloc[max(0, n - max(spy, n - spinup * spy)) :]
    return {c: float(np.nanmean(tail[c].to_numpy(dtype=float))) for c in cols}


def _optional_metric(results, metric: str, years: int, spinup: int) -> dict[str, float]:
    """Best-effort read of an optional per-species metric; {} if it was never produced.

    In-memory Python results raise FileNotFoundError for an output type that was never written
    (e.g. output.size.enabled=false); disk-backed Java results (strict=False) just return an
    empty frame instead. Both collapse to the same "not available" signal here so ensemble()
    never has to special-case which engine is asking.
    """
    try:
        df = _reader(results, metric)
    except FileNotFoundError:
        return {}
    if df.empty:
        return {}
    return _final_mean(df, years, spinup)


def _read_prefix(config: Path) -> str:
    """The Java output-file prefix (OsmoseResults globs '{prefix}_{type}*.csv')."""
    raw = dict(OsmoseConfigReader().read(str(config)))
    return raw.get("output.file.prefix", config.parent.name)


def inject_java_bioen_keys(master: Path, raw: dict[str, str]) -> int:
    """Java 4.3.3 reads predation.ingestion.rate.max.bioen / .coef...larvae.bioen / predation.c.bioen
    for EVERY predator index (focal + background) and exits on a missing key; the writer's reverse
    alias cannot losslessly reconstruct predation.ingestion.rate.max.bioen from the merged canonical
    predation.ingestion.rate.max (that merge is intentionally lossy — see osmose/config/aliases.py's
    _INVERSE_440 comment). Append the .bioen trio from the canonical values instead. No-op unless
    bioen is on. The other two keys may already be correctly present via the reverse alias (the
    larvae-ratio one) or pass through unaliased (predation.c.bioen); appending them again anyway is
    harmless — Java's Configuration loads parameters into a HashMap keyed by name, so a repeated
    identical line is just an overwrite with the same value, not a duplicate-key error.
    """
    if str(raw.get("module.bioenergetics.enabled", "false")).lower() != "true":
        return 0
    n_sp = int(raw.get("simulation.nspecies", "0"))
    idx = list(range(n_sp)) + sorted(
        int(k.split(".sp")[-1])
        for k, v in raw.items()
        if k.startswith("species.type.sp") and str(v).strip().lower() == "background"
    )
    lines = []
    for i in idx:
        imax = raw.get(f"predation.ingestion.rate.max.sp{i}")
        if imax is None:
            raise KeyError(
                f"predation.ingestion.rate.max.sp{i} missing; cannot stage bioen for Java 4.3.3"
            )
        lines.append(f"predation.ingestion.rate.max.bioen.sp{i} ; {imax}\n")
        lines.append(
            f"predation.coef.ingestion.rate.max.larvae.bioen.sp{i} ; "
            f"{raw.get(f'predation.larval.ingestion.rate.increase.ratio.sp{i}', '1.0')}\n"
        )
        lines.append(f"predation.c.bioen.sp{i} ; {raw.get(f'predation.c.bioen.sp{i}', '0.0')}\n")
    with master.open("a") as fh:
        fh.writelines(lines)
    return len(lines)


def inject_java_resource_nsteps_year(master: Path, raw: dict[str, str]) -> int:
    """Java 4.3.3's ResourceForcing.init() requires species.biomass.nsteps.year(.spN) for any
    NetCDF-file-forced resource species (``species.file.spN`` set) when the per-species key is
    absent; a global (non-indexed) fallback is accepted (see ResourceForcing.java:150-153).
    NOT bioen-specific — this bites the bioen-OFF control run too.

    data/eec_full's own fixture already carries this key (eec_param-ltl.csv line 77:
    'species.biomass.nsteps.year;24'), and osmose/config/aliases.py's
    _emit_resource_biomass_forcing() synthesizes the per-species form for a >=4.4.0 target from
    the very same source — but data/examples / data/examples_bioen (the Bay-of-Biscay demo) were
    bundled without it, and nothing synthesizes it on the <4.4.0 (reverse-alias) branch of
    to_target_keys(). The Python engine never needed it (its own resource loader reads the step
    count straight off the NetCDF), so this went unnoticed until a direct Java 4.3.3 run of this
    fixture — Gate B's first. No-op if the key is already present or no resource species is
    file-forced. Sourced from simulation.time.ndtperyear, matching the NetCDF's own frame count
    (verified for data/examples_bioen: 24 steps/year in both the config and the .nc file).
    """
    if "species.biomass.nsteps.year" in raw:
        return 0
    if not any(k.startswith("species.file.sp") for k in raw):
        return 0
    ndt = raw.get("simulation.time.ndtperyear")
    if not ndt:
        return 0
    with master.open("a") as fh:
        fh.write(f"species.biomass.nsteps.year ; {ndt}\n")
    return 1


def comparable_species(
    py_metric: dict[str, np.ndarray], ens: dict, present: list[str], m: str
) -> list[str]:
    """Species with a value from the python arm AND every present java arm, for metric ``m``.

    Empty means metric ``m`` could not be compared at all for at least one arm (e.g. that
    metric's CSV came back empty for one engine while other metrics still had data — the
    per-metric sibling of the whole-arm ``empty_arms`` check). Callers MUST treat an empty
    result as "not evaluated", never as a silent pass: a metric with zero rows prints nothing
    and, unguarded, would leave the gate looking clean over a comparison that never ran (this is
    a narrower instance of the same "gate green over code it never executed" defect that
    ``empty_arms``, above, guards against at the whole-arm level).

    This is a different signal from a *degenerate but real* comparison (e.g. ``tost()``'s
    ``se == 0`` short-circuit on a species with zero measured variance in both engines, which
    still appears here — it has a value in every arm, just an uninformative one). This function
    only asks whether the species key exists in every arm's dict for this metric; it does not
    look at the values.
    """
    return [s for s in py_metric if all(s in ens[v][m] for v in present)]


def gate_verdict(
    empty_arms: list[str],
    uncompared_metrics: list[str],
    degenerate_report: list[str],
    overall_fail: list[str],
) -> str:
    """Roll the four failure classes (checked in this priority order) into the final GATE line."""
    if empty_arms:
        return f"FAIL (dropped arm(s) with zero comparable species: {', '.join(empty_arms)})"
    if uncompared_metrics:
        return (
            "FAIL (metric(s) with zero comparable species across present arms: "
            f"{', '.join(uncompared_metrics)})"
        )
    if degenerate_report:
        return "FAIL (degenerate: " + ", ".join(degenerate_report) + ")"
    if overall_fail:
        return "REVIEW: " + ", ".join(overall_fail)
    return "PASS"


def nondegenerate(ens, metric: str, n: int, floor: float, frac: float = 0.9) -> dict[str, bool]:
    """Per-species: did this engine's ensemble avoid collapsing to ~zero in >=10% of reps?

    ``ens`` is one engine's aggregated {metric: {species: array(N)}} (e.g. the ``py`` or
    ``ens["4.3.3"]`` dict from main(), NOT the full multi-engine ``ens`` dict). A rep counts as
    "ok" only if it is finite AND more than 100x the metric's collapse floor; a species must be
    "ok" in at least ``frac`` of exactly ``n`` reps (a short array — some reps never reported this
    species at all — also fails, via the size check).
    """
    out = {}
    for sp, arr in ens[metric].items():
        ok = np.isfinite(arr) & (arr > 100.0 * floor)
        out[sp] = bool(arr.size == n and ok.mean() >= frac)
    return out


def python_rep(config: Path, years: int, seed: int, spinup: int) -> dict[str, dict[str, float]]:
    raw = dict(OsmoseConfigReader().read(str(config)))
    raw["simulation.time.nyear"] = str(years)
    res = PythonEngine().run_in_memory(raw, seed=seed)
    out = {m: _final_mean(_reader(res, m), years, spinup) for m in METRICS}
    out.update({m: _optional_metric(res, m, years, spinup) for m in OPTIONAL_METRICS})
    return out


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
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if cp.returncode != 0:
        print(f"[java {ver}] rep FAILED (exit {cp.returncode}); stderr tail:\n{cp.stderr[-2000:]}")
        return None
    res = OsmoseResults(
        odir, prefix=prefix, strict=False
    )  # strict=False: report-empty, don't crash
    out = {m: _final_mean(_reader(res, m), years, spinup) for m in METRICS}
    out.update({m: _optional_metric(res, m, years, spinup) for m in OPTIONAL_METRICS})
    return out


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
        if engine == "4.3.3":
            n_injected = inject_java_bioen_keys(master, raw)
            if n_injected:
                print(
                    f"[stage] {engine}: injected {n_injected} .bioen key line(s) for bioen "
                    "predation (predation.ingestion.rate.max.bioen and friends)"
                )
            n_nsteps = inject_java_resource_nsteps_year(master, raw)
            if n_nsteps:
                print(
                    f"[stage] {engine}: injected species.biomass.nsteps.year (not bioen-specific — "
                    "required by any NetCDF-file-forced resource species on this jar)"
                )
        for s in range(n):
            r = java_rep(engine, master, years, tmp / f"out_{engine}_{s}", spinup, prefix)
            if r is not None:
                reps.append(r)
        if len(reps) != n:
            raise RuntimeError(
                f"engine {engine}: {len(reps)}/{n} reps succeeded — see stderr of the first "
                "failure above"
            )
    out: dict[str, dict[str, np.ndarray]] = {}
    for m in METRICS + OPTIONAL_METRICS:
        species = sorted({k for r in reps for k in r[m]})
        out[m] = {sp: np.array([r[m].get(sp, np.nan) for r in reps], dtype=float) for sp in species}
    return out


def _log(a, floor: float = COLLAPSE):
    return np.log10(np.clip(a, floor, None))


def tost(py, jv, delta: float, floor: float = COLLAPSE):
    """Formal TOST: returns (mean_log_diff, ci90_halfwidth, p_tost, equivalent, ks_p, var_ratio).

    ``se == 0`` (both engines' N reps have exactly zero measured variance on the log scale for
    this species/metric -- seen throughout the mean_weight rows, whose reported values are so
    tightly clamped by the model that 16 reps don't separate) is a real, valid comparison, not a
    dropped one: `d`/`eq` are still computed from genuine data. It just can't run a t-test with
    zero pooled variance, so `ci90` is reported as the literal 0.0 (not a computed interval) and
    `KS` as `nan` (`ks_2samp` is never called on this branch -- not "not significant", "not
    applicable"). `eq` here is a bare `abs(d) <= delta` threshold check, not a p-value verdict.
    This is unrelated to a metric having zero comparable species at all (see
    ``comparable_species``/``uncompared_metrics`` in main()) -- that case has no `d`/`eq` to show
    because the species never reaches this function in the first place.
    """
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
        "--delta-mean-weight",
        type=float,
        default=np.log10(1.5),
        help="equivalence margin for mean_weight and mean_size, log10 units (tighter than --delta)",
    )
    ap.add_argument(
        "--config", type=Path, default=ROOT / "data" / "eec_full" / "eec_all-parameters.csv"
    )
    ap.add_argument("--engines", default="python,4.4.1,4.3.3", help="comma list subset")
    ap.add_argument(
        "--gate-engine",
        choices=["4.4.1", "4.3.3"],
        default="4.4.1",
        help="which Java arm is the PRIMARY (gated) comparison; the other is reported only",
    )
    ap.add_argument(
        "--require-nondegenerate",
        action="store_true",
        help="FAIL when any species collapses (near-zero biomass/abundance) in >=10%% of reps "
        "on the python arm or the gate arm",
    )
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
    gate_ver = args.gate_engine
    report_ver = "4.3.3" if gate_ver == "4.4.1" else "4.4.1"
    j_gate = ens.get(gate_ver)
    j_report = ens.get(report_ver)
    print(
        f"[run] done in {time.perf_counter() - t0:.0f}s  (delta={args.delta:.2f} log10 = {10**args.delta:.1f}x, "
        f"gate-engine={gate_ver})\n"
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
    floors = {
        "mean_weight": 1e-9,
        "mean_size": 1e-6,
    }  # 1e-6 cm: below any real length, avoids log(0)
    deltas = {"mean_weight": args.delta_mean_weight, "mean_size": args.delta_mean_weight}

    if py is None:
        print("no python arm — nothing to gate")
        return
    present = [v for v in ("4.4.1", "4.3.3") if ens.get(v) is not None]
    # mean_size is only meaningful (and only added to the report) when the python arm AND every
    # present java arm actually wrote a non-empty meanSize frame (not every config turns on
    # output.size.enabled, and a java rep's output is independent of the python engine's).
    mean_size_ok = bool(py.get("mean_size")) and all(bool(ens[v].get("mean_size")) for v in present)
    if mean_size_ok:
        analysis_metrics = analysis_metrics + ("mean_size",)
    else:
        print(
            "[note] mean_size metric skipped: results.mean_size() frame was empty for the python "
            "arm and/or a present java arm"
        )
    # A selected Java arm that produced ZERO comparable species (every replicate errored) must be
    # reported, not silently omitted — otherwise present/sp_all filter it out and the gate prints a
    # vacuous PASS (the design's "a dropped arm degrades the reference invisibly" hazard).
    empty_arms = [v for v in present if not any(ens[v][m] for m in analysis_metrics)]
    for v in empty_arms:
        print(f"[warn] engine {v} produced ZERO comparable species — dropped arm, NOT a clean PASS")

    degenerate_report: list[str] = []
    if args.require_nondegenerate:
        check_engines = {"python": py}
        if j_gate is not None:
            check_engines[gate_ver] = j_gate
        print(f"\n[non-degeneracy check] (>=10% of {args.n} reps collapsed => FAIL)")
        for ename, edata in check_engines.items():
            for m in ("biomass", "abundance"):
                nd = nondegenerate(edata, m, args.n, COLLAPSE, frac=0.9)
                for sp, arr in edata[m].items():
                    ok = np.isfinite(arr) & (arr > 100.0 * COLLAPSE)
                    cf = 1.0 - float(ok.mean()) if arr.size else 1.0
                    tag = "" if nd.get(sp, False) else "  <-- DEGENERATE"
                    print(f"  {ename:<8} {m:<10} {sp:<22} collapse_frac={cf:.2f}{tag}")
                    if not nd.get(sp, False):
                        degenerate_report.append(f"{ename}:{m}:{sp}(collapse_frac={cf:.2f})")
        print()

    overall_fail = []
    uncompared_metrics: list[str] = []
    for m in analysis_metrics:
        floor = floors.get(m, COLLAPSE)
        delta_m = deltas.get(m, args.delta)
        sp_all = comparable_species(py[m], ens, present, m)
        print(f"==================== METRIC: {m} ====================")
        if not sp_all:
            uncompared_metrics.append(m)
            print(
                f"  [warn] metric {m}: zero species had a value in every present arm — NOT evaluated, NOT a pass"
            )
            continue
        for sp in sp_all:
            row = f"{sp:<22}"
            if j_gate is not None:
                d1, ci1, p1, eq1, ks1, vr1 = tost(py[m][sp], j_gate[m][sp], delta_m, floor)
                if not eq1 or abs(d1) >= 1.0:  # PRIMARY gate: absolute equivalence + 1-OoM tripwire
                    overall_fail.append(f"{m}:{sp}")
                row += f"  {gate_ver} d={d1:>6.2f} ci90=+-{ci1:.2f} eq={'Y' if eq1 else 'n'} KS={ks1:.2f}"
            if j_report is not None:  # reference / reported only, never gated
                d3, ci3, _, eq3, _, _ = tost(py[m][sp], j_report[m][sp], delta_m, floor)
                row += f"  | {report_ver} d={d3:>6.2f} ci90=+-{ci3:.2f} eq={'Y' if eq3 else 'n'}"
            print(row)
    tag = (
        f"absolute Python<->{gate_ver} equivalence + within 1 OoM"
        if j_gate is not None
        else f"reference run (no {gate_ver} arm — not gated)"
    )
    verdict = gate_verdict(empty_arms, uncompared_metrics, degenerate_report, overall_fail)
    print(f"GATE ({tag}): {verdict}")

    if args.persist_results:
        args.persist_results.mkdir(parents=True, exist_ok=True)
        for ver in [e for e in engines if e != "python"]:  # only selected java arms
            raw = dict(OsmoseConfigReader().read(str(args.config)))
            st = tmp / f"persist_stage_{ver}"
            write_temp_config(raw, st, source_dir=args.config.parent, target_version=ver)
            master = st / "osm_all-parameters.csv"
            if ver == "4.3.3":
                inject_java_bioen_keys(master, raw)
                inject_java_resource_nsteps_year(master, raw)
            java_rep(
                ver,
                master,
                args.years,
                args.persist_results / f"{prefix}_{ver}",
                args.spinup_years,
                prefix,
            )
        # Python arm is in-memory; Task 12 recomputes it directly (no persist needed).


if __name__ == "__main__":
    main()
