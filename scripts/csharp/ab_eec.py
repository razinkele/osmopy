"""C# engine performance on EEC, spec 10.4's protocol (plan task 1.11).

Each engine runs in its own long-lived worker process, warmed once, then driven run by run:
pairs are interleaved (the engine going first alternates), only one engine runs at a time,
and both use ``output.minimal.enabled``. Reported:

- median wall time per engine and the ratio Python / C# (gate 1.0x, target 1.2x), with every
  pair's ratio and whether all pairs agree in sign;
- the C# first-call overhead in a fresh process: a cold 1-year run minus a warm one (budget 2 s);
- the C# bundle build plus marshalling as a share of a run (budget 5%);
- each worker's peak RSS (budget: C# no more than Python), and its baseline after the imports and
  the config are loaded but before any run, so the run's own share is visible;
- the C# engine's phase split (RunSummary).

The timed span matches ``scripts/benchmark_engine.py``: Python ``simulate``; C# the bundle,
the buffers, ``RunSimulation`` and the ``StepOutput`` list. Config parsing is outside both.

Usage: PYTHONPATH=. python scripts/csharp/ab_eec.py [--pairs 7] [--threads 1 14] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "data" / "eec_full" / "eec_all-parameters.csv"


def _prepare(years: int):
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig

    raw = dict(OsmoseConfigReader().read(CONFIG))
    raw["simulation.time.nyear"] = str(years)
    raw["output.minimal.enabled"] = "true"
    return raw, EngineConfig.from_dict(raw), PythonEngine()._resolve_grid(raw)


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def worker(engine: str, threads: int, years: int, seed: int) -> None:
    """Line protocol on stdin/stdout: 'run' -> one JSON result line; 'quit' -> peak RSS."""
    from osmose.engine.thread_policy import apply_single_run_threads

    apply_single_run_threads(threads)
    import numpy as np

    def timed(raw, cfg, grid) -> dict:
        if engine == "csharp":
            from osmose.engine_cs import bridge
            from osmose.engine_cs.build import allocate_results, build_bundle
            from osmose.engine_cs.engine import _step_outputs

            t0 = time.perf_counter()
            bundle = build_bundle(cfg, raw, grid, seed=seed, threads=threads)
            buffers = allocate_results(bundle, cfg)
            t1 = time.perf_counter()
            summary = bridge.run(bundle, buffers.control)
            _step_outputs(buffers, summary[2])
            wall = time.perf_counter() - t0
            return {"wall": wall, "bundle": t1 - t0, "engine": summary[4], "phases": summary[6]}
        from osmose.engine.simulate import simulate

        t0 = time.perf_counter()
        simulate(cfg, grid, np.random.default_rng(seed))
        return {"wall": time.perf_counter() - t0}

    out: dict = {"engine": engine, "threads": threads}
    short, full = _prepare(1), _prepare(years)
    if engine == "python":
        import osmose.engine.simulate  # noqa: F401  (the C# side's imports are its run cost)
    out["rss_baseline_mb"] = _rss_mb()
    if engine == "csharp":  # first-call overhead: cold 1-year run minus warm 1-year run
        out["cold_1yr"] = timed(*short)["wall"]
        out["warm_1yr"] = timed(*short)["wall"]
    out["warmup"] = timed(*full)["wall"]
    print(json.dumps({"ready": out}), flush=True)
    for line in sys.stdin:
        if line.strip() == "run":
            print(json.dumps(timed(*full)), flush=True)
        elif line.strip() == "quit":
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(json.dumps({"peak_rss_mb": rss_kb / 1024}), flush=True)
            return


class Worker:
    def __init__(self, engine: str, threads: int, years: int, seed: int):
        env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), PYTHONPATH=str(ROOT))
        self.p = subprocess.Popen(
            [sys.executable, __file__, "--worker", engine, str(threads), str(years), str(seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, cwd=ROOT,
        )  # fmt: skip
        self.ready = self._read()["ready"]

    def _read(self) -> dict:
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("worker exited")
            if line.startswith("{"):
                return json.loads(line)

    def run(self) -> dict:
        self.p.stdin.write("run\n")
        self.p.stdin.flush()
        return self._read()

    def quit(self) -> float:
        self.p.stdin.write("quit\n")
        self.p.stdin.flush()
        rss = self._read()["peak_rss_mb"]
        self.p.wait(timeout=60)
        return rss


def measure(threads: int, pairs: int, years: int, seed: int) -> dict:
    py = Worker(
        "python", threads, years, seed
    )  # warm-ups one after the other: one engine at a time
    cs = Worker("csharp", threads, years, seed)
    rows = []
    for i in range(pairs):
        order = (py, cs) if i % 2 == 0 else (cs, py)
        res = {w: w.run() for w in order}
        rows.append(
            {
                "python": res[py]["wall"],
                "csharp": res[cs]["wall"],
                "bundle": res[cs]["bundle"],
                "phases": res[cs]["phases"],
            }
        )
        print(
            f"  threads {threads} pair {i + 1}: python {rows[-1]['python']:.3f}s  csharp {rows[-1]['csharp']:.3f}s",
            flush=True,
        )
    rss = {"python": py.quit(), "csharp": cs.quit()}
    baseline = {"python": py.ready["rss_baseline_mb"], "csharp": cs.ready["rss_baseline_mb"]}
    med_py = statistics.median(r["python"] for r in rows)
    med_cs = statistics.median(r["csharp"] for r in rows)
    ratios = [r["python"] / r["csharp"] for r in rows]
    phases = {k: statistics.median(r["phases"][k] for r in rows) for k in rows[0]["phases"]}
    return {
        "threads": threads, "pairs": rows, "median_python_s": med_py, "median_csharp_s": med_cs,
        "ratio": med_py / med_cs, "pair_ratios": ratios, "all_pairs_agree": all(r > 1 for r in ratios) or all(r < 1 for r in ratios),
        "bundle_share": statistics.median(r["bundle"] / r["csharp"] for r in rows),
        "first_call_overhead_s": cs.ready["cold_1yr"] - cs.ready["warm_1yr"],
        "peak_rss_mb": rss, "rss_baseline_mb": baseline, "csharp_phases_s": phases,
    }  # fmt: skip


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
        return
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--pairs", type=int, default=7)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 14])
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    load = os.getloadavg()
    print(
        f"load average {load[0]:.2f} {load[1]:.2f} {load[2]:.2f} on {os.cpu_count()} CPUs"
        + ("  (NOT idle)" if load[0] > 2 else "")
    )
    results = [measure(t, args.pairs, args.years, args.seed) for t in args.threads]
    print()
    for r in results:
        verdict = "PASS" if r["ratio"] >= 1.0 else "FAIL"
        print(
            f"threads {r['threads']}: python {r['median_python_s']:.3f}s  csharp {r['median_csharp_s']:.3f}s  "
            f"ratio {r['ratio']:.2f}x (gate 1.0x {verdict}, target 1.2x); pairs agree: {r['all_pairs_agree']}"
        )
        print(
            f"  first call {r['first_call_overhead_s']:.2f}s (budget 2 s); bundle share {r['bundle_share'] * 100:.1f}% (budget 5%); "
            f"peak RSS python {r['peak_rss_mb']['python']:.0f} MB (baseline {r['rss_baseline_mb']['python']:.0f}), "
            f"csharp {r['peak_rss_mb']['csharp']:.0f} MB (baseline {r['rss_baseline_mb']['csharp']:.0f})"
        )
        print("  C# phases: " + ", ".join(f"{k} {v:.3f}s" for k, v in r["csharp_phases_s"].items()))
    if args.json:
        args.json.write_text(
            json.dumps({"load_average": load, "years": args.years, "results": results}, indent=1)
        )


if __name__ == "__main__":
    main()
