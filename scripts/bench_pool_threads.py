#!/usr/bin/env python3
"""Worker x threads-per-worker sweep for the calibration process pool (perf plan item 1).

Each cell runs in its own process with NUMBA_NUM_THREADS / OMP_NUM_THREADS / MKL_NUM_THREADS /
OPENBLAS_NUM_THREADS and OSMOSE_WORKER_THREADS set to T. "unr" sets OSMOSE_WORKER_THREADS=unrestricted
(Numba's default, the pre-2026-11-14 behaviour); "policy" sets nothing, so the pools apply their own
split rule (osmose.engine.thread_policy). The cell builds an
``OsmoseCalibrationProblem`` with ``parallel_backend="process"`` and ``n_parallel=W`` on the
production Baltic config (its default horizon), discards one warm-up round of W evaluations, then
times R rounds of W evaluations in one batch on the same pool. W=1 evaluates serially in the cell
process (the problem only uses a pool for W > 1).

Criteria: docs/diagnostics/worker_thread_sweep_prestated_2026-11-14.md.

Usage (from the repo root; one engine job at a time):
    .venv/bin/python scripts/bench_pool_threads.py grid --out sweep.jsonl
    .venv/bin/python scripts/bench_pool_threads.py confirm --cells 14:1 8:unr 16:1 --repeats 3 \\
        --out confirm.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BALTIC_CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
THREAD_VARS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
WORKER_THREADS_ENV = "OSMOSE_WORKER_THREADS"  # osmose.engine.thread_policy override
WORKERS = (1, 4, 8, 12, 14, 16)
THREADS = ("1", "2", "unr")
SPLIT_EXTRA = ((1, "14"), (4, "3"))  # T = max(1, 14 // W) where it is not already in the grid


class _BenchObjective:
    """Picklable total-biomass objective (as in scripts/benchmark_calibration.py)."""

    def __call__(self, results) -> float:
        df = results.biomass()
        numeric = df.select_dtypes(include="number").drop(columns=["Time"], errors="ignore")
        return float(numeric.iloc[-1].sum())


class _Sampler:
    """Polls this process and all its descendants: peak total RSS and cumulative CPU seconds."""

    def __init__(self, interval: float = 0.5) -> None:
        import psutil

        self._psutil = psutil
        self._me = psutil.Process()
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _procs(self):
        try:
            return [self._me, *self._me.children(recursive=True)]
        except self._psutil.Error:
            return [self._me]

    def cpu_seconds(self, exclude_self: bool) -> float:
        total = 0.0
        for p in self._procs():
            if exclude_self and p.pid == self._me.pid:
                continue
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except self._psutil.Error:
                pass
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = 0
            for p in self._procs():
                try:
                    rss += p.memory_info().rss
                except self._psutil.Error:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def run_cell(workers: int, rounds: int) -> dict:
    """Run one cell in THIS process (its thread env vars were set by the caller)."""
    import numpy as np

    from osmose.calibration import FreeParameter, Transform
    from osmose.calibration.problem import OsmoseCalibrationProblem
    from osmose.schema import build_registry

    rng = np.random.default_rng(12345)
    with tempfile.TemporaryDirectory() as d, _Sampler() as sampler:
        problem = OsmoseCalibrationProblem(
            free_params=[FreeParameter("mortality.fishing.rate.sp0", 0.1, 0.5, Transform.LINEAR)],
            objective_fns=[_BenchObjective()],
            base_config_path=BALTIC_CONFIG,
            work_dir=Path(d),
            registry=build_registry(),
            n_parallel=workers,
            parallel_backend="process",
            enable_cache=False,
        )
        try:
            t0 = time.perf_counter()
            problem._evaluate(rng.uniform(0.1, 0.5, size=(workers, 1)), {})
            warmup_s = time.perf_counter() - t0
            in_pool = workers > 1
            cpu0 = sampler.cpu_seconds(exclude_self=in_pool)
            t1 = time.perf_counter()
            out: dict = {}
            problem._evaluate(rng.uniform(0.1, 0.5, size=(workers * rounds, 1)), out)
            timed_s = time.perf_counter() - t1
            cpu = sampler.cpu_seconds(exclude_self=in_pool) - cpu0
        finally:
            problem.shutdown_pool()
    n_eval = workers * rounds
    return {
        "workers": workers,
        "rounds": rounds,
        "n_eval": n_eval,
        "warmup_s": round(warmup_s, 2),
        "timed_s": round(timed_s, 2),
        "evals_per_h": round(n_eval / timed_s * 3600, 1),
        "s_per_eval_per_worker": round(timed_s / rounds, 2),
        "cores_per_worker": round(cpu / timed_s / workers, 2),
        "peak_rss_gb": round(sampler.peak_rss / 1e9, 2),
        "all_finite": bool(np.all(np.isfinite(out.get("F", np.array([np.inf]))))),
    }


def launch(workers: int, threads: str, rounds: int) -> dict:
    """Run one cell in a fresh process with the thread env vars set (or unset for 'unr')."""
    env = dict(os.environ)
    # OSMOSE_NSGA2_WORKERS would silently replace W in the pool (_resolve_worker_count) while
    # the record still said W, so it is cleared with the thread variables.
    for v in (*THREAD_VARS, WORKER_THREADS_ENV, "OSMOSE_NSGA2_WORKERS"):
        env.pop(v, None)
    if threads == "unr":
        env[WORKER_THREADS_ENV] = "unrestricted"  # pools would otherwise apply their own rule
    elif threads != "policy":  # "policy": nothing set, the pools' built-in rule decides
        for v in THREAD_VARS:
            env[v] = threads
        env[WORKER_THREADS_ENV] = threads
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "cell",
        "--workers",
        str(workers),
        "--rounds",
        str(rounds),
    ]
    load = os.getloadavg()[0]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"cell W={workers} T={threads} failed:\n{proc.stderr[-2000:]}")
    rec = json.loads(lines[-1])
    rec.update(
        {"threads": threads, "load_at_start": round(load, 2), "started": time.strftime("%H:%M:%S")}
    )
    return rec


def _emit(rec: dict, out: Path) -> None:
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("cell", help="(internal) run one cell in this process")
    c.add_argument("--workers", type=int, required=True)
    c.add_argument("--rounds", type=int, default=3)
    g = sub.add_parser("grid", help="run the pre-stated grid sequentially")
    g.add_argument("--rounds", type=int, default=3)
    g.add_argument("--out", type=Path, required=True)
    k = sub.add_parser("confirm", help="interleaved repeats of chosen cells (W:T)")
    k.add_argument("--cells", nargs="+", required=True)
    k.add_argument("--repeats", type=int, default=3)
    k.add_argument("--rounds", type=int, default=3)
    k.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.mode == "cell":
        print(json.dumps(run_cell(args.workers, args.rounds)), flush=True)
        return 0
    if args.mode == "grid":
        cells = [(w, t) for w in WORKERS for t in THREADS] + list(SPLIT_EXTRA)
        for w, t in cells:
            _emit({"phase": "grid", **launch(w, t, args.rounds)}, args.out)
        return 0
    cells = [(int(s.split(":")[0]), s.split(":")[1]) for s in args.cells]
    for rep in range(args.repeats):
        for w, t in cells:
            _emit({"phase": "confirm", "repeat": rep, **launch(w, t, args.rounds)}, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
