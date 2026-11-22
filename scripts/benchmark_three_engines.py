#!/usr/bin/env python3
"""Time the Java, Python and C# engines doing the same job on the same model.

Every run is ONE FRESH PROCESS that does the whole job: read the config, simulate, write the
outputs. That is the only span the three engines have in common: the Java engine is a separate
program and can only be timed as a whole, so the in-process core that
``scripts/benchmark_engine.py`` times (Python and C# only) cannot be compared with it.

For each run the parent records the wall clock, the child's CPU time (user + system) and its
peak resident memory, all from ``os.wait4`` on that child. CPU / wall is the number of cores the
engine actually kept busy, which is how the table shows what "threads" meant for each engine
(the JVM uses helper threads even when the model runs on one).

What makes it the same job:

* the same config file and the same ``simulation.time.nyear``, with the config's own outputs;
* Java gets the config through ``stage_config_for_java``, exactly as the app does (for
  ``eec_full`` that adds the resource-forcing keys in the jar's spelling and changes no value),
  and the staging itself is done once, outside the timed runs;
* after every run the parent checks the output directory: the biomass series must have one row
  per saved step, and the file count is recorded.

Threads: the Python child calls ``apply_single_run_threads(n)`` and caps the BLAS pools; the C#
child passes ``threads=n`` to the engine; the Java engine has no thread setting for one
simulation and is run as the app runs it (``java -Xmx2g -jar ...``).

The engines are interleaved (one round runs every arm once, and the order rotates between
rounds), so that drift of the machine falls on all of them. The first ``--warmup`` rounds are
discarded: the first Python run in a checkout pays Numba's compilation, which is cached on disk
afterwards.

The parent never loads the .NET runtime (a process that has loaded it must not fork).

Usage:
    .venv/bin/python scripts/benchmark_three_engines.py --years 70 --repeats 5 \\
        --output docs/perf/three-engines.json
    .venv/bin/python scripts/benchmark_three_engines.py --years 5 --repeats 1 --warmup 0  # pilot
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # this checkout's osmose, not an editable install elsewhere

FIXTURES = {
    "eec_full": ROOT / "data" / "eec_full" / "eec_all-parameters.csv",
    "examples": ROOT / "data" / "examples" / "osm_all-parameters.csv",
    "minimal": ROOT / "data" / "minimal" / "osm_all-parameters.csv",
}
JAVA_VERSION = "4.4.1"
DEFAULT_JAR = ROOT / "osmose-java" / "osmose-4.4.1-jar-with-dependencies.jar"


# --------------------------------------------------------------------------- the child


def child(engine: str, config: str, years: int, threads: int, out_dir: str, seed: int) -> int:
    """One in-process engine, start to finish. Runs in a process of its own."""
    from osmose.config.reader import OsmoseConfigReader

    raw = OsmoseConfigReader().read(Path(config))
    raw["simulation.time.nyear"] = str(years)
    if engine == "python":
        from osmose.engine import PythonEngine
        from osmose.engine.thread_policy import apply_single_run_threads

        apply_single_run_threads(threads)
        PythonEngine().run(raw, Path(out_dir), seed=seed)
    elif engine == "csharp":
        from osmose.engine_cs.engine import CSharpEngine

        CSharpEngine(threads=threads).run(raw, Path(out_dir), seed=seed)
    else:
        raise SystemExit(f"unknown in-process engine: {engine}")
    return 0


# -------------------------------------------------------------------------- the parent


def _timed(cmd: list[str], env: dict[str, str], log: Path) -> dict:
    """Run ``cmd`` to the end and return its wall clock, CPU time and peak memory."""
    with log.open("w") as sink:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, env=env, stdout=sink, stderr=subprocess.STDOUT, cwd=ROOT)
        _, status, usage = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - start
    proc.returncode = os.waitstatus_to_exitcode(status)
    cpu = usage.ru_utime + usage.ru_stime
    return {
        "exit": proc.returncode,
        "wall_s": round(wall, 3),
        "cpu_s": round(cpu, 3),
        "cores_used": round(cpu / wall, 2) if wall > 0 else 0.0,
        "peak_rss_mb": round(usage.ru_maxrss / 1024, 1),  # Linux reports kilobytes
    }


def _outputs(out_dir: Path, years: int, steps_per_year: int, record_every: int) -> dict:
    """What the run left behind: file count, bytes, and the biomass series' row count."""
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    biomass = sorted(p for p in files if p.name.endswith("_biomass_Simu0.csv"))
    rows = None
    if biomass:
        lines = [ln for ln in biomass[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
        # A title line and a header line, then one row per saved step.
        rows = max(0, len(lines) - 2)
    return {
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "biomass_rows": rows,
        # The engines also write the window the run ends inside, complete or not.
        "expected_rows": -(-(years * steps_per_year) // record_every),
    }


def _thread_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = str(threads)
    return env


def _versions(jar: Path) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    def pkg(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    def first_line(cmd: list[str], starting: str = "") -> str | None:
        """The first line of the command's output that starts with ``starting``."""
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        text = [
            ln
            for ln in (done.stdout or done.stderr).strip().splitlines()
            if ln.startswith(starting)
        ]
        # Name and version only: `dotnet --list-runtimes` appends the install path in brackets.
        return text[0].split(" [")[0] if text else None

    dotnet = os.environ.get("DOTNET_ROOT") or str(Path.home() / ".dotnet")
    cpu = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": pkg("numpy"),
        "numba": pkg("numba"),
        "pandas": pkg("pandas"),
        "pythonnet": pkg("pythonnet"),
        # The runtime that hosts the engine, not the first one listed (ASP.NET Core).
        "dotnet_runtime": first_line(
            [str(Path(dotnet) / "dotnet"), "--list-runtimes"], "Microsoft.NETCore.App"
        ),
        "java": first_line(["java", "-version"]),
        "java_engine_jar": jar.name,
        "osmose": pkg("osmose-python"),
        "git": first_line(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
    }


def _run_ok(run: dict) -> bool:
    """Exit 0 AND an output directory whose biomass series has every expected row. A run that
    left no output at all has no counts, and must not pass as "None equals None"."""
    rows = run.get("biomass_rows")
    return run["exit"] == 0 and rows is not None and rows == run.get("expected_rows")


def _summary(runs: list[dict]) -> dict:
    out = {}
    for key in ("wall_s", "cpu_s", "cores_used", "peak_rss_mb"):
        values = [r[key] for r in runs]
        out[key] = {
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Time the Java, Python and C# engines doing the same job on the same model"
    )
    parser.add_argument("--config", default="eec_full", help="a fixture name or a config path")
    parser.add_argument("--years", type=int, default=70)
    parser.add_argument("--repeats", type=int, default=5, help="measured rounds")
    parser.add_argument("--warmup", type=int, default=1, help="discarded rounds before them")
    parser.add_argument("--threads", default="1,14", help="thread counts for Python and C#")
    parser.add_argument("--engines", default="java,python,csharp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--output", type=Path, default=None, help="write the JSON record here")
    parser.add_argument("--keep", type=Path, default=None, help="keep each arm's last outputs here")
    args = parser.parse_args()

    config = FIXTURES.get(args.config, Path(args.config)).resolve()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    thread_counts = [int(t) for t in args.threads.split(",") if t.strip()]

    from osmose.config.reader import OsmoseConfigReader

    raw = OsmoseConfigReader().read(config)
    steps_per_year = int(raw["simulation.time.ndtperyear"])
    record_every = int(raw.get("output.recordfrequency.ndt", steps_per_year))

    work = Path(tempfile.mkdtemp(prefix="osmose_bench3_"))
    arms: list[dict] = []
    if "java" in engines:
        from osmose.java_staging import stage_config_for_java

        staged = dict(raw)
        master, overrides = stage_config_for_java(
            staged, work / "java_staged", config.parent, target_version=JAVA_VERSION
        )
        arms.append(
            {
                "name": "java",
                "engine": "java",
                "threads": None,
                "master": master,
                "overrides": overrides,
            }
        )
    for n in thread_counts:
        for engine in ("python", "csharp"):
            if engine in engines:
                arms.append({"name": f"{engine}@{n}", "engine": engine, "threads": n})

    def command(arm: dict, out_dir: Path) -> tuple[list[str], dict[str, str]]:
        if arm["engine"] == "java":
            cmd = [
                "java",
                "-Xmx2g",  # the app's own default (osmose/runner.py)
                "-jar",
                str(args.jar),
                str(arm["master"]),
                f"-Poutput.dir.path={out_dir}",
                f"-Psimulation.time.nyear={args.years}",
                *(f"-P{k}={v}" for k, v in arm["overrides"].items()),
            ]
            return cmd, dict(os.environ)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            arm["engine"],
            str(config),
            str(args.years),
            str(arm["threads"]),
            str(out_dir),
            str(args.seed),
        ]
        return cmd, _thread_env(arm["threads"])

    results: dict[str, list[dict]] = {arm["name"]: [] for arm in arms}
    total = args.warmup + args.repeats
    print(
        f"{config.name}: {args.years} years, arms {[a['name'] for a in arms]}, {args.warmup} warm-up + {args.repeats} rounds"
    )
    for rnd in range(total):
        order = arms[rnd % len(arms) :] + arms[: rnd % len(arms)]
        for arm in order:
            out_dir = work / f"out_{arm['name'].replace('@', '_')}_{rnd}"
            cmd, env = command(arm, out_dir)
            run = _timed(cmd, env, work / f"log_{arm['name'].replace('@', '_')}_{rnd}.txt")
            run.update(
                _outputs(out_dir, args.years, steps_per_year, record_every)
                if out_dir.exists()
                else {}
            )
            run["round"] = rnd
            run["warmup"] = rnd < args.warmup
            ok = _run_ok(run)
            run["ok"] = ok
            tag = "warm-up" if run["warmup"] else f"round {rnd - args.warmup + 1}"
            print(
                f"  {tag:8s} {arm['name']:10s} wall {run['wall_s']:8.2f} s  cpu {run['cpu_s']:8.2f} s"
                f"  cores {run['cores_used']:5.2f}  peak {run['peak_rss_mb']:7.0f} MB"
                f"  files {run.get('files')}  rows {run.get('biomass_rows')}  {'ok' if ok else 'FAILED'}",
                flush=True,
            )
            results[arm["name"]].append(run)
            if not ok:
                print(f"    see {work}/log_{arm['name'].replace('@', '_')}_{rnd}.txt")
            if args.keep and rnd == total - 1 and out_dir.exists():
                args.keep.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    out_dir, args.keep / arm["name"].replace("@", "_"), dirs_exist_ok=True
                )
            shutil.rmtree(out_dir, ignore_errors=True)

    record = {
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(config.relative_to(ROOT)) if config.is_relative_to(ROOT) else str(config),
        "years": args.years,
        "seed": args.seed,
        "warmup_rounds": args.warmup,
        "rounds": args.repeats,
        "versions": _versions(args.jar),
        "arms": {},
    }
    failed = False
    print("\nmedian of the measured rounds (min-max):")
    for arm in arms:
        measured = [r for r in results[arm["name"]] if not r["warmup"]]
        if not measured or not all(r["ok"] for r in measured):
            failed = True
            print(f"  {arm['name']:10s} FAILED or not measured")
            record["arms"][arm["name"]] = {"runs": results[arm["name"]], "summary": None}
            continue
        s = _summary(measured)
        record["arms"][arm["name"]] = {
            "threads": arm["threads"],
            "runs": results[arm["name"]],
            "summary": s,
        }
        w = s["wall_s"]
        print(
            f"  {arm['name']:10s} wall {w['median']:8.2f} s ({w['min']:.2f}-{w['max']:.2f})"
            f"  cores {s['cores_used']['median']:5.2f}  peak {s['peak_rss_mb']['median']:7.0f} MB"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=1) + "\n")
        print(f"\nwrote {args.output}")
    shutil.rmtree(work, ignore_errors=True) if not failed else print(f"logs kept in {work}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, eng, cfg, yrs, thr, out, sd = sys.argv
        raise SystemExit(child(eng, cfg, int(yrs), int(thr), out, int(sd)))
    raise SystemExit(main())
