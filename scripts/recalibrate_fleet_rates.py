#!/usr/bin/env python
"""Re-solve Baltic fleet base rates to ICES 2020-2024 catch (spec 2026-11-28 fleet-rate recal).

Solves exactly four fleets (fsh0 cod_west, fsh8 cod_east, fsh1 herring, fsh3 flounder) to their
`catch` target in data/baltic/reference/biomass_targets.csv with osmose.calibration.fleet_rates.
Sprat (fsh2) keeps its ICES-F anchor and is never solved. Each evaluation = a 50-yr in-memory run
per seed on the SP1-off production base; catch = trailing 5-yr mean yield, averaged over seeds.

    PYTHONPATH=. python scripts/recalibrate_fleet_rates.py --baseline-only --engine python \
        --seeds 42 123 7 999 2024 --out docs/diagnostics/fleet_recal_baseline.json
    PYTHONPATH=. python scripts/recalibrate_fleet_rates.py --engine csharp --seeds 42 123 \
        --out docs/diagnostics/fleet_recal_search.json
    PYTHONPATH=. python scripts/recalibrate_fleet_rates.py --apply --rates-json <log.json>

Runs the engine in THIS process, one job at a time, and never forks after loading the CLR.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from osmose.calibration.fleet_rates import Solve, solve_rate, trailing_catch

TARGETS_CSV = REPO / "data" / "baltic" / "reference" / "biomass_targets.csv"
FISHING_CSV = REPO / "data" / "baltic" / "baltic_param-fishing.csv"

#: fleet id -> (species it catches, its base-rate key). Sprat fsh2 deliberately absent.
FLEETS: dict[str, tuple[str, str]] = {
    "fsh0": ("cod_west", "fisheries.rate.base.fsh0"),
    "fsh8": ("cod_east", "fisheries.rate.base.fsh8"),
    "fsh1": ("herring", "fisheries.rate.base.fsh1"),
    "fsh3": ("flounder", "fisheries.rate.base.fsh3"),
}


def load_catch_targets(path: Path = TARGETS_CSV) -> dict[str, float]:
    """Species -> target_tonnes of the `catch` rows."""
    import csv

    with open(path) as fh:  # whole-line comments only: notes may contain '#'
        rows = csv.DictReader(ln for ln in fh if not ln.startswith("#"))
        return {
            r["species"]: float(r["target_tonnes"])
            for r in rows
            if r["reference_point_type"] == "catch"
        }


def make_evaluator(
    base_cfg: Mapping[str, str],
    rate_key: str,
    species: str,
    engine,
    seeds: Iterable[int],
    n_years: int = 50,
    log: list | None = None,
) -> Callable[[float], float]:
    """rate -> seed-mean trailing-5-yr catch of ``species``, only ``rate_key`` changed."""
    seeds = list(seeds)

    def evaluate(rate: float) -> float:
        cfg = dict(base_cfg)
        cfg[rate_key] = repr(float(rate))
        cfg["simulation.time.nyear"] = str(n_years)
        t0 = time.perf_counter()
        catches = [
            trailing_catch(engine.run_in_memory(cfg, seed=s).yield_biomass(), species)
            for s in seeds
        ]
        mean = sum(catches) / len(catches)
        if log is not None:
            log.append(
                {"key": rate_key, "rate": rate, "catch": mean, "per_seed": catches,
                 "seconds": round(time.perf_counter() - t0, 1)}
            )  # fmt: skip
        print(f"  {rate_key} = {rate:.5g} -> {species} catch {mean:,.0f} t", flush=True)
        return mean

    return evaluate


def apply_rates(path: Path, rates: Mapping[str, float]) -> None:
    """Replace each ``fisheries.rate.base.<fsh>;<value>`` line in place (Java keeps the FIRST
    definition of a duplicated key, so appending would be dead). A missing key raises."""
    lines = Path(path).read_text().splitlines(keepends=True)
    for fsh, rate in rates.items():
        pat = re.compile(rf"^fisheries\.rate\.base\.{re.escape(fsh)}\s*;")
        hits = [i for i, ln in enumerate(lines) if pat.match(ln)]
        if len(hits) != 1:
            raise KeyError(f"fisheries.rate.base.{fsh}: {len(hits)} lines in {path}, need 1")
        end = "\n" if lines[hits[0]].endswith("\n") else ""
        lines[hits[0]] = f"fisheries.rate.base.{fsh};{rate:.6g}{end}"
    Path(path).write_text("".join(lines))


def _base_config() -> dict[str, str]:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo

    res = osmose_demo("baltic", Path(tempfile.mkdtemp()))
    return dict(OsmoseConfigReader().read(str(res["config_file"])))


def _engine(name: str):
    from osmose.engine.select import get_engine

    if name == "csharp":
        from osmose.engine.thread_policy import physical_budget

        return get_engine("csharp", threads=physical_budget())
    return get_engine(name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", choices=("csharp", "python"), default="csharp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--fleets", default=",".join(FLEETS))
    ap.add_argument("--rates-json", help="start rates: a previous --out log (its final_rates)")
    ap.add_argument("--out", help="JSON log path")
    ap.add_argument("--baseline-only", action="store_true", help="evaluate the rates once")
    ap.add_argument("--apply", action="store_true", help="write --rates-json into the CSV")
    a = ap.parse_args(argv)

    fleets = [f for f in a.fleets.split(",") if f]
    unknown = set(fleets) - set(FLEETS)
    if unknown:
        ap.error(f"unknown fleets {sorted(unknown)}; solvable: {sorted(FLEETS)}")
    start: dict[str, float] = {}
    if a.rates_json:
        start = {
            k: float(v)
            for k, v in json.loads(Path(a.rates_json).read_text())["final_rates"].items()
        }

    if a.apply:
        if not start:
            ap.error("--apply needs --rates-json")
        apply_rates(FISHING_CSV, start)
        print(f"applied {start} to {FISHING_CSV}")
        return 0

    base = _base_config()
    rates = {f: start.get(f, float(base[FLEETS[f][1]])) for f in FLEETS}
    for f, r in rates.items():
        base[FLEETS[f][1]] = repr(r)
    targets = load_catch_targets()
    engine = _engine(a.engine)
    log: list[dict] = []
    record: dict = {"engine": a.engine, "seeds": a.seeds, "years": a.years, "targets": targets,
                    "start_rates": dict(rates), "evaluations": log, "solves": []}  # fmt: skip

    def catch_of(f: str) -> float:
        sp, key = FLEETS[f]
        return make_evaluator(base, key, sp, engine, a.seeds, a.years, log)(rates[f])

    if a.baseline_only:
        # one run per seed gives every fleet's catch: evaluate once, read all four species
        cfg = dict(base, **{"simulation.time.nyear": str(a.years)})
        ys = [engine.run_in_memory(cfg, seed=s).yield_biomass() for s in a.seeds]
        out = {}
        for f in fleets:
            sp = FLEETS[f][0]
            c = sum(trailing_catch(y, sp) for y in ys) / len(ys)
            out[f] = {"species": sp, "rate": rates[f], "catch": c, "target": targets[sp],
                      "ratio": c / targets[sp]}  # fmt: skip
            print(f"{f} {sp:9s} rate {rates[f]:.5g}  catch {c:,.0f} t  target {targets[sp]:,.0f} t"
                  f"  ({c / targets[sp]:.2f}x)")  # fmt: skip
        sp2 = "sprat"
        c2 = sum(trailing_catch(y, sp2) for y in ys) / len(ys)
        print(f"fsh2 sprat     (anchored) catch {c2:,.0f} t  target {targets[sp2]:,.0f} t"
              f"  ({c2 / targets[sp2]:.2f}x)")  # fmt: skip
        record["baseline"] = out
        record["sprat_catch"] = c2
        record["final_rates"] = rates
    else:
        for p in range(a.passes):
            print(f"pass {p + 1}", flush=True)
            for f in fleets:
                sp, key = FLEETS[f]
                ev = make_evaluator(base, key, sp, engine, a.seeds, a.years, log)
                s: Solve = solve_rate(ev, rates[f], targets[sp], tol=a.tol)
                rates[f] = s.rate
                base[key] = repr(s.rate)
                record["solves"].append({"pass": p + 1, "fleet": f, "species": sp, "rate": s.rate,
                                         "catch": s.catch, "target": s.target,
                                         "reachable": s.reachable, "note": s.note})  # fmt: skip
                print(f"{f} -> rate {s.rate:.5g}, catch {s.catch:,.0f} / {s.target:,.0f}"
                      f" ({'ok' if s.reachable else s.note})", flush=True)  # fmt: skip
            # re-check: coupling may have pushed an earlier fleet out
            done = all(
                abs(catch_of(f) / targets[FLEETS[f][0]] - 1) <= a.tol
                or not next(x for x in reversed(record["solves"]) if x["fleet"] == f)["reachable"]
                for f in fleets
            )
            if done:
                break
        record["final_rates"] = rates
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(record, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
