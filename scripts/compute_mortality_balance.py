#!/usr/bin/env python3
"""Per-species F/M (fishing vs natural mortality) diagnostics for a finished run.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compute_mortality_balance.py \\
        --results-dir <path> [--prefix osm] [--window-years 10] \\
        [--steps-per-year N] [--config <param-dir-or-file>] \\
        [--species cod sprat ...] [--report out.md] [--json out.json] [--plot out_prefix]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_steps_per_year(args) -> int:
    if args.steps_per_year is not None:
        return args.steps_per_year
    search = [args.config] if args.config else []
    search += [args.results_dir, args.results_dir.parent]
    ndt = rec = None
    for base in search:
        if base is None:
            continue
        base = Path(base)
        if not base.exists():
            continue
        files = (
            [base]
            if base.is_file()
            else list(base.glob("*param-simulation.csv")) + list(base.glob("*param-output.csv"))
        )
        for f in files:
            try:
                for line in f.read_text().splitlines():
                    parts = [p.strip() for p in line.replace(",", ";").split(";")]
                    key = parts[0] if parts else ""
                    # Value is the last non-empty field (real configs have a
                    # trailing separator, e.g. "output.recordfrequency.ndt,24,").
                    vals = [p for p in parts[1:] if p]
                    if not vals:
                        continue
                    if key == "simulation.time.ndtPerYear":
                        ndt = int(float(vals[-1]))
                    elif key == "output.recordfrequency.ndt":
                        # Exact key only — must not be shadowed by
                        # "output.restart.recordfrequency.ndt".
                        rec = int(float(vals[-1]))
            except (OSError, ValueError):
                continue
    if ndt and rec:
        spy = max(1, ndt // rec)
        print(f"steps_per_year = {spy} (ndtPerYear={ndt} / recordfrequency.ndt={rec})")
        return spy
    print(
        "WARNING: could not derive steps_per_year from config; defaulting to 1 "
        "(correct iff output.recordfrequency.ndt == simulation.time.ndtPerYear). "
        "Pass --steps-per-year if record frequency is finer than annual.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--prefix", type=str, default="osm")
    p.add_argument("--window-years", type=int, default=10)
    p.add_argument("--steps-per-year", type=int, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--species", nargs="*", default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args(argv)
    from osmose.validation import fisheries as fz

    spy = _resolve_steps_per_year(args)
    balances = fz.compute_mortality_balance(
        args.results_dir,
        prefix=args.prefix,
        species_list=args.species,
        steps_per_year=spy,
        window_years=args.window_years,
    )
    report = fz.format_mortality_report(balances, window_years=args.window_years)
    if args.report:
        args.report.write_text(report)
    print(report)
    if args.json:
        args.json.write_text(json.dumps([asdict(b) for b in balances], indent=2))
    if args.plot:
        from osmose import plotting

        plotting.make_fm_ratio_bars(balances).write_html(f"{args.plot}_fm.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
