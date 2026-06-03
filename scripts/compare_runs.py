#!/usr/bin/env python3
"""Compare two finished OSMOSE runs: per-species output delta, ranked by % change.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compare_runs.py \\
        --baseline <dir> --variant <dir> [--prefix osm] \\
        [--baseline-prefix P] [--variant-prefix P] \\
        [--metric biomass|yield|abundance] [--window-years 10] [--top-n N] \\
        [--report out.md] [--json out.json] [--plot out_prefix]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--variant", required=True, type=Path)
    p.add_argument("--prefix", type=str, default="osm")
    p.add_argument("--baseline-prefix", type=str, default=None)
    p.add_argument("--variant-prefix", type=str, default=None)
    p.add_argument("--metric", type=str, default="biomass", choices=["biomass", "yield", "abundance"])
    p.add_argument("--window-years", type=int, default=10)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args(argv)
    if args.window_years < 1:
        p.error("--window-years must be >= 1")
    if args.top_n is not None and args.top_n < 1:
        p.error("--top-n must be >= 1")

    from osmose.results import OsmoseResults
    from osmose import analysis as az

    bpref = args.baseline_prefix or args.prefix
    vpref = args.variant_prefix or args.prefix
    baseline = OsmoseResults(args.baseline, prefix=bpref, strict=False)
    variant = OsmoseResults(args.variant, prefix=vpref, strict=False)
    deltas = az.run_delta(baseline, variant, metric=args.metric,
                          window_years=args.window_years, top_n=args.top_n)
    report = az.format_delta_report(deltas, metric=args.metric, window_years=args.window_years)
    if args.report:
        args.report.write_text(report)
    print(report)
    if args.json:
        args.json.write_text(json.dumps([asdict(d) for d in deltas], indent=2))
    if args.plot:
        from osmose import plotting
        plotting.make_run_delta_chart(deltas, metric=args.metric).write_html(f"{args.plot}_delta.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
