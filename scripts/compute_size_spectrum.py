#!/usr/bin/env python3
"""Compute community size-spectrum diagnostics for a finished OSMOSE run.

Reads the {prefix}_{metric}DistribBySize community output and reports the size
spectrum, its log-log slope (length-biomass spectrum, trend/comparison only — not
the Sheldon exponent), the Large-Fish Indicator, mean size, and the peak bin.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compute_size_spectrum.py \\
        --results-dir <dir> [--metric biomass|abundance] [--prefix osm] \\
        [--window-years 10] [--lfi-threshold-cm 40] [--min-size-cm N] \\
        [--report out.md] [--json out.json] [--plot out_prefix]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--metric", type=str, default="biomass", choices=["biomass", "abundance"])
    p.add_argument("--prefix", type=str, default="osm")
    p.add_argument("--window-years", type=int, default=10)
    p.add_argument("--lfi-threshold-cm", type=float, default=40.0)
    p.add_argument("--min-size-cm", type=float, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args(argv)
    if args.window_years < 1:
        p.error("--window-years must be >= 1")

    from osmose import size_spectrum as ss

    try:
        spec = ss.compute_size_spectrum(
            args.results_dir,
            metric=args.metric,
            prefix=args.prefix,
            window_years=args.window_years,
            lfi_threshold_cm=args.lfi_threshold_cm,
            min_size_cm=args.min_size_cm,
        )
    except (FileNotFoundError, __import__("pandas").errors.EmptyDataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    report = ss.format_size_spectrum_report(spec)
    if args.report:
        args.report.write_text(report)
    print(report)
    if args.json:
        import math

        # mean_size_cm / peak_size_cm can be NaN on a degenerate all-zero window;
        # json.dumps would emit invalid `NaN`, so map NaN floats -> null.
        payload = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in asdict(spec).items()
        }
        args.json.write_text(json.dumps(payload, indent=2))
    if args.plot:
        from osmose import plotting

        plotting.make_size_spectrum_plot(ss.spectrum_plot_df(spec)).write_html(
            f"{args.plot}_size_spectrum.html"
        )
        ts = ss.size_spectrum_timeseries(
            args.results_dir,
            metric=args.metric,
            prefix=args.prefix,
            lfi_threshold_cm=args.lfi_threshold_cm,
            min_size_cm=args.min_size_cm,
        )
        plotting.make_size_indicator_timeseries(ts).write_html(f"{args.plot}_size_indicators.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
