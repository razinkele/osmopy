#!/usr/bin/env python3
"""Validate OSMOSE model outputs against ICES SSB envelopes.

Reads a finished simulation's output directory, computes the mean
species biomass over the last N years, and compares against the ICES
Stock Assessment Graph (SAG) envelope loaded from frozen JSON snapshots.
Reports per-species in-range / out-of-range with magnitude factor.

Distinguishes from `validate_baltic_vs_ices_sag.py` which validates the
**config** (fishing rates + a hand-curated targets CSV); this script
validates the **post-run model output** — what the simulation actually
produced.

Usage:
    .venv/bin/python scripts/validate_outputs_vs_ices.py \\
        --results-dir <path> \\
        [--snapshots-dir data/baltic/reference/ices_snapshots] \\
        [--window-years 5] \\
        [--ices-window 2018-2022] \\
        [--prefix osm] \\
        [--report path/to/output.md] \\
        [--json path/to/output.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "baltic" / "reference" / "ices_snapshots"


def _parse_ices_window(spec: str) -> range:
    """Parse '2018-2022' or '2018' into a range. Inclusive endpoints."""
    if "-" in spec:
        a, b = spec.split("-", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(spec)
    if end < start:
        raise ValueError(f"ICES window {spec!r}: end {end} < start {start}")
    return range(start, end + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="OSMOSE output directory (containing biomass-*.csv etc.)",
    )
    parser.add_argument(
        "--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR,
        help=f"ICES snapshot directory (default: {DEFAULT_SNAPSHOT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--window-years", type=int, default=5,
        help="Trailing simulation years to average for the model mean (default: 5)",
    )
    parser.add_argument(
        "--ices-window", type=_parse_ices_window, default=range(2018, 2023),
        metavar="YYYY-YYYY",
        help="ICES SAG year range to compute envelope over (default: 2018-2022)",
    )
    parser.add_argument(
        "--prefix", default="osm",
        help="OSMOSE output filename prefix (default: osm)",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Write markdown report to PATH (default: stdout-only)",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Write JSON results to PATH (default: stdout-only)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress markdown output to stdout",
    )
    args = parser.parse_args(argv)

    # Defer imports so --help works without OSMOSE being installed.
    from osmose.results import OsmoseResults
    from osmose.validation.ices import (
        compare_outputs_to_ices,
        format_markdown_report,
        load_snapshot,
    )

    if not args.results_dir.is_dir():
        print(f"ERROR: --results-dir {args.results_dir!r} is not a directory", file=sys.stderr)
        return 2
    if not args.snapshots_dir.is_dir():
        print(f"ERROR: --snapshots-dir {args.snapshots_dir!r} is not a directory", file=sys.stderr)
        return 2

    snapshot = load_snapshot(args.snapshots_dir)
    with OsmoseResults(args.results_dir, prefix=args.prefix, strict=False) as results:
        comparisons = compare_outputs_to_ices(
            results,
            snapshot,
            window_years=args.window_years,
            ices_window=args.ices_window,
        )

    md = format_markdown_report(
        comparisons,
        snapshot_dir=args.snapshots_dir,
        window_years=args.window_years,
        ices_window=args.ices_window,
    )
    if not args.quiet:
        print(md)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md)
        print(f"Wrote markdown report: {args.report}", file=sys.stderr)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshot_dir": str(args.snapshots_dir),
            "results_dir": str(args.results_dir),
            "window_years": args.window_years,
            "ices_window_start": args.ices_window.start,
            "ices_window_end": args.ices_window.stop - 1,
            "comparisons": [asdict(c) for c in comparisons],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON report: {args.json}", file=sys.stderr)

    n_with_envelope = sum(1 for c in comparisons if c.in_range is not None)
    if n_with_envelope == 0:
        return 1  # no comparable species — likely a config or snapshot mismatch
    return 0


if __name__ == "__main__":
    sys.exit(main())
