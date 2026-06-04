#!/usr/bin/env python3
"""Report structured parse diagnostics for an OSMOSE config master file.

Reads the config (recursively) and prints line-located issues: unparseable lines,
empty keys, within-file duplicate keys, and recursive-reference problems (circular /
missing sub-config / path-escape). Exits 1 only when an ERROR-class issue is present
(unparseable / circular_ref / missing_subconfig / path_escape); empty-key and
duplicate-key warnings print but exit 0. For config MASTER files, not data/map CSVs.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/check_config.py --config <master.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args(argv)
    if not args.config.is_file():
        p.error(f"config file not found: {args.config}")

    from osmose.config.reader import (
        OsmoseConfigReader,
        diagnostics_have_errors,
        format_diagnostics,
    )

    reader = OsmoseConfigReader()
    reader.read(args.config)
    print(format_diagnostics(reader.diagnostics))
    return 1 if diagnostics_have_errors(reader.diagnostics) else 0


if __name__ == "__main__":
    sys.exit(main())
