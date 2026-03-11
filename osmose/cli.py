"""OSMOSE command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate an OSMOSE config file."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    from osmose.config.reader import OsmoseConfigReader
    from osmose.config.validator import (
        validate_config,
        check_file_references,
        check_species_consistency,
    )
    from osmose.schema import build_registry

    reader = OsmoseConfigReader()
    config = reader.read(config_path)
    registry = build_registry()

    errors, warnings = validate_config(config, registry)
    file_errors = check_file_references(config, str(config_path.parent))
    errors.extend(file_errors)
    species_warnings = check_species_consistency(config)
    warnings.extend(species_warnings)

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"Valid. {len(warnings)} warning(s)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run an OSMOSE simulation."""
    import asyncio
    from osmose.runner import OsmoseRunner

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    jar_path = Path(args.jar)
    if not jar_path.exists():
        print(f"Error: JAR not found: {jar_path}", file=sys.stderr)
        return 1

    runner = OsmoseRunner(jar_path=jar_path)
    output_dir = Path(args.output) if args.output else None
    java_opts = args.java_opts.split() if args.java_opts else None
    timeout = args.timeout

    result = asyncio.run(
        runner.run(
            config_path=config_path,
            output_dir=output_dir,
            java_opts=java_opts,
            timeout_sec=timeout,
            on_progress=lambda line: print(line),
        )
    )

    if result.returncode == 0:
        print(f"\nComplete. Output: {result.output_dir}")
    else:
        print(f"\nFailed (exit code {result.returncode})", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    return result.returncode


def cmd_report(args: argparse.Namespace) -> int:
    """Generate an HTML report from OSMOSE output."""
    from osmose.results import OsmoseResults
    from osmose.reporting import generate_report

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        return 1

    results = OsmoseResults(output_dir)
    config = {}  # Could load from output_dir if available
    report_path = Path(args.report_path) if args.report_path else output_dir / "report.html"

    generate_report(results, config, report_path)
    print(f"Report written to {report_path}")
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="osmose",
        description="OSMOSE marine ecosystem simulator CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate config file")
    p_validate.add_argument("config", help="Path to config CSV file")

    # run
    p_run = subparsers.add_parser("run", help="Run simulation")
    p_run.add_argument("config", help="Path to config CSV file")
    p_run.add_argument("--jar", required=True, help="Path to OSMOSE JAR")
    p_run.add_argument("--output", help="Output directory")
    p_run.add_argument("--java-opts", help="Java options (e.g. '-Xmx4g')")
    p_run.add_argument("--timeout", type=int, help="Timeout in seconds")

    # report
    p_report = subparsers.add_parser("report", help="Generate HTML report")
    p_report.add_argument("output_dir", help="OSMOSE output directory")
    p_report.add_argument("--report-path", help="Output report file path")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    handlers = {"validate": cmd_validate, "run": cmd_run, "report": cmd_report}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
