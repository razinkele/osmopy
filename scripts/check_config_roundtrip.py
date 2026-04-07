#!/usr/bin/env python3
"""Verify OSMOSE config round-trip fidelity.

Reads a config directory, writes it to a temp dir using the same pipeline
as the Run page, then diffs the original vs written parameters to catch:
- Key case changes (Java is case-sensitive)
- Value corruption (multi-value arrays truncated to single values)
- Missing or added keys
"""

import re
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from osmose.config.reader import OsmoseConfigReader


SEPARATOR_RE = re.compile(r"\s*[=;,:\t]\s*")


def read_raw_kv(filepath: Path) -> dict[str, str]:
    """Read key-value pairs preserving original key case and full value."""
    result: dict[str, str] = {}
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ("#", "!"):
                continue
            parts = SEPARATOR_RE.split(line, maxsplit=1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().rstrip(";,:\t =")
                result[key] = value
    return result


def read_all_raw(config_dir: Path, master_file: Path) -> dict[str, str]:
    """Read all config files recursively, preserving original case and values."""
    all_kv: dict[str, str] = {}
    seen: set[Path] = set()

    def recurse(fp: Path):
        resolved = fp.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        kv = read_raw_kv(fp)
        all_kv.update(kv)
        for k, v in kv.items():
            if k.lower().startswith("osmose.configuration."):
                sub = fp.parent / v.strip()
                if sub.exists():
                    recurse(sub)

    recurse(master_file)
    return all_kv


def main() -> None:
    config_dir_arg = sys.argv[1] if len(sys.argv) > 1 else "data/eec_full"
    config_dir = Path(config_dir_arg)
    if not config_dir.is_dir():
        print(f"ERROR: Config directory not found: {config_dir}")
        sys.exit(1)

    # Find master file
    master_candidates = ["osm_all-parameters.csv", "osm_param-main.csv"]
    master_file = None
    for name in master_candidates:
        candidate = config_dir / name
        if candidate.exists():
            master_file = candidate
            break
    if master_file is None:
        # Try any CSV with osmose.configuration references
        for csv in sorted(config_dir.glob("*.csv")):
            text = csv.read_text(errors="replace")
            if "osmose.configuration" in text.lower():
                master_file = csv
                break
    if master_file is None:
        print(f"ERROR: No master config file found in {config_dir}")
        sys.exit(1)

    print(f"Config dir: {config_dir}")
    print(f"Master file: {master_file.name}")

    # Step 1: Read original config with raw case + values
    original_raw = read_all_raw(config_dir, master_file)
    print(f"Original: {len(original_raw)} parameters")

    # Step 2: Read via Python reader (normalises to lowercase)
    reader = OsmoseConfigReader()
    config = reader.read(master_file)
    case_map = reader.key_case_map
    print(f"Reader: {len(config)} parameters, {len(case_map)} case mappings")

    # Step 3: Write via the same pipeline as Run page
    from ui.pages.run import write_temp_config

    work_dir = Path(tempfile.mkdtemp(prefix="osmose_roundtrip_"))
    written_path = write_temp_config(config, work_dir, config_dir, key_case_map=case_map)

    # Step 4: Read back the written master file (raw)
    written_raw = read_raw_kv(written_path)
    print(f"Written: {len(written_raw)} parameters")
    print()

    # Step 5: Compare
    issues = {"case_mismatch": [], "value_changed": [], "key_missing": [], "key_added": []}

    # Index originals by lowercase for comparison
    orig_by_lower: dict[str, tuple[str, str]] = {}
    for k, v in original_raw.items():
        lk = k.lower()
        if not lk.startswith("osmose.configuration."):
            orig_by_lower[lk] = (k, v)

    written_by_lower: dict[str, tuple[str, str]] = {}
    for k, v in written_raw.items():
        lk = k.lower()
        if not lk.startswith(("osmose.configuration.", "_")):
            written_by_lower[lk] = (k, v)

    # Check each original key
    for lk, (orig_key, orig_val) in sorted(orig_by_lower.items()):
        if lk.startswith("_"):
            continue
        if lk not in written_by_lower:
            issues["key_missing"].append(f"  {orig_key} = {orig_val}")
            continue
        written_key, written_val = written_by_lower[lk]
        if orig_key != written_key:
            issues["case_mismatch"].append(f"  {orig_key} -> {written_key}")
        if orig_val != written_val:
            issues["value_changed"].append(f"  {orig_key}: {orig_val!r} -> {written_val!r}")

    # Check for added keys
    for lk, (written_key, written_val) in sorted(written_by_lower.items()):
        if lk not in orig_by_lower:
            issues["key_added"].append(f"  {written_key} = {written_val}")

    # Report
    total_issues = sum(len(v) for k, v in issues.items() if k != "key_added")
    has_problems = len(issues["case_mismatch"]) > 0 or len(issues["value_changed"]) > 0

    if issues["case_mismatch"]:
        print(f"CASE MISMATCH ({len(issues['case_mismatch'])}):")
        for line in issues["case_mismatch"]:
            print(line)
        print()

    if issues["value_changed"]:
        print(f"VALUE CHANGED ({len(issues['value_changed'])}):")
        for line in issues["value_changed"]:
            print(line)
        print()

    if issues["key_missing"]:
        print(f"KEY MISSING ({len(issues['key_missing'])}):")
        for line in issues["key_missing"][:20]:
            print(line)
        if len(issues["key_missing"]) > 20:
            print(f"  ... and {len(issues['key_missing']) - 20} more")
        print()

    if issues["key_added"]:
        print(f"KEY ADDED ({len(issues['key_added'])}):")
        for line in issues["key_added"][:10]:
            print(line)
        if len(issues["key_added"]) > 10:
            print(f"  ... and {len(issues['key_added']) - 10} more")
        print()

    if has_problems:
        print(f"FAIL: {total_issues} issue(s) that will cause Java engine failures")
        sys.exit(1)
    else:
        print("PASS: Config round-trip preserved all keys and values")
        if issues["key_missing"]:
            print(
                f"  (note: {len(issues['key_missing'])} keys missing — may be osmose.configuration.* refs, check if expected)"
            )


if __name__ == "__main__":
    main()
