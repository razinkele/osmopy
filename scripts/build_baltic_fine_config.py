"""Assemble the Baltic fine-grid (4x resolution) OSMOSE config tree.

Copies every ``osmose.configuration.*``-referenced ``baltic_param-*.csv`` from
the coarse master aggregator (``data/baltic/baltic_all-parameters.csv``) into
``data/baltic-fine/``, editing only the resolution-dependent keys (grid
dimensions/mask/netcdf).

Two entrypoints are written, differing ONLY in the 6 percid
``movement.file.map{13..18}`` values (perch, pike-perch):
    data/baltic-fine/baltic_fine_upsampled_all-parameters.csv
    data/baltic-fine/baltic_fine_real_all-parameters.csv

THE C1 FIX
----------
The config reader's ``_read_recursive`` applies ``flat.update()`` per included
sub-file *after* the including file's own keys have already been merged --  so
an included sub-file always overwrites the same key set inline by its parent.
Naively inline-overriding ``movement.file.map13..18`` in the entrypoint while
still including the *unedited* ``baltic_param-movement.csv`` (which also
defines those same keys) would therefore be silently clobbered by the
include -- both variants would resolve identically (rung2 == rung3).

The fix applied here: the shared ``baltic_param-movement.csv`` written to
``data/baltic-fine/`` has the 6 percid ``movement.file.mapN`` lines removed
(their sibling keys -- species/initialage/lastage/steps -- are untouched,
since those never differ between variants and stay in the shared file). Each
entrypoint then sets those 6 keys directly, inline. Because the shared
sub-file no longer defines them, the include-recursion cannot clobber the
entrypoint's inline value -- so ``osmose.configuration.movement`` itself
stays byte-identical between the two entrypoints (pointing at the same
shared file), and the *only* resolved-config diff between the two variants
is exactly the 6 percid map keys.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "baltic"
OUT = ROOT / "data" / "baltic-fine"
MASTER = SRC / "baltic_all-parameters.csv"

MOVEMENT_KEY = "osmose.configuration.movement"

# perch: map13-15 (juvenile/adult/spawning); pikeperch: map16-18
PERCID_MAP_INDICES = frozenset(range(13, 19))

# grid.csv text edits: 50x40 coarse -> 200x160 fine, + fine mask/netcdf files.
GRID_EDITS = {
    "grid.nlon;50": "grid.nlon;200",
    "grid.nlat;40": "grid.nlat;160",
    "grid.mask.file;grid/baltic_mask.csv": "grid.mask.file;grid/baltic_fine_mask.csv",
    "grid.netcdf.file;baltic_grid.nc": "grid.netcdf.file;baltic_fine_grid.nc",
}

_MOVEMENT_MAP_RE = re.compile(r"^(movement\.file\.map(\d+));(maps/\S+)\.csv$")


def _parse_master(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Split the master aggregator into (includes, master_only_lines).

    includes: ``osmose.configuration.*`` (key, sub_filename) pairs, in file
    order. master_only_lines: every other non-blank, non-comment line
    verbatim (e.g. ``mortality.subdt``, ``simulation.nschool.sp*``,
    ``osmose.version``) -- these must be preserved in both entrypoints.
    """
    includes: list[tuple[str, str]] = []
    master_only: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#!":
            continue
        key, sep, value = line.partition(";")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key.startswith("osmose.configuration."):
            includes.append((key, value))
        else:
            master_only.append(line)
    return includes, master_only


def _edit_grid(text: str) -> str:
    for old, new in GRID_EDITS.items():
        if old not in text:
            raise ValueError(f"expected grid key {old!r} not found in source grid config")
        text = text.replace(old, new)
    return text


def _strip_percid_file_lines(text: str) -> tuple[str, dict[int, str]]:
    """Remove the 6 percid ``movement.file.mapN`` lines from the movement config.

    Returns (stripped_text, {map_index: "maps/<name>"} without the trailing
    ".csv"). The sibling keys for those maps (species/initialage/lastage/
    steps) are left untouched -- they're identical across variants. The
    caller re-adds the file key per-variant as an inline entrypoint key (see
    module docstring: this is what makes the C1 fix work).
    """
    real_paths: dict[int, str] = {}
    out_lines: list[str] = []
    for raw in text.splitlines():
        m = _MOVEMENT_MAP_RE.match(raw.strip())
        if m and int(m.group(2)) in PERCID_MAP_INDICES:
            real_paths[int(m.group(2))] = m.group(3)
            continue
        out_lines.append(raw)
    if set(real_paths) != PERCID_MAP_INDICES:
        missing = PERCID_MAP_INDICES - set(real_paths)
        raise ValueError(f"movement config missing expected percid map keys: {sorted(missing)}")
    return "\n".join(out_lines) + "\n", real_paths


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    includes, master_only_lines = _parse_master(MASTER.read_text())

    # Copy every include, editing grid keys and stripping the percid movement
    # file lines (see module docstring) where relevant.
    shared_includes: list[tuple[str, str]] = []
    percid_real_paths: dict[int, str] = {}
    for key, filename in includes:
        text = (SRC / filename).read_text()
        if filename == "baltic_param-grid.csv":
            text = _edit_grid(text)
        elif key == MOVEMENT_KEY:
            text, percid_real_paths = _strip_percid_file_lines(text)
        (OUT / filename).write_text(text)
        shared_includes.append((key, filename))
        print(f"wrote {OUT / filename}")

    percid_inline_lines = {
        "real": [
            f"movement.file.map{i};{percid_real_paths[i]}.csv" for i in sorted(percid_real_paths)
        ],
        "upsampled": [
            f"movement.file.map{i};{percid_real_paths[i]}_upsampled.csv"
            for i in sorted(percid_real_paths)
        ],
    }

    # Two entrypoints: identical includes + master-only keys, differing only
    # in the 6 inline percid movement.file.map lines.
    for variant, inline_lines in percid_inline_lines.items():
        lines = [
            "# Baltic Sea OSMOSE configuration - fine grid (4x, 200x160)",
            f"# Percid (perch, pike-perch) movement maps: {variant}",
            "# Generated by scripts/build_baltic_fine_config.py -- do not hand-edit.",
            "",
        ]
        for key, filename in shared_includes:
            lines.append(f"{key};{filename}")
        lines.append("")
        lines.extend(inline_lines)
        lines.append("")
        lines.extend(master_only_lines)
        lines.append("")
        out_path = OUT / f"baltic_fine_{variant}_all-parameters.csv"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
