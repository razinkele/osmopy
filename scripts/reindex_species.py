#!/usr/bin/env python
"""Re-index OSMOSE species keys when inserting a focal species.

OSMOSE indexes focal, LTL (resource), and background species contiguously in a
single ``species.name.sp{idx}`` namespace (Baltic: focal 0-7, LTL 8-13,
background 14-15). Adding a focal species therefore requires shifting the
indices of every species above the insertion point. This tool relabels every
``.sp{old}`` / ``.fsh{old}`` config KEY to its new index per a shift map,
across all CSVs in a config dir.

Key facts that make this safe and simple for the Baltic append case:

* The predation-accessibility matrix and ``movement.species.mapN`` refs are
  keyed by species NAME, not index (see ``osmose/engine/accessibility.py`` —
  ``AccessibilityMatrix.from_csv`` resolves ``sp_idx -> matrix row`` by name).
  So an index shift does NOT touch them; renaming (cod -> cod_west) and adding
  the new species' row/column is a separate step.
* Values may themselves contain sp-tokens (e.g.
  ``reproduction.season.file.sp0;reproduction/season-sp0.csv``). Only the KEY
  side of each line is rewritten; values, comments, blanks, and per-line
  separators are preserved verbatim.

The ``append_focal_species`` convenience computes the shift for appending one
focal species (shift every index >= nspecies up by one, bump nspecies).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Separators the OSMOSE config reader auto-detects, per line.
_SEPARATORS = ";=,:\t"
# A trailing sp/fsh index token in a KEY: preceded by a word boundary (a '.'),
# an sp/fsh prefix, and a run of digits that ends the token. `species` never
# matches (no digits follow "sp"); values are never seen (key-side only).
_INDEX_RE = re.compile(r"\b(sp|fsh)(\d+)\b")


def _split_key(line: str) -> tuple[str, str] | None:
    """Split a data line into (key, tail) at the first separator.

    ``tail`` retains the separator and everything after it, so the value —
    including any sp-tokens or trailing whitespace — round-trips unchanged.
    Returns None for blank lines and comments (leave them untouched).
    """
    if not line.strip() or line.lstrip().startswith("#"):
        return None
    positions = [line.find(sep) for sep in _SEPARATORS]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return None
    cut = min(positions)
    return line[:cut], line[cut:]


def _rewrite_key(key: str, shifts: dict[int, int]) -> str:
    """Rewrite every ``sp{n}`` / ``fsh{n}`` token in a key per ``shifts``."""

    def repl(m: re.Match) -> str:
        idx = int(m.group(2))
        return f"{m.group(1)}{shifts.get(idx, idx)}"

    return _INDEX_RE.sub(repl, key)


def _set_value(lines: list[str], key: str, value: object) -> list[str]:
    """Replace the value of an exact ``key`` (separator-preserving)."""
    out = []
    for line in lines:
        split = _split_key(line)
        if split is not None and split[0].strip().lower() == key.lower():
            head, tail = split
            # tail is e.g. ";8" or " = 2"; keep the separator + spacing, swap value
            m = re.match(r"^(\s*[" + re.escape(_SEPARATORS) + r"]\s*)(.*)$", tail)
            sep = m.group(1) if m else tail[:1]
            out.append(f"{head}{sep}{value}")
        else:
            out.append(line)
    return out


def reindex(
    config_dir: str | Path,
    shifts: dict[int, int],
    *,
    new_nspecies: int | None = None,
) -> None:
    """Relabel ``.sp{old}``/``.fsh{old}`` keys to ``shifts[old]`` in place.

    Rewrites every ``*.csv`` under ``config_dir``. Only the key side of each
    line is rewritten; values, comments, blank lines, and separators are
    preserved. If ``new_nspecies`` is given, ``simulation.nspecies`` is set to
    it (in whichever file defines it).

    ``shifts`` must be injective (each old index maps to one new index) — the
    append case ``{i: i+1}`` is. Lines are mapped independently to a fresh
    output, so an ascending shift never collides with a not-yet-shifted key.
    """
    config_dir = Path(config_dir)
    for path in sorted(config_dir.glob("*.csv")):
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        out = []
        for line in lines:
            split = _split_key(line)
            if split is None:
                out.append(line)
                continue
            head, tail = split
            new_head = _rewrite_key(head, shifts)
            out.append(new_head + tail)
            changed = changed or new_head != head
        if new_nspecies is not None:
            new_out = _set_value(out, "simulation.nspecies", new_nspecies)
            changed = changed or new_out != out
            out = new_out
        if changed:
            path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _read_int_key(config_dir: Path, key: str) -> int | None:
    for path in config_dir.glob("*.csv"):
        for line in path.read_text(encoding="utf-8").splitlines():
            split = _split_key(line)
            if split is not None and split[0].strip().lower() == key.lower():
                val = split[1].lstrip("".join(_SEPARATORS)).strip()
                try:
                    return int(val)
                except ValueError:
                    return None
    return None


def append_focal_species(config_dir: str | Path) -> dict[int, int]:
    """Shift every index >= nspecies up by one and bump nspecies.

    This frees ``sp{nspecies}`` for a newly-appended focal species while
    pushing LTL and background species up one slot each. Returns the shift map
    applied (for logging / tests).
    """
    config_dir = Path(config_dir)
    nspecies = _read_int_key(config_dir, "simulation.nspecies")
    nresource = _read_int_key(config_dir, "simulation.nresource") or 0
    nbackground = _read_int_key(config_dir, "simulation.nbackground") or 0
    if nspecies is None:
        raise ValueError("simulation.nspecies not found in config")
    total = nspecies + nresource + nbackground
    shifts = {i: i + 1 for i in range(nspecies, total)}
    reindex(config_dir, shifts, new_nspecies=nspecies + 1)
    return shifts


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-index OSMOSE species keys")
    ap.add_argument("config_dir", type=Path)
    ap.add_argument(
        "--append-focal",
        action="store_true",
        help="Shift all species >= nspecies up one and bump nspecies (frees sp{nspecies}).",
    )
    ap.add_argument(
        "--shifts",
        help="Explicit map, e.g. '8:9,9:10,10:11'. Mutually exclusive with --append-focal.",
    )
    ap.add_argument("--new-nspecies", type=int, default=None)
    args = ap.parse_args()

    if args.append_focal:
        shifts = append_focal_species(args.config_dir)
        print(f"appended focal species; shifts={shifts}")
    elif args.shifts:
        shifts = {}
        for pair in args.shifts.split(","):
            old, new = pair.split(":")
            shifts[int(old)] = int(new)
        reindex(args.config_dir, shifts, new_nspecies=args.new_nspecies)
        print(f"reindexed; shifts={shifts}, new_nspecies={args.new_nspecies}")
    else:
        ap.error("one of --append-focal or --shifts is required")


if __name__ == "__main__":
    main()
