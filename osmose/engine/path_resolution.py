"""Consolidated file path resolution for OSMOSE data files.

All engine modules that resolve relative data file paths (config, background,
resources, movement maps) use this single implementation. Security: rejects
paths containing '..' directory traversal segments.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.path")


def resolve_data_path(
    file_key: str,
    config_dir: str = "",
) -> Path | None:
    """Resolve a relative file path against standard search directories.

    Search order:
      1. As-is (works for absolute paths under config_dir)
      2. Relative to config_dir
      3. Relative to CWD
      4. Relative to data/examples/
      5. Relative to data/*/

    Returns None if the file is not found or the path is rejected.
    Rejects paths containing '..' segments to prevent directory traversal.
    """
    if not file_key:
        return None

    p = Path(file_key)

    if ".." in p.parts:
        _log.warning("Rejecting file key with '..' traversal: %s", file_key)
        return None

    if p.is_absolute():
        if p.exists():
            return p
        return None

    for base in _build_search_dirs(config_dir):
        candidate = base / file_key
        if candidate.exists():
            return candidate

    return None


def _build_search_dirs(config_dir: str = "") -> list[Path]:
    """Build ordered list of directories to search for data files."""
    dirs: list[Path] = []
    if config_dir:
        dirs.append(Path(config_dir))
    dirs.append(Path("."))
    dirs.append(Path("data/examples"))
    dirs.extend(Path(d) for d in sorted(_glob.glob("data/*/")))
    return dirs
