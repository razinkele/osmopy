"""Temporary directory management for osmose-python.

osmose_* temp directories are created during runs, calibration, export, and
demo loading.  Without explicit cleanup they accumulate over time, especially
during calibration with hundreds of runs.  This module provides:

- ``cleanup_old_temp_dirs()`` — removes osmose temp dirs older than a given age
- ``register_cleanup()``      — registers an atexit handler that cleans on exit
"""

import atexit
import logging
import shutil
import tempfile
import time
from pathlib import Path

_log = logging.getLogger("osmose.cleanup")

_OSMOSE_PREFIXES = (
    "osmose_run_",
    "osmose_cal_",
    "osmose_sens_",
    "osmose_export_",
    "osmose_demo_",
)
_MAX_AGE_HOURS = 24


def cleanup_old_temp_dirs(max_age_hours: int = _MAX_AGE_HOURS) -> int:
    """Remove osmose temp directories older than *max_age_hours*.

    Iterates over the system temp directory and removes any subdirectory whose
    name starts with a known osmose prefix and whose last-modified time is
    older than the specified threshold.

    Args:
        max_age_hours: Age threshold in hours.  Directories older than this
            are deleted.  Pass ``0`` to remove all osmose temp dirs regardless
            of age (useful for shutdown cleanup).

    Returns:
        Number of directories removed.
    """
    tmp_root = Path(tempfile.gettempdir())
    now = time.time()
    max_age_secs = max_age_hours * 3600
    removed = 0

    try:
        entries = list(tmp_root.iterdir())
    except OSError as exc:
        _log.warning("Cannot list temp directory %s: %s", tmp_root, exc)
        return 0

    for entry in entries:
        if not entry.is_dir():
            continue
        if not any(entry.name.startswith(p) for p in _OSMOSE_PREFIXES):
            continue
        try:
            age = now - entry.stat().st_mtime
            if age > max_age_secs:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            pass

    if removed:
        _log.info(
            "Cleaned up %d old osmose temp director%s", removed, "y" if removed == 1 else "ies"
        )
    return removed


def register_cleanup() -> None:
    """Register an atexit handler that removes all osmose temp dirs on shutdown.

    Calling this at startup ensures that temp directories created during the
    current session are removed when the process exits normally.
    """
    atexit.register(cleanup_old_temp_dirs, max_age_hours=0)
