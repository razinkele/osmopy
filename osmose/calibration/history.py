"""Calibration run persistence — save/load/list/delete JSON run files."""

from __future__ import annotations

import json
import json as _json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Resolve relative to the project root (two levels up from osmose/calibration/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_DIR = _PROJECT_ROOT / "data" / "calibration_history"


def save_run(run_data: dict, history_dir: Path = HISTORY_DIR) -> Path:
    """Save a calibration run to JSON. Returns the path written."""
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = run_data["timestamp"].replace(":", "-")
    algo = run_data.get("algorithm", "unknown")
    filename = f"{ts}_{algo}.json"
    path = history_dir / filename
    path.write_text(json.dumps(run_data, indent=2))
    return path


def load_run(path: Path) -> dict:
    """Load a single run from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")
    return json.loads(path.read_text())


def list_runs(history_dir: Path = HISTORY_DIR) -> list[dict]:
    """List all runs, sorted by timestamp descending."""
    if not history_dir.is_dir():
        return []
    entries = []
    for f in history_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entries.append(
            {
                "path": str(f),
                "timestamp": data.get("timestamp", ""),
                "algorithm": data.get("algorithm", "unknown"),
                "best_objective": data.get("results", {}).get("best_objective", float("inf")),
                "n_params": len(data.get("parameters", [])),
                "duration_seconds": data.get("results", {}).get("duration_seconds", 0),
            }
        )
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries


def delete_run(path: Path) -> None:
    """Delete a run file."""
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")
    path.unlink()


def _save_run_safe(
    payload: dict,
    *,
    logger: logging.Logger,
    with_fallback: bool = True,
) -> None:
    """Persist a completed run via save_run(), tolerating expected failures.

    Layered exception scope: catches the expected serialisation/OS failure
    classes plus a defensive `except Exception` catch-all. On failure (if
    with_fallback): writes a fallback JSON to tempfile.gettempdir() with
    mode 0o600, O_EXCL so we don't clobber prior fallbacks. Logs via
    logger.exception in every failure branch.

    History-dir resolution: reads HISTORY_DIR at call time (NOT via save_run's
    default arg, which Python captures at function-definition time). This
    makes test monkeypatching of HISTORY_DIR effective.
    """
    from osmose.calibration import history as hist_mod

    try:
        save_run(payload, history_dir=hist_mod.HISTORY_DIR)
        return
    except (OSError, TypeError, ValueError, OverflowError,
            UnicodeError, RecursionError, MemoryError) as e:
        if with_fallback:
            _save_run_fallback(payload, e, logger)
        else:
            logger.exception("save_run failed (no fallback): %s", e.__class__.__name__)
    except Exception as e:  # noqa: BLE001
        if with_fallback:
            _save_run_fallback(payload, e, logger)
        else:
            logger.exception("save_run failed unexpectedly: %s", e.__class__.__name__)


def _save_run_fallback(payload: dict, e: BaseException, logger: logging.Logger) -> None:
    """Write a fallback JSON when canonical save failed.

    O_EXCL is INTENTIONAL: if a prior fallback exists, do not overwrite — the
    prior file is still the most-likely-recoverable copy. Spec §9 contract.
    """
    logger.exception(
        "save_run failed: %s; payload keys=%s", e.__class__.__name__, list(payload.keys()),
    )
    ts = payload.get("timestamp", datetime.now(timezone.utc).isoformat()).replace(":", "-")
    raw_algo = str(payload.get("algorithm", "unknown"))
    algo_sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", raw_algo)[:32]
    path = Path(tempfile.gettempdir()) / f"calibration_history_fallback_{ts}_{algo_sanitized}.json"
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            _json.dump(payload, f, indent=2, allow_nan=False, default=str)
    except (OSError, TypeError, ValueError) as fb_e:
        logger.exception("fallback write also failed: %s; path=%s", fb_e.__class__.__name__, path)
