"""Calibration run persistence — save/load/list/delete JSON run files."""

from __future__ import annotations

import json
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
