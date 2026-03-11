"""Run history tracking for OSMOSE simulations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class RunRecord:
    """A single OSMOSE run record."""

    timestamp: str = ""
    config_snapshot: dict[str, str] = field(default_factory=dict)
    duration_sec: float = 0.0
    output_dir: str = ""
    summary: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RunHistory:
    """Manage run history as JSON records."""

    def __init__(self, history_dir: Path) -> None:
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: RunRecord) -> Path:
        """Save a run record to a JSON file."""
        safe_ts = record.timestamp.replace(":", "-")
        path = self.history_dir / f"run_{safe_ts}.json"
        with open(path, "w") as f:
            json.dump(asdict(record), f, indent=2)
        return path

    def list_runs(self) -> list[RunRecord]:
        """List all run records, sorted newest first."""
        records = []
        for path in self.history_dir.glob("run_*.json"):
            with open(path) as f:
                data = json.load(f)
            records.append(RunRecord(**data))
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    def load_run(self, timestamp: str) -> RunRecord:
        """Load a specific run record by timestamp."""
        safe_ts = timestamp.replace(":", "-")
        path = self.history_dir / f"run_{safe_ts}.json"
        with open(path) as f:
            data = json.load(f)
        return RunRecord(**data)

    def compare_runs(self, ts_a: str, ts_b: str) -> list[dict]:
        """Compare two runs by config snapshot, return differences."""
        a = self.load_run(ts_a)
        b = self.load_run(ts_b)
        all_keys = sorted(set(a.config_snapshot) | set(b.config_snapshot))
        diffs = []
        for key in all_keys:
            va = a.config_snapshot.get(key)
            vb = b.config_snapshot.get(key)
            if va != vb:
                diffs.append({"key": key, "value_a": va, "value_b": vb})
        return diffs
