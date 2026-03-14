"""Run history tracking for OSMOSE simulations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.history")


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
            try:
                with open(path) as f:
                    data = json.load(f)
                records.append(RunRecord(**data))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                _log.warning("Corrupt history file %s: %s", path, exc)
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    def load_run(self, timestamp: str) -> RunRecord:
        """Load a specific run record by timestamp."""
        safe_ts = timestamp.replace(":", "-")
        raw = Path(f"run_{safe_ts}.json")
        if any(part == ".." for part in raw.parts) or raw.is_absolute():
            raise ValueError(f"Unsafe timestamp: {timestamp!r}")
        path = (self.history_dir / raw).resolve()
        if not path.is_relative_to(self.history_dir.resolve()):
            raise ValueError(f"Unsafe timestamp: {timestamp!r}")
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

    def compare_runs_multi(self, timestamps: list[str]) -> list[dict]:
        """Compare N runs by config snapshot, return parameters that differ.

        Args:
            timestamps: List of run timestamps to compare.

        Returns:
            List of {"key": str, "values": list[str | None]} for each
            parameter that differs across any of the selected runs.
        """
        if len(timestamps) < 2:
            return []

        records = [self.load_run(ts) for ts in timestamps]
        all_keys: set[str] = set()
        for r in records:
            all_keys |= set(r.config_snapshot)

        diffs = []
        for key in sorted(all_keys):
            values = [r.config_snapshot.get(key) for r in records]
            if len(set(values)) > 1:
                diffs.append({"key": key, "values": values})

        return diffs
