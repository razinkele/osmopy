"""Scenario management for OSMOSE configurations."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

_log = logging.getLogger("osmose.scenarios")


@dataclass
class Scenario:
    """A named, versioned OSMOSE configuration snapshot."""

    name: str
    description: str = ""
    created_at: str = ""  # ISO format
    modified_at: str = ""  # ISO format
    config: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_scenario: str | None = None

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.modified_at:
            self.modified_at = now


@dataclass
class ParamDiff:
    """A single parameter difference between two scenarios."""

    key: str
    value_a: str | None
    value_b: str | None


class ScenarioManager:
    """Save, load, compare, and fork OSMOSE scenarios."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, name: str) -> Path:
        """Validate a scenario name resolves within storage_dir."""
        if not name or not name.strip():
            raise ValueError(f"Invalid scenario name: {name!r}")
        target = (self.storage_dir / name).resolve()
        if not target.is_relative_to(self.storage_dir.resolve()):
            raise ValueError(f"Unsafe scenario name: {name!r}")
        return target

    def save(self, scenario: Scenario) -> Path:
        """Save a scenario to disk using atomic write pattern."""
        scenario.modified_at = datetime.now().isoformat()
        target = self._validate_path(scenario.name)

        tmp_dir = Path(tempfile.mkdtemp(dir=self.storage_dir))
        data = asdict(scenario)
        try:
            with open(tmp_dir / "scenario.json", "w") as f:
                json.dump(data, f, indent=2)

            backup = None
            if target.exists():
                backup = target.with_suffix(".bak")
                if backup.exists():
                    shutil.rmtree(backup)
                os.rename(target, backup)
            os.rename(tmp_dir, target)
            if backup and backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

        return target

    def load(self, name: str) -> Scenario:
        """Load a named scenario from disk."""
        target = self._validate_path(name)
        path = target / "scenario.json"
        with open(path) as f:
            data = json.load(f)
        return Scenario(**data)

    def list_scenarios(self) -> list[dict[str, str]]:
        """List all saved scenarios with basic metadata."""
        results = []
        for d in sorted(self.storage_dir.iterdir()):
            json_path = d / "scenario.json"
            if d.is_dir() and json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, KeyError) as exc:
                    _log.warning("Corrupt scenario file %s: %s", json_path, exc)
                    continue
                results.append(
                    {
                        "name": data["name"],
                        "description": data.get("description", ""),
                        "modified_at": data.get("modified_at", ""),
                        "tags": data.get("tags", []),
                    }
                )
        return results

    def delete(self, name: str) -> None:
        """Delete a saved scenario."""
        path = self._validate_path(name)
        if path.exists():
            shutil.rmtree(path)

    def compare(self, name_a: str, name_b: str) -> list[ParamDiff]:
        """Compare two scenarios and return parameter differences."""
        a = self.load(name_a)
        b = self.load(name_b)
        all_keys = sorted(set(a.config.keys()) | set(b.config.keys()))
        diffs = []
        for key in all_keys:
            val_a = a.config.get(key)
            val_b = b.config.get(key)
            if val_a != val_b:
                diffs.append(ParamDiff(key=key, value_a=val_a, value_b=val_b))
        return diffs

    def fork(self, source_name: str, new_name: str, description: str = "") -> Scenario:
        """Create a new scenario based on an existing one."""
        self._validate_path(source_name)
        self._validate_path(new_name)
        source = self.load(source_name)
        forked = Scenario(
            name=new_name,
            description=description or f"Forked from {source_name}",
            config=dict(source.config),
            tags=list(source.tags),
            parent_scenario=source_name,
        )
        self.save(forked)
        return forked

    def export_all(self, zip_path: Path) -> None:
        """Export all scenarios to a ZIP file."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for info in self.list_scenarios():
                scenario = self.load(info["name"])
                data = {
                    "name": scenario.name,
                    "description": scenario.description,
                    "config": scenario.config,
                    "tags": scenario.tags,
                    "parent_scenario": scenario.parent_scenario,
                }
                zf.writestr(f"{scenario.name}.json", json.dumps(data, indent=2))

    def import_all(self, zip_path: Path) -> int:
        """Import scenarios from a ZIP file. Returns count of imported scenarios."""
        count = 0
        storage_resolved = self.storage_dir.resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name))
                scenario_name = data["name"]
                # Validate name does not escape storage directory
                target = (self.storage_dir / scenario_name).resolve()
                if not target.is_relative_to(storage_resolved):
                    _log.warning("Skipping scenario with unsafe name: %s", scenario_name)
                    continue
                scenario = Scenario(
                    name=scenario_name,
                    description=data.get("description", ""),
                    config=data.get("config", {}),
                    tags=data.get("tags", []),
                    parent_scenario=data.get("parent_scenario"),
                )
                self.save(scenario)
                count += 1
        return count
