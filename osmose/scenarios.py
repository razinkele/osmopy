"""Scenario management for OSMOSE configurations."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.scenarios")


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
    key_case_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Scenario name must not be empty")
        if (
            "/" in self.name
            or "\\" in self.name
            or ".." in self.name
            or self.name.strip()
            in (
                ".",
                "..",
            )
        ):
            raise ValueError(f"Scenario name contains invalid characters: {self.name!r}")
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
        storage = self.storage_dir.resolve()
        # target == storage means a reserved/relative name like "." resolved to the
        # store root itself (is_relative_to is True for equal paths) — reject it so a
        # save can't rename/clobber the whole store.
        if target == storage or not target.is_relative_to(storage):
            raise ValueError(f"Unsafe scenario name: {name!r}")
        return target

    def save(self, scenario: Scenario, preserve_modified_at: bool = False) -> Path:
        """Save a scenario to disk using atomic write pattern."""
        if not preserve_modified_at:
            scenario.modified_at = datetime.now().isoformat()
        target = self._validate_path(scenario.name)

        tmp_dir = Path(tempfile.mkdtemp(dir=self.storage_dir))
        data = asdict(scenario)
        backup = None
        try:
            with open(tmp_dir / "scenario.json", "w") as f:
                json.dump(data, f, indent=2)

            if target.exists():
                # Append '.bak' (do NOT use with_suffix, which REPLACES the last
                # dotted segment: a scenario named 'v1.2' would back up to 'v1.bak'
                # and rmtree an unrelated scenario literally named 'v1.bak').
                backup = target.parent / (target.name + ".bak")
                if backup.exists():
                    shutil.rmtree(backup)
                os.rename(target, backup)
            os.rename(tmp_dir, target)
            if backup and backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            if backup is not None and backup.exists() and not target.exists():
                backup.rename(target)
            raise

        return target

    def load(self, name: str) -> Scenario:
        """Load a named scenario from disk."""
        target = self._validate_path(name)
        path = target / "scenario.json"
        with open(path) as f:
            data = json.load(f)
        from osmose.config.aliases import canonicalize_config

        config, _ = canonicalize_config(data.get("config", {}))
        data["config"] = config
        # Filter to known dataclass fields so an extra / forward-version key in the
        # JSON doesn't make the scenario permanently unloadable (TypeError on **data).
        known = {f.name for f in dataclasses.fields(Scenario)}
        return Scenario(**{k: v for k, v in data.items() if k in known})

    def list_scenarios(self) -> list[dict[str, str]]:
        """List all saved scenarios with basic metadata."""
        results = []
        for d in sorted(self.storage_dir.iterdir()):
            json_path = d / "scenario.json"
            if d.is_dir() and json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as exc:
                    _log.warning("Skipping corrupt scenario file %s: %s", json_path, exc)
                    continue
                name = data.get("name")
                if not name:
                    _log.warning("Skipping scenario file with no 'name': %s", json_path)
                    continue
                results.append(
                    {
                        "name": name,
                        "description": data.get("description", ""),
                        "modified_at": data.get("modified_at", ""),
                        "tags": data.get("tags", []),
                    }
                )
        return results

    def delete(self, name: str) -> None:
        """Delete a saved scenario."""
        path = self._validate_path(name)
        if not path.exists():
            _log.warning("Cannot delete scenario %r: path does not exist (%s)", name, path)
            return
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
            key_case_map=dict(source.key_case_map),
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
                    "created_at": scenario.created_at,
                    "modified_at": scenario.modified_at,
                    "config": scenario.config,
                    "tags": scenario.tags,
                    "parent_scenario": scenario.parent_scenario,
                    "key_case_map": scenario.key_case_map,
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
                info = zf.getinfo(name)
                if info.file_size > 10 * 1024 * 1024:
                    _log.warning(
                        "Skipping oversized ZIP entry: %s (%d bytes)",
                        name,
                        info.file_size,
                    )
                    continue
                scenario_name = None
                try:
                    data = json.loads(zf.read(name))
                    scenario_name = data.get("name")
                    if not scenario_name:
                        _log.warning("Skipping ZIP entry with no 'name': %s", name)
                        continue
                    # Validate name does not escape storage directory
                    target = (self.storage_dir / scenario_name).resolve()
                    if target == storage_resolved or not target.is_relative_to(storage_resolved):
                        _log.warning("Skipping scenario with unsafe name: %s", scenario_name)
                        continue
                    scenario = Scenario(
                        name=scenario_name,
                        description=data.get("description", ""),
                        created_at=data.get("created_at", ""),
                        modified_at=data.get("modified_at", ""),
                        config=data.get("config", {}),
                        tags=data.get("tags", []),
                        parent_scenario=data.get("parent_scenario"),
                        key_case_map=data.get("key_case_map", {}),
                    )
                    self.save(scenario, preserve_modified_at=True)
                    count += 1
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    _log.warning(
                        "Skipping malformed ZIP entry %r (%s): %s",
                        scenario_name or name,
                        type(exc).__name__,
                        exc,
                    )
        return count
