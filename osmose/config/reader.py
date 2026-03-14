"""Parse OSMOSE .properties/.csv configuration files."""

from __future__ import annotations

import re
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.config")


class OsmoseConfigReader:
    """Read OSMOSE configuration files with recursive sub-file loading.

    OSMOSE config files use key-value pairs with auto-detected separators
    (=, ;, comma, tab, colon). Lines starting with # or ! are comments.
    Sub-configs are referenced via osmose.configuration.* keys.
    """

    SEPARATORS = re.compile(r"\s*[=;,:\t]\s*")
    COMMENT_CHARS = {"#", "!"}

    def read(self, master_file: Path) -> dict[str, str]:
        """Recursively read a master config and all referenced sub-configs."""
        _log.info("Reading config from %s", master_file)
        flat: dict[str, str] = {}
        self._read_recursive(master_file, flat)
        return flat

    def _read_recursive(
        self, filepath: Path, flat: dict[str, str], _seen: set[Path] | None = None
    ) -> None:
        if _seen is None:
            _seen = set()
        resolved = filepath.resolve()
        if resolved in _seen:
            _log.warning("Circular config reference skipped: %s", filepath)
            return
        _seen.add(resolved)
        file_params = self.read_file(filepath)
        flat.update(file_params)
        config_dir = filepath.parent.resolve()
        for key, value in file_params.items():
            if key.startswith("osmose.configuration."):
                sub_path = filepath.parent / value.strip()
                resolved_sub = sub_path.resolve()
                if not resolved_sub.is_relative_to(config_dir):
                    _log.warning(
                        "Sub-file path escapes config directory, skipping: %s (from key %s)",
                        sub_path,
                        key,
                    )
                    continue
                if sub_path.exists():
                    self._read_recursive(sub_path, flat, _seen)
                else:
                    _log.warning("Referenced sub-config not found: %s (from key %s)", sub_path, key)

    def read_file(self, filepath: Path) -> dict[str, str]:
        """Parse a single OSMOSE config file into a flat key-value dict."""
        result: dict[str, str] = {}
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in self.COMMENT_CHARS:
                    continue
                parts = self.SEPARATORS.split(line, maxsplit=1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    # Strip trailing separators (e.g., "true," → "true")
                    value = value.rstrip(";,:\t =")
                    result[key] = value
        return result
