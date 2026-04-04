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

    After reading, ``self.key_case_map`` maps each lowercase key to the
    original case as it appeared in the config file.  Writers use this to
    restore Java's expected case when writing config back.
    """

    SEPARATORS = re.compile(r"\s*[=;,:\t]\s*")
    COMMENT_CHARS = {"#", "!"}

    def __init__(self) -> None:
        self.key_case_map: dict[str, str] = {}
        self.skipped_lines: int = 0

    def read(self, master_file: Path) -> dict[str, str]:
        """Recursively read a master config and all referenced sub-configs."""
        self.skipped_lines = 0
        master_file = Path(master_file)
        _log.info("Reading config from %s", master_file)
        flat: dict[str, str] = {}
        self._read_recursive(master_file, flat)
        flat["_osmose.config.dir"] = str(master_file.parent.resolve())
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
        """Parse a single OSMOSE config file into a flat key-value dict.

        Keys are stored as lowercase for internal lookups. The original
        case is preserved in ``self.key_case_map`` so that writers can
        restore the case Java expects.
        """
        if filepath.stat().st_size > 10_000_000:  # 10MB
            raise ValueError(f"Config file too large: {filepath} ({filepath.stat().st_size} bytes)")
        result: dict[str, str] = {}
        skipped = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in self.COMMENT_CHARS:
                    continue
                parts = self.SEPARATORS.split(line, maxsplit=1)
                if len(parts) == 2:
                    raw_key = parts[0].strip()
                    key = raw_key.lower()
                    value = parts[1].strip()
                    # Strip trailing separators (e.g., "true," → "true")
                    value = value.rstrip(";,:\t =")
                    result[key] = value
                    self.key_case_map[key] = raw_key
                else:
                    _log.warning("Skipping unparseable line in %s: %r", filepath.name, line)
                    skipped += 1
        self.skipped_lines += skipped
        return result
