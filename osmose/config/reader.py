"""Parse OSMOSE .properties/.csv configuration files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.config")


@dataclass(frozen=True)
class ConfigDiagnostic:
    """A structured, line-located config-parse issue."""

    file: str
    lineno: int | None
    line: str
    reason: str  # unparseable|empty_key|duplicate_key|circular_ref|missing_subconfig|path_escape
    detail: str


_ERROR_REASONS: frozenset[str] = frozenset(
    {"unparseable", "circular_ref", "missing_subconfig", "path_escape"}
)


def diagnostics_have_errors(diagnostics: list[ConfigDiagnostic]) -> bool:
    """True if any diagnostic is ERROR-class (vs the empty_key/duplicate_key warnings)."""
    return any(d.reason in _ERROR_REASONS for d in diagnostics)


def format_diagnostics(diagnostics: list[ConfigDiagnostic]) -> str:
    """Human-readable report grouped by file; one line per diagnostic + a summary."""
    if not diagnostics:
        return "No config issues found."
    out: list[str] = []
    for d in diagnostics:
        if d.lineno is not None:
            body = f"{d.file}:{d.lineno}: {d.reason}"
            if d.line:
                body += f" — {d.line}"
        else:
            body = f"{d.file}: {d.reason}"
            if d.detail:
                body += f" — {d.detail}"
        out.append(body)
    counts: dict[str, int] = {}
    for d in diagnostics:
        counts[d.reason] = counts.get(d.reason, 0) + 1
    summary = ", ".join(f"{n} {r}" for r, n in sorted(counts.items()))
    out.append(f"{len(diagnostics)} issue(s): {summary}")
    return "\n".join(out)


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
        self.deprecated_keys: list[str] = []
        self.skipped_lines: int = 0
        self.diagnostics: list[ConfigDiagnostic] = []

    def read(self, master_file: Path) -> dict[str, str]:
        """Recursively read a master config and all referenced sub-configs."""
        self.skipped_lines = 0
        self.key_case_map = {}
        self.deprecated_keys = []
        self.diagnostics = []
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
            self.diagnostics.append(
                ConfigDiagnostic(filepath.name, None, "", "circular_ref", str(filepath))
            )
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
                    self.diagnostics.append(
                        ConfigDiagnostic(
                            filepath.name, None, "", "path_escape", f"{sub_path} (from key {key})"
                        )
                    )
                    continue
                if sub_path.exists():
                    self._read_recursive(sub_path, flat, _seen)
                else:
                    _log.warning("Referenced sub-config not found: %s (from key %s)", sub_path, key)
                    self.diagnostics.append(
                        ConfigDiagnostic(
                            filepath.name,
                            None,
                            "",
                            "missing_subconfig",
                            f"{sub_path} (from key {key})",
                        )
                    )

    def read_file(self, filepath: Path) -> dict[str, str]:
        """Parse a single OSMOSE config file into a flat key-value dict.

        Keys are stored as lowercase for internal lookups. The original
        case is preserved in ``self.key_case_map`` so that writers can
        restore the case Java expects.
        """
        st = filepath.stat()
        if st.st_size > 10_000_000:  # 10MB
            raise ValueError(f"Config file too large: {filepath} ({st.st_size} bytes)")
        result: dict[str, str] = {}
        skipped = 0
        seen_keys: set[str] = set()
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for lineno, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line[0] in self.COMMENT_CHARS:
                    continue
                parts = self.SEPARATORS.split(line, maxsplit=1)
                if len(parts) == 2:
                    raw_key = parts[0].strip()
                    key = raw_key.lower()
                    value = parts[1].strip()
                    # Strip trailing separators (e.g., "true," → "true")
                    value = value.rstrip(";,:\t =")
                    if key == "":
                        # Separator-led line. ",,"/";;" (empty value) are benign blank rows;
                        # "=value" (lost its key) is a real error. Storage is unchanged below;
                        # empty keys are never tracked for duplicates.
                        if value != "":
                            self.diagnostics.append(
                                ConfigDiagnostic(
                                    filepath.name,
                                    lineno,
                                    line,
                                    "empty_key",
                                    "missing key before separator",
                                )
                            )
                    elif key in seen_keys:
                        self.diagnostics.append(
                            ConfigDiagnostic(
                                filepath.name,
                                lineno,
                                line,
                                "duplicate_key",
                                f"overrides earlier '{self.key_case_map.get(key, key)}'",
                            )
                        )
                    result[key] = value
                    self.key_case_map[key] = raw_key
                    if key != "":
                        seen_keys.add(key)
                else:
                    self.diagnostics.append(
                        ConfigDiagnostic(filepath.name, lineno, line, "unparseable", "")
                    )
                    _log.warning(
                        "Skipping unparseable line %d in %s: %r", lineno, filepath.name, line
                    )
                    skipped += 1
        self.skipped_lines += skipped

        from osmose.config.aliases import canonicalize_config

        had_version = "osmose.version" in result
        canon, deprecated = canonicalize_config(result)
        # canonicalize_config stamps osmose.version=4.4.0; don't fabricate it on a
        # per-file dict that never declared one (the master file carries the version).
        if not had_version:
            canon.pop("osmose.version", None)
        self.deprecated_keys.extend(d for d in deprecated if d not in self.deprecated_keys)
        if deprecated:
            # KEEP the renamed-old case_map entries (do NOT pop them): the writer reverse-maps
            # NEW->OLD before serializing and looks the case_map up by the OLD key, so it must
            # still find the source casing (e.g. output.fishery.byAge.enabled). Only ADD new keys.
            for new_key in canon:
                self.key_case_map.setdefault(new_key, new_key)
        result = canon
        return result
