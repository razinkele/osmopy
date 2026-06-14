"""Read and parse the project's README/CHANGELOG for in-app rendering.

Pure module (no UI import). Reads the maintained docs from the repo root by a fixed
whitelist and parses the latest changelog entry for the startup "what's new" modal.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # osmose/docs_content.py -> repo root
_DOC_FILES = {"readme": "README.md", "changelog": "CHANGELOG.md"}
_FALLBACK = "_Documentation unavailable._"

# First dated version heading "## [x.y.z] - YYYY-MM-DD" (date optional); skips [Unreleased].
_VERSION_HEADING = re.compile(
    r"^##\s*\[(?!Unreleased\])([^\]]+)\]\s*(?:-\s*(\S.*?))?\s*$", re.MULTILINE
)


def read_doc(kind: str, *, root: Path | None = None) -> str:
    """Return a whitelisted doc file's text ("readme"/"changelog").

    Returns a fallback string if the file is missing/unreadable; raises ValueError
    for an unknown kind.
    """
    try:
        filename = _DOC_FILES[kind]
    except KeyError:
        raise ValueError(f"Unknown doc kind: {kind!r}") from None
    path = (root or _PROJECT_ROOT) / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK


def _section_body(text: str, start: int) -> str:
    """Text from ``start`` up to the next '## ' heading, stripped."""
    nxt = re.search(r"^##\s", text[start:], re.MULTILINE)
    end = start + nxt.start() if nxt else len(text)
    return text[start:end].strip()


def latest_changelog_entry(text: str) -> dict:
    """Parse the first dated version section of a changelog (pure, never raises).

    Returns {"version": str|None, "date": str|None, "body": str}. version/date come
    from the first "## [x.y.z] - date" heading (date None if absent); body = lines
    AFTER that heading up to the next "## " heading, stripped (heading excluded).
    Falls back to the [Unreleased] body, else "No release notes yet.".
    """
    m = _VERSION_HEADING.search(text)
    if m is None:
        um = re.search(r"^##\s*\[Unreleased\]\s*$", text, re.MULTILINE)
        if um is not None:
            body = _section_body(text, um.end())
            if body:
                return {"version": None, "date": None, "body": body}
        return {"version": None, "date": None, "body": "No release notes yet."}
    version = m.group(1).strip()
    date = m.group(2).strip() if m.group(2) else None
    return {"version": version, "date": date, "body": _section_body(text, m.end())}
