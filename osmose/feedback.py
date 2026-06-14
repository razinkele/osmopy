"""Feedback store + token check (pure core — no web/UI imports).

Bug reports / suggestions submitted from the Shiny UI are appended as JSON lines to
``FEEDBACK_FILE`` (overridable via ``OSMOSE_FEEDBACK_FILE``). A token-gated read endpoint in
``app.py`` serves them back to a maintainer. Mirrors the repo's JSON-on-disk convention.
"""

from __future__ import annotations

import json
import os
import secrets
import uuid
from datetime import datetime
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.feedback")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # osmose/feedback.py -> repo root
FEEDBACK_FILE = _PROJECT_ROOT / "data" / "feedback" / "feedback.jsonl"  # default
_FILE_ENV = "OSMOSE_FEEDBACK_FILE"
_TOKEN_ENV = "OSMOSE_FEEDBACK_TOKEN"
_VALID_TYPES = {"bug", "suggestion", "other"}
_MAX_MESSAGE = 5000


def _resolve(path: Path | None) -> Path:
    """Resolve the store path at call time: explicit arg > OSMOSE_FEEDBACK_FILE > default."""
    if path is not None:
        return Path(path)
    env = os.environ.get(_FILE_ENV)
    return Path(env) if env else FEEDBACK_FILE


def build_feedback_record(
    type: str, message: str, *, contact: str = "", version: str = "", nav_tab: str = ""
) -> dict:
    """Validated feedback record. Raises ValueError on unknown type / empty message; truncates."""
    if type not in _VALID_TYPES:
        raise ValueError(f"Unknown feedback type: {type!r}")
    msg = (message or "").strip()
    if not msg:
        raise ValueError("Feedback message is empty")
    return {
        "id": uuid.uuid4().hex,
        "ts": datetime.now().isoformat(),
        "type": type,
        "message": msg[:_MAX_MESSAGE],
        "contact": (contact or "").strip(),
        "version": version,
        "nav_tab": nav_tab,
    }


def append_feedback(record: dict, *, path: Path | None = None) -> None:
    """Append one record as a JSON line (creates parent dir; POSIX flock; single-worker safe)."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # non-POSIX / unsupported — single-worker deploy is safe
            pass
        f.write(line)


def read_feedback(*, path: Path | None = None) -> list[dict]:
    """All records newest-first; missing file -> []; corrupt lines skipped (path resolved lazily)."""
    p = _resolve(path)
    if not p.is_file():
        return []
    out: list[dict] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except Exception:  # noqa: BLE001 — skip a corrupt line, don't fail the read
            _log.warning("Skipping corrupt feedback line")
            continue
    out.reverse()
    return out


def check_feedback_token(provided: str | None) -> bool:
    """Constant-time token check; total (never raises).

    False if OSMOSE_FEEDBACK_TOKEN is unset (endpoint disabled) or provided is None. Compares on
    UTF-8 bytes — compare_digest raises TypeError on non-ASCII str, and a header byte >= 0x80
    decodes (latin-1) to non-ASCII, so a str compare would let an unauth request crash the handler.
    """
    tok = os.environ.get(_TOKEN_ENV)
    if not tok or provided is None:
        return False
    return secrets.compare_digest(provided.encode("utf-8"), tok.encode("utf-8"))
