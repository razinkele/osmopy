"""Feedback store + token check (pure core — no web/UI imports).

Bug reports / suggestions submitted from the Shiny UI are appended as JSON lines to
``FEEDBACK_FILE`` (overridable via ``OSMOSE_FEEDBACK_FILE``). A token-gated read endpoint in
``app.py`` serves them back to a maintainer. Mirrors the repo's JSON-on-disk convention.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import uuid
from datetime import datetime
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.feedback")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # osmose/feedback.py -> repo root
FEEDBACK_FILE = _PROJECT_ROOT / "data" / "feedback" / "feedback.jsonl"  # default
CONTACTS_FILE = _PROJECT_ROOT / "data" / "feedback" / "contacts.jsonl"  # default; PII, gitignored
_FILE_ENV = "OSMOSE_FEEDBACK_FILE"
_CONTACTS_ENV = "OSMOSE_CONTACTS_FILE"
_TOKEN_ENV = "OSMOSE_FEEDBACK_TOKEN"
VALID_TYPES = frozenset({"bug", "suggestion", "other"})
"""The accepted feedback types. PUBLIC because two layers validate against it: this
module's ``build_feedback_record`` and, earlier in the request, the submit handler's
``_classify_and_consume`` -- which must reject an unknown type BEFORE its honeypot check
or the differing responses name the trap field. One constant so the two cannot drift."""
_MAX_MESSAGE = 5000
MAX_CONTACT = 254  # RFC 5321 practical maximum for an address
MAX_STORE_BYTES = 50 * 1024 * 1024
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")


def _resolve(path: Path | None) -> Path:
    """Resolve the store path at call time: explicit arg > OSMOSE_FEEDBACK_FILE > default."""
    if path is not None:
        return Path(path)
    env = os.environ.get(_FILE_ENV)
    return Path(env) if env else FEEDBACK_FILE


def _resolve_contacts(path: Path | None) -> Path:
    """Resolve the contacts side-store path: explicit arg > OSMOSE_CONTACTS_FILE > default."""
    if path is not None:
        return Path(path)
    env = os.environ.get(_CONTACTS_ENV)
    return Path(env) if env else CONTACTS_FILE


def looks_like_email(s: str) -> bool:
    """Shape check only — deliberately NOT an RFC validator and NOT a deliverability check."""
    return bool(_EMAIL_RE.match((s or "").strip()))


def build_feedback_record(
    type: str,
    message: str,
    *,
    contact: str = "",
    version: str = "",
    nav_tab: str = "",
    honeypot: str = "",
) -> dict:
    """Validated feedback record. Raises ValueError on unknown type / empty message; truncates.

    ``contact`` is used only to set the boolean ``has_contact`` flag on the returned record --
    the address itself never lands here. Callers that want the address stored must separately
    call ``save_contact(record["id"], contact)``, which writes it to the private side-store.
    This is a deliberate structural split: the record is designed to be copied into a public
    GitHub issue, so the address must never be reachable through it.
    """
    if (honeypot or "").strip():
        raise ValueError("honeypot field was filled — rejecting as automated submission")
    if type not in VALID_TYPES:
        raise ValueError(f"Unknown feedback type: {type!r}")
    msg = (message or "").strip()
    if not msg:
        raise ValueError("Feedback message is empty")
    return {
        "id": uuid.uuid4().hex,
        "ts": datetime.now().isoformat(),
        "type": type,
        "message": msg[:_MAX_MESSAGE],
        "has_contact": bool((contact or "").strip()),
        "version": version,
        "nav_tab": nav_tab,
    }


def _append_json_line(p: Path, payload: dict, *, what: str) -> None:
    """Append one JSON line under a size cap and an exclusive POSIX lock (creates parent dir).

    Shared by BOTH stores deliberately. Until 2026-09-14 the size cap and the flock lived only
    in ``append_feedback``, so ``feedback.jsonl`` — the file designed to be pasted into a public
    issue — had both protections while ``contacts.jsonl``, the file holding email addresses, had
    neither. The asymmetry ran backwards: the more sensitive store was the less protected one.
    One implementation means the next change cannot reintroduce it on one side only.

    Raises ``RuntimeError`` when the target is at or over ``MAX_STORE_BYTES``. Callers must
    treat that as terminal — retrying cannot clear it, only rotation can.
    """
    if p.is_file() and p.stat().st_size >= MAX_STORE_BYTES:
        raise RuntimeError(f"{what} is full ({p.stat().st_size} bytes) — rotate {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # non-POSIX / unsupported — single-worker deploy is safe
            pass
        f.write(line)


def append_feedback(record: dict, *, path: Path | None = None) -> None:
    """Append one record as a JSON line (creates parent dir; POSIX flock; single-worker safe)."""
    _append_json_line(_resolve(path), record, what="feedback store")


def save_contact(feedback_id: str, email: str, *, path: Path | None = None) -> None:
    """Store an address OUT OF BAND, keyed by feedback id. Never goes in the main record.

    Lives in ``CONTACTS_FILE`` (default ``data/feedback/contacts.jsonl``, gitignored — it holds
    PII), a file structurally separate from the public-facing feedback record. A no-op for an
    empty address. Truncates to ``MAX_CONTACT``, same cap as the (now-removed) record field.

    Goes through ``_append_json_line``, so it carries the SAME size cap and exclusive lock as
    ``append_feedback`` — see that helper for why the two must not drift apart again. Raising
    here is safe: the caller (``ui.components.feedback_modal._store_submission``) guards this
    call separately, logs the address as lost, and still reports success, because by then the
    feedback record itself is already stored.
    """
    email = (email or "").strip()[:MAX_CONTACT]
    if not email:
        return
    _append_json_line(
        _resolve_contacts(path), {"id": feedback_id, "email": email}, what="contacts store"
    )


def lookup_contact(feedback_id: str, *, path: Path | None = None) -> str | None:
    """Look up the address stored for ``feedback_id``, or None if missing/never saved."""
    p = _resolve_contacts(path)
    if not p.is_file():
        return None
    for raw in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 — skip a corrupt line, don't fail the lookup
            continue
        if rec.get("id") == feedback_id:
            return rec.get("email")
    return None


def read_feedback(*, path: Path | None = None) -> list[dict]:
    """All records newest-first; missing file -> []; corrupt lines skipped (path resolved lazily).

    Normalises v1 records that still carry a literal ``contact`` key (D6 backwards
    compatibility): the address is dropped and folded into the boolean ``has_contact`` flag, so
    callers never see an address here regardless of which code version wrote the line.
    """
    p = _resolve(path)
    if not p.is_file():
        return []
    out: list[dict] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except Exception:  # noqa: BLE001 — skip a corrupt line, don't fail the read
            _log.warning("Skipping corrupt feedback line")
            continue
        if not isinstance(rec, dict):
            # Valid JSON but not an object (`null`, `[]`, `"x"`). Without this the `.setdefault`
            # below raises AttributeError and the whole read fails, which takes the maintainer
            # review page down entirely — one bad line would hide every good record.
            _log.warning("Skipping non-object feedback line")
            continue
        rec.setdefault("has_contact", bool(rec.pop("contact", "")))
        out.append(rec)
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
