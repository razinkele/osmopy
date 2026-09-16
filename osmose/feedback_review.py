"""Maintainer-facing HTML rendering of the feedback store (pure core — no web/UI imports).

SECURITY: every value rendered here was typed into a PUBLIC form by an anonymous submitter, so
this module is the primary injection surface of the feedback feature. Nothing is interpolated
without ``_esc`` (``html.escape(..., quote=True)``) — message, type, timestamp, version, nav tab
and id alike. Assume a stored message contains ``<script>``, quotes and attribute-breaking
sequences, because eventually one will.

PRIVACY: a record carries ``has_contact`` (a boolean) and never the address itself; the address
lives only in the contacts side-store behind ``osmose.feedback.lookup_contact``. This module
renders the flag and never calls that lookup — the page is designed to be safe to screenshot.
Each card also carries a one-click "promote to a GitHub issue" link built by ``github_issue_url``.
That URL is the one place a record's content leaves this page for a PUBLIC destination, so it is
assembled from an explicit allowlist of fields (message, version, nav tab, id, has_contact flag)
rather than by serialising the record — a legacy v1 line that still carries a literal ``contact``
key therefore cannot leak an address into a filed issue.
"""

from __future__ import annotations

import html
from urllib.parse import urlencode, urlsplit

# Badge classes are whitelisted rather than derived from the record: `type` is attacker-supplied
# on a hand-written store line, and user data must not reach a CSS class name even escaped.
_TYPE_CLASSES = {"bug": "fb-bug", "suggestion": "fb-suggestion", "other": "fb-other"}
_DEFAULT_TYPE_CLASS = "fb-unknown"
_EMPTY = "&mdash;"  # HTML form, for the card
_EMPTY_TEXT = "—"  # plain-text form, for the issue body — same glyph, same meaning

# GitHub label per feedback type. Whitelisted for the same reason as the badge class above:
# `type` is attacker-supplied on a hand-written store line and must not reach the issue URL raw.
_LABEL = {"bug": "bug", "suggestion": "enhancement", "other": "question"}
_DEFAULT_LABEL = "question"
# Only these two schemes may reach an href. `html.escape` does NOT neutralise `javascript:` or
# `data:` — escaping an attribute value cannot make its scheme safe, so the scheme is validated
# rather than escaped (R17).
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_TITLE_CHARS = 60  # first line of the message, truncated, as the issue title
_NO_MESSAGE_TITLE = "(no message)"

_STYLE = """
:root { color-scheme: light dark; }
body { margin: 0; padding: 1.5rem; font: 14px/1.5 system-ui, -apple-system, sans-serif;
       background: #0f1b24; color: #dfe9f0; }
a { color: #6fc3df; }
h1 { font-size: 1.25rem; margin: 0 0 .25rem; }
.fb-head { border-bottom: 1px solid #24404f; padding-bottom: .75rem; margin-bottom: 1.25rem; }
.fb-count { color: #8fa9b8; }
.fb-card { border: 1px solid #24404f; border-radius: 6px; padding: .75rem 1rem;
           margin-bottom: .75rem; background: #16262f; }
.fb-card-head { display: flex; flex-wrap: wrap; gap: .75rem; align-items: baseline; }
.fb-badge { border-radius: 3px; padding: .1rem .5rem; font-weight: 600; font-size: .8rem;
            background: #24404f; }
.fb-bug { background: #6e2b2b; }
.fb-suggestion { background: #2b5a6e; }
.fb-other { background: #3a3f52; }
.fb-unknown { background: #4a4a4a; }
.fb-ts, .fb-contact { color: #8fa9b8; font-size: .85rem; }
.fb-msg { white-space: pre-wrap; word-break: break-word; margin: .6rem 0 .4rem;
          background: #0f1b24; border-radius: 4px; padding: .6rem; }
.fb-meta { color: #8fa9b8; font-size: .8rem; display: flex; flex-wrap: wrap; gap: 1rem; }
.fb-empty { color: #8fa9b8; }
.fb-promote { font-weight: 600; }
"""


def _esc(value: object) -> str:
    """Escape any value for either a text or an attribute context. ``None`` -> empty string.

    The ``None`` branch is cosmetic, not a safety guard: without it a null field would render as
    the literal string ``"None"`` (``html.escape(str(None))`` returns ``'None'`` and does not
    raise). Returning ``""`` instead lets ``_or_dash`` show an em dash for a missing field.
    """
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _text(value: object) -> str:
    """``str(value)``, with ``None`` as the empty string. No escaping — plain text."""
    return "" if value is None else str(value)


def _blank(value: object) -> bool:
    """The card's em-dash condition: missing, null or whitespace-only.

    Note what this is NOT: a truthiness test. ``0``, ``False`` and ``[]`` are falsy but are real
    values a hand-written store line can carry, and the card renders them (``0``, ``False``,
    ``[]``). ``x or default`` would swallow all three.
    """
    return not _text(value).strip()


def _or_dash(value: object) -> str:
    """Escaped value, or an em dash when the field is missing/blank."""
    return _EMPTY if _blank(value) else _esc(value)


def _or_dash_text(value: object) -> str:
    """Plain-text twin of ``_or_dash``, for the issue body.

    The card and the issue it promotes to must not disagree about the same record: a maintainer
    reading an em dash on the page and the literal word ``None`` in the filed issue has no way to
    tell which one is the record.
    """
    return _EMPTY_TEXT if _blank(value) else _text(value)


def github_issue_url(record: dict, repo_url: str) -> str:
    """Prefilled ``issues/new`` link for one record. NEVER includes contact details (D2).

    A prefilled URL rather than the REST API (D4): no token, no bot account, nothing to rotate or
    leak, and the maintainer reviews the issue before it is filed — the gate that makes a public
    submission channel safe.

    PRIVACY: the body is assembled from a fixed allowlist — message, version, nav tab, id, and the
    ``has_contact`` flag. The record is never serialised wholesale, so a legacy v1 line that still
    carries a literal ``contact`` key cannot carry an address into a public issue. As on the card,
    ``has_contact`` is tested with ``is True``: ``bool("false")`` is True and a hand-written string
    flag would otherwise be reported as "yes".

    TWO of those fields are reporter-controlled free text, not one. ``message`` is the obvious one.
    ``nav_tab`` is the other: it comes from ``_safe_nav`` (``ui/components/feedback_modal.py``),
    which reads ``input.main_nav()`` — a client-settable Shiny input with no whitelist — so a
    crafted client can put arbitrary text in it and that text lands in the public issue body. This
    is self-disclosure by the submitter, not a leak of anyone else's data, and the maintainer's
    review before filing (D4) is the gate. But do not read ``nav_tab`` as machine-generated and
    safe to paste unread: treat it exactly as you treat ``message``.

    SECURITY: the return value lands in an ``href``. ``html.escape`` does not neutralise a
    ``javascript:`` or ``data:`` scheme, so ``repo_url``'s scheme is *validated* here — only http
    and https are accepted, and anything else (including a scheme-relative ``//host``, which parses
    to an empty scheme) raises ``ValueError``. ``repo_url`` is an operator-controlled constant
    (``app.py:_REPO_URL``), never attacker data, so a bad one is a deployment bug rather than an
    availability risk.

    Know what that costs, though: the raise propagates through ``render_review_html`` and
    ``app.py``'s route catches it, so a bad ``repo_url`` loses the ENTIRE page — every card, a bare
    ``500 internal``. An earlier version of this docstring called that "loud". It was not; measured
    2026-09-14, the route logged nothing at all. It is loud now only because that route logs the
    traceback server-side (``app.py:_log.exception``) while still disclosing nothing to the caller.
    If that logging is ever removed, this becomes a silent total failure again.

    AVAILABILITY: the store is a plain JSONL file that ``read_feedback`` does not validate
    field-by-field, so a hand-written or legacy line can carry an empty, whitespace-only, missing
    or non-string ``message``/``type``. All of those must still produce a usable URL — this
    function is called once per card, so one bad line raising would blank the whole review page.
    """
    scheme = urlsplit(repo_url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"repo_url must use http or https (got scheme {scheme!r} from {repo_url!r}) — "
            "an issue link is rendered into an href and escaping cannot make a scheme safe"
        )
    # `_text`/`_or_dash_text`, not `x or default`: every field below is rendered on the card too,
    # and the two must agree about the same record. `.get(k, "?")` does NOT default on an explicit
    # null (the key is present), and `x or ""` swallows the falsy-but-real values `0`, `False` and
    # `[]` that the card happily renders. The str() coercion also mirrors _esc/_TYPE_CLASSES: a
    # non-string field must not raise part-way through the page.
    raw_type = record.get("type")
    kind = _text(raw_type)  # label lookup key; the whitelist below is what actually reaches GitHub
    message = _text(record.get("message"))
    # `"".splitlines()` is `[]`, so indexing [0] unguarded raises IndexError on an empty or
    # whitespace-only message and takes the page down with it.
    lines = message.strip().splitlines()
    title_text = lines[0][:_TITLE_CHARS] if lines else _NO_MESSAGE_TITLE
    body = (
        f"{message}\n\n---\n"
        f"- app version: `{_or_dash_text(record.get('version'))}`\n"
        f"- tab: `{_or_dash_text(record.get('nav_tab'))}`\n"
        f"- feedback id: `{_or_dash_text(record.get('id'))}`\n"
        f"- reporter left contact details: {'yes' if record.get('has_contact') is True else 'no'}\n"
    )
    query = urlencode(
        {
            "title": f"[{_or_dash_text(raw_type)}] {title_text}",
            "body": body,
            "labels": _LABEL.get(kind, _DEFAULT_LABEL),
        }
    )
    return f"{repo_url.rstrip('/')}/issues/new?{query}"


def _render_card(record: dict, repo_url: str) -> str:
    """One feedback record as a card. Every interpolation goes through ``_esc``."""
    raw_type = record.get("type")
    badge_class = _TYPE_CLASSES.get(str(raw_type), _DEFAULT_TYPE_CLASS)
    # Never render the value of has_contact, and never look the address up — just the flag.
    # `is True`, not truthiness: a hand-written `"has_contact": "false"` is a non-empty string
    # and would otherwise be reported as "yes". Only a real JSON boolean reads as yes.
    has_contact = "yes" if record.get("has_contact") is True else "no"
    # Escaped as well as scheme-validated: the query string is already percent-encoded, but
    # urlencode's `&` separators must become `&amp;` to be a well-formed attribute value.
    issue_href = _esc(github_issue_url(record, repo_url))
    return (
        f'<article class="fb-card" data-has-contact="{has_contact}">'
        f'<div class="fb-card-head">'
        f'<span class="fb-badge {badge_class}">{_or_dash(raw_type)}</span>'
        f'<span class="fb-ts">{_or_dash(record.get("ts"))}</span>'
        f'<span class="fb-contact">contact: {has_contact}</span>'
        f"</div>"
        f'<pre class="fb-msg">{_esc(record.get("message"))}</pre>'
        f'<div class="fb-meta">'
        f"<span>id {_or_dash(record.get('id'))}</span>"
        f"<span>version {_or_dash(record.get('version'))}</span>"
        f"<span>tab {_or_dash(record.get('nav_tab'))}</span>"
        f'<a class="fb-promote" href="{issue_href}" rel="noopener noreferrer" '
        f'target="_blank">promote to a GitHub issue</a>'
        f"</div>"
        f"</article>"
    )


def render_review_html(records: list[dict], repo_url: str) -> str:
    """Full HTML page listing ``records`` (newest first, as ``read_feedback`` returns them).

    ``repo_url`` is rendered once in the page header as a link back to the project repository,
    and once per card as the target of that record's prefilled ``issues/new`` promotion link
    (``github_issue_url``). It must be an http/https URL; anything else raises ``ValueError``
    rather than emitting an unsafe href.
    """
    cards = "".join(_render_card(rec, repo_url) for rec in records)
    body = cards or '<p class="fb-empty">No feedback has been submitted yet.</p>'
    count = len(records)
    plural = "" if count == 1 else "s"
    repo = _esc(repo_url)
    return (
        "<!doctype html>"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Feedback review</title>"
        f"<style>{_STYLE}</style></head><body>"
        f'<div class="fb-head"><h1>Feedback review</h1>'
        f'<span class="fb-count">{count} record{plural}</span> &middot; '
        f'<a class="fb-repo" href="{repo}" rel="noopener noreferrer" target="_blank">{repo}</a>'
        f"</div>{body}</body></html>"
    )
