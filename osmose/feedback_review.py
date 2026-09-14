"""Maintainer-facing HTML rendering of the feedback store (pure core — no web/UI imports).

SECURITY: every value rendered here was typed into a PUBLIC form by an anonymous submitter, so
this module is the primary injection surface of the feedback feature. Nothing is interpolated
without ``_esc`` (``html.escape(..., quote=True)``) — message, type, timestamp, version, nav tab
and id alike. Assume a stored message contains ``<script>``, quotes and attribute-breaking
sequences, because eventually one will.

PRIVACY: a record carries ``has_contact`` (a boolean) and never the address itself; the address
lives only in the contacts side-store behind ``osmose.feedback.lookup_contact``. This module
renders the flag and never calls that lookup — the page is designed to be safe to screenshot.
"""

from __future__ import annotations

import html

# Badge classes are whitelisted rather than derived from the record: `type` is attacker-supplied
# on a hand-written store line, and user data must not reach a CSS class name even escaped.
_TYPE_CLASSES = {"bug": "fb-bug", "suggestion": "fb-suggestion", "other": "fb-other"}
_DEFAULT_TYPE_CLASS = "fb-unknown"
_EMPTY = "&mdash;"

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


def _or_dash(value: object) -> str:
    """Escaped value, or an em dash when the field is missing/blank."""
    esc = _esc(value)
    return esc if esc.strip() else _EMPTY


def _render_card(record: dict) -> str:
    """One feedback record as a card. Every interpolation goes through ``_esc``."""
    raw_type = record.get("type")
    badge_class = _TYPE_CLASSES.get(str(raw_type), _DEFAULT_TYPE_CLASS)
    # Never render the value of has_contact, and never look the address up — just the flag.
    # `is True`, not truthiness: a hand-written `"has_contact": "false"` is a non-empty string
    # and would otherwise be reported as "yes". Only a real JSON boolean reads as yes.
    has_contact = "yes" if record.get("has_contact") is True else "no"
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
        f"</div>"
        f"</article>"
    )


def render_review_html(records: list[dict], repo_url: str) -> str:
    """Full HTML page listing ``records`` (newest first, as ``read_feedback`` returns them).

    ``repo_url`` is rendered once in the page header as a link back to the project repository.
    """
    cards = "".join(_render_card(rec) for rec in records)
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
