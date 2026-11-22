"""Read and parse the project's README/CHANGELOG for in-app rendering.

Pure module (no UI import). Reads the maintained docs from the repo root by a fixed
whitelist and parses the latest changelog entry for the startup "what's new" modal.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

#: Where the repository's files live on the web. The app serves none of them, so a relative
#: link in the README or CHANGELOG is rewritten to this, at the running version's tag.
GITHUB_REPO = "https://github.com/razinkele/osmopy"

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


# ------------------------------------------------------------- rendering inside the app

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ANCHOR = re.compile(r'<a href="([^"]*)"')
_HEADING = re.compile(r"<h([1-6])>(.*?)</h\1>", re.DOTALL)
_IMG = re.compile(r"<img(?![^>]*\sloading=)")
_TAG = re.compile(r"<[^>]+>")


def is_relative(target: str) -> bool:
    """A path inside the repository: not empty, not an in-page anchor, not absolute, no scheme."""
    return bool(target) and target[0] not in "#/" and not _SCHEME.match(target)


def github_url(target: str, ref: str) -> str:
    """The repository file or folder a relative README link names, at ``ref`` on GitHub.

    Files go to ``blob``, a path ending in ``/`` to ``tree``; a ``#fragment`` is kept.
    """
    path, sep, fragment = target.partition("#")
    path = path.removeprefix("./")
    kind = "tree" if not path or path.endswith("/") else "blob"
    return f"{GITHUB_REPO}/{kind}/{ref}/{path}" + (f"#{fragment}" if sep else "")


def heading_slug(heading_html: str, seen: dict[str, int]) -> str:
    """The id GitHub gives a heading, so the README's own ``#...`` links work in the app too.

    Lowercase, drop every character that is not a letter, digit, underscore, hyphen or space,
    then each space becomes a hyphen; a repeat gets ``-1``, ``-2``... (GitHub's rule).
    """
    text = html.unescape(_TAG.sub("", heading_html)).strip().lower()
    slug = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
    count = seen.get(slug, 0)
    seen[slug] = count + 1
    return slug if count == 0 else f"{slug}-{count}"


def html_for_app(rendered: str, ref: str, id_prefix: str) -> str:
    """Rendered README/CHANGELOG HTML made to work inside the app.

    - A relative link becomes the repository file on GitHub at ``ref`` (the app serves none of
      them), and every web link opens in a new tab: inside the About dialog a link that
      replaced the page would drop the user's session.
    - Headings get GitHub's ids, with ``id_prefix`` in front, and the document's in-page links
      (the README's Contents list) are rewritten to match. Several documents share the app's
      page (the CHANGELOG twice: the About tab and the what's-new dialog), so bare ids would
      repeat across them, or meet one of the app's own; a prefix per document prevents both.
    - Images load lazily: the dialogs are built with the page, and would otherwise fetch every
      picture on every visit.

    Works on the rendered HTML, where code blocks are already escaped text, so a link written
    inside a code block is never touched.
    """

    def link(match: re.Match) -> str:
        target = html.unescape(match.group(1))
        if is_relative(target):
            target = github_url(target, ref)
        elif target.startswith("#") and len(target) > 1:
            target = f"#{id_prefix}{target[1:]}"
        attrs = f'href="{html.escape(target, quote=True)}"'
        if target.startswith(("http://", "https://")):
            attrs += ' target="_blank" rel="noopener"'
        return f"<a {attrs}"

    seen: dict[str, int] = {}

    def heading(match: re.Match) -> str:
        level, inner = match.group(1), match.group(2)
        return f'<h{level} id="{id_prefix}{heading_slug(inner, seen)}">{inner}</h{level}>'

    out = _ANCHOR.sub(link, rendered)
    out = _HEADING.sub(heading, out)
    return _IMG.sub('<img loading="lazy"', out)
