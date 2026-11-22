"""The README and CHANGELOG work inside the app: their links resolve and their pictures load.

The About dialog renders README.md and CHANGELOG.md, but the app serves none of the repository's
files, so every relative link in them used to be a 404, the README's own "Contents" links led
nowhere (rendered headings had no ids), and its two screenshots were missing from every page
(the dialogs are built with the page). ``html_for_app`` rewrites the rendered HTML, and the app
serves ``screenshots/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from shiny import ui

from osmose import __version__
from osmose.docs_content import (
    GITHUB_REPO,
    github_url,
    heading_slug,
    html_for_app,
    is_relative,
    read_doc,
)

ROOT = Path(__file__).resolve().parents[1]
REF = "v9.9.9"
PREFIX = "doc-"


def _render(markdown: str) -> str:
    return html_for_app(str(ui.markdown(markdown)), REF, PREFIX)


# ------------------------------------------------------------------------ the parts


@pytest.mark.parametrize(
    ("target", "relative"),
    [
        ("docs/usage-guide.md", True),
        ("LICENSE", True),
        ("docs/plans/", True),
        ("./docs/x.md", True),
        ("#performance", False),
        ("/osmose/", False),
        ("https://github.com/razinkele/osmopy", False),
        ("mailto:someone@example.org", False),
        ("", False),
    ],
)
def test_is_relative(target, relative):
    assert is_relative(target) is relative


@pytest.mark.parametrize(
    ("target", "url"),
    [
        ("docs/usage-guide.md", f"{GITHUB_REPO}/blob/{REF}/docs/usage-guide.md"),
        (
            "docs/usage-guide.md#2-read-outputs",
            f"{GITHUB_REPO}/blob/{REF}/docs/usage-guide.md#2-read-outputs",
        ),
        ("docs/plans/", f"{GITHUB_REPO}/tree/{REF}/docs/plans/"),
        ("./CHANGELOG.md", f"{GITHUB_REPO}/blob/{REF}/CHANGELOG.md"),
        ("LICENSE", f"{GITHUB_REPO}/blob/{REF}/LICENSE"),
    ],
)
def test_github_url(target, url):
    assert github_url(target, REF) == url


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        ("Simulation engines", "simulation-engines"),
        ("Engine parity validation", "engine-parity-validation"),
        ("Testing and linting", "testing-and-linting"),
        ("C# engine &amp; Java", "c-engine--java"),  # GitHub keeps both hyphens
        ("<code>osmose/</code> layout", "osmose-layout"),
        ("Baltic Sea", "baltic-sea"),
    ],
)
def test_heading_slug_follows_github(heading, slug):
    assert heading_slug(heading, {}) == slug


def test_a_repeated_heading_gets_a_suffix():
    seen: dict[str, int] = {}
    assert [heading_slug("Usage", seen) for _ in range(3)] == ["usage", "usage-1", "usage-2"]


# ------------------------------------------------------------------ the rendered HTML


def test_relative_links_go_to_github_and_open_in_a_new_tab():
    out = _render("See [the guide](docs/usage-guide.md#outputs) and [plans](docs/plans/).")
    assert (
        f'<a href="{GITHUB_REPO}/blob/{REF}/docs/usage-guide.md#outputs" target="_blank" rel="noopener">'
        in out
    )
    assert f'<a href="{GITHUB_REPO}/tree/{REF}/docs/plans/" target="_blank" rel="noopener">' in out


def test_web_links_open_in_a_new_tab_and_in_page_links_do_not():
    out = _render("# Top\n\n[web](https://osmose-model.org) and [up](#top)")
    assert '<a href="https://osmose-model.org" target="_blank" rel="noopener">' in out
    assert '<a href="#doc-top">' in out
    assert 'target="_blank" rel="noopener">up<' not in out


def test_headings_get_prefixed_ids_and_in_page_links_follow():
    out = _render("## Simulation engines\n\n[jump](#simulation-engines)")
    assert '<h2 id="doc-simulation-engines">Simulation engines</h2>' in out
    assert '<a href="#doc-simulation-engines">jump</a>' in out


def test_two_documents_on_one_page_do_not_share_an_id():
    one = html_for_app(str(ui.markdown("## Performance")), REF, "readme-")
    two = html_for_app(str(ui.markdown("## Performance")), REF, "changelog-")
    assert 'id="readme-performance"' in one and 'id="changelog-performance"' in two


def test_a_link_inside_a_code_block_is_left_alone():
    out = _render("```\n[not a link](docs/x.md)\n```")
    assert "[not a link](docs/x.md)" in out
    assert GITHUB_REPO not in out


def test_images_load_lazily_and_keep_their_paths():
    out = _render('<img src="screenshots/a.png" alt="A">\n\n![B](screenshots/b.png)')
    assert '<img loading="lazy" src="screenshots/a.png" alt="A">' in out
    assert re.search(r'<img loading="lazy" src="screenshots/b.png"', out)


def test_an_image_that_already_says_how_to_load_is_kept():
    out = _render('<img loading="eager" src="screenshots/a.png">')
    assert out.count("loading=") == 1


# ------------------------------------------------------------------- the real README


def _rendered(kind: str) -> str:
    return html_for_app(str(ui.markdown(read_doc(kind))), f"v{__version__}", f"{kind}-")


@pytest.mark.parametrize("kind", ["readme", "changelog"])
def test_no_relative_link_survives_in_the_app(kind):
    hrefs = re.findall(r'<a href="([^"]*)"', _rendered(kind))
    assert hrefs, "no links found: the rendered document is not what this test expects"
    assert [h for h in hrefs if is_relative(h)] == []


@pytest.mark.parametrize("kind", ["readme", "changelog"])
def test_every_web_link_opens_in_a_new_tab(kind):
    for tag in re.findall(r"<a [^>]*>", _rendered(kind)):
        href = re.search(r'href="([^"]*)"', tag)
        if href and href.group(1).startswith(("http://", "https://")):
            assert 'target="_blank"' in tag and 'rel="noopener"' in tag, tag


def test_every_in_page_link_of_the_readme_has_its_heading():
    out = _rendered("readme")
    ids = set(re.findall(r'<h[1-6] id="([^"]+)"', out))
    anchors = {h[1:] for h in re.findall(r'<a href="(#[^"]*)"', out)}
    assert anchors, "the README's Contents list has no in-page links any more"
    assert anchors <= ids, sorted(anchors - ids)
    assert all(a.startswith("readme-") for a in anchors)


def test_every_relative_link_in_the_readme_names_a_file_in_the_repository():
    """The rewritten link opens that path at the version's tag; it must exist now."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)", readme)
    relative = [t.partition("#")[0] for t in targets if is_relative(t)]
    assert relative, "the README has no relative links: this test no longer looks at anything"
    missing = [t for t in relative if not (ROOT / t).exists()]
    assert missing == []


def test_every_picture_in_the_readme_is_in_the_folder_the_app_serves():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sources = re.findall(r'<img [^>]*src="([^"]+)"', readme)
    sources += re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", readme)
    relative = [s for s in sources if is_relative(s)]
    assert relative, "the README has no pictures: this test no longer looks at anything"
    for src in relative:
        assert src.startswith("screenshots/"), src
        assert (ROOT / src).is_file(), src


def test_the_app_serves_the_screenshots_folder_next_to_www():
    import app as app_module

    assets = app_module._STATIC_ASSETS
    assert Path(assets["/screenshots"]) == ROOT / "screenshots"
    assert Path(assets["/"]) == ROOT / "www"
    # The more specific mount first, in case the mounts are matched in order.
    assert list(assets) == ["/screenshots", "/"]


def test_the_app_page_has_no_repeated_id():
    """The README, the CHANGELOG (twice), the About and Help texts share the app's page with
    the app's own elements; a repeated id breaks in-page links and getElementById. Measured
    2026-11-22 before the per-document prefix: six ids appeared twice (performance, bug-fixes,
    cicd, documentation, other, tests)."""
    from collections import Counter

    import app as app_module

    page = str(app_module.app_ui)
    ids = Counter(re.findall(r'\sid="([^"]+)"', page))
    assert len(ids) > 100, "the rendered page is not what this test expects"
    assert [i for i, n in ids.items() if n > 1] == []


def test_the_app_page_carries_the_rewritten_documents():
    """On the page the app actually builds (not just through the function): no relative link
    is left in any document, and the README's headings and Contents links are the prefixed
    ones. A dialog that went back to rendering a document raw fails here."""
    import app as app_module

    page = str(app_module.app_ui)
    anchors = re.findall(r'<a [^>]*href="([^"]*)"', page)
    assert len(anchors) > 50, "the rendered page is not what this test expects"
    assert [a for a in anchors if is_relative(a)] == []
    assert 'id="readme-performance"' in page and 'href="#readme-performance"' in page
    assert f'href="{GITHUB_REPO}/blob/v{__version__}/docs/usage-guide.md"' in page
    # Every rendered document: the CHANGELOG tab, the what's-new dialog, About and Help texts.
    for prefix in ("readme-", "changelog-", "whatsnew-", "about-", "help-"):
        assert f'id="{prefix}' in page, prefix
    # Every link to the web opens in a new tab, wherever it is on the page.
    for tag in re.findall(r'<a [^>]*href="https?://[^"]*"[^>]*>', page):
        assert 'target="_blank"' in tag, tag
