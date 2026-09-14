"""Integration tests for the token-gated maintainer review page (`GET /feedback/review`).

Every value this page renders arrives from a PUBLIC form, so the tests below are written as
injection probes: each negative assertion ("no raw markup", "no address") is paired with a
positive control proving the very record it is about actually reached the page. A negative
assertion on its own passes against a blank page, which is not evidence of anything.

The second half of the file probes `github_issue_url` directly. That URL is the one place a
record's content leaves the page for a PUBLIC destination, so its negatives (no address, no
non-http scheme) are asserted against the GENERATED URL, not against the page -- and each is
paired with a positive control naming a benign substring of that record's own message. A
`return ""` mutation of `github_issue_url` must red every one of those controls.
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.testclient import TestClient

from osmose.feedback import (
    append_feedback,
    build_feedback_record,
    lookup_contact,
    save_contact,
)
from osmose.feedback_review import github_issue_url

_MARKER = "BENIGN_MARKER_42"
# Closes the <pre> the message is rendered in, then injects a script. If the message is not
# escaped, `_pre_blocks` below stops at the injected `</pre>` and the marker falls outside the
# block -- which is exactly how the positive control catches an unescaped render.
_XSS_MESSAGE = f"</pre><script>alert('xss')</script>{_MARKER}"
_EMAIL = "maintainer.probe@example.org"
# Spelled out rather than imported from app: this asserts the page links to THE repository,
# not merely to whatever string app.py happened to pass in.
_REPO_URL = "https://github.com/razinkele/osmopy"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Same shape as tests/test_feedback_api.py: env first, `app` imported INSIDE the fixture.

    `OSMOSE_CONTACTS_FILE` is redirected too -- without it `save_contact` would write a real
    address into the repo's gitignored `data/feedback/contacts.jsonl`, where `git status` would
    never show it.
    """
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(tmp_path / "fb.jsonl"))
    monkeypatch.setenv("OSMOSE_CONTACTS_FILE", str(tmp_path / "contacts.jsonl"))
    monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")
    from app import app

    return TestClient(app.starlette_app)


def _pre_blocks(body: str) -> list[str]:
    """Contents of every `<pre>` element -- lets a test assert on the message, not the page."""
    return re.findall(r"<pre[^>]*>(.*?)</pre>", body, re.S)


def _get(client, token: str | None = "secret"):
    headers = {} if token is None else {"x-feedback-token": token}
    return client.get("/feedback/review", headers=headers)


def _param(url: str, name: str) -> str:
    """One decoded query parameter of `url`, or "" when absent.

    Deliberately total. A vacuity mutation that makes `github_issue_url` return "" must red each
    test on its OWN assertion; `parse_qs("")[name]` would raise KeyError instead, which the brief
    rules out as red for the wrong reason.
    """
    values = parse_qs(urlsplit(url).query, keep_blank_values=True).get(name, [])
    return values[0] if values else ""


def _promote_hrefs(body: str) -> list[str]:
    """Raw (still HTML-escaped) href of every per-record promotion link, in page order."""
    return re.findall(r'<a class="fb-promote" href="([^"]*)"', body)


def test_review_requires_token(client):
    append_feedback(build_feedback_record("bug", "token gate probe"))
    anon = _get(client, token=None)
    wrong = _get(client, token="nope")
    assert anon.status_code == 403
    assert wrong.status_code == 403
    for resp in (anon, wrong):
        assert "secret" not in resp.text  # the token itself must never come back
        assert "token gate probe" not in resp.text  # no record content without auth


def test_review_renders_records_and_escapes_html(client):
    append_feedback(build_feedback_record("bug", _XSS_MESSAGE, version="1.2.3", nav_tab="Results"))
    resp = _get(client)
    assert resp.status_code == 200
    body = resp.text

    blocks = _pre_blocks(body)
    assert len(blocks) == 1, f"expected exactly one message block, got {len(blocks)}"
    msg = blocks[0]
    assert _MARKER in msg  # positive control: this record's message really rendered...
    assert "&lt;script&gt;" in msg  # ... and rendered escaped
    assert "&lt;/pre&gt;" in msg  # ... including the block-breakout attempt
    assert "<script" not in msg
    assert "<script>alert" not in body  # page-wide: no raw markup anywhere


def test_review_links_to_the_repository(client):
    """R15: `repo_url` must be a live parameter, not a dead one the page ignores."""
    append_feedback(build_feedback_record("other", "repo link probe"))
    body = _get(client).text
    assert "repo link probe" in body  # positive control: the page rendered at all
    assert f'href="{_REPO_URL}"' in body


def test_review_escapes_record_metadata(client):
    # Hand-built (not via build_feedback_record) so that `ts`, `type` and `id` carry markup too:
    # append_feedback takes any dict, and a corrupt/hostile line on disk must render safely.
    append_feedback(
        {
            "id": 'abc"><b>ID_MARK</b>',
            "ts": '2026-09-14T12:00:00"><b>TS_MARK</b>',
            "type": 'bug"><b>TYPE_MARK</b>',
            "message": "metadata probe MSG_MARK",
            "has_contact": False,
            "version": '9.9"><img src=x onerror=alert(1)>VER_MARK',
            "nav_tab": 'Results"><b>TAB_MARK</b>',
        }
    )
    body = _get(client).text

    # Positive controls: every interpolated field actually reached the page.
    assert "metadata probe MSG_MARK" in body
    for mark in ("ID_MARK", "TS_MARK", "TYPE_MARK", "VER_MARK", "TAB_MARK"):
        assert mark in body, f"{mark} never rendered -- the negatives below would be vacuous"

    assert "&quot;&gt;&lt;b&gt;" in body  # the attribute-breakout sequence, escaped
    assert '"><b>' not in body  # ... and never raw
    assert "&lt;img src=x" in body  # the img payload survives only as inert text
    assert "<img" not in body  # ... never as a tag ("onerror=..." as text is harmless)


def test_review_never_renders_an_email_address(client):
    rec = build_feedback_record("suggestion", "please add EMAIL_PROBE_MSG", contact=_EMAIL)
    append_feedback(rec)
    save_contact(rec["id"], _EMAIL)
    # The address IS on disk and IS retrievable by id -- so this test can genuinely fail.
    assert lookup_contact(rec["id"]) == _EMAIL
    assert rec["has_contact"] is True

    body = _get(client).text
    assert "EMAIL_PROBE_MSG" in body  # positive control: the record rendered...
    assert 'data-has-contact="yes"' in body  # ... and is flagged as having a contact
    assert _EMAIL not in body
    assert "maintainer.probe" not in body
    assert "@example.org" not in body

    # The page now also carries a promotion href whose query is percent-encoded, so `@` appears
    # there as `%40` and the raw-string assertions above would be vacuous for it. Decode and
    # re-check. `maintainer.probe` is the part that survives quote_plus unchanged.
    hrefs = _promote_hrefs(body)
    assert len(hrefs) == 1, f"expected one promotion link, got {len(hrefs)}"
    decoded = _param(html.unescape(hrefs[0]), "body")
    assert "EMAIL_PROBE_MSG" in decoded  # positive control: the link really carries this record
    assert _EMAIL not in decoded
    assert "maintainer.probe" not in decoded


def test_review_survives_a_non_object_store_line(client, tmp_path):
    """One bad line must not deny the maintainer the whole page (the cost of the read raising).

    The page-level half of the `read_feedback` guard: the unit test proves the line is skipped,
    this proves the consequence that justified fixing it -- the good records still render.
    """
    append_feedback(build_feedback_record("bug", "record BEFORE the bad line"))
    with open(tmp_path / "fb.jsonl", "a", encoding="utf-8") as f:  # same path as the fixture env
        f.write("null\n")
    append_feedback(build_feedback_record("other", "record AFTER the bad line"))

    resp = _get(client)
    assert resp.status_code == 200  # not the 500 an unguarded read would produce
    assert "record BEFORE the bad line" in resp.text
    assert "record AFTER the bad line" in resp.text


def test_review_reports_only_a_real_boolean_as_having_contact(client):
    """`has_contact` is a FLAG, not a truthiness test: `"false"` is a non-empty string."""
    append_feedback(
        {
            "id": "flagprobe",
            "ts": "2026-09-14T12:00:00",
            "type": "bug",
            "message": "string-flag probe FLAG_MARK",
            "has_contact": "false",
            "version": "",
            "nav_tab": "",
        }
    )
    body = _get(client).text
    assert "FLAG_MARK" in body  # positive control: the record rendered
    assert 'data-has-contact="no"' in body
    assert 'data-has-contact="yes"' not in body


def test_review_response_is_not_cacheable(client):
    append_feedback(build_feedback_record("bug", "cache header probe"))
    resp = _get(client)
    assert "cache header probe" in resp.text  # positive control: this is the real page
    assert resp.headers.get("cache-control") == "no-store"


def test_review_error_path_returns_bare_internal(client, monkeypatch):
    import app as app_module

    def _boom(*args, **kwargs):
        raise RuntimeError("store exploded while holding token secret")

    monkeypatch.setattr(app_module, "read_feedback", _boom)
    resp = _get(client)
    assert resp.status_code == 500
    assert resp.text.strip() == "internal"
    assert "Traceback" not in resp.text
    assert "store exploded" not in resp.text


# ---------------------------------------------------------------------------------------------
# `github_issue_url` -- the promotion URL itself. Pure function, no page, no fixture.
#
# These are the feature's privacy boundary: the URL is the one artefact that carries a record's
# content to a PUBLIC destination. Every negative below is asserted against the GENERATED URL and
# paired with a positive control naming a benign substring of that record's own message, because a
# "no address" assertion passes trivially against a URL that was never built.
# ---------------------------------------------------------------------------------------------

_LEGACY_EMAIL = "legacy.probe@example.net"


def test_issue_url_targets_the_repo_and_sets_label():
    rec = {
        "type": "bug",
        "message": "boom ISSUE_MARK",
        "version": "1.2.3",
        "nav_tab": "run",
        "id": "abc",
    }
    url = github_issue_url(rec, _REPO_URL)
    assert url.startswith(f"{_REPO_URL}/issues/new?")
    assert _param(url, "labels") == "bug"
    assert _param(url, "title") == "[bug] boom ISSUE_MARK"
    body = _param(url, "body")
    assert "ISSUE_MARK" in body
    assert "`1.2.3`" in body and "`run`" in body and "`abc`" in body


@pytest.mark.parametrize(
    ("kind", "label"),
    [
        ("bug", "bug"),
        ("suggestion", "enhancement"),
        ("other", "question"),
        ("wat", "question"),  # unknown type -> the whitelist default, never the raw value
        (None, "question"),
    ],
)
def test_issue_type_maps_to_a_whitelisted_label(kind, label):
    url = github_issue_url({"type": kind, "message": "LABEL_MARK m"}, "https://x/y")
    assert "LABEL_MARK" in _param(url, "body")  # positive control: the URL was really built
    assert _param(url, "labels") == label


def test_issue_body_is_url_encoded():
    url = github_issue_url({"type": "bug", "message": "a&b c"}, "https://x/y")
    assert "a%26b" in url and " " not in url
    assert _param(url, "body").startswith("a&b c")  # ... and decodes back to the original


def test_issue_url_never_contains_an_email():
    """R16: a legacy v1 line still carrying `contact` must not leak it into a PUBLIC issue.

    `read_feedback` folds `contact` into `has_contact` before the page ever sees a record, but
    `github_issue_url` accepts any dict and must not depend on that normalisation having run --
    this is the last gate before the content leaves for github.com.
    """
    rec = {
        "id": "legacy1",
        "ts": "2026-09-14T12:00:00",
        "type": "bug",
        "message": "v1 record LEGACY_MSG_MARK",
        "contact": _LEGACY_EMAIL,  # the v1 field, still present on an old store line
        "has_contact": True,
        "version": "1.0.0",
        "nav_tab": "Run",
    }
    url = github_issue_url(rec, _REPO_URL)
    decoded_body = _param(url, "body")

    # Positive controls FIRST: the URL really was built, and really carries THIS record.
    assert "LEGACY_MSG_MARK" in decoded_body
    assert "`legacy1`" in decoded_body
    assert "LEGACY_MSG_MARK" in url  # underscores/letters survive quote_plus unchanged

    # Negatives against the DECODED body. `@` percent-encodes to `%40`, so a raw-URL check for the
    # whole address would pass even with the address present -- exactly the vacuous negative the
    # positive controls above exist to rule out.
    assert _LEGACY_EMAIL not in decoded_body
    assert "legacy.probe" not in decoded_body
    assert "example.net" not in decoded_body
    # `legacy.probe` and `example.net` are made only of quote_plus-safe characters, so unlike the
    # full address these two are also meaningful against the raw URL.
    assert "legacy.probe" not in url
    assert "example.net" not in url

    # The flag itself still travels, truthfully -- dropping the address is not dropping the fact.
    assert "- reporter left contact details: yes\n" in decoded_body


@pytest.mark.parametrize(
    "bad",
    [
        "javascript:alert(1)",
        "JavaScript:alert(1)",  # urlsplit lowercases the scheme; the gate must not be case-blind
        "data:text/html,<script>alert(1)</script>",
        "//evil.example",  # scheme-relative: parses to an EMPTY scheme, must not slip through
        "/relative/path",
        "",
        "ftp://evil.example/x",
    ],
)
def test_issue_url_rejects_a_non_http_scheme(bad):
    """R17: `html.escape` does not neutralise `javascript:` in an href -- validate, don't escape.

    Raising beats emitting a link the caller might escape and render anyway: escaping an attribute
    value cannot make its scheme safe, so there is no "safe" way to render one of these.
    """
    with pytest.raises(ValueError):
        github_issue_url({"type": "bug", "message": "SCHEME_MARK m"}, bad)


@pytest.mark.parametrize(
    "good",
    ["https://x/y", "http://x/y", "HTTPS://x/y", "https://github.com/razinkele/osmopy/"],
)
def test_issue_url_accepts_http_and_https(good):
    """The other half of the scheme gate: it must not reject the schemes it exists to allow."""
    url = github_issue_url({"type": "bug", "message": "SCHEME_MARK ok"}, good)
    assert "SCHEME_MARK" in _param(url, "body")  # positive control
    assert url.startswith(good.rstrip("/"))
    assert "/issues/new?" in url


@pytest.mark.parametrize(
    ("case", "rec"),
    [
        ("empty message", {"type": "bug", "message": "", "id": "deg1"}),
        ("whitespace-only message", {"type": "bug", "message": "   \n\t  ", "id": "deg2"}),
        ("missing message key", {"type": "bug", "id": "deg3"}),
        ("null message", {"type": "bug", "message": None, "id": "deg4"}),
        ("non-string message", {"type": "bug", "message": ["a", "b"], "id": "deg5"}),
        ("non-string type", {"type": ["bug"], "message": "x", "id": "deg6"}),
    ],
)
def test_issue_url_survives_a_degenerate_record(case, rec):
    """A hand-written or legacy store line must not blank the whole review page.

    `"".splitlines()` is `[]`, so the obvious `.strip().splitlines()[0]` raises IndexError on an
    empty or whitespace-only message; `.strip()` on a list raises AttributeError; an unhashable
    `type` raises TypeError from the label lookup. `build_feedback_record` produces none of these,
    but `read_feedback` does not validate field-by-field and this function is called once per card
    -- one bad line raising takes down every good record with it.
    """
    try:
        url = github_issue_url(rec, _REPO_URL)
    except Exception as exc:  # noqa: BLE001 -- ANY exception here is the failure under test
        pytest.fail(
            f"{case}: github_issue_url raised {exc!r}; one bad store line would blank "
            "the whole review page"
        )
    assert url.startswith(f"{_REPO_URL}/issues/new?")
    assert f"`{rec['id']}`" in _param(url, "body")  # positive control: this record, not a stub
    assert _param(url, "title").strip(), f"{case}: issue title is blank"
    assert _param(url, "labels"), f"{case}: issue label is blank"


@pytest.mark.parametrize(
    ("flag", "expected"),
    [(True, "yes"), (False, "no"), ("false", "no"), ("true", "no"), (1, "no"), (None, "no")],
)
def test_issue_url_reports_only_a_real_boolean_as_having_contact(flag, expected):
    """Matches the card (`is True`, not truthiness): `bool("false")` is True, and `1 is True` is
    False -- a hand-written string or integer flag must read as "no", not be believed."""
    url = github_issue_url(
        {"type": "bug", "message": "FLAG_URL_MARK", "has_contact": flag}, _REPO_URL
    )
    body = _param(url, "body")
    assert "FLAG_URL_MARK" in body  # positive control
    assert f"- reporter left contact details: {expected}\n" in body


# ---------------------------------------------------------------------------------------------
# The call site: the link has to actually be on the card (R2).
# ---------------------------------------------------------------------------------------------


def test_review_card_promotes_each_record_to_a_prefilled_issue(client):
    """R2: one promotion link PER RECORD, and the page-level repository link left alone."""
    rec_a = {
        "id": "wire-a",
        "ts": "2026-09-14T10:00:00",
        "type": "bug",
        "message": "WIRE_MARK_A crash on run",
        "has_contact": False,
        "version": "1.2.3",
        "nav_tab": "Run",
    }
    rec_b = {
        "id": "wire-b",
        "ts": "2026-09-14T11:00:00",
        "type": "suggestion",
        "message": "WIRE_MARK_B nicer charts",
        "has_contact": False,
        "version": "1.2.3",
        "nav_tab": "Results",
    }
    append_feedback(rec_a)
    append_feedback(rec_b)
    body = _get(client).text

    hrefs = [html.unescape(h) for h in _promote_hrefs(body)]
    assert len(hrefs) == 2, f"expected one promotion link per record, got {len(hrefs)}"
    # Literal controls, independent of `github_issue_url`: comparing only against the function's
    # own output would agree with itself under a mutation that empties it.
    assert "WIRE_MARK_B" in hrefs[0] and _param(hrefs[0], "labels") == "enhancement"
    assert "WIRE_MARK_A" in hrefs[1] and _param(hrefs[1], "labels") == "bug"
    # ... and then the exact wiring: each card's href IS that record's issue URL, newest first.
    assert hrefs == [github_issue_url(rec_b, _REPO_URL), github_issue_url(rec_a, _REPO_URL)]
    # The header link from Task 5 is untouched.
    assert f'href="{_REPO_URL}"' in body


def test_review_promotion_href_is_escaped_as_well_as_scheme_validated(client):
    """Both layers, asserted on the extracted attribute rather than the page.

    percent-encoding (urlencode) keeps the payload out of the URL grammar; HTML escaping (`_esc`)
    keeps urlencode's `&` separators out of the attribute grammar. Neither substitutes for the
    scheme gate, which is what the `javascript:`/`data:` tests above cover.
    """
    append_feedback(
        {
            "id": "hrefprobe",
            "ts": "2026-09-14T12:00:00",
            "type": "bug",
            "message": 'HREF_MARK "><img src=x onerror=alert(1)>',
            "has_contact": False,
            "version": "",
            "nav_tab": "",
        }
    )
    body = _get(client).text
    hrefs = _promote_hrefs(body)  # raw, still escaped
    assert len(hrefs) == 1, f"expected one promotion link, got {len(hrefs)}"
    href = hrefs[0]

    assert "HREF_MARK" in href  # positive control: this record's link was really built
    assert href.startswith(f"{_REPO_URL}/issues/new?")  # ... at the validated http(s) target
    # urlencode layer: the payload can only reach the attribute percent-encoded.
    assert "%3Cimg" in href
    assert "<img" not in href
    assert "%22%3E" in href  # the `"> ` attribute-breakout attempt
    # _esc layer: `&body=` is NOT a substring of `&amp;body=`, so these four assertions really do
    # discriminate between an escaped and an unescaped href.
    assert "&amp;body=" in href and "&amp;labels=" in href
    assert "&body=" not in href and "&labels=" not in href
