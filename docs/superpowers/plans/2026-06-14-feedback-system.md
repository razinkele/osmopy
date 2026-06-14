# Feedback System Implementation Plan (Sub-project 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An in-app feedback modal (bug/suggestion) that appends to a JSONL store, plus a token-gated read-only `GET /osmose/api/feedback`.

**Architecture:** Pure `osmose/feedback.py` (build/append/read + total `check_feedback_token` + env-overridable path); a header "Feedback" modal + `feedback_server` (submit appends — no HTTP POST); a Starlette read route **inserted before** Shiny's catch-all `Mount("/")`. Core imports no web/UI.

**Tech Stack:** Python 3.12, Shiny for Python, Starlette (ships with Shiny), pytest + `starlette.testclient.TestClient` (httpx already transitive), Playwright.

**Spec:** `docs/superpowers/specs/2026-06-14-feedback-system-design.md`

---

## File Structure

- **Create** `osmose/feedback.py` (core) + `tests/test_feedback.py` (Task 1)
- **Create** `ui/components/feedback_modal.py` (Task 2)
- **Modify** `app.py` — imports, header link, modal placement, endpoint + route insert, server call (Task 3)
- **Modify** `.gitignore` — add `data/feedback/` (Task 3)
- **Create** `tests/test_feedback_api.py` (Task 3)
- **Modify** `tests/test_app_structure.py` — modal + wiring asserts (Tasks 2 & 3)
- **Create** `tests/test_e2e_feedback.py` + **Modify** `CHANGELOG.md` (Task 4)

Per-task gate: `.venv/bin/ruff check osmose/ ui/ tests/`, `.venv/bin/ruff format osmose/ ui/ tests/` (**not** app.py — out of scope), `.venv/bin/pyright` (`--pythonpath .venv/bin/python` when app.py is touched).

---

### Task 1: `osmose/feedback.py` core + unit tests

**Files:** Create `osmose/feedback.py`, `tests/test_feedback.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_feedback.py`:

```python
"""Unit tests for osmose.feedback (store + token check)."""

from __future__ import annotations

import pytest

from osmose.feedback import (
    append_feedback,
    build_feedback_record,
    check_feedback_token,
    read_feedback,
)


def test_build_record_fields():
    r = build_feedback_record(
        "bug", "  it broke  ", contact="me@x.io", version="0.13.0", nav_tab="run"
    )
    assert r["type"] == "bug" and r["message"] == "it broke"  # stripped
    assert r["contact"] == "me@x.io" and r["version"] == "0.13.0" and r["nav_tab"] == "run"
    assert r["id"] and r["ts"]


def test_build_record_unknown_type_raises():
    with pytest.raises(ValueError):
        build_feedback_record("spam", "x")


def test_build_record_empty_message_raises():
    with pytest.raises(ValueError):
        build_feedback_record("bug", "   ")


def test_build_record_truncates_long_message():
    r = build_feedback_record("bug", "x" * 6000)
    assert len(r["message"]) == 5000


def test_append_read_round_trip_newest_first(tmp_path):
    p = tmp_path / "fb.jsonl"
    append_feedback(build_feedback_record("bug", "first"), path=p)
    append_feedback(build_feedback_record("suggestion", "second"), path=p)
    out = read_feedback(path=p)
    assert [r["message"] for r in out] == ["second", "first"]  # newest-first


def test_read_missing_file_is_empty(tmp_path):
    assert read_feedback(path=tmp_path / "nope.jsonl") == []


def test_read_skips_corrupt_line(tmp_path):
    p = tmp_path / "fb.jsonl"
    append_feedback(build_feedback_record("bug", "ok"), path=p)
    with open(p, "a", encoding="utf-8") as f:
        f.write("{ not json\n")
    out = read_feedback(path=p)
    assert len(out) == 1 and out[0]["message"] == "ok"


def test_check_token_unset_env_is_false(monkeypatch):
    monkeypatch.delenv("OSMOSE_FEEDBACK_TOKEN", raising=False)
    assert check_feedback_token("anything") is False


def test_check_token_matching_and_mismatch(monkeypatch):
    monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")
    assert check_feedback_token("secret") is True
    assert check_feedback_token("wrong") is False
    assert check_feedback_token(None) is False


def test_check_token_non_ascii_is_false_not_raise(monkeypatch):
    monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")
    assert check_feedback_token("café") is False  # must not raise TypeError


def test_env_override_path(tmp_path, monkeypatch):
    p = tmp_path / "override.jsonl"
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(p))
    append_feedback(build_feedback_record("bug", "via env"))  # no path= → uses env
    assert [r["message"] for r in read_feedback()] == ["via env"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_feedback.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.feedback'`.

- [ ] **Step 3: Implement `osmose/feedback.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_feedback.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright osmose/feedback.py tests/test_feedback.py` → 0 errors.

- [ ] **Step 6: Commit**

```bash
git add osmose/feedback.py tests/test_feedback.py
git commit -m "feat(feedback): osmose/feedback store + token check (pure core)"
```

---

### Task 2: Feedback modal + submit handler

**Files:** Create `ui/components/feedback_modal.py`; Modify `tests/test_app_structure.py`

- [ ] **Step 1: Write the failing structure test**

In `tests/test_app_structure.py`, append:

```python
def test_feedback_modal_has_form_ids():
    from ui.components.feedback_modal import feedback_modal

    html = str(feedback_modal())
    assert "feedbackModal" in html
    for wid in ["feedback_type", "feedback_message", "feedback_contact", "feedback_submit"]:
        assert wid in html, f"Missing feedback widget id: {wid}"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py::test_feedback_modal_has_form_ids -q`
Expected: FAIL (`ModuleNotFoundError: ui.components.feedback_modal`).

- [ ] **Step 3: Implement `ui/components/feedback_modal.py`**

```python
"""Feedback modal (bug report / suggestion) + submit handler.

Reuses help_modal._bs_modal (static Bootstrap, header-triggered). Submission is a server-side
reactive.effect that appends to the feedback store — no HTTP POST. The read side is a
token-gated GET wired in app.py.
"""

from __future__ import annotations

from shiny import reactive, ui
from shiny.types import SilentException

from osmose import __version__
from osmose.feedback import append_feedback, build_feedback_record
from osmose.logging import setup_logging
from ui.components.help_modal import _bs_modal

_log = setup_logging("osmose.feedback_modal")


def feedback_modal():
    """The Send-feedback modal (header-triggered, static Bootstrap)."""
    body = ui.TagList(
        ui.input_radio_buttons(
            "feedback_type",
            "Type",
            {"bug": "Bug report", "suggestion": "Suggestion", "other": "Other"},
            selected="bug",
        ),
        ui.input_text_area(
            "feedback_message",
            "Message",
            rows=5,
            placeholder="What happened, or what would you like to see?",
            width="100%",
        ),
        ui.input_text("feedback_contact", "Contact (optional)", width="100%"),
        ui.input_action_button("feedback_submit", "Send", class_="btn-primary"),
        ui.tags.p(
            "Stored locally with the app version and current tab. Contact is optional.",
            class_="text-muted small mt-2",
        ),
    )
    return _bs_modal("feedbackModal", "Send feedback", body, size="lg")


def _safe_nav(input) -> str:
    try:
        return input.main_nav() or ""
    except (SilentException, AttributeError):
        return ""


def feedback_server(input, output, session, state):
    """Wire the submit handler. `output`/`state` unused; kept for call-signature uniformity."""

    @reactive.effect
    @reactive.event(input.feedback_submit)
    def _submit():
        msg = (input.feedback_message() or "").strip()
        if not msg:
            ui.notification_show("Enter a message before sending.", type="warning", duration=5)
            return
        try:
            rec = build_feedback_record(
                input.feedback_type(),
                msg,
                contact=(input.feedback_contact() or "").strip(),
                version=__version__,
                nav_tab=_safe_nav(input),
            )
            append_feedback(rec)
        except Exception:  # noqa: BLE001 — never crash the session on a save failure
            _log.error("feedback save failed", exc_info=True)
            ui.notification_show("Couldn't save feedback — try again.", type="error", duration=8)
            return
        ui.notification_show("Thanks — feedback saved.", type="message", duration=4)
        ui.update_text_area("feedback_message", value="")
        ui.update_text("feedback_contact", value="")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py::test_feedback_modal_has_form_ids -q`
Expected: PASS.

- [ ] **Step 5: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright ui/components/feedback_modal.py tests/test_app_structure.py` → 0 errors.

- [ ] **Step 6: Commit**

```bash
git add ui/components/feedback_modal.py tests/test_app_structure.py
git commit -m "feat(ui): feedback modal + submit handler (appends, no HTTP POST)"
```

---

### Task 3: app.py wiring + read API + .gitignore

**Files:** Modify `app.py`, `.gitignore`; Create `tests/test_feedback_api.py`; Modify `tests/test_app_structure.py`

- [ ] **Step 1: Write the failing API integration test**

Create `tests/test_feedback_api.py`:

```python
"""Integration test for the token-gated read-only feedback API route."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from osmose.feedback import append_feedback, build_feedback_record


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(tmp_path / "fb.jsonl"))
    monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")
    append_feedback(build_feedback_record("bug", "api round trip"))  # writes to env path
    from app import app

    return TestClient(app.starlette_app)


def test_no_token_forbidden(client):
    assert client.get("/api/feedback").status_code == 403


def test_wrong_token_forbidden(client):
    assert client.get("/api/feedback", headers={"X-Feedback-Token": "nope"}).status_code == 403


def test_correct_token_returns_records(client):
    r = client.get("/api/feedback", headers={"X-Feedback-Token": "secret"})
    assert r.status_code == 200
    assert any(rec["message"] == "api round trip" for rec in r.json())
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_feedback_api.py -q`
Expected: FAIL — the route isn't mounted yet (404, so the 403/200 assertions fail).

- [ ] **Step 3: Add imports to `app.py`**

Add to the import block at the top of `app.py` (near the other `from osmose...`/`from ui...` imports):

```python
from starlette.responses import JSONResponse
from starlette.routing import Route

from osmose.feedback import check_feedback_token, read_feedback
from ui.components.feedback_modal import feedback_modal, feedback_server
```

- [ ] **Step 4: Add the header "Feedback" link**

In `app.py`, immediately after the "Help" header anchor (the `ui.tags.a("Help", ... "#helpModal")` block), add:

```python
            ui.tags.a(
                "Feedback",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#feedbackModal"},
            ),
```

- [ ] **Step 5: Place the modal**

In `app.py`, after `changelog_modal(),` in the modals block, add:

```python
    feedback_modal(),
```

- [ ] **Step 6: Add the endpoint + mount the route (insert before the catch-all)**

In `app.py`, immediately after `app = App(app_ui, server, static_assets=_WWW)`, add:

```python


async def _feedback_endpoint(request):
    try:
        if not check_feedback_token(request.headers.get("x-feedback-token")):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        return JSONResponse(read_feedback())
    except Exception:  # noqa: BLE001 — never leak a traceback to an unauth caller
        return JSONResponse({"error": "internal"}, status_code=500)


# Mount the read-only feedback API BEFORE Shiny's catch-all Mount("/") — add_route would
# append AFTER it and the route would 404. See the feedback-system spec.
app.starlette_app.routes.insert(0, Route("/api/feedback", _feedback_endpoint, methods=["GET"]))
```

**Route path note:** the app serves the route at **`/api/feedback`** (what TestClient and the API test
hit). The externally-visible `/osmose/api/feedback` (goal/CHANGELOG prose) is the same route — the
`/osmose` prefix is added by the reverse-proxy / `--root-path /osmose` deploy, **not** in the route
definition. Do NOT change the route to `/osmose/api/feedback` (it would 404 under TestClient).

- [ ] **Step 7: Call the server wiring**

In `app.py`'s `server()`, after `diagnostics_server(input, output, session, state)`, add:

```python
    feedback_server(input, output, session, state)
```

- [ ] **Step 8: Run the API test to verify pass**

Run: `.venv/bin/python -m pytest tests/test_feedback_api.py -q`
Expected: PASS (3 passed — route reachable, token gate works).

- [ ] **Step 9: Add the structure/wiring test + .gitignore**

In `tests/test_app_structure.py`, append:

```python
def test_feedback_wired_into_app():
    from app import app_ui

    html = str(app_ui)
    assert "feedbackModal" in html
    assert 'data-bs-target="#feedbackModal"' in html
    for wid in ["feedback_type", "feedback_message", "feedback_contact", "feedback_submit"]:
        assert wid in html

    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "app.py").read_text()
    assert "feedback_server" in src
    assert 'Route("/api/feedback"' in src
```

In `.gitignore`, add (near `data/history/`):

```
data/feedback/
```

- [ ] **Step 10: Run structure test + verify gates**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py tests/test_feedback_api.py -q`
Expected: PASS (all).
Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/` (NOT app.py); `.venv/bin/pyright --pythonpath .venv/bin/python app.py tests/test_feedback_api.py tests/test_app_structure.py` → 0 errors.

- [ ] **Step 11: Commit**

```bash
git add app.py .gitignore tests/test_feedback_api.py tests/test_app_structure.py
git commit -m "feat(feedback): header modal + token-gated read API route (insert before catch-all)"
```

---

### Task 4: e2e + CHANGELOG

**Files:** Create `tests/test_e2e_feedback.py`; Modify `CHANGELOG.md`

- [ ] **Step 1: Write the e2e test**

Create `tests/test_e2e_feedback.py`:

```python
"""End-to-end test for the feedback modal (submit → JSONL store).

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_feedback.py -v -m e2e
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

_REPO = Path(__file__).resolve().parent.parent
# Point the app subprocess at a DEDICATED file (never the real feedback.jsonl). Set at import
# time so the create_app_fixture subprocess inherits it at launch.
_E2E_FILE = _REPO / "data" / "feedback" / "_e2e_feedback.jsonl"
os.environ["OSMOSE_FEEDBACK_FILE"] = str(_E2E_FILE)

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000


@pytest.fixture
def clean_e2e_file():
    _E2E_FILE.unlink(missing_ok=True)
    yield _E2E_FILE
    _E2E_FILE.unlink(missing_ok=True)


def test_feedback_submit_writes_record(page: Page, app: ShinyAppProc, clean_e2e_file):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # A startup changelog modal may overlay the header — dismiss it via its close button
    # (focus-independent) before clicking the header Feedback link.
    if page.locator("#changelogModal").is_visible():
        page.locator("#changelogModal [data-bs-dismiss='modal']").click()
        expect(page.locator("#changelogModal")).not_to_be_visible(timeout=_LOAD_TIMEOUT)

    page.get_by_role("link", name="Feedback").click()
    expect(page.locator("#feedbackModal")).to_be_visible(timeout=_LOAD_TIMEOUT)

    page.locator("#feedbackModal #feedback_message").fill("e2e bug report alpha")
    page.locator("#feedback_submit").click()

    # Success notification appears.
    expect(page.locator(".shiny-notification")).to_contain_text("saved", timeout=_LOAD_TIMEOUT)

    # The record landed in the dedicated store.
    def _written():
        return _E2E_FILE.is_file() and "e2e bug report alpha" in _E2E_FILE.read_text()

    page.wait_for_timeout(500)
    assert _written(), "feedback record was not written to the store"
```

- [ ] **Step 2: Run the e2e**

Run: `.venv/bin/python -m pytest tests/test_e2e_feedback.py -v -m e2e`
Expected: PASS (1 passed). If the notification selector differs, adjust to the actual Shiny notification container, but assert on the "saved" text + the written file.

- [ ] **Step 3: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added`, add:

```markdown
- **ui (feedback):** a header "Feedback" modal for bug reports / suggestions that appends to a JSONL
  store (`data/feedback/feedback.jsonl`), plus a token-gated read-only `GET /osmose/api/feedback`
  (disabled unless `OSMOSE_FEEDBACK_TOKEN` is set) for a maintainer to retrieve reports. New pure
  `osmose/feedback.py`; submission is in-session (no public write endpoint).
```

- [ ] **Step 4: Lint / format / type-check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`; `.venv/bin/ruff format osmose/ ui/ tests/`; `.venv/bin/pyright tests/test_e2e_feedback.py` → 0 errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_feedback.py CHANGELOG.md
git commit -m "test(feedback): e2e submit→store + CHANGELOG entry"
```

---

## Final verification (after all tasks)

- [ ] Full non-e2e suite: `.venv/bin/python -m pytest -m 'not e2e' -n auto -q`
- [ ] e2e: `.venv/bin/python -m pytest tests/test_e2e_feedback.py -v -m e2e`
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/` + `.venv/bin/ruff format --check osmose/ ui/ tests/` clean (app.py out of scope)
- [ ] `.venv/bin/pyright --pythonpath .venv/bin/python osmose/feedback.py ui/components/feedback_modal.py app.py tests/test_feedback.py tests/test_feedback_api.py tests/test_app_structure.py tests/test_e2e_feedback.py` → 0 errors
- [ ] Final whole-implementation review before finishing the branch.

## Self-Review (plan author)

- **Spec coverage:** core store + token (Task 1) ↔ spec §1; modal + submit (Task 2) ↔ §3; app.py header link + modal + endpoint + insert-before-Mount route + server call + .gitignore (Task 3) ↔ §2/§3/§4; tests (Tasks 1-4) ↔ spec Testing 1/2/3/4; CHANGELOG (Task 4) ↔ spec note. No spec requirement without a task.
- **Contract consistency:** `build_feedback_record`/`append_feedback`/`read_feedback`/`check_feedback_token`/`_resolve` signatures identical between `osmose/feedback.py` (Task 1), its tests (Task 1), the modal server (Task 2), and the endpoint (Task 3); the env keys `OSMOSE_FEEDBACK_FILE`/`OSMOSE_FEEDBACK_TOKEN` and the JSONL/header names are consistent across core, API test, and e2e; the route is mounted via `routes.insert(0, Route("/api/feedback", ...))` (NOT `add_route`) and the structure test greps `Route("/api/feedback"`.
- **Ordering / safety:** core first; the API test (Task 3) is the red→green guard for the route mount; the e2e uses a dedicated `OSMOSE_FEEDBACK_FILE` so the real `feedback.jsonl` is never clobbered; app.py is excluded from ruff and pyright uses `--pythonpath`.
- **No placeholders:** every code/edit step shows concrete content; commands have expected output.
