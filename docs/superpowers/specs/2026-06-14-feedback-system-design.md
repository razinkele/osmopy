# Feedback system — design (Sub-project 2)

**Date:** 2026-06-14
**Status:** Approved (design phase)
**Feature:** An in-app feedback form (bug reports + suggestions) that persists to a JSONL store,
plus a token-protected backend REST endpoint to retrieve the records.

This is sub-project 2 of a two-part request; sub-project 1 (docs-in-app) shipped to origin/master
(`044db54`).

## Motivation

There is no way for users to report bugs or suggest improvements from inside the running app, and no
way for a maintainer to collect that signal. This adds a lightweight feedback channel: a header
"Feedback" modal that writes structured records to disk, and a read-only HTTP endpoint (disabled by
default, token-gated) so a maintainer can pull the reports programmatically.

## Decisions (locked during brainstorming)

1. **API surface = read-only + UI submit.** Submission happens only through the in-app Shiny form
   (inside the authenticated WebSocket session); the backend exposes a **token-protected `GET`** to
   retrieve records. **No public POST** (avoids an unauthenticated write/abuse/CSRF surface).
2. **Storage = JSONL append log** at `data/feedback/feedback.jsonl` (one JSON object per line) —
   matches the repo's JSON-on-disk ethos, zero new deps, append-friendly for low-volume manual input.
3. **UI = header "Feedback" link → Bootstrap modal** (next to About/Help), consistent with the modals
   shipped in sub-project 1.

## Reuse (do not rebuild)

- `ui/components/help_modal.py:_bs_modal(modal_id, title, body, *, size)` — the static Bootstrap-5
  modal builder (client-side `data-bs-toggle`); the feedback modal reuses it.
- `app.py` (anchor by stable text, not line numbers — app.py is out of ruff scope and drifts): the
  header About/Help `data-bs-toggle="modal"` anchors block; the modal placement block (after the
  `changelog_modal()` call); the `app = App(app_ui, server, static_assets=_WWW)` line; the page-server
  `*_server(input, output, session, state)` calls in `server()`; `from osmose import __version__`
  (already imported); the navset `id="main_nav"`. Verified: Shiny `App` exposes a stable
  `.starlette_app` instance — but route mounting must **insert before** its catch-all `Mount` (see §2),
  not `add_route`.
- `osmose/history.py` — the JSON-on-disk + path-safety + `_PROJECT_ROOT` precedent to mirror.
- Form idiom (`ui/pages/scenarios.py:23-28,105-143`): `input_text`/`input_text_area`/
  `input_radio_buttons` + `input_action_button` + `@reactive.event` + `ui.notification_show` +
  `ui.update_*` to clear fields.
- `.env` credential convention (gitignored; `mcp_servers/copernicus/server.py` `_require_creds`) — the
  pattern for `OSMOSE_FEEDBACK_TOKEN`.
- `osmose/__version__.py` (`__version__`) and `input.main_nav()` (the navset id) for auto-captured
  context.

## Architecture — four units (core has NO web imports)

### 1. `osmose/feedback.py` (pure core — file I/O + token check, unit-tested)

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[1]          # osmose/feedback.py -> repo root
FEEDBACK_FILE = _PROJECT_ROOT / "data" / "feedback" / "feedback.jsonl"   # default
_VALID_TYPES = {"bug", "suggestion", "other"}
_MAX_MESSAGE = 5000
_TOKEN_ENV = "OSMOSE_FEEDBACK_TOKEN"
_FILE_ENV = "OSMOSE_FEEDBACK_FILE"   # optional override (relocate store; test/e2e isolation)

def _resolve(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(_FILE_ENV)
    return Path(env) if env else FEEDBACK_FILE

def build_feedback_record(
    type: str, message: str, *, contact: str = "", version: str = "", nav_tab: str = ""
) -> dict: ...
def append_feedback(record: dict, *, path: Path | None = None) -> None: ...
def read_feedback(*, path: Path | None = None) -> list[dict]: ...
def check_feedback_token(provided: str | None) -> bool: ...
```

- `build_feedback_record`: strips `message`; raises `ValueError` if empty or if `type not in
  _VALID_TYPES`; **truncates** `message` to `_MAX_MESSAGE` (never lose a report). Returns
  `{"id": uuid4().hex, "ts": datetime.now().isoformat(), "type", "message", "contact", "version",
  "nav_tab"}`.
- `append_feedback`: `p = _resolve(path)`; `p.parent.mkdir(parents=True, exist_ok=True)`; append
  `json.dumps(record) + "\n"` with `open(p, "a", encoding="utf-8")`. **Concurrency:** the deploy is
  single-worker (`uvicorn app:app`, no `--workers` — `deploy.sh`), so a synchronous append from the
  one event loop never interleaves; on POSIX, wrap the write in an `fcntl.flock(f, LOCK_EX)`
  (guarded `try: import fcntl`) as cheap belt-and-suspenders. (See the single-worker note under
  Error handling — multi-worker would require the lock.)
- `read_feedback`: `p = _resolve(path)`; missing file → `[]`; parse each line, **skip corrupt**
  (`# noqa: BLE001` continue); return **newest-first** (reverse insertion order). Resolves the path at
  **call time** (never bind it as a default arg) so the env override / monkeypatch takes effect.
- `check_feedback_token`: `tok = os.environ.get(_TOKEN_ENV)`; return `False` if `tok` is falsy
  (endpoint disabled by default) or `provided` is None; else **compare on UTF-8 bytes**
  `secrets.compare_digest(provided.encode("utf-8"), tok.encode("utf-8"))`. **Byte comparison is
  mandatory** — `compare_digest` on `str` raises `TypeError` for non-ASCII input, and a header byte
  ≥ 0x80 decodes (latin-1) to a non-ASCII `str`, so a `str` compare would let an unauthenticated
  request crash the handler (500/DoS). `check_feedback_token` must be **total** (never raise). Reads
  the env at call time (tests monkeypatch via `setenv`).

### 2. Read API (wired in `app.py`, web layer)

```python
async def _feedback_endpoint(request):
    try:
        if not check_feedback_token(request.headers.get("x-feedback-token")):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        return JSONResponse(read_feedback())
    except Exception:  # noqa: BLE001 — never leak a traceback to an unauth caller
        return JSONResponse({"error": "internal"}, status_code=500)
```

**Mounting (verified subtlety — do NOT use `add_route`):** Shiny's `App.__init__` builds
`starlette_app` ending with a catch-all `Mount("/", app=self._dependency_handler)`. Starlette matches
in order, so `app.starlette_app.add_route(...)` **appends after the catch-all and the route 404s**
(confirmed empirically via `TestClient`). Instead **insert before** the catch-all, after `app = App(...)`:

```python
from starlette.routing import Route
app.starlette_app.routes.insert(0, Route("/api/feedback", _feedback_endpoint, methods=["GET"]))
```

Verified: `TestClient(app).get("/api/feedback")` → 200 and the Shiny root `/` still serves. Externally
`GET /osmose/api/feedback` (the `--root-path /osmose` deploy). `from starlette.responses import
JSONResponse` (starlette ships with Shiny; `httpx`, which `starlette.testclient.TestClient` needs, is
already a transitive dep). Read-only; no POST route.

### 3. Feedback modal + submit (`ui/components/feedback_modal.py`)

- `feedback_modal()` — reuses `_bs_modal("feedbackModal", "Send feedback", body, size="lg")`; body =
  `input_radio_buttons("feedback_type", "Type", {"bug": "Bug report", "suggestion": "Suggestion",
  "other": "Other"}, selected="bug")`, `input_text_area("feedback_message", "Message", rows=5,
  placeholder=...)`, `input_text("feedback_contact", "Contact (optional)")`,
  `input_action_button("feedback_submit", "Send", class_="btn-primary")`, and a small privacy note
  ("Stored locally; contact is optional"). Imports `_bs_modal` from `ui.components.help_modal`.
- `feedback_server(input, output, session, state)`:
  ```python
  @reactive.effect
  @reactive.event(input.feedback_submit)
  def _submit():
      msg = (input.feedback_message() or "").strip()
      if not msg:
          ui.notification_show("Enter a message before sending.", type="warning", duration=5)
          return
      try:
          rec = build_feedback_record(
              input.feedback_type(), msg,
              contact=(input.feedback_contact() or "").strip(),
              version=__version__, nav_tab=_safe_nav(input),
          )
          append_feedback(rec)
      except Exception:  # noqa: BLE001
          _log.error("feedback save failed", exc_info=True)
          ui.notification_show("Couldn't save feedback — try again.", type="error", duration=8)
          return
      ui.notification_show("Thanks — feedback saved.", type="message", duration=4)
      ui.update_text_area("feedback_message", value="")
      ui.update_text("feedback_contact", value="")
  ```
  `_safe_nav(input)` reads `input.main_nav()` defensively (`try/except (SilentException, AttributeError)`
  → ""; import `from shiny.types import SilentException`).
- `app.py` wiring: a header "Feedback" link in the `app.py:218-229` block
  (`data-bs-target="#feedbackModal"`); `feedback_modal()` placed with the other modals
  (`app.py:293-296`); `feedback_server(input, output, session, state)` called in `server()`.

### 4. `.gitignore`

Add `data/feedback/` (user-submitted content with optional contact PII — gitignored like
`data/history/`).

## Data flow

```
#feedbackModal form ──submit──► feedback_server @reactive.event(input.feedback_submit)
   └─ build_feedback_record(type, message, contact, version=__version__, nav_tab=input.main_nav())
      └─ append_feedback(rec) ──► data/feedback/feedback.jsonl   (one JSON line)
         └─ notification_show + clear message/contact fields

maintainer ──GET /osmose/api/feedback  (X-Feedback-Token: <env token>)──►
   check_feedback_token(header)  ?  JSONResponse(read_feedback())  :  403
```

No public write path. Core (`osmose/feedback.py`) imports no web/UI; `app.py` is the only HTTP layer.

## Error handling

- `append_feedback` mkdir+append; the submit handler wraps it (`# noqa: BLE001`) → error notification,
  never crashes the session.
- `read_feedback`: missing file → `[]`; corrupt line → skipped.
- `build_feedback_record`: `ValueError` on empty message / bad type (UI checks empty first); message
  truncated to 5000.
- API: token unset/mismatch → 403; **non-ASCII / malformed token → 403** (byte-compare, never raises);
  read error → 500 JSON; the whole handler body is wrapped so an unauth caller never sees a traceback.
- `_safe_nav`: `input.main_nav()` guarded (`SilentException`/`AttributeError` → "").
- **Single-worker invariant:** JSONL append is interleave-safe because the deploy runs one uvicorn
  worker (one event loop). This is a deliberate constraint — **do not add `--workers`/gunicorn workers
  without** switching to a file lock (`fcntl.flock`) or another store. The append uses a guarded
  `fcntl.flock` on POSIX as belt-and-suspenders.

## Testing

1. **Unit `tests/test_feedback.py`**: `build_feedback_record` (returns id/ts/type/message/contact/
   version/nav_tab; unknown type → `ValueError`; empty/whitespace message → `ValueError`; a
   >5000-char message is truncated to 5000); `append_feedback`+`read_feedback` round-trip
   (`path=tmp_path` — one line per record, newest-first, skips a deliberately corrupt line; missing
   file → `[]`); `check_feedback_token` (env unset via `monkeypatch.delenv` → False; set+matching →
   True; set+wrong → False; `provided=None` → False; **a non-ASCII `provided` (e.g. "café") → False
   without raising** — the byte-compare guard).
2. **API integration `tests/test_feedback_api.py`**: `from starlette.testclient import TestClient`
   (httpx is already a transitive dep); `from app import app`; `client = TestClient(app.starlette_app)`.
   Isolate the store via **env override** — `monkeypatch.setenv("OSMOSE_FEEDBACK_FILE", str(tmp))` and
   `monkeypatch.setenv("OSMOSE_FEEDBACK_TOKEN", "secret")`, then `append_feedback(rec)` (writes to the
   tmp path via the env). `GET /api/feedback` with **no** header → 403; **wrong** token → 403;
   **correct** `X-Feedback-Token: secret` → 200 with the record in the JSON. (Proves the route is
   reachable on the wired app — the guardrail for the insert-before-Mount mounting.) Env override is
   used instead of monkeypatching `FEEDBACK_FILE` so the endpoint and the test agree on the path.
3. **Structure `tests/test_app_structure.py`**: `str(app_ui)` contains `feedbackModal`, the four form
   ids (`feedback_type`/`feedback_message`/`feedback_contact`/`feedback_submit`), and a header
   `data-bs-target="#feedbackModal"` link; `app.py` source contains `feedback_server` and
   `Route("/api/feedback"` (the insert-before-Mount mount — **not** `add_route`).
4. **e2e `tests/test_e2e_feedback.py`**: set `os.environ["OSMOSE_FEEDBACK_FILE"]` at **module import**
   (before the `create_app_fixture` subprocess launches) to a **dedicated** repo path
   (`data/feedback/_e2e_feedback.jsonl`, NOT the real `feedback.jsonl`) so the subprocess writes there
   and a real maintainer file is never touched. Open Feedback from the header (click the header link —
   focus-independent), select a type, fill the message, click Send → assert a success notification;
   then read the dedicated file and assert the submitted message is present. Fixture `unlink(missing_ok)`
   the dedicated file after the yield (cleanup even on failure). **Note (lesson from sub-project 1):**
   dismiss any startup changelog modal first if it overlays the header (click its close button — not
   Escape), and assert on notification/content rather than racing visibility.

**Gates:** every task runs `ruff check`, `ruff format` (osmose/ ui/ tests/ — **not** app.py, outside
scope by design), and `pyright` (`--pythonpath .venv/bin/python` when app.py is touched). **CHANGELOG**
entry under `[Unreleased]`.

## Out of scope (YAGNI)

- A public POST API; an in-app admin viewer of feedback (the read endpoint serves that).
- Accounts/auth beyond the single env token; email/Slack notification on new feedback.
- Rate-limiting middleware (submission is WebSocket-session-gated; the 5000-char cap bounds size).
- Edit/delete of records via API (read-only); attachments/screenshots.
- Migrating the store to SQLite (JSONL chosen; revisit only if volume demands queries).
