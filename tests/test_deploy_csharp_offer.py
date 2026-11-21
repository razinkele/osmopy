"""After building the C# engine, deploy.sh restarts the app and requires it to OFFER the engine.

The app decides once, at import, whether to offer the engine (``CSHARP_ENGINE_OFFERED`` in
``app.py``). Every deploy removes the previous build before the checkout and Step 6 builds the
new one after the restart, so the process started in Step 4 never saw it. Measured 2026-11-21 on
the first C# deploy to production: the summary said "activated", the build loaded from a fresh
process, and the running app served no C# button until it was restarted by hand.

The probe is tested by running the function from deploy.sh, under the script's own shell options,
against a local server; the rest of Step 6 is checked by the order of its commands.
"""

from __future__ import annotations

import http.server
import re
import subprocess
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy.sh"
BUTTON = 'id="engineBtnCsharp"'


def _step6() -> str:
    text = DEPLOY.read_text()
    return text[
        text.index("# --- Step 6 (opt-in): the C# engine ---") : text.index("# --- Summary ---")
    ]


def _code(block: str) -> list[str]:
    return [s for s in (ln.strip() for ln in block.splitlines()) if s and not s.startswith("#")]


def _probe() -> str:
    """The ``app_offers_csharp`` function, as deploy.sh defines it."""
    found = re.search(
        r"^( *)app_offers_csharp\(\) \{\n.*?^\1\}\n", _step6(), re.DOTALL | re.MULTILINE
    )
    assert found, "deploy.sh no longer defines app_offers_csharp() in Step 6"
    return found.group(0)


class _Page(http.server.BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        try:
            self.wfile.write(self.body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # a client that stopped reading early: the case the probe must not create

    def log_message(self, *args):
        pass


@pytest.fixture
def serve():
    servers = []

    def start(body: bytes) -> int:
        handler = type("Handler", (_Page,), {"body": body})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1]

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _offers(port: int) -> bool:
    script = f"set -euo pipefail\nAPP_PORT={port}\n{_probe()}\napp_offers_csharp\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, timeout=60).returncode == 0


# The real page is a few hundred kB over thousands of lines. The filler makes a probe written as
# `curl | grep -q` fail the way it would there: grep leaves at the first matching LINE, curl dies
# writing the rest, and under `pipefail` (which deploy.sh sets) the pipeline is then non-zero
# although the button was found. Many lines, not one long one: grep reads a whole line before
# it can leave, so a single 4 MB line lets curl finish first and hides the fault.
_FILLER = b"\n".join([b"<p>" + b"x" * 100 + b"</p>"] * 40_000)


def test_the_probe_finds_the_button_in_a_page_that_has_it(serve):
    page = b"<html><button " + BUTTON.encode() + b">C#</button>\n" + _FILLER + b"\n</html>"
    assert _offers(serve(page)) is True


def test_the_probe_does_not_find_it_in_a_page_without_it(serve):
    page = b'<html><button id="engineBtnPython">Python</button>\n' + _FILLER + b"\n</html>"
    assert _offers(serve(page)) is False


def test_the_probe_fails_when_nothing_answers(serve):
    port = serve(b"")
    # Free the port again: nothing listens there now.
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    assert dead != port
    assert _offers(dead) is False


def test_the_probe_looks_for_the_id_the_app_renders():
    assert BUTTON in _probe()
    assert BUTTON in (ROOT / "app.py").read_text(), (
        "app.py no longer renders the C# button under this id; the deploy would then remove a "
        "good build as 'not offered'"
    )


def _index(lines: list[str], needle: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    raise AssertionError(f"not found after line {start}: {needle}")


def test_step6_restarts_after_activating_and_only_then_reports_the_engine():
    lines = _code(_step6())
    activated = _index(lines, 'as_shiny mv "$STAGE" "$LIVE_BIN"')
    restart = _index(lines, 'systemctl restart "$SERVICE_NAME"', activated)
    asked = _index(lines, "if app_offers_csharp; then", restart)
    reported = _index(lines, 'CSHARP_RESULT="built from', asked)
    assert activated < restart < asked < reported
    assert "offered by the app" in lines[reported]


def test_a_build_the_app_does_not_offer_is_removed_and_the_app_restarted_without_it():
    lines = _code(_step6())
    not_offered = _index(lines, 'if [[ "$OFFERED" != "1" ]]; then')
    removed = _index(lines, 'as_shiny rm -rf "$LIVE_BIN"', not_offered)
    restarted = _index(lines, 'systemctl restart "$SERVICE_NAME"', removed)
    failed = _index(lines, "exit 1", restarted)
    reported = _index(lines, 'CSHARP_RESULT="built from', not_offered)
    assert not_offered < removed < restarted < failed < reported
