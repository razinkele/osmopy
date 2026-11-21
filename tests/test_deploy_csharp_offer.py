"""After building the C# engine, deploy.sh restarts the app and requires it to OFFER the engine.

The app decides once, at import, whether to offer the engine (``CSHARP_ENGINE_OFFERED`` in
``app.py``). Every deploy removes the previous build before the checkout and Step 6 builds the
new one after the restart, so the process started in Step 4 never saw it. Measured 2026-11-21 on
the first C# deploy to production: the summary said "activated", the build loaded from a fresh
process, and the running app served no C# button until it was restarted by hand.

The probe is tested by running the function from deploy.sh, under the script's own shell options,
against a local server. The restart-and-check block is then run whole the same way, with
``systemctl`` and the other commands that need root replaced by shell functions: the order of
its commands alone cannot show an early exit under ``set -e``.
"""

from __future__ import annotations

import http.server
import re
import shlex
import socket
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


def _dead_port() -> int:
    """A port nothing listens on: bound once to get a free number, then released."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_probe_fails_when_nothing_answers():
    assert _offers(_dead_port()) is False


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


def _offer_block() -> str:
    """Step 6 from the probe's definition to the line that reports the engine, as written."""
    step = _step6()
    start = step.index("    app_offers_csharp() {")
    end = step.index('offered by the app"\n', start) + len('offered by the app"\n')
    return step[start:end]


def _run_block(port: int, tmp_path: Path, *, systemctl_rc: int = 0, rm_fails: bool = False):
    """Run the block under deploy.sh's shell options. ``systemctl`` only records its arguments
    and returns ``systemctl_rc``; ``as_shiny`` runs its command directly (or, with ``rm_fails``,
    refuses an ``rm``); ``sleep`` does nothing. Returns the process, whether the build is still
    there, and the systemctl calls."""
    live = tmp_path / "bin"
    live.mkdir()
    (live / "Osmose.Engine.dll").write_text("build")
    log = tmp_path / "systemctl.log"
    script = f"""set -euo pipefail
APP_PORT={port}
SERVICE_NAME=osmose-shiny
DEPLOYED_SHA=abc1234
LIVE_BIN={shlex.quote(str(live))}
info() {{ echo "INFO $*"; }}
error() {{ echo "ERROR $*" >&2; }}
sleep() {{ :; }}
as_shiny() {{ if [[ "$1" == "rm" && "{int(rm_fails)}" == "1" ]]; then return 1; fi; "$@"; }}
systemctl() {{ echo "$*" >> {shlex.quote(str(log))}; return {systemctl_rc}; }}
{_offer_block()}
echo "RESULT $CSHARP_RESULT"
"""
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=180)
    calls = log.read_text().splitlines() if log.exists() else []
    return done, live.exists(), calls


_WITH = b"<html><button " + BUTTON.encode() + b">C#</button></html>\n"
_WITHOUT = b'<html><button id="engineBtnPython">Python</button></html>\n'


def test_an_offered_engine_is_kept_and_reported(serve, tmp_path):
    done, build_kept, calls = _run_block(serve(_WITH), tmp_path)
    assert done.returncode == 0, done.stderr
    assert build_kept
    assert calls == ["restart osmose-shiny"]
    assert "RESULT built from abc1234, self-tested, activated, offered by the app" in done.stdout


def test_an_engine_the_app_does_not_offer_is_removed(serve, tmp_path):
    done, build_kept, calls = _run_block(serve(_WITHOUT), tmp_path)
    assert done.returncode == 1
    assert not build_kept
    assert calls == ["restart osmose-shiny", "restart osmose-shiny"]
    assert "the build was removed" in done.stderr
    assert "The app is running abc1234 WITHOUT the C# engine." in done.stderr
    assert "RESULT" not in done.stdout


def test_a_service_that_does_not_come_back_is_reported_as_not_answering(tmp_path):
    done, build_kept, calls = _run_block(_dead_port(), tmp_path)
    assert done.returncode == 1
    assert not build_kept
    assert len(calls) == 2
    assert "The service is NOT answering (HTTP 000)." in done.stderr
    assert "WITHOUT the C# engine" not in done.stderr


def test_a_failed_restart_still_removes_the_build_and_says_so(serve, tmp_path):
    # Under `set -e` a bare `systemctl restart` that fails would end the script on the spot:
    # the build left in place, and no word about the engine.
    done, build_kept, calls = _run_block(serve(_WITHOUT), tmp_path, systemctl_rc=1)
    assert done.returncode == 1
    assert not build_kept
    assert len(calls) == 2
    assert "the build was removed" in done.stderr


def test_a_build_that_cannot_be_removed_is_not_reported_as_removed(serve, tmp_path):
    done, build_kept, calls = _run_block(serve(_WITHOUT), tmp_path, rm_fails=True)
    assert done.returncode == 1
    assert build_kept
    assert len(calls) == 2
    assert "could NOT be removed" in done.stderr
    assert "the rollback is incomplete" in done.stderr
    assert "so the build was removed" not in done.stderr
    assert "WITHOUT the C# engine" not in done.stderr
