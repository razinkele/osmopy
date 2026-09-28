"""Drive ``mcp_servers/copernicus/server.py`` over stdio JSON-RPC under a given interpreter.

Purpose: confirm which ``fastmcp`` majors the server actually runs on, so the ``[mcp]``
extra's ``fastmcp`` bound in ``pyproject.toml`` is a MEASURED range rather than a guess.
It is version-agnostic on purpose -- fastmcp's Python API for listing tools changed
between majors, but the MCP wire protocol is what a client speaks -- and it needs no
credentials, because tool LISTING and the credential-free ``list_datasets`` tool are
what it exercises.

Usage::

    uv venv /tmp/fm-4 --python 3.12
    uv pip install --python /tmp/fm-4/bin/python "fastmcp==4.0.10" "copernicusmarine>=2,<3" python-dotenv
    .venv/bin/python scripts/probe_copernicus_mcp_stdio.py /tmp/fm-4/bin/python

Exit status 0 = initialize + tools/list + tools/call(list_datasets) all succeeded.
Verified 2026-09-28 against fastmcp 2.0.0, 2.14.7, 3.4.7 and 4.0.10 (5 tools, identical
3426-char ``list_datasets`` payload on every one).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "mcp_servers" / "copernicus" / "server.py"
EXPECTED_TOOLS = {
    "check_credentials",
    "download_field",
    "generate_osmose_ltl",
    "generate_osmose_physics",
    "list_datasets",
}


def _recv(proc: subprocess.Popen, want_id: int, limit: int = 50) -> dict | None:
    for _ in range(limit):
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line.startswith("{"):
            continue  # fastmcp banners / logging on stdout in some versions
        msg = json.loads(line)
        if msg.get("id") == want_id:
            return msg
    return None


def main(py: str) -> int:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMEMS")}
    env["PYTHONUNBUFFERED"] = "1"

    ver = subprocess.run(
        [py, "-c", "import importlib.metadata as m; print(m.version('fastmcp'))"],
        capture_output=True,
        text=True,
        env=env,
    )
    print("fastmcp:", ver.stdout.strip() or ver.stderr.strip()[-200:])

    proc = subprocess.Popen(
        [py, str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    def send(obj: dict) -> None:
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def fail(msg: str) -> int:
        print("FAIL:", msg)
        print("--- server stderr (tail) ---")
        print(proc.stderr.read()[-1500:])
        return 1

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "0"},
                },
            }
        )
        init = _recv(proc, 1)
        if init is None or "result" not in init:
            return fail(f"no initialize result: {init}")
        print("server:", init["result"].get("serverInfo"))

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tl = _recv(proc, 2)
        if tl is None or "result" not in tl:
            return fail(f"tools/list: {tl}")
        names = {t["name"] for t in tl["result"]["tools"]}
        print("tools:", len(names), sorted(names))
        if names != EXPECTED_TOOLS:
            return fail(f"tool set differs from expected: {sorted(names ^ EXPECTED_TOOLS)}")

        send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "list_datasets", "arguments": {}},
            }
        )
        call = _recv(proc, 3)
        if call is None or "result" not in call:
            return fail(f"tools/call list_datasets: {call}")
        content = call["result"].get("content", [])
        text = next((c.get("text", "") for c in content if c.get("type") == "text"), "")
        is_error = call["result"].get("isError", False)
        print(f"call list_datasets: isError={is_error}, {len(text)} chars, starts {text[:50]!r}")
        if is_error or not text:
            return fail("list_datasets returned an error or empty text")
        print("OK")
        return 0
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
