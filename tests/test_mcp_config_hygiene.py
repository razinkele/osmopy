from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mcp_json_has_no_cmems_password():
    cfg = json.loads((REPO_ROOT / ".mcp.json").read_text())
    text = json.dumps(cfg)
    assert "CMEMS_PASSWORD" not in text or '"CMEMS_PASSWORD": ""' in text, (
        ".mcp.json must not ship a CMEMS_PASSWORD value"
    )
    assert "Razinka@2026" not in text


def test_env_example_documents_cmems_vars():
    p = REPO_ROOT / ".env.example"
    assert p.exists(), ".env.example must document required env vars"
    body = p.read_text()
    assert "CMEMS_USERNAME" in body
    assert "CMEMS_PASSWORD" in body


def test_burned_credential_absent_from_whole_tracked_tree():
    """The burned CMEMS/ICES password must not appear in ANY tracked file except
    the two hygiene detectors that scan for it.

    The original scan only checked ``.mcp.json`` — which is why the literal leaked
    into plan docs undetected (deep-review v2, 2026-06-20). This greps the whole
    tracked tree so any re-introduction (docs, code, config) fails CI.
    """
    import subprocess

    burned = "Razinka@2026"  # this file is in the allow-list below, so its own copy is fine
    out = subprocess.run(
        ["git", "grep", "--no-color", "-l", burned],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    matches = {line.strip() for line in out.stdout.splitlines() if line.strip()}
    allowed = {
        "tests/test_mcp_config_hygiene.py",
        "tests/test_copernicus_mcp_env.py",
    }
    leaked = matches - allowed
    assert not leaked, (
        f"burned credential literal found in tracked file(s) outside the detectors: "
        f"{sorted(leaked)} — redact and rotate"
    )


def test_mcp_json_has_no_literal_cmems_credentials():
    """Detect literal CMEMS credentials under any server's env block.

    Allows empty-string placeholders and `${ENV_VAR}` references;
    rejects anything else (including the rotated password).
    """
    import re

    env_ref = re.compile(r"^\$\{[A-Z_][A-Z0-9_]*\}$")
    cfg = json.loads((REPO_ROOT / ".mcp.json").read_text())
    for srv_name, srv in cfg.get("mcpServers", {}).items():
        env = srv.get("env") or {}
        for var, val in env.items():
            if var.upper() not in {"CMEMS_PASSWORD", "CMEMS_USERNAME"}:
                continue
            assert val == "" or env_ref.match(str(val)), (
                f"literal CMEMS credential found in .mcp.json under "
                f"mcpServers.{srv_name}.env.{var}: {val!r}"
            )
