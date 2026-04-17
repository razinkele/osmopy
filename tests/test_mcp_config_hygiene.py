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
