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
