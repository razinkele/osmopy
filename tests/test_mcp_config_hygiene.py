from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# M9: generic credential sniffer over every MCP server source file.
#
# The tests further down pin the *one* credential that was actually burned.
# These two are the forward-looking guard: any future server under
# mcp_servers/ that assigns a literal to a credential-looking name, or embeds a
# pasted token, fails CI regardless of which service it talks to.
# ---------------------------------------------------------------------------

_CRED_ASSIGN = re.compile(
    r"""(?ix)
    \b(?:password|passwd|pwd|secret|token|api_?key|access_?key|credential)s?
    \s*(?::\s*[\w\[\], |]+)?      # optional type annotation
    \s*=\s*
    (?P<q>["'])(?P<val>[^"'\n]{6,})(?P=q)
    """
)
# Values that look like config plumbing rather than secrets.
_PLACEHOLDER = re.compile(
    r"^(?:\$\{[A-Z_][A-Z0-9_]*\}|<[^>]+>|your[-_ ].*|changeme|x{3,}|\*{3,}|\.{3})$", re.IGNORECASE
)
_ENV_VAR_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def _mcp_python_files() -> list[Path]:
    files = sorted((REPO_ROOT / "mcp_servers").rglob("*.py"))
    assert files, "expected at least one MCP server module under mcp_servers/"
    return files


def _shannon_entropy_bits(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def test_no_literal_credential_assignments_in_mcp_servers():
    """`password = "..."`-style literals are forbidden in every MCP server module.

    Env-var names (``"CMEMS_PASSWORD"``), ``${VAR}`` references and obvious
    placeholders are allowed; anything else that is assigned to a
    credential-looking name is a leak candidate.
    """
    offenders: list[str] = []
    for py in _mcp_python_files():
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            for m in _CRED_ASSIGN.finditer(line):
                val = m.group("val")
                if _PLACEHOLDER.match(val) or _ENV_VAR_NAME.match(val):
                    continue
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "literal credential assignment(s) in mcp_servers/:\n" + "\n".join(
        offenders
    )


def test_no_high_entropy_string_literals_in_mcp_servers():
    """Pasted tokens/keys that are not on a ``password = ...`` line still fail.

    Flags string constants that are >= 20 chars, contain no whitespace or path
    separators, mix upper/lower/digits, and have Shannon entropy >= 4.5 bits per
    character — the shape of API keys and session tokens, not of dataset IDs,
    URLs, or English prose.
    """
    offenders: list[str] = []
    for py in _mcp_python_files():
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            s = node.value
            if len(s) < 20 or any(ch.isspace() for ch in s) or "/" in s:
                continue
            if not (
                any(c.isupper() for c in s)
                and any(c.islower() for c in s)
                and any(c.isdigit() for c in s)
            ):
                continue
            if _shannon_entropy_bits(s) < 4.5:
                continue
            offenders.append(f"{py.relative_to(REPO_ROOT)}:{node.lineno}: {s[:12]}…")
    assert not offenders, "high-entropy string literal(s) in mcp_servers/:\n" + "\n".join(offenders)


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
