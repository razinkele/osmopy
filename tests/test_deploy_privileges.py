"""deploy.sh runs as root; it must never execute, or write through, anything shiny can modify.

The prod clone (``.git`` included) and the shared env (``/opt/micromamba/envs/shiny``: its
``python3``, ``pip`` and ``site-packages``) are owned by the service user. Root running git
there executes the clone's hooks, ``core.fsmonitor`` and filter drivers. Root running the env's
interpreter executes its ``.pth`` files, or a replaced binary. Root's ``cp``/``mkdir``/``mv``/
``rm`` there can be redirected by a planted symlink. Each is a path from a compromised service
account to root at the next deploy, so each must go through ``as_shiny`` (or ``as_service``).

deploy.sh needs root, systemd and the production paths, so it cannot run here. This is a
line-level lint of the script instead: every invocation of those commands is checked.
"""

from __future__ import annotations

import re
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / "deploy.sh"
WRAPPED = re.compile(r"\bas_(shiny|service)\b")
# A command position: line start, after ; & | ( { ! or $( , or after if/elif/then/do.
CMD = r"(?:^|[;&|({!]\s*|\$\(\s*|\b(?:if|elif|then|do)\s+(?:!\s*)?)"
GIT = re.compile(CMD + r"git\s")
ENV_TOOLS = re.compile(CMD + r'"(?:\$SHINY_PYTHON|\$SHINY_PIP|\$\{PIP_INSTALL\[@\]\})"')
WRITES_INTO_CLONE = re.compile(
    CMD + r"(?:cp|mkdir|mv|rm|ln|tee|touch)\b[^\n]*(?:\$\{?PROD_SRC\b|\$\{?LIVE_BIN\b)"
)
# Root's own git config, not the clone's; the uninstall's removal of the whole clone (rm -rf of
# a directory does not follow symlinks inside it); the served symlink, which lives in the
# root-owned /srv/shiny-server and only POINTS at the clone.
ALLOWED = (
    re.compile(r"^git config --global "),
    re.compile(r'^rm -rf "\$PROD_SRC"$'),
    re.compile(r'^ln -s "\$PROD_SRC" "\$LINK_PATH"$'),
)


def _code_lines(text: str) -> list[tuple[int, str]]:
    """Non-comment, non-message lines, stripped, with heredoc bodies skipped."""
    out, heredoc_end = [], None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if heredoc_end is not None:
            if line == heredoc_end:
                heredoc_end = None
            continue
        m = re.search(r"<<-?'?([A-Z_]+)'?", line)
        if m:
            heredoc_end = m.group(1)
        if not line or line.startswith("#"):
            continue
        if re.match(r"(info|warn|error|echo)\b", line):
            continue
        out.append((n, line))
    return out


def _violations(text: str) -> list[str]:
    found = []
    for n, line in _code_lines(text):
        if any(a.search(line) for a in ALLOWED) or WRAPPED.search(line):
            continue
        if line.startswith(("as_shiny()", "as_service()")):
            continue  # the helpers' own definitions
        if "=" in line.split()[0] and "$(" not in line:
            continue  # a plain assignment; one with $(...) runs a command and is checked
        for name, rx in (("git", GIT), ("env python/pip", ENV_TOOLS), ("write", WRITES_INTO_CLONE)):
            if rx.search(line):
                found.append(f"deploy.sh:{n}: root runs {name}: {line}")
    return found


def test_root_never_runs_git_the_env_or_writes_into_the_clone():
    assert _violations(DEPLOY.read_text()) == []


def test_every_runuser_child_has_no_controlling_terminal():
    """A shiny child sharing root's terminal could inject keystrokes into it (TIOCSTI) for
    root's shell to run after the deploy: every runuser call runs under setsid."""
    calls = [(n, line) for n, line in _code_lines(DEPLOY.read_text()) if "runuser" in line]
    assert calls, "no runuser call found: the helpers moved, update this test"
    bare = [f"deploy.sh:{n}: {line}" for n, line in calls if "setsid -w runuser" not in line]
    assert bare == []


def test_the_lint_catches_each_kind():
    """The lint itself: each pattern it must catch, and the forms it must allow."""
    bad = [
        'git -C "$PROD_SRC" checkout --quiet --force --detach "$DEPLOY_REF"',
        'DEPLOYED_SHA="$(git -C "$PROD_SRC" rev-parse --short HEAD)"',
        'if ! "$SHINY_PYTHON" -c "import pyvis"; then',
        '"${PIP_INSTALL[@]}" --quiet "packaging" || true',
        'cp -f "$JAR_STAGE/"*.jar "${PROD_SRC}/osmose-java/"',
        'rm -rf "$LIVE_BIN"',
    ]
    for line in bad:
        assert _violations(line), line
    good = [
        'as_shiny git -C "$PROD_SRC" checkout --quiet --force --detach "$DEPLOY_REF"',
        'DEPLOYED_SHA="$(as_shiny git -C "$PROD_SRC" rev-parse --short HEAD)"',
        'git config --global --fixed-value --unset-all safe.directory "$PROD_SRC" || true',
        'SHINY_PIP="/opt/micromamba/envs/shiny/bin/pip"',
        'info "Updating prod clone at ${PROD_SRC} (git fetch)..."',
        '# git -C "$PROD_SRC" checkout  (a comment)',
        'rm -rf "$PROD_SRC"',
        'ln -s "$PROD_SRC" "$LINK_PATH"',
        'PIP_INSTALL=("$SHINY_PIP" install --root-user-action=ignore)',
    ]
    for line in good:
        assert not _violations(line), line
