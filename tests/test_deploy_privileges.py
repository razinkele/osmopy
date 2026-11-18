"""deploy.sh runs as root; it must never execute, or write through, anything shiny can modify.

The prod clone (``.git`` included) and the shared env (``/opt/micromamba/envs/shiny``: its
``python3``, ``pip`` and ``site-packages``) are owned by the service user. Root running git
there executes the clone's hooks, ``core.fsmonitor`` and filter drivers. Root running the env's
interpreter executes its ``.pth`` files, or a replaced binary. Root's ``cp``/``mkdir``/``mv``/
``rm``/``install`` there, or a redirection root's shell opens there, can be redirected by a
planted symlink. Each is a path from a compromised service account to root at the next deploy,
so each must go through ``as_shiny`` (or ``as_service``).

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
CLONE = r"\$\{?(?:PROD_SRC|LIVE_BIN)\b"
WRITES_INTO_CLONE = re.compile(CMD + r"(?:cp|mkdir|mv|rm|ln|tee|touch|install)\b[^\n]*" + CLONE)
# A redirection into the clone is opened by ROOT's shell, even before a wrapped command runs.
REDIRECT_TARGET = re.compile(r'>>?\s*"?' + CLONE)
# Root's own git config, not the clone's; the uninstall's removal of the whole clone (rm -rf of
# a directory does not follow symlinks inside it); the served symlink, which lives in the
# root-owned /srv/shiny-server and only POINTS at the clone; creating the empty clone directory
# in that root-owned parent.
ALLOWED = (
    re.compile(r"^git config --global "),
    re.compile(r'^rm -rf "\$PROD_SRC"$'),
    re.compile(r'^ln -s "\$PROD_SRC" "\$LINK_PATH"$'),
    re.compile(r'^install -d -o shiny -g shiny "\$PROD_SRC"$'),
)


def _redirects_into_clone(line: str) -> bool:
    """A ``>``/``>>`` outside quotes whose target is in the clone (a quoted ``->`` is text)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == ">" and (i == 0 or line[i - 1] != ">") and REDIRECT_TARGET.match(line, i):
            return True
    return False


def _strip_comment(line: str) -> str:
    """The line without a trailing ``# comment`` (a ``#`` outside quotes, after whitespace)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line


def _code_lines(text: str, messages: bool = False) -> list[tuple[int, str]]:
    """Non-comment lines, stripped of trailing comments, heredoc bodies skipped; message lines
    (info, warn, error, echo) only when ``messages`` is set."""
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
        line = _strip_comment(line)
        if not line:
            continue
        if not messages and re.match(r"(info|warn|error|echo)\b", line):
            continue
        out.append((n, line))
    return out


def _violations(text: str) -> list[str]:
    found = [
        f"deploy.sh:{n}: root opens a redirection into the clone: {line}"
        for n, line in _code_lines(text, messages=True)
        if _redirects_into_clone(line)
    ]
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
        'install -d "$PROD_SRC/cache"',
        'as_shiny printf x > "${PROD_SRC}/flag"',  # wrapped, but root's shell opens the file
        'echo x >> "$LIVE_BIN/log"',
        'git -C "$PROD_SRC" fetch  # a trailing comment does not hide it',
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
        'install -d -o shiny -g shiny "$PROD_SRC"  # root-owned parent: root creates the empty dir',
        'install -m 644 "${SOURCE_DIR}/osmose-java/"*.jar "$JAR_STAGE/"',
        'rmdir "$PROD_SRC" 2>/dev/null || true  # only if git left it empty',
        "as_shiny find \"$SHINY_SITE_PACKAGES\" -maxdepth 1 -name '~*' 2>/dev/null",
        'warn "Could not provision the JARs # into ${PROD_SRC}"',  # a # inside quotes
        'info "Creating prod clone: ${REPO_URL} -> ${PROD_SRC}"',  # a quoted -> is text
    ]
    for line in good:
        assert not _violations(line), line
