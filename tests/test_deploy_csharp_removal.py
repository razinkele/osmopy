"""A deploy that deletes the C# engine without rebuilding it says so, and releases carry the flag.

Every deploy deletes the build a previous deploy made (a stale assembly would otherwise load
against new Python code), and only ``OSMOSE_DEPLOY_CSHARP=1`` rebuilds it. Production has had the
engine since 2026-11-21, so a deploy without the flag takes it away. Until this test's change that
showed only as ``C# engine: not built`` in the summary, the same line a server that never had the
engine prints.

The blocks are taken from deploy.sh and run under its own shell options, with ``as_shiny`` and
the message functions replaced by shell functions.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy.sh"
SKILL = ROOT / ".claude" / "skills" / "release" / "SKILL.md"

BUILT = "built from abc1234, self-tested, activated, offered by the app"


def _blocks() -> tuple[str, str, str]:
    """The removal before the checkout, the default result line, and the summary to the end.

    Cut at lines the change does not touch (the start of the removal, the checkout, the Step 6
    header and its opening ``if``), not at the conditions under test: a changed condition must
    fail an assertion below, not the cutting.
    """
    text = DEPLOY.read_text()
    a = text.index("CSHARP_REMOVED=0\n")
    removal = text[a : text.index('as_shiny git -C "$PROD_SRC" checkout ', a)]
    step6 = text[text.index("# --- Step 6 (opt-in): the C# engine ---") :].splitlines(keepends=True)
    first = next(i for i, ln in enumerate(step6) if not ln.startswith("#"))
    opening = next(
        i for i, ln in enumerate(step6) if ln.startswith('if [[ "$DEPLOY_CSHARP" == "1" ]]; then')
    )
    assert first < opening
    default = "".join(step6[first:opening])
    summary = text[text.index("# --- Summary ---") :]
    return removal, default, summary


def _deploy(tmp_path: Path, *, build: str | None, flag: str, rm_fails: bool = False):
    """Run the three blocks in order. ``build`` is "dir", "link" (to a directory outside the
    clone), "dangling" (a link to nothing) or None; with the flag on, the line Step 6 writes on
    success stands in for Step 6 itself."""
    clone = tmp_path / "clone"
    bin_dir = clone / "osmose" / "engine_cs" / "bin"
    bin_dir.parent.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    if build == "dir":
        bin_dir.mkdir()
        (bin_dir / "Osmose.Engine.dll").write_text("build")
    elif build == "link":
        elsewhere.mkdir()
        (elsewhere / "keep.txt").write_text("not part of the clone")
        bin_dir.symlink_to(elsewhere)
    elif build == "dangling":
        bin_dir.symlink_to(tmp_path / "nowhere")
    removal, default, summary = _blocks()
    step6 = f'CSHARP_RESULT="{BUILT}"' if flag == "1" else ":"
    script = f"""set -euo pipefail
PROD_SRC={shlex.quote(str(clone))}
DEPLOY_CSHARP={flag}
DEPLOY_REF=v9.9.9
DEPLOYED_SHA=abc1234
SERVICE_NAME=osmose-shiny
APP_PORT=8838
LINK_PATH=/srv/shiny-server/osmose
SOURCE_DIR=/home/dev/osmopy
SERVICE_FILE=/etc/systemd/system/osmose-shiny.service
info() {{ echo "INFO $*"; }}
warn() {{ echo "WARN $*"; }}
error() {{ echo "ERROR $*" >&2; }}
as_shiny() {{ if [[ "$1" == "rm" && "{int(rm_fails)}" == "1" ]]; then return 1; fi; "$@"; }}
{removal}
echo "AFTER_REMOVAL"
{default}
{step6}
{summary}
"""
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    still_there = bin_dir.exists() or bin_dir.is_symlink()
    return done, still_there, elsewhere


def _warnings(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.startswith("WARN ")]


@pytest.mark.parametrize("build", ["dir", "link", "dangling"])
def test_a_plain_deploy_that_deletes_the_engine_says_so_three_times(tmp_path, build):
    done, still_there, elsewhere = _deploy(tmp_path, build=build, flag="0")
    assert done.returncode == 0, done.stderr
    assert not still_there
    out = done.stdout
    # Where it happens, in the summary, and as the last lines on the screen.
    before, _, after = out.partition("AFTER_REMOVAL")
    assert any("WITHOUT the C# engine" in w for w in _warnings(before)), before
    assert "  C# engine:   REMOVED" in after
    last = _warnings(after)
    assert last and out.rstrip().endswith(last[-1])
    restore = "sudo OSMOSE_DEPLOY_REF=v9.9.9 OSMOSE_DEPLOY_CSHARP=1 bash /home/dev/osmopy/deploy.sh"
    assert restore in last[-1]
    if build == "link":
        # Only the link went: rm did not follow it out of the clone.
        assert (elsewhere / "keep.txt").read_text() == "not part of the clone"


def test_a_deploy_that_rebuilds_the_engine_does_not_warn(tmp_path):
    done, still_there, _ = _deploy(tmp_path, build="dir", flag="1")
    assert done.returncode == 0, done.stderr
    assert not still_there  # the old build goes either way; Step 6 makes the new one
    assert _warnings(done.stdout) == []
    assert "REMOVED" not in done.stdout
    assert f"  C# engine:   {BUILT}" in done.stdout


def test_a_server_that_never_had_the_engine_is_not_warned(tmp_path):
    done, still_there, _ = _deploy(tmp_path, build=None, flag="0")
    assert done.returncode == 0, done.stderr
    assert not still_there
    assert _warnings(done.stdout) == []
    assert "  C# engine:   not built (OSMOSE_DEPLOY_CSHARP=1 builds it)" in done.stdout


def test_a_removal_that_fails_stops_the_deploy_before_any_claim(tmp_path):
    # The old build then still matches the old ref: the deploy must stop, not go on to the
    # checkout, and must not say the build was removed.
    done, still_there, _ = _deploy(tmp_path, build="dir", flag="0", rm_fails=True)
    assert done.returncode != 0
    assert still_there
    assert "AFTER_REMOVAL" not in done.stdout
    assert _warnings(done.stdout) == []


def test_the_release_procedure_deploys_with_the_flag():
    """Production has the engine, so the release skill's deploy command must rebuild it."""
    commands = [
        ln.strip()
        for ln in SKILL.read_text().splitlines()
        if "bash /home/razinka/osmopy/deploy.sh" in ln and "OSMOSE_DEPLOY_REF=v{version}" in ln
    ]
    assert len(commands) == 1, commands
    assert "OSMOSE_DEPLOY_CSHARP=1" in commands[0].split("bash ")[0]
    # The name the skill sets is the one deploy.sh reads.
    assert 'DEPLOY_CSHARP="${OSMOSE_DEPLOY_CSHARP:-0}"' in DEPLOY.read_text()
