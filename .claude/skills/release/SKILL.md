---
name: release
description: Cut a new release with version bump, changelog section, release PR, tag on the merge commit, and deploy hand-off
disable-model-invocation: true
---

Cut a new OSMOSE release.

## Arguments

- `bump` (required): "patch", "minor", or "major"
- `dry-run` (optional): if "yes", show what would happen without making changes

## Pre-flight Checks

Run ALL of these from `/home/razinka/osmopy`. If any fail, stop and report:

1. **On master, up to date with origin, no tracked changes**:
   ```
   git -C /home/razinka/osmopy branch --show-current
   git -C /home/razinka/osmopy status --short --untracked-files=no
   git -C /home/razinka/osmopy fetch origin master
   git -C /home/razinka/osmopy rev-list --left-right --count origin/master...master
   ```
   Branch must be `master`. Status must be empty (untracked files are fine). Both `rev-list`
   counts must be `0`, or the release would be cut from a stale master.

2. **Master's own CI is green on the commit being released.** A release tag permanently stamps a
   version on one commit; don't stamp an unverified one. PR checks are not the same evidence:
   they ran on a different commit, and master's run includes coverage.
   ```
   gh run list --repo razinkele/osmopy --branch master --commit <HEAD sha>
   ```

## Release

```
git -C /home/razinka/osmopy switch -c release/v{version}
.venv/bin/python scripts/release.py {bump}
```

If dry-run=yes, skip the `switch` (stay on master), run only
`.venv/bin/python scripts/release.py {bump} --dry-run`, and stop there. A dry run exits 1 whenever
the real bump would.

`release.py` inserts only the new `## [X.Y.Z]` section and leaves released sections untouched.
Check that anyway, because it is the property that broke before (2026-11):
```
git -C /home/razinka/osmopy diff --numstat -- CHANGELOG.md
```
The second column (deletions) must be `0`. Do NOT use `--changelog-only` for a release; it rebuilds
the whole file from git and rewrites released history.

## Post-release

1. Commit `osmose/__version__.py` + `CHANGELOG.md` as `release: v{version}`, push the branch, open a
   PR. Ask the user before pushing.
2. When the PR's CI is green, merge it pinned to the head you checked
   (`gh pr merge N --merge --match-head-commit <sha>`).
3. Tag the **merge commit** (what master and prod carry, not the branch tip), annotated.
   The PR's CI ran on GitHub's merge ref, which is the same tree as the merge commit **only if master
   did not move** between that CI run and the merge. If it moved, wait until master's own CI
   succeeds on the merge commit before tagging:
   ```
   gh run list --repo razinkele/osmopy --branch master --workflow CI --commit <merge-commit> --status success
   ```
   An empty result means the run hasn't finished or didn't succeed. Do not tag yet.
   ```
   git -C /home/razinka/osmopy tag -a v{version} <merge-commit> -m v{version}
   git -C /home/razinka/osmopy push origin v{version}
   git -C /home/razinka/osmopy ls-remote origin "refs/tags/v{version}^{}"
   ```
   The `^{}` line must print the merge commit.
4. Deploy is the user's step (needs sudo and a TTY, so it can't go through `!`). Give them the
   **dev-checkout** path, never the prod clone's copy, and only after the tag is pushed:
   ```
   sudo OSMOSE_DEPLOY_REF=v{version} bash /home/razinka/osmopy/deploy.sh
   ```
5. Verify from the prod clone `/srv/shiny-server/osmose-src`:
   - `git describe --tags --exact-match` = v{version}, with no offset.
   - `osmose/__version__.py` matches.
   - `.git/FETCH_HEAD` mtime is new.
   - `systemctl show osmose-shiny -p ActiveEnterTimestamp` is after that fetch.
   - `GET :8838/` returns 200.
   - `GET :8838/feedback/review` and `/api/feedback` return **403**. A 200 there means the token
     gate broke.

## Rules

- Never skip pre-flight checks
- Never push without explicit user confirmation
- Use `.venv/bin/python`, never system python
