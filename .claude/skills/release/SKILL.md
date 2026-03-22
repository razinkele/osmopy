---
name: release
description: Cut a new release with version bump, changelog generation, git tag, and push
disable-model-invocation: true
---

Cut a new OSMOSE release.

## Arguments

- `bump` (required): "patch", "minor", or "major"
- `dry-run` (optional): if "yes", show what would happen without making changes

## Pre-flight Checks

Run ALL of these from `/home/razinka/osmose/osmose-python/` before releasing. If any fail, stop and report:

1. **Clean working tree**:
   ```
   git -C /home/razinka/osmose/osmose-python status --porcelain
   ```
   Must be empty. If not, ask user to commit or stash.

2. **Tests pass**:
   ```
   .venv/bin/python -m pytest -x -q
   ```

3. **Lint clean**:
   ```
   .venv/bin/ruff check osmose/ ui/ tests/
   ```

4. **On main branch**:
   ```
   git -C /home/razinka/osmose/osmose-python branch --show-current
   ```
   Must be "main". If not, warn and confirm with user.

## Release

If all checks pass (or dry-run=yes):

```
.venv/bin/python scripts/release.py {bump}
```

If dry-run=yes, append `--dry-run` to the command.

## Post-release

After a successful (non-dry-run) release:

1. Show the new version from `osmose/__version__.py`
2. Show the latest CHANGELOG.md entry
3. Ask user: "Push tag and commit to origin? (y/n)"
4. If yes:
   ```
   git -C /home/razinka/osmose/osmose-python push origin main --tags
   ```

## Rules

- Never skip pre-flight checks
- Never push without explicit user confirmation
- Use `.venv/bin/python`, never system python
