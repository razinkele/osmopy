---
name: Versioning system + release script mechanics
description: Where the version lives, how it propagates through pyproject/app, and the release script interface. Non-obvious chain of dynamic reads.
type: project
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
**Single source of truth:** `osmose/__version__.py` contains `__version__ = "x.y.z"`.

**Propagation chain:**

- `osmose/__init__.py` re-exports `__version__`.
- `pyproject.toml` reads it dynamically via `[tool.setuptools.dynamic]` (so no duplication, no pre-commit bumps).
- `app.py` header badge reads `f"v{__version__}"` at runtime.
- About modal version reads `__version__` dynamically.

**Release workflow:** `python scripts/release.py patch|minor|major [--changelog-only] [--dry-run]`.

- Auto-generates CHANGELOG.md from conventional commits (`feat:`, `fix:`, `docs:`, `test:`, etc.).
- Bumps `osmose/__version__.py`, commits, tags `vX.Y.Z`.

**Why:** Centralized version avoids drift between package metadata, UI display, and git tags — bump once, everything follows.

**How to apply:** When doing a release, never hand-edit version in multiple places. Use the script. When mentioning the current version in docs, prefer dynamic references where possible.
