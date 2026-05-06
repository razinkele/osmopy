"""H11: importing app.py must not trigger filesystem side effects.

Pre-fix, `cleanup_old_temp_dirs()` and `register_cleanup()` ran at module
import time. Any test, type-check, or schema-only consumer that imported
`app.py` paid for a filesystem sweep + an atexit handler. H11 moves the
calls into a server-time guard.
"""

from __future__ import annotations

import importlib
import sys


def test_import_does_not_run_cleanup() -> None:
    """Re-import app and assert the cleanup-done flag is False after import."""
    # Force a clean re-import even if a prior test imported app.
    if "app" in sys.modules:
        del sys.modules["app"]
    app = importlib.import_module("app")
    assert app._cleanup_done is False, (
        "cleanup_old_temp_dirs() ran at import time — should be deferred to server()"
    )


def test_ensure_cleanup_registered_is_idempotent() -> None:
    """Calling _ensure_cleanup_registered twice must not re-cleanup."""
    if "app" in sys.modules:
        del sys.modules["app"]
    app = importlib.import_module("app")
    # Pre-state
    assert app._cleanup_done is False
    app._ensure_cleanup_registered()
    assert app._cleanup_done is True
    # Idempotent on second call
    app._ensure_cleanup_registered()
    assert app._cleanup_done is True
