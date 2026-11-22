"""Environment for subprocesses that must import THIS checkout's ``osmose``.

A child ``python script.py`` puts the script's own directory at ``sys.path[0]``, not the repo,
so it imports ``osmose`` only if the package is installed. CI installs it (``pip install -e``);
the shared micromamba ``shiny`` env deliberately does not (installing a dev checkout there
would shadow production's own clone). Prepending the repo root to ``PYTHONPATH`` makes the
child import this checkout either way.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p)
    return env
