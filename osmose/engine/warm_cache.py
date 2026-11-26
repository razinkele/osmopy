"""Fill the Numba cache: one in-memory year of the shipped Baltic model.

Numba 0.65 (pinned in pyproject.toml) compiles this engine's kernels in about 90 s on a cold
cache, against ~16 s with 0.66+, which in exchange runs them ~50% slower
(``docs/perf/2026-11-26-environment-and-collection.md``). The compiled code is cached next to the
sources, so paying it once, ahead of time, keeps it off a user's first run:

* ``deploy.sh`` runs this as the service user after the restart (a deploy that changed engine
  code invalidates the cache entries of the changed files);
* CI runs it before the browser tests, whose first Baltic run in the UI has a 120 s budget.

Usage: ``python -m osmose.engine.warm_cache`` (exit status 0 on success).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    master = Path(__file__).resolve().parents[2] / "data" / "baltic" / "baltic_all-parameters.csv"
    cfg = dict(OsmoseConfigReader().read(master))
    cfg["simulation.time.nyear"] = "1"
    start = time.perf_counter()
    PythonEngine().run_in_memory(cfg, seed=1)
    print(f"Numba cache warm: one Baltic year in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
