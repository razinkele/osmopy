"""Provenance & call-path guards. Run FIRST, fail loudly.

Guards the two documented benchmark traps (perf-arc-overview.md:103-104):
importing the wrong osmose, and timing the non-numba dead-code path.
"""
from __future__ import annotations

from pathlib import Path


def assert_provenance(worktree_root: Path) -> dict:
    import numba

    from osmose.engine.processes import mortality

    mfile = Path(mortality.__file__).resolve()
    root = Path(worktree_root).resolve()
    # mortality.py must live at <root>/osmose/engine/processes/mortality.py, so `root`
    # must be one of its ACTUAL parent directories — rejecting a path that merely shares a
    # string prefix (e.g. /home/razinka). The guard's real job: if the WRONG osmose is
    # imported, mfile resolves outside this tree and `root` is absent from its parents.
    if str(root) not in {str(p) for p in mfile.parents}:
        raise RuntimeError(
            f"mortality.py resolved to {mfile}, not under worktree {root}. "
            "Set PYTHONPATH to the worktree before running the spike."
        )
    if not getattr(mortality, "_HAS_NUMBA", False):
        raise RuntimeError(
            "mortality._HAS_NUMBA is False — the per-cell Python path is dead "
            "code in production; timing it measures the wrong kernel."
        )
    return {
        "mortality_file": str(mfile),
        "has_numba": True,
        "numba_version": numba.__version__,
    }


def capture_flag_config(diet_enabled: bool, tl_tracking: bool,
                        use_stage_access: bool, has_access: bool) -> dict[str, bool]:
    return {
        "diet_enabled": bool(diet_enabled),
        "tl_tracking": bool(tl_tracking),
        "use_stage_access": bool(use_stage_access),
        "has_access": bool(has_access),
    }
