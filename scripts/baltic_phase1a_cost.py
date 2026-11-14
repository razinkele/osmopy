#!/usr/bin/env python3
"""In-situ cost of the predation kernel's Phase 1a size-window scan on the Baltic (perf plan item 6).

Method: in a SCRATCH worktree, wrap Phase 1a of ``_apply_predation_numba`` (the school-prey scan,
with its ``total_available``/``n_prey`` reset) in ``for _rep in range(N):`` (``--passes N``,
default 2). The repeats recompute the same buffers from unchanged inputs, so the outputs must stay
identical; the ``time`` command prints a biomass hash to check this. The extra ``mortality()`` time
is then the scan's in-situ cost.

``--test-only`` instead adds passes that run only the per-pair tests (self, abundance, length,
size window), without the eligible-prey work. That separates the testing a length prefilter could
remove from the work it must keep.

A repeat can err in either direction (a warm cache understates, a restructured loop nest can
overstate), so check linearity with ``--passes 3``.

Usage (from the repo root):
    git worktree add --detach ../osmopy-scratch origin/master
    .venv/bin/python scripts/baltic_phase1a_cost.py apply-double ../osmopy-scratch
    # interleave, at least two pairs:
    .venv/bin/python scripts/baltic_phase1a_cost.py time .
    .venv/bin/python scripts/baltic_phase1a_cost.py time ../osmopy-scratch
    git worktree remove --force ../osmopy-scratch

Never apply the edit to a checkout you work in: ``apply-double`` refuses this repository's own
root, any worktree that is on a branch (the scratch must be detached), and any worktree with
uncommitted changes.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = Path("osmose/engine/processes/mortality.py")
MARKER = "# PERF PROBE: Phase 1a"
BLOCK_START = "        total_available = 0.0\n        n_prey = 0\n\n        # 1a: School prey\n"
BLOCK_END = "        # 1b: Resource prey\n"


def apply_double(worktree: Path, passes: int = 2, test_only: bool = False) -> None:
    worktree = worktree.resolve()
    if worktree == ROOT:
        raise SystemExit(f"refusing to edit this repository's own checkout: {worktree}")
    on_branch = (
        subprocess.run(
            ["git", "-C", str(worktree), "symbolic-ref", "-q", "HEAD"], capture_output=True
        ).returncode
        == 0
    )
    if on_branch:
        raise SystemExit(
            f"refusing: {worktree} is on a branch; use a detached scratch worktree "
            "(git worktree add --detach <path> origin/master)"
        )
    dirty = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty.strip():
        raise SystemExit(f"refusing: {worktree} has uncommitted changes")
    path = worktree / KERNEL
    src = path.read_text()
    start = src.index(BLOCK_START)
    end = src.index(BLOCK_END, start)
    block = src[start:end]
    if block.count("for q_pos in range(n_local):") != 1:
        raise SystemExit("Phase 1a block not recognised; the kernel has changed, update this probe")
    if passes < 2:
        raise SystemExit("--passes must be at least 2")
    if test_only:
        # Extra passes run only the per-pair tests (self, abundance, length, size window) and do
        # none of the eligible-prey work (accessibility, prey biomass, buffer writes). A sum of
        # the ratios feeds a branch that is never taken (every ratio is positive), so the
        # compiler cannot drop the loop and the outputs cannot change.
        label = f"{MARKER} test-only x{passes - 1} extra"
        extra = (
            f"        _probe_sum = 0.0  {label}\n"
            f"        for _rep in range({passes - 1}):\n"
            "            for _q_pos in range(n_local):\n"
            "                _q = cell_indices[_q_pos]\n"
            "                if _q == p_idx:\n"
            "                    continue\n"
            "                _abd = max(inst_abd[_q] - egg_retained[_q], 0.0)\n"
            "                if _abd <= 0:\n"
            "                    continue\n"
            "                _pl = length[_q]\n"
            "                if _pl <= 0:\n"
            "                    continue\n"
            "                _ratio = pred_len / _pl\n"
            "                if _ratio < r_min or _ratio >= r_max:\n"
            "                    continue\n"
            "                _probe_sum += _ratio\n"
            "        if _probe_sum < -1.0:\n"
            "            total_available += _probe_sum\n\n"
        )
        path.write_text(src[:end] + extra + src[end:])
    else:
        label = f"{MARKER} full x{passes}"
        indented = "".join(("    " + ln if ln.strip() else ln) for ln in block.splitlines(True))
        path.write_text(
            src[:start] + f"        for _rep in range({passes}):  {label}\n" + indented + src[end:]
        )
    print(f"Phase 1a probe applied in {path}: {label}")


def time_arm(repo: Path, years: int, seed: int) -> None:
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    import numba

    import osmose.engine.simulate as sim
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    # A path test, not a string-prefix one: "/x/osmopy-wt-a" starts with "/x/osmopy".
    if not Path(sim.__file__).resolve().is_relative_to(repo):
        raise SystemExit(f"imported {sim.__file__}, not {repo}")
    numba.set_num_threads(1)
    cfg = dict(OsmoseConfigReader().read(repo / "data/baltic/baltic_all-parameters.csv"))
    cfg["simulation.time.nyear"] = str(years)
    PythonEngine().run_in_memory(cfg, seed=seed)  # warm-up (JIT)

    spent = [0.0]
    original = sim._mortality

    def timed(*args, **kwargs):
        t = time.perf_counter()
        out = original(*args, **kwargs)
        spent[0] += time.perf_counter() - t
        return out

    sim._mortality = timed
    t = time.perf_counter()
    results = PythonEngine().run_in_memory(cfg, seed=seed)
    wall = time.perf_counter() - t
    digest = hashlib.sha256(results.biomass().to_csv().encode()).hexdigest()[:16]
    probe = [ln for ln in (repo / KERNEL).read_text().splitlines() if MARKER in ln]
    arm = probe[0].split(MARKER, 1)[1].strip() if probe else "control"
    print(f"{arm:22s} mortality {spent[0]:.2f} s  wall {wall:.2f} s  biomass-sha {digest}")


def main() -> int:
    ap = argparse.ArgumentParser(description="In-situ cost of the Phase 1a scan (Baltic).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply-double", help="repeat Phase 1a in a scratch worktree")
    a.add_argument("worktree", type=Path)
    a.add_argument("--passes", type=int, default=2, help="total passes of the scan (default 2)")
    a.add_argument(
        "--test-only",
        action="store_true",
        help="extra passes run only the per-pair tests, none of the eligible-prey work",
    )
    t = sub.add_parser("time", help="time mortality() for one arm, one thread, warm")
    t.add_argument("repo", type=Path)
    t.add_argument("--years", type=int, default=10)
    t.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.cmd == "apply-double":
        apply_double(args.worktree, passes=args.passes, test_only=args.test_only)
    else:
        time_arm(args.repo, args.years, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
