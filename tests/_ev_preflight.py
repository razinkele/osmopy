"""Shared viability gate for the baltic_ev FIE / genetics integration tests.

The FIE-on-cod demo and the genetics-activation smokes only produce a
meaningful signal when the baltic_ev bioen fixture yields a *viable* cod
population: cod must reach the l50=35cm gear (otherwise size-selective fishing
catches nothing and the FIE differential on imax is structurally zero) AND hold
a 0.5-2.0x biomass envelope over 50y (otherwise founder-effect drift or
selection collapse swamps the FIE signal, and reproduction can fire with no
viable parents). Until the bioen params are tuned (plan Task 7.4) the population
is degenerate, so these tests must SKIP rather than fail.

``_probe_baseline_viability`` runs the 50y baseline once and checks both
criteria. ``ensure_preflight_result`` is the shared entry point: it is called
both by the dedicated pre-flight test
(``test_baltic_ev_fixture_bioen.py::test_baltic_ev_baseline_viable_for_fie``)
and by every downstream dependant via ``require_baltic_ev_preflight()``.
Whichever caller reaches it FIRST in a given pytest run pays the ~60s compute
cost; every other caller -- on any pytest-xdist worker, in any file, running
before, during, or after that first caller -- converges on the same cached
answer instead of re-running the simulation or guessing from a half-written
file.

Why not a file the pre-flight test alone writes on success. An earlier design
had the pre-flight test ``unlink()`` the sentinel at the START of its ~60s run
and ``touch()`` it at the END, with dependants just checking existence. Under
``pytest -n auto --dist loadfile`` (this repo's actual CI invocation), the
pre-flight test's file and a dependant's file are scheduled to independent,
concurrently-running xdist workers with NO ordering guarantee between them.
That made whether a dependant saw the sentinel -- and therefore whether it ran
at all -- depend on which worker happened to be scheduled first, and BOTH
outcomes reported green: a test suite whose coverage silently differs between
otherwise-identical runs. Making the unlink/touch atomic would not have fixed
this: the problem was never a torn read of a partially-written file (unlink
and touch are each already atomic), it was that a dependant could observe
"sentinel absent" and have no way to tell "not computed yet" apart from
"computed and not viable". A dependant that instead ensures the answer is
computed -- via ``ensure_preflight_result`` below -- never has to make that
distinction: it either reads an already-cached answer or computes one itself.

Cross-process synchronization uses a POSIX ``flock`` on ``_LOCK`` (same
pattern as ``osmose/feedback.py::append_feedback``): the first caller to reach
the lock computes the probe and atomically replaces ``SENTINEL`` with a JSON
payload (write-temp-then-``os.replace`` -- the final file is always either
absent, or fully written; a reader entering mid-write is impossible); every
other caller blocks on the same lock, then re-checks the cache before
recomputing (standard double-checked locking -- avoids paying the 60s cost
twice just because two workers reached the lock close together).

The cached payload is stamped with ``PYTEST_XDIST_TESTRUNUID`` (identical
across every worker of ONE ``pytest`` invocation; unset on a serial run) so a
result from an EARLIER invocation is never trusted: a stale "viable" left on
disk from three weeks ago must not silently gate today's run forever, and a
stale "not viable" must not permanently block a fixture that has since been
tuned (plan Task 7.4). On a serial run (no run id to stamp with) there is no
cross-process race to guard against, so the result is memoized in-process only
and the file is not touched at all.

``SENTINEL`` is resolved relative to this file so the path is cwd-independent
and single-sourced across every caller (it is gitignored:
tests/.preflight_wired*, which also covers the lock file and the transient
per-pid temp files used for the atomic write).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

try:
    import fcntl
except ImportError:  # pragma: no cover — non-POSIX; CI and prod are Linux
    fcntl = None  # type: ignore[assignment]

SENTINEL = Path(__file__).parent / ".preflight_wired"
_LOCK = Path(__file__).parent / ".preflight_wired.lock"

_SKIP_REASON = (
    "baltic_ev FIE pre-flight not satisfied this run — the bioen fixture "
    "has not been confirmed viable (cod must reach the 35cm gear and hold a "
    "0.5-2.0x 50y biomass envelope). Tune bioen params (plan Task 7.4)."
)

# In-process memo for the serial (no xdist run id) case — see module docstring.
_MEMO: dict[str, tuple[bool, str]] = {}


def _run_id() -> str | None:
    """Value shared by every xdist worker of ONE pytest invocation; None serially."""
    return os.environ.get("PYTEST_XDIST_TESTRUNUID") or None


def _read_cached_result() -> tuple[bool, str] | None:
    """The cached (viable, detail), or None if absent/corrupt/from a different run."""
    if not SENTINEL.exists():
        return None
    try:
        data = json.loads(SENTINEL.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("run_id") != _run_id():
        return None  # stale — from a different pytest invocation, do not trust it
    try:
        return bool(data["viable"]), str(data["detail"])
    except KeyError:
        return None


def _write_cached_result(result: tuple[bool, str]) -> None:
    """Atomically publish ``result`` so no reader ever observes a partial write."""
    viable, detail = result
    payload = {"viable": viable, "detail": detail, "run_id": _run_id()}
    tmp = SENTINEL.with_name(f".preflight_wired.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, SENTINEL)  # atomic on POSIX (same filesystem)


def _probe_baseline_viability() -> tuple[bool, str]:
    """Run the 50y baltic_ev baseline once and return (viable, detail).

    Deliberately does NOT catch engine/config exceptions — those must
    propagate and fail the caller, not be folded into "not viable". Conflating
    an environment error with a genuine, documented population collapse is
    exactly the trap this branch's own review flagged elsewhere (an
    xfail(strict=True) with no ``raises=`` reporting an ImportError as if it
    were the modelled outcome): the two must stay distinguishable here too.
    """
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(Path("data/baltic_ev/baltic_ev_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "50"
    # Reader canonicalizes to the NEW 4.4.0 key; set that one (setting the old
    # key would be a no-op the from_dict merge silently drops).
    cfg["module.genetics.enabled"] = "false"
    # Zero fishing so the cod size distribution reflects bioen alone.
    cfg["fisheries.rate.base.fsh0"] = "0.0"
    result = PythonEngine().run_in_memory(cfg, seed=0)

    # Criterion 1: cod reach the 35cm gear at the final year.
    # biomass_by_size returns long-form [time, species, bin, value] where `bin`
    # is the size-bin lower edge as a string (e.g. "35.0"); see
    # osmose/engine/output.py:_build_distribution_dataframes.
    bbs = result.biomass_by_size("cod")
    if bbs.empty:
        return False, (
            "baltic_ev pre-flight: biomass_by_size('cod') is empty — fixture un-tuned (Task 7.4)."
        )
    bbs = bbs.assign(bin_lower=bbs["bin"].astype(float))
    t_max = bbs["time"].max()
    last_year = bbs[bbs["time"] >= t_max - 1.0]
    biomass_ge35 = float(last_year[last_year["bin_lower"] >= 35.0]["value"].sum())
    biomass_total = float(last_year["value"].sum())
    max_occupied_bin = float(last_year[last_year["value"] > 0]["bin_lower"].max())

    # Criterion 2: 50y/5y (post-burnin) biomass envelope.
    bio = result.biomass().sort_values("Time")
    # See test_baltic_ev_runs_5_years_without_genetics for the wide-form note.
    if "cod" not in bio.columns:
        return False, (
            f"baltic_ev pre-flight: biomass output missing 'cod' column; "
            f"got columns={list(bio.columns)}."
        )
    burnin = float(bio[(bio["Time"] >= 5.0) & (bio["Time"] < 6.0)]["cod"].mean())
    end = float(bio[bio["Time"] >= 49.0]["cod"].mean())
    ratio = end / burnin if burnin > 0 else float("inf")

    if not (biomass_ge35 > 0.0 and 0.5 <= ratio <= 2.0):
        return False, (
            "baltic_ev FIE pre-flight not viable — tune bioen params (Task 7.4). "
            f"cod biomass >=35cm at year {t_max:.1f} = {biomass_ge35:.3e} "
            f"(total = {biomass_total:.3e}, largest occupied bin = "
            f"{max_occupied_bin:.1f}cm); 50y/5y envelope ratio = {ratio:.2f} "
            "(need cod >=35cm present and 0.5 <= ratio <= 2.0)."
        )
    return True, (
        f"baltic_ev pre-flight viable: cod biomass >=35cm at year {t_max:.1f} = "
        f"{biomass_ge35:.3e} (total = {biomass_total:.3e}); 50y/5y envelope ratio = "
        f"{ratio:.2f}."
    )


def ensure_preflight_result() -> tuple[bool, str]:
    """Return (viable, detail), computing + caching at most once per pytest run.

    See the module docstring for why this replaces a plain "check whether a
    dedicated test already wrote a sentinel" design. Deterministic under
    ``pytest -n auto``: no matter which test — the dedicated pre-flight test or
    a downstream dependant — calls this first, the result is the same, and no
    caller ever observes a half-computed state.
    """
    run_id = _run_id()
    if run_id is None:
        # Serial run: only this one process exists, so there is no cross-process
        # race to guard against — a plain in-process memo is sufficient and
        # avoids touching the filesystem at all.
        if "result" not in _MEMO:
            _MEMO["result"] = _probe_baseline_viability()
        return _MEMO["result"]

    cached = _read_cached_result()
    if cached is not None:
        return cached

    with open(_LOCK, "a") as lockfile:
        have_lock = False
        if fcntl is not None:
            try:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
                have_lock = True
            except OSError:
                have_lock = False
        try:
            # Re-check: another worker may have computed it while we waited.
            cached = _read_cached_result()
            if cached is not None:
                return cached
            result = _probe_baseline_viability()
            _write_cached_result(result)
            return result
        finally:
            if have_lock:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)


def require_baltic_ev_preflight() -> None:
    """Skip the calling test unless the baltic_ev viability pre-flight passed.

    Computes the pre-flight (or reuses an already-cached result from this
    pytest run — see ``ensure_preflight_result``) rather than checking for a
    sentinel some OTHER, separately-scheduled test may or may not have written
    yet. Conservative by construction: on any non-viable outcome the calling
    test skips rather than running on a degenerate population.
    """
    viable, detail = ensure_preflight_result()
    if not viable:
        pytest.skip(f"{_SKIP_REASON} {detail}")
