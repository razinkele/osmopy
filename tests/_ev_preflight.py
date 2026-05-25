"""Shared viability gate for the baltic_ev FIE / genetics integration tests.

The FIE-on-cod demo and the genetics-activation smokes only produce a
meaningful signal when the baltic_ev bioen fixture yields a *viable* cod
population: cod must reach the l50=35cm gear (otherwise size-selective fishing
catches nothing and the FIE differential on imax is structurally zero) AND hold
a 0.5-2.0x biomass envelope over 50y (otherwise founder-effect drift or
selection collapse swamps the FIE signal, and reproduction can fire with no
viable parents). Until the bioen params are tuned (plan Task 7.4) the population
is degenerate, so these tests must SKIP rather than fail.

The single pre-flight test
`test_baltic_ev_fixture_bioen.py::test_baltic_ev_baseline_viable_for_fie`
probes both criteria from one 50y baseline run and, on success, touches
`SENTINEL`. Dependent tests call `require_baltic_ev_preflight()`, which skips
when `SENTINEL` is absent.

`SENTINEL` is resolved relative to this file so the path is cwd-independent and
single-sourced across the writer and all readers (it is gitignored:
tests/.preflight_wired).
"""

from pathlib import Path

import pytest

SENTINEL = Path(__file__).parent / ".preflight_wired"

_SKIP_REASON = (
    "baltic_ev FIE pre-flight not satisfied this session — the bioen fixture "
    "has not been confirmed viable (cod must reach the 35cm gear and hold a "
    "0.5-2.0x 50y biomass envelope). Tune bioen params (plan Task 7.4); the "
    "pre-flight test touches tests/.preflight_wired on success."
)


def require_baltic_ev_preflight() -> None:
    """Skip the calling test unless the baltic_ev viability pre-flight passed.

    Conservative by construction: if the pre-flight test did not run (e.g. the
    file is run in isolation) or did not pass, the sentinel is absent and the
    dependent test skips rather than running on a degenerate population.
    """
    if not SENTINEL.exists():
        pytest.skip(_SKIP_REASON)
