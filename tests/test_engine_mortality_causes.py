"""Tests for `_get_mortality_causes` — the interleaved-loop mortality cause set.

Java builds the set as every `MortalityCause` minus DISCARDS and AGING, minus FORAGING
when bioen is off (`MortalityProcess.java:506-517`); OUT is handled by the post-loop
out-school pass. So STARVATION is in the set on BOTH paths, and FORAGING only under bioen.

Bioen starvation reads the PREVIOUS step's `e_net`, because Java's step order is
mortality -> EnergyBudget -> reproduction (`SimulationStep.java:190-198`) and
`BioenStarvationMortality.computeStarvation` is called from inside the interleaved loop.
`_bioen_step` therefore must NOT apply starvation as well — the earlier port did, and
excluded STARVATION from this set to avoid the double count. Task 4 moved starvation back
into the loop where Java has it and removed `_bioen_step`'s copy; this test guards the
cause set against a regression to either half of that arrangement.
"""

from __future__ import annotations

from types import SimpleNamespace

from osmose.engine.processes.mortality import (
    _ADDITIONAL,
    _FISHING,
    _FORAGING,
    _PREDATION,
    _STARVATION,
    _get_mortality_causes,
)


def test_non_bioen_includes_starvation_excludes_foraging():
    causes = _get_mortality_causes(SimpleNamespace(bioen_enabled=False))
    assert set(causes) == {_PREDATION, _STARVATION, _ADDITIONAL, _FISHING}
    assert _FORAGING not in causes


def test_bioen_includes_starvation_and_foraging():
    # Bioen starvation competes with the other causes inside the loop, on the previous
    # step's e_net. Dropping it here would silence starvation entirely, since
    # `_bioen_step` no longer applies it.
    causes = _get_mortality_causes(SimpleNamespace(bioen_enabled=True))
    assert _STARVATION in causes
    assert set(causes) == {_PREDATION, _STARVATION, _ADDITIONAL, _FISHING, _FORAGING}
