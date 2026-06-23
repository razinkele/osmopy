"""Tests for `_get_mortality_causes` — the interleaved-loop mortality cause set.

Regression guard for the bioen double-starvation bug (deep review 2026-06-22,
critical): when bioenergetics is enabled, starvation mortality is applied
authoritatively in `_bioen_step` using the freshly-computed current-step energy
budget. Including STARVATION in the interleaved mortality loop as well applied it
a SECOND time (with the previous step's stale `e_net`), double-counting starvation
deaths in `n_dead[:, STARVATION]`. So the interleaved cause set must EXCLUDE
STARVATION when bioen is enabled (but keep it when bioen is off, where the loop is
the only place standard starvation runs).
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


def test_bioen_excludes_starvation_from_interleaved_loop():
    # Bioen starvation is owned by _bioen_step (current-step e_net); the
    # interleaved loop must NOT also apply it, or deaths are double-counted.
    causes = _get_mortality_causes(SimpleNamespace(bioen_enabled=True))
    assert _STARVATION not in causes
    # The other interleaved causes — including bioen-only FORAGING — still run.
    assert set(causes) == {_PREDATION, _ADDITIONAL, _FISHING, _FORAGING}
