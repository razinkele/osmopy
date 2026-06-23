"""Regression: standard starvation must be suppressed on the Numba mortality path
when bioen is enabled.

Production runs `_HAS_NUMBA=True`, so mortality routes through the Numba kernels,
which apply standard (pred-success) starvation from `_precompute_effective_rates`'
`eff_starv`. `_bioen_step` SEPARATELY applies bioen gonad-depletion starvation.
With no suppression, both hit `n_dead[:, STARVATION]` and both reduce abundance —
a production double-count of starvation for every bioen run.

This complements the `_get_mortality_causes` fix (commit 457ac55), which handled
only the innermost pure-Python interleaved loop (the `_HAS_NUMBA=False` fallback).
Bioen starvation must be applied exactly once, by `_bioen_step`, on every path.

Source: deep review 2026-06-22 (critical) + spec review 2026-06-23 (dispatch-path).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from osmose.engine.processes.mortality import _precompute_effective_rates


class _MockState:
    def __init__(self, n: int, starvation_rate) -> None:
        self.species_id = np.zeros(n, dtype=np.int32)
        self.is_background = np.zeros(n, dtype=bool)
        self.age_dt = np.full(n, 10, dtype=np.int32)
        self.first_feeding_age_dt = np.ones(n, dtype=np.int32)
        self.starvation_rate = np.asarray(starvation_rate, dtype=np.float64)
        self.cell_y = np.zeros(n, dtype=np.int32)
        self.cell_x = np.zeros(n, dtype=np.int32)
        self.length = np.full(n, 15.0)
        self._n = n

    def __len__(self) -> int:
        return self._n


def _config(*, bioen_enabled: bool) -> SimpleNamespace:
    # Minimal config touched by _precompute_effective_rates when fishing is off
    # and fleet_state is None.
    return SimpleNamespace(
        n_species=1,
        n_dt_per_year=24,
        bioen_enabled=bioen_enabled,
        additional_mortality_rate=np.array([0.0]),
        additional_mortality_by_dt=None,
        additional_mortality_spatial=None,
        fishing_enabled=False,
    )


def test_bioen_suppresses_standard_starvation_on_numba_path():
    # Under-fed schools => non-zero standard starvation_rate.
    state = _MockState(2, [0.5, 0.3])
    eff_s, _, _, _ = _precompute_effective_rates(
        state, _config(bioen_enabled=True), n_subdt=1, step=0
    )
    # Bioen owns starvation (via _bioen_step); the Numba kernel must NOT also apply it.
    assert np.all(eff_s == 0.0)


def test_non_bioen_keeps_standard_starvation():
    state = _MockState(2, [0.5, 0.3])
    eff_s, _, _, _ = _precompute_effective_rates(
        state, _config(bioen_enabled=False), n_subdt=1, step=0
    )
    denom = 24 * 1
    assert eff_s[0] == pytest.approx(0.5 / denom)
    assert eff_s[1] == pytest.approx(0.3 / denom)
