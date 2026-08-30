"""Regression: standard starvation must be suppressed on the Numba mortality path
when bioen is enabled.

Under bioen, starvation is the gonad-depletion formula
(`BioenStarvationMortality.computeStarvation`) applied inside the interleaved loop.
The standard (pred-success) rate must not be applied as well, or `n_dead[:, STARVATION]`
carries both. `_precompute_effective_rates` zeroes `eff_starv` for that reason.

Since Task 4, `mortality()` never dispatches to the batched Numba kernels under bioen
(spec decision 14), so this suppression is defence in depth rather than the load-bearing
guard it was — `eff_starv` is only read by the Numba paths. Keeping it means a future
bioen-aware kernel cannot silently reintroduce the double count.

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
    # Bioen owns starvation (gonad depletion, inside the interleaved loop); the Numba
    # kernel must NOT also apply the standard pred-success rate.
    assert np.all(eff_s == 0.0)


def test_non_bioen_keeps_standard_starvation():
    state = _MockState(2, [0.5, 0.3])
    eff_s, _, _, _ = _precompute_effective_rates(
        state, _config(bioen_enabled=False), n_subdt=1, step=0
    )
    denom = 24 * 1
    assert eff_s[0] == pytest.approx(0.5 / denom)
    assert eff_s[1] == pytest.approx(0.3 / denom)
