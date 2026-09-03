"""Regression: standard starvation must be suppressed on the Numba mortality path
when bioen is enabled.

Under bioen, starvation is the gonad-depletion formula
(`BioenStarvationMortality.computeStarvation`) applied inside the interleaved loop.
The standard (pred-success) rate must not be applied as well, or `n_dead[:, STARVATION]`
carries both. `_precompute_effective_rates` zeroes `eff_starv` for that reason.

Bioen-Numba-kernel plan Task 3 (2026-09-03) reversed spec decision 14: `mortality()` now
DOES dispatch bioen runs to the batched Numba kernels (Task 2 taught them all five bioen
behaviours first), so `eff_starv` is read by the very kernel every bioen run now
executes. It is still defence in depth, not the only guard: `_apply_single_cause`'s
bioen branch (cause==1) has every one of its exit paths end in `return` before reaching
its `D = eff_starv[idx]` tail, so that tail is unreachable under bioen regardless of what
`eff_starv` holds. The REAL guard is that unconditional `return`; zeroing `eff_starv` is
a cheap belt that keeps the tail dead code even if those `return`s are ever
restructured. Final whole-branch review (2026-09-03, finding F2) corrected an earlier
"the ONLY guard" version of this docstring. The behavioural companion to this unit test
is
`test_bioen_starvation_fires_inside_the_interleaved_loop`
(`tests/test_engine_bioen_mortality_parity.py`), which pins the exact post-dispatch
`n_dead`/`e_net`/`gonad_weight` numbers end to end through `mortality()` — a double count
there would move those numbers, not just this suppression check.

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
