"""run_simulation must surface {sp}_yield_mean from results.yield_biomass()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.calibrate_baltic as cb


def test_yield_mean_added(monkeypatch):
    # Fake OsmoseResults exposing wide biomass + yield frames (15 rows each).
    # First 5 rows differ from the last 10 per species so a wrong-window slice
    # (e.g. mean over all 15 rows instead of the last 10) is caught: the
    # n_eval_years=10 truncation in run_simulation (bio.iloc[-10:] / yvals[-10:])
    # must select the *last* 10 rows, not the full 15.
    bio = pd.DataFrame(
        {
            "cod": np.concatenate([np.full(5, 999.0), np.full(10, 100.0)]),
            "sprat": np.concatenate([np.full(5, 555.0), np.full(10, 1000.0)]),
        }
    )
    yld = pd.DataFrame(
        {
            "cod": np.concatenate([np.full(5, 111.0), np.full(10, 800.0)]),
            "sprat": np.concatenate([np.full(5, 222.0), np.full(10, 1200.0)]),
        }
    )

    class _FakeResults:
        def __init__(self, *a, **k): ...
        def biomass(self):
            return bio

        def yield_biomass(self):
            return yld

        def close(self): ...

    class _FakeEngine:
        def run(self, *a, **k):
            class _R:  # run() return with returncode
                returncode = 0

            return _R()

    monkeypatch.setattr("osmose.results.OsmoseResults", _FakeResults)
    # run_simulation imports these names lazily inside the function; patch the module it imports from.
    monkeypatch.setattr("osmose.engine.PythonEngine", _FakeEngine, raising=False)

    stats = cb.run_simulation({"x": "1"}, {}, n_years=1, seed=0, timeout_s=None)

    # Last-10-row means (rows 5..14), NOT the full-15-row mean.
    assert stats["cod_yield_mean"] == pytest.approx(800.0)
    assert stats["sprat_yield_mean"] == pytest.approx(1200.0)
    assert stats["cod_mean"] == pytest.approx(100.0)
    assert stats["sprat_mean"] == pytest.approx(1000.0)
