"""run_simulation emits {sp}_recruitment_mean when recruitment_ages is passed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.calibrate_baltic as cb


def _fake_results(bio_df, yld_df, abd_df):
    class _R:
        def __init__(self, *a, **k): ...
        def biomass(self):
            return bio_df

        def yield_biomass(self):
            return yld_df

        def abundance_by_age(self):
            return abd_df

        def close(self): ...

    class _Engine:
        def run(self, *a, **k):
            class _Ret:
                returncode = 0

            return _Ret()

    return _R, _Engine


def _wire(monkeypatch, R, Engine):
    monkeypatch.setattr("osmose.engine.PythonEngine", Engine, raising=False)
    monkeypatch.setattr("osmose.results.OsmoseResults", R)


def test_recruitment_mean_from_age_bin(monkeypatch):
    bio = pd.DataFrame({"sprat": np.full(12, 1000.0), "herring": np.full(12, 2000.0)})
    yld = pd.DataFrame({"sprat": np.full(12, 100.0), "herring": np.full(12, 200.0)})
    # Long abundance-by-age. Vary EARLY (t<2) vs TRAILING (last 10) values so a wrong-window slice
    # (full-mean / head-slice / off-by-one) is actually caught; a decoy wrong-age bin proves age
    # selection. mean of the LAST 10 = 5e7/4e7; a full-12 mean would be inflated by the early decoy.
    rows = []
    for t in range(12):
        early = t < 2
        rows += [
            {"time": t, "species": "sprat", "bin": "1", "value": 9e9 if early else 5e7},
            {"time": t, "species": "sprat", "bin": "0", "value": 1e11},  # decoy wrong age
            {"time": t, "species": "herring", "bin": "0", "value": 9e9 if early else 4e7},
        ]
    abd = pd.DataFrame(rows)
    R, Engine = _fake_results(bio, yld, abd)
    _wire(monkeypatch, R, Engine)

    stats = cb.run_simulation(
        {"x": "1"}, {}, n_years=1, seed=0, recruitment_ages={"sprat": "1", "herring": "0"}
    )
    assert stats["sprat_recruitment_mean"] == pytest.approx(5e7)  # last-10 window, not full mean
    assert stats["herring_recruitment_mean"] == pytest.approx(4e7)


def test_missing_bin_for_one_species(monkeypatch):
    # Non-empty, correctly-columned frame but with NO herring rows: exercises the per-species
    # no-matching-(species,bin) path (distinct from the bare-empty-frame guard) — herring stat
    # stays unset while sprat is still emitted.
    bio = pd.DataFrame({"sprat": np.full(12, 1000.0), "herring": np.full(12, 2000.0)})
    yld = pd.DataFrame({"sprat": np.full(12, 100.0), "herring": np.full(12, 200.0)})
    abd = pd.DataFrame(
        [{"time": t, "species": "sprat", "bin": "1", "value": 5e7} for t in range(12)]
    )
    R, Engine = _fake_results(bio, yld, abd)
    _wire(monkeypatch, R, Engine)
    stats = cb.run_simulation(
        {"x": "1"}, {}, n_years=1, seed=0, recruitment_ages={"sprat": "1", "herring": "0"}
    )
    assert stats["sprat_recruitment_mean"] == pytest.approx(5e7)
    assert "herring_recruitment_mean" not in stats  # no matching rows -> unset, no crash


def test_no_recruitment_ages_emits_nothing(monkeypatch):
    bio = pd.DataFrame({"sprat": np.full(12, 1000.0)})
    R, Engine = _fake_results(bio, pd.DataFrame({"sprat": np.full(12, 100.0)}), pd.DataFrame())
    _wire(monkeypatch, R, Engine)
    stats = cb.run_simulation({"x": "1"}, {}, n_years=1, seed=0)  # recruitment_ages default None
    assert not any(k.endswith("_recruitment_mean") for k in stats)


def test_empty_abundance_frame_guarded(monkeypatch):
    bio = pd.DataFrame({"sprat": np.full(12, 1000.0)})
    # abundance-by-age off -> bare empty frame (no columns); must not KeyError.
    R, Engine = _fake_results(bio, pd.DataFrame({"sprat": np.full(12, 100.0)}), pd.DataFrame())
    _wire(monkeypatch, R, Engine)
    stats = cb.run_simulation({"x": "1"}, {}, n_years=1, seed=0, recruitment_ages={"sprat": "1"})
    assert "sprat_recruitment_mean" not in stats  # gracefully unset, no crash
