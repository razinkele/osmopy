from __future__ import annotations

import multiprocessing as mp

import pytest


def _build_wrapper(targets, species_names):
    """Construct an _ObjectiveWrapper using the verified real constructor."""
    from scripts.calibrate_baltic import _ObjectiveWrapper

    base_config = {"simulation.nspecies": str(len(species_names))}
    return _ObjectiveWrapper(
        base_config=base_config,
        targets=targets,
        param_keys=["k_a", "k_b"],
        n_years=1,
        seed=42,
        use_log_space=True,
    )


def test_last_per_species_residuals_none_before_first_call(synthetic_two_species_targets):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    assert w.last_per_species_residuals is None


def test_last_per_species_residuals_populated_after_call(
    synthetic_two_species_targets, synthetic_stats_in_band, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(w, "_simulate_and_compute_stats", lambda x: synthetic_stats_in_band)
    w([-0.3, 0.3])
    assert w.last_per_species_residuals is not None
    assert len(w.last_per_species_residuals) == 2
    for entry in w.last_per_species_residuals:
        assert len(entry) == 3
        sp, residual, sim_biomass = entry
        assert sp in species_names
        assert isinstance(residual, float)
        assert isinstance(sim_biomass, float)


def test_residuals_zero_when_in_band(
    synthetic_two_species_targets, synthetic_stats_in_band, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(w, "_simulate_and_compute_stats", lambda x: synthetic_stats_in_band)
    w([0.0, 0.0])
    by_sp = {sp: r for sp, r, _ in w.last_per_species_residuals}
    assert by_sp["sp_a"] == 0.0
    assert by_sp["sp_b"] == 0.0


def test_residual_positive_when_out_of_band(
    synthetic_two_species_targets, synthetic_stats_sp_b_out_of_band, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(
        w, "_simulate_and_compute_stats", lambda x: synthetic_stats_sp_b_out_of_band,
    )
    w([0.0, 0.0])
    by_sp = {sp: r for sp, r, _ in w.last_per_species_residuals}
    assert by_sp["sp_a"] == 0.0
    assert by_sp["sp_b"] > 0.0


def test_sim_biomass_captured_per_species(
    synthetic_two_species_targets, synthetic_stats_in_band, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(w, "_simulate_and_compute_stats", lambda x: synthetic_stats_in_band)
    w([0.0, 0.0])
    by_sp = {sp: b for sp, _, b in w.last_per_species_residuals}
    assert by_sp["sp_a"] == synthetic_stats_in_band["sp_a_mean"]
    assert by_sp["sp_b"] == synthetic_stats_in_band["sp_b_mean"]


def test_extinction_fast_path_records_100_loss_and_zero_biomass(
    synthetic_two_species_targets, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(
        w, "_simulate_and_compute_stats",
        lambda x: {
            "sp_a_mean": 0.0, "sp_a_cv": 0.0, "sp_a_trend": 0.0,
            "sp_b_mean": 2.0, "sp_b_cv": 0.0, "sp_b_trend": 0.0,
        },
    )
    w([0.0, 0.0])
    by_sp = {sp: (r, b) for sp, r, b in w.last_per_species_residuals}
    assert by_sp["sp_a"] == (100.0, 0.0)
    assert by_sp["sp_b"][0] == 0.0


def test_residuals_attribute_unset_when_call_raises_midway(
    synthetic_two_species_targets, monkeypatch,
):
    """Spec §6.5.1 load-bearing assign-at-end invariant."""
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)

    def boom(x):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(w, "_simulate_and_compute_stats", boom)
    with pytest.raises(RuntimeError):
        w([0.0, 0.0])
    assert w.last_per_species_residuals is None

    monkeypatch.setattr(
        w, "_simulate_and_compute_stats",
        lambda x: {"sp_a_mean": 1.0, "sp_a_cv": 0.0, "sp_a_trend": 0.0,
                   "sp_b_mean": 2.0, "sp_b_cv": 0.0, "sp_b_trend": 0.0},
    )
    w([0.0, 0.0])
    populated = w.last_per_species_residuals
    assert populated is not None
    monkeypatch.setattr(w, "_simulate_and_compute_stats", boom)
    with pytest.raises(RuntimeError):
        w([0.0, 0.0])
    assert w.last_per_species_residuals is populated


# Module-level helpers for multiprocessing round-trip test
_RTL_RESULT_KEYS_STATS = {
    "sp_a_mean": 1.0, "sp_a_cv": 0.0, "sp_a_trend": 0.0,
    "sp_b_mean": 2.0, "sp_b_cv": 0.0, "sp_b_trend": 0.0,
}


def _call_wrapper_then_return_residuals(wrapper):
    wrapper._simulate_and_compute_stats = lambda x: _RTL_RESULT_KEYS_STATS
    wrapper([0.0, 0.0])
    return wrapper.last_per_species_residuals


def test_evaluator_round_trips_through_multiprocessing(synthetic_two_species_targets):
    """DE workers > 1 ships the evaluator. Verify the new attribute survives."""
    targets, _ = synthetic_two_species_targets
    w = _build_wrapper(targets, ["sp_a", "sp_b"])
    with mp.Pool(1) as pool:
        worker_residuals = pool.apply(_call_wrapper_then_return_residuals, (w,))
    assert worker_residuals is not None
    assert len(worker_residuals) == 2
    # Parent process: attribute unchanged
    assert w.last_per_species_residuals is None
