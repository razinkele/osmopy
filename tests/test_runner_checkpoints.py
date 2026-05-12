from __future__ import annotations


def _toy_objective_2_param(x):
    """Picklable quadratic; min at (1.0, 2.0) → 0.0."""
    return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2


def test_de_writes_checkpoint_each_generation(tmp_results_dir):
    """Run DE for 3 generations on a synthetic toy; assert a CalibrationCheckpoint-
    shaped JSON file appears in tmp_results_dir after each generation."""
    from scipy.optimize import differential_evolution

    from osmose.calibration.checkpoint import read_checkpoint
    from scripts.calibrate_baltic import _make_checkpoint_callback

    bounds = [(-1.0, 1.0), (0.0, 3.0)]
    param_keys = ["k_a", "k_b"]
    checkpoint_path = tmp_results_dir / "phase_test_checkpoint.json"
    callback = _make_checkpoint_callback(
        checkpoint_path, 1, param_keys, bounds,
        phase="test",
        optimizer="de",
        evaluator=None,
        banded_targets=None,
        generation_budget=3,
    )
    differential_evolution(
        _toy_objective_2_param, bounds, maxiter=3, popsize=4, seed=42,
        workers=1, callback=callback, polish=False,
    )
    assert checkpoint_path.exists()
    result = read_checkpoint(checkpoint_path)
    assert result.kind == "ok"
    ckpt = result.checkpoint
    assert ckpt.optimizer == "de"
    assert ckpt.phase == "test"
    assert ckpt.generation == 3
    assert ckpt.generation_budget == 3
    assert ckpt.per_species_residuals is None
    assert ckpt.per_species_sim_biomass is None
    assert ckpt.species_labels is None
    assert ckpt.proxy_source == "objective_disabled"


def test_de_existing_test_callsite_still_works(tmp_path):
    """Regression pin: 14 existing _make_checkpoint_callback callers in
    tests/test_calibrate_baltic_parallelism.py pass only (path, every_n, param_keys, bounds)."""
    from types import SimpleNamespace
    from scripts.calibrate_baltic import _make_checkpoint_callback

    checkpoint_path = tmp_path / "phase_legacy_checkpoint.json"
    cb = _make_checkpoint_callback(
        checkpoint_path, every_n=2,
        param_keys=["k_a", "k_b"], bounds=[(-1.0, 1.0), (0.0, 3.0)],
    )
    cb(SimpleNamespace(x=[0.0, 1.0], fun=4.5))
    cb(SimpleNamespace(x=[-0.5, 0.5], fun=3.2))
    assert checkpoint_path.exists()
