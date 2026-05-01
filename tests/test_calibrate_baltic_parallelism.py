"""Verify the calibration script passes workers=-1 to scipy DE."""
import ast
import json
from pathlib import Path


def test_de_call_uses_workers():
    source = (Path(__file__).parent.parent / "scripts" / "calibrate_baltic.py").read_text()
    tree = ast.parse(source)
    de_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "differential_evolution"
    ]
    assert de_calls, "no differential_evolution() call found"
    for call in de_calls:
        kw_names = {kw.arg for kw in call.keywords}
        assert "workers" in kw_names, (
            f"differential_evolution call at line {call.lineno} "
            f"does not pass 'workers' kwarg"
        )


def test_phase12_returns_expected_params():
    from scripts.calibrate_baltic import get_phase12_params
    keys, bounds, x0 = get_phase12_params()
    # 16 mortality (8 larval + 8 adult) + 8 fishing + 3 B-H ssb_half (sp3/4/5)
    assert len(keys) == 27, f"phase 12 should expose 27 params, got {len(keys)}"
    assert len(bounds) == 27
    assert len(x0) == 27
    # Must contain mortality + fishing + recruitment keys
    assert any("mortality.additional.larva.rate.sp0" in k for k in keys)
    assert any("mortality.additional.rate.sp0" in k for k in keys)
    assert any("fisheries.rate.base.fsh0" in k for k in keys)
    for sp_idx in [3, 4, 5]:
        assert f"stock.recruitment.ssbhalf.sp{sp_idx}" in keys, (
            f"sp{sp_idx} ssb_half should be DE-tunable in phase 12"
        )
    # Cod (sp0) ssb_half is fixed at the literature prior, NOT a DE param
    assert "stock.recruitment.ssbhalf.sp0" not in keys
    # Flounder + pikeperch fishing bounds should be widened (from Task 1)
    fsh3_idx = keys.index("fisheries.rate.base.fsh3")
    fsh5_idx = keys.index("fisheries.rate.base.fsh5")
    assert bounds[fsh3_idx] == (-2.5, 0.5)
    assert bounds[fsh5_idx] == (-2.5, 0.5)
    # Adult mortality upper bounds widened for predated species (from T3alt)
    for sp_idx in [0, 1, 2, 3, 4, 5]:
        adult_idx = keys.index(f"mortality.additional.rate.sp{sp_idx}")
        assert bounds[adult_idx] == (-3.0, 0.7), f"sp{sp_idx} adult mortality should be widened"
    for sp_idx in [6, 7]:
        adult_idx = keys.index(f"mortality.additional.rate.sp{sp_idx}")
        assert bounds[adult_idx] == (-3.0, 0.3), f"sp{sp_idx} adult mortality should be default"
    # B-H ssb_half bounds: flounder wider than perch/pikeperch
    sp3_idx = keys.index("stock.recruitment.ssbhalf.sp3")
    sp4_idx = keys.index("stock.recruitment.ssbhalf.sp4")
    sp5_idx = keys.index("stock.recruitment.ssbhalf.sp5")
    assert bounds[sp3_idx] == (3.7, 5.3)
    assert bounds[sp4_idx] == (2.7, 4.7)
    assert bounds[sp5_idx] == (2.7, 4.7)


def test_optimizer_choices_and_dispatch():
    """All three optimizers must dispatch through to a normalized result dict."""
    import numpy as np
    from scripts.calibrate_baltic import _OPTIMIZER_CHOICES, _dispatch_optimizer

    assert _OPTIMIZER_CHOICES == ("de", "cmaes", "surrogate-de")

    # Tiny synthetic objective to keep the test cheap (~seconds).
    def quad(x):
        return float(np.sum((np.asarray(x) - 0.3) ** 2))

    bounds = [(-1.0, 1.0)] * 3
    x0 = [0.0, 0.0, 0.0]
    # init_pop only used by DE; minimal valid value
    init_pop = np.tile(np.asarray(x0), (10, 1))

    for opt in _OPTIMIZER_CHOICES:
        # Tiny budgets — just want to confirm dispatch + result shape, not convergence
        if opt == "surrogate-de":
            # surrogate-DE has fixed n_iterations=6, n_topk=30 hardcoded in dispatcher;
            # smoke-test would do ~165 evals which is acceptable for a synthetic quad
            pass
        result = _dispatch_optimizer(
            opt, quad, bounds, x0, init_pop,
            maxiter=5, popsize=5, tol=1e-3, workers=1, seed=42,
        )
        assert {"x", "fun", "nfev", "success", "message"} <= set(result.keys()), (
            f"optimizer {opt} returned incomplete result: {set(result.keys())}"
        )
        assert len(result["x"]) == 3
        assert isinstance(result["fun"], float)
        assert isinstance(result["nfev"], int)
        assert isinstance(result["message"], str)


def test_unknown_optimizer_raises():
    import pytest
    import numpy as np
    from scripts.calibrate_baltic import _dispatch_optimizer

    with pytest.raises(ValueError, match="unknown optimizer"):
        _dispatch_optimizer(
            "bogus", lambda x: 0.0, [(-1.0, 1.0)], [0.0],
            np.zeros((1, 1)),
            maxiter=1, popsize=1, tol=0.1, workers=1, seed=0,
        )


def test_cli_optimizer_choices_in_help():
    """--help must list all three optimizers so users know what's available."""
    import subprocess
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    result = subprocess.run(
        [str(venv_python), str(Path(__file__).resolve().parent.parent / "scripts" / "calibrate_baltic.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert "--optimizer" in result.stdout
    for opt in ("de", "cmaes", "surrogate-de"):
        assert opt in result.stdout, f"optimizer choice {opt!r} missing from --help"
    assert "--checkpoint-every" in result.stdout


def test_checkpoint_callback_writes_snapshot(tmp_path):
    """The DE checkpoint callback must serialise (gen, fun, x, params) to JSON."""
    import json
    from types import SimpleNamespace
    from scripts.calibrate_baltic import _make_checkpoint_callback

    checkpoint_path = tmp_path / "phase12_checkpoint.json"
    param_keys = ["mortality.additional.rate.sp0", "fisheries.rate.base.fsh0"]
    bounds = [(-3.0, 0.7), (-2.5, 0.0)]
    cb = _make_checkpoint_callback(
        checkpoint_path, every_n=2, param_keys=param_keys, bounds=bounds,
    )

    # Gen 1 — should NOT write (every_n=2)
    cb(SimpleNamespace(x=[0.0, -1.5], fun=4.5))
    assert not checkpoint_path.exists()

    # Gen 2 — writes
    cb(SimpleNamespace(x=[-1.0, -2.0], fun=3.2))
    assert checkpoint_path.exists()
    snap = json.loads(checkpoint_path.read_text())
    assert snap["generation"] == 2
    assert snap["best_fun"] == 3.2
    assert snap["best_x_log10"] == [-1.0, -2.0]
    # Linear params: 10^(-1.0) = 0.1, 10^(-2.0) = 0.01
    assert abs(snap["best_parameters"]["mortality.additional.rate.sp0"] - 0.1) < 1e-9
    assert abs(snap["best_parameters"]["fisheries.rate.base.fsh0"] - 0.01) < 1e-9
    assert "timestamp_iso" in snap

    # Gen 3 — does not write (only every 2nd gen)
    cb(SimpleNamespace(x=[0.5, -0.5], fun=999.0))
    snap = json.loads(checkpoint_path.read_text())
    assert snap["generation"] == 2  # unchanged
    assert snap["best_fun"] == 3.2

    # Gen 4 — writes again, overwriting
    cb(SimpleNamespace(x=[-2.0, -1.0], fun=2.1))
    snap = json.loads(checkpoint_path.read_text())
    assert snap["generation"] == 4
    assert snap["best_fun"] == 2.1


def test_checkpoint_callback_atomic_no_partial_file(tmp_path):
    """A kill mid-write must NOT leave a partial JSON — atomic via tmp + rename."""
    from types import SimpleNamespace
    from scripts.calibrate_baltic import _make_checkpoint_callback

    checkpoint_path = tmp_path / "snap.json"
    cb = _make_checkpoint_callback(
        checkpoint_path, every_n=1, param_keys=["k0"], bounds=[(-1.0, 1.0)],
    )
    cb(SimpleNamespace(x=[0.5], fun=1.5))
    # Verify only the final renamed file exists, no .tmp leftover
    assert checkpoint_path.exists()
    assert not (tmp_path / "snap.json.tmp").exists()


def test_checkpoint_callback_disabled_with_zero_every_n(tmp_path):
    """every_n=0 means no checkpoint writes — used by --checkpoint-every 0."""
    from types import SimpleNamespace
    from scripts.calibrate_baltic import _make_checkpoint_callback

    checkpoint_path = tmp_path / "snap.json"
    cb = _make_checkpoint_callback(
        checkpoint_path, every_n=0, param_keys=["k0"], bounds=[(-1.0, 1.0)],
    )
    for _ in range(5):
        cb(SimpleNamespace(x=[0.5], fun=1.5))
    assert not checkpoint_path.exists()


def test_checkpoint_callback_handles_legacy_signature(tmp_path):
    """If scipy passes the legacy (xk, convergence) signature, callback must
    not crash — just skip the snapshot."""
    import numpy as np
    from scripts.calibrate_baltic import _make_checkpoint_callback

    checkpoint_path = tmp_path / "snap.json"
    cb = _make_checkpoint_callback(
        checkpoint_path, every_n=1, param_keys=["k0"], bounds=[(-1.0, 1.0)],
    )
    # Legacy signature passes a numpy array as first positional, not OptimizeResult.
    # The callback should not crash — it will fail to access .x / .fun but
    # gracefully return None.
    cb(np.array([0.5]), convergence=0.1)
    assert not checkpoint_path.exists()


def test_apply_warm_start_overrides_only_known_keys(tmp_path):
    """warm-start: known keys overridden, unknown kept default, skipped excluded."""
    from scripts.calibrate_baltic import apply_warm_start

    fixture = tmp_path / "prior.json"
    fixture.write_text(json.dumps({
        "log10_parameters": {
            "mortality.additional.rate.sp0": 0.57,
            "fisheries.rate.base.fsh3": -0.81,
            "mortality.additional.larva.rate.sp1": 0.62,
        }
    }))

    param_keys = [
        "mortality.additional.rate.sp0",          # in JSON, but skipped
        "fisheries.rate.base.fsh3",               # in JSON, applied
        "mortality.additional.larva.rate.sp1",    # in JSON, applied
        "stock.recruitment.ssbhalf.sp3",          # NOT in JSON, kept default
    ]
    x0 = [-1.301, -1.398, 0.903, 4.699]
    skip = {"mortality.additional.rate.sp0"}

    new_x0, applied, skipped = apply_warm_start(fixture, param_keys, x0, skip)

    # sp0 stays at default x0 (it's in skip set)
    assert new_x0[0] == -1.301
    # fsh3 + larva.sp1 are overridden from JSON
    assert new_x0[1] == -0.81
    assert new_x0[2] == 0.62
    # ssb_half stays at default x0 (not in JSON)
    assert new_x0[3] == 4.699
    # tracking lists
    assert applied == ["fisheries.rate.base.fsh3", "mortality.additional.larva.rate.sp1"]
    assert skipped == ["mortality.additional.rate.sp0"]
