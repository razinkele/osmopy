"""Phase-14 (predator functional-response K) calibration scaffolding tests.

Asserts the phase-14 problem assembles correctly:
  * ``get_phase14_params()`` returns the 4 FR halfsat keys, log10-encoded.
  * The phase-14 base_config freezes the ~40 reconstructed phase-13 params,
    sets all 9 focal species to Shepherd SR, and fixes FR shape=type3 on the 4
    calibrated predators (cod_west sp0, pikeperch sp5, GreySeal sp15, Cormorant sp16).
  * The 4 free halfsat keys are DISJOINT from base_config (base sets shape, not halfsat).
  * A single objective evaluation runs without a config-validation error and
    returns a finite float (10**x lands as a raw float K in the engine config).

These are assembly/single-eval checks only — they do NOT run a multi-hour
calibration (that is PB3).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._data_guards import require_baltic_phase12

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_HALFSAT_KEYS = [
    "predation.functional.response.halfsat.sp0",
    "predation.functional.response.halfsat.sp5",
    "predation.functional.response.halfsat.sp15",
    "predation.functional.response.halfsat.sp16",
]


def _load_calibrate_module():
    """Load scripts/calibrate_baltic.py from disk (scripts/ has no __init__.py)."""
    script = PROJECT_ROOT / "scripts" / "calibrate_baltic.py"
    spec = importlib.util.spec_from_file_location("calibrate_baltic", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_baltic"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cal():
    module = _load_calibrate_module()
    try:
        yield module
    finally:
        sys.modules.pop("calibrate_baltic", None)


def _ensure_phase13_results() -> dict:
    """Regenerate phase13_results.json (gitignored artifact) and return it.

    Reconstruction reads the gitignored phase-12 results; skip when absent so a
    clean CI checkout reports a skip rather than a hard FileNotFoundError.
    """
    require_baltic_phase12()
    recon = PROJECT_ROOT / "scripts" / "reconstruct_phase13_results.py"
    spec = importlib.util.spec_from_file_location("reconstruct_phase13_results", recon)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    with open(mod.PHASE13_FILE) as f:
        return json.load(f)


def test_phase14_params_shape(cal):
    keys, bounds, x0 = cal.get_phase14_params()
    assert keys == EXPECTED_HALFSAT_KEYS
    assert len(bounds) == 4
    assert len(x0) == 4
    # log10 space; K in [0.5, 5.0]; x0 = log10(1.0) = 0.0
    for lo, hi in bounds:
        assert math.isclose(lo, math.log10(0.5))
        assert math.isclose(hi, math.log10(5.0))
    for v in x0:
        assert math.isclose(v, 0.0, abs_tol=1e-12)
    # 10**x0 -> K = 1.0, inside the engine's accepted FR-halfsat range [0.1, 5.0]
    for v in x0:
        assert 0.1 <= 10.0**v <= 5.0


def test_phase14_bound_corners_pass_the_engine_halfsat_check(cal):
    """Both corners of the search box must reach the engine as a K it accepts.

    The engine rejects a half-saturation outside [0.1, 5.0] (``_parse_reproduction_params``,
    ``osmose/engine/config.py``). ``10 ** log10(5.0)`` is 5.000000000000001, so a candidate on the
    upper bound used to fail its simulation and score the 1e6 sentinel (found 2026-11-11, #232)."""
    keys, bounds, _ = cal.get_phase14_params()
    for corner in ([b[0] for b in bounds], [b[1] for b in bounds]):
        overrides = cal.expand_param_overrides(keys, np.array(corner), True)
        for key in keys:
            assert 0.1 <= float(overrides[key]) <= 5.0, f"{key} = {overrides[key]}"


def test_reconstructed_phase13_has_40_params(cal):
    data = _ensure_phase13_results()
    params = data["parameters"]
    assert data["phase"] == "13"
    assert data.get("_reconstructed") is True
    assert len(params) == 40, f"expected 40 reconstructed phase-13 params, got {len(params)}"
    # 24 mortality + fishing (from phase 12)
    for i in range(8):
        assert f"mortality.additional.larva.rate.sp{i}" in params
        assert f"mortality.additional.rate.sp{i}" in params
        assert f"fisheries.rate.base.fsh{i}" in params
    # 16 SR params from the doc
    for i in range(8):
        assert f"stock.recruitment.ssbhalf.sp{i}" in params
        assert f"stock.recruitment.shape.sp{i}" in params
    # Spot-check documented values
    assert params["stock.recruitment.ssbhalf.sp0"] == 120000.0  # cod fixed Bpa
    assert params["stock.recruitment.shape.sp0"] == 1.88  # cod beta
    assert params["stock.recruitment.shape.sp6"] == 2.56  # smelt beta
    # No FR halfsat keys (must stay disjoint from phase-14 free params)
    assert not any("functional.response.halfsat" in k for k in params)


def _build_phase14_base_config(cal) -> tuple[dict, list[str]]:
    """The phase-14 base exactly as run_calibration builds it, via the shared
    ``apply_phase14_base``. This used to be a hand copy, and it drifted to the pre-split
    indices (0, 5, 14, 15): type III on Benthos, none on Cormorant. That made the assertions
    below check the copy instead of the calibrator."""
    from osmose.config.reader import OsmoseConfigReader

    _ensure_phase13_results()
    param_keys, _, _ = cal.get_phase14_params()
    base_config = OsmoseConfigReader().read(cal.BALTIC_CONFIG)
    cal.apply_phase14_base(base_config)
    return base_config, param_keys


def test_phase14_base_config_freezes_and_sets_fr(cal):
    base_config, param_keys = _build_phase14_base_config(cal)

    # All 9 focal species on Shepherd (cod_east sp8 included after the disaggregation)
    for sp_idx in range(9):
        assert base_config[f"stock.recruitment.type.sp{sp_idx}"] == "shepherd"
    # FR type3 on the 4 calibrated predators: cod_west, pikeperch, GreySeal, Cormorant
    for sp_idx in (0, 5, 15, 16):
        assert base_config[f"predation.functional.response.shape.sp{sp_idx}"] == "type3"
    # ...and not on Benthos (sp14), where the pre-split indices used to put it
    assert "predation.functional.response.shape.sp14" not in base_config
    # 39+ frozen phase-13 params present (40 minus cod ssbhalf.sp0 which would also be
    # set explicitly in phase 13; here it comes through the reconstructed JSON)
    frozen = [
        k
        for k in base_config
        if k.startswith(
            (
                "mortality.additional",
                "fisheries.rate.base",
                "stock.recruitment.ssbhalf",
                "stock.recruitment.shape",
            )
        )
    ]
    assert len(frozen) >= 39, f"expected >=39 frozen phase-13 params, got {len(frozen)}"
    # The 4 free halfsat keys must be DISJOINT from base_config
    for k in param_keys:
        assert k not in base_config, f"free param {k} must not be pre-set in base_config"


def test_phase14_single_objective_eval_is_finite(cal):
    """One objective eval must run with no FR config-validation error, finite result."""
    base_config, param_keys = _build_phase14_base_config(cal)
    targets = cal.load_targets()

    # Tiny run: 2 years keeps it fast; we only assert it assembles + evaluates clean.
    objective = cal.make_objective(
        base_config, targets, param_keys, n_years=2, seed=42, use_log_space=True
    )
    # x0 = log10(1.0) = 0.0 for all four -> K = 1.0 each (type-III, in range).
    x = np.zeros(len(param_keys))
    val = objective(x)
    assert isinstance(val, float)
    assert math.isfinite(val), f"objective returned non-finite value {val}"
    # 1e6 is the sentinel for a failed simulation (e.g. config-validation error).
    assert val < 1e6, (
        "objective returned the failed-simulation sentinel (1e6) — a single phase-14 "
        "eval should produce real biomass stats, not a validation/run failure."
    )
