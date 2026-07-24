import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from calibrate_baltic import get_phase13_shepherd_params  # noqa: E402


def test_shepherd_beta_bounds_forbid_undercompensation_and_overcrush():
    keys, bounds, _ = get_phase13_shepherd_params()
    beta = [(k, b) for k, b in zip(keys, bounds) if k.startswith("stock.recruitment.shape.")]
    assert len(beta) == 9  # 9 focal species after cod disaggregation (cod_west + cod_east)
    for k, (lo, hi) in beta:
        # lower bound >= 1.0 (no under-compensation), upper <= 3.0 (no extreme over-crush)
        assert math.isclose(10**lo, 1.0, rel_tol=1e-6), f"{k} lower {10**lo} != 1.0"
        assert math.isclose(10**hi, 3.0, rel_tol=1e-6), f"{k} upper {10**hi} != 3.0"
