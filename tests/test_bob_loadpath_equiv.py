from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "examples_433_orig").exists()
    or not (ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay_24step.nc").exists(),
    reason="need Task 3 snapshot + Task 2 24-step file",
)


def test_ltl_and_native_load_paths_are_bit_exact():
    from scripts.native_440_parity import bob_loadpath_equiv

    assert bob_loadpath_equiv(years=3, seed=42) == 0.0  # np.array_equal via max abs diff
