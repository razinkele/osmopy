import subprocess
import sys
from pathlib import Path

import pytest

_FINE_READY = (
    Path("data/baltic-fine/baltic_fine_grid.nc").exists()
    and Path("data/baltic-fine/maps/perch_adult.csv").exists()
)
pytestmark = pytest.mark.skipif(
    not _FINE_READY,
    reason="needs end-to-end fine-grid build (EMODnet fetch + real percid maps + fine salinity)",
)


def test_ladder_runs_three_rungs_and_verdict():
    out = subprocess.run(
        [sys.executable, "scripts/baltic_fine_grid_ladder.py", "--nyear", "2", "--seeds", "1"],
        capture_output=True,
        text=True,
        timeout=5400,
    )
    assert out.returncode == 0, out.stderr
    lo = out.stdout.lower()
    assert "coarse" in lo and "4x-upsampled" in lo and "4x-real" in lo
    assert "perch" in lo and "pikeperch" in lo and "area" in lo
    assert "go" in lo or "no-go" in lo
