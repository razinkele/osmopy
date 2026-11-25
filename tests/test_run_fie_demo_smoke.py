import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_run_fie_demo_short_smoke(tmp_path: Path) -> None:
    """Smoke: script must produce both scenario CSVs + a PNG within ~5 min on
    a 10-year, 1-seed override."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_fie_demo.py",
            "--n-years",
            "10",
            "--seeds",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "fie_imax_trajectory.png").exists()
    for arm in ("baltic_ev_high_f", "baltic_ev_low_f"):  # files carry output.file.prefix
        assert list((tmp_path / arm / "seed0").glob("*_genetic_trait_means_Simu0.csv"))
