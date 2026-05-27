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
    assert (tmp_path / "baltic_ev_high_f" / "seed0" / "osm_genetic_trait_means_Simu0.csv").exists()
    assert (tmp_path / "baltic_ev_low_f" / "seed0" / "osm_genetic_trait_means_Simu0.csv").exists()
