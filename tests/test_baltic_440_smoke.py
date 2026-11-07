"""``scripts/baltic_440_smoke.py`` must find this repo's Baltic config and stage it for Java.

It pointed at ``/home/razinka/osmose/osmose-python``, a checkout that no longer exists, so it died
on a missing file before reaching Java. Behind that, it staged with ``write_temp_config`` plus
background staging but no 4.4.x name reconcile, which the cod east/west split made necessary:
4.4.1 cannot resolve ``cod_west`` (GitHub #138). It now stages through
``osmose.java_staging.stage_config_for_java`` like every other Java run.

``subprocess.run`` is faked, so these need no jar and run in CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import baltic_440_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]


def test_its_paths_point_at_this_repo():
    assert smoke.BALTIC == ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
    assert smoke.BALTIC.exists()


def test_java_gets_a_staged_reconciled_baltic(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "")  # non-zero: skips the run's checks

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    assert smoke.stage_and_run(years=1, out=tmp_path) == 1

    (cmd,) = calls
    master = Path(cmd[cmd.index("-jar") + 2])
    names = [
        line.partition(";")[2].strip()
        for line in master.read_text().splitlines()
        if line.startswith("species.name.sp")
    ]
    assert names and not [n for n in names if "_" in n], (
        f"4.4.1 cannot resolve these staged names (GitHub #138): {[n for n in names if '_' in n]}"
    )
    assert "-Poutput.cutoff.enabled=false" in cmd, "background-staging override missing"
