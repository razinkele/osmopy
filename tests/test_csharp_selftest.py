"""Plan task 1.13: the deployment self-test, ``python -m osmose.engine_cs.selftest``.

The checks themselves (``problems``) are tested on synthetic frames, without the CLR. The command
runs in a child process, as it does on every deployment surface: once against the built engine
(it must pass), once against an empty engine directory (it must fail, loudly, with exit 1).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.selftest import problems

ROOT = Path(__file__).resolve().parents[1]


def _frame(values, times=None) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame(values, columns=[f"sp{i}" for i in range(values.shape[1])])
    frame.insert(0, "Time", np.arange(1, len(values) + 1) / 24 if times is None else times)
    frame["species"] = "all"
    return frame


GOOD = _frame([[1.0, 2.0], [1.5, 0.0], [2.0, 0.5]])


def test_a_healthy_run_has_no_problems():
    assert problems({"biomass": GOOD, "abundance": GOOD}, expected_rows=3) == []


@pytest.mark.parametrize(
    ("frames", "rows", "expected"),
    [
        ({"biomass": GOOD}, 4, "3 rows, expected 4"),
        ({"biomass": _frame([[1.0], [np.nan], [1.0]])}, 3, "non-finite"),
        ({"biomass": _frame([[1.0], [np.inf], [1.0]])}, 3, "non-finite"),
        ({"abundance": _frame([[1.0], [-1.0], [1.0]])}, 3, "negative"),
        ({"biomass": _frame([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])}, 3, "every species is at zero"),
        ({"biomass": _frame([[1.0], [1.0], [1.0]], times=[0.1, 0.1, 0.2])}, 3, "Time"),
        ({"biomass": GOOD[["Time", "species"]]}, 3, "no species columns"),
    ],
)
def test_each_failure_is_named(frames, rows, expected):
    found = problems(frames, expected_rows=rows)
    assert len(found) == 1 and expected in found[0], found


def test_an_extinct_abundance_alone_is_not_a_failure():
    """Only biomass must survive the year; abundance at zero on its own is reported by biomass."""
    zero_end = _frame([[1.0], [0.5], [0.0]])
    assert problems({"abundance": zero_end}, expected_rows=3) == []


def test_a_probe_that_raises_is_a_fail_line_not_a_traceback(monkeypatch, capsys):
    """An unreadable capabilities file makes ``csharp_engine_available`` raise. Nothing loads the
    CLR before the probe, so ``main`` can run in-process here."""
    from osmose import engine_cs
    from osmose.engine_cs.selftest import main

    def unreadable():
        raise PermissionError("Osmose.Engine.capabilities.json")

    monkeypatch.setattr(engine_cs, "csharp_engine_available", unreadable)
    assert main() == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL: the C# engine is not available: PermissionError"), out


def _selftest(env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    env = dict(os.environ, **env_extra)
    return subprocess.run(
        [sys.executable, "-m", "osmose.engine_cs.selftest"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )  # fmt: skip


def test_the_command_fails_loudly_without_the_engine(tmp_path):
    proc = _selftest({"OSMOSE_CSHARP_ENGINE_DIR": str(tmp_path)})
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout.startswith("FAIL: the C# engine is not available"), proc.stdout


def test_the_command_passes_with_the_built_engine():
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    proc = _selftest({})
    assert proc.returncode == 0, proc.stdout + proc.stderr[-4000:]
    assert proc.stdout.startswith("OK: C# engine ran 1 year (24 steps) of bay_of_biscay"), (
        proc.stdout
    )
