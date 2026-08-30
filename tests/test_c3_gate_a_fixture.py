from pathlib import Path

import numpy as np
import pandas as pd

from scripts.c3_gate_a_reference import check_against_fixture, load_gate_a_fixture

FIXTURE = (
    Path(__file__).resolve().parents[1] / "docs" / "diagnostics" / "c3_gate_a_master_baseline.json"
)


def test_fixture_exists_and_has_five_seeds_and_eleven_columns():
    fx = load_gate_a_fixture(FIXTURE)
    assert fx["seeds"] == [42, 123, 7, 999, 2024]
    assert len(fx["columns"]) == 11 and "cod_west" in fx["columns"] and "GreySeal" in fx["columns"]
    for s in fx["seeds"]:
        arr = np.asarray(fx["series"][str(s)], dtype=float)
        assert arr.shape == (50, 11) and np.all(np.isfinite(arr))
    assert len(fx["engine_commit"]) >= 7


def test_check_against_fixture_reports_only_differing_columns():
    fx = load_gate_a_fixture(FIXTURE)
    arr = np.asarray(fx["series"]["42"], dtype=float)
    df = pd.DataFrame(arr, columns=fx["columns"])
    df.insert(0, "Time", np.arange(50, dtype=float))
    assert check_against_fixture(fx, 42, df) == []
    df2 = df.copy()
    df2["herring"] = df2["herring"] * (1 + 1e-12)
    assert check_against_fixture(fx, 42, df2) == ["herring"]
