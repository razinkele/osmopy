"""`dietMatrix` must carry PERCENTAGE composition, matching Java (GitHub #144).

Java's dietMatrix holds each predator's diet as percentages; this engine wrote absolute tonnes under
the same filename, so comparing the two files divided tonnes by percentages. The absolute quantity now
has its own output (`predatorPressure`, Java's name for it), freeing dietMatrix to match Java.

`osmose/trophic_network.py` already documents that it renders "DIET COMPOSITION (% of a predator's
diet)" and does not normalise, so it was consuming tonnes as if they were percentages — this also
fixes that consumer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osmose.engine.output import write_diet_csv


def test_rows_are_per_predator_percentages(tmp_path):
    # cod eats 1 t sprat + 3 t smelt; herring eats 2 t sprat only
    mat = np.array([[1.0, 3.0], [2.0, 0.0]])
    path = tmp_path / "osm_dietMatrix_Simu0.csv"
    write_diet_csv(
        path=path, step_diet_matrices=[mat], step_times=[1.0],
        predator_names=["cod", "herring"], prey_names=["sprat", "smelt"],
    )
    df = pd.read_csv(path)
    row = df.iloc[0]
    assert row["cod_sprat"] == pytest.approx(25.0)
    assert row["cod_smelt"] == pytest.approx(75.0)
    assert row["herring_sprat"] == pytest.approx(100.0)
    assert row["herring_smelt"] == pytest.approx(0.0)


def test_each_predator_sums_to_100(tmp_path):
    rng = np.random.default_rng(0)
    mat = rng.random((3, 4)) * 10.0
    path = tmp_path / "d.csv"
    write_diet_csv(
        path=path, step_diet_matrices=[mat], step_times=[1.0],
        predator_names=["a", "b", "c"], prey_names=["w", "x", "y", "z"],
    )
    df = pd.read_csv(path)
    for pred in ("a", "b", "c"):
        cols = [c for c in df.columns if c.startswith(f"{pred}_")]
        assert df.iloc[0][cols].sum() == pytest.approx(100.0)


def test_predator_that_ate_nothing_is_all_zero_not_nan(tmp_path):
    mat = np.array([[0.0, 0.0], [2.0, 2.0]])
    path = tmp_path / "d.csv"
    write_diet_csv(
        path=path, step_diet_matrices=[mat], step_times=[1.0],
        predator_names=["starved", "fed"], prey_names=["x", "y"],
    )
    df = pd.read_csv(path)
    vals = df.iloc[0][["starved_x", "starved_y"]].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals)) and np.all(vals == 0.0)
    assert df.iloc[0]["fed_x"] == pytest.approx(50.0)
