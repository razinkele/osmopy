"""Python must emit ``predatorPressure`` with Java's semantics: absolute prey biomass eaten per pair.

Discovered while trying to compare predation kernels across engines: Java writes BOTH
``dietMatrix`` (percentage composition) and ``predatorPressure`` (absolute biomass eaten), while this
engine wrote only ``dietMatrix`` — and its ``dietMatrix`` holds absolute BIOMASS (see
``write_diet_csv``: "Values are BIOMASS EATEN in tonnes"), i.e. Java's *predatorPressure* quantity
under Java's *dietMatrix* name. So the cross-engine comparison had no matching pair of files.

Layout is prey-row / predator-column (long), matching Java's orientation. Java additionally resolves
predator size stages; this engine has no size dimension, so its columns are whole-predator totals —
comparable to Java after summing Java's stage columns per predator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from osmose.engine.output import write_predator_pressure_csv


def test_writes_absolute_biomass_prey_rows_predator_columns(tmp_path):
    # 2 predators x 3 prey, biomass eaten in tonnes
    mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = tmp_path / "osm_predatorPressure_Simu0.csv"
    write_predator_pressure_csv(
        path=path,
        step_diet_matrices=[mat],
        step_times=[1.0],
        predator_names=["cod", "herring"],
        prey_names=["sprat", "smelt", "benthos"],
    )
    df = pd.read_csv(path)
    assert list(df.columns) == ["Time", "Prey", "cod", "herring"]
    assert len(df) == 3, "one row per prey per recorded time"
    # absolute biomass preserved, not normalised to fractions
    sprat = df[df["Prey"] == "sprat"].iloc[0]
    assert sprat["cod"] == 1.0 and sprat["herring"] == 4.0
    benthos = df[df["Prey"] == "benthos"].iloc[0]
    assert benthos["cod"] == 3.0 and benthos["herring"] == 6.0


def test_multiple_times_stack_as_rows(tmp_path):
    m1 = np.array([[1.0, 0.0]])
    m2 = np.array([[0.0, 2.0]])
    path = tmp_path / "p.csv"
    write_predator_pressure_csv(
        path=path,
        step_diet_matrices=[m1, m2],
        step_times=[1.0, 2.0],
        predator_names=["cod"],
        prey_names=["sprat", "smelt"],
    )
    df = pd.read_csv(path)
    assert len(df) == 4 and sorted(df["Time"].unique().tolist()) == [1.0, 2.0]


def test_engine_run_emits_the_file(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine

    res = osmose_demo("baltic", tmp_path)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    cfg["simulation.time.nyear"] = "1"
    out = tmp_path / "out"
    out.mkdir()
    PythonEngine().run(cfg, out, seed=42)
    hits = list(out.rglob("*predatorPressure_Simu0.csv"))
    assert hits, "engine run wrote no predatorPressure output"
    df = pd.read_csv(hits[0])
    assert df.columns[0] == "Time" and df.columns[1] == "Prey"
    assert df.iloc[:, 2:].to_numpy(dtype=float).sum() > 0.0, "all-zero predation pressure"


def test_values_are_per_step_means_over_the_recording_window(tmp_path):
    """Java's interval row is the per-step MEAN over the window; this engine must match.

    Verified empirically: a Java interval row is ~1/24 of the sum of its 24 per-step rows, while
    `diet_by_species` arrives already summed. Without dividing by the window length the two engines
    differ by exactly that factor — which made the first kernel comparison read 17-100x backwards.
    """
    mat = np.array([[24.0, 48.0]])  # biomass summed over a 24-step window
    path = tmp_path / "p.csv"
    write_predator_pressure_csv(
        path=path,
        step_diet_matrices=[mat],
        step_times=[1.0],
        predator_names=["cod"],
        prey_names=["sprat", "smelt"],
        steps_per_record=24,
    )
    df = pd.read_csv(path)
    assert df[df["Prey"] == "sprat"].iloc[0]["cod"] == 1.0
    assert df[df["Prey"] == "smelt"].iloc[0]["cod"] == 2.0
