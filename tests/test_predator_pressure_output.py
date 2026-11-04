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
import pytest

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


# --- Background predators (GreySeal, Cormorant) -------------------------------------------------
# The kernel has always booked what background schools eat into ctx.diet_matrix; the per-step
# aggregation then dropped every row past the focal block, so neither output could attribute any
# predation to the seal or the cormorant. predatorPressure now carries them; dietMatrix stays
# focal-only (diet_distance returns inf on any shape change, so widening it is a separate call).


def _two_focal_one_bkg():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        n_dt_per_year=2,
        output_record_frequency=1,
        n_species=2,
        n_background=1,
        species_names=["cod", "sprat"],
        all_species_names=["cod", "sprat", "GreySeal"],
        resource_names=["zoo"],
    )
    # rows: cod, sprat, GreySeal; prey cols: cod, sprat, zoo
    mat = np.array([[0.0, 3.0, 1.0], [0.0, 0.0, 5.0], [2.0, 7.0, 0.0]])
    outputs = [SimpleNamespace(step=1, diet_by_species=mat)]
    return cfg, outputs


def test_in_memory_predator_pressure_has_background_columns():
    from osmose.engine.output import _build_predator_pressure_dataframe

    cfg, outputs = _two_focal_one_bkg()
    df = _build_predator_pressure_dataframe(outputs, cfg)["predatorPressure"]
    assert list(df.columns) == ["Time", "Prey", "cod", "sprat", "GreySeal"]
    sprat = df[df["Prey"] == "sprat"].iloc[0]
    assert sprat["GreySeal"] == 7.0 and sprat["cod"] == 3.0


def test_in_memory_diet_matrix_stays_focal_only():
    from osmose.engine.output import _build_diet_dataframe

    cfg, outputs = _two_focal_one_bkg()
    df = _build_diet_dataframe(outputs, cfg)["dietMatrix"]
    assert not any(c.startswith("GreySeal_") for c in df.columns)
    # cod's row is percentages over its own prey: 3 of 4 tonnes were sprat
    assert df.iloc[0]["cod_sprat"] == 75.0


@pytest.fixture(scope="module")
def baltic_run(tmp_path_factory):
    """1-yr Baltic demo run on disk, capturing the kernel's raw per-school diet each step."""
    from unittest import mock

    import osmose.engine.output as _output
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine

    tmp = tmp_path_factory.mktemp("bkg")
    res = osmose_demo("baltic", tmp)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    cfg["simulation.time.nyear"] = "1"
    out = tmp / "out"
    out.mkdir()
    real = _output.aggregate_diet_by_species
    raw: list[tuple[np.ndarray, np.ndarray]] = []

    def _capture(diet_matrix, species_id, n_pred_species):
        raw.append((np.array(diet_matrix), np.array(species_id)))
        return real(diet_matrix, species_id, n_pred_species)

    with mock.patch.object(_output, "aggregate_diet_by_species", _capture):
        PythonEngine().run(cfg, out, seed=42)
    (pp,) = out.rglob("*predatorPressure_Simu0.csv")
    (dm,) = out.rglob("*dietMatrix_Simu0.csv")
    return cfg, raw, pd.read_csv(pp), pd.read_csv(dm)


def test_engine_predator_pressure_attributes_background_predation(baltic_run):
    from osmose.engine.config import EngineConfig

    cfg, raw, pp, _ = baltic_run
    ec = EngineConfig.from_dict(cfg)
    assert ec.n_background == 2
    assert list(pp.columns) == ["Time", "Prey", *ec.all_species_names]
    steps = max(1, int(ec.output_record_frequency))
    for slot in range(ec.n_species, ec.n_species + ec.n_background):
        name = ec.all_species_names[slot]
        booked = sum(float(m[s == slot].sum()) for m, s in raw)
        assert booked > 0.0, f"kernel booked nothing for {name}"
        # the output carries exactly what the kernel booked (rows are per-step means)
        assert pp[name].sum() * steps == pytest.approx(booked, rel=1e-9), name


def test_engine_focal_columns_unchanged_by_background_rows(baltic_run):
    from osmose.engine.config import EngineConfig

    cfg, raw, pp, dm = baltic_run
    ec = EngineConfig.from_dict(cfg)
    steps = max(1, int(ec.output_record_frequency))
    for slot, name in enumerate(ec.species_names):
        booked = sum(float(m[s == slot].sum()) for m, s in raw)
        assert pp[name].sum() * steps == pytest.approx(booked, rel=1e-9), name
    assert not any(c.startswith(("GreySeal_", "Cormorant_")) for c in dm.columns)
