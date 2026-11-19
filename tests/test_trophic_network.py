from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from osmose.trophic_network import (
    _read_diet_matrix,
    _split_species,
    available_times,
    diet_column_pairs,
    diet_network_at,
    make_trophic_network_html,
    network_node_universe,
    species_layout,
)
from tests._data_guards import require_eec_output


def _write_diet(path, rows, cols):
    # rows: list of dicts with Time, Prey, <predator cols>; cols: predator column names
    df = pd.DataFrame(rows, columns=["Time", "Prey", *cols])
    df.to_csv(path, index=False)  # clean header, no preamble


def test_split_species():
    assert _split_species("cod in [10.000000, 30.000000[") == "cod"
    assert _split_species("Diatoms") == "Diatoms"


def test_read_diet_matrix_wildcard(tmp_path):
    d = tmp_path / "output" / "Trophic"
    d.mkdir(parents=True)
    _write_diet(
        d / "eec_dietMatrix_Simu0.csv",
        [{"Time": 1.0, "Prey": "herring", "cod in [0, 50[": 30.0}],
        ["cod in [0, 50["],
    )
    wide = _read_diet_matrix(tmp_path / "output")  # wildcard finds it under Trophic/
    assert list(wide.columns) == ["Time", "Prey", "cod in [0, 50["]


def test_read_diet_matrix_missing(tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(FileNotFoundError):
        _read_diet_matrix(tmp_path / "output")


def test_available_times(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_diet(
        d / "x_dietMatrix.csv",
        [
            {"Time": 2.0, "Prey": "a", "p in [0, 1[": 1.0},
            {"Time": 1.0, "Prey": "a", "p in [0, 1[": 1.0},
        ],
        ["p in [0, 1["],
    )
    assert available_times(d) == [1.0, 2.0]


def test_network_node_universe(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _write_diet(
        d / "x_dietMatrix.csv",
        [{"Time": 1.0, "Prey": "herring in [0, 5[", "cod in [0, 5[": 10.0, "cod in [5, 9[": 20.0}],
        ["cod in [0, 5[", "cod in [5, 9["],
    )
    assert network_node_universe(d, "species") == ["cod", "herring"]
    assert network_node_universe(d, "stage") == ["cod in [0, 5[", "cod in [5, 9[", "herring"]


def test_read_diet_matrix_eec_real():
    require_eec_output("*dietMatrix*")
    wide = _read_diet_matrix(Path("data/eec_full/output"))
    assert "Time" in wide.columns and "Prey" in wide.columns
    assert wide["Time"].nunique() == 70


def _diet_fixture(path):
    # herring has a DEAD [30,inf[ stage (all 0) — exercises dead-stage exclusion (NOT cod).
    # predator cols sum to ~100 per live stage. Includes a self-loop (cod eats cod) + a NaN.
    rows = [
        # prey-species "cod" split into 2 stages summed to species within a predator col
        {
            "Time": 1.0,
            "Prey": "cod in [0, 10[",
            "cod in [0, 50[": 5.0,
            "herring in [0, 10[": 0.0,
            "herring in [10, 30[": 0.0,
            "herring in [30, inf[": 0.0,
        },
        {
            "Time": 1.0,
            "Prey": "cod in [10, inf[",
            "cod in [0, 50[": 15.0,
            "herring in [0, 10[": 0.0,
            "herring in [10, 30[": 0.0,
            "herring in [30, inf[": 0.0,
        },
        {
            "Time": 1.0,
            "Prey": "herring in [0, 5[",
            "cod in [0, 50[": 80.0,
            "herring in [0, 10[": 60.0,
            "herring in [10, 30[": 40.0,
            "herring in [30, inf[": 0.0,
        },
        {
            "Time": 1.0,
            "Prey": "Diatoms",
            "cod in [0, 50[": float("nan"),
            "herring in [0, 10[": 40.0,
            "herring in [10, 30[": 60.0,
            "herring in [30, inf[": 0.0,
        },
    ]
    cols = ["cod in [0, 50[", "herring in [0, 10[", "herring in [10, 30[", "herring in [30, inf["]
    pd.DataFrame(rows, columns=["Time", "Prey", *cols]).to_csv(path, index=False)


def test_diet_network_species_prey_sum_and_dead_stage(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=0.0)
    m = {(r.predator, r.prey): r.proportion for r in net.itertuples()}
    # prey "cod" stages SUM within cod-predator: 5+15 = 20 (exact)
    assert m[("cod", "cod")] == pytest.approx(20.0)
    # herring predator: live stages are [0,10[ and [10,30[ (the [30,inf[ is all-zero=dead, excluded).
    # herring-on-Diatoms = mean(40, 60) over the 2 LIVE stages = 50 (NOT /3 incl. the dead stage)
    assert m[("herring", "Diatoms")] == pytest.approx(50.0)
    # herring-on-herring = mean(60, 40) = 50
    assert m[("herring", "herring")] == pytest.approx(50.0)


def test_diet_network_threshold_and_nan(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=30.0)
    # cod->cod is 20 -> filtered out at threshold 30; herring->Diatoms (50) kept
    assert ("cod", "cod") not in {(r.predator, r.prey) for r in net.itertuples()}
    assert (net["proportion"] >= 30.0).all()
    # cod->Diatoms was NaN -> dropped entirely
    assert ("cod", "Diatoms") not in {(r.predator, r.prey) for r in net.itertuples()}


def test_diet_network_stage_level(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    net = diet_network_at(d, time=1.0, threshold=0.0, predator_level="stage")
    preds = set(net["predator"])
    assert "cod in [0, 50[" in preds  # predator kept at stage granularity
    assert "cod" not in preds


def test_diet_network_bad_time(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    _diet_fixture(d / "x_dietMatrix.csv")
    with pytest.raises(ValueError):
        diet_network_at(d, time=99.0)


def test_diet_network_eec_real():
    require_eec_output("*dietMatrix*")
    net = diet_network_at(Path("data/eec_full/output"), time=1.0)  # no prefix (wildcard)
    assert list(net.columns) == ["predator", "prey", "proportion"]
    assert len(net) > 0 and (net["proportion"] >= 0).all()
    # species-level: no size suffix in node names
    assert not any(" in [" in s for s in set(net["predator"]) | set(net["prey"]))


def test_species_layout_deterministic():
    a = species_layout(["cod", "herring", "sprat"])
    b = species_layout(["sprat", "cod", "herring"])
    assert set(a) == {"cod", "herring", "sprat"}
    assert a["cod"] == b["cod"]  # deterministic (fixed seed), order-independent
    assert all(isinstance(v, tuple) and len(v) == 2 for v in a.values())


def test_make_trophic_network_html_self_contained_fixed_layout():
    pytest.importorskip("pyvis")
    import pandas as pd

    df = pd.DataFrame(
        {
            "predator": ["cod", "herring", "cod"],
            "prey": ["herring", "cod", "cod"],  # mutual cycle + self-loop
            "proportion": [70.0, 10.0, 20.0],
        }
    )
    pos = species_layout(["cod", "herring"])
    html = make_trophic_network_html(df, positions=pos, threshold=0.0)
    assert 'src="lib/' not in html  # self-contained (cdn_resources='in_line')
    assert '"physics"' in html and "false" in html  # physics disabled
    assert '"x"' in html and '"y"' in html  # fixed coords emitted
    assert "cod" in html and "herring" in html


def test_results_has_trophic_network_wiring():
    src = (Path(__file__).resolve().parent.parent / "ui" / "pages" / "results.py").read_text()
    assert "Trophic Network" in src  # the nav_panel
    assert "trophic_network" in src  # the output id / render fn
    assert "trophic_time" in src  # the time slider
    assert "make_trophic_network_html" in src  # the builder is used
    assert "update_slider" in src  # slider populated on load
    assert "_dietMatrix" not in src  # reads via the helper, not a hardcoded glob here


# ── Python-engine diet-matrix format (Time, <pred>_<prey> columns; species-level) ──
# The engine writes `Time, <predator>_<prey>` (predator-major, biomass eaten), NOT the
# Java `Time, Prey, <predator-stage cols>` layout. The trophic reader must accept both
# or it raises KeyError 'Prey' on every Python-engine run.


def _write_engine_diet(path, rows):
    """Engine format: comma-separated `Time, <pred>_<prey>` columns (no Prey column)."""
    pd.DataFrame(rows).to_csv(path, index=False)


def test_engine_format_node_universe(tmp_path):
    _write_engine_diet(
        tmp_path / "osm_dietMatrix_Simu0.csv",
        [{"Time": 0.0, "cod_herring": 10.0, "cod_sprat": 30.0, "herring_sprat": 5.0}],
    )
    assert set(network_node_universe(tmp_path)) == {"cod", "herring", "sprat"}
    # 'stage' falls back to species for engine output (no size-stages) — no error.
    assert set(network_node_universe(tmp_path, predator_level="stage")) == {
        "cod",
        "herring",
        "sprat",
    }


def test_engine_format_diet_network_at(tmp_path):
    # cod ate 40 t (herring 10, sprat 30) -> 25% / 75%; herring ate 5 t sprat -> 100%.
    _write_engine_diet(
        tmp_path / "osm_dietMatrix_Simu0.csv",
        [
            {
                "Time": 0.0,
                "cod_herring": 10.0,
                "cod_sprat": 30.0,
                "herring_sprat": 5.0,
                "herring_cod": 0.0,
            }
        ],
    )
    net = diet_network_at(tmp_path, time=0.0, threshold=1.0)
    cod = net[net["predator"] == "cod"].set_index("prey")["proportion"]
    assert cod["herring"] == pytest.approx(25.0)
    assert cod["sprat"] == pytest.approx(75.0)
    herring = net[net["predator"] == "herring"].set_index("prey")["proportion"]
    assert herring["sprat"] == pytest.approx(100.0)
    # 'stage' falls back to species (identical edges) for engine output.
    net_stage = diet_network_at(tmp_path, time=0.0, threshold=1.0, predator_level="stage")
    assert set(zip(net_stage["predator"], net_stage["prey"])) == set(
        zip(net["predator"], net["prey"])
    )


def test_engine_format_available_times(tmp_path):
    _write_engine_diet(
        tmp_path / "osm_dietMatrix_Simu0.csv",
        [{"Time": 0.0, "cod_herring": 1.0}, {"Time": 1.0, "cod_herring": 2.0}],
    )
    assert available_times(tmp_path) == [0.0, 1.0]


# ── Species names with an underscore (Baltic's cod_west, cod_east) ──
# A column is `<predator>_<prey>` and either name may hold an underscore, so no fixed
# underscore is the split. Splitting at the first one read `cod_west_herring` as the predator
# `cod` eating `west_herring`, on the shipped Baltic config.

_BALTIC_PREDATORS = ["cod_west", "herring", "cod_east", "GreySeal"]
_BALTIC_PREY = ["cod_west", "herring", "cod_east", "GreySeal", "Mesozooplankton"]


def _engine_columns(predators, prey):
    """The engine's column order: predator by predator, every predator over the same prey."""
    return [f"{p}_{q}" for p in predators for q in prey]


def test_a_name_with_an_underscore_is_split_where_the_matrix_says():
    cols = _engine_columns(_BALTIC_PREDATORS, _BALTIC_PREY)
    pairs = diet_column_pairs(["Time", *cols, "species"])
    assert [c for c, _, _ in pairs] == cols  # the metadata columns are not pairs
    assert [(p, q) for _, p, q in pairs] == [
        (p, q) for p in _BALTIC_PREDATORS for q in _BALTIC_PREY
    ]


def test_the_older_layout_without_background_prey_is_split_the_same_way():
    """Before the background species had prey columns, the background predators were not
    among the prey. The block structure alone still settles the split."""
    prey = ["cod_west", "herring", "cod_east", "Mesozooplankton"]
    pairs = diet_column_pairs(_engine_columns(_BALTIC_PREDATORS, prey))
    assert [(p, q) for _, p, q in pairs] == [(p, q) for p in _BALTIC_PREDATORS for q in prey]


def test_a_predator_whose_name_starts_another_predators_name():
    """`cod` next to `cod_west`: a run of columns starting `cod_` covers both blocks."""
    predators, prey = ["cod", "cod_west", "sprat"], ["cod", "cod_west", "sprat", "Zoo"]
    pairs = diet_column_pairs(_engine_columns(predators, prey))
    assert [(p, q) for _, p, q in pairs] == [(p, q) for p in predators for q in prey]


def test_resources_named_after_the_fish_do_not_turn_the_grid_on_its_side():
    """Found in review. Fish `cod` and `herring`, resources `west_cod` and `west_herring`: the
    columns are a 2 x 4 grid, and also a 4 x 2 grid of predators `cod`, `cod_west`, `herring`,
    `herring_west` over the prey `cod` and `herring`. In both, two predators are also prey.
    The engine's prey axis holds every fish and the resources besides, so the longer prey axis
    is taken. Split at the first underscore, this frame was read correctly: it must stay so."""
    predators, prey = ["cod", "herring"], ["cod", "herring", "west_cod", "west_herring"]
    pairs = diet_column_pairs(_engine_columns(predators, prey))
    assert [(p, q) for _, p, q in pairs] == [(p, q) for p in predators for q in prey]


def test_one_fish_and_a_resource_named_after_it_is_one_predator_over_two_prey():
    """`cod_cod, cod_west_cod`: the fish `cod` over the prey `cod` and `west_cod`, or the
    predators `cod` and `cod_west` over the one prey `cod`. Both are layouts the engine could
    write. The longer prey axis is taken: a run with no resource at all is the rarer one."""
    pairs = diet_column_pairs(["cod_cod", "cod_west_cod"])
    assert [(p, q) for _, p, q in pairs] == [("cod", "cod"), ("cod", "west_cod")]


def test_a_full_grid_in_another_order_than_the_engines_is_still_read_as_a_grid():
    """Prey in another order than the predators: not laid out as the engine does, but a full
    grid all the same."""
    predators, prey = ["a_x", "b"], ["b", "a_x"]
    pairs = diet_column_pairs(_engine_columns(predators, prey))
    assert [(p, q) for _, p, q in pairs] == [(p, q) for p in predators for q in prey]
    # One predator, so two readings: the one in which the predator is also a prey is taken.
    pairs = diet_column_pairs(_engine_columns(["a_x"], ["z", "a_x"]))
    assert [(p, q) for _, p, q in pairs] == [("a_x", "z"), ("a_x", "a_x")]
    # Nothing tells two readings apart: the first underscore, as before.
    assert diet_column_pairs(["a_b_x", "c_b_x"]) == [("a_b_x", "a", "b_x"), ("c_b_x", "c", "b_x")]


def test_a_single_predator_with_an_underscore_is_read_as_a_species_that_is_also_prey():
    """One predator leaves two readings (`cod` eating `west_...`, or `cod_west`). A fish
    species is always among its own prey names, and that settles it."""
    pairs = diet_column_pairs(_engine_columns(["cod_west"], ["cod_west", "Zoo"]))
    assert [(p, q) for _, p, q in pairs] == [("cod_west", "cod_west"), ("cod_west", "Zoo")]


def test_names_without_underscores_and_frames_made_by_hand_read_as_before():
    assert diet_column_pairs(_engine_columns(["cod", "sprat"], ["cod", "sprat", "Zoo"])) == [
        (f"{p}_{q}", p, q) for p in ["cod", "sprat"] for q in ["cod", "sprat", "Zoo"]
    ]
    # Not a full predator x prey grid: nothing to infer from, so the first underscore it is.
    ragged = ["cod_herring", "cod_sprat", "herring_sprat", "herring_cod"]
    assert diet_column_pairs(ragged) == [
        ("cod_herring", "cod", "herring"),
        ("cod_sprat", "cod", "sprat"),
        ("herring_sprat", "herring", "sprat"),
        ("herring_cod", "herring", "cod"),
    ]
    # The second block starts like the first and then differs: still no grid.
    assert diet_column_pairs(["a_x", "a_y", "b_x", "b_z"]) == [
        ("a_x", "a", "x"),
        ("a_y", "a", "y"),
        ("b_x", "b", "x"),
        ("b_z", "b", "z"),
    ]
    assert diet_column_pairs(["Time", "species"]) == []


def test_the_network_of_a_baltic_named_matrix_has_the_real_species(tmp_path):
    cols = _engine_columns(_BALTIC_PREDATORS, _BALTIC_PREY)
    row = {"Time": 0.0} | dict.fromkeys(cols, 0.0)
    row |= {"cod_west_herring": 30.0, "cod_west_cod_east": 10.0, "cod_east_Mesozooplankton": 5.0}
    _write_engine_diet(tmp_path / "osm_dietMatrix_Simu0.csv", [row])
    assert set(network_node_universe(tmp_path)) == set(_BALTIC_PREY)
    net = diet_network_at(tmp_path, time=0.0, threshold=1.0)
    west = net[net["predator"] == "cod_west"].set_index("prey")["proportion"]
    assert west["herring"] == pytest.approx(75.0) and west["cod_east"] == pytest.approx(25.0)
    east = net[net["predator"] == "cod_east"].set_index("prey")["proportion"]
    assert east["Mesozooplankton"] == pytest.approx(100.0)
    assert "cod" not in set(net["predator"]) | set(net["prey"])


def test_the_reader_splits_back_what_the_writer_wrote():
    """The names come from the engine's own writer, underscores in every kind of name: focal,
    background and resource. Whatever the prey axis is, the reader recovers both axes."""
    from types import SimpleNamespace

    import numpy as np

    from osmose.engine.output import (
        _build_diet_dataframe,
        diet_prey_names,
        pressure_predator_names,
    )

    cfg = SimpleNamespace(
        n_dt_per_year=2,
        output_record_frequency=1,
        n_species=3,
        n_background=1,
        species_names=["cod_west", "herring", "cod_east"],
        all_species_names=["cod_west", "herring", "cod_east", "Grey_Seal"],
        resource_names=["Meso_zooplankton", "Benthos"],
    )
    predators, prey = pressure_predator_names(cfg), diet_prey_names(cfg)
    mat = np.arange(len(predators) * len(prey), dtype=float).reshape(len(predators), len(prey))
    outputs = [SimpleNamespace(step=1, diet_by_species=mat, window_steps=1)]
    df = _build_diet_dataframe(outputs, cfg)["dietMatrix"]
    pairs = diet_column_pairs(df.columns)
    assert len(pairs) == len(predators) * len(prey)
    assert [(p, q) for _, p, q in pairs] == [(p, q) for p in predators for q in prey]
    assert all(c in df.columns for c, _, _ in pairs)


@pytest.fixture(scope="module")
def baltic_output(tmp_path_factory):
    """One year of the Baltic demo as shipped, written to disk. It writes its diet matrix."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig

    tmp = tmp_path_factory.mktemp("baltic_diet")
    cfg = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    cfg["simulation.time.nyear"] = "1"
    out = tmp / "out"
    out.mkdir()
    PythonEngine().run(cfg, out, seed=42)
    return out, EngineConfig.from_dict(cfg)


def test_a_real_baltic_run_is_read_with_its_own_species(baltic_output):
    """The defect and the fix on what the engine really writes, through the two entry points
    the results page uses. The split is inferred from the whole frame, so one stray column
    would send every name back to the first underscore with the synthetic tests still green:
    only the real file can show that the real file is read."""
    from osmose.results import OsmoseResults
    from ui.pages.results import make_diet_heatmap

    out, config = baltic_output
    fish = list(config.species_names)
    predators = list(config.all_species_names)[: config.n_species + config.n_background]
    assert {"cod_west", "cod_east"} <= set(fish)

    nodes = set(network_node_universe(out))
    assert set(predators) <= nodes
    assert "cod" not in nodes and not any(n.startswith(("west_", "east_")) for n in nodes)
    net = diet_network_at(out, time=available_times(out)[-1], threshold=0.0)
    assert set(net["predator"]) <= set(predators) and "cod_west" in set(net["predator"])

    fig = make_diet_heatmap(OsmoseResults(out, strict=False).diet_matrix())
    assert list(fig.data[0].y) == predators
    assert list(fig.data[0].x)[: len(fish)] == fish
