"""The predation-accessibility matrix covers cod_west + cod_east with distinct,
literature-grounded diets, and loads cleanly through the engine's name-resolving
AccessibilityMatrix (Phase 1 Task 4)."""

from pathlib import Path

import pandas as pd

MATRIX = Path("data/baltic/predation-accessibility.csv")

SPECIES_ORDER = [
    "cod_west", "herring", "sprat", "flounder", "perch", "pikeperch", "smelt",
    "stickleback", "cod_east", "Diatoms", "Dinoflagellates", "Microzooplankton",
    "Mesozooplankton", "Macrozooplankton", "Benthos",
]


def _df() -> pd.DataFrame:
    return pd.read_csv(MATRIX, sep=";", index_col=0)


def test_matrix_square_over_new_species_set():
    df = _df()
    assert list(df.index) == SPECIES_ORDER
    assert list(df.columns) == SPECIES_ORDER
    assert df.shape == (15, 15)


def test_no_nan_and_values_in_unit_range():
    df = _df()
    assert not df.isna().any().any(), "matrix has NaN"
    assert (df.values >= 0).all() and (df.values <= 1).all(), "accessibility outside [0,1]"


def test_cod_east_more_sprat_and_benthos_dependent():
    """Eastern cod are prey-limited: more sprat/benthos, less herring/coastal."""
    df = _df()
    # column = predator's diet; row = prey
    assert df.loc["sprat", "cod_east"] > df.loc["sprat", "cod_west"]
    assert df.loc["Benthos", "cod_east"] > df.loc["Benthos", "cod_west"]
    assert df.loc["herring", "cod_east"] < df.loc["herring", "cod_west"]


def test_no_cross_predation_between_stocks():
    """cod_west (SD22-24) and cod_east (deep basins) are spatially separated —
    neither preys on the other."""
    df = _df()
    assert df.loc["cod_east", "cod_west"] == 0
    assert df.loc["cod_west", "cod_east"] == 0
    # each retains within-stock cannibalism
    assert df.loc["cod_west", "cod_west"] > 0
    assert df.loc["cod_east", "cod_east"] > 0


def test_loads_through_engine_accessibility_matrix():
    from osmose.engine.accessibility import AccessibilityMatrix

    names = SPECIES_ORDER + ["GreySeal", "Cormorant"]  # background: absent from matrix, skipped
    am = AccessibilityMatrix.from_csv(str(MATRIX), names)
    assert am.raw_matrix.shape == (15, 15)
    # both cod species resolve to a matrix label
    assert am.resolve_name("cod_west") is not None
    assert am.resolve_name("cod_east") is not None
