"""fishery-discards.csv must stay structurally consistent with fishery-catchability.csv —
identical species rows and identical fishery columns. The cod disaggregation
(cod -> cod_west + cod_east, adding the trawlcod_east fishery) updated catchability but left
discards stale (aggregate 'cod', no cod_east row, no trawlcod_east column). Python then silently
assigns cod_west/cod_east a zero discard rate by name-omission, and Java aborts on the missing
prey/fishery. These guards keep the two fishery matrices from drifting apart again.
"""

from pathlib import Path

import pandas as pd

CATCHABILITY = Path("data/baltic/fishery-catchability.csv")
DISCARDS = Path("data/baltic/fishery-discards.csv")


def _df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def test_discards_species_rows_match_catchability():
    cat, disc = _df(CATCHABILITY), _df(DISCARDS)
    assert list(disc.index) == list(cat.index)


def test_discards_fishery_columns_match_catchability():
    cat, disc = _df(CATCHABILITY), _df(DISCARDS)
    assert list(disc.columns) == list(cat.columns)


def test_discards_reflect_cod_disaggregation():
    disc = _df(DISCARDS)
    assert "cod_west" in disc.index and "cod_east" in disc.index
    assert "cod" not in disc.index  # aggregate cod row must be gone
    assert "trawlcod_east" in disc.columns  # cod_east fishery column present
