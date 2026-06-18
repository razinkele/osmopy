"""Property-based tests: OSMOSE config writer->reader round-trip."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter
from tests.strategies import config_keys, config_kv_dicts, config_values


@given(d=config_kv_dicts())
def test_roundtrip_survives_and_keyset(d):
    with tempfile.TemporaryDirectory() as td:
        OsmoseConfigWriter().write(d, Path(td))
        result = OsmoseConfigReader().read(Path(td) / "osm_all-parameters.csv")
    # (a) every substantive key/value survives (exact STRING equality, no approx).
    for k, v in d.items():
        assert result[k] == v
    # (b) the substantive key set is preserved exactly — catches a routing change
    # that INVENTS a spurious substantive key (part (a) is blind to that). Exclude
    # writer/reader-injected framing metadata: the config-dir sentinel, the
    # `osmose.configuration.*` sub-file reference keys, and `osmose.version` (the
    # target-version writer stamps it into the master file). None are substantive
    # parameters the strategy generated; a spurious *substantive* key still fails.
    substantive = (
        set(result)
        - {"_osmose.config.dir", "osmose.version"}
        - {k for k in result if k.startswith("osmose.configuration.")}
    )
    assert substantive == set(d)


@given(key=config_keys(), value=config_values(), sep=st.sampled_from(["=", ";", ",", ":", "\t"]))
def test_separator_invariance(key, value, sep):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sub.csv"
        p.write_text(f"{key}{sep}{value}\n")
        result = OsmoseConfigReader().read_file(p)
    assert result[key] == value
