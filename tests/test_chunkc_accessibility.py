import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import chunkc_accessibility as cc  # noqa: E402

_DEPLOYED = (
    "v Prey / Predator >;cod;herring;sprat;smelt\n"
    "cod;0.05;0;0;0.05\n"
    "herring;0.4;0;0;0\n"
    "sprat;0.4;0;0;0\n"
    "smelt;0.1;0.2;0.2;0\n"
)


def _write_deployed(tmp_path) -> str:
    p = tmp_path / "deployed.csv"
    p.write_text(_DEPLOYED)
    return str(p)


def test_write_chunkc_matrix_sets_only_cod_to_clupeids(tmp_path):
    dep = _write_deployed(tmp_path)
    out = str(tmp_path / "chunkc.csv")
    assert cc.write_chunkc_matrix(dep, 0.3, out) == out
    d = pd.read_csv(dep, sep=";", index_col=0)
    v = pd.read_csv(out, sep=";", index_col=0)
    # cod -> herring and cod -> sprat set to 0.3
    assert v.loc["cod", "herring"] == 0.3
    assert v.loc["cod", "sprat"] == 0.3
    # cod cannibalism and every other cell unchanged
    changed = {(r, c) for r in v.index for c in v.columns if v.loc[r, c] != d.loc[r, c]}
    assert changed == {("cod", "herring"), ("cod", "sprat")}


def test_write_chunkc_matrix_missing_labels_raises(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("v Prey / Predator >;cod;flounder\ncod;0.05;0\nflounder;0.1;0\n")
    with pytest.raises(KeyError):
        cc.write_chunkc_matrix(str(p), 0.2, str(tmp_path / "o.csv"))
