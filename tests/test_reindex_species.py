"""Tests for the species re-indexing tool (scripts/reindex_species.py).

The tool relabels every `.sp{old}` / `.fsh{old}` config KEY to its new index
per a shift map, across all CSVs in a config dir, without touching values
(which may themselves contain sp-tokens, e.g. seasonality filenames) or
comments. It is the OSMOSE-mechanics prerequisite for inserting a focal
species (cod-east) into the contiguous sp{idx} namespace.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "reindex_species",
    Path(__file__).resolve().parents[1] / "scripts" / "reindex_species.py",
)
reindex_species = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reindex_species)
reindex = reindex_species.reindex


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_config(tmp_path: Path) -> Path:
    """A minimal config dir mirroring the append-shift scenario.

    Focal sp0-1, LTL sp2-3 (stand-ins), nspecies=2. Shifting LTL up by one
    (shifts={2:3, 3:4}) is the miniature of appending a focal species.
    """
    _write(
        tmp_path / "param-species.csv",
        [
            "# focal species",
            "species.name.sp0;cod",
            "species.name.sp1;herring",
            "species.linf.sp0;80.0",
        ],
    )
    _write(
        tmp_path / "param-reproduction.csv",
        [
            "# value contains an sp-token that MUST be preserved",
            "reproduction.season.file.sp0;reproduction/season-sp0.csv",
            "reproduction.season.file.sp2;reproduction/season-sp2.csv",
        ],
    )
    _write(
        tmp_path / "param-ltl.csv",
        [
            "species.name.sp2;Diatoms",
            "species.name.sp3;Dinoflagellates",
            "species.tl.sp3;1.0",
        ],
    )
    _write(
        tmp_path / "param-simulation.csv",
        ["simulation.nspecies;2", "simulation.nresource;2"],
    )
    return tmp_path


def _read_keys(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, val = line.partition(";")
        out[key.strip()] = val.strip()
    return out


def test_reindex_shifts_keys_up(synthetic_config: Path):
    reindex(synthetic_config, {2: 3, 3: 4}, new_nspecies=3)

    ltl = _read_keys(synthetic_config / "param-ltl.csv")
    # sp2 -> sp3, sp3 -> sp4; values preserved
    assert ltl["species.name.sp3"] == "Diatoms"
    assert ltl["species.name.sp4"] == "Dinoflagellates"
    assert ltl["species.tl.sp4"] == "1.0"
    assert "species.name.sp2" not in ltl  # old index gone


def test_reindex_leaves_unshifted_focal_untouched(synthetic_config: Path):
    reindex(synthetic_config, {2: 3, 3: 4}, new_nspecies=3)

    sp = _read_keys(synthetic_config / "param-species.csv")
    assert sp["species.name.sp0"] == "cod"  # 0 not in shifts
    assert sp["species.name.sp1"] == "herring"
    assert sp["species.linf.sp0"] == "80.0"


def test_reindex_rewrites_key_side_only_not_value(synthetic_config: Path):
    """The sp-token inside a VALUE (a filename) must survive unchanged even
    when the KEY's index is shifted."""
    reindex(synthetic_config, {2: 3, 3: 4}, new_nspecies=3)

    repro = _read_keys(synthetic_config / "param-reproduction.csv")
    # key sp2 -> sp3, but the value's 'season-sp2.csv' filename is untouched
    assert "reproduction.season.file.sp3" in repro
    assert repro["reproduction.season.file.sp3"] == "reproduction/season-sp2.csv"
    # sp0 key unchanged, value unchanged
    assert repro["reproduction.season.file.sp0"] == "reproduction/season-sp0.csv"


def test_reindex_bumps_nspecies(synthetic_config: Path):
    reindex(synthetic_config, {2: 3, 3: 4}, new_nspecies=3)

    sim = _read_keys(synthetic_config / "param-simulation.csv")
    assert sim["simulation.nspecies"] == "3"
    assert sim["simulation.nresource"] == "2"  # untouched


def test_reindex_preserves_comments_and_blank_lines(synthetic_config: Path):
    reindex(synthetic_config, {2: 3, 3: 4}, new_nspecies=3)
    text = (synthetic_config / "param-ltl.csv").read_text(encoding="utf-8")
    # no comment was present in ltl, but species.csv had one — check it survives
    sp_text = (synthetic_config / "param-species.csv").read_text(encoding="utf-8")
    assert "# focal species" in sp_text
    assert text.endswith("\n")
