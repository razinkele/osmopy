"""Maturity keys must mean the same thing to both engines.

Java 4.4.1 (`Species.<init>`, `Species.isSexuallyMature`) treats `species.maturity.size` and
`species.maturity.age` as ALTERNATIVES: if a size is set, it wins and the age is ignored; otherwise
the species matures by age. The Python engine requires `length >= size AND age >= age`, with an
absent size read as 0 and an absent age as 0. The two agree exactly when a species sets ONE of the
keys, never both. A species that set both would mature by size alone in Java and by size AND age in
Python.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader

ROOT = Path(__file__).resolve().parent.parent
MASTER_CONFIGS = [
    ROOT / "data" / "baltic" / "baltic_all-parameters.csv",
    ROOT / "data" / "eec_full" / "eec_all-parameters.csv",
    ROOT / "data" / "examples" / "osm_all-parameters.csv",
]


def _species_indices(cfg: dict[str, str]) -> list[int]:
    return sorted(int(k.rsplit("sp", 1)[1]) for k in cfg if k.startswith("species.name.sp"))


@pytest.mark.parametrize(
    "master", [p for p in MASTER_CONFIGS if p.exists()], ids=lambda p: p.parent.name
)
def test_no_species_sets_both_maturity_size_and_age(master):
    cfg = dict(OsmoseConfigReader().read(str(master)))
    both = [
        i
        for i in _species_indices(cfg)
        if f"species.maturity.size.sp{i}" in cfg and f"species.maturity.age.sp{i}" in cfg
    ]
    assert not both, (
        f"{master.name}: species {both} set both species.maturity.size and species.maturity.age. "
        "Java ignores the age when a size is set; the Python engine requires both. Keep one."
    )


def test_baltic_sprat_matures_by_age_only():
    """Sprat (sp2) spawns from age 1 (2026-11-13, docs/baltic_sprat_maturity_age_2026-11-13.md).
    Adding a maturity size back would make Java ignore the age."""
    cfg = dict(OsmoseConfigReader().read(str(MASTER_CONFIGS[0])))
    assert cfg["species.name.sp2"] == "sprat"
    assert float(cfg["species.maturity.age.sp2"]) == 1.0
    assert "species.maturity.size.sp2" not in cfg
