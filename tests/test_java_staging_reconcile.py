"""The Java staging path must reconcile names/matrices, or Java aborts on `cod_west`.

Java 4.4.1's ``Species.java`` strips ``_``/``-`` when building a species' internal name, but
leaves name-based *references* (``movement.species.mapN`` values, matrix headers) untouched, so
a disaggregated config aborts at init with "does not match any predefined species name".
``osmose.java_config_reconcile`` fixes that on the staged copy; the bug (GitHub #138) was that
only ``scripts/baltic_stability_certify.py`` called it while the Run tab did not.

These guards pin the staged OUTPUT rather than the call, so they fail whichever staging entry
point regresses. They need no jar — only the tracked Baltic config — so they run in CI, unlike
``tests/test_java_engine_thread.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from ui.pages.run import stage_config_for_java

_NAME_VALUE_PREFIXES = ("species.name.sp", "movement.species.map", "fisheries.name.fsh")


def _master_entries(master: Path) -> list[tuple[str, str]]:
    out = []
    for raw in master.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ";" not in line:
            continue
        key, _, value = line.partition(";")
        out.append((key.strip(), value.strip()))
    return out


@pytest.fixture(scope="module")
def baltic_config():
    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    return cfg, res["config_file"].parent


def test_staged_names_are_java_safe_at_441(baltic_config, tmp_path):
    """Every name-resolving VALUE must be alphanumeric, matching Java's stripped internal name."""
    cfg, source_dir = baltic_config
    master, _overrides = stage_config_for_java(
        cfg, tmp_path / "stage", source_dir, target_version="4.4.1"
    )
    offenders = [
        (k, v)
        for k, v in _master_entries(master)
        if k.startswith(_NAME_VALUE_PREFIXES) and ("_" in v or "-" in v)
    ]
    assert not offenders, (
        f"staged config carries names Java cannot resolve: {offenders[:5]} — "
        "reconcile_config_for_java did not run on this staging path (see GitHub #138)"
    )


def test_movement_species_maps_resolve_to_declared_species(baltic_config, tmp_path):
    """The exact failure from #138: movement.species.map0=cod_west with no matching species."""
    cfg, source_dir = baltic_config
    master, _overrides = stage_config_for_java(
        cfg, tmp_path / "stage", source_dir, target_version="4.4.1"
    )
    entries = _master_entries(master)
    declared = {v for k, v in entries if k.startswith("species.name.sp")}
    referenced = {v for k, v in entries if k.startswith("movement.species.map")}
    assert referenced, "no movement.species.map* entries found — fixture assumption broke"
    assert referenced <= declared, (
        f"movement maps reference undeclared species {sorted(referenced - declared)}; "
        f"declared: {sorted(declared)}"
    )


def test_pre_440_target_is_left_unmangled(baltic_config, tmp_path):
    """Name stripping mirrors 4.4.x behaviour; applying it to the 4.3.3 jar would corrupt names."""
    cfg, source_dir = baltic_config
    master, _overrides = stage_config_for_java(
        cfg, tmp_path / "stage433", source_dir, target_version="4.3.3"
    )
    names = [v for k, v in _master_entries(master) if k.startswith("species.name.sp")]
    assert any("_" in n for n in names), (
        "4.3.3 staging must preserve raw species names (cod_west), not Java-4.4.1-stripped ones"
    )
