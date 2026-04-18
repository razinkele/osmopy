"""Baltic config constraints that the Java 4.3.3 engine enforces.

The Python engine is permissive about fishery-name shape (it resolves
catchability by row/column index, not name), so a config that Python
happily loads can still break Java at boot time. These tests are a
cheap static lint over ``data/baltic/`` that catches the
known-regressing cases before anyone runs Java.

Scope is deliberately narrow: Baltic only. When another ecosystem
starts being Java-cross-checked, add a parametrized version.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALTIC_FISHING = PROJECT_ROOT / "data" / "baltic" / "baltic_param-fishing.csv"
BALTIC_CATCHABILITY = PROJECT_ROOT / "data" / "baltic" / "fishery-catchability.csv"
BALTIC_DISCARDS = PROJECT_ROOT / "data" / "baltic" / "fishery-discards.csv"

# Java's FishingGear.java:107-109 strips `_` and `-` from fishery-name values
# and then enforces `name.matches("^[a-zA-Z0-9]*$")`. Since we use the
# stripped form as the canonical name directly (no underscores or hyphens
# in the committed values), the regex pins the *committed* spelling to
# alphanumeric-only — that way any future reintroduction of `_` or `-`
# for human-readability trips CI instead of silently disabling every
# fishery at Java boot.
JAVA_FISHERY_NAME_RE = re.compile(r"^[a-zA-Z0-9]+$")


def _fishery_name_values() -> list[tuple[str, str]]:
    """Extract every ``fisheries.name.fshN`` value from the Baltic fishing CSV.

    Returns [(key, value), ...]. Skips blank/comment lines and tolerates
    the OSMOSE ';' separator plus inline comments.
    """
    if not BALTIC_FISHING.exists():
        pytest.skip("Baltic fishing config not present in this checkout")
    entries: list[tuple[str, str]] = []
    for raw in BALTIC_FISHING.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("fisheries.name.fsh"):
            continue
        key, sep, value = line.partition(";")
        assert sep == ";", f"unexpected separator in {raw!r}"
        entries.append((key.strip(), value.strip()))
    return entries


def _fishery_movement_map_values() -> list[tuple[str, str]]:
    """Extract every ``fisheries.movement.fishery.mapN`` value."""
    if not BALTIC_FISHING.exists():
        pytest.skip("Baltic fishing config not present in this checkout")
    entries: list[tuple[str, str]] = []
    for raw in BALTIC_FISHING.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("fisheries.movement.fishery.map"):
            continue
        key, sep, value = line.partition(";")
        assert sep == ";", f"unexpected separator in {raw!r}"
        entries.append((key.strip(), value.strip()))
    return entries


def test_baltic_fishery_names_are_java_safe():
    """Every ``fisheries.name.fshN`` value must pass Java's post-strip
    regex. Java normalizes by removing `_` and `-`, then rejects any
    remaining non-alphanumeric character. Since we already use the
    stripped form as canonical, the committed value must be purely
    alphanumeric — catching a future regression that reintroduces `_`
    for readability (the exact failure this test exists to prevent)."""
    entries = _fishery_name_values()
    assert entries, "Expected at least one fisheries.name.fshN line"
    bad = [(k, v) for k, v in entries if not JAVA_FISHERY_NAME_RE.match(v)]
    assert not bad, (
        f"Java-incompatible fishery names: {bad}. "
        "FishingGear.java:107-109 strips `_`/`-` then enforces "
        "`^[a-zA-Z0-9]*$`. Use the stripped form as the committed "
        "value, not the underscored human-readable one."
    )


def test_baltic_fishery_map_values_match_fishery_names():
    """Every ``fisheries.movement.fishery.mapN`` value must refer to a
    fishery that actually exists by the (stripped) name Java will use.
    Java compares these map-side values to the stripped FishingGear name
    with a case-sensitive ``equals()`` and NO stripping (see
    FisheryMapSet.java:209). If the map value has an underscore but the
    name doesn't, Java silently deactivates the fishery for all steps.
    """
    names = {v for _, v in _fishery_name_values()}
    map_values = {v for _, v in _fishery_movement_map_values()}
    orphan_maps = map_values - names
    assert not orphan_maps, (
        f"fisheries.movement.fishery.mapN values that don't match any "
        f"fisheries.name.fshN: {orphan_maps}. Java will deactivate the "
        "fishery for every step. Check spelling and underscore stripping."
    )


def test_baltic_catchability_headers_match_fishery_names():
    """The column header of ``fishery-catchability.csv`` must match the
    fishery names exactly (including case). Java's AccessibilityManager
    resolves gears by name, so a header mismatch produces
    ``[severe] No catchability found for fishery X`` at boot."""
    if not BALTIC_CATCHABILITY.exists():
        pytest.skip("Baltic catchability CSV not present in this checkout")
    header = BALTIC_CATCHABILITY.read_text().splitlines()[0].split(",")
    # First cell is empty/row-label placeholder; rest are fishery names.
    headers = [h.strip() for h in header[1:]]
    names = [v for _, v in _fishery_name_values()]
    assert headers == names, (
        f"Catchability header {headers} does not match fisheries.name.fshN "
        f"order {names}. Java's Matrix.java:167-174 looks up gear columns "
        "by exact string match."
    )


def test_baltic_discards_headers_match_fishery_names():
    """Same contract as catchability for ``fishery-discards.csv``."""
    if not BALTIC_DISCARDS.exists():
        pytest.skip("Baltic discards CSV not present in this checkout")
    header = BALTIC_DISCARDS.read_text().splitlines()[0].split(",")
    headers = [h.strip() for h in header[1:]]
    names = [v for _, v in _fishery_name_values()]
    assert headers == names, (
        f"Discards header {headers} does not match fisheries.name.fshN order {names}."
    )
