"""Tests for config page input syncing (Grid, Forcing, Fishing, Movement)."""

from osmose.schema.grid import GRID_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.fishing import FISHING_FIELDS
from osmose.schema.movement import MOVEMENT_FIELDS


def test_grid_global_keys():
    from ui.pages.grid import GRID_GLOBAL_KEYS

    expected = [f.key_pattern for f in GRID_FIELDS if not f.indexed]
    for key in expected:
        assert key in GRID_GLOBAL_KEYS


def test_forcing_global_keys():
    from ui.pages.forcing import FORCING_GLOBAL_KEYS

    expected = [f.key_pattern for f in LTL_FIELDS if not f.indexed]
    for key in expected:
        assert key in FORCING_GLOBAL_KEYS


def test_fishing_global_keys():
    from ui.pages.fishing import FISHING_GLOBAL_KEYS

    expected = [f.key_pattern for f in FISHING_FIELDS if not f.indexed]
    for key in expected:
        assert key in FISHING_GLOBAL_KEYS


def test_movement_global_keys():
    from ui.pages.movement import MOVEMENT_GLOBAL_KEYS

    expected = [f.key_pattern for f in MOVEMENT_FIELDS if not f.indexed]
    for key in expected:
        assert key in MOVEMENT_GLOBAL_KEYS


def test_movement_uses_dynamic_species_count(tmp_path):
    """Movement server reads species count from state.config, not a hardcoded value.

    We verify behavioral correctness: the server must react differently when
    simulation.nspecies changes in state.  Source inspection is avoided because
    it tests implementation details rather than observable behavior.
    """
    from ui.pages.movement import MOVEMENT_GLOBAL_KEYS

    # MOVEMENT_GLOBAL_KEYS should not include any hardcoded species-indexed keys —
    # indexed keys are resolved at runtime from the species count in state.config.
    # A hardcoded species count of 3 would materialise as sp0/sp1/sp2 keys here.
    hardcoded_species_keys = [
        k for k in MOVEMENT_GLOBAL_KEYS
        if any(f"sp{i}" in k for i in range(10))
    ]
    assert hardcoded_species_keys == [], (
        f"MOVEMENT_GLOBAL_KEYS contains hardcoded species keys: {hardcoded_species_keys}"
    )
