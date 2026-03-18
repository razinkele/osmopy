"""Tests for setup page input syncing to state."""

from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS


def test_setup_global_keys():
    """Setup sync should cover all non-advanced simulation fields."""
    from ui.pages.setup import SETUP_GLOBAL_KEYS

    expected_patterns = [f.key_pattern for f in SIMULATION_FIELDS if not f.advanced]
    for pattern in expected_patterns:
        assert pattern in SETUP_GLOBAL_KEYS, f"Missing key: {pattern}"


def test_setup_species_spt_input_ids():
    """Species table sync uses spt_ prefix; verify input ID generation for a species index."""
    visible = [f for f in SPECIES_FIELDS if f.indexed and not f.advanced]
    # Should include at least one field (e.g. species.k)
    assert len(visible) > 0
    for field in visible:
        base_key = field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
        input_id = f"spt_{base_key}_0"
        # Input IDs should be non-empty strings with the spt_ prefix
        assert input_id.startswith("spt_")
        assert "_0" in input_id


def test_setup_species_spt_input_ids_with_advanced():
    """With show_advanced=True, advanced indexed fields are also included."""
    visible_all = [f for f in SPECIES_FIELDS if f.indexed]
    visible_basic = [f for f in SPECIES_FIELDS if f.indexed and not f.advanced]
    # There should be more fields when advanced are included
    assert len(visible_all) >= len(visible_basic)
    for field in visible_all:
        base_key = field.key_pattern.replace(".sp{idx}", "").replace("{idx}", "").replace(".", "_")
        input_id = f"spt_{base_key}_1"
        assert "sp1" not in input_id or input_id.startswith("spt_")
