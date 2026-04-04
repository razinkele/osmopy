"""Tests for auto-generated parameter form components."""

from osmose.config.validator import validate_field
from osmose.schema.base import OsmoseField, ParamType
from ui.components.param_form import (
    render_field,
    render_category,
    _guess_step,
    constraint_hint,
    render_species_table,
)


def test_render_float_field():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=100.0,
        min_val=1.0,
        max_val=500.0,
        description="L-infinity",
        unit="cm",
        indexed=True,
    )
    widget = render_field(field, species_idx=0)
    # Should produce a Tag (Shiny UI element)
    assert widget is not None
    html = str(widget)
    assert "L-infinity" in html
    assert "cm" in html


def test_render_int_field():
    field = OsmoseField(
        key_pattern="simulation.nspecies",
        param_type=ParamType.INT,
        default=3,
        min_val=1,
        max_val=50,
        description="Number of species",
    )
    widget = render_field(field)
    assert widget is not None
    html = str(widget)
    assert "Number of species" in html


def test_render_bool_field():
    field = OsmoseField(
        key_pattern="simulation.bioen.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description="Enable bioenergetics",
    )
    widget = render_field(field)
    assert widget is not None


def test_render_enum_field():
    field = OsmoseField(
        key_pattern="grid.java.classname",
        param_type=ParamType.ENUM,
        choices=["OriginalGrid", "NcGrid"],
        default="OriginalGrid",
        description="Grid type",
    )
    widget = render_field(field)
    assert widget is not None
    html = str(widget)
    assert "OriginalGrid" in html


def test_render_file_field():
    field = OsmoseField(
        key_pattern="predation.accessibility.file",
        param_type=ParamType.FILE_PATH,
        description="Accessibility matrix",
    )
    widget = render_field(field)
    assert widget is not None


def test_render_category_filters_advanced():
    """Verify advanced fields are excluded when show_advanced=False and included when True."""
    fields = [
        OsmoseField(
            key_pattern="field.basic",
            param_type=ParamType.FLOAT,
            default=1.0,
            description="Basic field",
            advanced=False,
        ),
        OsmoseField(
            key_pattern="field.secret",
            param_type=ParamType.FLOAT,
            default=2.0,
            description="Advanced field",
            advanced=True,
        ),
    ]
    # Without advanced — only basic field should be rendered
    result = render_category(fields, show_advanced=False)
    html = str(result)
    assert "field_basic" in html
    assert "field_secret" not in html

    # With advanced — both fields should be present
    result_adv = render_category(fields, show_advanced=True)
    html_adv = str(result_adv)
    assert "field_basic" in html_adv
    assert "field_secret" in html_adv


def test_guess_step_small_range():
    field = OsmoseField(key_pattern="x", param_type=ParamType.FLOAT, min_val=0, max_val=1)
    assert _guess_step(field) == 0.01


def test_guess_step_medium_range():
    field = OsmoseField(key_pattern="x", param_type=ParamType.FLOAT, min_val=0, max_val=10)
    assert _guess_step(field) == 0.1


def test_guess_step_large_range():
    field = OsmoseField(key_pattern="x", param_type=ParamType.FLOAT, min_val=0, max_val=500)
    assert _guess_step(field) == 10.0


def test_constraint_hint_float():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Asymptotic length",
        category="species",
        min_val=1.0,
        max_val=200.0,
        unit="cm",
        indexed=True,
    )
    hint = constraint_hint(field)
    assert "1.0" in hint
    assert "200.0" in hint
    assert "cm" in hint


def test_constraint_hint_no_bounds():
    field = OsmoseField(
        key_pattern="simulation.name",
        param_type=ParamType.STRING,
        description="Simulation name",
        category="simulation",
    )
    hint = constraint_hint(field)
    assert hint == ""


def test_constraint_hint_min_only():
    field = OsmoseField(
        key_pattern="species.age.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Minimum age",
        min_val=0.0,
        unit="year",
        indexed=True,
    )
    hint = constraint_hint(field)
    assert "Min: 0.0" in hint
    assert "year" in hint
    assert "Max" not in hint


def test_constraint_hint_max_only():
    field = OsmoseField(
        key_pattern="species.mortality.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Mortality rate",
        max_val=10.0,
        unit="year^-1",
        indexed=True,
    )
    hint = constraint_hint(field)
    assert "Max: 10.0" in hint
    assert "year^-1" in hint
    assert "Min" not in hint


def test_constraint_hint_no_unit():
    field = OsmoseField(
        key_pattern="simulation.nspecies",
        param_type=ParamType.INT,
        description="Number of species",
        min_val=1,
        max_val=50,
    )
    hint = constraint_hint(field)
    assert "Range: 1 " in hint or "Range: 1 —" in hint
    assert "50" in hint


def test_render_float_field_with_hint():
    """Float field with min/max should include hint text in rendered HTML."""
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=100.0,
        min_val=1.0,
        max_val=500.0,
        description="L-infinity",
        unit="cm",
        indexed=True,
    )
    widget = render_field(field, species_idx=0)
    html = str(widget)
    assert "Range:" in html
    assert "1.0" in html
    assert "500.0" in html
    assert "cm" in html


def test_render_int_field_with_hint():
    """Int field with min/max should include hint text in rendered HTML."""
    field = OsmoseField(
        key_pattern="simulation.nspecies",
        param_type=ParamType.INT,
        default=3,
        min_val=1,
        max_val=50,
        description="Number of species",
    )
    widget = render_field(field)
    html = str(widget)
    assert "Range:" in html
    assert "50" in html


def test_validate_field_rejects_out_of_bounds():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=50.0,
        min_val=1.0,
        max_val=200.0,
        description="L-infinity",
        category="growth",
        indexed=True,
    )
    error = validate_field("species.linf.sp0", "500.0", field)
    assert error is not None
    assert "above maximum" in error


def test_validate_field_accepts_valid_value():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=50.0,
        min_val=1.0,
        max_val=200.0,
        description="L-infinity",
        category="growth",
        indexed=True,
    )
    assert validate_field("species.linf.sp0", "100.0", field) is None


def test_render_species_table_zero_species():
    """0 species shows placeholder message."""
    from osmose.schema.species import SPECIES_FIELDS

    result = render_species_table(SPECIES_FIELDS, n_species=0, species_names=[])
    html = str(result)
    assert "Load a configuration" in html


def test_render_species_table_one_species():
    """1 species renders table with header + data column."""
    from osmose.schema.species import SPECIES_FIELDS

    result = render_species_table(
        SPECIES_FIELDS,
        n_species=1,
        species_names=["Anchovy"],
    )
    html = str(result)
    assert "Anchovy" in html
    assert "Growth" in html  # Category header


def test_render_species_table_multiple_species():
    """Multiple species render as columns."""
    from osmose.schema.species import SPECIES_FIELDS

    result = render_species_table(
        SPECIES_FIELDS,
        n_species=3,
        species_names=["Anchovy", "Sardine", "Hake"],
    )
    html = str(result)
    assert "Anchovy" in html
    assert "Sardine" in html
    assert "Hake" in html


def test_render_species_table_hides_advanced():
    """Advanced fields hidden by default."""
    from osmose.schema.species import SPECIES_FIELDS

    result_basic = render_species_table(
        SPECIES_FIELDS,
        n_species=1,
        species_names=["A"],
        show_advanced=False,
    )
    result_adv = render_species_table(
        SPECIES_FIELDS,
        n_species=1,
        species_names=["A"],
        show_advanced=True,
    )
    assert len(str(result_adv)) > len(str(result_basic))


def test_render_species_table_uses_spt_prefix():
    """Input IDs use spt_ prefix to avoid collision."""
    from osmose.schema.species import SPECIES_FIELDS

    result = render_species_table(
        SPECIES_FIELDS,
        n_species=1,
        species_names=["A"],
    )
    html = str(result)
    assert "spt_" in html


def test_render_species_table_with_start_idx():
    """start_idx offsets species indexing for LTL resources."""
    from osmose.schema.ltl import LTL_FIELDS

    indexed = [f for f in LTL_FIELDS if f.indexed]
    result = render_species_table(
        indexed,
        n_species=2,
        species_names=["Phyto", "Zoo"],
        start_idx=8,
    )
    html = str(result)
    assert "spt_" in html
    assert "Phyto" in html


def test_tooltip_markup_in_render_field():
    """render_field should include tooltip (?) icon."""
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=100.0,
        min_val=1.0,
        max_val=500.0,
        description="L-infinity",
        unit="cm",
        indexed=True,
    )
    widget = render_field(field, species_idx=0)
    html = str(widget)
    assert "osm-tooltip-icon" in html
    assert "data-bs-toggle" in html
    assert "data-bs-content" in html
