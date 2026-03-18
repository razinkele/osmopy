# tests/test_schema.py
import pytest
from osmose.schema.base import OsmoseField, ParamType


def test_osmose_field_creation():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=100.0,
        min_val=1.0,
        max_val=500.0,
        description="L-infinity (asymptotic length)",
        category="growth",
        unit="cm",
        indexed=True,
    )
    assert field.key_pattern == "species.linf.sp{idx}"
    assert field.param_type == ParamType.FLOAT
    assert field.default == 100.0
    assert field.indexed is True
    assert field.required is True  # default
    assert field.advanced is False  # default


def test_osmose_field_resolve_key():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        indexed=True,
    )
    assert field.resolve_key(3) == "species.linf.sp3"


def test_osmose_field_resolve_key_non_indexed():
    field = OsmoseField(
        key_pattern="simulation.time.ndtperyear",
        param_type=ParamType.INT,
        indexed=False,
    )
    assert field.resolve_key() == "simulation.time.ndtperyear"


def test_osmose_field_validate_in_range():
    field = OsmoseField(
        key_pattern="species.k.sp{idx}",
        param_type=ParamType.FLOAT,
        min_val=0.01,
        max_val=2.0,
    )
    assert field.validate_value(0.5) == []
    errors = field.validate_value(5.0)
    assert len(errors) == 1
    assert "max" in errors[0].lower()


def test_osmose_field_validate_enum():
    field = OsmoseField(
        key_pattern="grid.java.classname",
        param_type=ParamType.ENUM,
        choices=["fr.ird.osmose.grid.OriginalGrid", "fr.ird.osmose.grid.NcGrid"],
    )
    assert field.validate_value("fr.ird.osmose.grid.OriginalGrid") == []
    errors = field.validate_value("InvalidGrid")
    assert len(errors) == 1


def test_resolve_key_indexed_without_idx_raises():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        indexed=True,
    )
    with pytest.raises(ValueError, match="Index required"):
        field.resolve_key()


def test_validate_value_below_min():
    field = OsmoseField(
        key_pattern="species.k.sp{idx}",
        param_type=ParamType.FLOAT,
        min_val=0.01,
        max_val=2.0,
    )
    errors = field.validate_value(-0.5)
    assert len(errors) == 1
    assert "min" in errors[0].lower()


def test_param_type_enum():
    assert ParamType.FLOAT.value == "float"
    assert ParamType.MATRIX.value == "matrix"
    assert ParamType.FILE_PATH.value == "file_path"


def test_grid_nlon_matches_schema():
    """Post-v3.3.3 grid keys must match schema fields."""
    from osmose.schema import build_registry

    registry = build_registry()
    field = registry.match_field("grid.nlon")
    assert field is not None, "grid.nlon must match a schema field"
    assert field.param_type.value == "int"
    field2 = registry.match_field("grid.nlat")
    assert field2 is not None, "grid.nlat must match a schema field"
    assert field2.param_type.value == "int"


def test_mortality_additional_matches_schema():
    """Post-v4.2.5 mortality.additional keys must match schema."""
    from osmose.schema import build_registry

    registry = build_registry()
    field = registry.match_field("mortality.additional.rate.sp0")
    assert field is not None, "mortality.additional.rate.sp0 must match schema"
    assert field.indexed is True
    field2 = registry.match_field("mortality.additional.larva.rate.sp0")
    assert field2 is not None, "mortality.additional.larva.rate.sp0 must match schema"
