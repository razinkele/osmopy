"""Tests for ENUM validation in config validator."""

from __future__ import annotations

import pytest

from osmose.config.validator import validate_config, validate_field
from osmose.schema.base import OsmoseField, ParamType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    from osmose.schema import build_registry

    return build_registry()


def _enum_field(choices: list[str], key: str = "test.param") -> OsmoseField:
    """Helper: build a non-indexed ENUM OsmoseField."""
    return OsmoseField(
        key_pattern=key,
        param_type=ParamType.ENUM,
        choices=choices,
    )


# ---------------------------------------------------------------------------
# validate_field — ENUM handling
# ---------------------------------------------------------------------------


class TestValidateFieldEnum:
    def test_valid_enum_value_returns_none(self):
        """A value that is in choices should return no error."""
        field = _enum_field(["focal", "resource", "background"])
        result = validate_field("test.param", "focal", field)
        assert result is None

    def test_invalid_enum_value_returns_error(self):
        """A value not in choices should return an error string."""
        field = _enum_field(["focal", "resource", "background"])
        result = validate_field("test.param", "unknown_type", field)
        assert result is not None
        assert "unknown_type" in result or "focal" in result

    def test_invalid_enum_error_contains_choices(self):
        """Error message should hint at valid choices."""
        choices = ["alpha", "beta", "gamma"]
        field = _enum_field(choices)
        error = validate_field("x", "delta", field)
        assert error is not None
        # At least one choice should appear in the error message
        assert any(c in error for c in choices)

    def test_enum_with_no_choices_returns_none(self):
        """ENUM field with choices=None should not produce an error (unconstrained)."""
        field = OsmoseField(
            key_pattern="test.unconstrained",
            param_type=ParamType.ENUM,
            choices=None,
        )
        result = validate_field("test.unconstrained", "anything", field)
        assert result is None

    def test_enum_case_sensitive(self):
        """ENUM validation is case-sensitive: 'Focal' != 'focal'."""
        field = _enum_field(["focal", "resource"])
        result = validate_field("test.param", "Focal", field)
        assert result is not None

    def test_enum_empty_string_is_invalid(self):
        """An empty string is not a valid ENUM value when choices are set."""
        field = _enum_field(["focal", "resource"])
        result = validate_field("test.param", "", field)
        assert result is not None

    def test_bool_field_unaffected(self):
        """BOOL field should still work correctly after ENUM branch was added."""
        field = OsmoseField(
            key_pattern="test.bool",
            param_type=ParamType.BOOL,
        )
        assert validate_field("test.bool", "true", field) is None
        assert validate_field("test.bool", "not_a_bool", field) is not None

    def test_float_field_unaffected(self):
        """FLOAT field should still work correctly."""
        field = OsmoseField(
            key_pattern="test.float",
            param_type=ParamType.FLOAT,
            min_val=0.0,
            max_val=100.0,
        )
        assert validate_field("test.float", "50.0", field) is None
        assert validate_field("test.float", "abc", field) is not None


# ---------------------------------------------------------------------------
# validate_config — ENUM handling via registry
# ---------------------------------------------------------------------------


class TestValidateConfigEnum:
    def test_invalid_species_type_produces_error(self, registry):
        """species.type.sp0 with an invalid value should produce a validation error."""
        config = {"species.type.sp0": "invalid_type"}
        errors, _ = validate_config(config, registry)
        assert len(errors) >= 1
        assert any("invalid_type" in e for e in errors)

    def test_valid_species_type_focal(self, registry):
        """species.type.sp0='focal' is a valid ENUM value and should pass."""
        config = {"species.type.sp0": "focal"}
        errors, _ = validate_config(config, registry)
        assert errors == []

    def test_valid_species_type_resource(self, registry):
        """species.type.sp0='resource' is a valid ENUM value and should pass."""
        config = {"species.type.sp0": "resource"}
        errors, _ = validate_config(config, registry)
        assert errors == []

    def test_invalid_enum_error_message_contains_value(self, registry):
        """Error message for invalid ENUM should include the bad value."""
        config = {"species.type.sp0": "totally_wrong"}
        errors, _ = validate_config(config, registry)
        assert any("totally_wrong" in e for e in errors)

    def test_null_enum_value_skipped(self, registry):
        """Null/empty ENUM values should be silently skipped (optional param)."""
        config = {"species.type.sp0": "null"}
        errors, _ = validate_config(config, registry)
        assert errors == []
