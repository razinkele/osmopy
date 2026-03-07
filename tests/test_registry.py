import pytest

from osmose.schema.base import OsmoseField, ParamType
from osmose.schema.registry import ParameterRegistry


@pytest.fixture
def registry():
    """Pre-populated registry for reuse across tests."""
    reg = ParameterRegistry()
    reg.register(
        OsmoseField(
            key_pattern="species.linf.sp{idx}",
            param_type=ParamType.FLOAT,
            category="growth",
            indexed=True,
        )
    )
    reg.register(
        OsmoseField(
            key_pattern="species.k.sp{idx}",
            param_type=ParamType.FLOAT,
            category="growth",
            indexed=True,
        )
    )
    reg.register(
        OsmoseField(
            key_pattern="simulation.time.ndtperyear",
            param_type=ParamType.INT,
            category="simulation",
        )
    )
    return reg


def test_match_field_uses_cache(registry):
    """match_field should be fast on repeated lookups (cached)."""
    field1 = registry.match_field("simulation.time.ndtperyear")
    field2 = registry.match_field("simulation.time.ndtperyear")
    assert field1 is field2


def test_categories_preserves_order(registry):
    """categories() should return unique categories in insertion order."""
    cats = registry.categories()
    assert len(cats) == len(set(cats))
    assert len(cats) > 0


def test_registry_register_and_retrieve():
    reg = ParameterRegistry()
    f = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        category="growth",
        indexed=True,
    )
    reg.register(f)
    assert len(reg.all_fields()) == 1
    assert reg.all_fields()[0] is f


def test_registry_fields_by_category():
    reg = ParameterRegistry()
    f1 = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        category="growth",
        indexed=True,
    )
    f2 = OsmoseField(
        key_pattern="species.k.sp{idx}", param_type=ParamType.FLOAT, category="growth", indexed=True
    )
    f3 = OsmoseField(
        key_pattern="simulation.time.ndtperyear", param_type=ParamType.INT, category="simulation"
    )
    reg.register(f1)
    reg.register(f2)
    reg.register(f3)
    growth = reg.fields_by_category("growth")
    assert len(growth) == 2
    sim = reg.fields_by_category("simulation")
    assert len(sim) == 1


def test_registry_get_field_by_pattern():
    reg = ParameterRegistry()
    f = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        category="growth",
        indexed=True,
    )
    reg.register(f)
    result = reg.get_field("species.linf.sp{idx}")
    assert result is f


def test_registry_get_field_not_found():
    reg = ParameterRegistry()
    assert reg.get_field("nonexistent") is None


def test_registry_categories():
    reg = ParameterRegistry()
    reg.register(OsmoseField(key_pattern="a", param_type=ParamType.FLOAT, category="growth"))
    reg.register(OsmoseField(key_pattern="b", param_type=ParamType.FLOAT, category="simulation"))
    reg.register(OsmoseField(key_pattern="c", param_type=ParamType.FLOAT, category="growth"))
    cats = reg.categories()
    assert set(cats) == {"growth", "simulation"}


def test_registry_validate_config():
    reg = ParameterRegistry()
    reg.register(
        OsmoseField(
            key_pattern="species.k.sp{idx}",
            param_type=ParamType.FLOAT,
            min_val=0.01,
            max_val=2.0,
            indexed=True,
            category="growth",
        )
    )
    errors = reg.validate({"species.k.sp0": 0.5})
    assert errors == []
    errors = reg.validate({"species.k.sp0": 5.0})
    assert len(errors) == 1


def test_registry_match_field():
    reg = ParameterRegistry()
    f = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        category="growth",
        indexed=True,
    )
    reg.register(f)
    assert reg.match_field("species.linf.sp0") is f
    assert reg.match_field("species.linf.sp12") is f
    assert reg.match_field("species.linf.spX") is None
    assert reg.match_field("nonexistent") is None
