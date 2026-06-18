"""Economics OSMOSE parameter definitions."""

from osmose.schema.base import OsmoseField, ParamType

ECONOMICS_FIELDS: list[OsmoseField] = [
    OsmoseField(
        key_pattern="module.bioeconomics.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description="Enable the economics module",
        category="economics",
    ),
    OsmoseField(
        key_pattern="economic.output.stage",
        param_type=ParamType.STRING,
        description="Size classes for economic output",
        category="economics",
        required=False,
    ),
]
