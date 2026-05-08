"""Global predation OSMOSE parameter definitions."""

from osmose.schema.base import OsmoseField, ParamType

PREDATION_FIELDS: list[OsmoseField] = [
    OsmoseField(
        key_pattern="predation.accessibility.file",
        param_type=ParamType.FILE_PATH,
        description="Accessibility matrix CSV",
        category="predation",
        # required=False because the engine falls back to per-species
        # accessibility2fish when this is absent (resources.py:140).
        required=False,
    ),
    OsmoseField(
        key_pattern="predation.accessibility.stage.structure",
        param_type=ParamType.ENUM,
        default="age",
        choices=["age", "size"],
        description="Stage structure used for accessibility matrix",
        category="predation",
    ),
    OsmoseField(
        key_pattern="predation.accessibility.stage.threshold.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Stage threshold for accessibility",
        category="predation",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="predation.predprey.stage.threshold.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Stage threshold for predator-prey interactions",
        category="predation",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="predation.predprey.stage.structure",
        param_type=ParamType.ENUM,
        default="size",
        choices=["age", "size"],
        description="Stage structure used for predator-prey size ratios",
        category="predation",
    ),
]
