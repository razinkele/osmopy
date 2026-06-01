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
    OsmoseField(
        key_pattern="predation.functional.response.shape.sp{idx}",
        param_type=ParamType.ENUM,
        default="type1",
        choices=["type1", "type2", "type3"],
        category="predation",
        indexed=True,
        required=False,
        description=(
            "Holling functional-response form for this predator. type1 (default) = "
            "existing linear-with-ration-ceiling behavior (bit-exact). type2 = saturating "
            "(Holling disc; classically destabilizing — paradox of enrichment). type3 = "
            "sigmoid with low-density prey refuge (recommended/validated form)."
        ),
    ),
    OsmoseField(
        key_pattern="predation.functional.response.halfsat.sp{idx}",
        param_type=ParamType.FLOAT,
        default=None,
        min_val=0.1,
        max_val=5.0,
        category="predation",
        indexed=True,
        required=False,
        description=(
            "Dimensionless ration-relative half-saturation K for type2/type3. "
            "Required when shape != type1. Range [0.1, 5.0]. Well-scaled for DE, "
            "not a transferable biological constant."
        ),
    ),
]
