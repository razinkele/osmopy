"""Movement OSMOSE parameter definitions."""

from osmose.schema.base import OsmoseField, ParamType

MOVEMENT_FIELDS: list[OsmoseField] = [
    # ── Per-species movement method ───────────────────────────────────────
    OsmoseField(
        key_pattern="movement.distribution.method.sp{idx}",
        param_type=ParamType.ENUM,
        default="maps",
        choices=["maps", "random"],
        description="Spatial distribution method",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.distribution.ncell.sp{idx}",
        param_type=ParamType.INT,
        default=None,
        description="Number of cells for random distribution (whole domain if omitted)",
        category="movement",
        indexed=True,
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.randomwalk.range.sp{idx}",
        param_type=ParamType.INT,
        default=1,
        description="Range of random walk in number of cells",
        category="movement",
        indexed=True,
        advanced=True,
    ),
    # ── Distribution maps (indexed by map{idx}) ──────────────────────────
    OsmoseField(
        key_pattern="movement.map{idx}.species",
        param_type=ParamType.STRING,
        description="Species for this distribution map",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.map{idx}.file",
        param_type=ParamType.FILE_PATH,
        description="CSV distribution map file",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.map{idx}.season",
        param_type=ParamType.STRING,
        description="Active time steps, comma-separated",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.map{idx}.initialage",
        param_type=ParamType.FLOAT,
        description="Minimum age (years) for this map",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.map{idx}.lastage",
        param_type=ParamType.FLOAT,
        description="Maximum age (years) for this map",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.map{idx}.year.min",
        param_type=ParamType.INT,
        default=0,
        description="Minimum simulation year for this map",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.map{idx}.year.max",
        param_type=ParamType.INT,
        default=999,
        description="Maximum simulation year for this map",
        category="movement",
        indexed=True,
    ),
    # ── Global movement settings ──────────────────────────────────────────
    OsmoseField(
        key_pattern="movement.randomseed.fixed",
        param_type=ParamType.BOOL,
        default=False,
        description="Fix random seed for movement (reproducibility)",
        category="movement",
        advanced=True,
    ),
]
