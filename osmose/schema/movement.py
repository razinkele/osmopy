"""Movement OSMOSE parameter definitions."""

from osmose.schema.base import OsmoseField, ParamType

MOVEMENT_FIELDS: list[OsmoseField] = [
    # ── Per-species movement method ───────────────────────────────────────
    OsmoseField(
        key_pattern="movement.distribution.method.sp{idx}",
        param_type=ParamType.ENUM,
        default="random",  # engine reads `cfg.get(..., "random")` at config.py:669
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
    # ── Distribution maps (engine reads `movement.{property}.map{idx}` —
    # see osmose/engine/movement_maps.py:129-178) ─────────────────────────
    # NB: the schema previously had these inverted as `movement.map{idx}.{property}`
    # which the engine never reads, so UI writes were a silent no-op. Closes
    # C1 from docs/plans/2026-05-05-deep-review-remediation-plan.md.
    OsmoseField(
        key_pattern="movement.species.map{idx}",
        param_type=ParamType.STRING,
        description="Species for this distribution map",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.file.map{idx}",
        param_type=ParamType.FILE_PATH,
        description="CSV distribution map file",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.steps.map{idx}",
        param_type=ParamType.STRING,
        description="Active time steps, comma-separated",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.initialage.map{idx}",
        param_type=ParamType.FLOAT,
        description="Minimum age (years) for this map",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.lastage.map{idx}",
        param_type=ParamType.FLOAT,
        description="Maximum age (years) for this map",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.initialyear.map{idx}",
        param_type=ParamType.INT,
        default=0,
        description="Minimum simulation year for this map (year.min alias)",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.lastyear.map{idx}",
        param_type=ParamType.INT,
        default=999,
        description="Maximum simulation year for this map (year.max alias)",
        category="movement",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="movement.years.map{idx}",
        param_type=ParamType.STRING,
        description="Comma-separated years of activity (alternative to initialyear/lastyear)",
        category="movement",
        indexed=True,
        advanced=True,
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
    # ── Salinity-gated occupancy (prototype spike) ────────────────────────
    OsmoseField(
        key_pattern="movement.salinity.gate.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description=(
            "Master switch for salinity-gated predator occupancy. When false "
            "the gate is inert and movement output is bit-identical."
        ),
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.gate.species.enabled.sp{idx}",
        param_type=ParamType.BOOL,
        default=False,
        description="Per-species enable for salinity occupancy gating (cod only for Baltic).",
        category="movement",
        indexed=True,
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.gate.s.low",
        param_type=ParamType.FLOAT,
        default=3.0,
        description="Salinity (psu) at/below which occupancy weight is 0.",
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.gate.s.high",
        param_type=ParamType.FLOAT,
        default=6.0,
        description="Salinity (psu) at/above which occupancy weight is 1.",
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.field.constant",
        param_type=ParamType.FLOAT,
        description="Constant salinity (psu) forcing (alternative to a NetCDF field).",
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.field.file",
        param_type=ParamType.FILE_PATH,
        description="NetCDF salinity field (alternative to a constant).",
        category="movement",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="movement.salinity.field.varname",
        param_type=ParamType.STRING,
        default="so",
        description="Variable name in the salinity NetCDF (CMEMS 'so').",
        category="movement",
        advanced=True,
    ),
]
