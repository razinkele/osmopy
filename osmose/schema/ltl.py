"""Lower Trophic Level (LTL) / resource OSMOSE parameter definitions."""

from osmose.schema.base import OsmoseField, ParamType

LTL_FIELDS: list[OsmoseField] = [
    # ── Global LTL settings ───────────────────────────────────────────────
    OsmoseField(
        key_pattern="ltl.java.classname",
        param_type=ParamType.ENUM,
        default="fr.ird.osmose.ltl.LTLFastForcing",
        choices=["fr.ird.osmose.ltl.LTLFastForcing"],
        description="LTL forcing method",
        category="ltl",
    ),
    OsmoseField(
        key_pattern="ltl.netcdf.file",
        param_type=ParamType.FILE_PATH,
        description="LTL biomass NetCDF file",
        category="ltl",
    ),
    OsmoseField(
        key_pattern="ltl.nstep",
        param_type=ParamType.INT,
        default=12,
        description="Number of time steps in LTL data",
        category="ltl",
    ),
    # ── Per-resource parameters (indexed, continuing species numbering) ───
    OsmoseField(
        key_pattern="species.size.min.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Minimum resource size",
        category="ltl",
        unit="cm",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.size.max.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Maximum resource size",
        category="ltl",
        unit="cm",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.tl.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Trophic level",
        category="ltl",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.accessibility2fish.sp{idx}",
        param_type=ParamType.FLOAT,
        min_val=0,
        max_val=1,
        description="Accessibility to fish predators",
        category="ltl",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.file.sp{idx}",
        param_type=ParamType.FILE_PATH,
        description="Biomass NetCDF file for this resource",
        category="ltl",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.biomass.total.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Uniform total biomass",
        category="ltl",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="species.multiplier.sp{idx}",
        param_type=ParamType.FLOAT,
        default=1.0,
        description="Biomass multiplier",
        category="ltl",
        indexed=True,
        advanced=True,
    ),
]
