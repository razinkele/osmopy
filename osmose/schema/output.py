"""Output OSMOSE parameter definitions."""

from osmose.schema.base import OsmoseField, ParamType

# ── General output settings ───────────────────────────────────────────────────

_GENERAL_OUTPUT_FIELDS: list[OsmoseField] = [
    OsmoseField(
        key_pattern="output.dir.path",
        param_type=ParamType.STRING,
        default="output",
        description="Output directory path",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.file.prefix",
        param_type=ParamType.STRING,
        default="osm",
        description="Prefix for output file names",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.start.year",
        param_type=ParamType.INT,
        default=0,
        description="First year of output recording",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.recordfrequency.ndt",
        param_type=ParamType.INT,
        default=12,
        description="Recording frequency in number of time steps",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.csv.separator",
        param_type=ParamType.STRING,
        default=",",
        description="CSV column separator character",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.flush.enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Flush output files after each write",
        category="output",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="output.restart.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description="Enable restart file output",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.cutoff.enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable output cutoff filtering",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.cutoff.age.sp{idx}",
        param_type=ParamType.FLOAT,
        default=0.08,
        description="Minimum age for output inclusion",
        category="output",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="output.distrib.bysize.min",
        param_type=ParamType.FLOAT,
        default=0,
        description="Minimum size for size distribution output",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.distrib.bysize.max",
        param_type=ParamType.FLOAT,
        default=205,
        description="Maximum size for size distribution output",
        category="output",
    ),
    OsmoseField(
        key_pattern="output.distrib.bysize.incr",
        param_type=ParamType.FLOAT,
        default=10,
        description="Size increment for size distribution output",
        category="output",
    ),
]

# ── Output enable flags (generated programmatically) ──────────────────────────

_OUTPUT_ENABLE_FLAGS = [
    "output.biomass.enabled",
    "output.abundance.enabled",
    "output.abundance.age1.enabled",
    "output.ssb.enabled",
    "output.biomass.bysize.enabled",
    "output.biomass.byage.enabled",
    "output.biomass.byweight.enabled",
    "output.biomass.bytl.enabled",
    "output.abundance.bysize.enabled",
    "output.abundance.byage.enabled",
    "output.abundance.byweight.enabled",
    "output.abundance.bytl.enabled",
    "output.size.enabled",
    "output.weight.enabled",
    "output.size.catch.enabled",
    "output.meansize.byage.enabled",
    "output.meanweight.byage.enabled",
    "output.tl.enabled",
    "output.tl.catch.enabled",
    "output.meantl.bysize.enabled",
    "output.meantl.byage.enabled",
    "output.diet.composition.enabled",
    "output.diet.composition.byage.enabled",
    "output.diet.composition.bysize.enabled",
    "output.diet.pressure.enabled",
    "output.diet.pressure.byage.enabled",
    "output.diet.pressure.bysize.enabled",
    "output.diet.success.enabled",
    "output.mortality.enabled",
    "output.mortality.perspecies.byage.enabled",
    "output.mortality.perspecies.bysize.enabled",
    "output.mortality.additional.bysize.enabled",
    "output.mortality.additional.byage.enabled",
    "output.yield.biomass.enabled",
    "output.yield.abundance.enabled",
    "output.yield.biomass.bysize.enabled",
    "output.yield.biomass.byage.enabled",
    "output.yield.abundance.bysize.enabled",
    "output.yield.abundance.byage.enabled",
    "output.fishery.enabled",
    "output.fishery.byage.enabled",
    "output.fishery.bysize.enabled",
    "output.spatial.enabled",
    "output.spatial.biomass.enabled",
    "output.spatial.abundance.enabled",
    "output.spatial.size.enabled",
    "output.spatial.ltl.enabled",
    "output.spatial.yield.biomass.enabled",
    "output.spatial.yield.abundance.enabled",
    "output.spatial.egg.enabled",
    "output.biomass.netcdf.enabled",
    "output.abundance.netcdf.enabled",
    "output.yield.biomass.netcdf.enabled",
    "output.yield.abundance.netcdf.enabled",
    "output.diet.composition.netcdf.enabled",
    "output.diet.pressure.netcdf.enabled",
    "output.nschool.enabled",
    "output.age.at.death.enabled",
    "output.bioen.ingest.enabled",
    "output.bioen.maint.enabled",
    "output.bioen.enet.enabled",
    "output.bioen.rho.enabled",
    "output.bioen.sizeInf.enabled",
]


def _make_flag_description(flag: str) -> str:
    """Derive a human-readable description from an output enable flag name.

    Example: "output.biomass.bysize.enabled" -> "Enable biomass by-size output"
    """
    # Strip "output." prefix and ".enabled" suffix
    middle = flag.removeprefix("output.").removesuffix(".enabled")
    # Replace dots with spaces, clean up common patterns
    words = (
        middle.replace(".", " ")
        .replace("bysize", "by-size")
        .replace("byage", "by-age")
        .replace("byweight", "by-weight")
        .replace("bytl", "by-trophic-level")
        .replace("meantl", "mean trophic level")
        .replace("meansize", "mean size")
        .replace("meanweight", "mean weight")
        .replace("perspecies", "per-species")
        .replace("netcdf", "NetCDF")
    )
    return f"Enable {words} output"


_FLAG_FIELDS: list[OsmoseField] = [
    OsmoseField(
        key_pattern=flag,
        param_type=ParamType.BOOL,
        default=False,
        description=_make_flag_description(flag),
        category="output",
        advanced=True,
    )
    for flag in _OUTPUT_ENABLE_FLAGS
]

# ── Combined export ───────────────────────────────────────────────────────────

OUTPUT_FIELDS: list[OsmoseField] = _GENERAL_OUTPUT_FIELDS + _FLAG_FIELDS
