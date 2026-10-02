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
        key_pattern="simulation.restart.enabled",
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
        default=10.0,
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
    "output.carbon.enabled",
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
    "output.fisheries.enabled",
    "output.fisheries.byage.enabled",
    "output.fisheries.bysize.enabled",
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
    "output.biomass.byage.netcdf.enabled",
    "output.abundance.byage.netcdf.enabled",
    "output.biomass.bysize.netcdf.enabled",
    "output.abundance.bysize.netcdf.enabled",
    "output.mortality.netcdf.enabled",
    "output.yield.abundance.netcdf.enabled",
    "output.size.netcdf.enabled",
    "output.ssb.netcdf.enabled",
    "output.carbon.netcdf.enabled",
    "output.diet.composition.netcdf.enabled",
    "output.diet.pressure.netcdf.enabled",
    "output.nschool.enabled",
    "output.age.at.death.enabled",
    "output.bioen.ingest.enabled",
    "output.bioen.maint.enabled",
    "output.bioen.enet.enabled",
    "output.bioen.rho.enabled",
    "output.bioen.sizeinf.enabled",
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

# ── Fish-mediated carbon flux coefficients (issue #134) ───────────────────────
# Opt-in diagnostic after Silvar-Viladomiu, Cavan, Martin et al. (2026), ICES J. Mar.
# Sci. 83(6), doi:10.1093/icesjms/fsag095: faecal pellets = eaten biomass x unassimilated
# fraction x pellet carbon factor; carcasses = non-predation, non-fishing deaths x carcass
# carbon factor. Defaults are that study's teleost values (U = 0.2, sensitivity 0.1-0.3;
# carbon factors 0.06-0.14 of wet weight). Baltic species-specific values are NOT derived
# here; override per species. Respiration, dissolved carbon and depth attenuation are
# excluded, as in the source.

_CARBON_DESC = (
    "Defaults follow Silvar-Viladomiu et al. (2026, ICES JMS, doi:10.1093/icesjms/fsag095) "
    "for teleosts; species-specific values are not derived here."
)

_CARBON_FIELDS: list[OsmoseField] = [
    OsmoseField(
        key_pattern="carbon.unassimilated.fraction.sp{idx}",
        param_type=ParamType.FLOAT,
        default=0.2,
        min_val=0.0,
        max_val=1.0,
        description=(
            "Unassimilated fraction of consumed biomass egested as faecal pellets "
            "(carbon-flux diagnostic; 0.1-0.3 explored in the source). " + _CARBON_DESC
        ),
        category="output",
        unit="fraction",
        indexed=True,
        advanced=True,
    ),
    OsmoseField(
        key_pattern="carbon.pellet.cfactor.sp{idx}",
        param_type=ParamType.FLOAT,
        default=0.10,
        min_val=0.0,
        max_val=1.0,
        description=(
            "Carbon content of egested wet weight for the faecal-pellet carbon flux "
            "(0.07-0.12 in the source). " + _CARBON_DESC
        ),
        category="output",
        unit="t C / t wet weight",
        indexed=True,
        advanced=True,
    ),
    OsmoseField(
        key_pattern="carbon.carcass.cfactor.sp{idx}",
        param_type=ParamType.FLOAT,
        default=0.10,
        min_val=0.0,
        max_val=1.0,
        description=(
            "Carbon content of carcass wet weight for the natural-mortality carbon flux "
            "(0.06-0.14 in the source). " + _CARBON_DESC
        ),
        category="output",
        unit="t C / t wet weight",
        indexed=True,
        advanced=True,
    ),
]

# ── Combined export ───────────────────────────────────────────────────────────

OUTPUT_FIELDS: list[OsmoseField] = _GENERAL_OUTPUT_FIELDS + _FLAG_FIELDS + _CARBON_FIELDS
