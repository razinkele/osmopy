"""Fisheries module OSMOSE parameter definitions."""

from osmose.schema.base import OsmoseField, ParamType

FISHING_FIELDS: list[OsmoseField] = [
    # ── General fisheries settings ────────────────────────────────────────
    OsmoseField(
        key_pattern="fisheries.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description="Enable the fisheries module",
        category="fishing",
    ),
    # ── Per-fishery parameters (indexed by fsh{idx}) ──────────────────────
    OsmoseField(
        key_pattern="fisheries.name.fsh{idx}",
        param_type=ParamType.STRING,
        description="Fishery name",
        category="fishing",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="fisheries.rate.base.fsh{idx}",
        param_type=ParamType.FLOAT,
        description="Base fishing rate",
        category="fishing",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="fisheries.period.number.fsh{idx}",
        param_type=ParamType.INT,
        default=4,
        description="Number of fishing periods per year",
        category="fishing",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="fisheries.period.start.fsh{idx}",
        param_type=ParamType.FLOAT,
        default=0.0,
        description="Start time of first fishing period",
        category="fishing",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="fisheries.seasonality.file.fsh{idx}",
        param_type=ParamType.FILE_PATH,
        description="Fishery seasonality CSV file",
        category="fishing",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        # Stored values stay as Java's int-strings ("0"/"1"/"2"/"3") so
        # `int(cfg.get(...))` at engine/config.py:309 reads them unchanged.
        # `choice_labels` adds friendly display strings so the UI dropdown
        # shows "knife-edge / sigmoid / Gaussian / log-normal" instead of
        # bare digits — pure UI affordance, no write-side translation.
        key_pattern="fisheries.selectivity.type.fsh{idx}",
        param_type=ParamType.ENUM,
        choices=["0", "1", "2", "3"],
        choice_labels={
            "0": "knife-edge",
            "1": "sigmoid",
            "2": "Gaussian",
            "3": "log-normal",
        },
        default="0",  # knife-edge — the Java engine default
        description="Selectivity curve type",
        category="fishing",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="fisheries.selectivity.l50.fsh{idx}",
        param_type=ParamType.FLOAT,
        description="Length at 50% selectivity",
        category="fishing",
        unit="cm",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="fisheries.selectivity.l75.fsh{idx}",
        param_type=ParamType.FLOAT,
        description="Length at 75% selectivity",
        category="fishing",
        unit="cm",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="fisheries.catchability.file",
        param_type=ParamType.FILE_PATH,
        description="Catchability matrix CSV file",
        category="fishing",
        required=False,
    ),
    OsmoseField(
        key_pattern="fisheries.discards.file",
        param_type=ParamType.FILE_PATH,
        description="Discards matrix CSV file",
        category="fishing",
        required=False,
    ),
    OsmoseField(
        key_pattern="fisheries.check.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description="Enable fisheries consistency checks",
        category="fishing",
        advanced=True,
    ),
    # ── Marine Protected Areas (indexed by mpa{idx}) ──────────────────────
    OsmoseField(
        key_pattern="mpa.file.mpa{idx}",
        param_type=ParamType.FILE_PATH,
        description="MPA spatial extent CSV",
        category="fishing",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="mpa.start.year.mpa{idx}",
        param_type=ParamType.INT,
        description="MPA start year",
        category="fishing",
        indexed=True,
        required=False,
    ),
    OsmoseField(
        key_pattern="mpa.end.year.mpa{idx}",
        param_type=ParamType.INT,
        description="MPA end year",
        category="fishing",
        indexed=True,
        required=False,
    ),
]
