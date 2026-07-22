"""Map a calibration target to its distinct UQ emulator output-stat key."""

from __future__ import annotations

from osmose.calibration.targets import BiomassTarget

# Distinct per reference_point_type: biomass and ssb do NOT collide here (they
# both map to "_mean" in losses.quantity_key). The emulator needs one GP per
# distinct output, so biomass and ssb must be separately keyed.
_UQ_OUTPUT_SUFFIX = {
    "biomass": "_biomass_mean",
    "ssb": "_ssb_mean",
    "catch": "_yield_mean",
}


def target_to_output_key(target: BiomassTarget) -> str:
    """Return the UQ output-stat key a target is scored against.

    ``biomass`` -> ``"{species}_biomass_mean"``; ``ssb`` -> ``"{species}_ssb_mean"``;
    ``catch`` -> ``"{species}_yield_mean"``. Raises ``ValueError`` on an unknown
    ``reference_point_type``.
    """
    rpt = getattr(target, "reference_point_type", "biomass")
    try:
        suffix = _UQ_OUTPUT_SUFFIX[rpt]
    except KeyError:
        raise ValueError(
            f"unknown reference_point_type {rpt!r}; expected one of {sorted(_UQ_OUTPUT_SUFFIX)}"
        ) from None
    return f"{target.species}{suffix}"
