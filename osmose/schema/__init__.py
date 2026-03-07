from osmose.schema.base import OsmoseField, ParamType
from osmose.schema.registry import ParameterRegistry

from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from osmose.schema.grid import GRID_FIELDS
from osmose.schema.predation import PREDATION_FIELDS
from osmose.schema.fishing import FISHING_FIELDS
from osmose.schema.movement import MOVEMENT_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.output import OUTPUT_FIELDS
from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.economics import ECONOMICS_FIELDS

ALL_FIELDS: list[list[OsmoseField]] = [
    SIMULATION_FIELDS,
    SPECIES_FIELDS,
    GRID_FIELDS,
    PREDATION_FIELDS,
    FISHING_FIELDS,
    MOVEMENT_FIELDS,
    LTL_FIELDS,
    OUTPUT_FIELDS,
    BIOENERGETICS_FIELDS,
    ECONOMICS_FIELDS,
]


def build_registry() -> ParameterRegistry:
    """Build a ParameterRegistry with all OSMOSE parameter definitions."""
    reg = ParameterRegistry()
    for fields in ALL_FIELDS:
        for f in fields:
            reg.register(f)
    return reg


__all__ = ["OsmoseField", "ParamType", "ParameterRegistry", "ALL_FIELDS", "build_registry"]
