"""ICES validation harness for OSMOSE model outputs.

Compares post-run species biomass against ICES Stock Assessment Graph (SAG)
envelopes loaded from frozen JSON snapshots. Replaces the manual
"compute mean biomass → eyeball ICES table" workflow with a structured
in-range / out-of-range diagnostic.
"""

from osmose.validation.ices import (
    IcesSnapshot,
    SpeciesBiomassComparison,
    compare_outputs_to_ices,
    load_snapshot,
    model_biomass_window_mean,
)

__all__ = [
    "IcesSnapshot",
    "SpeciesBiomassComparison",
    "compare_outputs_to_ices",
    "load_snapshot",
    "model_biomass_window_mean",
]
