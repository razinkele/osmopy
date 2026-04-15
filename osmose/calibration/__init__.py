"""OSMOSE calibration module — optimization, surrogate modelling, and sensitivity analysis."""

from osmose.calibration.objectives import (
    biomass_rmse,
    abundance_rmse,
    diet_distance,
    normalized_rmse,
    yield_rmse,
    catch_at_size_distance,
    size_at_age_rmse,
    weighted_multi_objective,
)
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem, Transform
from osmose.calibration.surrogate import SurrogateCalibrator
from osmose.calibration.sensitivity import SensitivityAnalyzer
from osmose.calibration.targets import BiomassTarget, load_targets
from osmose.calibration.losses import (
    banded_log_ratio_loss,
    stability_penalty,
    worst_species_penalty,
    make_banded_objective,
)
from osmose.calibration.multiseed import validate_multiseed, rank_candidates_multiseed

__all__ = [
    "biomass_rmse",
    "abundance_rmse",
    "diet_distance",
    "normalized_rmse",
    "yield_rmse",
    "catch_at_size_distance",
    "size_at_age_rmse",
    "weighted_multi_objective",
    "FreeParameter",
    "Transform",
    "OsmoseCalibrationProblem",
    "SurrogateCalibrator",
    "SensitivityAnalyzer",
    "BiomassTarget",
    "load_targets",
    "banded_log_ratio_loss",
    "stability_penalty",
    "worst_species_penalty",
    "make_banded_objective",
    "validate_multiseed",
    "rank_candidates_multiseed",
]
