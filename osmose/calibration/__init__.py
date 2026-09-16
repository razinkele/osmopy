"""OSMOSE calibration module — optimization, surrogate modelling, and sensitivity analysis."""

from osmose.calibration.history import delete_run, list_runs, load_run, save_run
from osmose.calibration.losses import (
    banded_log_ratio_loss,
    make_banded_objective,
    stability_penalty,
    worst_species_penalty,
)
from osmose.calibration.multiseed import rank_candidates_multiseed, validate_multiseed
from osmose.calibration.objectives import (
    abundance_rmse,
    biomass_rmse,
    catch_at_size_distance,
    diet_distance,
    normalized_rmse,
    size_at_age_rmse,
    weighted_multi_objective,
    yield_rmse,
)
from osmose.calibration.preflight import (
    IssueCategory,
    IssueSeverity,
    ParameterScreening,
    PreflightIssue,
    PreflightResult,
    make_preflight_eval_fn,
    run_preflight,
)
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem, Transform
from osmose.calibration.sensitivity import SensitivityAnalyzer
from osmose.calibration.surrogate import SurrogateCalibrator
from osmose.calibration.targets import BiomassTarget, load_targets

__all__ = [
    "BiomassTarget",
    "FreeParameter",
    "IssueCategory",
    "IssueSeverity",
    "OsmoseCalibrationProblem",
    "ParameterScreening",
    "PreflightIssue",
    "PreflightResult",
    "SensitivityAnalyzer",
    "SurrogateCalibrator",
    "Transform",
    "abundance_rmse",
    "banded_log_ratio_loss",
    "biomass_rmse",
    "catch_at_size_distance",
    "delete_run",
    "diet_distance",
    "list_runs",
    "load_run",
    "load_targets",
    "make_banded_objective",
    "make_preflight_eval_fn",
    "normalized_rmse",
    "rank_candidates_multiseed",
    "run_preflight",
    "save_run",
    "size_at_age_rmse",
    "stability_penalty",
    "validate_multiseed",
    "weighted_multi_objective",
    "worst_species_penalty",
    "yield_rmse",
]
