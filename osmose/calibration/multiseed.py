"""Multi-seed validation and candidate re-ranking for OSMOSE calibration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def validate_multiseed(
    make_objective: Callable[[int], Callable[[np.ndarray], float]],
    x: np.ndarray,
    seeds: Sequence[int] = (42, 123, 7, 999, 2024),
) -> dict:
    """Re-evaluate a candidate across multiple random seeds.

    Args:
        make_objective: Factory(seed) -> objective(x) -> float.
        x: Parameter vector to evaluate.
        seeds: Random seeds to test against.

    Returns:
        Dict with per_seed, mean, std, cv, worst_seed, worst_value.
    """
    per_seed: list[float] = []
    for seed in seeds:
        obj_fn = make_objective(seed)
        per_seed.append(float(obj_fn(x)))

    mean = float(np.mean(per_seed))
    std = float(np.std(per_seed))
    cv = std / mean if mean != 0 else float("inf")
    worst_idx = int(np.argmax(per_seed))

    return {
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
        "cv": cv,
        "worst_seed": seeds[worst_idx],
        "worst_value": per_seed[worst_idx],
    }


def rank_candidates_multiseed(
    make_objective: Callable[[int], Callable[[np.ndarray], float]],
    candidates: np.ndarray,
    seeds: Sequence[int] = (42, 123, 7, 999, 2024),
) -> dict:
    """Re-rank candidates by mean objective across multiple seeds.

    Args:
        candidates: Array of shape (n_candidates, n_params).

    Returns:
        Dict with rankings (sorted candidate indices) and scores (per-candidate dicts).
    """
    scores: list[dict] = []
    for i in range(len(candidates)):
        score = validate_multiseed(make_objective, candidates[i], seeds)
        scores.append(score)

    means = [s["mean"] for s in scores]
    rankings = list(int(i) for i in np.argsort(means))

    return {"rankings": rankings, "scores": scores}
