"""Pareto-front solution selection helpers for the calibration UI.

These are small pure functions for picking one solution out of an
already-in-memory multi-objective front (the population ``X`` and objective
matrix ``F`` held by the calibration page). This is NOT a run loader — the front
is produced by the live optimizer or read back by the History tab; see
``osmose.calibration.history`` for persistence.
"""

from __future__ import annotations

import numpy as np

from osmose.calibration.surrogate import _non_dominated_indices


def nondominated_indices(F):
    """Indices of the non-dominated rows of objective matrix ``F`` (minimization).

    Thin public wrapper around the surrogate module's helper so the UI does not
    reach into a private symbol.
    """
    return _non_dominated_indices(np.asarray(F, dtype=float))


def select_solution(X, F, param_keys, index):
    """Pick one solution from a front.

    Parameters
    ----------
    X : array-like, shape (n_solutions, n_params)
        Decision vectors (parameter values).
    F : array-like, shape (n_solutions, n_obj)
        Objective values.
    param_keys : sequence[str]
        Full config keys, one per parameter column of ``X``.
    index : int
        Row to select.

    Returns
    -------
    dict
        ``{"index": int, "params": {key: value}, "objectives": [float, ...]}``.

    Raises
    ------
    IndexError
        ``index`` is out of range.
    ValueError
        ``len(param_keys)`` does not match the number of parameter columns.
    """
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float)
    n_solutions, n_params = X.shape
    if not (0 <= index < n_solutions):
        raise IndexError(f"solution index {index} out of range for {n_solutions} solutions")
    if len(param_keys) != n_params:
        raise ValueError(
            f"param_keys has {len(param_keys)} entries but X has {n_params} parameter columns"
        )
    params = {str(k): float(v) for k, v in zip(param_keys, X[index])}
    objectives = [float(v) for v in F[index]]
    return {"index": int(index), "params": params, "objectives": objectives}


def solution_overrides_csv(params):
    """Render a ``{key: value}`` mapping as OSMOSE config ``key ; value`` lines.

    Suitable for download as a parameter-override file that can be merged into a
    base config. Returns an empty string for an empty mapping.
    """
    if not params:
        return ""
    return "\n".join(f"{k} ; {v}" for k, v in params.items()) + "\n"


def apply_solution_overrides(config, params):
    """Merge a picked solution's ``{key: value}`` params into an OSMOSE config dict.

    OSMOSE config values are strings; solution params are floats — each is rendered with
    ``str(value)``, identical to :func:`solution_overrides_csv`, so Apply and Download never
    diverge. Returns ``(new_config, keys_changed)`` where ``keys_changed`` counts params whose
    stringified value differs from the config's current value (a not-yet-present key counts as
    changed). Does not mutate the input config.
    """
    new_config = dict(config)
    changed = 0
    for k, v in params.items():
        sv = str(v)
        if new_config.get(k) != sv:
            changed += 1
        new_config[k] = sv
    return new_config, changed
