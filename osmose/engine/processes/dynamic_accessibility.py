"""Density-dependent dynamic predation accessibility.

Scales the static accessibility matrix based on prey biomass relative to
a reference level.  When prey biomass drops below reference, accessibility
decreases (prey become harder to find).  When prey biomass is at or above
reference, static accessibility is used unchanged.

Updated once per year at the start of each simulation year.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_prey_density_scale(
    prey_biomass: NDArray[np.float64],
    reference_biomass: NDArray[np.float64],
    exponent: float = 1.0,
    floor: float = 0.05,
) -> NDArray[np.float64]:
    """Compute per-species scaling factors from biomass ratios.

    Parameters
    ----------
    prey_biomass : (n_species,)
        Current total biomass per focal species.
    reference_biomass : (n_species,)
        Reference biomass per species (carrying capacity or initial).
        Must be strictly positive.
    exponent : float
        Controls sensitivity.  1.0 = linear, <1 = saturating, >1 = sharp
        threshold.
    floor : float
        Minimum scale factor.  Prevents prey from becoming completely
        invisible even at very low abundance.

    Returns
    -------
    scale : (n_species,)
        Scale factors in [floor, 1.0] for each species.
    """
    safe_ref = np.maximum(reference_biomass, 1e-10)
    ratio = prey_biomass / safe_ref
    scale = np.clip(np.power(ratio, exponent), floor, 1.0)
    return scale


def apply_prey_scale_to_matrix(
    matrix: NDArray[np.float64],
    scale: NDArray[np.float64],
    n_species: int,
    is_stage_indexed: bool,
    stage_to_species: NDArray[np.int32] | None = None,
) -> NDArray[np.float64]:
    """Apply per-prey-species density scaling to an accessibility matrix.

    Parameters
    ----------
    matrix : 2D array
        The static accessibility matrix.
        - Non-stage: shape (n_total, n_total), indexed [predator, prey].
        - Stage-indexed: shape (n_prey_stages, n_pred_stages), indexed
          [prey_stage, pred_stage].
    scale : (n_species,)
        Per-species scale factors from ``compute_prey_density_scale``.
    n_species : int
        Number of focal species.
    is_stage_indexed : bool
        True when using stage-indexed accessibility.
    stage_to_species : (n_prey_stages,) or None
        Maps each prey stage row to its species index.  Required when
        ``is_stage_indexed`` is True.

    Returns
    -------
    scaled : 2D array
        Copy of matrix with prey-side scaling applied.
    """
    scaled = matrix.copy()

    if is_stage_indexed:
        if stage_to_species is None:
            return scaled
        # Stage matrix: rows are prey stages → scale each row by its species
        for row_idx in range(scaled.shape[0]):
            sp = stage_to_species[row_idx]
            if 0 <= sp < len(scale):
                scaled[row_idx, :] *= scale[sp]
    else:
        # Species matrix: columns are prey → scale each column by species
        for sp in range(min(n_species, scaled.shape[1])):
            scaled[:, sp] *= scale[sp]

    return scaled
