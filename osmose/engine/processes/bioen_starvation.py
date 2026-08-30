"""Energy-deficit starvation with gonad buffer.

Matches Java BioenStarvationMortality: internally loops over
n_subdt, flushing gonad before computing death toll at each sub-step.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def bioen_starvation(
    e_net: NDArray[np.float64],
    gonad_weight: NDArray[np.float64],
    weight: NDArray[np.float64],
    eta: float,
    n_subdt: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute starvation deaths and gonad depletion.

    Per sub-timestep: if E_net < 0, deficit is absorbed by gonad first.
    If gonad insufficient, gonad is flushed to 0 BEFORE computing death toll.

    Returns: (n_dead, new_gonad_weight)
    """
    n_dead = np.zeros_like(e_net)
    new_gonad = gonad_weight.copy()

    for _ in range(n_subdt):
        e_sub = e_net / n_subdt
        deficit = np.maximum(-e_sub, 0.0)

        sufficient = new_gonad >= eta * deficit
        # Sufficient: gonad absorbs
        new_gonad = np.where(sufficient, new_gonad - eta * deficit, new_gonad)

        # Insufficient: flush gonad, compute deaths
        insufficient = (~sufficient) & (deficit > 0)
        gonad_buffered = np.where(insufficient, new_gonad / eta, 0.0)
        remaining = np.where(insufficient, deficit - gonad_buffered, 0.0)
        safe_weight = np.where(weight > 0, weight, 1.0)
        n_dead += np.where(insufficient, remaining / safe_weight, 0.0)
        new_gonad = np.where(insufficient, 0.0, new_gonad)

    return n_dead, new_gonad


def bioen_starvation_substep(
    e_net: float,
    gonad_weight: float,
    weight: float,
    eta: float,
    n_subdt: int,
) -> tuple[float, float, float]:
    """Java ``BioenStarvationMortality.computeStarvation`` for ONE school, ONE sub-step.

    Replicates Java's ordering quirks on purpose (spec §0): the deficit is the SCHOOL's
    ``E_net`` (tonnes per school) divided by ``subdt`` while the gonad is per FISH; and in
    the insufficient branch the gonad is flushed BEFORE the repayment credit and the death
    toll are read from it, so the credit is zero and the toll is the whole deficit
    (``BioenStarvationMortality.java:186-193``).

    Called once per sub-step from the interleaved mortality loop, on the PREVIOUS step's
    ``e_net`` (Java step order: mortality -> EnergyBudget -> reproduction). Each call sees
    the ``e_net`` left by the previous sub-step, so the toll is not simply ``|E_net_0|``.

    Returns:
        ``(n_dead, new_gonad_weight, new_e_net)``.
    """
    if e_net >= 0.0:
        return 0.0, gonad_weight, e_net
    deficit = abs(e_net) / n_subdt
    if gonad_weight >= eta * deficit:
        # Enough gonadic energy: pay maintenance from the gonad and repay E_net.
        return 0.0, gonad_weight - eta * deficit, e_net + deficit
    if weight <= 0.0:
        # Java would divide by zero here; flush the gonad as it does and kill nobody.
        return 0.0, 0.0, e_net
    return deficit / weight, 0.0, e_net
