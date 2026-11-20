"""The share of a resource's size range that a predator can eat (OSMOSE ``computePercent``).

A predator eats a resource in proportion to the part of the resource's size range
``[size_min, size_max]`` that falls in its prey window; the caller has already clipped the window
to the range, giving ``[overlap_min, overlap_max]`` with ``overlap_min < overlap_max``.

- Linear form (the default, Java's ``computePercentLegacy``): the share of the range in size.
- Log form (OSMOSE 4.4.0, ``simulation.resources.computePercent.legacy = false``; Java's
  ``ResourceSpecies.computePercentLog``): the share of the range in log size. A window from
  0.02 to 0.2 cm on a resource from 0.02 to 2 cm is a tenth of the range in size and half of it
  in log size.

The mortality kernels call a Numba-compiled copy of this function, so both paths compute the
same numbers. ``math.log`` lowers to libm in Numba and is libm in Python.
"""

from __future__ import annotations

import math


def resource_percent(
    overlap_min: float, overlap_max: float, size_min: float, size_max: float, log_form: bool
) -> float:
    if log_form:
        return (math.log(overlap_max) - math.log(overlap_min)) / (
            math.log(size_max) - math.log(size_min)
        )
    return (overlap_max - overlap_min) / (size_max - size_min)


LEGACY_KEY = "simulation.resources.computepercent.legacy"  # the reader lowercases keys


def log_form_requested(cfg) -> bool:
    """Whether a config asks for the log form, read as Java reads the key.

    Absent (or empty, or ``null``) keeps the linear form, as Java does when the key is not
    defined (``ResourceSpecies``, v4.4.1 lines 144-149). Otherwise Java takes
    ``Boolean.valueOf``: ``true`` in any case is the linear form, and ANY other value is
    ``false``, so the log form. A value like ``0`` or ``no`` is the log form.
    """
    value = cfg.get(LEGACY_KEY)
    if value is None or value.strip() == "" or value.strip().lower() == "null":
        return False
    return value.strip().lower() != "true"
