---
name: SchoolState Optional fields need None-guards in 5 iteration sites
description: Adding `x: T | None = None` to SchoolState (@dataclass(frozen=True)) breaks 5 iteration sites that assume every field is a non-None ndarray
type: feedback
originSessionId: fdd6c641-6530-4066-acf8-1127627cd75e
---
Any new `Optional[NDArray]` field on `SchoolState` (`osmose/engine/state.py`) must be accompanied by `if val is None: continue` guards in every site that iterates `fields(self)` / `fields(state)` and calls `.ndim` / `np.concatenate` / `arr[mask]` on the value.

**Why:** On 2026-04-17, adding `imax_trait: NDArray | None = None` naively broke 59 tests with `AttributeError: 'NoneType' has no attribute 'ndim'`. The fix required guards in five places:
  1. `state.py::__post_init__` (validator `val.ndim`)
  2. `state.py::append` (`np.concatenate([a, b])`)
  3. `state.py::compact` (`arr[alive]`)
  4. `simulate.py::_strip_background` (`arr.ndim`)
  5. `processes/reproduction.py` batch-merge (`np.concatenate` on parts)

**How to apply:** Before adding an Optional ndarray field, grep for `fields(self)` and `fields(state)` across the engine. Apply the skip-None guard at every site. For `np.concatenate` paths, decide whether partial-None should be an error (reproduction.py raises `ValueError` today) or a skip. See commit `59ccca1` on master for the reference pattern.
