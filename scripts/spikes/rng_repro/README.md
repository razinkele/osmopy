# RNG Reproduction Feasibility Spike

This is a throwaway spike exploring the feasibility of reproducing per-cell random draws from OSMOSE's Numba-compiled mortality loop. The spike validates that an @njit oracle mirroring mortality.py's cell-loop semantics reproduces bit-identically with CPython's numpy.random.RandomState (the documented legacy MT19937 algorithm). To run the spike, execute `run_spike.py` from the project root. See the specification at `.superpowers/sdd/spec.md` for full context.
