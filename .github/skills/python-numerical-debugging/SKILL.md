---
name: python-numerical-debugging
description: >-
  Use this skill when the user reports NaN, Inf, incorrect numerical results, array shape mismatches,
  floating-point precision issues, or silent numerical corruption in Python code that uses NumPy, SciPy,
  Numba, pandas, or xarray. Trigger for prompts like "why am I getting NaN", "results are wrong",
  "numerical instability", "values don't match", "array shape mismatch", or "floating point error".
  Also trigger when comparing two numerical implementations for parity (e.g. Python vs Java engine output).
  Do not trigger for general Python exceptions, import errors, or non-numerical logic bugs.
license: MIT
compatibility: 'Cross-platform. Requires Python 3.10+ with NumPy installed.'
metadata:
  version: "1.0"
  categories:
    - Scientific computing
    - Numerical simulation
    - Data validation
argument-hint: 'Optional: describe the symptom, e.g. "NaN in biomass after year 3", "predation matrix has negative values"'
---

# Python Numerical Debugging

Systematically diagnoses numerical bugs in Python scientific code. Produces a root-cause analysis
with a targeted fix, not a shotgun of suggestions.

## Output Contract (Required)

Before finishing, all of the following must be true:

1. The **symptom** is reproduced or precisely located (file, function, line).
2. The **root cause** is identified with evidence (not guessed).
3. A **minimal fix** is proposed and validated.
4. Any **upstream contamination** (earlier code that fed bad values) is traced.
5. A **guard** is suggested to prevent recurrence (assertion, test, or runtime check).

## Workflow

Copy and track this checklist:

```
- [ ] Phase 1: Reproduce and characterize the symptom
- [ ] Phase 2: Instrument and isolate the source
- [ ] Phase 3: Identify root cause with evidence
- [ ] Phase 4: Fix, validate, and add guards
```

### Phase 1: Reproduce and Characterize

1. **Identify the symptom class** from this table:

   | Symptom | Likely Cause Family |
   |---------|-------------------|
   | NaN in output | 0/0, log(≤0), sqrt(<0), NaN input propagation |
   | Inf / -Inf | Division by near-zero, exp() overflow, unbounded accumulation |
   | Wrong magnitude | Unit mismatch, missing scaling factor, integer vs float division |
   | Slight imprecision | Float64 vs float32, accumulation order, catastrophic cancellation |
   | Shape mismatch | Broadcasting error, transposed axis, off-by-one in slicing |
   | Negative values where impossible | Unsigned subtraction, predation exceeding biomass, sign flip |
   | Silently zero | Uninitialized array, dtype truncation (int vs float), masked values |
   | Results differ between runs | Race condition, unseeded RNG, dict iteration order |
   | Python vs reference mismatch | Algorithm difference, parameter indexing, boundary condition |

2. **Get a minimal reproducer.** Ask the user for:
   - Input data or config that triggers the bug
   - Expected vs actual output
   - Whether the issue is deterministic

3. **Snapshot array state** at the symptom site:
   ```python
   import numpy as np
   def array_health(name, arr):
       """Print diagnostic summary for a numerical array."""
       a = np.asarray(arr, dtype=float)
       print(f"--- {name} ---")
       print(f"  shape={a.shape}  dtype={a.dtype}")
       print(f"  min={np.nanmin(a):.6g}  max={np.nanmax(a):.6g}  mean={np.nanmean(a):.6g}")
       print(f"  NaN={np.isnan(a).sum()}  Inf={np.isinf(a).sum()}  Zero={np.count_nonzero(a == 0)}")
       print(f"  Negative={np.count_nonzero(a < 0)}  Finite={np.isfinite(a).all()}")
       if a.ndim >= 2:
           print(f"  Row sums range: [{np.nanmin(a.sum(axis=1)):.6g}, {np.nanmax(a.sum(axis=1)):.6g}]")
   ```

### Phase 2: Instrument and Isolate

1. **Enable NumPy error trapping** to catch the exact operation:
   ```python
   np.seterr(all='raise')  # Raises FloatingPointError on invalid ops
   ```
   Or for warnings with tracebacks:
   ```python
   import warnings
   warnings.filterwarnings('error', category=RuntimeWarning)
   ```

2. **Binary search the timeline.** For simulations with time steps:
   - Check array health at step 0 (initial conditions).
   - Check at the midpoint.
   - Narrow to the first step where corruption appears.

3. **Binary search the call chain.** Within the corrupted step:
   - Insert `array_health()` before and after each process call.
   - The bug is between the last clean checkpoint and the first corrupt one.

4. **Check common traps** at the isolated site:

   | Trap | How to check |
   |------|-------------|
   | Division by zero | `assert np.all(denominator != 0)` or check `denominator.min()` |
   | Log of non-positive | `assert np.all(x > 0)` before `np.log(x)` |
   | Exp overflow | Check `x.max() < 709` before `np.exp(x)` (float64 limit) |
   | Integer division | Verify dtypes: `arr.dtype` should be `float64`, not `int64` |
   | Broadcasting mismatch | Print shapes of all operands: `a.shape, b.shape` |
   | In-place mutation aliasing | Check `a is b` or `np.shares_memory(a, b)` |
   | Numba type coercion | Check `numba.typeof(x)` matches expectations |
   | Stale array reference | Verify array was freshly allocated, not reused from prior step |
   | Off-by-one indexing | Check `arr[idx]` bounds: `0 <= idx < arr.shape[axis]` |
   | Uninitialized `np.empty` | Replace with `np.zeros` or `np.full(shape, np.nan)` to surface gaps |

### Phase 3: Root Cause Analysis

1. **State the root cause** in one sentence: what operation, on what input, produces the bad value.
2. **Show the evidence**: the specific array values, shapes, or dtypes that prove it.
3. **Trace upstream**: if the bad input came from an earlier computation, trace back until you reach
   either user input, config, or a correct intermediate value.
4. **Check for systemic contamination**: once NaN/Inf enters an array, it propagates everywhere.
   The *first* occurrence is the bug; downstream NaNs are symptoms.

### Phase 4: Fix, Validate, and Guard

1. **Apply the minimal fix.** Prefer:
   - Correcting the math over clamping/masking
   - Adding epsilon only when mathematically justified (document why)
   - Fixing dtypes at the source, not casting at every use site

2. **Validate the fix:**
   - Re-run the reproducer — symptom must be gone
   - Check `array_health()` at the previously-corrupt checkpoint
   - Run existing tests if available

3. **Add a guard** to prevent recurrence. Choose the lightest option that catches the bug:

   | Guard type | When to use |
   |-----------|-------------|
   | `assert` statement | Development-time check, zero production cost |
   | `np.errstate(invalid='raise')` context | Wrap suspect functions during testing |
   | Explicit validation function | Reusable check for domain constraints (e.g., biomass ≥ 0) |
   | Unit test with known-bad input | Regression test for the specific trigger |
   | Runtime bounds check | Production code where silent corruption is unacceptable |

4. **Report** the fix with:
   - Root cause (one sentence)
   - The fix (diff or description)
   - The guard added
   - Any remaining `[TODO]` items

---

## Parity Debugging (Python vs Reference Implementation)

When comparing two implementations that should produce identical results:

1. **Align inputs exactly.** Same config values, same initial conditions, same RNG seeds.
2. **Compare at each process step**, not just final output.
3. **Use tolerance-aware comparison:**
   ```python
   np.testing.assert_allclose(python_result, reference_result, rtol=1e-10, atol=1e-12)
   ```
4. **Common parity divergence causes:**
   - Different iteration order (row-major vs column-major)
   - Different floating-point accumulation order (sum reduction)
   - Off-by-one in species/age indexing (0-based vs 1-based)
   - Different interpolation or rounding at boundaries
   - Stochastic processes with different RNG algorithms

---

## Quick Reference: NumPy/SciPy Numerical Limits

| Constant | Value | Relevance |
|----------|-------|-----------|
| `np.finfo(np.float64).max` | ~1.8e308 | Overflow threshold |
| `np.finfo(np.float64).tiny` | ~2.2e-308 | Underflow threshold |
| `np.finfo(np.float64).eps` | ~2.2e-16 | Machine epsilon |
| `np.finfo(np.float32).eps` | ~1.2e-7 | Float32 precision limit |
| Max safe `np.exp(x)` | x < 709.78 | Beyond this → Inf |
| `np.log(0)` | -Inf | Not NaN — but propagates oddly |
| `0.0 / 0.0` | NaN | The classic silent killer |
| `1.0 / 0.0` | Inf | Often from empty populations |

---

## Anti-Patterns

| ❌ Don't | ✅ Do instead |
|---------|--------------|
| Add `np.nan_to_num()` everywhere | Find and fix the source of NaN |
| Clamp to `[0, ∞)` without understanding why values go negative | Fix the math that produces negatives |
| Use `try/except` around array ops to silence errors | Use `np.seterr(all='raise')` to find them |
| Compare floats with `==` | Use `np.isclose()` or `np.testing.assert_allclose()` |
| Assume `dtype=float` without checking | Always verify with `arr.dtype` |
| Debug by reading code alone | Instrument with `array_health()` and inspect actual values |
| Blame Numba/NumPy for "wrong" results | Check your indexing and dtypes first |
