---
name: profile-engine
description: Profile the Python engine to identify performance bottlenecks in NumPy/Numba code
disable-model-invocation: true
---

Profile the OSMOSE Python engine to identify performance bottlenecks and Numba JIT compilation issues.

## Arguments

- `config` (optional): "bob" (Bay of Biscay) or "eec" (EEC) — default: "bob"
- `years` (optional): simulation years (default: 5)
- `focus` (optional): specific module to deep-profile (e.g., "predation", "movement")

## Steps

1. **Run cProfile** to get a high-level function-time breakdown:
   ```
   .venv/bin/python -m cProfile -s cumulative -m osmose.engine.cli --config {config} --years {years} 2>&1 | head -40
   ```
   If no CLI entry point, use this inline approach:
   ```
   .venv/bin/python -c "
   import cProfile
   from scripts.benchmark_engine import run_benchmark
   cProfile.run('run_benchmark()', sort='cumulative')
   " 2>&1 | head -40
   ```

2. **Identify top 5 hotspots**: From cProfile output, list the functions consuming the most cumulative time. Focus on `osmose/engine/` functions, not NumPy/Numba internals.

3. **Check Numba JIT status** for hotspot functions:
   ```
   .venv/bin/python -c "
   import numba
   numba.config.DEVELOPER_MODE = True
   # Import the module to trigger JIT compilation
   from osmose.engine.processes import predation
   print('Numba JIT compilation successful')
   "
   ```
   Look for:
   - Functions falling back to object mode (kills performance)
   - Type inference failures
   - Unsupported Python features inside `@njit`

4. **Check for non-vectorized loops**: Search hotspot files for Python loops over arrays that should be vectorized:
   ```
   grep -n "for.*in range" osmose/engine/processes/{focus}.py
   ```
   Flag any loop iterating over array elements that could use NumPy broadcasting.

5. **Check memory allocation patterns**: Look for array allocations inside hot loops:
   ```
   grep -n "np.zeros\|np.empty\|np.array" osmose/engine/processes/{focus}.py
   ```
   Arrays should be pre-allocated outside loops where possible.

6. **Compare against baseline timing**:
   - Bay of Biscay 5yr baseline: ~2.0s
   - EEC 5yr baseline: ~5.2s

   If current timing exceeds baseline by >10%, flag as regression.

7. **Report findings** as a table:

   | Rank | Function | Time (s) | % Total | Issue |
   |------|----------|----------|---------|-------|
   | 1 | predation._compute_kernel | X.XX | XX% | — |
   | 2 | movement._distribute | X.XX | XX% | Non-vectorized loop |

8. **Suggest optimizations** for any identified bottlenecks:
   - Python loop → NumPy vectorization
   - NumPy operation → Numba `@njit`
   - Repeated allocation → pre-allocated buffer
   - Object-mode Numba → fix type annotations

## Rules

- Always use `.venv/bin/python`, never system python
- Run from `/home/razinka/osmose/osmose-python/`
- First JIT compilation is slow — run twice and report the second (warm) timing
- Do NOT modify engine code during profiling — this is a read-only analysis
- Always verify parity after any suggested optimization is applied
- Performance baselines: BoB 5yr ~2.0s, EEC 5yr ~5.2s (Python), Java BoB ~2.3s, EEC ~7.2s
