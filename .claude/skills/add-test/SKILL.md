---
name: add-test
description: Scaffold a new engine parity test for a specific biological process
disable-model-invocation: true
---

Scaffold a new parity test that verifies a Python engine process produces identical output to the Java baseline.

## Arguments

- `process` (required): the engine process to test (e.g., "growth", "predation", "reproduction")
- `config` (optional): "bob" (Bay of Biscay) or "eec" (EEC) — default: "bob"

## Steps

1. **Identify the engine module**: Find the process implementation in `osmose/engine/`:
   ```
   ls osmose/engine/processes/
   ```
   Map the process name to its module (e.g., growth → `growth.py`, predation → `predation.py`).

2. **Check existing tests**: Search for tests that already cover this process:
   ```
   grep -rn "{process}" tests/test_engine_parity.py tests/test_engine*.py
   ```

3. **Check the baseline exists**: Parity tests compare against saved baselines:
   ```
   ls tests/baselines/
   ```
   If no baseline exists, generate one first:
   ```
   .venv/bin/python scripts/save_parity_baseline.py --years 1 --seed 42
   ```

4. **Create the test function** in `tests/test_engine_parity.py` following this pattern:

   ```python
   @pytest.mark.skipif(
       not _baseline_path().exists(),
       reason="No baseline — run scripts/save_parity_baseline.py first",
   )
   def test_{process}_parity():
       """Verify {process} output matches baseline."""
       baseline = np.load(_baseline_path())
       biomass, abundance, mortality = _run_engine(DEFAULT_YEARS, DEFAULT_SEED)

       # Extract process-specific outputs from baseline
       np.testing.assert_allclose(
           biomass,
           baseline["biomass"],
           rtol=1e-10,
           err_msg="{process}: biomass drift from baseline",
       )
   ```

   Adapt the assertions to the specific process:
   - **Growth**: check length-at-age arrays
   - **Predation**: check mortality rates and diet composition
   - **Reproduction**: check egg/larval production arrays
   - **Movement**: check spatial distribution arrays
   - **Fishing**: check catch arrays and fishing mortality
   - **Starvation**: check starvation mortality rates

5. **Run the new test**:
   ```
   .venv/bin/python -m pytest tests/test_engine_parity.py -v -k "{process}"
   ```

6. **Run full suite** to check for regressions:
   ```
   .venv/bin/python -m pytest tests/ -x -q
   ```

## Rules

- Always use `.venv/bin/python`, never system python
- Run from `/home/razinka/osmose/osmose-python/`
- Use `rtol=1e-10` for floating-point comparisons (matches Java double precision)
- Test names must follow pattern `test_{process}_parity`
- Include a docstring explaining what the test verifies
- Skip gracefully if baseline doesn't exist (CI generates baselines separately)
- Do NOT modify existing parity tests — add new functions only
