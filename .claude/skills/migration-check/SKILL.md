---
name: migration-check
description: Verify that engine changes maintain numerical parity with Java reference outputs
user-invocable: false
---

Automatically verify numerical parity after engine modifications.

## When to Trigger

Invoke this skill after modifying any file in `osmose/engine/` that implements a biological process (growth, predation, mortality, reproduction, movement, fishing, starvation, bioenergetics).

## Steps

1. **Run targeted parity tests** from `/home/razinka/osmose/osmose-python/`:
   ```
   .venv/bin/python -m pytest tests/test_engine_parity.py -v --no-header -x
   ```

2. **Run Bay of Biscay validation** (quick — ~2 seconds):
   ```
   .venv/bin/python scripts/validate_engines.py --years 1
   ```
   - All 8 species must PASS
   - Any FAIL = the change broke numerical parity

3. **Run EEC parity tests**:
   ```
   .venv/bin/python -m pytest tests/ -k "eec" -v --no-header -x
   ```
   - Baseline: 14/14 PASS
   - Any regression from baseline = the change broke parity

4. **If any test fails**: Stop and report which process diverged, the relative error magnitude, and the specific line of code that likely caused it. Do NOT proceed with further changes until parity is restored.

5. **If all pass**: Report "Parity maintained" and continue.

## Rules

- Always use `.venv/bin/python`, never system python
- Do NOT skip this check when modifying engine process code
- If the Java JAR is missing, run only the Python-side parity tests (step 1)
- A relative error > 1e-10 is a parity failure
