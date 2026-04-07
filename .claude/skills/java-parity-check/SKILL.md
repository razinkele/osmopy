---
name: java-parity-check
description: Verify config round-trip fidelity — read config, write to temp dir, diff against original to catch value corruption (multi-value arrays, case changes, missing keys)
disable-model-invocation: true
---

Verify that the Python config reader/writer pipeline preserves all OSMOSE parameters correctly for Java consumption.

## When to Use

- After modifying `osmose/config/reader.py`, `osmose/config/writer.py`, or `ui/state.py`
- After changing schema field definitions in `osmose/schema/`
- After modifying `write_temp_config()` in `ui/pages/run.py`
- Before releasing a version that will run the Java engine

## Arguments

- `config` (optional): path to config directory (default: `data/eec_full`)

## Steps

1. Run from `/home/razinka/osmose/osmose-python/`

2. **Run the config round-trip check script**:
   ```
   .venv/bin/python scripts/check_config_roundtrip.py {config}
   ```

3. **Interpret results**:
   - `PASS` = all keys and values survive the round-trip
   - `CASE MISMATCH` = key case changed (Java is case-sensitive, will fail)
   - `VALUE CHANGED` = value was modified (multi-value arrays truncated, defaults substituted)
   - `KEY MISSING` = key present in original but absent in output
   - `KEY ADDED` = key in output that wasn't in original (internal keys like `_osmose.*`)

4. **If any CASE MISMATCH or VALUE CHANGED**: Report the specific keys and stop. These WILL cause Java engine failures (ArrayIndexOutOfBounds, missing parameters, wrong values).

5. **If only KEY ADDED for `_osmose.*` keys**: These are internal Python keys and are expected — PASS.

## Rules

- Always use `.venv/bin/python`, never system python
- Do NOT modify config code based on results — only report findings
- Multi-value entries like `2.3;1.8` must survive as `2.3;1.8`, not `2.3` or `3.5`
- Key case must be preserved: `predation.predPrey.sizeRatio.max.sp5` not `predation.predprey.sizeratio.max.sp5`
