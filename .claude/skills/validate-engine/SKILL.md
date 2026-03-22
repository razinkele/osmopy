---
name: validate-engine
description: Run engine validation against reference Java outputs (Bay of Biscay + EEC parity)
disable-model-invocation: true
---

Validate the Python OSMOSE engine against the Java reference engine.

## Arguments

- `suite` (optional): "bob" (Bay of Biscay only), "eec" (EEC only), or "all" (default: "all")
- `years` (optional): number of simulation years (default: 1)

## Steps

1. Run from `/home/razinka/osmose/osmose-python/`

2. **Bay of Biscay validation** (unless suite=eec):
   ```
   .venv/bin/python scripts/validate_engines.py --years {years}
   ```
   - Expect 8/8 species PASS at year 1
   - Report any biomass mismatches with relative error

3. **EEC parity check** (unless suite=bob):
   - Run the EEC test configuration manually:
   ```
   .venv/bin/python -m pytest tests/ -k "eec" -v --no-header
   ```
   - Current baseline: 9/14 PASS
   - Report which species/processes fail and their error magnitudes

4. **Summary**: Print a table with suite, species, status (PASS/FAIL), and relative error

## Rules

- Always use `.venv/bin/python`, never system python
- Do NOT modify engine code based on validation results — only report findings
- If Java JAR is missing, report the error and skip Java-dependent validations
- Compare at year 1 unless the user specifies otherwise
