---
name: diagnose-divergence
description: Diagnose numerical divergence between Python and Java engine outputs for a specific species and timestep
disable-model-invocation: true
---

Diagnose where the Python OSMOSE engine diverges from Java reference outputs.

## Arguments

- `species` (optional): species name or index to focus on (default: all species)
- `step` (optional): timestep at which divergence first appears (default: auto-detect)
- `config` (optional): "bob" (Bay of Biscay) or "eec" (EEC) (default: "bob")

## Steps

1. Run from `/home/razinka/osmose/osmose-python/`

2. **Run existing diagnostic script** if doing a general check:
   ```
   .venv/bin/python scripts/diagnose_divergence.py
   ```

3. **Run validation to identify which species diverge**:
   ```
   .venv/bin/python scripts/validate_engines.py --years 1
   ```
   Parse output to find FAIL species and their relative errors.

4. **Narrow down the divergent process** by examining the simulation step-by-step:
   - Read the species config: egg weight, growth params, mortality rates
   - Check each process in order: initialization -> growth -> predation -> natural mortality -> fishing -> starvation -> reproduction -> movement
   - For each process, compare Python output arrays against Java reference baselines in `tests/baselines/`

5. **Identify root cause** — common patterns:
   - **Weight units**: grams vs tonnes (multiply by 1e-6)
   - **Rate conversion**: per-year vs per-timestep (divide by n_dt_per_year)
   - **Iteration order**: Java processes species in index order; verify Python matches
   - **Integer division**: Java uses `(int)` casts; Python uses `//` or `int()`
   - **Boundary conditions**: age 0, max age, zero biomass, empty cells
   - **Config key mismatch**: check exact key names match Java

6. **Report findings** as a table:

   | Species | Process | Error Magnitude | Root Cause |
   |---------|---------|-----------------|------------|
   | Anchovy | Growth | 1e-6x | Weight in grams, should be tonnes |

## Rules

- Always use `.venv/bin/python`, never system python
- Run from `/home/razinka/osmose/osmose-python/`
- Do NOT modify engine code — only diagnose and report
- Reference Java source in `osmose-java/` for formula comparison (read-only)
- Check `tests/baselines/` for frozen reference outputs
- If the Java JAR is missing, report it and focus on Python-only analysis
