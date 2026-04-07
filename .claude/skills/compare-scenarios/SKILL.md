---
name: compare-scenarios
description: Compare two saved OSMOSE scenario JSON files, highlighting parameter differences and their scientific impact
disable-model-invocation: true
---

Compare two OSMOSE scenarios and explain the scientific implications of their differences.

## Arguments

- `a`: name or path of the first scenario
- `b`: name or path of the second scenario
- `storage` (optional): scenario storage directory (default: `data/scenarios/`)

## Steps

1. Run from `/home/razinka/osmose/osmose-python/`

2. **Load and compare scenarios** using the built-in ScenarioManager:
   ```
   .venv/bin/python -c "
   from pathlib import Path
   from osmose.scenarios import ScenarioManager
   mgr = ScenarioManager(Path('{storage}'))
   diffs = mgr.compare('{a}', '{b}')
   for d in diffs:
       marker = '<' if d.value_b is None else ('>' if d.value_a is None else '~')
       print(f'{marker} {d.key}: {d.value_a} -> {d.value_b}')
   print(f'Total differences: {len(diffs)}')
   "
   ```

3. **Categorize differences** by process domain:
   - **Grid/spatial**: `grid.*`, `map.*` — affects spatial distribution
   - **Species biology**: `species.*`, `growth.*`, `predation.*` — affects population dynamics
   - **Mortality**: `mortality.*`, `fishing.*` — affects survival rates
   - **Reproduction**: `reproduction.*`, `egg.*` — affects recruitment
   - **Movement**: `movement.*` — affects spatial connectivity
   - **Simulation**: `simulation.*`, `output.*` — affects run configuration
   - **Calibration**: parameters with `calibration` in the key

4. **Assess scientific impact** for each category:
   - **High impact**: Changes to predation efficiency, mortality rates, growth parameters (Linf, K, allometric power) — these fundamentally alter ecosystem dynamics
   - **Medium impact**: Changes to movement maps, fishing seasonality, reproduction timing — these shift spatial/temporal patterns
   - **Low impact**: Changes to output frequency, grid resolution, simulation years — these affect precision but not dynamics

5. **Report** in this format:

   ### Scenario Comparison: {a} vs {b}

   | Category | Key | {a} | {b} | Impact |
   |----------|-----|-----|-----|--------|
   | Species biology | species.linf.sp0 | 20.5 | 22.0 | HIGH — changes asymptotic length |

   **Summary**: X parameters differ across Y categories. Z are high-impact changes that will significantly affect simulation outputs.

## Rules

- Always use `.venv/bin/python`, never system python
- Run from `/home/razinka/osmose/osmose-python/`
- If scenario files don't exist, list available scenarios and ask the user to choose
- Do NOT modify scenario files — only read and compare
- If scenarios share the same parent, mention the fork lineage
