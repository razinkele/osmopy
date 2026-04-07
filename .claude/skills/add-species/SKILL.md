---
name: add-species
description: Scaffold all config/schema/engine entries needed for a new species in an OSMOSE configuration
disable-model-invocation: true
---

Scaffold all the files and config entries required to add a new species to an OSMOSE configuration.

## Arguments

- `name` (required): Species common name (e.g., "European anchovy")
- `type` (required): "focal", "resource", or "background"
- `config_dir` (optional): Path to the OSMOSE configuration directory (default: prompt user)

## Steps

1. **Determine the species index**: Read the existing config to find the next available `sp{idx}`:
   ```
   grep -r 'species.name.sp' {config_dir}/ | sort -t'p' -k2 -n | tail -1
   ```
   The new species gets `sp{max_idx + 1}`.

2. **Update species count**: Increment `simulation.nspecies` in the main config file.

3. **Add required schema fields**: For each required field in the species schema, create the config entry with the new index. At minimum:
   - `species.name.sp{idx}` — the species name
   - `species.type.sp{idx}` — "focal", "resource", or "background"
   - `species.linf.sp{idx}` — asymptotic length (cm)
   - `species.k.sp{idx}` — von Bertalanffy K
   - `species.t0.sp{idx}` — von Bertalanffy t0
   - `species.lw.condition.factor.sp{idx}` — length-weight condition factor
   - `species.lw.allometry.power.sp{idx}` — length-weight allometry exponent
   - `species.lifespan.sp{idx}` — max age (years)
   - `species.egg.size.sp{idx}` — egg size (cm)
   - `species.egg.weight.sp{idx}` — egg weight (tonnes)
   - `species.maturity.size.sp{idx}` — size at maturity (cm)
   - `species.vonbertalanffy.threshold.age.sp{idx}` — age threshold

4. **Create species-specific config file** (if focal): `{config_dir}/species/osm_param-{name_slug}.csv` with species-specific parameters.

5. **Create distribution maps**: Prompt the user for spatial distribution approach:
   - Uniform across all cells
   - Specific cells (user provides list)
   - Map file (user provides CSV path)

6. **Add to predation matrix**: If other focal species exist, the predation accessibility matrix needs a new row and column. Prompt the user for predation interactions.

7. **Verify**: Run a quick config validation:
   ```
   .venv/bin/python -c "from osmose.config import read_config; c = read_config('{config_dir}'); print(f'Species count: {len([k for k in c if k.startswith(\"species.name.sp\")])}')"
   ```

## Rules

- Always use `.venv/bin/python`, never system python
- Species indices are 0-based and must be contiguous
- Config keys are case-sensitive — use exact lowercase as in Java
- Do NOT modify engine code — this skill only touches configuration files
- For resource species, skip predation matrix and reproduction parameters
- For background species, only `species.name`, `species.type`, and distribution maps are needed
