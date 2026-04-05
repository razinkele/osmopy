#!/usr/bin/env python3
"""Diagnostic v2: Pinpointed weight-unit bug causing ~10^6 divergence.

This verifies the root cause and shows exact magnitude of the discrepancy.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import initialize, _growth, _reset_step_variables


def main():
    reader = OsmoseConfigReader()
    cfg = reader.read(Path("data/examples/osm_all-parameters.csv"))
    config = EngineConfig.from_dict(cfg)
    grid = Grid(ny=10, nx=10, ocean_mask=np.ones((10, 10), dtype=np.bool_))
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("WEIGHT UNIT BUG VERIFICATION")
    print("=" * 80)

    # ---- 1. Check weight units throughout the codebase ----
    print("\n--- 1. Weight unit comparison ---")
    sp = 0
    name = config.species_names[sp]
    egg_w = config.egg_weight_override[sp]  # 0.0005

    print(f"\n  {name}:")
    print(f"  species.egg.weight.sp0 = {egg_w}")
    print("  Java docs: 'Weight (gram) of eggs' -> 0.0005 GRAMS")
    print("  Java School constructor: this.weight = weight * 1e-6 -> 5e-10 TONNES")
    print("  Java incrementLength: setWeight(computeWeight(length) * 1e-6f) -> TONNES")
    print("  Java biomass = abundance * weight_tonnes -> TONNES")
    print()
    print(f"  Python initialize(): weights = egg_weight_override = {egg_w} (no conversion)")
    print("  Python growth: new_weight = cf * L^b (GRAMS, no conversion)")
    print("  Python biomass = abundance * weight_GRAMS -> GRAMS (should be TONNES)")

    # ---- 2. Demonstrate the magnitude ----
    print("\n\n--- 2. Magnitude of biomass error ---")

    state = initialize(config, grid, rng)

    # At initialization, Python's biomass is set directly = seeding_biomass/n_schools (tonnes)
    # But abundance = biomass_tonnes / weight_grams -> NOT individual count
    for sp in range(config.n_species):
        mask = state.species_id == sp
        py_weight = state.weight[mask][0]
        py_abundance = state.abundance[mask][0]
        py_biomass = state.biomass[mask][0]

        java_weight_tonnes = py_weight * 1e-6
        java_abundance = py_biomass / java_weight_tonnes  # correct individual count

        print(f"\n  {config.species_names[sp]}:")
        print(f"    Python weight = {py_weight:.6e} (grams)")
        print(f"    Java weight   = {java_weight_tonnes:.6e} (tonnes)")
        print(f"    Python abundance = {py_abundance:.6e} (tonnes/grams = meaningless)")
        print(f"    Java abundance   = {java_abundance:.6e} (correct individual count)")
        print(f"    Ratio (Java/Py)  = {java_abundance / py_abundance:.0f}x")

    # ---- 3. Growth makes it worse ----
    print("\n\n--- 3. After growth: weight diverges further ---")

    # Reset for step, then apply growth
    state = _reset_step_variables(state)
    # Give all schools a fake pred_success_rate to trigger growth
    state = state.replace(pred_success_rate=np.ones(len(state), dtype=np.float64))
    grown = _growth(state, config, rng)

    for sp in [0, 5]:  # Anchovy and Hake
        mask = grown.species_id == sp
        if not mask.any():
            continue
        w = grown.weight[mask][0]
        length = grown.length[mask][0]
        b = grown.biomass[mask][0]
        abd = grown.abundance[mask][0]
        print(f"\n  {config.species_names[sp]} after 1 growth step:")
        print(f"    length = {length:.4f} cm")
        print(f"    weight (Python, grams) = {w:.6e}")
        print(f"    weight (Java, tonnes)  = {w * 1e-6:.6e}")
        print(f"    biomass = abd * weight = {abd:.6e} * {w:.6e} = {b:.6e}")
        print("    This biomass is in GRAMS (Python) vs TONNES (Java)")
        print(f"    Factor: {1e6}x")

    # ---- 4. Reproduction amplifies the error ----
    print("\n\n--- 4. Reproduction: SSB computation ---")
    print()
    print("  Python's SSB = sum(abundance * weight) for mature schools")
    print("  abundance is (tonnes / grams), weight is grams")
    print("  So SSB = (tonnes/grams) * grams = tonnes")
    print("  -> SSB is accidentally correct! The units cancel out.")
    print()
    print("  But after growth, biomass = abundance * weight_grams")
    print("  And the output collects biomass per species.")
    print("  So output biomass is in GRAMS, not TONNES.")
    print()

    # ---- 5. All locations where weight in grams causes errors ----
    print("\n--- 5. All affected code locations ---")
    print()
    print("  1. initialize() line 222:")
    print("     abundance = biomass_per_school / weights")
    print("     weights is in grams -> abundance is 1e6x too small")
    print()
    print("  2. growth.py line 102:")
    print("     new_weight = cf * L^b  (grams, should be * 1e-6 for tonnes)")
    print()
    print("  3. growth.py line 105:")
    print("     new_biomass = abundance * new_weight  (grams units)")
    print()
    print("  4. reproduction.py line 38:")
    print("     SSB = abundance * weight  (cancels out: (t/g)*g = t)")
    print("     -> SSB and egg count are accidentally correct")
    print()
    print("  5. reproduction.py line 72:")
    print("     egg_weight = allometry or override (grams)")
    print()
    print("  6. reproduction.py line 82:")
    print("     biomass = eggs_per_school * egg_weight (grams units)")
    print()
    print("  7. predation.py line 103 (Numba):")
    print("     biomass_p = abundance * weight  (grams units for max_eatable)")
    print()
    print("  8. predation.py line 151 (Numba):")
    print("     prey_bio = abundance * weight  (grams units)")
    print()
    print("  9. predation.py line 171 (Numba):")
    print("     n_dead = eaten_from_prey / weight  (eaten in grams, weight in grams -> correct)")
    print()
    print("  10. predation.py line 571:")
    print("      new_biomass = abundance * weight  (grams units)")
    print()
    print("  Summary of what's wrong:")
    print("  - weight is in grams, should be in tonnes")
    print("  - biomass = abundance * weight is in grams, should be in tonnes")
    print("  - abundance = biomass_tonnes / weight_grams is 1e6x too small")
    print()
    print("  What accidentally works:")
    print("  - SSB in reproduction: (tonnes/grams) * grams = tonnes")
    print("  - Predation n_dead: eaten_grams / weight_grams = correct count")
    print("  - Starvation rate: depends on pred_success_rate (dimensionless)")

    # ---- 6. Verify output biomass magnitudes ----
    print("\n\n--- 6. Output biomass comparison ---")
    print()
    print("  At initialization:")
    total_py = sum(state.biomass[state.species_id == sp].sum() for sp in range(config.n_species))
    print(f"    Python total biomass = {total_py:.6e}")
    print("    This is in TONNES (set directly from seeding_biomass)")
    print()
    print("  After growth (abundance * weight_grams):")
    total_grown = sum(grown.biomass[grown.species_id == sp].sum() for sp in range(config.n_species))
    print(f"    Python total biomass = {total_grown:.6e}")
    print(f"    Java equivalent (tonnes) = {total_grown * 1e-6:.6e}")
    print(f"    Ratio = {total_grown / (total_grown * 1e-6):.0f}x")

    # ---- 7. The fix ----
    print("\n\n" + "=" * 80)
    print("RECOMMENDED FIX")
    print("=" * 80)
    print("""
Convert all weights to TONNES at the point of creation, matching Java:

1. config.py: Convert egg_weight_override from grams to tonnes:
   egg_weight_override[i] = float(v) * 1e-6

2. initialize() in simulate.py:
   weights = config.condition_factor[species_ids] * lengths ** config.allometric_power[species_ids]
   weights *= 1e-6  # grams -> tonnes
   (egg_weight_override is already in tonnes after fix #1)

3. growth.py line 102:
   new_weight = config.condition_factor[sp] * new_length ** config.allometric_power[sp]
   new_weight *= 1e-6  # grams -> tonnes

4. reproduction.py line 72:
   egg_weight = config.condition_factor[sp] * egg_len ** config.allometric_power[sp]
   egg_weight *= 1e-6  # grams -> tonnes
   (egg_weight_override already in tonnes after fix #1)

5. No changes needed to predation.py — it uses abundance * weight
   which will now be in tonnes, matching Java.

6. The 1e6 in reproduction's nEgg formula is CORRECT and must stay:
   nEgg = sex_ratio * fecundity * SSB_tonnes * season * 1e6
   (converts SSB from tonnes to grams for fecundity which is eggs/gram)
""")


if __name__ == "__main__":
    main()
