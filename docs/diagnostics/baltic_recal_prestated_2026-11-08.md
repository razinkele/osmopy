# Step 10b: targeted recalibration on the literature-corrected config

Written 2026-11-08 19:30, BEFORE the first evaluation.

## Config held fixed
master (#227 GreySeal, #230 herring packing removed, #233 T3 floor + T1 no L-inf cap) plus:
- the corrected LTL file (#234 generator: column integral + hole fill);
- C2: cod_east spawning on the 14 SD 25 cells;
- herring autumn egg share 0 (sourced: about 1% or less today), folded into spring.

Step 10a measured on that config (50 yr x 5 seeds): cod_west 24.9-26.1 kt (ceiling 25), herring
0.798-0.807 Mt (floor 0.8), cod_east 80.8-82.2 kt, sprat and flounder in.

## Solve
- Free: mortality.additional.larva.rate.sp1 (herring) and .sp0 (cod_west), in CSV units
  (per year; the engine divides by ndt).
- Objective per stock: final-decade (years 40-49) mean of a 50-yr single-seed (seed 42) run,
  the certification's own quantity.
- Method: 1-D bisection in log-rate per stock, Gauss-Seidel, herring first (it is the food),
  2 sweeps.
- Targets (biomass_targets.csv): herring 1.5 Mt, cod_west 10 kt. Tolerance +-10% on its own axis.
- Fallback if herring saturates below 1.5 Mt (the gate run with autumn eggs reached only
  1.16 Mt): accept herring >= 1.0 Mt. Accept cod_west <= 20 kt.
- cod_east rule: after each sweep, if cod_east's single-seed final-decade mean leaves 60-85 kt,
  add mortality.additional.larva.rate.sp8 as a third dimension targeting 70 kt (its target).

## Acceptance
5-seed certification 5/5 assessed. Then the SP1b re-solve, which is mandatory: cod_west's larval
rate moves d0, and sp1_on_config raises on the mismatch. The PR does NOT auto-merge; it waits for
the user's word.
