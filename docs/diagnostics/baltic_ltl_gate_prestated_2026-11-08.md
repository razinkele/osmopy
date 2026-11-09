# Step 9 gate: corrected LTL (column integral + hole fill) with CURRENT parameters

Written 2026-11-08 19:00, BEFORE the run.

Run: 50 yr x 5 seeds (scripts/baltic_stability_certify.py --params current), on the latest
engine and config: master (GreySeal #227 + C1a #230) + T3 + T1 (fix/engine-sub-one-fish-floor),
with data/baltic/baltic_ltl_biomass.nc replaced by the corrected build (phyto 0.52-0.54x,
zoo/benthos 0.48x of the committed file). Measure only: no parameter changes.

Reading, judged on the 5 ASSESSED stocks (cod_west, cod_east, herring, sprat, flounder):
- GREEN: all 5 persist and every final-decade mean lies within 2x of its envelope
  (lower/2 <= mean <= 2*upper). A bounded recalibration (larval mortalities) is plausible.
  Proceed to the joint recalibration (step 10).
- AMBER: all 5 persist, but at least one lies beyond 2x of its envelope. Recalibrating would
  need a broad parameter set (larval mortalities + accessibilities), which is a project-scale
  decision for the user. Report and stop.
- RED: any assessed stock collapses (persists = no). Park the LTL rebuild with its measurements
  (step 11: ship C2 alone with a 1-D cod_east solve instead).

Precedent to keep in mind: depletable plankton cut herring 80% and cod_east 70%, and a bounded
6-parameter refit still failed its gate. A food cut of about half is a shock of similar size.
