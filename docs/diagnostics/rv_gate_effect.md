# RV recruitment-gate effect measurement (Task 6)

Measured with the Step-5 snippet from `.superpowers/sdd/task-6-brief.md`, run
against `data/baltic/baltic_all-parameters.csv` at `simulation.time.nyear=15`,
`seed=0`, comparing the RV gate off (baseline) vs on
(`reproduction.rv.gate.mode=mean_preserving`,
`reproduction.rv.gate.series.file=data/baltic/forcing/baltic_rv_gate_series.csv`,
`reproduction.rv.gate.start.year=1993`, `reproduction.rv.gate.species.enabled.sp0=true`).
Cod biomass (`sp0`) windowed to model years 3-14 (spin-up excluded).

## Result

```
boom/bust off=2.6 on=3.9 reduction=-52%
mean cod biomass off=2054 on=2150 delta=+5%
```

- **Boom/bust ratio**: off = 2.6x, on = 3.9x → the gate *increased* the
  boom/bust ratio by 52%, the opposite of the ≥25% reduction success target
  (spec §10.2).
- **Mean cod biomass**: off = 2054, on = 2150 → +5%, well within the ±10%
  success band.

## Interpretation

The mean-preserving normalisation held cod biomass close to the ungated
baseline (+5%, inside tolerance) but did not reduce — and in this 15-year
window made worse — the boom/bust ratio. A plausible mechanism: the gate
series (`baltic_rv_gate_series.csv`) starts at `spawning_rv=0.0` in its first
year (1993) and only rises gradually (0.069, 0.057, 0.043, ...) over the next
few years. With `reproduction.rv.gate.start.year=1993` landing near the start
of the 15-year run, recruitment is heavily suppressed during the early
transient and then released as the RV series recovers, which can sharpen
rather than dampen the early bust/rebound cycle that dominates the 3-14
window's max/min. This is a genuine dynamical interaction between the gate's
transient ramp-up and the model's own spin-up transient, not a wiring bug —
the config keys were checked against `osmose/engine/config.py:_load_rv_gate`
and match exactly.

## Status

Boom/bust target (≥25% reduction) **not met** — reversed sign, -52% (i.e. a
52% *increase*). Mean-preservation target (±10%) **met** (+5%). This is
recorded as an honest finding per the task brief (spec §3.2's closed-loop
mean-shift caveat), not tuned to hit the target. Candidates for follow-up
(out of scope for Task 6): fit/shift the gate series' effective start year
away from the model spin-up window, or evaluate the boom/bust window over a
longer run (nyear > 15) where the early-transient interaction is a smaller
fraction of the windowed years.
