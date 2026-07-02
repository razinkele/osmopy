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
few years. This was initially hypothesised as a spin-up collision (RV=0 at
1993 landing on the model's own bootstrap transient). **That hypothesis was
subsequently tested and refuted** — see the start-year sweep below. The config
keys were checked against `osmose/engine/config.py:_load_rv_gate` and match
exactly, so this is a genuine dynamical outcome, not a wiring bug.

## Start-year sweep (refutes the spin-up hypothesis)

Re-ran the measurement varying `reproduction.rv.gate.start.year` (mean_preserving,
nyear=15, seed=0), so model-year-0 lands on years with different RV. Baseline
(gate off) boom/bust = 2.58.

```
start_year | rv@yr0 | boom/bust | vs_off | mean_delta
   1993    | 0.000  |   3.91    | +52%   |  +5%
   1994    | 0.069  |   3.84    | +49%   |  -9%
   1996    | 0.043  |   2.55    |  -1%   |  -2%
   1999    | 0.039  |   3.52    | +36%   | +49%
   2003    | 0.094  |   4.56    | +77%   | -12%
   2005    | 0.169  |   4.93    | +91%   | -25%
   2007    | 0.146  |   2.88    | +12%   | -11%
```

**No start year damps the overshoot.** The best case (1996) is neutral within
noise (-1%, far short of the ≥25% target); every other start makes boom/bust
worse, and the highest-RV-window starts (2003, 2005 — which include the 2004
and 2016 inflow pulses) make it *much* worse (+77%, +91%). Moving off the 1993
spin-up window does not help, so the reversal is not a spin-up artifact — it is
intrinsic to mean_preserving gating: normalising a strongly-pulsed RV series to
mean-1 amplifies recruitment variance in years around the inflow pulses, which
the model converts into larger biomass swings, not smaller ones.

## Status

Boom/bust target (≥25% reduction) **not met at any tested start year** — the
best case is -1% (noise), and start-year variation makes it worse, not better.
Mean-preservation target (±10%) is met only for some starts (e.g. 1993 +5%,
1996 -2%) and violated for others (1999 +49%, 2005 -25%). The negative result
is **robust**: RV recruitment gating in mean_preserving mode does not stabilise
the Baltic cod overshoot. Recorded as an honest finding, not tuned.

## raw_cap mode + ssb_half recalibration

The literal environmental cap `m = clip(rv/ref, 0, 1)` (ref=0.20, start=1994)
was tested and, per the spec §3.2/§10 caveat, `ssb_half` was swept to try to keep
cod viable. Baseline (gate off) boom/bust = 2.58, mean = 2054.

```
ssb_half | boom/bust | vs_off | mean | mean_vs_off | viable
 120000  |   6.40    | +148%  | 1617 |   -21%      | yes
  40000  |   5.97    | +131%  |  396 |   -81%      | yes
  15000  |   6.30    | +144%  |  126 |   -94%      | yes
   6000  |   5.76    |   -    |   41 |   -98%      | COLLAPSE
   2500  |   6.06    |   -    |   14 |   -99%      | COLLAPSE
   1000  |   5.89    |   -    |    7 |  -100%      | COLLAPSE
```

raw_cap is **worse than mean_preserving**: boom/bust rises to ~6× (+130–148%)
and the mean drops even at the current ssb_half (−21%). **Recalibrating
`ssb_half` cannot rescue it** — this is structural: cod's B-H term is near-inactive
(ssb_half=120000 ≫ cod SSB ~2000), and B-H can only *suppress* recruitment below
the linear rate, never restore the mean the gate removes. Lowering `ssb_half`
therefore adds suppression on top of the gate and drives cod to collapse
(mean 1617 → 41 → 7) without ever damping boom/bust. Raising `ssb_half` does
nothing (the term is already ≈1). The mean-restoring lever would be larval
mortality / fecundity, not `ssb_half` — but with raw_cap already worsening
boom/bust at the viable end, restoring the mean would not change the sign.

## Overall conclusion

RV recruitment gating **does not stabilise the Baltic cod overshoot in either
mode**, across start years (mean_preserving) and across `ssb_half` (raw_cap):
every viable configuration leaves boom/bust unchanged-to-much-worse. The
mechanism injects the real, strongly-pulsed interannual RV signal into
recruitment, and the model amplifies that variance into larger biomass swings
rather than damping them. The reproductive-volume-as-recruitment-multiplier
hypothesis is **not supported** by this model. The gate remains a correct,
inert-by-default, config-gated mechanism (useful for further LTL/coupling
experiments), but it is not the lever that stabilises Baltic cod here.
