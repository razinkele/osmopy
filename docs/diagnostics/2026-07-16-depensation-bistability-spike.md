# Recruitment depensation/Allee de-risk spike — GO (depensation manufactures bistability)

**Date:** 2026-07-16 · **Status:** GO (proof-of-mechanism) · **Kind:** throwaway de-risk spike
**Script:** `scripts/spikes/depensation_bistability_spike.py`

## Question

After the SSB-trajectory F-hindcast NO-GO (`2026-07-15-ssb-f-hindcast-spike.md`), the three-mechanism
exploration concluded that **recruitment depensation/Allee is the ROOT lever** — the only one that
can *create* the alternative stable states the model currently lacks (fishing hysteresis and
historical-state init are downstream diagnostics that need bistability to exist first). The engine's
4 stock-recruitment forms (Beverton-Holt, Ricker, hockey-stick, Shepherd) are all *compensatory* —
per-capita recruitment is maximal as SSB→0, so one attractor. This spike asks, before building a
depensation feature: **can a depensatory/Allee recruitment term manufacture bistability in this model
at all?**

## Method

- **Monkeypatch** (throwaway, no feature build): a cod (sp0) Allee factor
  `A(SSB) = SSB^θ / (S50^θ + SSB^θ)` (θ=4, the sharpest trap = most favorable) applied to
  `apply_stock_recruitment` using the real per-step cod SSB.
- **Validated warm-start reciprocal-invasion contrast** (reuses `scripts/baltic_bistability_chunk0.py`):
  cod-rich (300 kt) vs cod-poor (1 kt) standing-stock initial conditions, evolved under identical
  parameters. Warm-start disables egg-rescue, so a low-SSB trap is genuine. **Bistable** iff the two
  ICs land in different cod basins (one persists, one collapses) at the *same* parameters.
- **Grid:** larval-mortality scale ∈ {0.3, 0.5, 0.7, 1.0} (low = cod-viable/overshoot; 1.0 = deployed)
  × S50 ∈ {30 kt, 90 kt}, θ=4, 2 seeds, 15-yr runs. Plus a **no-Allee baseline control** (must
  reproduce the known MONOSTABLE result).

## Result — GO

```
BASELINE (no Allee):
  scale=1.0: cod_rich=185,348   cod_poor=892       gap=1.00   (transient — see caveat)
  scale=0.5: cod_rich=6,842,327 cod_poor=6,316,056 gap=0.08   MONOSTABLE (both overshoot) ✓

ALLEE (θ=4):  every point a BASIN SPLIT
  scale=0.3 S50=30k: rich=9,391,580 poor=218   gap=1.00
  scale=0.3 S50=90k: rich=8,417,851 poor=218   gap=1.00
  scale=0.5 S50=30k: rich=7,107,813 poor=221   gap=1.00
  scale=0.5 S50=90k: rich=6,833,630 poor=223   gap=1.00
  scale=0.7 S50=30k: rich=3,575,780 poor=221   gap=1.00
  scale=0.7 S50=90k: rich=3,549,041 poor=224   gap=1.00
  scale=1.0 S50=30k: rich=187,050   poor=226   gap=1.00   (contaminated — see caveat)
  scale=1.0 S50=90k: rich=183,934   poor=225   gap=1.00   (contaminated)
```

**Depensation manufactures genuine bistability.** At the low larval scales (0.3–0.7), where the
no-Allee baseline is unambiguously monostable (both ICs → ~6.5M at scale 0.5, gap 0.08), the Allee
gate splits them: the cod-rich IC self-sustains and overshoots (3.5–9.4M), the cod-poor IC falls into
the low-SSB trap and collapses to ~220 t — at *identical* parameters. The split is caused by the
depensation term (the baseline control at the same scale shows the poor IC *recovering* to 6.3M
without it), is robust across both S50 values, and is seed-consistent (poor basin 218–226t across all
points). This is the alternative-stable-state structure the compensatory SR forms cannot produce, and
that the warm-start/Chunk-C/Chunk-A2 investigations all failed to find — now demonstrated.

## Caveats (so the GO isn't oversold)

- **Proof-of-mechanism, not a calibrated model.** The bistability appears in *cod-viable* regimes
  (reduced larval-M, scales 0.3–0.7). At the **deployed** larval scale (1.0) cod is in the collapse
  regime: the scale=1.0 "split" is **transient-contaminated** — the baseline *also* splits there
  (rich=185kt vs poor=892t) because the rich IC is mid-collapse at a 15-yr horizon, not in a stable
  high basin (the warm-start study found deployed cod-rich → ~0 at longer horizon). So this spike does
  **not** yet show bistability *at deployed parameters* — only that the mechanism works where cod can
  persist.
- **Neither basin is in-band.** Rich overshoots (millions), poor ≈ collapse (~220t); neither matches
  the ICES cod SSB band (~120kt). Calibrating S50/θ **jointly with larval-M** so the two basins
  bracket realistic cod SSB (healthy ~Bpa vs collapsed) is the central task of the full build, and the
  real open risk (S50 too high → always sub-threshold collapse; too low → inert).
- **2 seeds, θ=4 only.** Values are tight, but a proper build should confirm across more seeds/θ and a
  finer grid. The 15-yr "instrument-limited" horizon should be lengthened to separate stable basins
  from slow transients.
- **SSB==0 seeding-rescue** (`reproduction.py:122-128`) must stay disabled (warm-start does) or the
  trap saturate at low-but-nonzero SSB, or the bootstrap would silently rescue the collapsed basin.

## Conclusion → GO to build the depensation feature

The make-or-break question — *can depensation create bistability here?* — is answered **yes,
decisively**. This unblocks the SSB-trajectory/regime-shift capability that the F-hindcast and prior
bistability work had walled off. Recommended next: brainstorm → spec → build the depensation as a
proper config-plumbed recruitment gate (mirroring the RV/thermal gates), with the **joint S50/θ ×
larval-M calibration to place the bistable regime at realistic cod SSB** as the central task, and the
fishing-hysteresis F-ramp + warm-start basin check as built-in validation. Historical-state init
remains shelved (dependent, and data-limited). Related: `2026-07-15-ssb-f-hindcast-spike.md`,
`baltic_chunka2_results_2026-07-09.md`.
