# SSB-trajectory F-hindcast de-risk spike — HONEST NEGATIVE (NO-GO for Spec 3)

**Date:** 2026-07-15 · **Status:** NO-GO · **Kind:** throwaway de-risk spike (not a CI gate)
**Script:** `scripts/spikes/ssb_f_hindcast_spike.py`

## Question

Spec 3 of the ICES-strengthening sequence would add an SSB-**trajectory** target/diagnostic to the
Baltic calibration. It was flagged HIGH RISK because the interannual reproductive-volume (RV)
hindcast (PR #109) was an honest-negative on a closely related question. This spike de-risks it
**before** building anything.

The RV hindcast held fishing mortality **F constant** and varied only environmental forcing. But SSB
of exploited stocks is dominated by F, and F is a model **input**. So the untested, genuinely
different lever is: **drive the model with the historical annual F trajectory — does modeled SSB then
track observed ICES SSB better than constant F does?** (Skill-delta framing, identical to PR #109.)

## Method

- **Mechanism (verified first):** `mortality.fishing.rate.byyear.file.sp{i}` is fully wired through the
  production Numba mortality path even in the Baltic *fisheries-based* config. A mechanical check
  (cod F forced to 5× base) dropped cod SSB to 0.24× — the override demonstrably takes effect.
- **Species:** cod (sp0, ICES `cod.27.24-32`) and sprat (sp2, ICES `spr.27.22-32`) — the two most
  cleanly ICES-derivable stocks. Metric is **correlation, which is scale-invariant**, so cod's
  index-scale ICES SSB and sprat's tonnes-scale ICES SSB are both usable directly from the frozen
  snapshots (each stock's SSB and F taken from the *same* assessment record → internally consistent;
  no unit-mixing).
- **F input (relative scaling):** `F_model[year] = base_F × ICES_F[year] / mean(ICES_F[2018–2022])`.
  Relative, so the model stays in its calibrated 2018–22 regime instead of being fished out of it —
  the choice most *favorable* to tracking, isolating trajectory **shape**.
- **Arms:** `null` (constant base F, current config) vs `fdriven` (cod+sprat byyear-F). 5 seeds,
  30-year runs (1993–2022), `output.ssb.enabled`, `PythonEngine().run_in_memory(...).ssb()`.
- **Metric:** `skill_delta = corr(SSB_fdriven, ICES) − corr(SSB_null, ICES)` over post-spinup windows.

## Result

```
scaled byyear-F:  cod min=0.014 max=1.168 (1993=0.332, 2020=0.033);  sprat min=0.189 max=0.408

skill delta (corr(fdriven,ICES) − corr(null,ICES)):
  window yr4-29:  cod +0.009 ±0.004      sprat +0.049 ±0.007
  window yr4-15:  cod +0.348 ±0.005      sprat +0.001 ±0.005
```

**All skill deltas are negligible except cod yr4-15 (+0.348), which is spurious** (see below).

Mean trajectories expose why:
- **cod** — both arms are a spin-up transient (0 → peak at ~yr5) then a **monotonic collapse to ~0**
  by yr20-25 (the model's known monostable larval-M regime). Observed ICES cod *rose* over 1997–2008
  (index 0.32→0.79). Model declines while observed rises → correlation is strongly **negative** in
  both arms (`corr_null`(yr4-15) = −0.893). The fdriven "+0.348" is an artifact: heavier relative F
  makes cod crash to zero *faster* and **flatline**, mechanically shrinking the anti-correlation
  magnitude. A degenerate flatline, not tracking.
- **sprat** — both arms ramp monotonically 47k → 6.4M then plateau; observed fluctuates flat around
  ~1M. F-forcing barely changes the trajectory (null plateau 6.45M vs fdriven 6.50M). Correlation is
  negative in both arms; skill delta ≈ 0.

## Conclusion — NO-GO

Even with the genuinely-different **F-forcing** lever (relative-scaled to stay in-regime), the
deployed Baltic model **cannot reproduce the observed SSB trajectory shape** for either of the two
cleanest stocks. The model's SSB relaxes to its own **intrinsic attractor** — monostable larval-M
collapse for cod, carrying-capacity plateau for sprat — within ~5–10 years, washing out any imposed
forcing. This is the **same failure family** as the RV hindcast negative, now confirmed for F.

**Do not build Spec 3 as an SSB-trajectory hindcast/objective.** The ICES-strengthening sequence is
complete at Spec 1 (catch objective, PR #114) + Spec 2 (recruitment diagnostic, PR #115).

### Caveats (stated so the negative isn't oversold)
- Relative F scaling was the choice most favorable to tracking; absolute ICES F would only crash cod
  harder (mechanical check + cod fdriven both show this).
- Only cod+sprat tested; herring (multi-stock) and flounder (index, no recruitment) targets are
  messier — no reason to expect tracking when the two cleanest cases fail.
- The model is not initialized to 1993's true stock state, but the spin-up transient shows it does
  not *hold* any imposed state — it relaxes to its attractor regardless, so correct ICs would wash
  out too. This reinforces (does not undermine) the NO-GO.
- Magnitude mismatch (sprat ~6× high, cod → 0) is a separate calibration issue; correlation is
  scale-invariant, so it did not drive the negative — the trajectory **shape** simply doesn't match.

### If ever revisited (a DIFFERENT mechanism, not this lever)
A real SSB-trajectory capability needs the model to *carry* an interannual signal it currently does
not: recruitment-level depensation/Allee, fishing hysteresis, or historical-state initialization —
i.e. structural dynamics changes, not a new calibration target. The RV doc pointed the same way.
