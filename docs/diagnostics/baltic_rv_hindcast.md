# Baltic interannual reproductive-volume cod hindcast — results (Phase 3 A/B)

**Question.** Does forcing the Baltic cod model with a *real year-by-year* (1993–2021)
reproductive-volume (RV) field move modeled cod SSB toward observed ICES SSB, versus the
stationary **climatology** of that same field? The two enabled arms share one forced
`RV_ref` (36.15), so the A/B isolates *temporal structure*, not a scaling difference.

**Verdict (headline).** **No meaningful skill gain.** Over 5 seeds the interannual arm beats
the climatology arm by a **skill delta of +0.0042 ± 0.0015** (all 5 seeds non-nan, all
marginally positive) — consistently signed but negligible in magnitude. Interannual RV
*timing* is not a controlling driver of modeled eastern-Baltic cod at this scale. This
corroborates the weak Phase-0 offline correlation (RV↔recruitment best-lag corr −0.04).

## Design and framing

- **Intrinsic boom-bust is the null.** The deployed Baltic config is robustly monostable and
  its cod undergoes an intrinsic post-peak decline (larval-M / fishing / food-web structure).
  The A/B does not ask "does the model match observations" — it asks "does adding *interannual*
  RV structure beat the *stationary* climatology," holding the shared `RV_ref` fixed.
- **Low correlation power — read the delta, not the absolute skill.** Over the usable window
  (sim-yr 6–15 ≈ **1999–2008**) observed cod SSB *rises* (the post-2003 recovery, 52→134 kt)
  while the model's intrinsic cod *declines* in every arm. So the raw correlation with observed
  is strongly negative for *all* arms by construction; only the **delta** (interannual − climatology)
  is interpretable, and it is ~0.
- **Metric distinction.** Phase 0 uses a bottom-slice areal RV *fraction*; Phase 3 uses the
  full-column RV *thickness* field. They are related but not identical — qualitative agreement
  (both weak), do not over-equate the numbers. Phase 0's observed cod is data-limited post-~2014.
- **Jensen caveat.** The egg-survival clip is nonlinear, so the arms do **not** have identical
  mean suppression; see the systematic interannual−climatology offset below.

## Cod SSB trajectories, window 1999–2008 (5-seed mean, tonnes-equivalent proxy)

| year | observed ssb_t | off (no RV) | clim | inter |
|---|---|---|---|---|
| 1999 | 51 971 | 3302 | 3352 | 3366 |
| 2000 | 61 608 | 2786 | 2581 | 2587 |
| 2001 | 75 403 | 2552 | 2009 | 1997 |
| 2002 | 85 029 | 2475 | 1586 | 1557 |
| 2003 | 86 704 | 2403 | 1261 | 1218 |
| 2004 | 75 587 | 2237 | 999 | 946 |
| 2005 | 94 282 | 1967 | 784 | 731 |
| 2006 | 94 986 | 1646 | 608 | 561 |
| 2007 | 93 791 | 1316 | 467 | 429 |
| 2008 | 134 284 | 1019 | 356 | 327 |

(The model proxy is on the engine's own SSB scale, not calibrated to absolute ICES tonnes; the
comparison is about *shape/correlation*, not level.)

## Skill delta (interannual − climatology), per seed

| seed | 0 | 1 | 2 | 3 | 4 | mean ± sd |
|---|---|---|---|---|---|---|
| skill Δ | 0.003 | 0.004 | 0.006 | 0.006 | 0.002 | **+0.0042 ± 0.0015** (5/5 non-nan) |

Absolute correlations with observed SSB over the window (5-seed-mean trajectories):
`corr(off) = −0.909`, `corr(clim) = −0.843`, `corr(inter) = −0.839`. Enabling the RV mechanism
(off→clim) nudges the correlation from −0.909 to −0.843; interannual over climatology adds a
further +0.004. Both moves are relative to a wrong-signed baseline, so neither is meaningful
absolute skill.

## Two secondary, honest observations

1. **The RV egg-survival gate suppresses cod** — by the end of the window, `clim` and `inter`
   cod are ~35% / ~32% of the `off` arm. The mechanism does what it should (reproductive
   limitation where RV is low); it just does not help track the *observed recovery*.
2. **Interannual sits systematically *below* climatology** (interannual − climatology =
   +14, +6, −12, −29, −43, −53, −53, −47, −38, −29 over the window; ratio → 0.92). This is the
   **Jensen effect**: the nonlinear survival clip on a *variable* input yields net lower mean
   survival than on the *smoothed* climatology.

## 2003 Major Baltic Inflow (MBI) feature check

If the interannual field captured the large 2003 MBI (a salt/O₂ pulse that should *enlarge* the
reproductive volume in 2003–2004), the `inter` arm should rise *above* `clim` around 2003–2005.
It does not: interannual − climatology in 2003/2004/2005 = **−43 / −53 / −53** — interannual
stays *below* climatology. **No distinct MBI cod feature is recoverable**; the intrinsic decline
and the Jensen suppression dominate any single-year RV pulse. (The 2016 MBI is beyond the usable
window, after modeled cod has collapsed.)

## Conclusion

The interannual reproductive-volume field is a scientifically sound construction (validated:
the shipped climatology is bit-identical to this field's 29-year climatological mean, so the A/B
is a clean temporal-structure isolation), and the harness scores like-for-like maturity-based SSB.
Within that clean test, **interannual RV timing does not improve the eastern-Baltic cod hindcast
over a stationary climatology** (skill Δ ≈ +0.004). The signal is dominated by the model's
intrinsic dynamics. This is an informative negative result: it argues against interannual RV
*forcing timing* as a lever for cod dynamics in this configuration, and points future work toward
recruitment-level mechanisms (depensation/Allee, fishing hysteresis) rather than RV forcing.

*Run:* `scripts/_run_rv_hindcast.py` → `run_hindcast(seeds=(0,1,2,3,4))`, 3 arms × 29 yr, shared
`RV_ref=36.15`. Emergent — not a CI gate. Deployed `data/baltic/` unchanged (field lives in
`data/baltic_rv/`).
