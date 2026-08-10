# Why computed and prescribed RV disagree: the prescribed series is a recruitment narrative, not a reproductive volume

**Date:** 2026-08-10
**Follows:** `docs/baltic_computed_rv_divergence_2026-08-10.md` (the measurement), which asked why
the two series show no rank agreement. This answers it.

## The answer is in the file's own README

`data/baltic/reference/baltic_cod_reproductive_volume.README.md` states it plainly:

> **This file (PoC status — literature-informed, not a data product)**
> The values encode the documented **relative interannual pattern**, NOT a calibrated data
> product: high and variable through the MBI-rich 1970s–80s … the long 1983–1993 stagnation
> decline, the large 1993 and 2003 inflow spikes, a moderate 2014 inflow, and **the sustained low
> after 2003 that underlies the eastern-cod recruitment failure**.

So the prescribed series was never a measurement. It is a hand-drawn curve encoding a narrative,
and the narrative's organising feature — "the sustained low that underlies the recruitment
failure" — is the *stock outcome*, not the physical driver. The README also anticipates exactly
the replacement that was attempted, and specifies the gate consumes the new file unchanged.

## The measurement, side by side

Computed = viable column thickness (S ≥ 11 PSU, O₂ ≥ 89.3 mmol m⁻³) summed over the Bornholm
Basin (54.9–56.0 °N, 14.5–17.0 °E), May–August mean, from
`data/baltic_rv/baltic_rv_field_interannual.nc`:

| year | computed | prescribed | |
|---|---|---|---|
| 1993 | 311 | 300 | 1993 MBI |
| 1996 | 286 | 150 | |
| 2000 | 278 | 100 | |
| 2002 | 240 | 90 | pre-inflow minimum in both |
| **2003** | **388** | 240 | 2003 MBI — highest computed value to that point |
| 2008 | 299 | 70 | |
| 2013 | 264 | 50 | |
| **2014** | **362** | 130 | Dec-2014 MBI |
| 2016 | 379 | 70 | |
| 2019 | 383 | 50 | |
| 2020 | 347 | **48** | |
| 2021 | 406 | — | highest in the record |

Spearman ρ = +0.042 (p = 0.83) over the overlap; −0.001 excluding the 1993 spin-up year.

**The prescribed series falls 84% (300 → 48). The computed series has no trend at all** — it
oscillates around ~300 and ends at its record high. That single structural difference accounts for
the absence of correlation; it is not a threshold, season, or units artefact, and no domain choice
repairs it (Bornholm-only +0.04, SD 25+26+28 −0.16, cod_east mask −0.18).

## Both series are defensible — of different things

The computed result is **consistent with the literature**, not at odds with it. Köster et al.
(2005, *ICES JMS* 62:1408–1425, doi:10.1016/j.icesjms.2005.05.004) report that the Bornholm Basin
was the one spawning ground that *continued* to sustain successful egg development while the
Gdańsk Deep and Gotland Basin lost reproductive volume from the late 1980s; MacKenzie, Hinrichsen
& Plikshs (2000, *MEPS* 193:143–156, doi:10.3354/meps193143) find Bornholm has the highest mean RV
and the *lowest* variability of the four sites. A Bornholm-only computation showing sustained,
weakly-varying RV is what that literature predicts.

The prescribed series' decline is therefore not "Bornholm RV falling". It is a whole-stock
composite dominated by the basins that went anoxic, further shaped by the post-2003 recruitment
failure it was drawn to explain.

And the post-2014 divergence is the sharpest case: the reanalysis says the December-2014 MBI
**ventilated the basins** (computed RV rises and stays high through 2021), while the prescribed
series continues falling to its minimum. The eastern stock did keep failing over that period — but
attributing that to reproductive volume is precisely the inference the physics does not support.

## What this means for the model

The gate is labelled "reproductive volume" and is **the dominant control on cod_east** (gate-off
→ 137,302 t, 1.61× over ceiling). What it actually applies is a prescribed recruitment-failure
trajectory. That is defensible as a calibration device and is honestly documented in the README as
a PoC — but it should not be described, in papers or in the config, as a physical
reproductive-volume forcing. The distinction matters because:

* a narrative curve **cannot be projected** — the scenario ambition (spec C1/B2) needs a computed
  quantity, and the computed quantity does not reproduce the narrative;
* swapping in the computed series would remove the post-2003 decline and, with it, most of the
  recruitment suppression holding cod_east inside its envelope. Given cod_east now sits 8.0% above
  its floor with an admissible `ref` band of ~115–161, a flat computed series would need a much
  smaller `ref` to bite at all, and the resulting factor would be near-constant — the exact
  degradation the withdrawn spec's A1 was written to reject.

## Recommendation

1. **Do not swap.** The computed series is the better *physics* and the worse *fit*; adopting it
   would trade a documented calibration device for a physically-derived constant. Keep the
   prescribed series and correct how it is described: it is a recruitment-failure index, not a
   measured RV. A one-line change in `baltic_param-reproduction.csv`'s comment and the README's
   opening sentence.
2. **The scenario track needs a different mechanism.** If cod recruitment must respond to future
   climate, the honest route is a mechanism whose driver the physics can actually supply —
   temperature-dependent stock–recruitment (spec C1, Voss & Quaas 2026) — not a projected RV.
3. **Fix the config defect this uncovered**, independent of everything above: the cod E/W split
   assigned the Bornholm Deep to `cod_west` (`maps/cod_west_spawning.csv`) while the RV gate runs
   on `cod_east` only, and `cod_east_spawning.csv` contains no Bornholm cells at all. The stock's
   principal spawning ground is attached to the wrong stock.
4. **Preserve the computed field.** `data/baltic_rv/baltic_rv_field_interannual.nc` is a genuine,
   literature-consistent product. It is the right input for a *spatial* egg-survival field
   (`reproduction.rv.spatial.*`, implemented but unused) where its basin-level structure is the
   signal, rather than being collapsed to a scalar that must mimic a narrative.

## Method note

The whole investigation — measurement, domain sensitivity, and this explanation — used data
already in the repository and cost under an hour. The one thing that would have made it cheaper is
reading `baltic_cod_reproductive_volume.README.md` before writing the spec: it declares the series'
PoC status in its second heading.
