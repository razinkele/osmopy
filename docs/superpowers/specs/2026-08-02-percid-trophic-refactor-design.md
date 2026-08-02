# Percid trophic refactor — design

**Date:** 2026-08-02
**Status:** design, not started
**Supersedes the lever search in:** `docs/baltic_percid_overshoot_investigation_2026-08-02.md`
**Proposal:** `docs/proposals/2026-08-02-percid-trophic-refactor.md`

## 1. Problem

The committed Baltic config lets both percids forage year-round on the offshore pelagic stocks. That
is ecologically wrong and it is the supply the pikeperch overshoot runs on.

| | final-decade mean | ICES envelope | |
|---|---|---|---|
| pikeperch | 1,453,313 t | 4,000–25,000 t | **~56–58× over** |
| smelt | 680,125 t | 20,000–120,000 t | ~5.7× over |
| perch | 45,382 t | 8,000–50,000 t | in envelope |

### 1.1 Why this is a supply problem, not a parameter problem

Both demand-side levers were measured and both are closed:

* **Diet accessibility**, `herring`+`sprat` cut 10× (0.3 → 0.03): **−25%**. Pikeperch switched onto
  benthos and smelt.
* **Fishing mortality**, swept 0.5 → 4.0: **−10.6%, non-monotonic**, and the envelope count degrades
  monotonically 7/9 → 6/9 → 5/9 → 4/9. At F = 4.0 (~98%/yr removal) every *other* stock rises —
  cod_west +35%, smelt +26%, sprat +20%, herring +17% — releasing the forage base that regrows
  pikeperch straight back.

The stock is **supply-limited, not mortality-limited: it cannot be fished down.**

### 1.2 The mechanism, measured

| | |
|---|---|
| pikeperch adult map | 27 cells |
| herring adult map | 593 cells; all 27 pikeperch cells fall inside it |
| herring distribution weight in pikeperch cells | 4.55% ≈ 118 kt of 2.60 Mt |
| pikeperch consumption of herring | **1.46 Mt/yr ≈ 12× the locally available standing stock** |

Maps are binary presence (0/1, −99 nodata), so herring spreads uniformly over its range, and OSMOSE
redistributes schools across each species' map every timestep. The coastal cells are therefore
**continuously restocked from the basin-wide pool**: a bay-resident predator holds a replenished tap
into the entire Baltic Proper herring stock.

### 1.3 Ecological basis

* **Bay-resident with sub-bay feeding ranges.** Limited home range *within bays* (Christensen et al.,
  2020; Hansson et al., 2019, reviewed in Hall et al., 2022). Stable isotopes resolve feeding-range
  differences between closely located sites in one littoral area with no migration barriers — perch are
  sedentary at sub-bay scale (Ahlbeck Bergendahl et al., 2017).
* **Locally structured stocks.** Pikeperch has "a very local population structure" (Björklund et al.,
  2007, via Olsson et al., 2015). Perch populations <50 km apart are differentiated with 3–5% gene flow
  and show reproductive homing (Hall et al., 2022). Deep water acts as a barrier to gene flow; the
  species is "suitable for local management" (Olsson et al., 2011).
* **No offshore foraging phase.** No support was found for seasonal entry into the Baltic Proper food
  web. Perch's anadromous ecotype migrates into *freshwater* to spawn (Hall et al., 2022) — the
  opposite direction. The error is offshore access at all, not its duration.
* **Smelt is seasonal prey.** Smelt is available to coastal percids only during its winter spawning
  migration into low-salinity water.

## 2. Current coefficients (the edit surface)

`data/baltic/predation-accessibility.csv`, rows = prey, cols = predators, `;`-separated.

| predator | herring | sprat | smelt | stickleback | perch | Mesozoo | Macrozoo | Benthos |
|---|---|---|---|---|---|---|---|---|
| perch | **0.2** | **0.2** | **0.5** | 0.3 | 0.05 | 0.5 | 0.4 | 0.6 |
| pikeperch | **0.3** | **0.3** | **0.6** | 0.3 | 0.15 | 0.3 | 0.3 | 0.4 |

Bold = targeted. `pikeperch` also holds cod_west/cod_east 0.05 and cannibalism 0.05 — **out of scope,
leave alone.**

## 3. Engine constraint

`AccessibilityMatrix` (`osmose/engine/accessibility.py`) is **stage-indexed only** — labels encode age
thresholds (`"smelt < 0.45"`), giving an ontogenetic dimension. **There is no time/season axis**: one
matrix applies at every timestep. `predation.accessibility.dynamic.*` is density-dependent scaling, not
seasonal.

Consequence: **smelt seasonality cannot be expressed today.** Tier 0 must use a time-averaged
surrogate; Tier 1 adds the axis.

## 4. Design

### 4.1 Tier 0 — accessibility corrections (config only, no code)

1. **Zero the offshore pelagic links.** `perch→herring`, `perch→sprat`, `pikeperch→herring`,
   `pikeperch→sprat` → `0`.
2. **Time-average smelt over its availability window.** Multiply by the fraction of the year the
   window occupies. For a window of `W` months: `perch→smelt = 0.5 × W/12`,
   `pikeperch→smelt = 0.6 × W/12`.

   **`W` is the single free parameter and results scale linearly in it.** It must come from local
   spawning phenology, not from a placeholder. `W = 3` (→ 0.125 / 0.15) is the working default and must
   be recorded as an assumption, not a finding.

Leaves percids on stickleback, benthos, zooplankton, and (pikeperch) perch and conspecifics.

### 4.2 Tier 1 — seasonal accessibility (engine feature), conditional

Mirrors the existing per-timestep vector idiom already in this config format
(`fisheries.seasonality.fshN` is a 24-value vector):

```
predation.accessibility.seasonality.enabled;true
predation.accessibility.seasonality.pair0;smelt,pikeperch
predation.accessibility.seasonality.values.pair0;<n_dt values>
```

**Sparse by design** — only declared pairs deviate; everything else is a constant 1.0 multiplier, so
existing configs are untouched and the default path costs nothing.

**Implementation constraint:** the predation kernel is Numba-compiled, so this must resolve to a dense
`(n_dt, n_prey, n_pred)` float array built once at config time. No dict lookup in the hot loop.

**Conditional on Tier 0's outcome.** If the time-averaged surrogate reproduces the intended behaviour,
Tier 1 buys phenological timing rather than a different annual answer, and is scheduled on its own
merits.

### 4.3 Out of scope

**Tier 2 — percid stocks as separate coastal units.** The only change that addresses supply rather than
links, and the only one that makes the target like-for-like: the ICES envelope is a **per-stock** figure
for locally assessed populations while the model carries one aggregated basin-wide pikeperch. High
cost; the cod E/W disaggregation is a cautionary precedent (could not be fitted, remains a flagged
experiment). **Not to be started on the strength of this design.**

## 5. Acceptance criteria

1. **Non-regression (hard).** **≥ 7/9 in envelope** on final-decade means, 50 yr, **≥ 3 seeds**. Below
   7/9 is a fail, *including* a herring breach. Not a trade-off to argue about.
2. **Mechanism demonstrated (hard).** Post-change diet composition, read from the corrected
   `dietMatrix` (#146): percid diets dominated by benthos, zooplankton and stickleback, with
   **herring and sprat absent**. A config edit that does not show up in realised diet has not been
   demonstrated to work. *(This engine has repeatedly shipped switches that silently no-op'd — assume
   nothing from configuration alone.)*
3. **Pikeperch materially closer to envelope**, with the residual factor stated honestly. "10× over
   instead of 56× over" is a result to report, not a success to claim.
4. **Explicitly NOT a criterion:** smelt reaching envelope. Smelt is a separate `in_envelope` failure
   at 5.7× and this change has no mechanism to fix it — claiming credit there would be spurious.

## 6. Risks

**Herring breach is the material one.** Percid predation is a real loss term for herring, which sits at
2.60 Mt against a 3.00 Mt ceiling — 15% headroom. The 0.03 test *already* breached it at 3.05 Mt;
zeroing entirely pushes harder.

If it breaches, the reading is that **herring's mortality budget was implicitly leaning on percid
predation that should not exist**, and needs correcting alongside — not that the percid change was
wrong. Record it as the next finding rather than reverting Tier 0.

**Secondary:** sprat is also released (+13% in the 0.03 test) but sits at 1.06 Mt against a 2.50 Mt
ceiling, so it has room. Stickleback takes on more percid predation and may fall; it is at 80 kt
against a 50 kt floor, so it has less room than it looks.

## 7. References

Ahlbeck Bergendahl, I., Holliland, P. B., & Hansson, S. (2017). Feeding range of age 1+ year Eurasian
perch *Perca fluviatilis* in the Baltic Sea. *Journal of Fish Biology, 90*(5), 2060–2072.
https://doi.org/10.1111/jfb.13285

Hall, M., Koch‐Schmidt, P., & Larsson, P. (2022). Reproductive homing and fine‐scaled genetic
structuring of anadromous Baltic Sea perch (*Perca fluviatilis*). *Fisheries Management and Ecology,
29*(5), 586–596. https://doi.org/10.1111/fme.12542

Olsson, J., Mo, K., & Florin, A.-B. (2011). Genetic population structure of perch *Perca fluviatilis*
along the Swedish coast of the Baltic Sea. *Journal of Fish Biology, 79*(1), 122–137.
https://doi.org/10.1111/j.1095-8649.2011.02998.x

Olsson, J., Tomczak, M. T., & Ojaveer, H. (2015). Temporal development of coastal ecosystems in the
Baltic Sea over the past two decades. *ICES Journal of Marine Science, 72*(9), 2539–2548.
https://doi.org/10.1093/icesjms/fsv143
