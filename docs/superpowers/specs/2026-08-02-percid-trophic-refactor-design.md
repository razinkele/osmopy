# Percid trophic refactor — design

**Date:** 2026-08-02 · **Revised:** 2026-08-03 after adversarial review (11 agents, 8 findings upheld)
**Status:** design, not started
**Evidence:** `docs/baltic_percid_overshoot_investigation_2026-08-02.md`
**Proposal:** `docs/proposals/2026-08-02-percid-trophic-refactor.md`

> **Revision note.** The first draft diagnosed the overshoot as a *spatial supply* problem — a
> bay-resident predator with a continuously replenished tap into the basin-wide herring pool — and
> proposed zeroing percid access to herring and sprat. **Review refuted the diagnosis and the
> remedy**, both reproduced independently before this rewrite (§1.3, §1.4). The supply framing is
> retained below only as a rejected hypothesis, because the numbers that killed it are the ones that
> point at the replacement.

## 1. Problem

| | final-decade mean | ICES envelope | |
|---|---|---|---|
| pikeperch | 1,453,313 t | 4,000–25,000 t | **~56–58× over** |
| smelt | 680,125 t | 20,000–120,000 t | ~5.7× over |
| perch | 45,382 t | 8,000–50,000 t | in envelope |

Perch is the control: same family, same coastal habitat, comparable fishery, and it passes.

### 1.1 Demand-side levers are closed (measured)

* **Diet accessibility**, `herring`+`sprat` cut 10× (0.3 → 0.03): **−25%**; pikeperch switched onto
  benthos and smelt.
* **Fishing mortality**, swept 0.5 → 4.0: **−10.6%, non-monotonic**, envelope count degrading
  7/9 → 6/9 → 5/9 → 4/9. At F = 4.0 every *other* stock rises (cod_west +35%, smelt +26%, sprat +20%,
  herring +17%), releasing the forage base that regrows pikeperch.

Pikeperch cannot be fished down. That much stands.

### 1.2 Bioenergetics are sound

Q/B at equilibrium: pikeperch 2.55, perch 3.41, cod_east 3.44, herring 4.59, sprat 4.77 — all inside
the published range. No conservation violation, no growth-without-eating; pikeperch has the *lowest*
Q/B of any fish. There is not anomalously much eating going on; there is anomalously much pikeperch.

### 1.3 ~~Spatial supply~~ — REJECTED, with the corrected numbers

The first draft claimed pikeperch's 27 cells hold 4.55% of herring (≈118 kt) against 1.46 Mt/yr
consumption — "**12× the locally available standing stock**". **That figure is wrong.** Adult herring
occupies the 593-cell map for only **10 of 24 timesteps** (`baltic_param-movement.csv`, honoured per
`movement.steps` in `osmose/engine/movement_maps.py`):

| steps | herring map | cells | share in pikeperch cells |
|---|---|---|---|
| 0–3, 12–15, 22–23 | herring_adult | 593 | 4.55% |
| 4–11 | herring_spawning | 51 | 3.9–9.8% |
| 16–21 | herring_spawning_autumn | 10 | **50.0%** |

Time-weighted: **16.19% ≈ 421 kt**, so **1.46 Mt/yr ≈ 3.5×**, not 12×. The all-juvenile bound gives
9.3×, so 12× is outside the range achievable under *any* adult/juvenile combination.

The dominant term is autumn, where all adult herring sits on a **10-cell coastal map of which
pikeperch occupies half**. That is **seasonal spawning aggregation into percid cells** — a different
mechanism from "continuous replenishment", with a different fix.

### 1.4 What actually killed the supply hypothesis

Time-weighted herring share, computed identically for both percids:

| | herring | sprat |
|---|---|---|
| pikeperch | 16.19% | 3.73% |
| **perch** | **31.56%** | 7.90% |

**Perch has twice pikeperch's spatial access to herring and is in envelope.** Spatial supply does not
discriminate between the two, so it cannot explain pikeperch specifically. *(Reproduced directly from
`data/baltic/maps/` and `baltic_param-movement.csv` before this rewrite.)*

### 1.5 ~~Predation release~~ — CORRECTED to size-window escape

**Revised 2026-08-03 (round-2 review).** The first rewrite claimed pikeperch is "the least-predated
fish in the system" (accessibility-as-prey 0.60 vs perch 1.05). **That claim is false**, and it failed
the same control test §1.4 used to kill the supply hypothesis — a test this document invented and then
did not apply to its own replacement. Row sums of `predation-accessibility.csv`:

| cod_west | cod_east | flounder | **pikeperch** | perch | herring | sprat | stickleback | smelt |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 0.20 | 0.35 | **0.60** | 1.05 | 1.35 | 1.55 | 1.65 | 2.55 |

Pikeperch is **4th-least** predated. Three fish carry less and all three are in envelope; smelt carries
the **most** (2.55) and is 5.7× over. The axis is contradicted at both ends.

**The real mechanism is the size window, and it is not a coefficient at all.** The predation kernel
applies the size-ratio gate **before** reading accessibility (`osmose/engine/processes/mortality.py`
:412–414 `if ratio < r_min or ratio >= r_max: continue`, with the accessibility read at :416+), so a
coefficient outside the window is never applied.

| | Linf | ratio_min | largest prey it can take |
|---|---|---|---|
| cod_west / cod_east | 110 cm | 3.5 | **31.4 cm** |
| pikeperch (cannibalism) | 90 cm | 2.5 | **36.0 cm** |

**Pikeperch matures at 40.0 cm** (`baltic_param-species.csv`). **No predator in the configuration can
take a pikeperch above 36 cm** — it escapes the entire predator field *before maturity* and accumulates
biomass in a structurally invulnerable adult class. Perch, with Linf 45 cm, spends far more of its life
inside cod's 31.4 cm window.

This reframes `Linf = 90 cm` from a contributing factor to **the mechanism itself**, and it explains
every failed lever: fishing is the only adult removal, and demand-side cuts act on a class that was
never predation-limited.

**Candidate remedies now point elsewhere entirely** — grey seal predation on large pikeperch (real, and
currently absent from the matrix: GreySeal has no pikeperch entry), the cod size-ratio window, or
`Linf` itself. **Tiers A and B below do not act on this mechanism** and are retained only as the record
of what was tried.

## 2. Ecological basis, and its limits

* **Bay residency / small feeding ranges.** Limited home range *within bays* (Christensen et al.,
  2020; Hansson et al., 2019, via Hall et al., 2022); stable isotopes resolve feeding-range differences
  between closely located sites in one littoral area (Ahlbeck Bergendahl et al., 2017). **Scope: perch.
  Do not extend to pikeperch without a pikeperch source.**
* **Locally structured stocks.** Pikeperch "very local population structure" (Björklund et al., 2007,
  via Olsson et al., 2015); perch differentiated at <50 km with 3–5% gene flow and reproductive homing
  (Hall et al., 2022); deep water a barrier to gene flow (Olsson et al., 2011). **These are population-
  genetic results. They bound dispersal and stock identity, not daily foraging range** — the first
  draft used them as evidence about feeding and that inference is withdrawn.
* **Percids DO eat herring inside bays.** Jensen, Hansson & Didrikas (2011) examine YOY herring and
  "one of their major predators, pikeperch" in **Himmerfjärden, a brackish Baltic bay, in summer**,
  with piscivorous targets at **>45 cm** — above this config's 40 cm maturity size. The size window
  already restricts the link to the coastal size class (ratio 2.5–30 → a 60 cm pikeperch takes 2–24 cm
  prey; herring Linf 27 cm). **The link is real and must not be deleted.**
* **Herring ≠ sprat.** Baltic herring (*C. h. membras*) is a coastal spring spawner whose 0- and
  1-group juveniles are bay-resident; sprat is genuinely offshore and deeper-dwelling. The first draft
  treated them as one object.
* **Smelt.** Anadromous form spawning in coastal low-salinity zones and rivers, **in spring after
  ice-out (April–May), with runs of 20–45 days** — not winter, and ~1 month, not 3 (Sendek & Bogdanov,
  2019). It is also among the most abundant fishes in the eastern Gulf of Finland year-round, so
  "available *only* during the run" is too strong.

## 3. Engine constraint

`AccessibilityMatrix` (`osmose/engine/accessibility.py`) is **stage-indexed only** — labels encode age
thresholds (`"smelt < 0.45"`). **There is no time axis.** `predation.accessibility.dynamic.*` is
density-dependent scaling, not seasonal.

Two consequences: smelt seasonality needs a time-averaged surrogate (Tier B) or a new axis (Tier C);
and the Baltic config uses a **single stage** for percids, so an ontogenetic diet shift (juvenile
pikeperch planktivorous → adult piscivorous) cannot currently be expressed either.

## 4. Design — SUPERSEDED by §1.5

> **The tiers below are retained as a record, not as a recommendation.** Round-2 review established
> that Tier A raises coefficients on size classes holding ~5% of pikeperch mass (every predator window
> closes below maturity), and that Tier B's herring coefficient multiplies accessibility by a spatial
> overlap the cell-based kernel already applies — while §1.4 had just rejected that same quantity as
> non-discriminating. A narrower version of Tier B was already measured and failed 7/9 → 5/9. Any
> revision should start from the size window, not from these tiers.

## 4a. Design as originally proposed

### Tier A — test the predation-release hypothesis first (config only)

The primary lever per §1.5, and it must be tested **before** any trophic edit, so the trophic change is
not credited with an effect predation would have produced.

Raise predation on pikeperch toward perch's level, justified by the documented cormorant pathway
(Heikinheimo et al., 2015: 5–30% of total annual mortality for ages 2–4):

* `Cormorant → pikeperch` 0.4 → 0.6 (perch's value)
* `cod_west → pikeperch` 0.1 → 0.15, `cod_east → pikeperch` 0.05 → 0.1 (perch's values)

Total as prey: 0.60 → 0.90. **This is a diagnostic, not a proposed calibration** — it asks whether the
asymmetry in §1.5 has the leverage the hypothesis needs.

### Tier B — trophic corrections (config only)

1. **`sprat` → 0** for both percids (0.2 perch, 0.3 pikeperch). Sprat is genuinely offshore; this is
   the defensible half of the original proposal.
2. **`herring` → scaled, not zeroed.** Retain the documented bay link at the coastal-available
   fraction. Using §1.3's time-weighted overlap as the supply-side scale:
   `pikeperch→herring 0.3 × 0.162 ≈ 0.05`, `perch→herring 0.2 × 0.316 ≈ 0.06`.
   **Reversible and expressible**, unlike exact 0.
3. **`smelt` → time-averaged over a corrected window.** Spring (April–May), run 20–45 days → `W ≈ 1`
   month, not 3: `perch→smelt 0.5 × 1/12 ≈ 0.04`, `pikeperch→smelt 0.6 × 1/12 ≈ 0.05`.
   **Caveat:** smelt is present year-round in some basins, so a pure run-length average understates
   availability. Treat these as a lower bound and state the assumption.

### Tier C — seasonal accessibility (engine), conditional

Sparse per-pair per-timestep multiplier, mirroring the existing `fisheries.seasonality.fshN` 24-value
idiom; resolves to a dense `(n_dt, n_prey, n_pred)` array built at config time (the predation kernel is
Numba-compiled — no dict lookup in the hot loop). Only for the smelt pulse, and only if Tier B shows
the annual mean is insufficient.

### Out of scope

**Percid stocks as separate coastal units.** Also the only change that makes the target like-for-like:
the ICES envelope is a **per-stock** figure while the model carries one aggregated basin-wide
pikeperch. High cost; the cod E/W disaggregation (could not be fitted, remains a flagged experiment) is
the cautionary precedent.

## 5. Acceptance criteria

Applied per tier, measured on final-decade means, 50 yr, **5 seeds** (matching
`scripts/baltic_stability_certify.py`, which produced the 7/9 baseline — a 3-seed run is not comparable
to it).

1. **Non-regression (hard).** **≥ 7/9 in envelope.** A drop below 7/9 is a fail. This is a *floor, not a
   target* — the baseline already passes it, so it cannot be the sole criterion.
2. **Effect demonstrated (hard).** Pikeperch's final-decade mean must fall by **>2× the 5-seed spread**
   of the baseline, so the change is distinguishable from noise. Report the residual factor plainly —
   "10× over instead of 56×" is a result, not a success.
3. **Mechanism demonstrated (hard), and not by restating the config.** For Tier A: pikeperch's
   realised predation mortality must rise as a share of total Z (read from `mortalityRate` output),
   not merely its accessibility coefficient. For Tier B: percid realised diet must shift *and*
   `herring` must remain **present but reduced** — its disappearance would indicate the coefficient is
   being applied as a hard gate rather than a scale, which is a defect, not a pass.
4. **Not a criterion:** smelt reaching envelope (a separate 5.7× failure with no mechanism here);
   pikeperch reaching envelope in one tier.
5. **Collapse guard.** Pikeperch must not fall below its envelope *floor* (4,000 t) on the final-decade
   minimum. Reaching envelope by starvation or collapse dynamics is a fail, not a fix.

## 6. Risks

**Ranked by headroom to the nearest envelope bound, from the 5-seed baseline** — not herring alone, as
the first draft had it:

| species | current | nearest bound | headroom |
|---|---|---|---|
| **cod_east** | 83,122 t | 85,000 upper | **2.3%** |
| **perch** | 45,382 t | 50,000 upper | 9.2% |
| herring | 2,600,112 t | 3,000,000 upper | 15.4% |
| stickleback | 80,159 t | 50,000 lower | 60.2% |

**cod_east is the tightest, not herring.** Tier A raises `cod → pikeperch` accessibility, which feeds
cod — and cod_east has 2.3% of room. Tier B releases herring and sprat from percid predation.

**No escape clause.** The first draft allowed a herring breach to be reclassified as "the next finding".
That is withdrawn: **any species leaving envelope is a fail against criterion 1**, whatever the
explanation. A breach may still be *informative* — it would suggest that species' mortality budget was
leaning on a link that should not exist — but it is recorded as a failed tier, not a passed one.

## 7. References

Ahlbeck Bergendahl, I., Holliland, P. B., & Hansson, S. (2017). Feeding range of age 1+ year Eurasian
perch *Perca fluviatilis* in the Baltic Sea. *Journal of Fish Biology, 90*(5), 2060–2072.
https://doi.org/10.1111/jfb.13285

Hall, M., Koch‐Schmidt, P., & Larsson, P. (2022). Reproductive homing and fine‐scaled genetic
structuring of anadromous Baltic Sea perch (*Perca fluviatilis*). *Fisheries Management and Ecology,
29*(5), 586–596. https://doi.org/10.1111/fme.12542

Heikinheimo, O., et al. (2015). Cited in Jakubavičiūtė, E., Arula, T., & Dainys, J. (2022), *Status and
future perspectives for pikeperch (Sander lucioperca) stocks in Europe*, openRxiv.
https://doi.org/10.1101/2022.12.20.521162

Jensen, O. P., Hansson, S., & Didrikas, T. (2011). Foraging, bioenergetic and predation constraints on
diel vertical migration. *Journal of Fish Biology, 78*(2), 449–465.
https://doi.org/10.1111/j.1095-8649.2010.02855.x

Olsson, J., Mo, K., & Florin, A.-B. (2011). Genetic population structure of perch *Perca fluviatilis*
along the Swedish coast of the Baltic Sea. *Journal of Fish Biology, 79*(1), 122–137.
https://doi.org/10.1111/j.1095-8649.2011.02998.x

Olsson, J., Tomczak, M. T., & Ojaveer, H. (2015). Temporal development of coastal ecosystems in the
Baltic Sea over the past two decades. *ICES Journal of Marine Science, 72*(9), 2539–2548.
https://doi.org/10.1093/icesjms/fsv143

Sendek, D. S., & Bogdanov, D. V. (2019). European smelt *Osmerus eperlanus* in the eastern Gulf of
Finland, Baltic Sea: Stock status and fishery. *Journal of Fish Biology*.
https://doi.org/10.1111/jfb.14009
