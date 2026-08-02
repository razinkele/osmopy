# Refactor proposal — percid trophic links: no offshore pelagics, seasonal smelt

**Date:** 2026-08-02
**Motivation:** Baltic percids are bay-resident with sub-bay feeding ranges and locally structured
stocks; they do not forage on the offshore pelagic stocks. Smelt is available to them only during its
winter spawning migration into low-salinity coastal waters. The config currently gives both percids
year-round access to herring and sprat, which is the supply that the ~56× pikeperch overshoot runs on
(`docs/baltic_percid_overshoot_investigation_2026-08-02.md`).

## Current coefficients

| predator | herring | sprat | smelt | stickleback | perch | Mesozoo | Macrozoo | Benthos |
|---|---|---|---|---|---|---|---|---|
| perch | **0.2** | **0.2** | **0.5** | 0.3 | 0.05 | 0.5 | 0.4 | 0.6 |
| pikeperch | **0.3** | **0.3** | **0.6** | 0.3 | 0.15 | 0.3 | 0.3 | 0.4 |

Bold = the links this proposal targets. (`pikeperch` also has cod_west/cod_east 0.05 and cannibalism
0.05, left alone.)

## Engine constraint that shapes the design

`AccessibilityMatrix` is **stage-indexed** — labels encode age thresholds (`"smelt < 0.45"`), giving an
ontogenetic dimension only. There is **no time/season axis**: one matrix applies to every timestep.
`predation.accessibility.dynamic.*` is density-dependent scaling, not seasonal. So the smelt
seasonality cannot be expressed directly today, and Tier 0 must use a time-averaged surrogate.

---

## Tier 0 — accessibility corrections (CSV only, no code)

Immediately testable, no engine change.

1. **Cut the offshore pelagic links to zero.**
   `perch→herring 0.2 → 0`, `perch→sprat 0.2 → 0`, `pikeperch→herring 0.3 → 0`,
   `pikeperch→sprat 0.3 → 0`.

2. **Scale smelt to its seasonal availability window.** Smelt is accessible to coastal percids only
   during the winter/early-spring spawning migration into low-salinity water. As a time-averaged
   surrogate, multiply by the fraction of the year that window occupies. For a ~3-month window
   (25%): `perch→smelt 0.5 → 0.125`, `pikeperch→smelt 0.6 → 0.15`.
   **Set the window length from the local spawning phenology** rather than taking 3 months as given;
   it is the one free parameter here and the result scales linearly in it.

This leaves percids feeding on stickleback, benthos, zooplankton, and (pikeperch) perch and
conspecifics — a coastal percid diet.

### Expected outcome, and the risk that decides whether Tier 0 is enough

A weaker version of change 1 alone has already been measured: cutting `herring`/`sprat` to 0.03 (10×,
not zero) gave **−25%** on pikeperch, because it switched onto benthos and smelt. Tier 0 closes both
of those escape routes — smelt is throttled 4×, and benthos is a **per-cell resource with local
regrowth**, so unlike herring it is *not* replenished from a basin-wide pool. That is the reason to
expect Tier 0 to bind where the earlier test did not. It is a reason to expect, not a result: measure it.

**The material risk is herring.** Percid predation is currently a real loss term for herring, which
sits at 2.60 Mt against a 3.00 Mt ceiling — only 15% headroom. The 0.03 test *already* pushed herring
to 3.05 Mt, a breach. Zeroing the link entirely will push harder. **Tier 0 may well trade the pikeperch
failure for a herring failure**, which by the non-regression standard is not a fix. If that happens,
the honest conclusion is that herring's own mortality budget was implicitly relying on percid predation
that should not exist, and that needs correcting alongside — not that the percid change was wrong.

### Acceptance criteria

* Non-regression: **≥ 7/9 in envelope** on the final-decade mean, 50 yr, ≥ 3 seeds. A drop below 7/9 is
  a fail, including a herring breach.
* Pikeperch materially closer to its envelope — and state the residual factor honestly rather than
  claiming success at, say, 10× over.
* Post-change diet composition checked with the corrected `dietMatrix` (#146): percid diets should be
  dominated by benthos, zooplankton and stickleback, with herring/sprat absent.

---

## Tier 1 — seasonal accessibility (engine feature)

Makes the smelt representation correct rather than time-averaged, and is reusable for any seasonal
predator–prey overlap.

**Design, mirroring the existing fishing-seasonality pattern** (`fisheries.seasonality.fshN` is already
a 24-value per-timestep vector, so the idiom exists in this config format):

```
predation.accessibility.seasonality.enabled;true
predation.accessibility.seasonality.pair0;smelt,pikeperch
predation.accessibility.seasonality.values.pair0;0.9;0.9;0.9;0.6;0.1;0;0;0;0;0;0;0.3; ... (n_dt values)
```

Sparse by design: only declared pairs deviate, everything else stays at a constant 1.0 multiplier, so
existing configs are unaffected and the default path costs nothing.

Implementation touches the accessibility lookup in the predation kernel — a per-pair multiplier indexed
by `step % n_dt_per_year`. Note the kernel is Numba-compiled, so this wants a dense
`(n_dt, n_prey, n_pred)` float array built once at config time rather than a dict lookup in the hot
loop.

**Do Tier 0 first.** If the time-averaged surrogate already reproduces the intended behaviour, Tier 1
buys phenological realism (right seasonal timing of the percid–smelt interaction) rather than a
different annual answer, and can be scheduled on its own merits.

---

## Tier 2 — percid stocks as separate coastal units (structural)

The full fix implied by the literature, and the only one that addresses the *supply* rather than the
*links*.

Percids show local population structure (Björklund et al., 2007), fine-scale genetic differentiation
with 3–5% gene flow between populations <50 km apart and reproductive homing (Hall et al., 2022),
isolation by distance with **deep water acting as a barrier to gene flow**, and are explicitly
"suitable for local management" (Olsson et al., 2011). Olsson et al. (2015) treat Gulf of Riga SW/NE,
Gulf of Finland E/W, Archipelago Sea, Holmön (Bothnian Bay) and the Curonian/Vistula Lagoons as
**separate coastal ecosystems**.

This follows the cod E/W disaggregation recipe (`docs/superpowers/specs/2026-07-24-baltic-stock-disaggregation-design.md`)
— append per-basin percid stocks, per-stock maps, per-stock fisheries, name-labelled accessibility
rows/columns. It also fixes a comparison error that persists through Tiers 0–1: **the ICES envelope
(4,000–25,000 t) is a per-stock figure for locally assessed populations, while the model carries one
aggregated basin-wide pikeperch.** Until stocks are separated, the target itself is not like-for-like.

Cost is high and the cod E/W precedent is cautionary — that disaggregation could not be fitted and
remains a flagged experiment. Do not start Tier 2 on the strength of this proposal alone.

---

## Recommended sequence

1. **Tier 0**, measured against the acceptance criteria above, ≥3 seeds. Cheap, reversible, and it
   tests the supply-side diagnosis directly.
2. If herring breaches, treat that as the next finding — herring's mortality budget was leaning on
   percid predation that should not exist — rather than reverting Tier 0.
3. **Tier 1** only if the seasonal timing (not just the annual mean) turns out to matter.
4. **Tier 2** as a separate, explicitly scoped piece of work.

## References

Björklund, M., et al. (2007). Cited in Olsson et al. (2015).

Hall, M., Koch‐Schmidt, P., & Larsson, P. (2022). Reproductive homing and fine‐scaled genetic
structuring of anadromous Baltic Sea perch (*Perca fluviatilis*). *Fisheries Management and Ecology,
29*(5), 586–596. https://doi.org/10.1111/fme.12542

Olsson, J., Mo, K., & Florin, A.-B. (2011). Genetic population structure of perch *Perca fluviatilis*
along the Swedish coast of the Baltic Sea. *Journal of Fish Biology, 79*(1), 122–137.
https://doi.org/10.1111/j.1095-8649.2011.02998.x

Olsson, J., Tomczak, M. T., & Ojaveer, H. (2015). Temporal development of coastal ecosystems in the
Baltic Sea over the past two decades. *ICES Journal of Marine Science, 72*(9), 2539–2548.
https://doi.org/10.1093/icesjms/fsv143
