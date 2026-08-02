# Percid overshoot — what has been eliminated, and where the anomaly now sits

**Date:** 2026-08-02
**Config:** 9-species Baltic master, `--params current`, Python engine
**Status:** open investigation. This records eliminations and one refuted hypothesis so the next
attempt does not repeat them.

## The target

After the SP-B re-derivation (`a2c3254`) narrowed the genuine failures to three species, the
remaining calibration problem on master is the percid overshoot:

| species | ICES envelope | final-decade mean | over by |
|---|---|---|---|
| pikeperch | 4,000 – 25,000 t | **1,339,138 – 1,461,750 t** | **~56×** |
| smelt | 20,000 – 120,000 t | 676,793 – 688,088 t | ~5.7× |
| perch | 8,000 – 50,000 t | 43,542 – 46,636 t | **in envelope** |

Perch is the useful control: same family, same coastal habitat, comparable fishery, and it passes.

## Eliminated by measurement

* **Missing fishing mortality.** Pikeperch has `fisheries.rate.base.fsh5 = 0.50`, *higher* than
  perch's 0.40.
* **Fishery not wired to the species.** `fishery-catchability.csv` maps
  `pikeperch → coastalpikeperch = 1`. Worth checking given `trawlcodeast` shipped declared-but-mapless
  (#139), but it is sound here.
* **Spatial mismatch between stock and fishery.** 100% of pikeperch's 27 occupied cells overlap the
  fishery distribution; same for perch (62 cells) and smelt (297).
* **Higher resource accessibility than perch.** Perch has *more* (`Mesozoo 0.5 / Macrozoo 0.4 /
  Benthos 0.6`) than pikeperch (`0.3 / 0.3 / 0.4`), and perch is in envelope.

## Refuted: the LTL-subsidy hypothesis

**Hypothesis:** pikeperch is sustained by regrowing resource pools rather than the fish forage base,
so no fish-side lever can bound it.

**It looked strongly confirmed on a 3-year run** — 51.8% Benthos + 23.8% Macrozooplankton, i.e. ~76%
resource-derived. **At equilibrium (years 40–50) it is false:**

| species | final-decade B | resource share of diet | top prey |
|---|---|---|---|
| **pikeperch** | 1,453,313 t | **36.9%** | herring 39%, Benthos 32%, smelt 9% |
| perch | 45,382 t | **42.1%** | Benthos 35%, herring 34%, smelt 17% |
| cod_west | 13,623 t | 48.4% | Benthos 39%, herring 33%, sprat 15% |
| cod_east | 83,122 t | 31.5% | sprat 47%, Benthos 28%, herring 16% |
| smelt | 680,125 t | 100% | Mesozoo 62%, Macrozoo 19%, Benthos 15% |
| herring | 2,600,112 t | 100% | Mesozoo 61%, Macrozoo 20%, Benthos 12% |

Pikeperch's resource share is **below** perch's and cod_west's. It is not unusually subsidised; it is
mid-pack. The 76% figure was a seeding-transient artifact.

**Method note worth keeping:** the 3-year number was measured only because the equilibrium run had not
finished, and it pointed the opposite way from the truth. Diet composition on this config is not
stable until the transient clears — do not characterise it on short runs.

## Where the anomaly sits now

The chain is `Mesozooplankton → herring (2.60 Mt) → pikeperch (1.45 Mt)`.

**Pikeperch stands at 56% of herring's biomass.** cod_east, by contrast, is 3% of it. That ratio is
the thing now requiring explanation, and it carries a thermodynamic problem: a predator at 1.45 Mt
implies prey production far above what a 2.6 Mt herring stock can deliver — yet **herring sits
comfortably inside its own envelope**, so it is not being drawn down the way a predator that large
should draw it down.

Both readings were tested and **both are refuted.** Measured consumption (final decade, annualised):

| predator | B (t) | Q (t/yr) | **Q/B** |
|---|---|---|---|
| pikeperch | 1,453,313 | 3,712,747 | **2.55** |
| perch | 45,382 | 154,920 | 3.41 |
| cod_east | 83,122 | 285,817 | 3.44 |
| cod_west | 13,623 | 46,255 | 3.40 |
| smelt | 680,125 | 1,519,300 | 2.23 |
| herring | 2,600,112 | 11,925,219 | 4.59 |
| sprat | 1,059,941 | 5,056,861 | 4.77 |
| stickleback | 80,159 | 368,876 | 4.60 |

Every value is in the published range — planktivores 4.6–4.8, piscivores 2.5–3.4. **There is no
conservation violation and no growth-without-eating; the engine is internally consistent.** All
predators together remove 1.57 Mt/yr of herring, 60.3% of its standing stock, which a 2.6 Mt stock
sustains at a plausible P/B.

Notably **pikeperch has the LOWEST Q/B of any fish in the system (2.55)** — consistent with a
long-lived (15 yr), large-bodied (Linf 90 cm) predator. It is not eating anomalously much; there is
simply a great deal of it.

So the overshoot is not an engine defect. The model's Baltic is very productive — herring alone
consumes 11.9 Mt/yr of zooplankton — and that productivity propagates up to a predator with almost
nothing eating it.

## Standing structural contrasts (not yet ruled out)

| | Linf | eaten-by (total accessibility) | cells | biomass |
|---|---|---|---|---|
| perch | 45 cm | **1.05** (5 predators) | 62 | 45 kt — IN |
| **pikeperch** | **90 cm** | **0.60** (4 predators) | 27 | 1,453 kt |

Pikeperch is double perch's length (≈8× mass per fish at Linf) and carries the weakest predation
pressure in the food web — nothing but cod (0.05–0.1), cannibalism (0.05) and Cormorant (0.4) eats it.
It is also absent from perch's diet while perch is in *its* diet at 0.15.

`Linf = 90 cm` is at the top of the plausible range for Baltic pikeperch and biomass scales with
roughly the cube of length, so it is worth questioning before concluding the overshoot is unfixable.

## What the literature says about the direction

The model's error is directional, not marginal. Jakubavičiūtė et al. (2022) assessed nine European
pikeperch stocks with Bayesian surplus production models and found three strongly depleted —
**including two in the Baltic Sea** — with biomasses considerably below B<sub>MSY</sub>. The model has
the stock 56× *above* envelope.

Recreational catch is a genuine missing removal: Dainys et al. (2022) showed it can greatly exceed
commercial catch and impede recovery after a commercial closure, and the same holds in the Archipelago
Sea (Heikinheimo et al., 2015). The model's F is commercial-only. But doubling F does not close 56×.

Cormorant predation is real and already represented (`Cormorant → pikeperch = 0.4`); Heikinheimo et al.
(2015) put it at 5–30% of total annual mortality for ages 2–4, so that pathway cannot absorb the gap
either.

Kokkonen et al. (2019) found pikeperch abundance *suppresses* perch recruitment via predation and
competition. The config encodes the predation link (0.15), yet the model has perch in envelope while
pikeperch runs away — the documented asymmetry is inverted.

## Where that leaves the problem

With the LTL subsidy and the engine-defect readings both eliminated, what remains is a **regulation**
problem, and the surviving candidates are all top-down:

1. **Predation on pikeperch is negligible** — total accessibility 0.60, and only cod (0.05–0.1),
   cannibalism (0.05) and Cormorant (0.4) contribute at all. Perch, which passes, carries 1.05.
2. **F = 0.5 is small relative to the productivity** feeding the stock. Recreational catch is a real
   missing removal (see below) but cannot plausibly close 56×.
3. **`Linf = 90 cm`** gives a large biomass-per-recruit, and biomass scales with roughly the cube of
   length.
4. **Its access to the offshore forage base may be unrealistic.** 39% of its diet is herring, at
   accessibility 0.3 — the same as its access to sprat. Real Baltic pikeperch is a coastal
   bay/lagoon predator that would rarely encounter the offshore herring stock. This is the lever the
   prior work labelled "percid diet constraint (×217, grid-unfixable)"; it is worth revisiting now
   that the diagnostic is trustworthy, because the diet share it targets is now measurable.

## Note on the diagnostic itself

Every diet figure above required fixing #146 first: the diet prey axis was labelled with background
species instead of resources, so resource prey were mislabelled or silently discarded and **every
predator reported as 100% fish / 0% resource**. Any percid conclusion drawn from `dietMatrix` before
`e121c6d` is unreliable.

`e121c6d` was itself **incomplete**: it corrected three of the four prey-axis sites and left
`_build_predator_pressure_dataframe` on the old axis, so in-memory `predatorPressure` still dropped
four of six resource groups. The full suite stayed green because nothing asserted on absolute
consumption. It surfaced only as a biologically impossible figure — herring at Q/B 0.12 while
feeding 61% on Mesozooplankton. Fixed, with a Q/B sanity assertion added that catches a dropped prey
block wherever it originates.

## References

Dainys, J., et al. (2022). Cited in Jakubavičiūtė et al. (2022).

Heikinheimo, O., et al. (2015). Cited in Jakubavičiūtė et al. (2022).

Jakubavičiūtė, E., Arula, T., & Dainys, J. (2022). Status and future perspectives for pikeperch
(*Sander lucioperca*) stocks in Europe. *openRxiv*. https://doi.org/10.1101/2022.12.20.521162

Kokkonen, E., Heikinheimo, O., & Pekcan‐Hekim, Z. (2019). Effects of water temperature and pikeperch
(*Sander lucioperca*) abundance on the stock–recruitment relationship of Eurasian perch (*Perca
fluviatilis*) in the northern Baltic Sea. *Hydrobiologia, 841*(1), 79–94.
https://doi.org/10.1007/s10750-019-04008-z

Olin, M., Heikinheimo, O., & Lehtonen, T. K. (2023). Long‐term monitoring of pikeperch (*Sander
lucioperca*) populations under increasing temperatures and predator abundances in the Finnish coastal
waters of the Baltic Sea. *Ecology of Freshwater Fish, 32*(4), 750–764.
https://doi.org/10.1111/eff.12721
