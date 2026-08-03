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
4. ~~**Its access to the offshore forage base may be unrealistic.**~~ **Tested and rejected** — see
   below. It corroborates the prior "percid diet constraint (×217, grid-unfixable)" verdict, now on a
   trustworthy diagnostic.

## Tested and rejected: closing the offshore forage pathway

Sensitivity test, not a proposed calibration: pikeperch accessibility to **herring and sprat** cut
10× (0.3 → 0.03), i.e. roughly half its diet closed off. 50 yr, seed 42.

| species | baseline | access 0.03 | ratio | envelope |
|---|---|---|---|---|
| **pikeperch** | 1,453,313 | 1,084,623 | **0.746** | still **43.4× over** |
| herring | 2,600,112 | 3,052,372 | 1.174 | **breached** (3.05 vs 3.00 Mt ceiling) |
| cod_east | 83,122 | 87,212 | 1.049 | **breached** (87.2 vs 85.0 kt) |
| sprat | 1,059,941 | 1,196,715 | 1.129 | in |
| smelt | 680,125 | 651,891 | 0.958 | still 5.4× over |
| perch | 45,382 | 41,869 | 0.923 | in |
| stickleback | 80,159 | 65,037 | 0.811 | in |

**A 10× cut to half its diet bought a 25% biomass reduction** — 56× over becomes 43× over. Pikeperch
routed around the closed pathway onto benthos and smelt, which is what an opportunistic predator with
a broad prey base and no predators does.

**It also fails non-regression: 7/9 → 5/9.** Releasing herring from its dominant predator pushed
herring through its own ceiling, and cod_east followed. This is the failure moving, not closing —
precisely the outcome the Phase 0 non-regression criterion exists to reject.

*Limits:* single seed; the herring (+1.7%) and cod_east (+2.6%) breaches are marginal and within the
range where seed noise matters (0.3–2.3% observed elsewhere on this config). The **primary** result —
pikeperch remaining 43× over — is far too large to be noise and is what decides the lever.

**Read-through:** the binding constraint is not what pikeperch eats. With diet closed off it still
sustains 1.08 Mt, so the constraint is that **nothing removes it** — total predation accessibility
0.60 against perch's 1.05, plus F = 0.5. The remaining candidates are levers 1–3 (predation on
pikeperch, F including the missing recreational component, and `Linf = 90 cm`), which act on removal
and biomass-per-recruit rather than on intake.

## The structural defect: a bay-resident predator with a tap into the basin-wide forage pool

Raised by the user 2026-08-02 and **confirmed against the literature and the config**. This supersedes
the lever-by-lever search above: none of those levers can work, because the supply side is wrong.

### What the literature says

**Percids are bay-resident with small feeding ranges — the association is even stronger than
"coastal".** Site-fidelity and home-range work in the Baltic indicates a limited home range *within
bays* (Christensen et al., 2020; Hansson et al., 2019, as reviewed in Hall et al., 2022). Stable-isotope
work resolves feeding-range differences *between closely located sites inside one littoral area with no
migration barriers*, i.e. individual perch are sedentary at sub-bay scale (Ahlbeck Bergendahl et al.,
2017).

**Stocks are separate, at fine spatial scale.** Pikeperch has "a very local population structure"
(Björklund et al., 2007, as cited in Olsson et al., 2015). Anadromous Baltic perch populations <50 km
apart are genetically differentiated with only 3–5% gene flow, and show reproductive homing (Hall et
al., 2022). Along the Swedish coast, perch shows isolation by distance and departure from panmixia,
with differentiation between the central Baltic and the Gulf of Bothnia — and critically **stretches of
deep water act as barriers to gene flow**, the species being explicitly "suitable for local management"
(Olsson et al., 2011). Olsson et al. (2015) accordingly treat Gulf of Riga SW/NE, Gulf of Finland E/W,
Archipelago Sea, Holmön (Bothnian Bay), and the Curonian and Vistula Lagoons as **separate coastal
ecosystems**.

**On seasonal entry to the Baltic Proper food web:** the evidence found points the *other* way — toward
percids largely not entering the offshore food web at all, rather than entering it seasonally. Perch's
anadromous ecotype migrates into **freshwater** to spawn, then returns to coastal brackish foraging
areas (Hall et al., 2022); that is the opposite direction from the open sea. So the model error is
larger than "year-round instead of summer-only": it is offshore access at all.

### What the config does instead

| | value |
|---|---|
| pikeperch adult map | **27 cells** |
| herring adult map | **593 cells**, and all 27 pikeperch cells fall inside it |
| herring distribution weight inside pikeperch cells | **4.55%** ≈ 118 kt of the 2.60 Mt |
| pikeperch consumption of herring | **1.46 Mt/yr** |

Maps are binary presence (0/1, with −99 nodata), so herring is spread uniformly over its 593 cells.
Pikeperch therefore sees ~118 kt of herring locally at any instant, yet consumes **1.46 Mt/yr — about
12× the locally available standing stock every year.** OSMOSE redistributes schools across each
species' map every timestep, so those coastal cells are *continuously restocked from the basin-wide
pool*.

**The model gives a bay-resident predator a continuously replenished tap into the entire Baltic Proper
herring stock.** That is the supply the overshoot runs on.

### Why every lever tested was weak

This explains the pattern rather than adding to it:

* **Diet accessibility (10× cut on herring+sprat): −25% only.** Throttling the coefficient leaves the
  replenishment intact, so pikeperch switched to benthos and smelt.
* **Fishing mortality: inert beyond F = 1.0.** Full sweep, 50 yr, seed 42:

  | F | pikeperch | over 25 kt | envelope |
  |---|---|---|---|
  | 0.5 (base) | 1,453,313 | 58.1× | **7/9** |
  | 1.0 | 1,284,118 | 51.4× | 6/9 |
  | 2.0 | 1,333,002 | 53.3× | 5/9 |
  | 4.0 | 1,298,663 | 51.9× | **4/9** |

  An **8× increase in F** — to a physically absurd ~98%/yr removal, far beyond any real fishery
  including the missing recreational component — moves pikeperch by **10.6%**, and non-monotonically
  (F = 2.0 sits *above* F = 1.0). The envelope count meanwhile degrades monotonically, 7 → 6 → 5 → 4.

  The mechanism is visible in the F = 4.0 column: **every other stock rises** — cod_west +35%, smelt
  +26%, sprat +20%, herring +17%, flounder +16%, perch +15%. Fishing pikeperch down releases the
  forage base, which regrows pikeperch straight back. The compensation loop closes because the forage
  supply is replenished from the basin pool regardless of local depletion.

Both act on the *demand* side. The defect is on the *supply* side, so neither can close a 56× gap —
**the stock is supply-limited, not mortality-limited, and cannot be fished down.**

### Implication for the fix, and for the envelope comparison

The representation needs percid stocks confined to their local prey base — separate coastal units with
their own forage, rather than one basin-wide pikeperch drawing on a well-mixed pelagic pool. Note also
that the ICES envelope (4,000–25,000 t) is a **per-stock** figure for locally-assessed populations,
while the model carries a single aggregated pikeperch; the comparison is not like-for-like even before
the supply problem is addressed.

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

Ahlbeck Bergendahl, I., Holliland, P. B., & Hansson, S. (2017). Feeding range of age 1+ year Eurasian
perch *Perca fluviatilis* in the Baltic Sea. *Journal of Fish Biology, 90*(5), 2060–2072.
https://doi.org/10.1111/jfb.13285

Björklund, M., et al. (2007). Cited in Olsson et al. (2015).

Christensen, E. A. F., et al. (2020); Hansson, S., et al. (2019). Cited in Hall et al. (2022).

Hall, M., Koch‐Schmidt, P., & Larsson, P. (2022). Reproductive homing and fine‐scaled genetic
structuring of anadromous Baltic Sea perch (*Perca fluviatilis*). *Fisheries Management and Ecology,
29*(5), 586–596. https://doi.org/10.1111/fme.12542

Olsson, J., Mo, K., & Florin, A.-B. (2011). Genetic population structure of perch *Perca fluviatilis*
along the Swedish coast of the Baltic Sea. *Journal of Fish Biology, 79*(1), 122–137.
https://doi.org/10.1111/j.1095-8649.2011.02998.x

Olsson, J., Tomczak, M. T., & Ojaveer, H. (2015). Temporal development of coastal ecosystems in the
Baltic Sea over the past two decades. *ICES Journal of Marine Science, 72*(9), 2539–2548.
https://doi.org/10.1093/icesjms/fsv143

Olin, M., Heikinheimo, O., & Lehtonen, T. K. (2023). Long‐term monitoring of pikeperch (*Sander
lucioperca*) populations under increasing temperatures and predator abundances in the Finnish coastal
waters of the Baltic Sea. *Ecology of Freshwater Fish, 32*(4), 750–764.
https://doi.org/10.1111/eff.12721


---

## 2026-08-03 — third diagnosis: escape from the predator size window

Both hypotheses above are withdrawn (spatial supply, §1.4 of the design spec; predation release,
round-2 review). The third position, and the first with a mechanism in the engine rather than in a
coefficient:

**The predation kernel applies the size-ratio gate BEFORE reading accessibility**
(`osmose/engine/processes/mortality.py`:412–414, accessibility at :416), so a coefficient outside the
size window is never applied at all.

| predator | length | ratio_min | largest prey it can take | pikeperch access |
|---|---|---|---|---|
| cod_west / cod_east | 110 cm | 3.5 | 31.4 cm | 0.1 / 0.05 |
| Cormorant | 70–85 cm | 2.5 | **34.0 cm** | 0.4 |
| pikeperch (cannibalism) | 90 cm | 2.5 | 36.0 cm | 0.05 |
| **GreySeal** | **110–170 cm** | **3** | **56.7 cm** | **no matrix column** |

**Pikeperch matures at 40.0 cm** and grows to 90 cm. Every predator holding a pikeperch accessibility
entry closes its window *below maturity*. Perch (Linf 45 cm) stays inside cod's 31.4 cm window for far
more of its life. This makes `Linf = 90` the mechanism rather than a modifier, and explains why every
lever failed: fishing is the only adult removal, and demand-side cuts act on a class that was never
predation-limited.

### Unresolved: what a missing predator column actually means

The accessibility matrix is **16 columns — 9 focal + 6 resources + Cormorant. GreySeal is absent**,
though it is declared as background sp15 with a size window reaching 56.7 cm.

`AccessibilityMatrix.get_index` returns **−1** for a species not in the matrix, and the kernel does:

```
access_coeff = 1.0
if ... and p_acc >= 0 and q_acc >= 0:
    access_coeff = access_matrix[q_acc, p_acc]
    if access_coeff <= 0: continue
```

A −1 index skips the block, leaving `access_coeff = 1.0` — **a missing predator column means maximum
access, not zero**, and the `<= 0` skip never fires. Whether this school-vs-school path governs
background predators (which are biomass pools rather than schools) is **not yet established**, and it
decides the sign of the GreySeal question entirely:

* if background predation takes this path, GreySeal already has *full* access to adult pikeperch, and
  the size-window story needs re-examining;
* if it takes another path, GreySeal may not predate pikeperch at all, and adding it is the candidate
  remedy.

**An intervention test was attempted and is invalid:** setting `df.loc["pikeperch", "GreySeal"]` adds a
17th column and changes the matrix dimensions rather than editing a cell. Resolve the path question
before retrying.

### Resolved: GreySeal already has full access, and is ~30× too small to matter

Background species **are** schools carrying an `is_background` flag. The four `is_background` guards in
`mortality.py` (:94, :138, :190, :290) exempt them from *suffering* starvation, additional, fishing and
foraging mortality — **none excludes them from acting as predators.** So GreySeal predates through the
ordinary school path with `p_acc = −1`, the `p_acc >= 0` guard fails, and `access_coeff` stays at its
**1.0** initialiser.

**GreySeal therefore already has full accessibility to adult pikeperch** within its 56.7 cm window. The
config comment confirms this was intended: *"herring, sprat, medium cod, flounder, small-medium
pikeperch"*.

It cannot matter at the configured scale. GreySeal is **4,500 t** standing biomass (30,000 individuals
× 150 kg) with `predation.ingestion.rate.max = 13.0`, i.e. **~58,500 t/yr of total consumption across
all prey** — herring (2.60 Mt), sprat (1.06 Mt), cod, flounder and pikeperch together. Even if seals
ate nothing but pikeperch, that is ~4% of a 1.45 Mt stock per year; the realistic share is a small
fraction of that.

### The diagnosis, complete

Adult pikeperch is effectively unpredated, by two independent routes:

* **cod, cormorant and cannibalism cannot reach it by size** — windows close at 31.4 / 34.0 / 36.0 cm
  against 40 cm maturity;
* **grey seal can reach it (to 56.7 cm) but is ~30× too small in biomass** to exert control, and
  **nothing at all can take a pikeperch above 56.7 cm**, against Linf 90 cm and a 15-year lifespan.

Biomass accumulates in a size class no predator in the configuration can touch. This is why fishing was
the only lever with any effect (it is the sole adult removal) and why every demand-side cut failed
(they act on a class that was never predation-limited).

### Candidate levers, in order of defensibility

1. **`Linf = 90 cm`** — at the top of the plausible range for Baltic pikeperch and the parameter that
   places the stock beyond every predator. Biomass scales with roughly the cube of length.
2. **Cod `sizeratio.min = 3.5`** — a 110 cm cod restricted to ≤31.4 cm prey is arguably too tight for
   Baltic cod; loosening it re-opens a predation route on sub-adult pikeperch.
3. **GreySeal biomass** — realistic at 30,000 individuals; not a free parameter.

### Status

The size-window diagnosis is now specified end to end and each number is verified. It remains the
**third** position in this investigation, and the two it replaced were each supported by verified
numbers too, right up to the control test that killed them. **It has not been subjected to an
intervention test**, which is what would settle it.


---

## 2026-08-03 (later) — the size-window diagnosis fails its intervention test

Six levers have now been measured. All fail.

| intervention | pikeperch | vs baseline | envelope |
|---|---|---|---|
| baseline | 1,453,313 | — | 7/9 |
| diet accessibility, herring+sprat 10x cut | 1,084,623 | −25.4% | 5/9 |
| F swept 0.5 → 4.0 | 1,298,663 | −10.6% | 4/9 |
| gear: size-selective l50 = 40 cm | 1,345,161 | −7.4% | 7/9 |
| gear: size-selective l50 = 45 cm | 1,565,986 | +7.8% | 7/9 |
| **cod size window 3.5 → 2.5 (prey ≤44 cm)** | **1,502,194** | **+3.4%** | 7/9 |
| l50 = 45 + cod 2.5 | 1,521,761 | +4.7% | 6/9 |
| l50 = 45 + Linf 70 | 1,435,425 | −1.2% | 6/9 |
| l50 = 45 + cod 2.5 + Linf 70 | 1,521,579 | +4.7% | 6/9 |

**The cod-window arm is the decisive one and it failed.** Opening cod's feeding window from a 31.4 cm
ceiling to 44 cm crosses pikeperch's 40 cm maturity, giving a predator access to the adult class for the
first time. Biomass moved **the wrong way**.

### Why: the predator field is too small by an order of magnitude

Arithmetic that should have preceded the test. Cod totals **~97 kt** (cod_west 13.6 + cod_east 83.1)
against pikeperch's **1.45 Mt** — a stock 15× its own size. Cod's entire annual consumption is ~330 kt
across all prey; even at 100% pikeperch that is 23% of the stock, realistically a few percent. GreySeal,
the only other predator reaching adult sizes, consumes ~58.5 kt/yr across all prey.

**No predator in this configuration has the biomass to regulate pikeperch, whatever the size window
permits.** The size-window finding is therefore *necessary but not sufficient*: adults are genuinely
unreachable, and making them reachable changes nothing.

### Status: four diagnoses, four failures

Spatial supply (refuted by the perch control), predation release (refuted by the accessibility
ordering), size-window escape (refuted by this intervention test), and the gear-selectivity hypothesis
(refuted, worse at realistic landing sizes). Each was supported by verified numbers up to the point it
was tested.

**Recommendation: stop testing removal-side levers.** Six have produced between −25% and +8% against a
58× gap. The remaining untested candidates are on the production side —

* **recruitment**: pikeperch's stock–recruitment parameterisation, which no test so far has touched;
* **the target itself**: the ICES envelope is per-stock for locally assessed populations while the model
  carries one aggregated basin-wide stock. Even summing across the ~9 recognised Baltic coastal stocks
  gives at most ~225 kt, still 6.4× below the model, so this cannot explain the gap alone — but it means
  the reference point is wrong by roughly an order of magnitude before any dynamics are considered.

`cod_east` breached in every `l50 = 45` arm and is confirmed as the fragile species (2.3% headroom).
