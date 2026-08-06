# Re-deriving the pikeperch and smelt biomass targets

**Date:** 2026-08-06
**Scope:** `data/baltic/reference/biomass_targets.csv`, rows `pikeperch` and `smelt`
**Prompted by:** those two being the only failing species in Baltic certification, and both carrying the
file's lowest confidence weights (0.2 and 0.3).

## 1. What the targets are, and are not

**ICES does not assess any of them.** `list_stocks` for 2023, area `27.2[0-9]`, returns 13 stocks — cod
(27.22-24 and 27.24-32), herring (three), sprat, plaice (two), dab, sole, brill, ray. **No pikeperch, no
perch, no smelt, no stickleback.** This is consistent with Jakubavičiūtė et al. (2022), who had to apply
Bayesian surplus production models to standardised CPUE precisely because formalised pikeperch
assessments "remain rare".

So "ICES envelope" — the phrase used throughout this project's certification notes and in
`baltic_stability_certify.py`'s own comment — is **wrong for these four species**. The file states the
real provenance: *"Literature estimate for coastal Baltic"*, weight 0.2, note *"Concentrated in
estuaries/lagoons; coarse grid under-resolves"*.

## 2. Correction to an earlier claim of mine

Earlier in this investigation I proposed that the target might be a **per-stock** figure wrongly applied
to an aggregated basin-wide model stock, and that this could account for an order of magnitude of the
58× pikeperch overshoot.

**That was wrong, and the file says so in its own header:**

```
# Biomass type: total stock biomass (not SSB), whole Baltic model domain (10-30E, 54-66N)
```

The targets are **already declared as whole-domain aggregates**. The scope is correct as documented, and
the per-stock/aggregate ambiguity I raised does not exist. The 58× gap is against a correctly-scoped
target.

## 3. Independent cross-check on magnitude

The one quantitative anchor available for a Baltic coastal percid is Hansson et al. (2018,
Supplement 4), for perch in the Curonian Lagoon:

| quantity | value |
|---|---|
| lagoon area (entire area is perch production habitat, max depth 5 m) | 1,600 km² |
| perch production | 3,200 t → **2.0 t/km²** |
| fishery catch | 48 t |
| cormorant consumption | 118 t |
| total extraction | 166 t = **5.2% of production** |

Converting production to biomass with P/B ≈ 0.5–0.7 (Downing & Plante, 1993; Randall & Minns, 2000)
gives **2.9–4.0 t/km² of perch biomass in prime habitat**.

Scaling that to the model domain requires the percid-habitat fraction, which is the dominant unknown.
The Baltic is ~377,000 km²; sheltered bottoms shallower than 10 m are plausibly 5–15%, i.e.
19,000–57,000 km². Applied naively that gives **55–226 kt of perch**.

**But the Curonian Lagoon is an upper bound on density, not a mean.** Hansson et al. note its primary
productivity is *2–4× higher than the Baltic Proper*, and that much of the <10 m bottom area lies in
outer-coastal and offshore zones "where perch is uncommon". Applying a 2–4× productivity discount gives
roughly **14–113 kt**.

The committed perch target is **8–50 kt**. It sits at the low end of that band but inside it. **The
target is not obviously wrong.**

## 4. Conclusions per species

### pikeperch — target retained, 4,000–25,000 t

No production anchor comparable to the perch one was found. The target is supported indirectly:

* It must be **below** perch. Pikeperch is the less abundant of the two coastal percids, occurring in
  "separate and restricted areas" (Hansson et al., 2018, citing Lehtonen & Toivonen, 1988; Saulamo &
  Thoresson, 2005), whereas perch is ubiquitous along the coast. The committed targets put pikeperch at
  roughly half of perch, which is if anything **generous** to pikeperch.
* Baltic stocks are in poor condition: of the stocks Jakubavičiūtė et al. (2022) assessed, **the two
  Baltic ones (Curonian Lagoon and Galtfjärden) were both strongly depleted below B<sub>MSY</sub>**, with
  Pärnu Bay also considered poor. A target near the bottom of a plausible range is the correct reading
  of a depleted stock complex.

**Recommendation: retain 4,000–25,000 t.** Change the `source` field from "Literature estimate for
coastal Baltic" to record the reasoning and the fact that ICES does not assess the species. Keep
weight 0.2.

### smelt — target retained, 20,000–120,000 t, weight raised is NOT recommended

Smelt is the second-most abundant fish in the eastern Gulf of Finland after herring (Sendek & Bogdanov,
2019), so a target an order of magnitude above pikeperch's is directionally right. No quantitative
basin-wide anchor was retrieved. The 5.7× model overshoot is much smaller than pikeperch's 58× and may
be within what a coarse grid can be expected to deliver for a species whose distribution the grid
under-resolves.

**Recommendation: retain 20,000–120,000 t and weight 0.3.**

## 5. What this means for the overshoot

The pikeperch overshoot is **not** explained by a mis-scoped or mis-magnitude target. The target's scope
is correctly declared, its magnitude is consistent with the one available production anchor and with the
documented depleted state of the Baltic stocks, and its relationship to the perch target is internally
consistent.

**The 58× gap is real.** Combined with the seven failed interventions
(`docs/baltic_percid_overshoot_investigation_2026-08-02.md`), the reading is that this configuration
cannot represent Baltic pikeperch at a defensible biomass — which is what the file's own note
("coarse grid under-resolves") anticipated. The weight-aware certifier now keeps that failure out of the
headline verdict rather than resolving it.

## 6. Limits

* The perch cross-check rests on a single lagoon anchor scaled by a habitat fraction I estimated rather
  than measured. It bounds the target to within roughly an order of magnitude — enough to say "not
  obviously wrong", not enough to refine the numbers.
* No quantitative anchor was found for pikeperch or smelt specifically; both conclusions are relative
  (to perch, and to herring) rather than absolute.
* Retaining a target because it is *not refuted* is weaker than deriving it. These remain
  weight-0.2/0.3 literature estimates and should not drive calibration.

## References

Downing, J. A., & Plante, C. (1993); Randall, R. G., & Minns, C. K. (2000). Production/biomass
relationships, cited in Hansson et al. (2018).

Hansson, S., Bergström, U., & Bonsdorff, E. (2018). Competition for the fish — fish extraction from the
Baltic Sea by humans, aquatic mammals, and birds. *ICES Journal of Marine Science, 75*(3), 999–1008.
https://doi.org/10.1093/icesjms/fsx207

Jakubavičiūtė, E., Arula, T., & Dainys, J. (2022). Status and future perspectives for pikeperch (*Sander
lucioperca*) stocks in Europe. *openRxiv*. https://doi.org/10.1101/2022.12.20.521162

Sendek, D. S., & Bogdanov, D. V. (2019). European smelt *Osmerus eperlanus* in the eastern Gulf of
Finland, Baltic Sea: Stock status and fishery. *Journal of Fish Biology*.
https://doi.org/10.1111/jfb.14009
