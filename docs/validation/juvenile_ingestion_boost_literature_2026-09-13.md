# Literature validation — juvenile ingestion-cap multiplier (`j` = 1.44–4.64×)

**Date:** 2026-09-13 · **Tool:** scite MCP · **Subject:** commit `374bc26`
(`osmose/calibration/bioen_offline.py`, opt-in juvenile boost) and the `theta`/`c_rate` keys it
maps onto in `per_fish_ingestion_cap`.

**Claim under test.** Juvenile/larval fish sustain a maximum ingestion rate ABOVE what the adult
allometry `Imax·w^β` (β = 0.8) predicts, and a multiplier of ~1.4–4.6× over the first year is
defensible.

## Verdict

| | |
|---|---|
| Direction (early stages have higher mass-specific max ingestion) | ✅ **Verified** — two independent lines |
| Bioen-OSMOSE documents such a parameter | ⚠️ **Partially supported** — assumption stated, magnitude never published in retrievable text |
| The specific 1.44–4.64× magnitude | ❔ **Not verified** — no source gives a larva:adult ingestion ratio for any study species |
Separately, and **not a citation finding** — no paper contradicts anything here — my own hypothesis
that the β gap *generates* the fitted values did not survive the arithmetic: it has ≤1.5× of
dynamic range against a 3.24× spread. See the next section.

**No editorial notices** (retraction/correction/expression of concern) on any paper cited here.

## The quantitative test I ran, and why it failed to support the fit

Kiørboe & Hirst (2014) put maximum ingestion on a `w^0.75` law. Bioen-OSMOSE uses β = 0.8. If the
fitted boost were compensating for that 0.05 exponent gap, its size would be
`(w_ref/w_juv)^0.05` and it would *rank* with that quantity across species. Neither holds:

| species | fitted `j` | implied by β-gap (vBGF anchor / model trajectory) |
|---|---|---|
| cod_west | 2.88 | 1.36 / 1.50 |
| cod_east | 2.37 | 1.34 / 1.47 |
| herring | **4.64** | **1.15 / 1.33** |
| sprat | 2.43 | 1.15 / 1.29 |
| flounder | 3.30 | 1.26 / 1.40 |
| perch | 3.00 | 1.27 / 1.45 |
| pikeperch | 3.02 | 1.30 / 1.49 |
| smelt | 1.89 | 1.19 / 1.33 |
| stickleback | 1.44 | 1.15 / 1.22 |

**The load-bearing argument is the spread, and it needs no statistics.** Fitted `j` spans **3.24×**
(1.44–4.64). The β-gap predictor spans **1.19×** (vBGF anchor) or **1.23×** (trajectory anchor). A
predictor with 1.2× of dynamic range cannot generate a 3.24× spread, under any anchoring — to get
herring's 4.64 out of an exponent gap you would need β_emp ≈ 0.53, nowhere near the measured 0.75.
This is deterministic and it is the finding.

**Corroborating, but weak on its own:** the rank order does not track either. Spearman ρ = −0.05
(p = 0.90) vBGF anchor, ρ = +0.28 (p = 0.46) trajectory anchor — robust across both anchorings
tried, though *not* invariant to the choice (the two ρ differ materially; what is invariant is the
0.05 exponent, since ranks of `x^0.05` are ranks of `x`). Treat this as corroboration only: **at
n = 9 the critical |ρ| for p < 0.05 is 0.667**, so failing to detect a correlation against a
predictor with 1.2× of range is weak evidence of absence, not proof of it. The same lesson recorded
after the growth-account refutation applies here in reverse — nine species cannot reject a mechanism
any more than they can establish one.

**And the rank test was structurally uninformative from the start.** Residuals begin at
`age_dt = ndt = 24` (`fit_species`: *"ages >= 1 yr (spec 3.4: larval phase not fitted)"*) while the
boost applies at `age_dt < 24`. So `j` acts **entirely below the first residual point**: its fitted
value is whatever makes `w[24]` reach the age-1 target after 23 steps of nonlinear integration from
egg weight, given `Imax`, `c_m` and the temperature path. `j` is a knob on **entry weight into the
fitted window**, not a ration multiplier — there is no reason it would rank with a static allometric
ratio even if the underlying biology were exactly the β gap.

**Caveat on the anchoring.** The vBGF column extrapolates the growth curve below age 1, where it is
not valid (cod_west's first-year geometric-mean mass reads 9.76 g against a ~1e-3 g egg). The
trajectory column uses the fitted model's own egg→age-1 weight path instead. The genuinely soft
number is the **median fitted/implied** (2.12 vBGF vs 1.92 trajectory); the spread comparison is
barely anchor-dependent.

**What this does NOT show.** Kiørboe & Hirst is an *interspecific* law, and the paper says so
itself: *"There may be significant deviations in mass scaling, both during ontogeny within a
species and between species … but such variation is hidden in the current larger-scale analysis."*
So 1.15–1.50× is the answer to "what if the model used the measured interspecific exponent" — it is
**not** a ceiling on an intraspecific developmental effect, and must not be read as one.

## Evidence by claim

### 1 — Early stages have higher mass-specific maximum ingestion ✅
**DOI:** https://doi.org/10.1111/ele.70017 (Morell et al., 2024, *Ecology Letters*, OA cc-by) ·
notices: none
> "It emerges from the higher maximum mass-specific ingestion rate of early-life stages, an
> assumption originally included to reflect faster growth during the larval and postlarval periods
> (Figure S7) (Osse and Boogaart 1995)."

> "This is consistent with studies showing that the maximum mass-specific ingestion rate decreases
> with body mass in fish at the interspecific level (Kiørboe and Hirst 2014) and during development
> at the intraspecific level (Wuenschel and Werner 2004)."

This is the model family's own literature confirming the mechanism is a deliberate, documented
assumption — not an artifact. It separates the two levels explicitly, which is why the interspecific
β-gap cannot bound the intraspecific effect.

### 2 — Interspecific scaling ✅ (but shallow)
**DOI:** https://doi.org/10.1086/675241 (Kiørboe & Hirst, 2014, *Am. Nat.*) · notices: none
> "Plots of mass-specific ingestion rates reveal a rather consistent decline across taxa, and the
> average of power exponents is not significantly different from −1/4."

327 maximum-ingestion estimates. Gives β_emp ≈ 0.75 against the model's 0.8 — the 1.15–1.50× above.

### 3 — Intraspecific developmental magnitude ⚠️ (closest analogue: 2.2×)
**DOI:** https://doi.org/10.1111/j.0022-1112.2004.00479.x (Wuenschel & Werner, 2004, *J. Fish Biol.*)
· **abstract-only, full text access-restricted** · notices: none
> "The C_MAX model predicted an initial increase in specific feeding rate from 70 to 155% M_D day⁻¹
> for small larvae, before declining for larger larvae and juveniles."

> "Mass-specific midgut contents increased for small larvae <0.156 mg dry mass (c. 4 mm L_S), and
> decreased for larger larvae and juveniles."

70 → 155% = **2.21×**, which lands inside the fitted 1.44–4.64 range. Two hard caveats: this is a
rise *within the larval phase*, **not** a larva:adult ratio; and the species is a Gulf of Mexico
sciaenid (*Cynoscion nebulosus*), not a Baltic species. Note also the shape — a **hump** (rise below
0.156 mg, fall above) — which no single power law in `w` can represent. That is the structural
argument for having a separate larval parameter at all, independent of its value.

### 4 — Independent physiological anchor ⚠️ (2.5×, oxygen side)
**DOI:** https://doi.org/10.1242/jeb.150.1.343 (Kaufmann, 1990, *J. Exp. Biol.*, OA) · notices: none
> "To a lesser extent this also holds true for the larval cyprinids, with exponents of 0.87 and
> 0.84, but here the mass-specific active rates clearly decline with mass."

> "This is hardly surprising as they are 2.5-fold higher than those of adult fish (Blaxter, 1969;
> Beamish, 1978) and are, moreover, among the highest respiration rates ever recorded for fish.
> Higher oxygen consumptions have only been found in clupeid larvae."

Relevant because Morell et al. tie the two sides together explicitly: *"a higher maximum
mass-specific ingestion rate relies implicitly on a higher maximum mass-specific rate of oxygen
supply (Deutsch et al. 2015)."* So a measured **2.5× larval:adult mass-specific active metabolic
rate** is a demand-side analogue of the same multiplier. Caveats: cyprinids (fresh-water), and
metabolism, not ingestion. The note that clupeid larvae run higher still is the only retrieved hint
that herring/sprat might sit at the top of a range — it is a qualitative aside, not a measurement.

### 5 — Species-specific values for the eight Baltic species ❔ **Not found**
Three searches (Baltic/North Sea gadoid + clupeid larval ration; Cmax larval-vs-adult allometry;
the Bioen-OSMOSE papers themselves) returned **no** source giving a maximum-ingestion multiplier for
cod, herring, sprat, flounder, perch, pikeperch, smelt or stickleback. Nearest misses, neither
carrying the number: Folkvord et al. (2015) `10.1111/jfb.12783` (cod vs herring larvae co-reared at
variable prey — growth and survival, not Cmax) and Fiksen & Jørgensen (2011) `10.1371/journal.pone.0098205`
(Quirks larval foraging model, includes 7-mm Atlantic cod).

### 6 — `theta` / `c_rate` / `larvaeThresDt` published values ❔ **Not found**
Both Bioen-OSMOSE papers are **abstract-only** through scite (`contentDenied: true`):
`10.1016/j.pocean.2023.103064` returned zero readable characters; the preprint
`10.1101/2023.01.13.523601` returned the abstract only. Neither abstract names the larval ingestion
parameter. Morell et al. (2024) confirms the assumption **exists** and states its rationale; it does
not give its magnitude. Osse & van den Boogaart (1995) — the origin citation for the assumption —
was seen **only quoted inside citing papers** and could not be retrieved (ICES Mar. Sci. Symp.,
apparently no DOI). It is therefore **not** in the reference list below.

## Bottom line for the code

The boost is a **real, documented modelling construct with the right sign**, and its median fitted
value (2.88) sits between the two independent quantitative anchors I could retrieve (2.21× and
2.5×). But:

1. No retrieved source supports **4.64×** (herring) — that exceeds every anchor found.
2. The **between-species spread is unjustified**: fitted `j` spans 3.24× while the only mechanism
   I can quantify spans 1.2×. A 1.2× predictor cannot produce a 3.24× spread, so per-species `j` is
   absorbing something other than a size-scaling correction. (The absent rank correlation points
   the same way but carries little weight at n = 9.)
3. `theta`/`c_rate` remain at their disabled defaults in the committed overlay
   (`c3_bioen_arm.json`: `theta = 1.0`, `c_rate = 0.0`). **Nothing shipped depends on these values.**

This **confirms rather than clears** the caveat recorded at commit time: the direction is defensible;
"the curve fit wanted it" is still not a justification for the magnitude. If the boost is ever
enabled, a defensible move is a **single shared `j` ≈ 2.2–2.5** taken from the literature anchors
rather than nine free per-species values — which also costs 8 degrees of freedom the growth data
does not appear to constrain.

## References

Kaufmann, R. (1990). Respiratory cost of swimming in larval and juvenile cyprinids. *Journal of
Experimental Biology*, 150(1), 343–366. https://doi.org/10.1242/jeb.150.1.343

Kiørboe, T., & Hirst, A. G. (2014). Shifts in mass scaling of respiration, feeding, and growth rates
across life-form transitions in marine pelagic organisms. *The American Naturalist*, 183(4),
E118–E130. https://doi.org/10.1086/675241

Morell, A., Shin, Y.-J., & Barrier, N. (2024). Realised thermal niches in marine ectotherms are
shaped by ontogeny and trophic interactions. *Ecology Letters*, 27(11).
https://doi.org/10.1111/ele.70017

Wuenschel, M. J., & Werner, R. G. (2004). Consumption and gut evacuation rate of laboratory-reared
spotted seatrout (Sciaenidae) larvae and juveniles. *Journal of Fish Biology*, 65(3), 723–743.
https://doi.org/10.1111/j.0022-1112.2004.00479.x *(abstract only; full text access-restricted)*
