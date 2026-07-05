# Baltic foodweb spawning & recruitment — literature review and candidate model corrections

**Date:** 2026-07-03
**Scope:** Eastern/Western Baltic cod (*Gadus morhua*) and coastal percids (perch *Perca fluviatilis*,
pikeperch *Sander lucioperca*) — mechanisms controlling reproductive success and recruitment, and what
the OSMOSE Baltic configuration would need to represent to reproduce realistic (non-boom/bust) long-term
dynamics.
**Method:** scite MCP (literature discovery + DOI/retraction verification), ICES MCP (observed stock
series), and web fetch of open-access full text. All eight load-bearing DOIs were checked against scite's
retraction index — none are retracted.
**Motivation:** the Baltic config exhibits population-level cod boom/bust overshoot and percid overshoot
(×38–96). Five levers have already been tried and ruled out — parameters (SP-A), grid refinement (SP-B),
salinity-correct spawning areas, per-cell spatial egg-survival (SP1), and a per-year reproductive-volume
recruitment gate (RV gate, both `mean_preserving` and `raw_cap`). The RV gate **worsened** overshoot by
adding recruitment variance the model amplifies. This review asks what the *published science* says the
missing mechanism is.

---

## 1. TL;DR — the diagnosis and the one untried lever

1. **The real Baltic cod stock has recruitment that is decoupled from spawning-stock biomass.** ICES
   data show that after the 2019–2020 fishing moratorium drove F to near-zero and let SSB stabilise,
   recruitment did **not** recover — it stayed at roughly half the long-term mean. In nature, cod
   recruitment is gated by the environment (reproductive volume, hypoxia, body condition, seal-transmitted
   parasites), **not** by spawner abundance. A model in which recruitment is an increasing function of SSB
   is therefore structurally mis-specified for this stock.

2. **The overshoot is the classic failure mode of a stock-recruitment (SR) curve inside a multi-species
   model.** McGregor, Fulton & Dunn (2019) show analytically that a Beverton–Holt SR curve makes a
   *depleted* population "very productive under persistent predation release" — precisely the boom that
   the Baltic config exhibits. Their recommended fix is to **cap recruitment at its unfished level**.
   The OSMOSE Baltic config added an explicit Beverton–Holt / Shepherd SR relationship during calibration
   (phase-12), so this warning applies **directly**.

3. **The one lever not yet tried is a recruitment *ceiling* (a hard cap at the unfished/reference level),
   which is conceptually different from every lever already ruled out.** All previous attempts tried to
   *modulate* recruitment (environmental multipliers, spatial egg survival) — they add variance or shift
   the mean but do not remove the runaway upside of the SR curve. A ceiling removes the upside directly.

4. **For percids, the model is missing the mechanisms that make percids self-limiting in reality:**
   temperature-gated year-class strength (first-winter survival) and strong density-dependent
   **cannibalism** (especially pikeperch). Neither is represented, which is a sufficient explanation for
   unbounded percid growth.

---

## 2. What the observed data show (ICES 2023 assessments)

| Stock | Key signal | Numbers |
|---|---|---|
| **Eastern Baltic cod** `cod.27.24-32` | Recruitment stepped down to a persistent low regime; **SR link broken** | Peak R 11.9×10⁶ (1976); last strong classes 5.1–5.2×10⁶ (2011–12); 2013→ ~1.3–2.6×10⁶ (≈½ long-term mean). SSB < Blim (108,942 t) continuously since ~2012. After 2019–20 moratorium (F 0.35→0.015) recruitment stayed flat ~1.6–2.0×10⁶ — **no recovery despite reduced F and stable SSB.** |
| **Western Baltic cod** `cod.27.22-24` | Same broken SR signature | Near-zero recruitment 2008–2021 despite fishing effort; SSB < Blim since ~2020. |
| **Sprat** `spr.27.22-32` | "Cod down, sprat up" regime | SSB held 0.8–1.1 Mt throughout the cod collapse; recruitment varies ~55× (4.6×10⁶–259×10⁶). |
| **Central herring** `her.27.25-2932` | Slow SSB decline | Recruitment varies ~8× over the modern period. |
| **Perch, pikeperch** | **Not ICES-assessed** | Managed as coastal/national stocks (HELCOM / national). Use HELCOM or national data, not ICES, for percid time series. |

**Load-bearing observation:** the decoupling of recruitment from SSB (point 2 in the table) is the single
most important empirical fact. It says the cod recruitment signal in the real system is environmentally
and condition-driven, not spawner-driven. Any SR curve that ties recruitment tightly to SSB will
misbehave — either failing to collapse when it should, or over-producing when the population expands.

---

## 3. Mechanisms controlling **cod** recruitment (literature)

### 3.1 Reproductive volume (RV) — necessary but not sufficient
- RV = water volume simultaneously meeting egg-survival constraints: salinity ≥ 11 psu (egg neutral
  buoyancy) and O₂ ≥ 2 ml/l (egg development); driven by irregular Major Baltic Inflows (Köster et al.
  2005; Hüssy et al. 2017).
- Baltic cod eggs are neutrally buoyant at ~14.5 psu, which occurs only below the halocline (~55 m), so
  eggs settle toward the hypoxic deep basins — the "buoyancy trap" (Wieland et al. 1994).
- Sperm motility and fertilization drop sharply below ~11–12 psu, setting an independent salinity floor
  on fertilization (Westin & Nissling; Nissling et al.).
- **Predictive limitation (critical for us):** RV is "significantly but only partly" related to
  recruitment. Even when inflows improved RV in the Gdańsk/Gotland basins, cod larvae remained very few
  (Plikshs et al. 2015), and hydrodynamic modelling suggests ~two-thirds of eggs are spawned in
  unfavourable conditions regardless of basin RV (Hüssy et al. 2015). **This is exactly why the project's
  RV gate failed: RV is a weak, partial predictor, so forcing recruitment through it injects noise
  without adding the missing compensatory structure.**

### 3.2 Clupeid predation and the cod–sprat reversal
- Sprat and herring prey on pelagic cod eggs/larvae; daily clupeid consumption of cod eggs reached
  > 50% of daily egg production in some May/June periods (Neumann et al. 2014).
- Heavy cod fishing released clupeids from predation control; the large sprat population then both preys
  on cod eggs and depletes the zooplankton cod larvae need — a double suppression ("cultivation–
  depensation"), and a trophic-cascade regime shift with an identified planktivore threshold
  (~17×10¹⁰ individuals) separating two ecosystem states (Casini et al. 2008).
- **Relevance:** this is an *emergent* food-web feedback that OSMOSE should in principle capture through
  spatial predation — but only if sprat/herring predation on cod early-life stages is resolved at the
  model's grid/foodweb resolution. If it is weak in the model, one of nature's main cod-recruitment
  brakes is missing.

### 3.3 The post-2010 collapse — multi-driver, condition-mediated (Orio et al. 2023)
- Body condition (Fulton's K) fell to historic lows; L50 (length at maturity) collapsed from ~40 cm
  (early 1990s) to ≤ 20 cm; L95 fell to ~50 cm (lowest since the 1930s).
- Natural mortality "markedly increased"; hypoxic dead-zone expansion + high local density + scarce
  benthic prey (*Saduria entomon*) jointly drive poor condition (Casini et al. 2016).
- Grey-seal-transmitted liver nematode *Contracaecum osculatum* prevalence rose to 88–100% (2016–2020),
  reducing condition and metabolic performance (Ryberg et al. 2020).
- **Relevance:** the collapse is condition- and mortality-driven, not spawner-driven — reinforcing §2.

### 3.4 Density-dependent stabilisers in cod are weak
- Cod cannibalism in the Central Baltic is judged to have a "negligible" net effect on recruitment
  (Neuenfeldt & Köster 2000).
- Condition-dependent fecundity and density-dependent growth exist but are weak and largely overwhelmed
  by environmental + predation forcing (Hüssy et al. 2017).
- **Implication:** cod do **not** have a strong intrinsic density-dependent brake. So the stabilising
  structure in a cod model must come from (a) an environmentally-capped recruitment and/or (b) an
  explicit recruitment ceiling — not from cannibalism.

---

## 4. Mechanisms controlling **percid** recruitment (literature)

Percids are self-limiting in nature through several mechanisms — **none currently represented** in the
config, which is a sufficient explanation for percid overshoot:

- **Temperature-gated year-class strength.** Pikeperch recruitment is strongly summer-temperature-driven:
  mean June–July temperature explained ~40% of year-class variance in the Gulf of Finland and ~73%
  (July–August) in the Archipelago Sea; strong classes form above ~18.5 °C (Pekcan-Hekim et al. 2011).
  Mechanism is **first-winter survival** — fast summer growth → larger size → higher overwinter survival.
- **Density-dependent cannibalism (strong in pikeperch).** YOY pikeperch in conspecific stomachs rise with
  juvenile density, causing sharp late-summer YOY declines — a genuine self-regulating brake
  (Hydrobiologia 2024). Perch is an ontogenetic omnivore shaped by cannibalism and competition.
- **Interspecific control.** Perch stock-recruitment is jointly controlled by temperature (bottom-up) and
  pikeperch abundance (negative — competition/predation) (Olin et al. 2019, *Hydrobiologia*).
- **Habitat-limited coastal spawning + top-down control.** Percids are freshwater-origin species confined
  to sheltered warm coastal bays; spawning/nursery habitat caps abundance. Fishing (harvest before
  maturity) and cormorant predation are the dominant top-down limiters in declining stocks (Ambio 2013).

**Implication:** percid abundance is capped by a *stack* of mechanisms — thermal recruitment gating,
cannibalism, habitat. A model with none of these has no way to prevent unbounded growth.

---

## 5. The pivotal modelling insight — why an SR curve boom/busts under predation release

**McGregor, Fulton & Dunn (2019), *PeerJ* 7:e7308** (verbatim abstract):

> "The spawning stock recruitment (SSR) relationship … can produce dynamics that are counter-intuitive
> and change scenario outcomes. We analysed the Beverton-Holt SSR curve and found **a population with low
> resilience when depleted becomes very productive under persistent predation release**. To avoid
> implausible increases in biomass, **we propose limiting recruitment to its unfished level.** This allows
> for specification of resilience when a population is depleted, without sudden and excessive increase when
> the population expands."

How this maps onto the OSMOSE Baltic config:
- Base OSMOSE has **no** built-in SR curve — recruitment is emergent from spatial predation + starvation +
  a prescribed larval-mortality constant `Mlarval`, and it is *extremely* sensitive to `Mlarval`
  (a ~10% change can more than halve a species' biomass; Oliveros-Ramos et al. sensitivity work).
- The Baltic config **added** an explicit Beverton–Holt / Shepherd SR curve during phase-12 calibration.
- Adding a B-H curve on top of an emergent-density-dependence engine risks **double-counting** the
  compensatory reserve, and — per McGregor et al. — the B-H curve's productive upside dominates whenever
  the population expands (predation release), producing the boom. The subsequent food-limitation/starvation
  crash produces the bust. Boom/bust follows structurally.

**Contrast with the Atlantis Baltic approach:** Atlantis uses B-H **plus** environmental scaling of
recruits by temperature/salinity/oxygen, tuned to a stable 120-year equilibrium (Bossier et al. 2018).
The difference from the project's failed RV gate is that Atlantis's B-H is *saturating* (already capped)
and the environmental term only *suppresses*; the project applied an environmental multiplier without the
underlying ceiling. The ceiling is the missing ingredient.

---

## 6. Candidate corrections, ranked, and mapped to what's already ruled out

| # | Candidate correction | Literature basis | Distinct from ruled-out levers? | Est. effort |
|---|---|---|---|---|
| **1** | **Recruitment ceiling** — cap cod recruitment at its unfished/reference level (hard cap on the B-H output, not a multiplier). | McGregor et al. 2019 (the direct fix for this exact failure mode). | **Yes — fundamentally.** RV gate *modulated* recruitment (added variance → worse). A ceiling *removes the upside* of the SR curve. Never tried. | Small — one clamp in the reproduction step, config-gated, inert by default. Testable with the existing determinism harness. |
| **2** | **Percid cannibalism** — add intra-specific (and pikeperch→perch inter-specific) predation as a density-dependent juvenile-mortality term. | Hydrobiologia 2024; Olin et al. 2019. | Yes — no self-limiting term exists for percids today. | Medium — needs a cannibalism/predation kernel entry; percids already have life-stage maps. |
| **3** | **Percid thermal recruitment gate** — scale percid year-class strength by summer temperature / degree-days (first-winter survival). | Pekcan-Hekim et al. 2011 (40–73% of variance). | Partly — different species and a *cap* (warm-year ceiling), not the cod RV multiplier. CMEMS temperature forcing already exists in-repo. | Medium. |
| **4** | **Re-specify cod recruitment as environment-capped, weakly SSB-coupled** — replace/augment the SSB-driven B-H with a saturating recruitment that is *suppressed* by RV/hypoxia (Atlantis pattern), retaining the ceiling from #1. | Bossier et al. 2018; §2 decoupling; Hüssy et al. 2017. | Yes — combines the ceiling (#1) with suppression, which the failed RV-gate-alone did not. | Medium–large. |
| **5** | **Strengthen emergent density-dependence via density-dependent `Mlarval`** instead of the explicit SR curve — i.e. remove the added B-H and let predation/starvation + a density-scaled larval mortality do the compensation. | OSMOSE design (recruitment emergent + high `Mlarval` sensitivity). | Yes — the opposite direction (remove the SR curve rather than cap it). | Medium; risk of re-tuning the whole config. |

**Recommended first move: candidate #1 (recruitment ceiling).** It is the smallest, most directly
literature-supported change; it targets the mechanism the theory names; and it is conceptually orthogonal
to every lever already ruled out — so it is not "re-attempting params/grid/spawning/RV," which memory
correctly warns against. Build it the way the RV gate was built (config-gated, inert-by-default,
parity-bit-identical when off, TDD) and A/B the cod boom/bust ratio with the ceiling on vs off. If the
ceiling damps cod overshoot, follow with #2+#3 for percids.

**Important caveat:** no peer-reviewed paper documents OSMOSE-Baltic overshoot specifically; the
McGregor→OSMOSE mapping is a well-grounded inference from (a) the general SR-in-multispecies-models result,
(b) OSMOSE's documented structure and `Mlarval` sensitivity, and (c) the project's own diagnostics. Treat
these as hypotheses to be tested with the same rigor as the RV gate, not as settled fixes.

---

## References

*All DOIs below were verified present and non-retracted via scite. Items marked ⚠ were read via abstract /
review-citation rather than primary full text — verify author-year against the primary source before formal
citation.*

- Bossier, S., et al. (2018). The Baltic Sea Atlantis: An integrated end-to-end modelling framework.
  *PLOS ONE*, 13(7), e0199168. https://doi.org/10.1371/journal.pone.0199168
- Casini, M., et al. (2008). Multi-level trophic cascades in a heavily exploited open marine ecosystem
  / Trophic cascades promote threshold-like shifts. *PNAS*, 105(37).
  https://doi.org/10.1073/pnas.0806649105
- Casini, M., et al. (2016). Hypoxic areas, density-dependence and food limitation drive the body
  condition of a heavily exploited marine fish predator. *Royal Society Open Science*, 3(10), 160416.
  https://doi.org/10.1098/rsos.160416
- Köster, F. W., et al. (2005). Baltic cod recruitment — the impact of climate variability on key
  processes. *ICES Journal of Marine Science*, 62(7), 1408. https://doi.org/10.1016/j.icesjms.2005.05.004
- McGregor, V. L., Fulton, E. A., & Dunn, M. R. (2019). Spawning stock recruitment creates misleading
  dynamics under predation release in ecosystem and multi-species models. *PeerJ*, 7, e7308.
  https://doi.org/10.7717/peerj.7308
- Neuenfeldt, S., & Köster, F. W. (2000). Trophodynamic control on recruitment success in Baltic cod:
  the influence of cannibalism. *ICES Journal of Marine Science*, 57(2), 324.
  https://doi.org/10.1006/jmsc.2000.0647
- Olin, M., et al. (2019). Effects of water temperature and pikeperch abundance on the stock–recruitment
  relationship of Eurasian perch in the northern Baltic Sea. *Hydrobiologia*.
  https://doi.org/10.1007/s10750-019-04008-z
- Orio, A., et al. (2023). New insights into the recent collapse of Eastern Baltic cod from historical
  data on stock health. *PLOS ONE*, 18(6), e0286247. https://doi.org/10.1371/journal.pone.0286247
- Pekcan-Hekim, Z., et al. (2011). Climate warming and pikeperch year-class catches in the Baltic Sea.
  *Ambio*, 40(5), 447. https://doi.org/10.1007/s13280-011-0143-7
- ⚠ Cardinale, M., et al. (2017). Fish egg predation by Baltic sprat and herring. *CJFAS*, 74(9).
  https://doi.org/10.1139/cjfas-2017-0105
- ⚠ Hüssy, K., et al. (2017). Eastern Baltic cod recruitment revisited. *ICES JMS*, 74(1), 3.
  https://doi.org/10.1093/icesjms/fsr145 (review; also Hüssy et al. 2016, ICES JMS 73(9):2138)
- ⚠ Ryberg, M. P., et al. (2020). Physiological condition of Eastern Baltic cod infected with
  *Contracaecum osculatum*. *Conservation Physiology*, 8(1), coaa093.
  https://doi.org/10.1093/conphys/coaa093
- ⚠ Wieland, K., et al. (1994). Egg buoyancy of Baltic cod. *Marine Biology*.
  https://doi.org/10.1007/BF01986342

**Data source:** ICES 2023 stock assessments via ICES MCP — cod.27.24-32 (asmt 17782), cod.27.22-24
(17840), her.27.25-2932 (17816), spr.27.22-32 (17805).
