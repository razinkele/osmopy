# Documented OSMOSE Ecosystem Models & Available Model Setups Worldwide

*Reviewed, fact-checked and updated edition*
Last updated 6 July 2026 · Verified against GitHub, Zenodo, CRAN and the peer-reviewed literature

---

## Reviewer's summary — what changed

This edition verifies every checkable claim in the previous draft against live sources (the [osmose-model](https://github.com/osmose-model) GitHub organisation, Zenodo, CRAN and journal records) and adds work published through mid-2026. The most consequential corrections concern how the model and its configurations are actually distributed today.

| Item | Previous draft | This edition |
|---|---|---|
| CRAN R package | Claimed installable via `install.packages("osmose")`. | **Corrected:** the `osmose` package was archived / removed from CRAN on 2022-10-03. Only legacy v3.3.4 remains in the CRAN archive. Current v4.x installs from GitHub or the project's drat repository. |
| Latest code version | Cited Zenodo v4.3.3 (2023) as the latest. | **Updated:** latest archived release is v4.4.1 (18 June 2026, DOI 10.5281/zenodo.20744141); the core repo is actively developed (last push June 2026). |
| West Florida Shelf / Gulf of Mexico config | Stated the WFS input set "is not posted online." | **Corrected:** a public Gulf of Mexico configuration exists — `osmose-model/osmose-gom` (updated 2025). |
| Southern Benguela config | Stated there is "no open repository" for Benguela. | **Corrected:** a public Benguela configuration exists — `osmose-model/osmose-ben` (multiple versions, incl. v4.3). |
| Yellow Sea (OSMOSE-YS) | Attributed to "Sun et al. (2024)." | **Corrected year:** Sun, R. et al. (2023), *J. Mar. Syst.* 240:103946. |
| Eco-genetic module | Referred to "EvO-OSMOSE." | **Corrected name:** Ev-OSMOSE (eco-genetic extension). |
| New work added | — | Huang et al. (2025) offshore-wind EEC model; Moullec et al. (2023) Mediterranean rebuilding scenarios; the Fish-MIP global/regional intercomparison (*Earth's Future*, 2024); Bioen-OSMOSE (2023) and Ev-OSMOSE (2023) extensions. |

---

## Part 1 — Documented OSMOSE model applications by region

OSMOSE (Object-oriented Simulator of Marine ecOSystEms) is a spatially explicit, individual-based, multi-species model of size-based opportunistic predation. Entries below give scope, drivers, calibration/validation and primary references. Confirmed items are cited to their DOI; genuine gaps are flagged.

### Mediterranean Sea

**OSMOSE-MED — whole Mediterranean basin.** End-to-end model of the entire basin with 100 species (~95% of regional landings) on a ~20′×20′ grid, one-way coupled to the NEMOMED12 circulation model and the Eco3M-S biogeochemical model. Calibrated with an evolutionary algorithm to 2006–2013 biomass and catch, and validated against MEDITS survey data, diets and trophic levels. Under RCP8.5, total fish biomass is projected to change by roughly +5 to +22% by 2100, with strong spatial and species contrasts (small thermophilic and exotic species gain; large native predators decline).
*References:* Moullec et al. (2019), [Prog. Oceanogr. 178:102179](https://doi.org/10.1016/j.pocean.2019.102179); Moullec et al. (2019), [Front. Mar. Sci. 6:345](https://doi.org/10.3389/fmars.2019.00345). Follow-up rebuilding scenarios: Moullec et al. (2023), [Mar. Ecol. Prog. Ser. 708:1–20](https://doi.org/10.3354/meps14269) (new).

**OSMOSE-GoL — Gulf of Lions (NW Mediterranean).** Shelf-scale model of 10 exploited fish groups plus forage species (>70% of catches), two-way coupled to an Eco3M/ROMS lower-trophic model so that fish predation feeds back onto plankton. Parameterised for 2001–2004 and validated against satellite chlorophyll, biomass, landings and diets. Fish predation accounts for <30% of micro/mesozooplankton mortality yet still drives measurable bottom-up and top-down cascades in plankton and fish seasonal cycles.
*References:* Bănaru et al. (2019, *Ecol. Model.*, Part I — parameterisation & validation); Diaz et al. (2019, *Ecol. Model.*, Part II — coupling feedbacks).

**OSMOSE-GoG — Gulf of Gabès (SE Mediterranean, Tunisia).** End-to-end model of the shallow Gulf of Gabès (~11 exploited fish and invertebrate species: bony fishes, one shark, cuttlefish, and two shrimps), one-way coupled to an Eco3M/NPZD plankton model. Calibrated to mid-2000s biomass and catch and validated on diets, mean catch size and community trophic indicators. Used to test which size- and community-based indicators respond most sensitively to fishing pressure.
*Reference:* Halouani et al. (2019, *Ecol. Indicators*). A runnable configuration is public (see Part 2).

### European Atlantic & shelf seas

**Eastern English Channel (OSMOSE-EEC).** 14 key fish species (gadoids, flatfish, small pelagics, elasmobranchs, benthivores) with full life cycles, forced by prey-biomass fields and multi-fleet fishing, calibrated to 2000–2009. Used for climate projections (RCP4.5/8.5): for ~80% of species F_MSY declines under warming, so climate-adaptive management would reduce target fishing rates for cold-water stocks such as cod.
*References:* Travers-Trolet et al. (2020, [Front. Mar. Sci.](https://osmose-model.org/publications/)). **New:** Huang et al. (2025), "An ecosystem modelling approach to assess potential impacts of offshore wind farms," [ICES J. Mar. Sci. 82(9): fsaf153](https://doi.org/10.1093/icesjms/fsaf153) — extends the EEC model with new species, prey-field forcing, fleet dynamics and inter-annual calibration to assess cumulative offshore-wind and fishing effects.

**North Sea fish community.** Multi-species, size-structured model forced with five different plankton (zooplankton) prey fields to test sensitivity to lower-trophic input. Absolute fish biomass tracks total zooplankton production, but the relative composition of functional groups and spatial patterns stay consistent across plankton models — supporting robustness of community structure for management scenarios.
*Reference:* van de Wolfshaar et al. (2021, [Mar. Ecol. Prog. Ser.](https://www.wur.nl/en/publication-details.htm?publicationId=2d8746f6-4c02-4dea-9509-20e24df758ec)).

**Documented gaps: Baltic, Celtic Sea & Bay of Biscay.** No peer-reviewed OSMOSE model has been published for the Baltic Sea, the Celtic Sea or the Bay of Biscay. Baltic ensemble work has used other frameworks. These remain the clearest coverage gaps in the published OSMOSE record — directly relevant to prospective Baltic applications.

### Eastern boundary upwelling systems

**Southern Benguela (South Africa).** One of the earliest OSMOSE applications (Shin & Cury 2004, size-spectrum dynamics), later developed into a fully coupled ROMS–NPZD–OSMOSE end-to-end model (Travers-Trolet et al. 2014) with sardine, anchovy, hake and other groups. Smith et al. (2015) showed OSMOSE, Atlantis and Ecopath agree qualitatively on responses to heavy fishing; Briton et al. (2019) derived ecosystem-indicator reference levels at multi-species MSY, showing that robust reference points are strategy-specific.
*References:* Shin & Cury (2004, *J. Theor. Biol.*); Travers-Trolet et al. (2014, *Afr. J. Mar. Sci.*); Smith et al. (2015, *Science*); Briton et al. (2019, *ICES J. Mar. Sci.*). A public configuration exists (see Part 2).

**Northern Humboldt Current (Peru).** Marzloff et al. (2009) first applied OSMOSE to the Peruvian upwelling (hake harvesting strategies, 2000–2006). Hill-Cruz et al. (2022) built an end-to-end version coupled to the CROCO ocean model and BioEBUS biogeochemistry: interannual fish-biomass variability is strongly bottom-up (plankton-driven) but modulated by predation, so both environmental variability (e.g. El Niño) and species interactions are needed for skilful forecasts.
*References:* Marzloff et al. (2009, *J. Mar. Syst.*); Hill-Cruz et al. (2022, *Ecol. Model.*). No public repository located.

### East Asia, Indo-Pacific & Southern Ocean

**Yellow Sea (OSMOSE-YS).** Multi-species model of commercial pelagics, demersals and invertebrates, simulating 1970–2014 under low/moderate/high fishing. Long-term overfishing reduces total biomass, mean body size and longevity; heavy fishing on high-trophic predators releases low-trophic species (trophic cascades).
*Reference:* Sun, R. et al. (2023), [J. Mar. Syst. 240:103946](https://doi.org/10.1016/j.jmarsys.2023.103946) (authors incl. Shin, Barrier, Tian). *Corrected from "2024."*

**Jiaozhou Bay (OSMOSE-JZB).** Small temperate-bay model (Lei Xing and colleagues) centred on a short-lived migratory shrimp, Japanese mantis shrimp (*Oratosquilla oratoria*) and Korean rockfish (*Sebastes schlegelii*). Concentrated "race-to-fish" pulses cut community biomass and mean trophic level more sharply than evenly spread effort, so managing fishing seasonality mitigates impacts; migratory-shrimp biomass exerts strong bottom-up control on residents.
*References:* Xing et al. (2021, *Fish. Res.*); Xing et al. (2020, *Ecol. Model.* — sensitivity analysis); Xing et al. (2022, *Front. Mar. Sci.* — evaluation of a shrimp stock-enhancement programme).

**Cooperation Sea, East Antarctica (OSMOSE-CooperationSea).** Confirmed model for CCAMLR Division 58.4.2, including Antarctic and Patagonian toothfish and mid-trophic prey (krill, myctophids). Predicted biomasses fit observations and the model reaches a stable state; intensive toothfish harvesting can raise mid-trophic biomass through predator release but risks destabilising the food web. A follow-up adds global sensitivity / uncertainty analysis of the Antarctic configuration.
*References:* Xing et al. (2023), "Simulating impacts of fishing toothfish on the pelagic community in the Cooperation Sea, Southern Ocean," [Reg. Stud. Mar. Sci.](https://doi.org/10.1016/j.rsma.2023.103254); Xing et al. (2025, *Fish. Res.* — sensitivity/uncertainty). No public repository located.

### North & Central America

**West Florida Shelf / Gulf of Mexico (OSMOSE-WFS).** Spatial model of ~12 functional groups (grouper, snapper, sharks, forage fish, shrimp/crab) on the West Florida Shelf, forced by monthly plankton/benthos productivity and multi-fleet fishing, calibrated to 2005–2009 and validated against stomach data and an independent Ecopath model. Used for Management Strategy Evaluation and natural-mortality estimation: predation dominates juvenile red-grouper mortality, adult mortality is largely episodic red tide, and more conservative harvest rules raise long-term biomass at some short-term catch cost.
*References:* Grüss et al. (2015, 2016; *Ecol. Model.* and *J. Mar. Syst.*). **Correction:** a public Gulf of Mexico configuration is available — [osmose-model/osmose-gom](https://github.com/osmose-model/osmose-gom) (see Part 2).

**Pacific North Coast (PNCIMA, British Columbia).** Model of Pacific cod, lingcod, herring and other NE-Pacific shelf species, with plankton-productivity regimes imposed via larval mortality and growth. Ecosystem-based F_MSY is higher in high-plankton regimes; balanced (multi-species) harvest needs higher F for some stocks than single-species strategies; precautionary harvest control rules dampen variability. A follow-up examined cumulative stressors (reduced plankton, mammal predation, fishing).
*References:* Guo et al. (2019, *ICES J. Mar. Sci.*); Fu et al. (2020, *Front. Mar. Sci.*). No public repository located.

### Model extensions & cross-regional efforts

- **Bioen-OSMOSE** — bioenergetic extension with explicit physiological responses to temperature and oxygen (growth, reproduction, hypoxia). Morell et al. (2023), *Prog. Oceanogr.* 219:103064.
- **Ev-OSMOSE** — eco-genetic extension adding heritable trait variation and fisheries-induced evolution on top of the trophic core (2023 preprint; corrects the earlier "EvO-OSMOSE" label).
- **Fish-MIP intercomparison** — OSMOSE participates in the Fisheries & Marine Ecosystem Model Intercomparison Project. Recent global-vs-regional comparisons (*Earth's Future*, 2024) find regional models such as OSMOSE often project smaller climate-driven biomass declines than coarse global models, underscoring the value of fine-scale trophic detail.
- **Indicator meta-analyses** — Fu et al. (2019) used OSMOSE among multiple models across ~10 ecosystems, finding biomass:catch ratio and mean lifespan are comparatively robust fishing indicators, with threshold responses often near ~0.4–0.6 F_MSY.

---

## Part 2 — Available model setups, code & repositories

All code is open-source under the [osmose-model](https://github.com/osmose-model) GitHub organisation (20 repositories) and the [Osmose Zenodo community](https://zenodo.org/communities/osmose/records). The items below were verified live on 6 July 2026.

### Core code & installation

- **Engine (Java):** [github.com/osmose-model/osmose](https://github.com/osmose-model/osmose) — GPL-3.0, actively developed (last push June 2026).
- **Latest archived release:** v4.4.1, 18 June 2026 — [Zenodo DOI 10.5281/zenodo.20744141](https://doi.org/10.5281/zenodo.20744141) (35.8 MB). The concept DOI 10.5281/zenodo.7628348 always resolves to the newest release; v4.3.3 (DOI 10.5281/zenodo.7641728) is now superseded.
- **R interface — important change:** the `osmose` R package was **removed from CRAN on 2022-10-03** (unfixed check issues); only legacy v3.3.4 remains in the CRAN archive. Install the current v4.x from GitHub (`remotes::install_github("osmose-model/osmose")`) or from the project's [drat repository](https://github.com/osmose-model/drat).
- **Build/run helpers:** [osmose-conda](https://github.com/osmose-model/osmose-conda) (conda environments), [osmose-mpi](https://github.com/osmose-model/osmose-mpi) (C++ plugin for parallel runs), [osmose.calibrar](https://github.com/osmose-model/osmose.calibrar) (calibration templates), and [osmose.fishmip](https://github.com/osmose-model/osmose.fishmip) (Fish-MIP tools).

### Documentation & configuration tools

- **ODD / user documentation:** the full Overview-Design-Details description is archived on Zenodo ([record 5766976](https://zenodo.org/record/5766976)) and covers the core model plus the bioenergetic and eco-genetic extensions.
- **FishBase–OSMOSE bridge:** a live web app at [config.osmose-model.org](https://config.osmose-model.org) auto-generates OSMOSE input files from FishBase/SeaLifeBase (Grüss et al. 2019). Its source is public (archived): [osmose-web-api](https://github.com/osmose-model/osmose-web-api) and [the web UI](https://github.com/osmose-model/osmose-model.github.io). *This corrects the prior claim that the bridge is "not a standalone public repository."*
- **Configuration map viewer:** [configuration-leaflet-map](https://github.com/osmose-model/configuration-leaflet-map) for interactive display of a configuration's spatial inputs.

### Region-specific configurations you can download

**Four regional configurations are publicly downloadable today** (the previous draft listed only two and wrongly excluded the Gulf of Mexico and Benguela):

| Ecosystem | Repository | Contents / status |
|---|---|---|
| Eastern English Channel | [osmose-eec](https://github.com/osmose-model/osmose-eec) | Full EEC setup: species parameters, spatial grid, plankton forcing, accessibility matrix, two-phase fishing/mortality files. Basis for Travers-Trolet et al. (2019, 2020). |
| Gulf of Gabès | [osmose-gog](https://github.com/osmose-model/osmose-gog) | Complete config CSVs plus `calibrate.R` and `run.R` and spatial maps (11 species). Runnable test case also at `osmose-model/emibios_testcase`. |
| Gulf of Mexico / West Florida Shelf | [osmose-gom](https://github.com/osmose-model/osmose-gom) | Gulf of Mexico configuration (`gom_v3`), bundled OSMOSE jar, and R driver scripts (`dynamics_gom.R`, `analysis.R`). Updated 2025. |
| Southern Benguela | [osmose-ben](https://github.com/osmose-model/osmose-ben) | Benguela configuration in several OSMOSE versions (v3.2, v4.3, v4.x develop) with a `launcher.R` and utilities; 80+ commits. |

**Caveats:** these config repos were last updated 2022–2025 and target OSMOSE v3–v4.x; expect to reconcile parameter names with the current engine (v4.4.1). None carries a formal release tag, so clone the default branch.

### Described in the literature but not openly archived

- OSMOSE-MED, OSMOSE-GoL and the Northern Humboldt, PNCIMA and Cooperation Sea (Antarctic) setups: full parameterisations are published, but no plug-and-play input sets are in the public repositories — contact the authors or reconstruct via the bridge tool and the papers.
- Yellow Sea (OSMOSE-YS) and Jiaozhou Bay (OSMOSE-JZB): documented in detail in the cited papers; configurations not publicly posted.

### Practical starting points

- **To learn the workflow:** clone [osmose-eec](https://github.com/osmose-model/osmose-eec) or [osmose-gog](https://github.com/osmose-model/osmose-gog) (both include R calibration/run scripts) alongside the v4.4.1 engine.
- **To build a new ecosystem (e.g. Baltic):** use the [config.osmose-model.org](https://config.osmose-model.org) bridge to seed species/diet parameters from FishBase, then adapt an existing shelf-sea config (EEC or Benguela) for grid, forcing and fleets.
- **New users:** start from [osmose-welcome](https://github.com/osmose-model/osmose-welcome) and the ODD documentation.

---

## Key references (verified)

1. Moullec, F. et al. (2019). Capturing the big picture of Mediterranean marine biodiversity with an end-to-end model of climate and fishing impacts. *Prog. Oceanogr.* 178:102179. https://doi.org/10.1016/j.pocean.2019.102179
2. Moullec, F. et al. (2019). An end-to-end model reveals losers and winners in a warming Mediterranean Sea. *Front. Mar. Sci.* 6:345. https://doi.org/10.3389/fmars.2019.00345
3. Moullec, F. et al. (2023). Rebuilding Mediterranean marine resources under climate change. *Mar. Ecol. Prog. Ser.* 708:1–20. https://doi.org/10.3354/meps14269
4. Halouani, G. et al. (2019). Ecosystem indicators in the Gulf of Gabès (OSMOSE). *Ecol. Indicators.* https://doi.org/10.1016/j.ecolind.2018.12.005
5. Travers-Trolet, M. et al. (2020). Emergent effects of climate change on F_MSY in the Eastern English Channel. *Front. Mar. Sci.* https://osmose-model.org/publications/
6. Huang, Y. et al. (2025). An ecosystem modelling approach to assess potential impacts of offshore wind farms. *ICES J. Mar. Sci.* 82(9):fsaf153. https://doi.org/10.1093/icesjms/fsaf153
7. van de Wolfshaar, K. et al. (2021). North Sea fish community structure robust to plankton model choice. *Mar. Ecol. Prog. Ser.* https://www.wur.nl/en/publication-details.htm?publicationId=2d8746f6-4c02-4dea-9509-20e24df758ec
8. Smith, A.D.M. et al. (2015). Multi-model comparison, Southern Benguela. https://osmose-model.org/publications/
9. Briton, F. et al. (2019). Ecosystem indicator reference levels at multi-species MSY, Southern Benguela. *ICES J. Mar. Sci.* https://osmose-model.org/publications/
10. Marzloff, M. et al. (2009). Modelling the Peruvian upwelling with OSMOSE. *J. Mar. Syst.* https://doi.org/10.1016/j.jmarsys.2009.07.001
11. Hill-Cruz, M. et al. (2022). Drivers of interannual fish biomass variability in the Humboldt system. *Ecol. Model.* https://doi.org/10.1016/j.ecolmodel.2022.109937
12. Sun, R. et al. (2023). Exploring fishing impacts on the Yellow Sea ecosystem using an individual-based modelling approach. *J. Mar. Syst.* 240:103946. https://doi.org/10.1016/j.jmarsys.2023.103946
13. Xing, L. et al. (2023). Simulating impacts of fishing toothfish on the pelagic community in the Cooperation Sea, Southern Ocean. *Reg. Stud. Mar. Sci.* https://doi.org/10.1016/j.rsma.2023.103254
14. Grüss, A. et al. (2016). OSMOSE-WFS: MSE and natural-mortality estimation on the West Florida Shelf. *Ecol. Model. / J. Mar. Syst.* https://repository.library.noaa.gov/view/noaa/62637
15. Guo, C. et al. (2019). Environmentally driven F_MSY in PNCIMA. *ICES J. Mar. Sci.* https://osmose-model.org/publications/
16. Morell, A. et al. (2023). Bioen-OSMOSE: a bioenergetic marine ecosystem model with physiological response to temperature and oxygen. *Prog. Oceanogr.* 219:103064. https://doi.org/10.1016/j.pocean.2023.103064
17. Shin, Y.-J. et al. (2023). OSMOSE v4.3.3 (software). Zenodo. https://doi.org/10.5281/zenodo.7641728
18. Barrier, N. et al. (2026). OSMOSE v4.4.1 (software). Zenodo. https://doi.org/10.5281/zenodo.20744141

*Note: a few DOIs above are given at journal/landing-page level where the exact article DOI was not machine-verified in this pass; titles, authors and years were confirmed against publisher records and the OSMOSE publications list.*
