---
name: Tier 1 plan review (HELCOM + ICES + scite validated)
description: 2026-04-24 plan at docs/superpowers/plans/2026-04-24-tier1-baltic-improvements.md revised after HELCOM/scite validation. Key fix: OSMOSE expects standing biomass in NetCDF, not consumption-equivalent.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Iteratively reviewed the Tier 1 Baltic improvements plan against HELCOM MADS layers, ICES stock data, and peer-reviewed literature (via scite MCP). Five substantive corrections applied to the plan file.

**Critical semantic bug caught:** OSMOSE background species expects **standing biomass** in the NetCDF forcing; `predation.ingestion.rate.max` is the annual turnover multiplier (see `osmose/engine/background.py:89`). The original plan put consumption-equivalent biomass (body × turnover) in the NetCDF, which would have compounded with the ingestion rate to give 13× the intended consumption (~1,014,000 t/yr for seal instead of 60,000 t/yr). Plan now has:
- Seal NetCDF: 4,500 t standing biomass, ingestion rate 13/yr → consumption 58,500 t
- Cormorant NetCDF: 500 t (presence-weighted) standing biomass, ingestion rate 40/yr → consumption 20,000 t

**Literature values corrected:**
- Seal population 40k → 30k individuals (HELCOM Seal Database 2019 per Galatius et al. 2020 doi:10.2981/wlb.00711; stagnated 2014-2017)
- Seal standing biomass 6,000 → 4,500 t (30k × 150 kg)
- Cormorant: added presence-weighting (×0.48) for Oct-Apr absence from Baltic (Östman et al. 2013 doi:10.1371/journal.pone.0083763)
- Cormorant consumption: 500 g/bird/day × 150 days × 520k birds ≈ 20,000 t/yr (not 30,000)

**Spatial weights corrected:** shifted from uniform-ish to literature-grounded distribution:
- Seal: 40% Gulf of Bothnia, 40% Central Baltic/Stockholm, 10% Kalmarsund, 5% SW Baltic recolonizing, <3% other
- Cormorant: 22% Gulf of Riga, 20% Gulf of Finland, 15% N Baltic Proper, rest coastal

**Size-ratio predation corrected:** plan originally used prey-size proxies as "length"; OSMOSE wants predator body length for size-ratio computation. Now:
- Seal: length 110-170 cm body, size_ratio 3-12 → prey 12-50 cm (herring, sprat, cod, flounder)
- Cormorant: length 70-85 cm body, size_ratio 2.5-8 → prey 10-32 cm (perch, herring, juvenile pikeperch/cod)

**HELCOM layers consulted:**
- HOLAS3 mammals theme (grey seal, harbour seal, ringed seal assessment units)
- MADS/Indicators_and_assessments/MapServer layer 459 (Overall Seal Integrated Result)
- MADS/Indicators_and_assessments/MapServer layer 457 (Grey Seal Integrated Assessment)
- MADS/Indicators_and_assessments/MapServer layer 277 (Grey Seal Population Trends)
- MADS/Indicators_and_assessments/MapServer layer 269 (Distribution - grey seal)

**ICES cross-checks:** seal diet includes herring (her.27.20-24 + her.27.25-2932 stocks), sprat (spr.27.22-32), cod (cod.27.22-24 + cod.27.24-32), flounder. All in the focal-species list of OSMOSE Baltic — no missing stocks.

**References added to plan (for provenance trail):**
- Galatius et al. 2020 doi:10.2981/wlb.00711 (grey seal distribution)
- Lundström et al. 2010 doi:10.7557/3.2733 (seal diet composition)
- Gårdmark et al. 2012 doi:10.1093/icesjms/fss099 (seal impact on herring)
- Östman et al. 2013 doi:10.1371/journal.pone.0083763 (cormorant pop + consumption)
- Heikinheimo et al. 2021 doi:10.1093/icesjms/fsab258 (cormorant perch mortality)
- Heikinheimo et al. 2016 doi:10.1139/cjfas-2015-0033 (cormorant pikeperch)
- Bełdowska & Falkowska 2016 doi:10.1007/s11270-015-2735-5 (seal diet composition confirmation)

Plan now ready for execution. 9 tasks, ~6-11h unattended compute for the full sequence (dominated by Task 8 joint calibration at 5-10h with 8-core parallelism).

**Second review pass (2026-04-24) caught 8 more issues:**
- Task 3 commit message still said "consumption-equivalent" — fixed to "standing biomass"
- Task 6 Step 2 insertion anchor was imprecise — now uses explicit before/after blocks
- Task 2 Step 2 hardcoded line 593 in expected failure message — relaxed to line-number-agnostic
- Task 7 Step 2 only ran `--help` (weak check) — now asserts the `elif args.phase == "12"` branch exists via AST-free grep
- Plan header mentioned seal diet includes whitefish (17%) but whitefish isn't in OSMOSE Baltic since 2026-04-17; note added
- Task 8 runtime was 10-14h (overconservative for 8-core parallel) — revised to 5-10h
- **Task 5 `test_predators_depress_focal_biomass` had a real bug**: tried to disable predators by setting `simulation.nbackground=0`, but reading `osmose/engine/background.py:158-167` that key is only a consistency-warning check — the parser still picks up `species.type.spN=background`. Fixed by zeroing `species.biomass.multiplier.spN` instead.
- Task 5 threshold loosened from 3/5 to 2/5 species (trophic cascades can raise some prey biomass when predator removed)
- Cormorant condition factor 0.008 → 0.004 (biologically: 80 cm × 0.008 gave 4 kg, real cormorant is ~2 kg)

**Cross-consistency verified** via Python sanity script: script-vs-config values match; body-mass math gives realistic weights (seal 53-197 kg, cormorant 1.4-2.5 kg); consumption math matches targets within 3% (seal 58,500 vs 60,000 t/yr; cormorant 20,000 vs 20,000 t/yr); prey-size ranges cover realistic diets (seal 9-57 cm, cormorant 9-34 cm).

**Status: review-complete. No more issues found.**
