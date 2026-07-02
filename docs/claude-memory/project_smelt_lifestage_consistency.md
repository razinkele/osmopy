---
name: Smelt life-stage maps made consistent
description: 2026-04-21 — smelt_juvenile and smelt_adult extended to include Curonian/Vistula/Gulf of Riga/Gulf of Finland cells matching expanded smelt_spawning. Spawning ⊂ juvenile ⊂ adult now holds 100%.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Follow-up to the earlier smelt_spawning expansion (10→27 cells across 5 populations). The 5-year sanity simulation revealed smelt biomass at 1.7% of ICES target — diagnosed as geographic mismatch: new spawning cells in Curonian/Vistula/Gulf of Riga/Gulf of Finland had no juvenile or adult habitat to migrate to. Larvae from southern populations effectively vanished.

Fixed 2026-04-21:
- `smelt_juvenile.csv` 215 → 230 cells (+15): added Curonian coast (3), Vistula+Gdansk (6), Pärnu Bay expansion (6), Gulf of Finland Narva (3); 1 cell rejected as land.
- `smelt_adult.csv` 275 → 297 cells (+22): wider adult feeding range around each nursery; 5 cells rejected as land.

Consistency verified:
- spawning ⊂ juvenile: 27/27 cells (100%)
- juvenile ⊂ adult: 230/230 cells (100%)

**Sim re-run result (5 years, Python engine):** smelt biomass still 1026 t (was 1027 t, Δ = -0.1%). The structural fix is necessary but insufficient — 5-year unspun-up runs cannot reach equilibrium. ICES target (60k t) is for 50-year spin-up.

**Broader finding:** all Baltic distribution-map changes from this session require re-calibration (NSGA-II via scripts/calibrate_baltic.py) because parameters were tuned against old maps. 5-year smoke test shows 4/8 species in ICES range (sprat, flounder, pikeperch, cod/perch near lower bound), 3 need re-calibration (herring, stickleback, smelt).

**How to apply:** no config changes. Full re-calibration follow-up recommended.

**References:**
- Lankov, A., Ojaveer, H., & Shpilev, H. (2005). Smelt (Osmerus eperlanus L.) in the Baltic Sea. Proc Estonian Acad Sci Biol Ecol 54(3): 230-241. https://doi.org/10.3176/biol.ecol.2005.3.04 — documents all 4 Baltic smelt populations with overlap between juvenile nurseries and adult feeding in each gulf.
