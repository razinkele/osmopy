---
name: Perch vs pikeperch spawning differentiated
description: 2026-04-21 — perch_spawning.csv and pikeperch_spawning.csv were cell-identical (88 each). Replaced with species-specific maps per Heikinheimo 2021, Järv 2000, Lappalainen 2003.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
`data/baltic/maps/perch_spawning.csv` and `pikeperch_spawning.csv` previously shared an identical 88-cell footprint heavily biased to SW Baltic (Mecklenburg 33, Belt/Sound 15). Neither matched its species' ecology.

Replaced on 2026-04-21 with literature-supported distributions:

- **perch_spawning.csv**: 42 cells — Bothnian Bay/Quark coastal (7), S Bothnian Sea coastal (7), Archipelago Sea (6), Gulf of Finland Estonian + Finnish coasts (10), Gulf of Riga / Pärnu / Matsalu / Viinameri (11), N Baltic Proper Swedish coast (1). Matches Heikinheimo et al. (2021) Finnish catch areas and Järv (2000) Estonian tagging.
- **pikeperch_spawning.csv**: 19 cells — Curonian Lagoon + Nemunas Delta (3), Vistula Lagoon + Gdansk Bay (5), Pomeranian Bay + Szczecin Lagoon (6), Arkona/Mecklenburg coastal (2), Gulf of Riga south (2), Gulf of Finland east lagoon-like (1). Matches Lappalainen et al. (2003), Bučas et al. (2022), Ivanauskas et al. (2025).
- **Overlap**: 2 cells in southern Gulf of Riga where both species co-occur (biologically correct).
- MD5 distinct; UI overlay pipeline loads both.

**Why:** part of the "13/26 maps share footprints" problem flagged during 2026-04-21 grid validation. The duplicate suggested the original maps were generated from a handful of habitat templates rather than per-species surveys.

**How to apply:** no config change needed; `movement.file.map15` (perch_spawning) and `movement.file.map18` (pikeperch_spawning) already point at the right filenames. A few desired cells were dropped because they fall on "land" per `baltic_grid.nc` at 0.4°×0.3° resolution — accepted tradeoff (rejected-cell list in commit trail).

**Remaining duplicate groups still to fix:**
- `cod_adult` + `flounder_adult` (431 cells — demersal generalist template)
- `herring_adult` + `herring_juvenile` + `sprat_juvenile` (544 cells — "everywhere pelagic" template)
- `herring_spawning` + `herring_spawning_autumn` + `perch_adult` + `perch_juvenile` (98 cells — coastal SW-biased template)
- `pikeperch_adult` + `pikeperch_juvenile` still match (now-obsolete) 88-cell footprint
- `stickleback_adult` + `stickleback_spawning` (147 cells)

**References:**
- Heikinheimo, O., Marjomäki, T. J., Olin, M., et al. (2021). Cormorant predation mortality of perch in coastal and archipelago areas, northern Baltic Sea. ICES J Mar Sci 79(2): 337–349. https://doi.org/10.1093/icesjms/fsab258
- Järv, L. (2000). Migrations of the perch in the coastal waters of western Estonia. Proc Estonian Acad Sci Biol Ecol 49(3): 270–276. https://doi.org/10.3176/biol.ecol.2000.3.03
- Lappalainen, J., Dörner, H., & Wysujack, K. (2003). Reproduction biology of pikeperch — a review. Ecol Freshw Fish. https://doi.org/10.1034/j.1600-0633.2003.00005.x
- Bučas, M., Lesutienė, J., Nika, N., et al. (2022). Juvenile Fish Associated With Pondweed and Charophyte Habitat in the Curonian Lagoon. Front Mar Sci. https://doi.org/10.3389/fmars.2022.862925
- Ivanauskas, E., Razinkovas-Baziukas, A., & Baziukė, D. (2025). Does the Fishery or Climate Change Drive Commercial Catches in a Shallow Eutrophic Lagoon? Fisheries Management and Ecology. https://doi.org/10.1111/fme.12792
