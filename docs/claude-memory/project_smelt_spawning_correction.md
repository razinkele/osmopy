---
name: Baltic smelt spawning map correction
description: 2026-04-21 drafted smelt_spawning_v2.csv — expanded from 10 Bothnian-Bay cells to 27 cells covering 5 populations per Lankov et al. (2005). Original kept; v2 awaiting user review.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
`data/baltic/maps/smelt_spawning.csv` only had 10 spawning cells, all clustered in Bothnian Bay. Baltic smelt (Osmerus eperlanus) has 4+ genetically distinct populations (McKeown et al. 2025) spawning across Gulf of Bothnia, Gulf of Finland, Gulf of Riga, and Curonian/Vistula Lagoons (Lankov et al. 2005).

Drafted `data/baltic/maps/smelt_spawning_v2.csv` with 27 cells:
- 10 original Bothnian Bay cells (unchanged)
- +2 Curonian Lagoon / Lithuanian coast (55.05-55.35°N, 21.0°E)
- +4 Vistula Lagoon / Gdansk Bay (54.45-54.75°N, 19.4-19.8°E)
- +5 Gulf of Riga / Pärnu Bay (57.75-58.35°N, 23.8-24.2°E)
- +3 Gulf of Finland / Narva coast (59.55°N, 27.0-27.8°E)
- +3 Archipelago Sea / S Bothnian Sea (59.85-60.15°N, 21.4-21.8°E)

4 candidate cells rejected because they fall on "land" per `baltic_grid.nc` mask (rows close to coastline at this 0.4°×0.3° resolution).

**Why:** validation run using HELCOM + ICES + peer-reviewed literature (scite MCP) revealed the map represents only 1 of the 4+ known Baltic smelt populations. Cookie-cutter duplicate-footprint problem affects 13/26 Baltic distribution maps; smelt_spawning is the clearest single-species error.

**How to apply:** To activate v2, edit `baltic_param-movement.csv:197` (`movement.file.map21`) from `maps/smelt_spawning.csv` to `maps/smelt_spawning_v2.csv`, OR copy v2 over the original after user review. Per-row convention verified: raw CSV row 0 = south (flipud on load). Writer: `.venv/bin/python` one-off script — not committed as a generator, but the cell list in this memory is the authoritative provenance.

**References:**
- Lankov, A., Ojaveer, H., & Shpilev, H. (2005). Smelt (Osmerus eperlanus L.) in the Baltic Sea. Proc. Estonian Acad. Sci. Biol. Ecol. 54(3): 230-241. https://doi.org/10.3176/biol.ecol.2005.3.04
- McKeown, N. J., Taylor, A. C., & Tysklind, N. (2025). Low genetic variation and strong genetic structure across a range of geographical scales in European smelt. J Fish Biol 107(3): 918-931. https://doi.org/10.1111/jfb.70093
- Naumenko, E. A., & Golubkova, T. A. (2025). Fertility of the smelt Osmerus eperlanus of the Curonian Lagoon. Vestnik ASTU Fishing Industry 2025(3): 13-19. https://doi.org/10.24143/2073-5529-2025-3-13-19
