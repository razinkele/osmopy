---
name: Cod/flounder + percid life-stage maps differentiated
description: 2026-04-21 — broke cod_adult≡flounder_adult duplicate (431 cells) and built adult+juvenile variants for perch and pikeperch from species-specific habitat rules.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Continuation of the 2026-04-21 duplicate-map fix. Resolved 3 more duplicate groups:

**cod_adult (324 cells) vs flounder_adult (580 cells)** — previously shared identical 431-cell "generic demersal" template:
- Cod: core habitat SDs 22-28 (SW Baltic + central basins). Region rules: Mecklenburg/Belt/Kiel/Arkona/Bornholm-Gdansk/Gotland = full coverage; N Baltic Proper = 50%; Gulf of Riga = 25% (salinity-limited sparse); Gulf of Finland/Bothnian Sea/Bothnian Bay = absent. ICES stocks cod.27.22-24 + cod.27.24-32.
- Flounder: full coverage everywhere except Bothnian Bay (50%, low-salinity tolerance). Tolerates lower salinity than cod per Lehtonen et al. (2023) — southern Baltic pelagic spawner, northern demersal spawner.
- Deterministic PRNG (seed=42) for the partial-coverage regions so results are reproducible.

**perch_adult (58) / perch_juvenile (27)** — previously both 98 cells identical to herring_spawning template:
- Adult: today's 42-cell perch_spawning + 17 feeding-range extensions (Gulf of Riga deeper waters, Gulf of Finland central, Archipelago, Bothnian shelf). Wider than spawning range.
- Juvenile: 27 cells — deterministic 70% subset of spawning (sheltered nursery bays only) per Kallasvuo et al. (2016) "sheltered shallow sea areas" preference.

**pikeperch_adult (25) / pikeperch_juvenile (12)** — previously both 88 cells identical to obsolete perch_spawning template:
- Adult: today's 19-cell pikeperch_spawning + 7 brackish-bay extensions (Kaliningrad coast, Pomeranian Bay, Gulf of Finland historical, Gulf of Riga inner). Adults emigrate from natal lagoons.
- Juvenile: 12 cells — restricted to natal lagoons only (Curonian, Vistula, Pomeranian/Szczecin). Per Bučas et al. (2022) pikeperch juveniles tied to lagoon pondweed/charophyte habitat.

Cod ⊂ Flounder (cod 324 all overlap with flounder's 580) — biologically correct since cod's habitat is a strict subset of flounder's range.

**Why:** completing the duplicate-template cleanup identified during 2026-04-21 grid validation.

**How to apply:** no config changes needed — all maps keep their canonical filenames. `movement.file.map*` entries in `baltic_param-movement.csv` unchanged.

**Fingerprint audit after this session:** 11 maps now differentiated. 2 duplicate groups remain:
- `herring_adult` ≡ `herring_juvenile` ≡ `sprat_juvenile` (544 cells, pelagic "everywhere" template)
- `stickleback_adult` ≡ `stickleback_spawning` (147 cells)

**References:**
- Lehtonen, T. K., Gilljam, D., & Veneranta, L. (2023). The ecology and fishery of the vendace (Coregonus albula) in the Baltic Sea. J Fish Biol. https://doi.org/10.1111/jfb.15542 (cites pelagic vs demersal flounder split)
- Berkström, C., Wennerström, L., & Bergström, U. (2021). Ecological connectivity of the MPA network in the Baltic, Kattegat and Skagerrak. Ambio. https://doi.org/10.1007/s13280-021-01684-x (flounder dispersal distances)
- Kallasvuo, M. et al. (2016) — perch prefers "relatively sheltered shallow sea areas", cited by Heikinheimo et al. 2021
- Bučas, M. et al. (2022). Juvenile Fish Associated With Pondweed and Charophyte Habitat in the Curonian Lagoon. https://doi.org/10.3389/fmars.2022.862925
